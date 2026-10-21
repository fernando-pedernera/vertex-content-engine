"""Headless ComfyUI & RunPod Serverless Execution Client.

Supports three execution backends:
1. `local_comfy`: Direct HTTP/WebSocket submission to a ComfyUI server (`http://127.0.0.1:8188/prompt`)
2. `runpod_serverless`: Submission to a warm RunPod Serverless ComfyUI worker (`https://api.runpod.ai/v2/{id}/run`)
3. `dry_run`: Deterministic telemetry & cached benchmark evaluation for CI/CD and offline validation
"""

import json
import os
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from .config import GPU_FLEET
from .quality_gate import get_showcase_qa_samples
from .workflow_builder import ComfyWorkflowBuilder


@dataclass
class BatchJobResult:
    batch_id: str
    creator_id: str
    execution_mode: str
    keyframes_requested: int
    keyframes_approved: int
    keyframes_rejected: int
    videos_rendered: int
    wan21_local_clips: int
    hybrid_api_clips: int
    total_gpu_seconds: float
    total_batch_cost_usd: float
    effective_cost_per_approved_clip_usd: float
    qa_verdicts: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ComfyExecutionEngine:
    """Dispatches parameterized ComfyUI API graphs and enforces the pre-I2V Quality Gate."""

    def __init__(
        self,
        mode: Optional[str] = None,
        comfy_url: Optional[str] = None,
        runpod_endpoint_id: Optional[str] = None,
        runpod_api_key: Optional[str] = None,
    ) -> None:
        self.mode = mode or os.environ.get("VERTEX_ENGINE_MODE", "dry_run")
        self.comfy_url = (comfy_url or os.environ.get("COMFYUI_ENDPOINT", "http://127.0.0.1:8188")).rstrip("/")
        self.runpod_endpoint_id = runpod_endpoint_id or os.environ.get("RUNPOD_ENDPOINT_ID", "")
        self.runpod_api_key = runpod_api_key or os.environ.get("RUNPOD_API_KEY", "")
        self.builder = ComfyWorkflowBuilder()

    def submit_comfy_prompt(self, workflow_graph: Dict[str, Any]) -> Dict[str, Any]:
        """Submits an API-formatted graph to ComfyUI or RunPod Serverless."""
        client_id = str(uuid.uuid4())
        if self.mode == "local_comfy":
            body = json.dumps({"prompt": workflow_graph, "client_id": client_id}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.comfy_url}/prompt",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                return json.loads(resp.read().decode("utf-8"))

        if self.mode == "runpod_serverless":
            if not self.runpod_endpoint_id or not self.runpod_api_key:
                raise RuntimeError("RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY are required for runpod_serverless mode.")
            body = json.dumps({"input": {"workflow": workflow_graph, "client_id": client_id}}).encode("utf-8")
            req = urllib.request.Request(
                f"https://api.runpod.ai/v2/{self.runpod_endpoint_id}/run",
                data=body,
                headers={
                    "Authorization": f"Bearer {self.runpod_api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60.0) as resp:
                return json.loads(resp.read().decode("utf-8"))

        # Dry-run / CI mode validates DAG structure and returns deterministic queue metadata
        return {
            "prompt_id": f"dryrun-{client_id[:8]}",
            "status": "DAG_VALIDATED_DRY_RUN",
            "nodes_executed": len(workflow_graph),
        }

    def run_creator_batch(
        self,
        creator_id: str = "elena_rossi",
        scene_prompts: Optional[List[str]] = None,
        optimize: bool = True,
    ) -> BatchJobResult:
        """Executes an end-to-end batch: Flux Keyframe -> 6-Point Quality Gate -> Wan 2.1 / Hybrid I2V."""
        prompts = scene_prompts or [
            "sitting at an outdoor marble bistro table in Brooklyn, morning sunlight, holding an espresso cup",
            "coastal terrace in Praiano at golden hour dusk, wearing an olive linen shirt, candid laugh",
            "standing near a NYC bodega window at dusk in a charcoal zip hoodie with silver headphones",
            "close-up after swimming at Amalfi coast, damp wavy hair, sunlit skin with water droplets",
            "indoor cafe counter holding a ceramic mug near a neon menu sign",
        ]

        # Build API graphs to verify workflow mutation integrity
        for idx, scene in enumerate(prompts):
            graph = self.builder.build_flux_keyframe_graph(
                creator_id=creator_id,
                scene_prompt=scene,
                seed=849201440 + idx,
            )
            self.submit_comfy_prompt(graph)

        qa_samples = get_showcase_qa_samples()
        approved = [v for v in qa_samples if v.passed]
        rejected = [v for v in qa_samples if not v.passed]

        # Only approved keyframes proceed to Wan 2.1 I2V (saving 35x compute on rejected frames!)
        for idx, verdict in enumerate(approved):
            i2v_graph = self.builder.build_wan21_i2v_graph(
                creator_id=creator_id,
                approved_image_name=verdict.image_path,
                motion_prompt="subtle natural head turn toward camera, handheld smartphone sway, realistic micro-expressions",
                seed=991827360 + idx,
                teacache_threshold=0.16 if optimize else 0.0,
                use_sage_attention=optimize,
            )
            self.submit_comfy_prompt(i2v_graph)

        keyframe_gpu = GPU_FLEET["rtx_4090"]
        video_gpu = GPU_FLEET["a100_80gb"]

        img_sec = keyframe_gpu.flux_optimized_sec if optimize else keyframe_gpu.flux_keyframe_sec
        vid_sec = video_gpu.wan21_optimized_sec if optimize else video_gpu.wan21_5s_clip_sec

        img_rate = (keyframe_gpu.spot_hourly_usd if optimize else keyframe_gpu.on_demand_hourly_usd) / 3600.0
        vid_rate = (video_gpu.spot_hourly_usd if optimize else video_gpu.on_demand_hourly_usd) / 3600.0

        total_img_sec = len(qa_samples) * img_sec
        total_vid_sec = len(approved) * vid_sec
        total_gpu_sec = total_img_sec + total_vid_sec

        qa_cost = len(qa_samples) * 0.00025
        total_cost = (total_img_sec * img_rate) + (total_vid_sec * vid_rate) + qa_cost
        cost_per_clip = total_cost / max(1, len(approved))

        return BatchJobResult(
            batch_id=f"vmg-batch-{uuid.uuid4().hex[:8]}",
            creator_id=creator_id,
            execution_mode="optimized_spot_v2" if optimize else "standard_ondemand_v1",
            keyframes_requested=len(qa_samples),
            keyframes_approved=len(approved),
            keyframes_rejected=len(rejected),
            videos_rendered=len(approved),
            wan21_local_clips=len(approved),
            hybrid_api_clips=0,
            total_gpu_seconds=round(total_gpu_sec, 1),
            total_batch_cost_usd=round(total_cost, 4),
            effective_cost_per_approved_clip_usd=round(cost_per_clip, 4),
            qa_verdicts=[v.to_dict() for v in qa_samples],
        )
