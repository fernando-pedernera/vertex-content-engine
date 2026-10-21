"""Programmatic ComfyUI API Graph Builder & Mutator.

Loads base ComfyUI API JSON templates and dynamically injects character LoRA weights,
anti-plastic skin guidance parameters, TeaCache acceleration thresholds, and metadata stripping.
"""

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .config import CREATOR_REGISTRY, CreatorProfile

WORKFLOWS_DIR = Path(__file__).resolve().parents[2] / "workflows"


class ComfyWorkflowBuilder:
    """Mutates headless ComfyUI API graphs for batch image and video generation."""

    def __init__(self, workflows_dir: Optional[Path] = None) -> None:
        self.workflows_dir = workflows_dir or WORKFLOWS_DIR

    def load_template(self, filename: str) -> Dict[str, Any]:
        path = self.workflows_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Workflow template not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def build_flux_keyframe_graph(
        self,
        creator_id: str,
        scene_prompt: str,
        seed: int = 849201442,
        width: int = 896,
        height: int = 1152,
        flux_guidance: Optional[float] = None,
        lora_strength: Optional[float] = None,
        film_grain: float = 0.018,
    ) -> Dict[str, Any]:
        """Builds a parameterized Flux.1-dev + Character LoRA API payload."""
        if creator_id not in CREATOR_REGISTRY:
            raise ValueError(f"Unknown creator_id '{creator_id}'. Available: {list(CREATOR_REGISTRY)}")

        creator: CreatorProfile = CREATOR_REGISTRY[creator_id]
        graph = copy.deepcopy(self.load_template("01_flux_dev_character_lora_iphone_api.json"))

        effective_guidance = flux_guidance if flux_guidance is not None else creator.default_flux_guidance
        effective_lora = lora_strength if lora_strength is not None else creator.flux_lora_strength

        # Node 15: Character Identity LoRA
        graph["15"]["inputs"]["lora_name"] = creator.flux_lora_file
        graph["15"]["inputs"]["strength_model"] = effective_lora
        graph["15"]["inputs"]["strength_clip"] = effective_lora

        # Node 20: Dual prompt + low guidance (2.0-2.3 avoids waxy Flux skin sheen)
        full_prompt = (
            f"IMG_{seed % 9000 + 1000}.HEIC, raw unedited iPhone 15 Pro photo of {creator.trigger_word}, "
            f"{scene_prompt}. Shot on 24mm smartphone lens, natural unretouched skin texture with visible pores, "
            f"anatomically accurate hands with five fingers, authentic lighting, zero studio airbrushing."
        )
        graph["20"]["inputs"]["clip_l"] = full_prompt[:240]
        graph["20"]["inputs"]["t5xxl"] = full_prompt
        graph["20"]["inputs"]["guidance"] = effective_guidance

        # Node 25 & 30: Resolution
        graph["25"]["inputs"]["width"] = width
        graph["25"]["inputs"]["height"] = height
        graph["30"]["inputs"]["width"] = width
        graph["30"]["inputs"]["height"] = height

        # Node 33 & 45: Deterministic seeds
        graph["33"]["inputs"]["noise_seed"] = seed
        graph["45"]["inputs"]["seed"] = seed

        # Node 50: Sensor grain
        graph["50"]["inputs"]["grain_intensity"] = film_grain

        # Node 60: Output prefix
        graph["60"]["inputs"]["filename_prefix"] = f"vertex_engine/{creator.creator_id}/keyframe_{seed}"
        return graph

    def build_wan21_i2v_graph(
        self,
        creator_id: str,
        approved_image_name: str,
        motion_prompt: str,
        seed: int = 991827364,
        teacache_threshold: float = 0.16,
        use_sage_attention: bool = True,
        strip_metadata: bool = True,
    ) -> Dict[str, Any]:
        """Builds a Wan 2.1 14B Image-to-Video API graph with TeaCache and SageAttention2."""
        if creator_id not in CREATOR_REGISTRY:
            raise ValueError(f"Unknown creator_id '{creator_id}'. Available: {list(CREATOR_REGISTRY)}")

        creator: CreatorProfile = CREATOR_REGISTRY[creator_id]
        graph = copy.deepcopy(self.load_template("02_wan21_14b_i2v_teacache_api.json"))

        # Node 10: Attention mode (SageAttention2 cuts inference time ~28%)
        graph["10"]["inputs"]["attention_mode"] = "sageattn" if use_sage_attention else "sdpa"

        # Node 12: TeaCache threshold (0.15-0.17 is the sweet spot before temporal flicker)
        graph["12"]["inputs"]["rel_l1_thresh"] = teacache_threshold

        # Node 14: Character Video LoRA
        graph["14"]["inputs"]["lora"] = creator.wan_lora_file
        graph["14"]["inputs"]["strength"] = creator.wan_lora_strength

        # Node 20: Input keyframe that passed the 6-point Quality Gate
        graph["20"]["inputs"]["image"] = approved_image_name

        # Node 25: Motion prompt
        graph["25"]["inputs"]["positive_prompt"] = (
            f"Handheld iPhone video of {creator.trigger_word}, {motion_prompt}, "
            "natural facial micro-expressions, stable cranial proportions, realistic smartphone camera sway."
        )

        # Node 35: Sampler seed
        graph["35"]["inputs"]["seed"] = seed

        # Node 60: Output & metadata stripping for multi-account safety
        graph["60"]["inputs"]["filename_prefix"] = f"vertex_engine/{creator.creator_id}/reel_{seed}"
        graph["60"]["inputs"]["save_metadata"] = not strip_metadata
        return graph
