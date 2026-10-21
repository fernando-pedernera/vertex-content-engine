"""Model Context Protocol (MCP) Server for Claude Code & Agent Orchestration.

Exposes 5 production tools to Claude Code (via `.mcp.json`):
1. `dispatch_creator_batch`: Runs end-to-end Flux LoRA -> 6-point Quality Gate -> Wan 2.1 I2V batch
2. `evaluate_quality_gate`: Runs the 2-second 6-axis realism inspection on keyframes
3. `analyze_unit_economics`: Computes $/image, $/clip, and the 5-lever cost-halving breakdown
4. `audit_format_performance`: Evaluates viral formats (3s hook rate, watch time, shares) -> DOUBLE_DOWN / KILL
5. `build_comfy_api_payload`: Dynamically mutates ComfyUI API JSON graphs for Flux or Wan 2.1

Supports standard MCP JSON-RPC 2.0 over stdio with zero external setup required.
"""

import json
import sys
from typing import Any, Dict, List

from .comfy_runner import ComfyExecutionEngine
from .format_analytics import FormatGrowthEngine
from .quality_gate import VisionQualityGate, get_showcase_qa_samples
from .unit_economics import CostEngine
from .workflow_builder import ComfyWorkflowBuilder

TOOLS_SPEC: List[Dict[str, Any]] = [
    {
        "name": "dispatch_creator_batch",
        "description": (
            "Dispatches a batch of photoreal AI creator keyframes (Flux.1-dev + Character LoRA), "
            "filters them through the 6-point Vision Quality Gate, and renders approved frames "
            "into vertical 720p reels via Wan 2.1 14B (FP8 + TeaCache + SageAttention2)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "creator_id": {
                    "type": "string",
                    "description": "Creator ID from registry (e.g. 'elena_rossi' or 'chiara_benedetti')",
                    "default": "elena_rossi",
                },
                "optimize": {
                    "type": "boolean",
                    "description": "Enable TeaCache, SageAttention2, FP8, and Spot GPU pricing",
                    "default": True,
                },
            },
        },
    },
    {
        "name": "evaluate_quality_gate",
        "description": (
            "Evaluates a generated keyframe against Vertex's 6 realism axes "
            "(light, skin, hands, head_size, sign_text, edit_style) and ArcFace identity lock."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "asset_id": {"type": "string", "default": "sample_frame_01"},
                "image_path": {"type": "string", "default": "assets/showcase/elena_01_cafe.jpg"},
                "identity_similarity": {"type": "number", "default": 0.94},
                "scores": {
                    "type": "object",
                    "description": "Scores (0-10) for light, skin, hands, head_size, sign_text, edit_style",
                },
            },
        },
    },
    {
        "name": "analyze_unit_economics",
        "description": (
            "Computes exact GPU cost per piece ($/image and $/video) across Hosted SaaS tools, "
            "Standard ComfyUI v1, and Cost-Halved Engine v2."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "gpu_key": {
                    "type": "string",
                    "enum": ["rtx_4090", "l40s", "a100_80gb", "h100_80gb"],
                    "default": "a100_80gb",
                },
                "monthly_images": {"type": "integer", "default": 3000},
                "monthly_videos": {"type": "integer", "default": 1000},
            },
        },
    },
    {
        "name": "audit_format_performance",
        "description": (
            "Reads view velocity, 3-second hook retention, and share rate across active creator "
            "formats and classifies each into DOUBLE_DOWN, ITERATE, or KILL."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "build_comfy_api_payload",
        "description": "Generates a headless ComfyUI API JSON graph for Flux.1-dev LoRA or Wan 2.1 I2V.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pipeline_type": {
                    "type": "string",
                    "enum": ["flux_keyframe", "wan21_i2v"],
                    "default": "flux_keyframe",
                },
                "creator_id": {"type": "string", "default": "elena_rossi"},
                "prompt": {
                    "type": "string",
                    "default": "candid iPhone snapshot at an outdoor espresso bar in Brooklyn",
                },
                "seed": {"type": "integer", "default": 849201442},
            },
            "required": ["pipeline_type", "creator_id", "prompt"],
        },
    },
]


def execute_mcp_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Executes the requested MCP tool and returns a structured dictionary payload."""
    if name == "dispatch_creator_batch":
        engine = ComfyExecutionEngine()
        result = engine.run_creator_batch(
            creator_id=arguments.get("creator_id", "elena_rossi"),
            optimize=bool(arguments.get("optimize", True)),
        )
        return result.to_dict()

    if name == "evaluate_quality_gate":
        if "scores" not in arguments:
            return {"showcase_verdicts": [v.to_dict() for v in get_showcase_qa_samples()]}
        gate = VisionQualityGate()
        verdict = gate.evaluate_metrics(
            asset_id=arguments.get("asset_id", "custom_check"),
            image_path=arguments.get("image_path", "custom.jpg"),
            identity_similarity=float(arguments.get("identity_similarity", 0.93)),
            raw_scores=arguments["scores"],
            notes=arguments.get("notes", {}),
        )
        return verdict.to_dict()

    if name == "analyze_unit_economics":
        cost_engine = CostEngine(gpu_key=arguments.get("gpu_key", "a100_80gb"))
        return cost_engine.compare_all(
            monthly_images=int(arguments.get("monthly_images", 3000)),
            monthly_videos=int(arguments.get("monthly_videos", 1000)),
        )

    if name == "audit_format_performance":
        growth = FormatGrowthEngine()
        return {"experiments": [exp.to_dict() for exp in growth.evaluate_experiments()]}

    if name == "build_comfy_api_payload":
        builder = ComfyWorkflowBuilder()
        p_type = arguments.get("pipeline_type", "flux_keyframe")
        creator_id = arguments.get("creator_id", "elena_rossi")
        prompt = arguments.get("prompt", "candid iPhone photo")
        seed = int(arguments.get("seed", 849201442))
        if p_type == "wan21_i2v":
            return builder.build_wan21_i2v_graph(
                creator_id=creator_id,
                approved_image_name="assets/showcase/elena_01_cafe.jpg",
                motion_prompt=prompt,
                seed=seed,
            )
        return builder.build_flux_keyframe_graph(
            creator_id=creator_id,
            scene_prompt=prompt,
            seed=seed,
        )

    raise ValueError(f"Unknown MCP tool: {name}")


def handle_jsonrpc_request(request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Processes a single MCP JSON-RPC 2.0 message."""
    method = request.get("method", "")
    req_id = request.get("id")
    params = request.get("params") or {}

    if method == "notifications/initialized":
        return None

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "vertex-content-engine", "version": "1.0.0"},
            },
        }

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOLS_SPEC},
        }

    if method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments") or {}
        try:
            payload = execute_mcp_tool(tool_name, tool_args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
                    "isError": False,
                },
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error executing {tool_name}: {exc}"}],
                    "isError": True,
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main() -> None:
    """Runs the stdio MCP server loop for Claude Code."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = handle_jsonrpc_request(req)
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
        except json.JSONDecodeError:
            continue


if __name__ == "__main__":
    main()
