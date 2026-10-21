"""Automated Test Suite for Vertex Autonomous AI Content Engine."""

from src.vertex_engine.comfy_runner import ComfyExecutionEngine
from src.vertex_engine.format_analytics import FormatGrowthEngine
from src.vertex_engine.mcp_server import execute_mcp_tool, handle_jsonrpc_request
from src.vertex_engine.quality_gate import VisionQualityGate, get_showcase_qa_samples
from src.vertex_engine.unit_economics import CostEngine
from src.vertex_engine.workflow_builder import ComfyWorkflowBuilder


def test_flux_workflow_mutation() -> None:
    builder = ComfyWorkflowBuilder()
    graph = builder.build_flux_keyframe_graph(
        creator_id="elena_rossi",
        scene_prompt="sitting at a cafe in Brooklyn",
        seed=123456,
        flux_guidance=2.2,
    )
    assert graph["15"]["inputs"]["lora_name"] == "elena_rossi_flux_v2.safetensors"
    assert graph["15"]["inputs"]["strength_model"] == 0.85
    assert graph["20"]["inputs"]["guidance"] == 2.2
    assert graph["33"]["inputs"]["noise_seed"] == 123456
    assert "elenav2" in graph["20"]["inputs"]["t5xxl"]


def test_wan21_i2v_workflow_mutation() -> None:
    builder = ComfyWorkflowBuilder()
    graph = builder.build_wan21_i2v_graph(
        creator_id="elena_rossi",
        approved_image_name="elena_01_cafe.jpg",
        motion_prompt="turning toward camera",
        seed=654321,
        teacache_threshold=0.16,
        use_sage_attention=True,
        strip_metadata=True,
    )
    assert graph["10"]["inputs"]["attention_mode"] == "sageattn"
    assert graph["12"]["inputs"]["rel_l1_thresh"] == 0.16
    assert graph["20"]["inputs"]["image"] == "elena_01_cafe.jpg"
    assert graph["60"]["inputs"]["save_metadata"] is False


def test_quality_gate_pass_and_reject() -> None:
    samples = get_showcase_qa_samples()
    assert len(samples) == 5
    passed = [s for s in samples if s.passed]
    rejected = [s for s in samples if not s.passed]
    assert len(passed) == 4
    assert len(rejected) == 1
    assert rejected[0].asset_id == "qa_rejected_sample"
    assert any("HANDS_FAIL" in r for r in rejected[0].rejection_reasons)
    assert any("SKIN_FAIL" in r for r in rejected[0].rejection_reasons)


def test_unit_economics_halves_cost() -> None:
    engine = CostEngine(gpu_key="a100_80gb")
    report = engine.compare_all(monthly_images=3000, monthly_videos=1000)
    proof = report["halving_proof"]
    # Engine v2 must reduce both image and video cost by >50% vs Engine v1
    assert proof["image_cost_reduction_pct"] >= 50.0
    assert proof["video_cost_reduction_pct"] >= 50.0
    assert proof["v2_video_usd"] < 0.08


def test_format_growth_decisions() -> None:
    growth = FormatGrowthEngine()
    experiments = growth.evaluate_experiments()
    decisions = {e.format_id: e.decision for e in experiments}
    assert decisions["fmt_01_bodega_pov"] == "DOUBLE_DOWN"
    assert decisions["fmt_04_polished_studio_walk"] == "KILL"


def test_mcp_server_jsonrpc_tools() -> None:
    init_resp = handle_jsonrpc_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert init_resp is not None
    assert init_resp["result"]["serverInfo"]["name"] == "vertex-content-engine"

    list_resp = handle_jsonrpc_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert list_resp is not None
    tool_names = [t["name"] for t in list_resp["result"]["tools"]]
    assert "dispatch_creator_batch" in tool_names
    assert "evaluate_quality_gate" in tool_names
    assert "analyze_unit_economics" in tool_names

    batch_out = execute_mcp_tool("dispatch_creator_batch", {"creator_id": "elena_rossi", "optimize": True})
    assert batch_out["keyframes_approved"] == 4
    assert batch_out["keyframes_rejected"] == 1
    assert batch_out["videos_rendered"] == 4


if __name__ == "__main__":
    test_flux_workflow_mutation()
    test_wan21_i2v_workflow_mutation()
    test_quality_gate_pass_and_reject()
    test_unit_economics_halves_cost()
    test_format_growth_decisions()
    test_mcp_server_jsonrpc_tools()
    print("ALL 6 VERTEX ENGINE TEST SUITES PASSED [OK]")
