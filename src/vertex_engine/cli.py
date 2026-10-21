"""CLI Entrypoint for Vertex Autonomous AI Content Engine."""

import argparse
import json
from pathlib import Path

from .comfy_runner import ComfyExecutionEngine
from .format_analytics import FormatGrowthEngine
from .quality_gate import get_showcase_qa_samples
from .unit_economics import CostEngine


def print_banner() -> None:
    print("=" * 78)
    print(" VERTEX AI CONTENT ENGINE v1.0 | ComfyUI + LoRA + Wan 2.1 + VLM Quality Gate")
    print("=" * 78)


def cmd_benchmark(args: argparse.Namespace) -> None:
    print_banner()
    engine = CostEngine(gpu_key=args.gpu)
    report = engine.compare_all(monthly_images=args.images, monthly_videos=args.videos)

    print(f"\nMonthly Target Volume: {args.images:,} Approved Stills | {args.videos:,} Approved 5s Video Clips\n")
    for tier in report["tiers"]:
        print(f"[{tier['mode_name']}]")
        print(f"  Stack        : {tier['gpu_tier']} ({tier['billing_mode']})")
        print(f"  $/Image (QA) : ${tier['cost_per_approved_image_usd']:.4f}")
        print(f"  $/Video (QA) : ${tier['cost_per_approved_video_usd']:.4f}")
        print(f"  Monthly Spend: ${tier['total_monthly_spend_usd']:,.2f} (Savings vs Hosted: {tier['savings_vs_hosted_pct']}%)")
        print()

    proof = report["halving_proof"]
    print("-" * 78)
    print(" HOW WE HALVE THE COST PER PIECE (Engine v1 -> Engine v2):")
    print(f"  * Approved Image Cost : ${proof['v1_image_usd']:.4f} -> ${proof['v2_image_usd']:.4f} (-{proof['image_cost_reduction_pct']}%)")
    print(f"  * Approved Video Cost : ${proof['v1_video_usd']:.4f} -> ${proof['v2_video_usd']:.4f} (-{proof['video_cost_reduction_pct']}%)")
    print(f"  * Annual Agency Saving: ${proof['annual_agency_savings_vs_hosted_usd']:,.2f} / year vs Hosted Tools")
    print("-" * 78)


def cmd_qa_check(_args: argparse.Namespace) -> None:
    print_banner()
    print("\nRunning 2-Second 6-Point Vision Quality Gate (Light, Skin, Hands, Head, Text, Edit)...\n")
    for v in get_showcase_qa_samples():
        status = "PASSED -> QUEUED FOR WAN 2.1 I2V" if v.passed else "REJECTED -> DROPPED PRE-VIDEO"
        print(f"Asset: {v.asset_id:<20} | Score: {v.composite_score}/10 | ArcFace: {v.identity_similarity:.3f} | {status}")
        if not v.passed:
            for reason in v.rejection_reasons:
                print(f"   [x] {reason}")
            print(f"   [Fix] {v.recommended_fix}")
        print()


def cmd_format_audit(_args: argparse.Namespace) -> None:
    print_banner()
    print("\nViral Format Growth Matrix (Read Views -> Kill What Doesn't Travel -> Double Down):\n")
    growth = FormatGrowthEngine()
    for exp in growth.evaluate_experiments():
        print(f"[{exp.decision:<11}] {exp.name}")
        print(f"  Views: {exp.total_views:,} ({exp.avg_views_per_post:,}/post) | 3s Hook: {exp.hook_retention_3s_pct}% | Watch: {exp.avg_watch_time_pct}% | Shares/1k: {exp.share_rate_per_1k}")
        print(f"  Agent Action: {exp.agent_action}\n")


def cmd_run_batch(args: argparse.Namespace) -> None:
    print_banner()
    runner = ComfyExecutionEngine()
    res = runner.run_creator_batch(creator_id=args.character, optimize=args.optimize)
    print(f"\nBatch ID          : {res.batch_id}")
    print(f"Creator           : {res.creator_id}")
    print(f"Execution Mode    : {res.execution_mode}")
    print(f"Keyframes (QA)    : {res.keyframes_approved} Approved / {res.keyframes_rejected} Rejected (Saved wasted I2V GPU!)")
    print(f"Videos Rendered   : {res.videos_rendered} (720x1280 32fps H.264, metadata stripped)")
    print(f"Total GPU Time    : {res.total_gpu_seconds}s")
    print(f"Total Batch Cost  : ${res.total_batch_cost_usd:.4f}")
    print(f"Cost / Valid Clip : ${res.effective_cost_per_approved_clip_usd:.4f}\n")

    out_dir = Path(__file__).resolve().parents[2] / "dashboard" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "latest_run.json"
    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(res.to_dict(), fh, indent=2)
    print(f"Exported telemetry snapshot -> {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Vertex AI Content Engine CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_batch = sub.add_parser("run-batch", help="Run end-to-end creator batch pipeline")
    p_batch.add_argument("--character", default="elena_rossi")
    p_batch.add_argument("--optimize", action="store_true", default=True)
    p_batch.set_defaults(func=cmd_run_batch)

    p_bench = sub.add_parser("benchmark", help="Compute GPU unit economics & cost-halving report")
    p_bench.add_argument("--gpu", default="a100_80gb", choices=["rtx_4090", "l40s", "a100_80gb", "h100_80gb"])
    p_bench.add_argument("--images", type=int, default=3000)
    p_bench.add_argument("--videos", type=int, default=1000)
    p_bench.set_defaults(func=cmd_benchmark)

    p_qa = sub.add_parser("qa-check", help="Run the 6-point Vision Quality Gate on showcase frames")
    p_qa.set_defaults(func=cmd_qa_check)

    p_fmt = sub.add_parser("format-audit", help="Audit viral formats (Double Down vs Kill)")
    p_fmt.set_defaults(func=cmd_format_audit)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
