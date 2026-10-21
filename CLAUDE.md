# CLAUDE.md — Vertex Autonomous AI Content Engine

This repository contains an end-to-end, headless AI Creator production pipeline orchestrated via **Claude Code**, **Model Context Protocol (MCP)**, **ComfyUI API workflows**, and **rented GPU workers (RunPod Serverless / Spot instances)**.

## Core Mission & Unit Economics Targets

1. **Identity Consistency at Scale:** Maintain a single photoreal character face (`>0.91` ArcFace cosine similarity) across 500+ monthly stills and video clips using Flux.1-dev + Character LoRAs and Wan 2.1 14B I2V / Kling hybrid routing.
2. **Strict Unit Economics:**
   - Target Keyframe Cost: **`<= $0.0020 / approved image`** on RunPod RTX 4090.
   - Target Video Clip Cost (5s 720p): **`<= $0.0600 / approved clip`** on RunPod A100/H100 (a **96% reduction** vs. $1.80–$3.50 on hosted web tools).
3. **Zero-Touch Quality Gate (Pre-I2V Filtering):**
   - Never dispatch an unverified keyframe to video generation. Video compute costs ~35x more than image generation.
   - Every frame must pass the 6-point Vision Quality Gate (`lighting_coherence`, `skin_realism`, `hand_anatomy`, `head_body_ratio`, `sign_text_legibility`, `edit_framing_style`) before entering the Wan 2.1 / Kling I2V queue.

## Available Commands

```bash
# Run the full pipeline simulation & generate telemetry report
python -m src.vertex_engine.cli run-batch --character elena_rossi --count 24 --optimize

# Run the unit economics & "Halve the Cost" benchmark comparison
python -m src.vertex_engine.cli benchmark

# Evaluate the 2-second Visual Quality Gate on a batch of keyframes
python -m src.vertex_engine.cli qa-check

# Audit viral formats (Kill vs. Double Down matrix)
python -m src.vertex_engine.cli format-audit

# Start the MCP Server for Claude Code integration
python -m src.vertex_engine.mcp_server

# Run automated test suite
pytest -v
```

## Autonomous Agent Workflow (SOP for Claude Code)

When asked to produce a daily content batch for a creator:
1. Call MCP tool `audit_format_performance` to identify winning hooks (`DOUBLE_DOWN`) and terminate dead formats (`KILL`).
2. Call MCP tool `build_comfy_api_payload` with `flux_guidance=2.2` (prevents plastic Flux skin sheen) and character LoRA strength `0.85` (prevents oversized head bias from close-up training crops).
3. Call MCP tool `dispatch_creator_batch` to generate keyframes on RunPod GPU workers.
4. Call MCP tool `evaluate_quality_gate` on all keyframes. Drop any frame failing the 6-point visual inspection before I2V stage.
5. Route approved keyframes:
   - **85% standard motion (turning, walking, coffee b-roll, subtle expressions):** Route to local/RunPod **Wan 2.1 14B FP8 + TeaCache + SageAttention2** (`~$0.052/clip`).
   - **15% complex hand-object interaction or fast dance physics:** Route via API to **Kling 1.6 Pro / Seedance** (`~$0.28/clip`).
6. Log final batch telemetry via `analyze_unit_economics`.
