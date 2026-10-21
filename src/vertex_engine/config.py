"""Configuration, GPU Pricing Tables, and Creator Registry for Vertex Content Engine."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class GPUTier:
    name: str
    vram_gb: int
    on_demand_hourly_usd: float
    spot_hourly_usd: float
    flux_keyframe_sec: float
    flux_optimized_sec: float
    wan21_5s_clip_sec: float
    wan21_optimized_sec: float


GPU_FLEET: Dict[str, GPUTier] = {
    "rtx_4090": GPUTier(
        name="RunPod NVIDIA RTX 4090 (24GB)",
        vram_gb=24,
        on_demand_hourly_usd=0.69,
        spot_hourly_usd=0.39,
        flux_keyframe_sec=19.5,
        flux_optimized_sec=8.2,   # FP8 + TeaCache + torch.compile + persistent worker
        wan21_5s_clip_sec=310.0,
        wan21_optimized_sec=155.0,
    ),
    "l40s": GPUTier(
        name="RunPod NVIDIA L40S (48GB)",
        vram_gb=48,
        on_demand_hourly_usd=1.09,
        spot_hourly_usd=0.68,
        flux_keyframe_sec=15.0,
        flux_optimized_sec=6.4,
        wan21_5s_clip_sec=195.0,
        wan21_optimized_sec=96.0,
    ),
    "a100_80gb": GPUTier(
        name="RunPod NVIDIA A100 SXM (80GB)",
        vram_gb=80,
        on_demand_hourly_usd=1.89,
        spot_hourly_usd=1.19,
        flux_keyframe_sec=11.0,
        flux_optimized_sec=4.8,
        wan21_5s_clip_sec=145.0,
        wan21_optimized_sec=72.0,  # FP8 + SageAttention2 + TeaCache (rel_l1=0.16)
    ),
    "h100_80gb": GPUTier(
        name="RunPod NVIDIA H100 SXM (80GB)",
        vram_gb=80,
        on_demand_hourly_usd=2.99,
        spot_hourly_usd=1.89,
        flux_keyframe_sec=6.5,
        flux_optimized_sec=2.9,
        wan21_5s_clip_sec=84.0,
        wan21_optimized_sec=41.0,  # Native FP8 Tensor Cores + SageAttention2 + TeaCache
    ),
}

HOSTED_TOOL_PRICING_USD: Dict[str, float] = {
    "kling_1_6_pro_web": 1.40,       # Avg per usable 5s clip factoring 35% reroll rate
    "higgsfield_cinema": 2.25,       # Avg per usable clip factoring rerolls
    "seedance_api_hd": 0.85,         # API fallback for complex multi-limb motion
    "krea_video_hosted": 1.90,
}


@dataclass(frozen=True)
class CreatorProfile:
    creator_id: str
    handle: str
    display_name: str
    age: int
    locations: List[str]
    trigger_word: str
    flux_lora_file: str
    flux_lora_strength: float
    wan_lora_file: str
    wan_lora_strength: float
    default_flux_guidance: float = 2.2
    persona_hook: str = ""
    target_surfaces: List[str] = field(default_factory=lambda: ["instagram_reels", "tiktok", "threads"])


CREATOR_REGISTRY: Dict[str, CreatorProfile] = {
    "elena_rossi": CreatorProfile(
        creator_id="elena_rossi",
        handle="@elena.rossi.nyc",
        display_name="Elena Rossi",
        age=23,
        locations=["Brooklyn, NY", "Amalfi Coast, IT"],
        trigger_word="elenav2",
        flux_lora_file="elena_rossi_flux_v2.safetensors",
        flux_lora_strength=0.85,
        wan_lora_file="elena_rossi_wan21_motion_v1.safetensors",
        wan_lora_strength=0.80,
        default_flux_guidance=2.2,
        persona_hook="Architecture grad student in Brooklyn, espresso snob, summers in Atrani.",
    ),
    "chiara_benedetti": CreatorProfile(
        creator_id="chiara_benedetti",
        handle="@chiara.benedetti",
        display_name="Chiara Benedetti",
        age=21,
        locations=["Brooklyn, NY", "Praiano, IT"],
        trigger_word="chiarav1",
        flux_lora_file="chiara_benedetti_flux_v1.safetensors",
        flux_lora_strength=0.86,
        wan_lora_file="chiara_benedetti_wan21_v1.safetensors",
        wan_lora_strength=0.82,
        default_flux_guidance=2.1,
        persona_hook="Jersey girl in Brooklyn, fashion school senior. Always asks if you ate.",
    ),
}
