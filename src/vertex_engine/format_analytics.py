"""Viral Format Engineering & Reach Decision Matrix.

Implements Requirement #3 of "What you'll do":
"Engineer characters and formats for reach: test, read the views,
kill what does not travel, double down on what does"
"""

from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass
class FormatExperiment:
    format_id: str
    name: str
    hook_mechanic: str
    posts_published: int
    total_views: int
    avg_views_per_post: int
    hook_retention_3s_pct: float
    avg_watch_time_pct: float
    share_rate_per_1k: float
    cost_per_1k_views_usd: float
    decision: str  # DOUBLE_DOWN | ITERATE | KILL
    agent_action: str

    def to_dict(self) -> Dict:
        return asdict(self)


class FormatGrowthEngine:
    """Evaluates content formats by retention & share velocity to automate batch allocation."""

    @staticmethod
    def classify_format(
        hook_3s_pct: float,
        watch_time_pct: float,
        share_rate_per_1k: float,
    ) -> str:
        if hook_3s_pct >= 68.0 and watch_time_pct >= 52.0 and share_rate_per_1k >= 8.0:
            return "DOUBLE_DOWN"
        if hook_3s_pct >= 50.0 and watch_time_pct >= 38.0:
            return "ITERATE"
        return "KILL"

    def evaluate_experiments(self) -> List[FormatExperiment]:
        raw_experiments = [
            {
                "format_id": "fmt_01_bodega_pov",
                "name": "Late-Night NYC Bodega / Espresso POV + Relatable Text Hook",
                "hook_mechanic": "2.8s eye-contact turn + 'did you eat today or are you being dramatic again' overlay",
                "posts_published": 18,
                "total_views": 2_840_000,
                "hook_retention_3s_pct": 76.4,
                "avg_watch_time_pct": 64.2,
                "share_rate_per_1k": 14.8,
                "production_cost_usd": 1.08,
            },
            {
                "format_id": "fmt_02_amalfi_camera_roll",
                "name": "6-Slide Carousel + 1 Loop Video ('Brooklyn vs. Praiano Camera Roll')",
                "hook_mechanic": "Mixed photo/video carousel triggering Instagram 2nd-slide re-delivery algorithm",
                "posts_published": 24,
                "total_views": 1_920_000,
                "hook_retention_3s_pct": 71.2,
                "avg_watch_time_pct": 58.0,
                "share_rate_per_1k": 11.2,
                "production_cost_usd": 0.42,
            },
            {
                "format_id": "fmt_03_fit_check_mirror",
                "name": "Quick Fit-Check Mirror Transition (2 Outfits)",
                "hook_mechanic": "Snap cut on beat drop at 1.4s",
                "posts_published": 15,
                "total_views": 410_000,
                "hook_retention_3s_pct": 56.0,
                "avg_watch_time_pct": 41.5,
                "share_rate_per_1k": 4.1,
                "production_cost_usd": 0.90,
            },
            {
                "format_id": "fmt_04_polished_studio_walk",
                "name": "Slow-Motion Studio Runway Walk (High-Glamour)",
                "hook_mechanic": "Cinematic slow-mo pan with luxury fashion prompt",
                "posts_published": 12,
                "total_views": 28_400,
                "hook_retention_3s_pct": 31.5,
                "avg_watch_time_pct": 19.2,
                "share_rate_per_1k": 0.6,
                "production_cost_usd": 0.72,
            },
        ]

        results: List[FormatExperiment] = []
        for item in raw_experiments:
            decision = self.classify_format(
                item["hook_retention_3s_pct"],
                item["avg_watch_time_pct"],
                item["share_rate_per_1k"],
            )
            avg_views = item["total_views"] // item["posts_published"]
            cpm = (item["production_cost_usd"] / (item["total_views"] / 1000.0))

            if decision == "DOUBLE_DOWN":
                action = "Scale to 60% of tomorrow's GPU batch across 5 creator accounts; clone hook structure."
            elif decision == "ITERATE":
                action = "Keep visual setting; test 3 faster text hooks in first 1.5s to push 3s retention >68%."
            else:
                action = "TERMINATED automatically by agent. Looks too polished/AI-studio; zero organic travel."

            results.append(
                FormatExperiment(
                    format_id=item["format_id"],
                    name=item["name"],
                    hook_mechanic=item["hook_mechanic"],
                    posts_published=item["posts_published"],
                    total_views=item["total_views"],
                    avg_views_per_post=avg_views,
                    hook_retention_3s_pct=item["hook_retention_3s_pct"],
                    avg_watch_time_pct=item["avg_watch_time_pct"],
                    share_rate_per_1k=item["share_rate_per_1k"],
                    cost_per_1k_views_usd=round(cpm, 5),
                    decision=decision,
                    agent_action=action,
                )
            )
        return results
