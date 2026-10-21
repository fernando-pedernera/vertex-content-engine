"""Interactive Full-Stack Web Application Server for Vertex Content Engine.

Runs a zero-dependency HTTP server on http://localhost:8080 that:
1. Serves the interactive Dashboard UI (`dashboard/index.html`) and visual assets (`assets/showcase/`)
2. Provides live REST API endpoints backed directly by our Python engine:
   - GET  /api/status            -> Engine health & registered creators
   - POST /api/run-batch         -> Runs ComfyExecutionEngine batch & returns live telemetry
   - POST /api/build-workflow    -> Generates parameterized ComfyUI API JSON (Flux or Wan 2.1)
   - GET  /api/benchmark         -> Runs CostEngine unit economics comparison
   - GET  /api/formats           -> Runs FormatGrowthEngine viral format audit
"""

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

from .comfy_runner import ComfyExecutionEngine
from .config import CREATOR_REGISTRY, GPU_FLEET
from .format_analytics import FormatGrowthEngine
from .quality_gate import get_showcase_qa_samples
from .unit_economics import CostEngine
from .workflow_builder import ComfyWorkflowBuilder

ROOT_DIR = Path(__file__).resolve().parents[2]


class VertexWebHandler(BaseHTTPRequestHandler):
    """Handles both static files (Dashboard + Images) and live Python Engine API endpoints."""

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def _serve_file(self, file_path: Path) -> None:
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404, f"File not found: {file_path.name}")
            return
        mime, _ = mimetypes.guess_type(str(file_path))
        content = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path
        query = parse_qs(parsed.query)

        if route in ("/", "/index.html", "/dashboard", "/dashboard/"):
            return self._serve_file(ROOT_DIR / "dashboard" / "index.html")

        if route.startswith("/assets/showcase/"):
            rel_name = route.split("/assets/showcase/")[-1]
            return self._serve_file(ROOT_DIR / "assets" / "showcase" / rel_name)

        if route == "/api/status":
            return self._send_json({
                "status": "ok",
                "creators": list(CREATOR_REGISTRY.keys()),
                "gpu_tiers": list(GPU_FLEET.keys()),
                "qa_samples": [v.to_dict() for v in get_showcase_qa_samples()],
            })

        if route == "/api/benchmark":
            gpu_key = query.get("gpu", ["a100_80gb"])[0]
            imgs = int(query.get("images", ["3000"])[0])
            vids = int(query.get("videos", ["1000"])[0])
            engine = CostEngine(gpu_key=gpu_key)
            return self._send_json(engine.compare_all(monthly_images=imgs, monthly_videos=vids))

        if route == "/api/formats":
            growth = FormatGrowthEngine()
            return self._send_json({"experiments": [e.to_dict() for e in growth.evaluate_experiments()]})

        self.send_error(404, "Endpoint not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        body_bytes = self.rfile.read(length) if length > 0 else b"{}"
        try:
            data = json.loads(body_bytes.decode("utf-8"))
        except json.JSONDecodeError:
            data = {}

        if parsed.path == "/api/run-batch":
            creator_id = data.get("creator_id", "elena_rossi")
            optimize = bool(data.get("optimize", True))
            custom_prompt = data.get("scene_prompt", "").strip()
            prompts = [custom_prompt] if custom_prompt else None

            runner = ComfyExecutionEngine()
            res = runner.run_creator_batch(creator_id=creator_id, scene_prompts=prompts, optimize=optimize)
            return self._send_json(res.to_dict())

        if parsed.path == "/api/build-workflow":
            builder = ComfyWorkflowBuilder()
            wf_type = data.get("pipeline_type", "flux_keyframe")
            creator_id = data.get("creator_id", "elena_rossi")
            prompt = data.get("scene_prompt", "sitting at a Brooklyn outdoor cafe holding an espresso cup")
            seed = int(data.get("seed", 849201442))
            guidance = float(data.get("flux_guidance", 2.2))
            lora_strength = float(data.get("lora_strength", 0.85))

            if wf_type == "wan21_i2v":
                graph = builder.build_wan21_i2v_graph(
                    creator_id=creator_id,
                    approved_image_name="assets/showcase/elena_01_cafe.jpg",
                    motion_prompt=prompt,
                    seed=seed,
                )
            else:
                graph = builder.build_flux_keyframe_graph(
                    creator_id=creator_id,
                    scene_prompt=prompt,
                    seed=seed,
                    flux_guidance=guidance,
                    lora_strength=lora_strength,
                )
            return self._send_json({"workflow_type": wf_type, "node_count": len(graph), "comfy_api_graph": graph})

        self.send_error(404, "POST endpoint not found")

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[Vertex WebApp] {self.address_string()} - {fmt % args}")


def run_server(port: int = 8080) -> None:
    server = HTTPServer(("127.0.0.1", port), VertexWebHandler)
    print("=" * 74)
    print(f" VERTEX INTERACTIVE WEB APP RUNNING AT: http://localhost:{port}")
    print(" Press Ctrl+C to stop the server.")
    print("=" * 74)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Vertex WebApp...")
        server.server_close()


if __name__ == "__main__":
    run_server()
