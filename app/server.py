"""基于标准库的 HTTP 服务：静态页面 + 校准 API + 健康检查。"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .solver import ValidationError, solve

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _solution_to_dict(sol, meets: bool, min_pairs: int, payload: dict) -> dict:
    result = {
        "meets_threshold": meets,
        "min_pairs": min_pairs,
        "offset": sol.offset,
        "pair_count": sol.pair_count,
        "sum_abs_residual": sol.sum_abs_residual,
        "max_abs_residual": sol.max_abs_residual,
        "pairs": [
            {
                "index_a": p.index_a,
                "index_b": p.index_b,
                "time_a_corrected": p.time_a_corrected,
                "time_b": p.time_b,
                "residual": p.residual,
                "abs_residual": p.abs_residual,
            }
            for p in sol.pairs
        ],
        "unpaired_a": sol.unpaired_a,
        "unpaired_b": sol.unpaired_b,
    }
    if not meets:
        result["calibration_possible"] = False
        result["reason"] = (
            f"在给定偏移区间与符合容差下，最多只能形成 {sol.pair_count} 对符合事件，"
            f"低于要求的最低配对数 {min_pairs}，无法据此形成可信校准。"
        )
    else:
        result["calibration_possible"] = True
    return result


class Handler(BaseHTTPRequestHandler):
    server_version = "CoincidenceCalib/1.0"

    def log_message(self, fmt, *args):  # 静默默认访问日志，保留简洁
        pass

    def _send_json(self, status: int, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path, content_type: str) -> None:
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            self._send_json(404, {"error": "not found"})
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/health" or path == "/healthz":
            self._send_json(200, {"status": "ok"})
        elif path == "/" or path == "/index.html":
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif path == "/app.js":
            self._send_file(STATIC_DIR / "app.js", "application/javascript; charset=utf-8")
        elif path == "/styles.css":
            self._send_file(STATIC_DIR / "styles.css", "text/css; charset=utf-8")
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path != "/api/calibrate":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b""
            payload = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._send_json(400, {"error": "请求体必须是合法的 JSON"})
            return
        if not isinstance(payload, dict):
            self._send_json(400, {"error": "请求体必须是 JSON 对象"})
            return

        required = ("times_a", "times_b", "offset_lo", "offset_hi",
                    "tolerance", "min_pairs")
        missing = [k for k in required if k not in payload]
        if missing:
            self._send_json(400, {"error": f"缺少参数: {', '.join(missing)}"})
            return
        try:
            sol, meets = solve(
                payload["times_a"],
                payload["times_b"],
                payload["offset_lo"],
                payload["offset_hi"],
                payload["tolerance"],
                payload["min_pairs"],
            )
        except ValidationError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        self._send_json(200, _solution_to_dict(sol, meets, payload["min_pairs"], payload))


def create_server(port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("0.0.0.0", port), Handler)


def main() -> None:
    port = int(os.environ.get("APP_PORT", "8080"))
    server = create_server(port)
    print(f"校准服务已启动，监听 0.0.0.0:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
