"""一次性 verify 服务入口。

在应用通过健康检查后（compose 的 depends_on: service_healthy 已保证），
依次执行：

1. 代码构建校验：全部源码与测试可编译、可导入；
2. 单元/集成测试（求解器暴力对照、校验、HTTP API）；
3. 宽偏移区间（跨度 20 亿纳秒）真实 API 冒烟；

任一步失败即以非零退出码结束，供 CI / ``docker compose up`` 判断。
"""

from __future__ import annotations

import json
import os
import py_compile
import sys
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
APP_HOST = os.environ.get("APP_HOST", "app")
APP_PORT = os.environ.get("APP_PORT", "8080")
BASE_URL = f"http://{APP_HOST}:{APP_PORT}"
LONG_NS = 1_000_000_000


def wait_for_health(timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=2) as resp:
                if resp.status == 200:
                    print(f"[verify] 应用健康检查通过：{BASE_URL}/health")
                    return
        except (urllib.error.URLError, OSError) as exc:
            last_err = exc
        time.sleep(1)
    print(f"[verify] 等待应用健康超时：{last_err}", file=sys.stderr)
    raise SystemExit(1)


def build_check() -> None:
    print("[verify] 1/3 构建校验：编译全部源码与测试")
    files = sorted(BASE_DIR.rglob("*.py"))
    for path in files:
        py_compile.compile(str(path), doraise=True)
    print(f"[verify]   编译通过（{len(files)} 个文件）")


def run_tests() -> None:
    print("[verify] 2/3 代码测试：单元 + HTTP API 集成")
    loader = unittest.TestLoader()
    suite = loader.discover(str(BASE_DIR / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def api_smoke() -> None:
    print("[verify] 3/3 宽偏移区间 API 冒烟（跨度 2_000_000_000 ns）")
    payload = {
        "times_a": [0, 100, 200, 300, 400, 500, 600, 700],
        "times_b": [x + LONG_NS for x in [0, 100, 200, 300, 400, 500, 600, 700]],
        "offset_lo": -LONG_NS,
        "offset_hi": LONG_NS,
        "tolerance": 0,
        "min_pairs": 8,
    }
    req = urllib.request.Request(
        f"{BASE_URL}/api/calibrate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status == 200, f"HTTP {resp.status}"
        data = json.loads(resp.read().decode("utf-8"))
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert data["meets_threshold"] is True, data
    assert data["offset"] == LONG_NS, data["offset"]
    assert data["pair_count"] == 8, data["pair_count"]
    assert data["sum_abs_residual"] == 0
    assert len(data["pairs"]) == 8
    for p in data["pairs"]:
        assert p["time_a_corrected"] == p["time_b"]
        assert p["residual"] == p["time_a_corrected"] - p["time_b"]
    print(f"[verify]   冒烟通过：offset={data['offset']}，8 对零残差，{elapsed_ms:.0f}ms")

    # 低于门槛时不得伪造校准值：门槛合法（<= 脉冲数）但物理上无法达到
    a_bad = [0, 100, 200, 300, 400, 500, 600, 700]
    b_bad = [x + LONG_NS + d for x, d in zip(a_bad, [0, 1, -1, 0, 1, -1, 2, -2])]
    bad = {
        "times_a": a_bad,
        "times_b": b_bad,
        "offset_lo": -LONG_NS,
        "offset_hi": LONG_NS,
        "tolerance": 1,
        "min_pairs": 7,  # 8 个脉冲合法；但容差 1 下最多 6 对
    }
    req2 = urllib.request.Request(
        f"{BASE_URL}/api/calibrate",
        data=json.dumps(bad).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req2, timeout=10) as resp:
        data2 = json.loads(resp.read().decode("utf-8"))
    assert data2["meets_threshold"] is False
    assert data2["calibration_possible"] is False
    assert data2["pair_count"] == 6, data2["pair_count"]
    assert "6" in data2["reason"] and "7" in data2["reason"]
    print("[verify]   低门槛场景正确拒绝并给出实际最大配对数与原因")


def main() -> int:
    wait_for_health()
    build_check()
    run_tests()
    api_smoke()
    print("[verify] 全部检查通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
