"""HTTP API 集成测试：启动真实服务，走网络请求验证。"""

import json
import os
import threading
import time
import unittest
import urllib.error
import urllib.request

from app.server import create_server

LONG_NS = 1_000_000_000


class ApiIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = create_server(0)  # 由内核分配空闲端口
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.05)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _post(self, payload):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/calibrate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode("utf-8"))

    def _get(self, path):
        with urllib.request.urlopen(
            f"http://127.0.0.1:{self.port}{path}", timeout=5
        ) as resp:
            return resp.status, resp.read()

    def test_health(self):
        status, body = self._get("/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "ok")

    def test_index_served(self):
        status, body = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn("双探头".encode("utf-8"), body)

    def test_calibrate_success_wide_range(self):
        payload = {
            "times_a": [0, 100, 200, 300, 400, 500, 600, 700],
            "times_b": [x + LONG_NS for x in [0, 100, 200, 300, 400, 500, 600, 700]],
            "offset_lo": -LONG_NS,
            "offset_hi": LONG_NS,
            "tolerance": 0,
            "min_pairs": 8,
        }
        status, data = self._post(payload)
        self.assertEqual(status, 200)
        self.assertTrue(data["meets_threshold"])
        self.assertTrue(data["calibration_possible"])
        self.assertEqual(data["offset"], LONG_NS)
        self.assertEqual(data["pair_count"], 8)
        self.assertEqual(data["sum_abs_residual"], 0)
        self.assertEqual(len(data["pairs"]), 8)
        # 每对返回校正时间与带符号残差
        p0 = data["pairs"][0]
        self.assertEqual(p0["time_a_corrected"], LONG_NS)
        self.assertEqual(p0["time_b"], LONG_NS)
        self.assertEqual(p0["residual"], 0)
        self.assertEqual(data["unpaired_a"], [])
        self.assertEqual(data["unpaired_b"], [])

    def test_calibrate_below_threshold_not_fabricated(self):
        payload = {
            "times_a": [0, 1000, 2000, 3000, 4000, 5000],
            "times_b": [50, 1050, 2050, 3050, 4050, 5050],
            "offset_lo": 0,
            "offset_hi": 0,
            "tolerance": 0,
            "min_pairs": 4,
        }
        status, data = self._post(payload)
        self.assertEqual(status, 200)
        self.assertFalse(data["meets_threshold"])
        self.assertFalse(data["calibration_possible"])
        self.assertEqual(data["pair_count"], 0)
        # 必须说明实际最大配对数与原因，而不是伪造校准值
        self.assertIn("0", data["reason"])
        self.assertIn("4", data["reason"])

    def test_validation_error_returns_400(self):
        payload = {
            "times_a": [1, 2, 3, 4, 5],  # 少于 6 个
            "times_b": [1, 2, 3, 4, 5, 6],
            "offset_lo": -1,
            "offset_hi": 1,
            "tolerance": 0,
            "min_pairs": 1,
        }
        status, data = self._post(payload)
        self.assertEqual(status, 400)
        self.assertIn("error", data)

    def test_missing_field_400(self):
        status, data = self._post({"times_a": [1] * 6})
        self.assertEqual(status, 400)

    def test_bad_json_400(self):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/calibrate",
            data=b"not-json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(cm.exception.code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
