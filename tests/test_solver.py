"""求解器测试：含小规模暴力枚举对照与宽区间性能/正确性测试。"""

import random
import time
import unittest

from app.solver import ValidationError, solve

LONG_NS = 1_000_000_000


def brute_best_matching(a, b, delta, tol):
    """枚举全部可行边，O(m^2) 最长链 DP，返回与求解器同口径的最优链。"""
    edges = []
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            if abs(ai + delta - bj) <= tol:
                edges.append((i, j))
    m = len(edges)
    if m == 0:
        return 0, 0, None, []

    # 按 i 升序、j 升序排列；DP 只允许 i'<i 且 j'<j 的前驱。
    # 每条边按 (count, sum, max) 目标组合分别保留最小序号序列：
    # 序号裁决对边扩展不保序（两条前缀的 max 可能被后续更大的残差
    # 追平，届时才轮到序列分胜负），按四级键只留一条链会丢真解。
    order = sorted(range(m), key=lambda k: (edges[k][0], edges[k][1]))
    best = [None] * m  # {(count, sum, max): 最小序号序列}
    for rank, k in enumerate(order):
        i, j = edges[k]
        w = abs(a[i] + delta - b[j])
        states = {(1, w, w): ((i + 1, j + 1),)}
        for rank2 in range(rank):
            k2 = order[rank2]
            i2, j2 = edges[k2]
            if i2 < i and j2 < j:
                for (c2, s2, m2), seq2 in best[k2].items():
                    key3 = (c2 + 1, s2 + w, max(m2, w))
                    seq = seq2 + ((i + 1, j + 1),)
                    if key3 not in states or seq < states[key3]:
                        states[key3] = seq
        best[k] = states

    all_chains = [
        (c, s, mx, seq)
        for states in best
        for (c, s, mx), seq in states.items()
    ]
    count, sumr, maxr, seq = max(all_chains, key=key_of)
    chosen = [(i - 1, j - 1) for (i, j) in seq]
    return count, sumr, maxr, chosen


def key_of(cand):
    count, sumr, maxr, seq = cand
    return (count, -sumr, -maxr, _Rev(seq))


class _Rev:
    def __init__(self, v):
        self.v = v

    def __lt__(self, o):
        return self.v > o.v

    def __gt__(self, o):
        return self.v < o.v

    def __eq__(self, o):
        return isinstance(o, _Rev) and self.v == o.v


def brute_solve(a, b, lo, hi, tol, min_pairs):
    """逐纳秒枚举 + 暴力配对，作为小规模真值。"""
    best_key = None
    best_chosen = []
    best_delta = lo
    for delta in range(lo, hi + 1):
        count, sumr, maxr, chosen = brute_best_matching(a, b, delta, tol)
        maxr_key = 0 if maxr is None else maxr
        seq = tuple((i + 1, j + 1) for i, j in chosen)
        k = (count, -sumr, -maxr_key, -delta, _Rev(seq))
        if best_key is None or k > best_key:
            best_key = k
            best_chosen = chosen
            best_delta = delta
    count, neg_sum, neg_max, _, _ = best_key
    return best_delta, count, -neg_sum, (None if count == 0 else -neg_max), best_chosen


class TestAgainstBruteForce(unittest.TestCase):
    def test_random_instances(self):
        rng = random.Random(20260929)
        for trial in range(60):
            na = rng.randint(6, 9)
            nb = rng.randint(6, 9)
            a = sorted(rng.sample(range(-200, 200), na))
            b = sorted(rng.sample(range(-200, 200), nb))
            tol = rng.randint(0, 12)
            lo, hi = -25, 25
            min_pairs = rng.randint(1, min(na, nb))

            sol, meets = solve(a, b, lo, hi, tol, min_pairs)
            d, count, sumr, maxr, chosen = brute_solve(a, b, lo, hi, tol, min_pairs)

            self.assertEqual(sol.offset, d, (a, b, tol, trial))
            self.assertEqual(sol.pair_count, count, (a, b, tol, trial))
            self.assertEqual(sol.sum_abs_residual, sumr, (a, b, tol, trial))
            self.assertEqual(sol.max_abs_residual, maxr, (a, b, tol, trial))
            got_chosen = [(p.index_a - 1, p.index_b - 1) for p in sol.pairs]
            self.assertEqual(got_chosen, chosen, (a, b, tol, trial))
            self.assertEqual(meets, count >= min_pairs)

            # 配对必须保序、一一对应，且残差在容差内
            idxs_a = [p.index_a for p in sol.pairs]
            idxs_b = [p.index_b for p in sol.pairs]
            self.assertEqual(idxs_a, sorted(idxs_a))
            self.assertEqual(idxs_b, sorted(idxs_b))
            self.assertEqual(len(set(idxs_a)), len(idxs_a))
            self.assertEqual(len(set(idxs_b)), len(idxs_b))
            for p in sol.pairs:
                self.assertLessEqual(p.abs_residual, tol)
                self.assertEqual(p.residual, p.time_a_corrected - p.time_b)

    def test_dense_instances(self):
        rng = random.Random(4242)
        for trial in range(20):
            n = rng.randint(6, 10)
            a = sorted(rng.sample(range(0, 40), n))
            b = [x + rng.randint(-3, 3) for x in a]
            for k in range(1, len(b)):  # 抖动后强制严格递增
                if b[k] <= b[k - 1]:
                    b[k] = b[k - 1] + 1
            tol = 2
            lo, hi = -6, 6
            sol, _ = solve(a, b, lo, hi, tol, 1)
            d, count, sumr, maxr, chosen = brute_solve(a, b, lo, hi, tol, 1)
            self.assertEqual(
                (sol.offset, sol.pair_count, sol.sum_abs_residual,
                 sol.max_abs_residual),
                (d, count, sumr, maxr),
                (a, b, trial),
            )
            got = [(p.index_a - 1, p.index_b - 1) for p in sol.pairs]
            self.assertEqual(got, chosen)


class TestKnownCases(unittest.TestCase):
    def test_identical_sequences_zero_offset(self):
        a = [10, 20, 30, 40, 50, 60]
        b = [10, 20, 30, 40, 50, 60]
        sol, meets = solve(a, b, -100, 100, 0, 6)
        self.assertTrue(meets)
        self.assertEqual(sol.offset, 0)
        self.assertEqual(sol.pair_count, 6)
        self.assertEqual(sol.sum_abs_residual, 0)
        self.assertEqual(sol.max_abs_residual, 0)
        self.assertEqual(sol.unpaired_a, [])
        self.assertEqual(sol.unpaired_b, [])

    def test_constant_shift(self):
        shift = 5
        a = [0, 100, 200, 300, 400, 500, 600]
        b = [x + shift for x in a]
        sol, meets = solve(a, b, -50, 50, 0, 7)
        self.assertTrue(meets)
        self.assertEqual(sol.offset, shift)  # a + 5 = b
        self.assertEqual(sol.pair_count, 7)
        self.assertTrue(all(p.residual == 0 for p in sol.pairs))

    def test_noise_does_not_steal_real_coincidence(self):
        # 若贪心先把 A 与“噪声”B 配上，会挤掉真实链；联合优化必须保住 4 对真值。
        a = [0, 100, 200, 300, 400, 500]
        # 真值：b 中 1,101,201,301,401,501（偏移 +1，容差 0 可配 6 对）；
        # 插入若干靠近但会诱导短视贪心的噪声点。
        b = sorted([1, 99, 101, 199, 201, 301, 399, 401, 501])
        sol, meets = solve(a, b, -3, 3, 1, 6)
        self.assertTrue(meets)
        self.assertEqual(sol.offset, 1)
        self.assertEqual(sol.pair_count, 6)
        self.assertEqual(sol.sum_abs_residual, 0)

    def test_threshold_not_met(self):
        a = [0, 1000, 2000, 3000, 4000, 5000]
        # 两组几乎完全错开，容差内最多配上 2 对
        b = [50, 1050, 2050, 3050, 4050, 5050]
        sol, meets = solve(a, b, 0, 0, 0, 4)
        self.assertFalse(meets)
        self.assertLess(sol.pair_count, 4)
        self.assertEqual(sol.pair_count, 0)

    def test_threshold_partial(self):
        a = [0, 100, 200, 300, 400, 500]
        b = [0, 100, 200, 350, 460, 570]
        sol, meets = solve(a, b, 0, 0, 1, 5)
        self.assertFalse(meets)
        self.assertEqual(sol.pair_count, 3)

    def test_offset_picks_minimum_residual_sum(self):
        # 多个偏移都能达到 6 对，但残差和 6*|delta-3| 在 delta=3 处最小
        a = [0, 10, 20, 30, 40, 50]
        b = [3, 13, 23, 33, 43, 53]
        sol, _ = solve(a, b, -10, 10, 5, 6)
        self.assertEqual(sol.offset, 3)
        self.assertEqual(sol.pair_count, 6)
        self.assertEqual(sol.sum_abs_residual, 0)

    def test_offset_tie_mirror_plateau(self):
        # 中心 0（3 条边）与 6（3 条边）：delta=0 与 delta=6 的残差多重集
        # 镜像，sum=18、max=6 完全相同；而 delta=3 处 max=3 严格更优。
        a = [0, 100, 200, 300, 400, 500]
        b = [0, 100, 200, 306, 406, 506]
        sol, meets = solve(a, b, -6, 12, 6, 6)
        self.assertTrue(meets)
        self.assertEqual(sol.offset, 3)
        self.assertEqual(sol.pair_count, 6)
        self.assertEqual(sol.sum_abs_residual, 18)
        self.assertEqual(sol.max_abs_residual, 3)

    def test_stable_determinism(self):
        a = [1, 5, 9, 20, 24, 30]
        b = [2, 6, 10, 21, 25, 31, 40]
        results = [solve(a, b, -50, 50, 3, 4) for _ in range(5)]
        offsets = [s.offset for s, _ in results]
        self.assertEqual(len(set(offsets)), 1)
        chosen = [[(p.index_a, p.index_b) for p in s.pairs] for s, _ in results]
        self.assertTrue(all(c == chosen[0] for c in chosen))

    def test_tie_broken_by_smallest_index_sequence(self):
        # 同分扩展场景：[(1,2),(2,3),(3,4),(4,5),(5,6),(7,7)] 与
        # [(1,1),(2,2),(3,4),(4,5),(5,6),(7,7)] 的配对数、残差和、
        # 最大残差三项目标完全相同（6 / 23 / 8），必须按输入序号序列
        # 字典序稳定选择后者，未配对集合随之确定。
        a = [4, 7, 11, 18, 22, 23, 24]
        b = [0, 5, 8, 11, 12, 13, 28, 34]
        sol, meets = solve(a, b, -1, -1, 8, 6)
        self.assertTrue(meets)
        self.assertEqual(sol.offset, -1)
        self.assertEqual(sol.pair_count, 6)
        self.assertEqual(sol.sum_abs_residual, 23)
        self.assertEqual(sol.max_abs_residual, 8)
        self.assertEqual(
            [(p.index_a, p.index_b) for p in sol.pairs],
            [(1, 1), (2, 2), (3, 4), (4, 5), (5, 6), (7, 7)],
        )
        self.assertEqual(sol.unpaired_a, [6])
        self.assertEqual(sol.unpaired_b, [3, 8])


class TestWideOffsetRange(unittest.TestCase):
    def test_two_billion_span(self):
        # 真实偏移约 +1e9，区间跨度 20 亿，不得逐纳秒扫描
        a = [10, 210, 410, 610, 810, 1010, 1210, 1410]
        b = [x + LONG_NS for x in a]  # delta = 1_000_000_000
        lo, hi = -LONG_NS, LONG_NS  # 跨度 2_000_000_000
        started = time.perf_counter()
        sol, meets = solve(a, b, lo, hi, 0, 8)
        elapsed = time.perf_counter() - started
        self.assertTrue(meets)
        self.assertEqual(sol.offset, LONG_NS)
        self.assertEqual(sol.pair_count, 8)
        self.assertEqual(sol.sum_abs_residual, 0)
        self.assertLess(elapsed, 3.0, f"宽区间求解耗时 {elapsed:.2f}s，疑似逐纳秒扫描")

    def test_two_billion_span_with_jitter_and_fail(self):
        a = [0, 100, 200, 300, 400, 500, 600]
        b = [x + LONG_NS + d for x, d in zip(a, [0, 1, -1, 0, 1, -1, 2])]
        # 相对偏移为 [0,1,-1,0,1,-1,2]：容差 1 下 δ=0 可配前 6 个
        # （残差 0,-1,1,0,-1,1，和为 4），第 7 个抖动 2 无法纳入；
        # δ=1 只能保 5 个（偏移 -1 的两处脱落）。
        sol, meets = solve(a, b, -LONG_NS, LONG_NS, 1, 6)
        self.assertTrue(meets)
        # δ=1e9：A0..A5 与 B0..B5 配对（残差 0,-1,1,0,-1,1）；
        # δ=1e9-100：A1..A6 与 B0..B5 错位配对，残差多重集完全相同，
        # count=6/sum=4/max=1 三级全部持平，按第 4 级取更小偏移。
        self.assertEqual(sol.offset, LONG_NS - 100)
        self.assertEqual(sol.pair_count, 6)
        self.assertEqual(sol.sum_abs_residual, 4)
        self.assertEqual(sol.max_abs_residual, 1)
        self.assertEqual(sol.unpaired_a, [1])
        self.assertEqual(sol.unpaired_b, [7])
        # 门槛 7 无法满足：实际最大配对数 6
        sol3, meets3 = solve(a, b, -LONG_NS, LONG_NS, 1, 7)
        self.assertEqual(sol3.pair_count, 6)
        self.assertFalse(meets3)

    def test_large_negative_offset_boundary(self):
        a = [5, 15, 25, 35, 45, 55]
        b = [x - LONG_NS for x in a]  # delta = -1e9，恰为区间端点
        sol, meets = solve(a, b, -LONG_NS, LONG_NS, 0, 6)
        self.assertTrue(meets)
        self.assertEqual(sol.offset, -LONG_NS)
        self.assertEqual(sol.pair_count, 6)


class TestValidation(unittest.TestCase):
    def _good_kwargs(self):
        return dict(
            a=[1, 2, 3, 4, 5, 6],
            b=[2, 3, 4, 5, 6, 7],
            offset_lo=-10,
            offset_hi=10,
            tolerance=1,
            min_pairs=3,
        )

    def assertInvalid(self, **overrides):
        kw = self._good_kwargs()
        kw.update(overrides)
        with self.assertRaises(ValidationError):
            solve(**kw)

    def test_too_few_or_many(self):
        self.assertInvalid(a=[1, 2, 3, 4, 5])
        self.assertInvalid(a=list(range(25)))

    def test_not_strictly_increasing(self):
        self.assertInvalid(a=[1, 2, 2, 4, 5, 6])
        self.assertInvalid(a=[1, 2, 0, 4, 5, 6])

    def test_bad_types(self):
        self.assertInvalid(a=[1, 2, 3, 4, 5, 6.0])
        self.assertInvalid(tolerance=-1)
        self.assertInvalid(offset_lo=10, offset_hi=-10)
        self.assertInvalid(min_pairs=0)
        self.assertInvalid(min_pairs=7)

    def test_empty_interval_still_evaluated(self):
        kw = self._good_kwargs()
        kw.update(offset_lo=100, offset_hi=100, min_pairs=1)
        sol, meets = solve(**kw)
        self.assertFalse(meets)
        self.assertEqual(sol.offset, 100)
        self.assertEqual(sol.pair_count, 0)
        self.assertIsNone(sol.max_abs_residual)


if __name__ == "__main__":
    unittest.main(verbosity=2)
