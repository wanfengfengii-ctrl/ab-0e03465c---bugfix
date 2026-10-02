"""联合最优整数时钟偏移与保序一对一符合配对求解器。

问题
----
探头 A、B 分别记录到严格递增的整数纳秒时间序列 ``a``、``b``。
选择一个整数时钟偏移 ``delta``（A 的校正时间为 ``a_i + delta``）以及一个
保持输入顺序的一对一配对，按以下字典序依次优化：

1. 配对数最多；
2. 在配对数最多的前提下，配对残差 ``|a_i + delta - b_j|`` 的总和最小；
3. 再使最大残差最小；
4. 再使偏移最小；
5. 再按输入序号稳定裁决（配对边的 (i, j) 1 基序号序列字典序最小）。

只有残差绝对值不超过 ``tolerance`` 的脉冲对才允许配对。

算法（不逐纳秒扫描）
--------------------
对固定偏移 d，可行配对边 e = (i, j) 满足 ``|a_i + d - b_j| <= tolerance``，
即 d 落在闭区间 ``[c - tolerance, c + tolerance]``（c = b_j - a_i）内。
以全部 ``c - tol``、``c``、``c + tol`` 为“段边界”，则在相邻边界之间的
开区间内：可行边集恒定、每条边的残差符号恒定。

* 第一阶段：只评估段边界（含 offset_lo/offset_hi，至多
  ``3 * n_a * n_b + 2`` 个）。段内任一链的残差和是 d 的仿射单调函数，
  故前两级目标（配对数最大、残差和最小）的最优值必在某段边界取得。
* 第二阶段：仅当某个段两端点的（配对数、残差和）追平全局最优时，
  第三级“最大残差”才可能在段内翻转；固定符号下链的最大残差是若干
  斜率 ±1 直线的 max，其拐点只可能落在两个中心 c 的（半整数）中点处，
  故只需补评估这些中点的 floor/ceil。再用前向/反向最长链 DP 把中点
  限制在“可承载全局最优链”的边中心上，候选数进一步收敛。

候选偏移总数与偏移区间跨度（可达 20 亿纳秒）无关，不逐纳秒扫描。

每个候选偏移上把可行边视为 DAG（i、j 均严格递增的链即合法配对），
用 Fenwick 树做带权最长路 DP。边权为残差绝对值；链间按
（配对数↑, 残差和↓, 最大残差↓）比较。追加一条边时残差和相加、
最大残差取 max，该字典序在同权扩展下保持，故可安全合并；
同一 i 的边先统一查询、再统一更新，保证一个脉冲至多使用一次。
第 5 级“输入序号序列”对边扩展不保序：两条前缀若仅最大残差不同，
后续一条更大残差的边会把 max 追平，届时才轮到序列分胜负，
被剪掉的前缀可能反而更优——故序列裁决不能并入 DP 剪枝，
改为在最终选定的偏移上用后缀最小残差和表贪心重构字典序最小的
最优链（见 ``_smallest_seq_chain``）。
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------


class ValidationError(ValueError):
    """请求参数不合法。"""


def _parse_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{name} 必须是整数")
    return value


def _parse_times(value, name: str) -> list[int]:
    if not isinstance(value, list) or not (6 <= len(value) <= 24):
        raise ValidationError(f"{name} 必须包含 6 至 24 个时间")
    result: list[int] = []
    for raw in value:
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValidationError(f"{name} 中的时间必须是整数纳秒值")
        result.append(raw)
    for prev, cur in zip(result, result[1:]):
        if cur <= prev:
            raise ValidationError(f"{name} 必须严格递增")
    return result


# ---------------------------------------------------------------------------
# 结果数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    index_a: int           # 1 基输入序号
    index_b: int
    time_a_corrected: int  # a_i + delta
    time_b: int
    residual: int          # 带符号残差：(a_i + delta) - b_j
    abs_residual: int


@dataclass(frozen=True)
class Solution:
    offset: int
    pairs: list[Pair]
    unpaired_a: list[int]  # 未配对脉冲的 1 基序号
    unpaired_b: list[int]
    pair_count: int
    sum_abs_residual: int
    max_abs_residual: int | None  # 无配对时为 None


# ---------------------------------------------------------------------------
# Fenwick 树：j 前缀上的最优链（可合并半群）
# ---------------------------------------------------------------------------

# 链的比较键：（配对数↑, 残差和↓, 最大残差↓），最大化该键即为字典序最优。
# 该三项目标对“追加同一条边”保持顺序（count/sum 平移保序、max 与同一后缀
# 再取 max 保序），故按前缀合并与按边扩展都是精确的。输入序号序列不具备
# 该性质，不参与此处比较（见 _smallest_seq_chain）。


class _FenwickMax:
    __slots__ = ("_n", "_count", "_sum", "_maxr")

    def __init__(self, n: int) -> None:
        self._n = n
        self._count = [0] * (n + 1)
        self._sum = [0] * (n + 1)
        self._maxr = [0] * (n + 1)

    @staticmethod
    def _better(count, sumr, maxr, best_count, best_sum, best_maxr):
        if count != best_count:
            return count > best_count
        if sumr != best_sum:
            return sumr < best_sum
        return maxr < best_maxr

    def update(self, pos, count, sumr, maxr):
        i = pos + 1
        while i <= self._n:
            if self._better(
                count, sumr, maxr,
                self._count[i], self._sum[i], self._maxr[i],
            ):
                self._count[i] = count
                self._sum[i] = sumr
                self._maxr[i] = maxr
            i += i & -i

    def query(self, pos):
        """返回位置 [0, pos] 内最优链的 (count, sum, max)。"""
        bc, bs, bm = 0, 0, 0
        i = pos + 1
        while i > 0:
            if self._better(
                self._count[i], self._sum[i], self._maxr[i],
                bc, bs, bm,
            ):
                bc = self._count[i]
                bs = self._sum[i]
                bm = self._maxr[i]
            i -= i & -i
        return bc, bs, bm


# ---------------------------------------------------------------------------
# 二分查找
# ---------------------------------------------------------------------------


def _lower_bound(values: list[int], target: int) -> int:
    lo, hi = 0, len(values)
    while lo < hi:
        mid = (lo + hi) // 2
        if values[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _upper_bound(values: list[int], target: int) -> int:
    lo, hi = 0, len(values)
    while lo < hi:
        mid = (lo + hi) // 2
        if values[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------------------
# 固定偏移下的最优配对
# ---------------------------------------------------------------------------


def _best_at_delta(a: list[int], b: list[int], delta: int, tol: int):
    """返回固定偏移下最优链的三项目标 (count, sum_abs, max_abs)。

    max_abs 在无配对时为 None。输入序号序列裁决不在此进行——它对边扩展
    不保序，若并入剪枝会丢掉“max 被后续追平后序列更优”的链。
    """
    na, nb = len(a), len(b)

    # 按 i 升序处理可行边，同 i 的边先统一查询、再统一更新，
    # 杜绝同一 i 被使用两次。
    fen = _FenwickMax(nb)
    for i in range(na):
        ai_d = a[i] + delta
        lo_j = _lower_bound(b, ai_d - tol)
        hi_j = _upper_bound(b, ai_d + tol)
        if hi_j <= lo_j:
            continue

        # 1) 全部查询（Fenwick 中只有 i' < i 的边）
        batch = []
        for j in range(hi_j - 1, lo_j - 1, -1):
            pc, ps, pm = fen.query(j - 1) if j > 0 else (0, 0, 0)
            w = abs(ai_d - b[j])
            batch.append((j, pc + 1, ps + w, max(pm, w)))

        # 2) 全部更新
        for j, count, sumr, maxr in batch:
            fen.update(j, count, sumr, maxr)

    count, sumr, maxr = fen.query(nb - 1)
    return count, sumr, (maxr if count else None)


def _smallest_seq_chain(
    a: list[int],
    b: list[int],
    delta: int,
    tol: int,
    need_count: int,
    need_sum: int,
    need_max: int,
):
    """固定偏移下，达到 (need_count, need_sum, need_max) 的输入序号序列
    字典序最小链，返回 0 基 (i, j) 升序列表。

    只保留残差 <= need_max 的可行边：此时任何达到 (need_count, need_sum)
    的链其最大残差自动等于 need_max（need_max 已是三项目标下的最优值，
    不存在更小的可行链）。suffix[k][i][j] 记录仅用 i' >= i、j' >= j 的边
    组成 k 条边链的最小残差和；据此贪心逐位选取仍能补全为
    (need_count, need_sum) 链的最小下一条边，即为字典序最小解。
    """
    if need_count == 0:
        return []
    na, nb = len(a), len(b)
    inf = float("inf")

    weights: list[list[int | None]] = [[None] * nb for _ in range(na)]
    for i in range(na):
        ai_d = a[i] + delta
        lo_j = _lower_bound(b, ai_d - tol)
        hi_j = _upper_bound(b, ai_d + tol)
        for j in range(lo_j, hi_j):
            res = abs(ai_d - b[j])
            if res <= need_max:
                weights[i][j] = res

    # suffix[k][i][j]：全部边满足 i' >= i、j' >= j 的 k 边链的最小残差和
    suffix: list[list[list[float]]] = [
        [[0] * (nb + 1) for _ in range(na + 1)]  # k = 0：空链，残差和 0
    ]
    for k in range(1, need_count + 1):
        prev = suffix[k - 1]
        table = [[inf] * (nb + 1) for _ in range(na + 1)]
        for i in range(na - 1, -1, -1):
            row, row_below, wrow, prev_below = (
                table[i], table[i + 1], weights[i], prev[i + 1],
            )
            for j in range(nb - 1, -1, -1):
                best = row_below[j]
                if row[j + 1] < best:
                    best = row[j + 1]
                wij = wrow[j]
                if wij is not None:
                    start = wij + prev_below[j + 1]
                    if start < best:
                        best = start
                row[j] = best
        suffix.append(table)

    chosen: list[tuple[int, int]] = []
    spent = 0
    ci = cj = 0  # 下一条边须满足 i >= ci、j >= cj（严格大于上一条）
    for k in range(need_count, 0, -1):
        rest = suffix[k - 1]
        budget = need_sum - spent
        found = False
        for i in range(ci, na):
            wrow, rrow = weights[i], rest[i + 1]
            for j in range(cj, nb):
                wij = wrow[j]
                if wij is not None and wij + rrow[j + 1] <= budget:
                    chosen.append((i, j))
                    spent += wij
                    ci, cj = i + 1, j + 1
                    found = True
                    break
            if found:
                break
        if not found:  # 不变式保证不会发生：suffix[k][ci][cj] <= 剩余预算
            raise AssertionError("最优链重构失败")
    return chosen


# ---------------------------------------------------------------------------
# 候选临界偏移
# ---------------------------------------------------------------------------


def _critical_points(
    a: list[int], b: list[int], tol: int, offset_lo: int, offset_hi: int
):
    """返回 (边界点, 中心列表)。

    边界点 = offset_lo/offset_hi 以及落在区间内的
    ``c - tol``、``c``、``c + tol``（c = b_j - a_i）。
    边集与残差符号只在边界点变化；相邻边界点之间的开区间为一个“段”。
    """
    boundaries = {offset_lo, offset_hi}
    centers: list[int] = []
    for ai in a:
        for bj in b:
            c = bj - ai
            centers.append(c)
            for critical in (c - tol, c, c + tol):
                if offset_lo <= critical <= offset_hi:
                    boundaries.add(critical)
    return sorted(boundaries), centers


def _midpoint_candidates(centers: list[int], offset_lo: int, offset_hi: int):
    """所有中心对的半整数中点 (c1+c2)/2 的 floor 与 ceil（去重排序）。

    段内残差符号固定时，一条链的最大残差是若干斜率 ±1 直线的 max，
    其拐点只可能出现在两个中心的中点处；整数决策取 floor/ceil 即可。
    """
    uniq = sorted(set(centers))
    out = set()
    for x in range(len(uniq)):
        cx = uniq[x]
        for y in range(x, len(uniq)):
            total = cx + uniq[y]
            low = total // 2
            high = low if total % 2 == 0 else low + 1
            if offset_lo <= low <= offset_hi:
                out.add(low)
            if offset_lo <= high <= offset_hi:
                out.add(high)
    return out


def _edges_at(a, b, delta, tol):
    """delta 下的可行边 (i, j)，按 i 升序、同 i 内 j 降序。"""
    edges = []
    for i, ai0 in enumerate(a):
        ai_d = ai0 + delta
        lo_j = _lower_bound(b, ai_d - tol)
        hi_j = _upper_bound(b, ai_d + tol)
        for j in range(hi_j - 1, lo_j - 1, -1):
            edges.append((i, j))
    return edges


def _optimal_edge_centers(a, b, delta, tol, need_count, need_sum):
    """标记在 delta 下可属于“（配对数, 残差和）全局最优链”的边，返回其中心。

    前向 Fenwick 求以每条边结尾的最优前缀，反向求最优后缀；
    拼接后恰为 (need_count, need_sum) 的边才可能承载第三级裁决。
    """
    edges = _edges_at(a, b, delta, tol)
    m = len(edges)
    if m == 0:
        return []

    weights = [abs(a[i] + delta - b[j]) for i, j in edges]

    # ---- 前向：以边 k 结尾的最优 (count, sum) ----
    fw_c, fw_s = _prefix_dp(edges, weights, len(b), forward=True)
    # ---- 反向：以边 k 开头的最优 (count, sum) ----
    bk_c, bk_s = _prefix_dp(edges, weights, len(b), forward=False)

    centers = []
    for k, (i, j) in enumerate(edges):
        if fw_c[k] + bk_c[k] - 1 == need_count and \
                fw_s[k] + bk_s[k] - weights[k] == need_sum:
            centers.append(b[j] - a[i])
    return centers


def _prefix_dp(edges, weights, nb, forward):
    """对边表做 (count↑, sum↓) 的 Fenwick 最长链 DP。

    forward=True 求以每条边结尾的最优前缀；False 求以每条边开头的
    最优后缀（翻转 i、j 的扫描方向与索引）。
    """
    m = len(edges)
    out_c = [0] * m
    out_s = [0] * m

    # Fenwick 内聚 (count, sum)；同 count 取 sum 小
    tree_c = [0] * (nb + 1)
    tree_s = [0] * (nb + 1)

    def better(c, s, tc, ts):
        return c > tc or (c == tc and s < ts)

    def tree_update(pos, c, s):
        p = pos + 1
        while p <= nb:
            if better(c, s, tree_c[p], tree_s[p]):
                tree_c[p] = c
                tree_s[p] = s
            p += p & -p

    def tree_query(pos):
        c, s = 0, 0
        p = pos + 1
        while p > 0:
            if better(tree_c[p], tree_s[p], c, s):
                c, s = tree_c[p], tree_s[p]
            p -= p & -p
        return c, s

    # 边表按 i 升序、同 i 内 j 降序。反向时 i 降序，同 i 内 j 升序，
    # 同样需要“同 i 先查询后更新”。按 i 分批：
    def run_batch(batch):
        queries = []
        for k in batch:
            _, j = edges[k]
            if forward:
                q = tree_query(j - 1) if j > 0 else (0, 0)
            else:
                rj = nb - 1 - j
                q = tree_query(rj - 1) if rj > 0 else (0, 0)
            queries.append(q)
        for k, (pc, ps) in zip(batch, queries):
            i, j = edges[k]
            c = pc + 1
            s = ps + weights[k]
            out_c[k], out_s[k] = c, s
            pos = j if forward else nb - 1 - j
            tree_update(pos, c, s)

    if forward:
        idx = 0
        while idx < m:
            i = edges[idx][0]
            batch = []
            while idx < m and edges[idx][0] == i:
                batch.append(idx)
                idx += 1
            run_batch(batch)
    else:
        idx = m - 1
        while idx >= 0:
            i = edges[idx][0]
            batch = []
            while idx >= 0 and edges[idx][0] == i:
                batch.append(idx)
                idx -= 1
            run_batch(batch)

    return out_c, out_s


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------


def solve(
    a: list[int],
    b: list[int],
    offset_lo: int,
    offset_hi: int,
    tolerance: int,
    min_pairs: int,
) -> tuple[Solution, bool]:
    """求联合最优解。

    返回 ``(solution, meets_threshold)``。当最大配对数低于 ``min_pairs``
    时 ``meets_threshold`` 为 False——调用方必须展示实际最大配对数与
    无法形成足够符合事件的事实，不得伪造校准值。
    """
    a = _parse_times(a, "探头A时间序列")
    b = _parse_times(b, "探头B时间序列")
    offset_lo = _parse_int(offset_lo, "偏移下界")
    offset_hi = _parse_int(offset_hi, "偏移上界")
    tolerance = _parse_int(tolerance, "符合容差")
    min_pairs = _parse_int(min_pairs, "最低配对数")

    if offset_hi < offset_lo:
        raise ValidationError("偏移区间下界不能大于上界")
    if tolerance < 0:
        raise ValidationError("符合容差不能为负")
    max_possible = min(len(a), len(b))
    if not (1 <= min_pairs <= max_possible):
        raise ValidationError(f"最低配对数须在 1 至 {max_possible} 之间")

    boundaries, centers = _critical_points(
        a, b, tolerance, offset_lo, offset_hi
    )

    # ---- 第一阶段：评估全部段边界点 ----
    # 最大化比较键：(count, -sum, -max, -delta)。第 5 级“输入序号序列”
    # 排在偏移之后，而候选偏移两两不同，跨候选比较不会触及该级；
    # 只需在最终选定的偏移上重构字典序最小的最优链。
    best_key = None
    best_delta = offset_lo

    # 记录每个边界点的 (count, sum) 与完整结果，供段存活判断复用
    boundary_values: dict[int, tuple[int, int]] = {}

    def consider(delta: int):
        nonlocal best_key, best_delta
        count, sumr, maxr = _best_at_delta(a, b, delta, tolerance)
        boundary_values[delta] = (count, sumr)
        maxr_key = 0 if maxr is None else maxr
        key = (count, -sumr, -maxr_key, -delta)
        if best_key is None or key > best_key:
            best_key = key
            best_delta = delta

    for delta in boundaries:
        consider(delta)

    global_count = best_key[0]
    global_sum = -best_key[1]

    # ---- 第二阶段：段内部 ----
    # 段内边集与残差符号恒定，最优 (count, sum) 为仿射单调函数，
    # 故其整数值由一个边界点取得。仅当某段的 (count, sum) 追平全局最优时，
    # 第三级“最大残差”才可能在段内的边中心中点处产生更优解。
    for seg_idx in range(len(boundaries) - 1):
        p_left = boundaries[seg_idx]
        p_right = boundaries[seg_idx + 1]
        if p_right <= p_left + 1:
            continue
        left_c, left_s = boundary_values[p_left]
        right_c, right_s = boundary_values[p_right]
        seg_count = max(left_c, right_c)
        seg_sum = min(
            (s for c, s in ((left_c, left_s), (right_c, right_s)) if c == seg_count),
            default=0,
        )
        if seg_count != global_count or seg_sum != global_sum:
            continue

        # 存活段：只保留能承载全局最优链的边中心（段内边集恒定，
        # 两端点的可行边集即段内边集），再枚举这些中心对的段内中点。
        useful_centers = set()
        for probe in (p_left, p_right):
            useful_centers.update(
                _optimal_edge_centers(
                    a, b, probe, tolerance, global_count, global_sum
                )
            )
        if not useful_centers:
            continue
        local_mids = _midpoint_candidates(
            sorted(useful_centers), p_left + 1, p_right - 1
        )
        for delta in sorted(local_mids):
            consider(delta)

    count, neg_sum, neg_max, neg_delta = best_key
    delta = -neg_delta
    sumr = -neg_sum
    maxr = None if count == 0 else -neg_max

    # 第 5 级裁决：在最终偏移上取达到三项目标的字典序最小链
    best_chosen = _smallest_seq_chain(a, b, delta, tolerance, count, sumr, maxr)

    used_a = {i for i, _ in best_chosen}
    used_b = {j for _, j in best_chosen}
    pairs = []
    for i, j in best_chosen:
        corrected = a[i] + delta
        pairs.append(
            Pair(
                index_a=i + 1,
                index_b=j + 1,
                time_a_corrected=corrected,
                time_b=b[j],
                residual=corrected - b[j],
                abs_residual=abs(corrected - b[j]),
            )
        )
    solution = Solution(
        offset=delta,
        pairs=pairs,
        unpaired_a=[i + 1 for i in range(len(a)) if i not in used_a],
        unpaired_b=[j + 1 for j in range(len(b)) if j not in used_b],
        pair_count=count,
        sum_abs_residual=sumr,
        max_abs_residual=maxr,
    )
    return solution, count >= min_pairs
