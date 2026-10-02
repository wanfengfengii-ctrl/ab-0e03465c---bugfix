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
用 Fenwick 树做最长链 DP。

注意字典序目标在“追加同一条边”的扩展下并非全部严格保持：配对数与
残差和是可加量，严格优势严格保持；但最大残差取 max，前缀的严格优势
可能被后缀中权值更大的边追平，追平后须由边序号元组裁决。因此
``(count, sum)`` 两级用普通 Fenwick 合并即可，第三级则在前向/反向
``(count, sum)`` DP 圈定的“可承载全局最优链”的边上，保留按
``(序号元组↑, 最大残差≤)`` 定义的 Pareto 前沿，杜绝把追平后会凭
序号翻盘的候选提前丢弃。同一 i 的边先统一查询、再统一更新，保证一个
脉冲至多使用一次。
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


def _prune_frontier(nodes: list[tuple]) -> list[tuple]:
    """按 (序号元组↑, 最大残差严格下降) 保留 Pareto 前沿。

    节点为四元组 ``(maxr, seq, edge_k, prev)``（``prev`` 为前驱节点或
    None），组内所有链的 (count, sum) 相同。若 seq 字典序更小且 maxr
    不更大，则追加任意相同后缀后前者恒优（等长前缀扩展严格保持 seq 的
    字典序），后者可安全丢弃。这正是第三级不能与前两级共用 Fenwick
    合并的原因：max 的前缀优势可能被后缀更大的权值追平，追平后须由
    seq 裁决，而被丢弃的前缀可能恰好持有更小的 seq。
    """
    nodes.sort(key=lambda n: (n[1], n[0]))  # seq 升序，maxr 次序之
    result: list[tuple] = []
    best_max = None
    for node in nodes:
        maxr, seq = node[0], node[1]
        if result and result[-1][1] == seq:
            continue  # seq 相同即同一条边序列，maxr 也必相同
        if best_max is None or maxr < best_max:
            result.append(node)
            best_max = maxr
    return result


def _scalar_at_delta(a: list[int], b: list[int], delta: int, tol: int):
    """返回 (count, sum_abs, max_abs)——三级目标的“值”，不含配对边。

    标量 Fenwick 在 (count↑, sum↓, maxr↓) 上合并：被丢弃的链与保留的
    链 (count, sum) 相同且 maxr 不更小，追加任意相同后缀后前者在三个
    值上都不会反超，故三级最优值正确；但打平后用于裁决的序号元组可能
    选错，序号正确的配对由 ``_exact_at_delta`` 在唯一获胜偏移上重建。
    """
    nb = len(b)
    tree_c = [0] * (nb + 1)
    tree_s = [0] * (nb + 1)
    tree_m = [0] * (nb + 1)

    def update(pos, c, s, mr):
        p = pos + 1
        while p <= nb:
            if c > tree_c[p] or (c == tree_c[p] and
                                 (s < tree_s[p] or (s == tree_s[p] and
                                                    mr < tree_m[p]))):
                tree_c[p] = c
                tree_s[p] = s
                tree_m[p] = mr
            p += p & -p

    def query(pos):
        c = s = mr = 0
        p = pos + 1
        while p > 0:
            if tree_c[p] > c or (tree_c[p] == c and
                                 (tree_s[p] < s or (tree_s[p] == s and
                                                    tree_m[p] < mr))):
                c, s, mr = tree_c[p], tree_s[p], tree_m[p]
            p -= p & -p
        return c, s, mr

    best_c = best_s = best_m = 0
    for i, ai0 in enumerate(a):
        ai_d = ai0 + delta
        lo_j = _lower_bound(b, ai_d - tol)
        hi_j = _upper_bound(b, ai_d + tol)
        js = list(range(hi_j - 1, lo_j - 1, -1))
        if not js:
            continue
        queries = [query(j - 1) if j > 0 else (0, 0, 0) for j in js]
        for j, (pc, ps, pm) in zip(js, queries):
            w = abs(ai_d - b[j])
            c = pc + 1
            s = ps + w
            mr = w if pc == 0 else max(pm, w)
            update(j, c, s, mr)
            if c > best_c or (c == best_c and
                              (s < best_s or (s == best_s and mr < best_m))):
                best_c, best_s, best_m = c, s, mr
    return best_c, best_s, (best_m if best_c else None)


def _exact_at_delta(a: list[int], b: list[int], delta: int, tol: int):
    """返回固定偏移下的 (count, sum_abs, max_abs, chosen_edges)。

    chosen_edges 为 (i, j) 列表（0 基，i、j 均升序），在三级目标值最优
    的前提下序号元组字典序最小。

    配对数与残差和是可加量，用 (count, sum) 的前向/反向 Fenwick 最长链
    DP 圈定可承载全局最优链的边；最大残差不具备严格可合并性（前缀优势
    可能被后缀更大的权值追平，追平后须由序号元组裁决），故第三级只在
    这些边上枚举以边结尾的候选，按 (seq↑, maxr) 保留 Pareto 前沿后回溯。
    """
    edges = _edges_at(a, b, delta, tol)
    m = len(edges)
    if m == 0:
        return 0, 0, None, []

    weights = [abs(a[i] + delta - b[j]) for i, j in edges]
    nb = len(b)
    fw_c, fw_s = _prefix_dp(edges, weights, nb, forward=True)
    bk_c, bk_s = _prefix_dp(edges, weights, nb, forward=False)

    best_count = max(fw_c)
    best_sum = min(s for c, s in zip(fw_c, fw_s) if c == best_count)

    # 可属于某条 (best_count, best_sum) 全局最优链的边
    rel_idx = [
        k for k in range(m)
        if fw_c[k] + bk_c[k] - 1 == best_count
        and fw_s[k] + bk_s[k] - weights[k] == best_sum
    ]

    # states[k]：以边 k 结尾、恰好达到 (fw_c[k], fw_s[k]) 的 Pareto 前沿
    states: list[list[tuple]] = [[] for _ in range(m)]
    for k in rel_idx:
        i, j = edges[k]
        w = weights[k]
        target_c, target_s = fw_c[k], fw_s[k]
        candidates: list[tuple] = []
        if target_c == 1 and target_s == w:
            candidates.append((w, ((i + 1, j + 1),), k, None))
        for k2 in rel_idx:
            if k2 == k:
                break  # 边表按 i 升序；其后不可能再有前驱
            i2, j2 = edges[k2]
            if i2 < i and j2 < j \
                    and fw_c[k2] == target_c - 1 \
                    and fw_s[k2] == target_s - w:
                for prev_node in states[k2]:
                    pmax, pseq = prev_node[0], prev_node[1]
                    candidates.append((
                        max(pmax, w),
                        pseq + ((i + 1, j + 1),),
                        k,
                        prev_node,
                    ))
        states[k] = _prune_frontier(candidates)

    # 链头必须自身就达到全局 (count, sum)；按 max↓、seq↑ 取最优
    winner = None
    for k in rel_idx:
        if fw_c[k] != best_count or fw_s[k] != best_sum:
            continue
        for node in states[k]:
            maxr, seq = node[0], node[1]
            if winner is None or maxr < winner[0] or \
                    (maxr == winner[0] and seq < winner[1]):
                winner = (maxr, seq, node)

    node = winner[2]
    chosen_uids: list[int] = []
    while node is not None:
        chosen_uids.append(node[2])
        node = node[3]
    chosen_uids.reverse()
    chosen = [edges[u] for u in chosen_uids]
    return best_count, best_sum, winner[0], chosen


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
    # 扫描键只含三个值目标与偏移：(count, -sum, -max, -delta)。
    # 序号元组是同偏移内的裁决，不能由合并标量的 DP 给出，故扫描结束后
    # 在唯一获胜偏移上用 _exact_at_delta 重建序号正确的配对。
    best_key = None
    best_delta = offset_lo

    # 记录每个边界点的 (count, sum)，供段存活判断复用
    boundary_values: dict[int, tuple[int, int]] = {}

    def consider(delta: int):
        nonlocal best_key, best_delta
        count, sumr, maxr = _scalar_at_delta(a, b, delta, tolerance)
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

    # 获胜偏移唯一（第 4 级“偏移最小”先于序号裁决）；在该偏移上做
    # 序号精确的 Pareto DP，三级目标值应与扫描结果完全一致。
    exact_count, exact_sum, exact_max, best_chosen = _exact_at_delta(
        a, b, delta, tolerance
    )
    assert (exact_count, exact_sum, exact_max) == (count, sumr, maxr), (
        exact_count, exact_sum, exact_max, count, sumr, maxr
    )

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
