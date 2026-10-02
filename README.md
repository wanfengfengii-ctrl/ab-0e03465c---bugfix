# 双探头脉冲符合校准服务

核医学质控场景：对齐同一标准源在两台探头上的脉冲记录。系统**联合**选择一个
整数时钟偏移与一个保持输入顺序的一对一符合配对，而不是先估时钟偏移再贪心配对
（后者会让噪声脉冲占用真实符合事件）。

## 优化目标（字典序）

对整数偏移 δ（A 的校正时间 = `a_i + δ`）和保序一对一配对，依次：

1. **配对数最多**；
2. 配对数相同时，**残差绝对值总和最小**（残差 = `a_i + δ - b_j`）；
3. 再使**最大残差最小**；
4. 再使**偏移最小**；
5. 再按**输入序号稳定裁决**（配对边的 (i, j) 序号序列字典序最小）。

仅当 `|a_i + δ - b_j| ≤ 符合容差` 时该配对才允许。

## 算法（不逐纳秒扫描）

- 固定 δ 下，允许的配对构成 DAG；最大保序一对一配对即最长链，用 **Fenwick 树**
  做带权最长路 DP（同属一个脉冲的边“先查后写”，杜绝重复使用）。
- 每条边的可行偏移是区间 `[b_j-a_i-τ, b_j-a_i+τ]`。以所有 `c-τ、c、c+τ`
  为段边界：段内边集与残差符号恒定，链的残差和是 δ 的仿射单调函数，因此前两级
  最优必落在段边界；第三级（最大残差）的拐点只可能在两个中心的半整数中点，
  第二阶段仅对能承载全局最优链的边中心补算其中点。
- 候选偏移数为 O((nₐn_b)²)（n≤24），**与偏移区间跨度无关**——跨度 20 亿纳秒
  也从配对关系的临界值精确求解，最坏约 1.5s，典型场景毫秒级。

若最大配对数低于最低配对数门槛，页面与 API 都**不会给出校准值**，而是如实返回
实际最大配对数及无法形成足够符合事件的原因。

## 运行

```bash
# 默认 8080
docker compose up app

# 自定义宿主机/容器端口
APP_PORT=9090 docker compose up app
```

打开 `http://localhost:8080`（或自定义端口）。

### 一次性 verify 服务

`verify` 服务在应用健康检查通过后自动执行：代码编译（构建校验）、全部单元/集成
测试（含对小实例逐纳秒暴力枚举的随机对照）、跨度 20 亿纳秒的真实 HTTP API 冒烟，
并以退出码报告结果：

```bash
docker compose up --abort-on-container-exit verify
# 查看退出码
docker inspect --format '{{.State.ExitCode}}' $(docker compose ps -q verify)
```

## API

`POST /api/calibrate`

```json
{
  "times_a": [0, 100, 200, 300, 400, 500],
  "times_b": [1, 101, 201, 301, 401, 501],
  "offset_lo": -1000000000,
  "offset_hi": 1000000000,
  "tolerance": 10,
  "min_pairs": 4
}
```

返回联合最优偏移、每对的校正时间/带符号残差/绝对残差、未配对脉冲序号，以及
`meets_threshold`、`calibration_possible` 与（未达标时的）`reason`。

健康检查：`GET /health` → `{"status":"ok"}`。

## 本地测试（无需 Docker）

应用与测试仅依赖 Python 3.11 标准库：

```bash
python3 -m unittest discover -s tests
APP_HOST=127.0.0.1 APP_PORT=8080 python3 verify.py   # 另起一个终端先运行服务
```

## 目录结构

```
app/solver.py          联合优化求解器（临界值 + Fenwick 最长链）
app/server.py          标准库 HTTP 服务（静态页面 + API + 健康检查）
app/static/            录入页面（修改即失效，提交调用真实 API）
tests/                 暴力对照随机测试、定点用例、HTTP API 集成测试
verify.py              一次性校验服务入口
Dockerfile / docker-compose.yml
```
