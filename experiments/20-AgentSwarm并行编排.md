# 实验 #20: Agent Swarm 并行编排 + LLM Handoff 路由

> **核心结论**: 用 ~150 行复用 `agent.call_llm` 实现了 Agent Swarm 的三块基本盘——
> **无状态 worker / 并行 fan-out-gather / LLM 动态 handoff 路由**。
> 真实数据给出反直觉发现：**并行 fan-out 阶段拿到 3.01x 加速，但串行的 gather(汇总) 是阿姆达尔瓶颈，
> 把整体墙钟收益从 3x 稀释到 1.32x（5830ms → 4429ms）**。
> Swarm 不是"agent 越多越快"，收益上限由"不可并行的汇总/串行依赖段"决定。

## 1. 假设与动机

单 agent 顺序干活，子任务之间明明无依赖却被迫排队。Swarm 的卖点是"该并行的并行、该交接的交接"。
但教材常把 swarm 讲成"加 agent 就提速"，本实验用真实延迟数据校准这个直觉。

**待验证假设**:
- H1: 子任务相互独立时，并行 fan-out 的墙钟 ≈ 串行 / N（N=worker 数）
- H2: 需要一个串行 gather 把结果合一时，整体加速比会显著低于 N（阿姆达尔定律）
- H3: 用一次 LLM 调用做 handoff 路由，能把异构问题正确分发给对口专家

**关键工程决策**：并行实验必须用云端 `bailian/qwen-turbo`。本地 Ollama 默认 `num_parallel=1`
会串行处理并发请求，用它测"并行"会得到延迟无收益的**假结论**——这本身是一个 swarm 落地踩坑。

## 2. Swarm 三块基本盘

[swarm_demo.py](swarm_demo.py)，全部复用 `agent.call_llm`：

```
① 无状态 worker     run_agent(role_prompt, task) → 一次 LLM 调用
                    状态不藏在 agent 里，全靠外部显式传入（可调试、可并行）
② fan-out / gather  ThreadPoolExecutor 并发发 N 个 worker，再用 1 次 LLM 合并
③ LLM handoff 路由  router agent 输出 JSON {agent, reason} → 动态分发给专家
```

| 拓扑 | 本实验对应 | 适用场景 |
|---|---|---|
| Orchestrator-Worker | 实验 A/B | 子任务可拆分、需汇总 |
| Handoff / 动态路由 | 实验 C | 异构请求分流给专家 |

## 3. 实验 A/B：串行 vs 并行 fan-out

**任务**：出海 SLG《Sand Empire》三语本地化 slogan（英/阿/泰），三份彼此独立 → 天然 fan-out。
`provider=bailian, model=qwen-turbo`，单次冷跑。

| 模式 | 墙钟 | 拆解 |
|---|---|---|
| A 串行 | **5830.5 ms** | 三 worker 顺序累计（1469 + 1881 + 2481）|
| B 并行 swarm | **4429.1 ms** | fan-out **1939.7 ms** + gather 2489.4 ms |

- **fan-out 加速比 = 5830 / 1940 ≈ 3.01x**：三 worker 并发，墙钟 ≈ 最慢的那个，正好等于 worker 数。H1 成立。
- **整体只 1.32x**：gather(汇总) 是一次不可省的串行 LLM 调用（2489ms），把并行省下的时间又吃回去大半。H2 成立。

> **阿姆达尔定律的活教材**：可并行段加速 3x，但串行段(gather)占了总时长的 56%，
> 整体加速比 = 1 / (0.44/3 + 0.56) ≈ 1.4x，与实测 1.32x 吻合。
> **优化 swarm 延迟的杠杆不在"加更多 worker"，而在"压缩或砍掉串行汇总段"**
> （例如：能直接拼接就别让 LLM 改写、gather 用更快的模型、或流式输出首个 worker）。

三语输出质量正常（节选）：
- EN: *"Conquer the desert, unite with allies, rise without spending a dime."*
- AR: `استمتع بحرب الصحراء الحقيقية وابنِ إمبراطوريتك دون دفع أي ريال!`
- TH: `สู้ในทะเลทราย สร้างจักรวรรดิ ไม่ต้องเสียเงินก็ขึ้นอันดับ!`

## 4. 实验 C：LLM Handoff 路由

一次 LLM 调用当调度器，输出 `{agent, reason}` JSON，动态分发给 billing/tech/policy 三专家。

| 用户问题 | 路由到 | 理由 | 专家回答（节选）|
|---|---|---|---|
| A100 一小时多少钱？ | **billing** ✅ | 计费查询 | 约 1.2 美元/小时 |
| gateway 报 429 啥原因？ | **tech** ✅ | 接口限流 | 请求过多触发限流 |
| 中东 UGC 要过哪些审核？ | **policy** ✅ | 审核政策 | 宗教/政治/文化/反恐合规… |

**3/3 正确分发**，H3 成立。路由各 ~1.3–2.5s + 专家应答 ~1–1.8s。
代价：handoff 让每个请求多一次 LLM 往返（延迟翻倍），所以路由该用快/小模型，且能用规则命中的就别走 LLM
（对齐实验 #09「规则快路径 + LLM 兜底」的混合架构思路）。

## 5. 结论与生产清单

1. **Swarm 收益上限 = 串行依赖段**：先画 DAG，看哪些真能并行；fan-out 容易，gather 才是瓶颈。
2. **并行必须配真并发后端**：本地 Ollama `num_parallel=1` 是隐形串行陷阱，云端 API 或 vLLM 才有效。
3. **handoff 不免费**：每跳一次专家 = 一次额外 LLM 往返，路由用小模型 + 规则前置。
4. **worker 无状态**：状态显式外传，否则 swarm 不可并行也不可调试。
5. 生产还需补：全局 token/步数熔断、worker 失败隔离重试、每跳 trace（接 gateway 统一记）。

## 6. 与既有实验的关系

- 对比 **#09 Multi-Agent 混合架构**：#09 是"规则快路径 + LLM 兜底"求**准 + 省**；
  本实验是"并行 + 动态路由"求**快 + 可扩展**——两者正交，可叠加（路由层用 #09 的规则前置）。
- 对比 **#11 LangGraph**：LangGraph 的 StateGraph 本质就是把本实验的 fan-out/gather/路由
  抽象成图节点；手写一遍后再用框架，才知道框架在替你管什么（状态合并、并发调度）。

---

**面试金句**：
> "做 swarm 我先测了一组数据：三个独立 worker 并行，fan-out 拿到 3x 加速，
> 但整体墙钟只快了 1.3x——因为汇总那一步是串行的，占了一半时长。
> 这让我意识到 swarm 调优的杠杆不是堆 agent，而是阿姆达尔定律里那段串行依赖。
> 而且并行得配真并发后端，本地 Ollama 默认串行处理，会让你的'并行'白做。"
