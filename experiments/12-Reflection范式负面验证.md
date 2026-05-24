# 实验 #12: Reflection 范式负面验证 —— 自我反思的边界

> **核心结论**: Reflection 在 Plan-Execute 之上加 Critic 回路，准确率持平（87.5% → 87.5%），延迟增加 38%。
> Self-Reflection 只能修「推理错误」，不能修「事实错误」。

## 1. 假设与动机

实验 #5 (ReAct) 在 1.5B 上准确率 25%，实验 #6 (Plan-Execute) 拉到 75%。
AgentGuide 推荐的下一步是 **Reflection 范式**——让模型对自己的答案做自我审查并必要时重做。

**待验证假设**:
- H1: Reflection 能进一步突破 Plan-Execute 的天花板
- H2: Reflection 触发重试的题目集中在 Plan-Execute 失败的子集上（精准发力）

## 2. 实现

在 [agent.py](agent.py) 复用 `run_plan_execute_agent` 作为执行体，外加 Critic 节点：

```python
def run_reflection_agent(question, max_retries=1, ...):
    for attempt in range(max_retries + 1):
        # 重试时把上轮 critique 注入 question
        augmented_q = question + (f"\n\n[上轮失败原因]: {critique_hint}" if critique_hint else "")
        result = run_plan_execute_agent(augmented_q, ...)

        # Critic LLM 评判
        verdict = call_llm(critic_prompt(question, result.trace, result.answer))
        if verdict["verdict"] == "pass":
            return result
        critique_hint = verdict["fix_hint"]
```

**Critic Prompt 核心规则**:
1. 答案是否充分基于执行步骤的 observation
2. 是否遗漏问题问到的关键信息
3. 关键数字是否与 observation 一致

## 3. 实验设计

- **Provider**: 百炼 qwen-turbo（消除模型能力地板影响，专注范式差异）
- **测试集**: 8 道混合题（数学/天气查询/RAG/多步推理/单位提取/多跳）
- **对照三组**: ReAct / Plan-Execute / Plan-Execute + Reflection
- **指标**: 准确率、平均延迟、平均迭代次数、Reflection 重试触发率

## 4. 结果

| 模式 | 准确率 | 平均延迟 (ms) | 平均迭代 |
|---|---|---|---|
| ReAct | 62.5% (5/8) | 2,881 | 1.4 |
| Plan-Execute | 87.5% (7/8) | 10,464 | 2.5 |
| **Reflection** | **87.5% (7/8)** | **14,350** | **2.4** |

**Reflection 触发重试的题目**: 1/8 (Q5 RAG 题)

## 5. 失败案例深度分析

### ReAct 失败的 3 题（多步算术漏步）

| Q | 正确答案 | ReAct 输出 | 错因 |
|---|---|---|---|
| Q3 湿度差 | 55 | 60 | 心算 85-30 错 |
| Q4 (杭州+北京)*2 | 74 | 86 | 运算优先级错（22+15*2）|
| Q8 (新加坡-杭州)+5 | 13 | 9 | 漏掉 +5 步 |

→ **ReAct 在单 prompt 内既要规划又要执行，多步任务漏步是结构性问题。**

### Q5 RAG 题——三种模式全军覆没

- **正确答案**: ¥6000 (¥1000 × 6 个月)
- **三种模式都答**: ¥210000 (¥35000 × 6)
- **Reflection 表现**: `attempts=2`，Critic 触发了 1 次重试，**重试后仍错**

**根因**: `kb_search` 返回的 chunk 包含多个产品的价格，LLM 抓错了那个数字（¥35000 是另一个产品的）。

**为何 Critic 救不了**:
```
Observation: "X 产品包月 ¥35000, Y 产品包月 ¥1000..."  ← 召回有偏，但确实包含 35000
Answer:      "X 产品 ¥35000, 6 个月 ¥210000"           ← 内部自洽
Critic:      pass ✓ (35000×6=210000，逻辑正确)
```

Critic 只能验证 **answer ←→ observation 的一致性**，无法验证 **observation ←→ 真实事实 的一致性**。

## 6. 核心洞察

### 6.1 Self-Reflection 的能力边界

> **Reflection 能修「推理错误」，不能修「事实错误」。**

| 错误类型 | Reflection 能修复? | 原因 |
|---|---|---|
| 算术错误 (Q4: 86 → 74) | ✓ | Critic 重新执行计算可验证 |
| 漏步骤 (Q8: 9 → 13) | ✓ | Critic 看 plan 是否完整 |
| 召回偏差 (Q5: 35000) | ✗ | Critic 看不到"真实价格"，只看 observation |
| 工具选错 | 部分 | 取决于 Critic 是否能识别工具用错 |

### 6.2 与实验 #6 的延续

实验 #6 论点："**范式 > 模型规模**"（Plan-Execute Turbo 75% vs ReAct 25%）。
本实验补充："**范式收益有递减**"——Plan-Execute → Reflection 在通用题集上准确率不再提升。

### 6.3 Reflection 的真正用武之地（未在本实验验证，但可推断）

Reflection 应该用在 **「Plan-Execute 容易过度自信」** 的场景:
- **代码生成**: 写完后让 Critic 检查类型/边界条件
- **数学证明**: 让 Critic 反向验证每步推导
- **结构化数据抽取**: Critic 检查字段完整性

而 **不该用在**:
- RAG 主导的事实问答（Critic 无独立真值）
- 单步任务（纯粹的延迟开销）
- 工具调用稳定的简单任务

## 7. 工程教训

1. **Critic 的 ground truth 来源是关键**——只看历史 trace 的 Critic 是"内部审查员"，无法发现 RAG 召回错误。要解决 Q5 类问题应该引入"独立检索一次做交叉验证"
2. **延迟成本不可忽视**——Reflection 模式 14.4s vs Plan-Execute 10.5s，多 4s 在 P99 场景是巨大开销
3. **负面结果是更强的论点**——比"Reflection 万能"更值钱的是"Reflection 的适用边界"

## 8. 下一步实验

- **实验 #13 (GraphRAG)**: Q5 失败暴露了 RAG 召回质量问题，GraphRAG 通过结构化检索有望改善
- **实验 #14 (Context Engineering)**: Reflection 增加的 LLM 调用次数提醒我们 token 成本，下一个实验直接做 Compress/Isolate 优化
- **Week 7 RAGAS**: 把 Q5 当负样本，给 RAG 模块加召回质量评测

## 9. 复现命令

```bash
cd ~/Calude-Learning/ai-gateway
source ../venv/bin/activate
python experiments/agent_reflection_demo.py
# 详细 JSON: experiments/reflection_results.json
```

## 10. 一句话总结

> **范式叠加不是单调收益**——Plan-Execute 已经吃掉了"任务分解"的大部分增益，Reflection 在事实型任务上额外贡献近零，但延迟成本是真实的。**真正提升 RAG 类任务准确率的方向是改进召回，而不是叠 Critic。**
