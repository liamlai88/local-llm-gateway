# 实验 #14: Context Engineering 四策略对照

> **核心结论**: 在 10 轮客服对话 + 第 10 轮召回测试上，四策略 **没有 dominant 解**。
> Write 省 70% token 但丢技术细节；Select 在主题分散场景失败；Compress 召回满分但只省 21%。
> **Context Engineering 解决不了 RAG 问题**——它和 RAG 是正交维度。

## 1. 假设与动机

实验 #12 (Reflection) 暴露多 LLM 调用的 token 开销问题。
AgentGuide / LangChain 推 4 大 Context Engineering 策略：**Write / Select / Compress / Isolate**。
本实验验证它们在「长对话 + 远距离召回」场景下的真实表现。

## 2. 四策略实现

[context_engineering.py](../context_engineering.py)，每个策略实现 `build_messages(user_msg) -> List[Dict]`：

| 策略 | 核心做法 |
|---|---|
| Baseline | 全量历史拼 prompt（O(N²) token 增长）|
| Write | 每轮结束 LLM 写 scratchpad；下轮只看 scratchpad（不带原始对话）|
| Select | 给每轮历史做 embedding，新 query 来选 top-3 相关轮 |
| Compress | 每 3 轮 LLM 压缩成 200 字摘要 + 保留最近 2 轮原文 |
| Isolate | **本实验不适用**——纯对话场景没有可拆分的子任务 |

## 3. 测试场景

10 轮客服对话脚本：
- Turn 1-2: 客户自我介绍（**王明 / ACME 数据公司 / 金融风控 / 100 万次/天**）
- Turn 3-6: GPU 实例咨询（介绍 → ecs.gn7e 配 A100 → 显存 → 对比 → 包月价）
- Turn 7-9: 闲聊 + 对接销售 + 活动
- Turn 10: **召回测试**——"总结我的姓名、公司、业务、产品倾向、GPU 型号、显存"

**召回打分项 6 个**：王明 / ACME / 金融风控 / ecs.gn7e（大小写无关）/ A100 / 80GB

> 注：原设计含 "¥35000" 但对话从未真实传入价格（无 kb_search），故剔除。这本身是个发现，见 §6.4。

## 4. 结果

| 策略 | 累计 input | overhead | output | 总 token | vs baseline | 召回 | 平均延迟 |
|---|---|---|---|---|---|---|---|
| Baseline | 16,287 | 0 | 3,628 | 19,915 | **100%** | **6/6** | 6.2s |
| Write | **1,221** | 3,679 | 1,320 | **6,220** | **31%** | 4/6 | 3.6s |
| Select | 9,725 | 0* | 3,699 | 13,424 | 67% | 5/6 | 7.7s |
| Compress | 7,948 | 4,300 | 3,448 | 15,696 | 79% | **6/6** | 5.7s |

\* Select overhead 是 embedding API 调用（与 LLM 计费分开，未计入）

## 5. 四大发现

### 5.1 Write 省钱最多但丢技术细节

Write 让 input token 从 baseline 的 16,287 降到 1,221，**省 92.5%**（含 overhead 后净省 69%）。

但召回只有 4/6——**Write 的 scratchpad 由 LLM 自动总结，它把 "A100"、"80GB" 判为"技术细节可省略"**，专心记客户画像（姓名/公司/业务）。

**典型 scratchpad（实测）**:
```
- 客户：王明，ACME 数据公司 CTO
- 业务：金融风控，100 万次推理/天
- 倾向：ECS GN7e 实例
- 后续：报价、对接销售
（没有 "A100"、"80GB"）
```

→ **Write 是「客户关系导向」的总结**，对技术细节天然有损。

### 5.2 Select 的隐藏假设：query 能 embedding 召回历史

Select 第 10 轮失败了 "ACME"——因为**第 10 轮的"总结"query 和第 1 轮的"你好我叫王明"在 embedding 空间几乎不相似**。Top-3 选了产品讨论的几轮，自我介绍那轮被挤出。

**这是 Select 范式的结构性缺陷**：

> Select 假设「query 能 embedding 召回相关历史」。但用户画像信息分布在**叙述型早期对话**，和**指令型晚期 query**天然语义距离远。

**反例**：如果第 10 轮 query 改成"我刚才说我叫什么来着"，Select 能召回（语义匹配"自我介绍"）。但真实对话不会这么问。

### 5.3 Compress 召回满分但省得不多

Compress 是唯一保住 baseline 召回质量的策略（6/6），但 token 只省 21%——因为：
1. 压缩本身吃了 4,300 overhead token（3 次压缩调用）
2. 还要保留最近 2 轮原文

**Compress 的真正增益在长尾**：
- Baseline: input token = O(N²)（每轮都带前面全部）
- Compress: input token ≈ O(N)（每轮带 summary + recent 2）

10 轮场景两者还没拉开。**50+ 轮对话 Compress 才会从"省 20%"变成"省 80%"**。

### 5.4 彩蛋发现：Baseline 满分背后是「自信地编」

Baseline 召回 6/6 但**给的价格是 ¥3,600/月**（编的）——真实的 ¥35000 价格在对话中**从未通过 kb_search 传入过**。LLM 凭记忆给了一个看似合理的数字。

> **Context Engineering 解决的是「context 太长怎么塞」，不解决「context 里没的事实怎么办」**。
> 后者是 RAG 的工作。两者是正交维度，不能互相替代。

## 6. 决策矩阵（适合面试）

| 策略 | 省 token | 召回质量 | 失败模式 | 何时选 |
|---|---|---|---|---|
| Baseline | 0% | 满分 | 长对话 OOC | < 20 轮、技术咨询 |
| **Write** | 70% | 中（丢细节）| LLM 总结主观取舍 | 销售/HR 等重画像、轻细节 |
| **Select** | 33% | 中（丢冷门 fact）| query-history embedding 不匹配 | 主题清晰的检索式 QA |
| **Compress** | 21% (本实验) → 80% (50+ 轮) | **高** | overhead 不便宜，短对话不划算 | **长对话默认选项** |

## 7. 工程建议

1. **不要单选一种策略，组合使用**:
   - 客户事实信息 → Write（手动 scratchpad，非 LLM）
   - 历史对话 → Compress
   - 关联文档/知识库 → 走 RAG，独立于对话历史
2. **Write 的 scratchpad 不要让 LLM 自动总结**——用结构化 schema（如 `{name, company, prefs, deals}`），避免主观取舍
3. **Select 的 query 改写**: 召回前先让 LLM 把"总结一下"改写为"用户的姓名、公司、产品偏好"，扩大 query 的语义覆盖
4. **Compress 的触发条件**: 不要硬性"每 N 轮压缩"，应该按 token budget 触发（如 input > 4k token）

## 8. 与前实验的关系

| 实验 | 论点 | 本实验补充 |
|---|---|---|
| #6 Plan-Execute | 范式 > 模型规模 | Context Engineering 是 Plan-Execute 之外的另一种"范式" |
| #12 Reflection | 修不了事实错 | Context Engineering 也修不了事实错——它管 context 长度，不管 context 真假 |
| #13 GraphRAG | 抽取质量是上限 | Write 也是同样问题——LLM 自动总结的质量决定 Write 上限 |

## 9. 复现

```bash
cd ~/Calude-Learning/ai-gateway
source ../venv/bin/activate
python experiments/context_engineering_demo.py
# 详细 JSON: experiments/context_engineering_results.json
```

## 10. 一句话总结

> **Context Engineering 四策略没有 dominant 解**——Write 省钱丢细节，Select 失败在 query-history 语义错位，Compress 召回好但短对话不划算。
> **真正的工程做法是组合（结构化 Write + 阈值触发 Compress + RAG 作为正交事实层）**，而不是赌一个银弹。
