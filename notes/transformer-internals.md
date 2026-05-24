# Transformer 内部机制笔记（实证导向）

> Week 6 Day 1 学习产出。每个概念挂到本项目 11 份实验里某个具体发现，避免脱离实践讲抽象架构。

## 1. Token 化：一切的起点

文本 → 整数序列。Qwen 用 BPE，词表 ~150k。

**关联实验 #8 (LoRA)**：100 条样本能"教会" 1.5B 做 extract_number，本质是微调改了 embedding 层 + attention 权重对**特定 token 模式**的响应。不是"理解数学"，是"看到这种 token 序列就输出那种 token 序列"。

## 2. Embedding：token → 向量

每个 token 查表得到 d 维向量（Qwen2.5 1.5B 的 d=1536）。

**关联实验 #2 (RAG)**：`text-embedding-v2` 输出 1536 维，**和 Qwen 内部 embedding 维度同源**（百炼整套配套）。向量检索能"语义匹配"——同一个语义空间。

## 3. Self-Attention：核心机制

每个 token 看其他所有 token，算"我该多关注谁"。

三件套：
- **Q (Query)**：我在找什么
- **K (Key)**：我能提供什么
- **V (Value)**：我实际的内容

公式：`attention = softmax(QK^T / √d) · V`

**关联实验 #3, #4**：Vector/BM25/Hybrid 都卡 50%——因为它们在"语义相似"层面打转。而 attention 在生成时是 token 级精细对齐，rerank 模型（gte-rerank）本质是小型 cross-attention，所以加 rerank 后从 50% → 100%。

## 4. 多头注意力（Multi-Head）

把 Q/K/V 切成 N 份并行算，最后拼起来。1.5B 通常 12-16 头。

**直觉**：不同头学不同关注模式——语法头、指代头、远距离依赖头。

## 5. 位置编码：Transformer 的"时间感"

self-attention 本身**对顺序不敏感**（打乱 token 结果一样）。Qwen 用 **RoPE**（旋转位置编码），把位置信息乘进 Q/K。

**关联**：32K context 版本的 "32K" 就是 RoPE 的有效外推范围。即将做的实验 #14（Context Engineering）的 Compress/Isolate，本质都是绕开 **RoPE 长距离衰减**问题。

## 6. FFN（前馈层）：知识存储

每层 attention 后跟两层 MLP，宽度通常 4×d。**模型的事实知识大部分存这里**。

**关联实验 #1**："1.5B + CoT 反退化"——1.5B 的 FFN 容量不够存复杂推理链所需的中间事实。"模型容量是地板"这个结论的物理基础就在 FFN。

## 7. 为什么 1.5B 不支持 tool_calls？

tool_calls 协议要求模型输出**严格 JSON Schema**：

```json
{"tool_calls": [{"name": "...", "arguments": {...}}]}
```

要做到这点需要：
1. **指令遵循能力**——RLHF/DPO 训出来的对齐
2. **结构化输出稳定性**——FFN 要记住 function calling 这套 token 模式
3. **足够 attention 头**理解"何时调用 vs 何时回答"

Qwen2.5-1.5B 缺的是**对齐数据规模**和**FFN 容量**。不是架构缺陷，是**训练数据缺陷**。

**这就是为什么 LoRA 实验 #8 成功**：100 条样本，在 attention 权重上加低秩适配，让它学会"看到这种 query 就输出这种 JSON"。LoRA 没改架构，只改输出分布。

## 8. 解码：从向量回到 token

最后一层输出 → vocab 大小 logits → softmax → 采样（temperature/top-p）。

**关联实验 #5**："Tool Use Laziness" 是采样问题。1.5B 在 temperature=0 时倾向输出训练集里高频的"直接回答"路径。Few-shot 之所以双刃剑——它在 prompt 里强行抬高"调工具"路径的概率，但也会**过拟合到示例模式**。

---

## 一句话总结

> Transformer = embedding 把 token 变向量 + attention 让 token 互相看 + FFN 存知识 + 位置编码给顺序 + 解码采样。
> **容量决定地板，对齐决定天花板，11 份实验全是在这两条线上验证。**

---

## 面试可能问到的点

- **Q: 为什么 RAG 能让 1.5B 闭域 0% → 100%？**
  A: 因为 RAG 把"知识"从 FFN 卸载到 context，1.5B 的 FFN 容量瓶颈被绕开，模型只需做"基于 context 的指代和拼接"——这是 attention 的本职工作。

- **Q: 为什么 Plan-Execute 比 ReAct 强 50 个点？**
  A: ReAct 每轮要在同一个 forward pass 里既"思考"又"决策调工具"，对 1.5B 的 attention 头是过载。Plan-Execute 把规划和执行**拆到不同 LLM 调用**，每次只解决一个子问题，避开了容量瓶颈。

- **Q: LoRA 到底改了什么？**
  A: 在 attention 的 Q/K/V 投影矩阵旁挂一对低秩矩阵 (rank=8 或 16)，前向时加到原矩阵。原参数冻结，只训低秩部分。10MB adapter 就能让模型在窄任务上完胜大模型。
