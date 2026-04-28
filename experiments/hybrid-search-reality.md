# Hybrid Search 实证报告 — 一次"失败"实验的三条真理

> **核心问题**：教科书都说 Hybrid Search 是 RAG 标配，但它真能解决一切吗？  
> **实验结论**：在 4 份专业文档 + 4 道题的小型测试中，Vector / BM25 / Hybrid **三种方法准确率都只有 50%** —— 这次"失败"的实验比成功更有价值。  
> **实验时间**：2026.04  
> **技术栈**：百炼 text-embedding-v2 + ChromaDB + rank-bm25 + jieba + RRF

---

## 实验设计

### 文档库（4 份阿里云 GPU 产品文档）

| ID | 内容要点 | 关键专有名词 |
|----|----------|--------------|
| `product_a` | A10 实例规格 | `ecs.gn7i-c8g1.2xlarge`, `NVIDIA A10` |
| `product_b` | A100 实例规格 | `ecs.gn7e-c12g1.3xlarge`, `NVIDIA A100` |
| `product_c` | 灵骏 H100 集群 | `PAI-Lingjun`, `NVIDIA H100` |
| `policy` | 使用规范 | "A10 系列"、"A100 实例"、"灵骏" |

**关键设计**：`policy` 故意写成"长杂文档"，覆盖多个产品；`product_*` 写成"短专文档"，只讲一个产品。这模拟了企业真实文档结构。

### 测试题（精心设计 4 类）

| # | 问题 | 类别 | 期望命中 |
|---|------|------|----------|
| Q1 | A100 实例多少钱一小时？ | 🔢 短英数字串 | product_b |
| Q2 | 我想做大模型推理，用哪个产品？ | 🧠 语义匹配 | product_b |
| Q3 | ecs.gn7i-c8g1.2xlarge 是什么型号？ | 🔣 长 ID 串 | product_a |
| Q4 | 训练超大规模模型走什么流程？ | 🌐 概念性 | policy |

### 三种检索方法
- **Vector**：百炼 text-embedding-v2，余弦相似度 Top-K
- **BM25**：jieba 中文分词 + rank-bm25
- **Hybrid**：RRF (Reciprocal Rank Fusion)，k=60

---

## 实验结果

### 准确率统计

| 方法 | 准确率 | Q1 | Q2 | Q3 | Q4 |
|------|--------|----|----|----|----|
| Vector | 2/4 = **50%** | ❌ policy | ❌ policy | ✅ product_a | ✅ policy |
| BM25 | 2/4 = **50%** | ❌ policy | ❌ product_a | ✅ product_a | ✅ policy |
| Hybrid | 2/4 = **50%** | ❌ policy | ❌ policy | ✅ product_a | ✅ policy |

### 关键失败案例

**Q1 "A100 多少钱"**：所有方法都选了 policy（"A10 系列"）而不是 product_b（"NVIDIA A100"）

```
BM25 score:
  Q1 A100 → policy:    0.1848  ← 信号微弱
  Q3 ecs.gn7i-... →    1.8980  ← 信号强 10 倍
```

**Q2 "大模型推理"**：Vector 选了 policy 而不是包含"千亿参数推理"的 product_b

```
product_b 文档内容: "适用场景: 大模型训练、千亿参数推理、HPC 高性能计算"
policy 文档内容:    "GPU 实例... AI 推理... 训练..." (整段语义密度高)
```

---

## 三条 RAG 真理（从失败中提炼）

### 真理 1: Tokenization 决定检索上限 🔪

```
jieba.lcut("A100 实例多少钱")  → ["a100", "实例", "多少钱"]
jieba.lcut("ecs.gn7i-c8g1...")  → ["ecs.gn7i-c8g1.2xlarge"]  ← 整体一个 token
```

| 类型 | BM25 表现 | 原因 |
|------|-----------|------|
| 短英数字串（A100、Q4） | ❌ 弱 | 容易和其他词冲突 |
| 长 ID 串（ecs.gn7e-...） | ✅ 强 | IDF 极高，独特性强 |
| 中文术语（"灵骏"） | 取决于词典 | jieba 词典覆盖度 |

**生产级方案**：
- 自定义 jieba 词典，把产品型号、专有名词作为整体 token
- 对短编号做预处理（A100 → `_A100_`）增强独特性
- 考虑用 BPE 而非词典分词

---

### 真理 2: Chunking 边界稀释语义信号 ✂️

`policy` 文档同时提到 A10、A100、H100 三个产品，整段语义向量是**三者的"平均"**。

当用户问"大模型推理"时：
- product_b 的向量很专（只讲 A100），但**整体语义密度**没 policy 高
- Vector 检索基于**整段相似度**，policy 反而胜出

**生产级方案**：
- **细粒度 chunking**：长杂文档按规则项/段落切，每块只讲一件事
- **多层次索引**：文档级 + 段落级 + 句子级，分别检索后融合
- **Late Chunking**（前沿）：先 embed 整篇文档，再切 token-level 向量

---

### 真理 3: Hybrid 是必要不充分条件 🔀

```
Q1: Vector → policy ❌
    BM25   → policy ❌
    RRF 融合 → policy ❌  (同质化错误，Hybrid 无能为力)
```

**RRF 融合的是"多样性"**，当两种方法都犯同一个错误时，融合也救不了。

**生产级方案** —— 真正的企业 RAG 三层架构：

```
[Query]
   │
   ├─→ Vector  (Top-20)  ┐
   │                     │
   ├─→ BM25    (Top-20)  ├─→ RRF 融合 → Top-10
   │                     │
   └─→ (可选) 多查询扩展  ┘             │
                                        ▼
                            ┌─── Rerank Model ───┐
                            │  bge-reranker-v2    │
                            │  Cohere Rerank      │
                            └─────────┬───────────┘
                                      │
                                      ▼
                                  Top-3 最终结果
```

**Rerank 的本质**：用专门训练的 cross-encoder 模型，把 query 和每个候选文档**真正放在一起**计算相关性，比向量内积准确得多。

---

## 与社区"标准答案"的差异

| 来源 | 主张 | 本实验观察 |
|------|------|-----------|
| OpenAI Cookbook | RAG = Embedding + Top-K | 简化得失真 |
| LangChain 教程 | Hybrid > Vector > BM25 | 不一定，看语料 |
| 各类博客 | "RAG 解决幻觉" | 解决知识缺失，但**检索本身可能误导**生成 |
| **本实验** | **Hybrid 不是银弹**，需要 tokenization + chunking + rerank 三管齐下 | ✅ |

---

## 商业决策模型 — 给客户讲方案时的话术

> "客户经常问：我能不能用 LangChain 的 RAG 模板直接套？
>
> 答案是：**Demo 可以，生产不行**。我们做过实验：4 份产品文档 + 4 道题，简单 Hybrid 准确率只有 50%。原因不是算法不行，而是**这三个隐藏陷阱**：
>
> 1. **专有名词的分词** — 你的产品型号在 jieba 默认词典里吗？
> 2. **文档的切块策略** — 你的'公司手册'有没有被切成单一文档？
> 3. **第三层 Rerank** — 你有没有把 Top-20 候选交给 reranker 精排？
>
> 这三个问题不解决，准确率就停在 50-70%。**真正生产级 RAG 不是接个向量库就行，是一整套 pipeline 工程**。"

---

## 下一步实验计划

- [ ] **加 Rerank 层**：用 bge-reranker 对 Top-10 精排，看准确率是否破 90%
- [ ] **改进 chunking**：policy 按规则项切，每块独立成文，重测准确率
- [ ] **自定义分词词典**：把 A100、A10、H100 等加入 jieba 词典
- [ ] **加权 Hybrid**：调整 vector/BM25 权重比，看是否突破 50% 天花板
- [ ] **扩大语料**：100+ 文档时 BM25 的 IDF 区分度是否改善

---

## 实验脚本

完整可复现：[`hybrid_search_demo.py`](hybrid_search_demo.py)  
RAG 模块（含三种检索）：[`../rag.py`](../rag.py)

```python
# 三种模式只需切换 mode 参数
POST /v1/rag/query
{
  "question": "...",
  "mode": "vector" | "bm25" | "hybrid",
  "top_k": 3
}
```

---

## 核心金句（面试用）

1. **"RAG 工程的难度被严重低估，OpenAI Cookbook 是 demo，不是生产"**
2. **"Hybrid Search 不是银弹，它解决的是'方法多样性'，不是'同质化错误'"**
3. **"Tokenization 是检索质量的天花板 —— jieba 默认词典 ≠ 你的业务词典"**
4. **"长杂文档会稀释短专业文档的语义信号 —— chunking 边界比 chunk 大小更重要"**
