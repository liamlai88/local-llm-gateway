# LoRA 微调突破报告 — 100 条样本让 1.5B 小模型超越 Turbo

> **核心问题**：Plan-Execute 实验里 1.5B 因为 JSON 输出不稳定准确率仅 25%。能否用 LoRA 微调救它？  
> **本实验**：用 100 条手工样本 + MLX-LM 在 M5 上做 LoRA 微调（25 分钟），让 1.5B 学会工具规划 JSON 协议。  
> **结论**：**工具正确率 40% → 100%，步数正确率 40% → 100%。1.5B 在 Agent 规划任务上的表现超过未微调的 Qwen-Turbo**。  
> **实验时间**：2026.04

---

## 实验设计

### 微调目标
让 Qwen2.5-1.5B-Instruct 学会输出标准的 Plan-Execute JSON 工具调用计划：

```json
{"plan": [
  {"step": 1, "tool": "calculator", "args": {"expression": "..."}, "purpose": "..."},
  ...
]}
```

### 数据生成（关键！）
**100 条手工模板生成的高质量样本**，覆盖 6 类场景：

| 类别 | 数量 | 特点 |
|------|------|------|
| 简单计算 | 25 | 单工具 calculator |
| 单工具天气 | 15 | get_weather |
| 多工具气温差 | 15 | **5 步链式工具** ⭐ |
| 单工具知识库 | 15 | kb_search |
| RAG + 计算 | 20 | **kb_search → extract_number → calculator** ⭐ |
| 无工具问候 | 10 | 边界场景：空 plan |

**8:1:1 划分**：80 train / 10 valid / 10 test

### 训练配置（M5 24GB 实测）

| 参数 | 值 | 理由 |
|------|----|----|
| 基础模型 | mlx-community/Qwen2.5-1.5B-Instruct-bf16 | MLX 优化版 |
| LoRA 层数 | 8 | M5 上只调最后 8 层（够用） |
| Iterations | 200 | 80 条 × batch 1 ≈ 10 epoch |
| Learning rate | 1e-5 | LoRA 标准小学习率 |
| Batch size | 1 | M5 保守设置 |
| Max seq length | 1024 | JSON 计划不会超 |
| **训练耗时** | **~25 分钟** | M5 风扇全速 |

---

## 实验结果

### 5 道独立测试题（不在训练集）

| ID | 类别 | 基础 1.5B | **LoRA 1.5B** |
|----|------|-----------|---------------|
| T1 | 简单计算 (37 × 8) | ✅ 1 步正确 | ✅ 1 步正确 |
| T2 | 单工具天气 (纽约) | ✅ 1 步正确 | ✅ 1 步正确 |
| T3 | **多工具气温差 (迪拜-首尔)** | ❌ **只想了 2 步**（伪表达式）| ✅ **完美 5 步** |
| T4 | **RAG + 计算 (H100×5天)** | ❌ **只想了 1 步**（缺提取+计算）| ✅ **完整 3 步** |
| T5 | 无工具问候 (你好) | ❌ 乱填 kb_search | ✅ **空 plan + direct_answer** |
| **总分** | **2/5 = 40%** | **5/5 = 100%** ⭐ |

### 详细对比 - T3 (多工具气温差)

**基础 1.5B 输出**（错）：
```json
{"plan": [
  {"step": 1, "tool": "kb_search", "args": {"query": "迪拜和首尔的气温差"}},
  {"step": 2, "tool": "calculator", "args": {"expression": "提取的迪拜气温 - 提取的首尔气温"}}
]}
```
↑ 工具选错（用 kb_search 查实时天气），表达式是伪占位符

**LoRA 1.5B 输出**（对）：
```json
{"plan": [
  {"step": 1, "tool": "get_weather", "args": {"city": "迪拜"}, "purpose": "查迪拜气温"},
  {"step": 2, "tool": "get_weather", "args": {"city": "首尔"}, "purpose": "查首尔气温"},
  {"step": 3, "tool": "extract_number", "args": {"text": "<step1_result>", "hint": "气温"}},
  {"step": 4, "tool": "extract_number", "args": {"text": "<step2_result>", "hint": "气温"}},
  {"step": 5, "tool": "calculator", "args": {"expression": "<step3_result> - <step4_result>"}}
]}
```
↑ 5 步完整链路，包含占位符语法

### 详细对比 - T4 (Plan-Execute 实验里的 Q4)

**基础 1.5B**（错）：只规划了 1 步 kb_search

**LoRA 1.5B**（对）：完整 3 步链
```json
{"plan": [
  {"step": 1, "tool": "kb_search", "args": {"query": "H100 按量付费每小时价格"}},
  {"step": 2, "tool": "extract_number", "args": {"text": "<step1_result>", "hint": "..."}},
  {"step": 3, "tool": "calculator", "args": {"expression": "<step2_result> * 120"}}
]}
```

### 详细对比 - T5 (无工具问候)

**基础 1.5B**：乱编 `kb_search(北京天气)` ← 训练分布外瞎填

**LoRA 1.5B**：
```json
{"plan": [], "direct_answer": "你好！我是任务规划器，我可以帮您做什么？"}
```
↑ **学会了"什么时候不调用工具"**，这是边界感的体现

---

## 三个核心洞察

### 洞察 1: LoRA = 给小模型"塞"特定任务的"专家肌肉"

```
模型容量: 1.5B 参数 (固定)
LoRA 加的额外参数: ~0.5% (8 层 × LoRA rank)
                   ↑
        "在不改变模型整体能力的前提下，给某个任务装一个专家模块"
```

**面试理解**：
- LoRA 不是让模型变聪明，是让模型变**专科**
- 通用任务（写邮件、聊天）不会更好，**特定任务（你的工具协议）能极致优化**

### 洞察 2: 100 条样本 + 25 分钟，超过 Qwen-Turbo

| 配置 | 部署成本 | 准确率 | 推理延迟 |
|------|----------|--------|----------|
| 基础 1.5B | 本地免费 | 40% | <1s |
| **LoRA 1.5B (本实验)** | **本地免费** | **100%** | **<1s** |
| Qwen-Turbo (Plan-Execute) | ¥/百万 token | 75% | 5-10s |
| Qwen-Max | 更贵 | ~95% | 10-20s |

**LoRA 1.5B 在我们这个特定任务上 = 比 Qwen-Turbo 又快又准又便宜**。

### 洞察 3: LoRA 学会了"边界感" (T5)

T5 测试是个隐藏陷阱：基础 1.5B 看到"你好"还是要硬编一个工具调用（训练数据偏向）。LoRA 微调因为见过 10 条无工具样本，**学会了"什么时候输出空 plan"**。

**这才是 Agent 工程的精髓**：知道**何时不行动**比知道何时行动更重要。

---

## 给客户讲 LoRA 的话术

> "客户经常问：我能不能不用 GPT-4，用便宜模型搞定 Agent？
>
> 答：可以，但要付出'数据收集成本'。
>
> 我做过实验：**100 条手工样本 + 25 分钟 LoRA 微调**，让 1.5B 模型在工具规划任务上从 40% → 100%，效果超过未微调的 Qwen-Turbo。
>
> **客户的真实选型公式**：
> ```
> 如果业务场景固定 + 流量大 → LoRA + 小模型 (本地零成本)
> 如果业务场景多变 + 流量小 → 直接 Qwen-Plus/Max (省研发成本)
> 如果有海量历史对话数据 → 全参微调 (终极方案)
> ```
>
> 决定因素是**数据是否能稳定收集**和**单 query 经济模型**，不是模型大小。"

---

## 完整 8 份报告认知曲线

```
1. Prompt Engineering ROI    → 模型容量是地板
2. RAG vs 纯 LLM            → RAG 让 0% → 100%（闭域）
3. Hybrid Search 失败        → Hybrid 不是银弹
4. Rerank 突破              → 三层架构是标配
5. ReAct Agent 边界          → Agent 不是万能
6. Plan-Execute 突破         → 范式 > 模型规模
7. MCP Server 实现           → 工具集 USB 化
8. LoRA 微调突破 ⭐ (本报告)  → 100 条样本 + 25 分钟 = 1.5B 超越 Turbo
```

完整覆盖 **Prompt → RAG → Agent → 协议 → 微调** 的 GenAI 工程全栈。

---

## 实验脚本

- 数据生成：[`generate_data.py`](../finetune/generate_data.py) - 100 条样本模板
- 训练脚本：[`train.sh`](../finetune/train.sh) - MLX-LM LoRA 训练
- 对比测试：[`compare.py`](../finetune/compare.py) - 5 题三层评判

```bash
cd finetune
python3 generate_data.py          # 生成数据 (秒级)
bash train.sh                      # 训练 (~25 分钟)
python3 compare.py                 # 对比测试 (~3 分钟)
```

---

## 后续优化方向

- [ ] **数据增量训练**：用 Plan-Execute 实际跑出来的 trace 当训练数据，无限滚雪球
- [ ] **QLoRA (4-bit 量化)**：进一步压缩，能在更小内存设备上训练
- [ ] **多任务联合微调**：把 ReAct 格式 + Plan-Execute 格式都学，让模型自适应
- [ ] **DPO 偏好对齐**：用 ReAct Turbo 失败案例做对比训练，学会"避免 Tool Use Laziness"
- [ ] **部署 LoRA 到 Ollama**：把 LoRA adapter 合并到模型权重，无缝接入现有 Gateway

---

## 核心金句（面试用）

1. **"100 条样本 + 25 分钟 LoRA = 1.5B 超过 Turbo"**：在特定任务上小模型靠微调能完胜未微调的大模型
2. **"LoRA 不是让模型变聪明，是让模型变专科"**：LoRA 适合稳定业务场景，不适合通用聊天
3. **"客户选型公式 = 业务稳定性 × 流量 × 单query经济模型"**：模型规模只是其中一个变量
4. **"边界感比能力更重要"**：T5 测试证明 LoRA 让模型学会"何时不调用"，这是 Agent 工程的精髓
