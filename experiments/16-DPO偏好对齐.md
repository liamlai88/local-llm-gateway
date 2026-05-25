# 实验 #16: DPO 偏好对齐 —— 当 Base 模型已经够强时

> **核心结论**: 在 100 条偏好对训练 Qwen2.5-1.5B-Instruct 后，DPO 让 Markdown 包裹率从 20% → 10%、平均输出长度 -8.5%，**但 JSON/Plan 合规率本来就 100%，DPO 边际增益小**。
> **当 base 已经经过完整 RLHF，DPO 主要修「边角缺陷」而非「能力地板」**——这与论文里"DPO 是 RLHF 替代品"的叙事很不一样。

## 1. 假设与动机

实验 #8 用 SFT-LoRA 让 1.5B 学会输出 JSON 工具调用计划，准确率 40% → 100%。
AgentGuide 推荐的下一步是 **DPO（Direct Preference Optimization）**——2023 年斯坦福提出的 RLHF 替代品，**用偏好对训练，不需要 reward model**。

**待验证假设**:
- H1: DPO 能进一步消除 base 模型的"格式漂移"问题（markdown 包裹、散文前缀）
- H2: 100 条偏好对足以让 1.5B 在窄任务上对齐
- H3: SFT 和 DPO 的分工是什么——什么场景该用哪个？

## 2. DPO 原理简述

DPO 的核心公式：

```
loss = -log(σ(β · [log(π(chosen)/π_ref(chosen)) - log(π(rejected)/π_ref(rejected))]))
```

直觉：让 policy 模型相比 reference 模型，**对 chosen 输出概率上升，对 rejected 输出概率下降**。β 是温度参数（实验设 0.1）。

**vs RLHF 三步法**（PPO + reward model + rollout）：
| 维度 | RLHF | DPO |
|---|---|---|
| 训练步骤 | 3 步（SFT → RM → PPO）| 1 步 |
| 需要 reward model | 是 | 否 |
| 训练稳定性 | 低（PPO 难调参）| 高 |
| 数据要求 | 偏好对 + 大量 rollout | 仅偏好对 |
| 算力 | 高 | 中 |

## 3. 实现

### 3.1 偏好数据集（100 条）

[build_dpo_dataset.py](../finetune/build_dpo_dataset.py)——复用实验 #8 的 100 条 SFT 数据，每条做 4 种 corruption 生成 rejected：

| Corruption 类型 | 占比 | 典型 rejected |
|---|---|---|
| Markdown 包裹 | 30% | ```` ```json {...} ``` ```` |
| 散文前缀 | 30% | "好的，以下是计划：\n{...}" |
| 删 purpose 字段 | 20% | `{"step":1,"tool":"calc","args":{}}`（缺 purpose）|
| 错误工具名 | 20% | "tool": "calc"（应为 "calculator"）|

输出格式 (TRL DPOTrainer 兼容)：
```json
{"prompt": "<|im_start|>system\n...\n<|im_start|>user\n...\n<|im_start|>assistant\n",
 "chosen": "{\"plan\": [...]}",
 "rejected": "```json\n{...}\n```"}
```

### 3.2 训练

[train_dpo.py](../finetune/train_dpo.py)：
- 基础模型: Qwen/Qwen2.5-1.5B-Instruct（HF safetensors，从 ModelScope 镜像下载）
- LoRA: r=8, alpha=16, target_modules=q_proj/v_proj
- DPO: β=0.1, lr=5e-6, 2 epochs, batch=1 + grad_accum=4
- 设备: Apple M5 MPS（fp32，MPS 不稳定支持 fp16/bf16 训练）
- 训练时间: ~15 分钟

## 4. 评测结果

10 道测试题，base vs DPO 对比：

| 指标 | Base | DPO | 差值 |
|---|---|---|---|
| JSON 合规率 | 100% | 100% | 持平 |
| Plan 结构正确率 | 100% | 100% | 持平 |
| **Markdown 包裹率** | **20%** | **10%** | **-50%** ✓ |
| 散文前缀率 | 0% | 0% | 持平 |
| 平均输出长度 | 183.4 | 167.8 | **-8.5%** |

**唯一显著的行为差异在 Q9 "谢谢"**：

| 模型 | 输出 |
|---|---|
| Base | ```` ```json {"plan": [{"tool": "calculator", "args": {}, "purpose": "无需使用工具"}]} ``` ```` (163 chars) |
| DPO | `{"plan": []}` (12 chars) |

DPO 不只去掉了 markdown，还**学会了"无需调工具就给空 plan"**，比 base 更合理。

**Q4 "再见" 仍未修复**：两个模型都还在用 markdown + 编造 calculator 步骤。说明 80 条训练样本里没覆盖到"空闲聊"边界。

## 5. 三个核心发现

### 5.1 Base 模型已经强，DPO 是「修边角」

Qwen2.5-1.5B-Instruct 经过完整 RLHF 训练，**JSON 合规率本来就 100%**。DPO 边际增益只能从 markdown 20% → 10%。

> **这与论文里"DPO 是 RLHF 替代品"的叙事很不一样**——当 base 已经对齐过，DPO 不能再造一遍奇迹，只能微调风格。

### 5.2 偏好数据覆盖度决定 DPO 上限

80 条训练样本覆盖了"计算/查询/RAG"类问题，但**没覆盖"空查询/闲聊"边界**——结果 Q4 ("再见") 和 Q9 ("谢谢") 中只有 Q9 被修复（因为它的失败模式正好被某条训练样本"撞上"了）。

**工程教训**: DPO 数据集**必须覆盖所有失败模式**，靠 LLM 生成的 corruption 不够，要从真实日志里挖。

### 5.3 DPO 的副作用：输出更简洁

平均长度 -8.5%。这是因为 chosen 都是无 markdown 无前缀的紧凑 JSON，模型学会了"长度的偏好"。

> **隐性效益**: 输出更短 → token 成本更低。对 100 万次/天的场景，每次省 15 token = 月省 ¥27（按 qwen-turbo 价格）。

## 6. SFT vs DPO：分工矩阵

| 维度 | SFT (#8 LoRA) | DPO (#16) |
|---|---|---|
| **教什么** | 「做什么」(behavior) | 「怎么做更好」(preference) |
| **数据** | 100 条对话 | 100 条偏好对 |
| **能力增量** | 40% → 100%（学会全新能力）| 80% → 90%（修边角）|
| **训练目标** | 最大化 chosen 概率 | 相对 ref 提升 chosen / 降低 rejected |
| **何时用** | base 模型缺这个能力 | base 模型偶尔失败的边界 |

**P7 面试金句**:
> **「SFT 教模型『做什么』，DPO 教模型『怎么做更好』。」** 我做的 #8 LoRA 让 1.5B 学会从 0 输出 JSON 工具调用（40%→100%），#16 DPO 把已经会输出的 1.5B 的 markdown 包裹率从 20% 降到 10%。**两者不替代，而是接力——SFT 拉地板，DPO 抬天花板。**

## 7. 与 RLHF 的对比（理论补充）

实际训练 RLHF 需要：
1. SFT 模型（已有 #8）
2. **Reward Model**：再训一个判分模型（需要更多偏好数据 + 算力）
3. **PPO**：用 RM 做 reward signal，rollout 训 policy

DPO 把 (2)+(3) 折叠成一个损失函数：**不显式建 reward model，而是把"偏好"直接编码进 policy loss**。

**关键洞察**: DPO 的简洁来自数学等价——在 KL 约束下，最优 policy 和 reward function 有闭式关系。DPO 利用这点直接绕过 RM。

## 8. 工程建议

1. **不要为了 DPO 而 DPO**：先看 base 模型的失败模式，如果失败率 < 5%，DPO 性价比低，不如改 prompt
2. **偏好数据必须覆盖真实失败模式**：从日志挖 > LLM 生成 corruption
3. **β 参数怎么调**：β 大（如 1.0）模型更激进偏离 ref；β 小（如 0.01）更保守。0.1 是常见起点
4. **MPS 训练限制**：M5 上 fp16 DPO 训练不稳，要用 fp32，显存占用更高
5. **DPO 之后还能继续叠 KTO/IPO**：2024 年新的偏好对齐变种，但 DPO 仍是工业标准

## 9. 跨实验回看

| 实验 | 微调技术 | 数据规模 | 能力增量 |
|---|---|---|---|
| #8 | SFT-LoRA | 100 对话 | 40% → 100% |
| **#16** | **DPO-LoRA** | **100 偏好对** | **80% → 90%（边角）** |

加上 RAG (#2/#4)、Plan-Execute (#6)、Multi-Agent (#9)、Context Engineering (#14)——**完整的"提升小模型能力"工具箱**。

## 10. 复现

```bash
cd ~/Calude-Learning/ai-gateway
source ../venv/bin/activate

# 步骤 1: 生成偏好数据集
python finetune/build_dpo_dataset.py

# 步骤 2: 训练（M5 ~15 分钟，需 ~3GB 模型缓存）
python finetune/train_dpo.py

# 步骤 3: 评测
python finetune/eval_dpo.py
# 详细结果: experiments/dpo_eval_results.json
```

## 11. 一句话总结

> **DPO 是「锦上添花」，不是「雪中送炭」**。
> 当 base 模型经过完整 RLHF，DPO 只能修边角；当 base 模型差，先用 SFT 拉地板，再用 DPO 抬天花板。
> **100 条偏好对就能改变模型行为**——但前提是这 100 条覆盖了真实的失败模式分布。
