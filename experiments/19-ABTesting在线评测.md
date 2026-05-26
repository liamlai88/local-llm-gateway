# 实验 #19: A/B Testing + 在线评测 + 自动回滚

> **核心结论**: 完整实现「Traffic Splitter + LLM-as-judge 在线评测 + Welch's t-test + 自动决策」工程链路。
> 真实数据反驳常识：**CoT 结构化 prompt 在简单客服场景反而 Relevance 略降（不显著）+ 长度 186% / 延迟 40%（显著）**。
> 自动决策引擎给出 `hold`——继续观察，避免拍脑袋上线。

## 1. 假设与动机

传统 prompt 优化是"改完直接全量上线"——风险高、回滚慢。
P7 SA 跟客户讲方案必须有数据驱动的发布机制。本实验做生产级 A/B 框架。

**待验证假设**:
- H1: CoT 结构化 prompt 应该提升 Faithfulness/Relevance（教材直觉）
- H2: 决策引擎能基于统计显著性自动给出 promote/rollback/hold
- H3: 30 条样本足以验证小型 prompt 改动

## 2. 工程链路设计

[ab_testing.py](../ab_testing.py):

```
新 prompt 上线
  ↓
TrafficSplitter（user_id MD5 哈希）→ 区间分流
  ↓
LLM-as-judge 在线打分（Faithfulness + Relevance 一次调用）
  ↓
Welch's t-test 单侧检验
  ↓
决策引擎: rollback (B 显著差) / promote (B 显著好) / hold (无显著差异)
```

### 核心模块

| 模块 | 功能 |
|---|---|
| `TrafficSplitter` | MD5 哈希分流，保证同 user_id 稳定路由 |
| `online_judge` | 一次 LLM 调用同打 Faithfulness + Relevance（省钱）|
| `welch_t_test` | 标准统计检验，含 Welch-Satterthwaite 自由度 |
| `ABTester.decision()` | 自动决策引擎，可配回滚阈值 |

## 3. 实验设计

### 3.1 两个 Prompt 变体

| 变体 | 策略 |
|---|---|
| **A_simple** | "简洁直接回答，2 句话内" |
| **B_cot** | "理解需求 → 给核心答案 → 补充建议，3-4 句" |

### 3.2 测试集

30 条客服 query（模拟 30 个用户），50/50 分流。覆盖 GPU 实例配置、价格、SLA、退款、技术支持等典型场景。

### 3.3 KB Context（给 RAGAS 评分用）

包含 ecs.gn7e/gn7i 配置、包月套餐、SLA、退款政策。

## 4. 结果

### 4.1 流量分流

30 个 user_id MD5 哈希后 **正好 15/15** ✓——哈希分流的优势：稳定、均匀、可复现。

### 4.2 Variant 对比

| 指标 | A_simple (n=15) | B_cot (n=15) | Delta | Welch p |
|---|---|---|---|---|
| **Faithfulness** | 0.849 | 0.828 | **-0.021** | p=0.57 ✗ |
| **Relevance** | **1.000** | 0.950 | -0.050 | p=0.92 ✗ |
| 长度（字符）| 39 | 112 | **+186%** | **p<0.001 ✓** |
| 延迟（ms）| 1617 | 2265 | **+40%** | **p<0.001 ✓** |
| Output tokens | 27 | 76 | +181% | — |

### 4.3 自动决策

```json
{"decision": "hold", "reason": "无统计显著差异，继续观察"}
```

**正确判断**——核心指标无显著差异，不该贸然上线 B。

## 5. 三个核心发现

### 5.1 CoT 结构化 prompt 不一定提升质量（反直觉）

教材直觉：CoT 让模型"思考更完整" → 答案更准。

实测：对简单客服问答（如"A100 显存多大"），B 的 Faithfulness 和 Relevance 都**略低于** A（虽不显著）。

**原因猜测**: 简单事实问答，A 直接 quote KB（"A100 显存 80GB"）；B 因为要"理解 → 答 → 建议"三步，**多说了不必要的话，反而稀释了关键信息**。

### 5.2 CoT 显著增加成本（+186% 长度，+40% 延迟）

| 场景 | A 月成本 | B 月成本 | 多花 |
|---|---|---|---|
| 1 万次/天 | ¥6.9 | ¥21.5 | **3.1×** |
| 100 万次/天 | ¥690 | ¥2150 | **+¥1460/月** |

→ **质量无显著提升，成本 3 倍**——这是典型的"工程师拍脑袋上线"陷阱。

### 5.3 自动决策引擎价值：把工程纪律编码进系统

很多团队会**贸然按均值微差就上线 B**（"B 长度显著更长，更专业"）。决策引擎拦住这种偏见：
- 看核心指标（Faithfulness/Relevance）而非辅助指标（长度/延迟）
- 看统计显著性（p<0.05）而非均值差
- 看绝对差距（>15% 才回滚）而非相对差距

> **决策引擎的真正价值**：把"工程师拍脑袋"变成"数据驱动的暂缓"。

## 6. 决策引擎完整规则

```python
def decision(rollback_threshold=0.15):
    # Rollback: B 在 Faithfulness/Relevance 任一项比 A 低 >15% 且显著
    for key in ["faithfulness", "relevance"]:
        if delta < -threshold and p < 0.05:
            return "rollback"
    # Promote: B 在 Faithfulness/Relevance 任一项比 A 高且显著
    if any(delta > 0 and p < 0.05 for key):
        return "promote"
    # Hold: 默认
    return "hold"
```

### 生产环境工作流

```
T+0: 上线 B (10% 流量)
T+24h: 检查 → hold (继续观察)
T+72h: 检查 → promote (扩到 50%)
T+168h: 检查 → promote (全量 100%)
[或] T+24h 异常: rollback (B Faithfulness 显著降 >15%)
```

## 7. 跨实验关联：CoT 的边界

| 实验 | 任务类型 | CoT 效果 |
|---|---|---|
| #1 | 数学推理 / 1.5B | **反退化**（CoT 让小模型更差）|
| #6 | 多步推理 / qwen-turbo | **正增益**（25% → 75%）|
| **#19** | **简单客服问答 / qwen-turbo** | **轻微负面**（Relevance -5%）|

→ **CoT 不是银弹**：要按 (任务复杂度) × (模型容量) 共同决定。这个结论需要 **3 个独立实验** 才能说服面试官。

## 8. 工程教训

1. **A/B 测试是 prompt 工程的底线** ——不上 A/B 直接全量改 prompt，等于盲改
2. **核心指标 ≠ 辅助指标**——长度/延迟显著不代表业务价值显著
3. **统计显著性 ≠ 业务显著性**——p<0.05 但 delta=0.01 仍不该改
4. **样本量决定结论强度**——30 条只能初步定向，1000+ 条才能下定论
5. **决策引擎要写死规则**——避免工程师上线决策被人情/KPI 干扰

## 9. 与生产环境对接的扩展点

| 维度 | 本实验（demo） | 生产环境 |
|---|---|---|
| 流量分流 | MD5 user_id | 加 layered experiment, holdout group |
| 在线评测 | LLM-as-judge | 加用户反馈（点赞/差评）+ session 留存 |
| 统计检验 | Welch's t-test | 加 sequential test（提前停止）|
| 决策 | 单一阈值 | 多维度（成本/质量/延迟）加权 |
| 监控 | print | Prometheus + Grafana 告警 |
| 回滚 | 函数返回 | 网关动态路由 + 全量配置中心 |

## 10. 复现

```bash
cd ~/Calude-Learning/ai-gateway
source ../venv/bin/activate
python experiments/ab_testing_demo.py
# 详细结果: experiments/ab_testing_results.json
```

预算：~¥0.25, 4-6 分钟。

## 11. 一句话总结

> **A/B 测试不是奢侈品，是生产环境 prompt 工程的底线**。
> 30 条小实验已经告诉我们：CoT 结构化 prompt 在简单客服场景**质量无显著提升、成本 3 倍**——这种发现没有 A/B 框架根本看不见。
> **决策引擎的价值不是替工程师做决定，而是拦住工程师拍脑袋上线**。
