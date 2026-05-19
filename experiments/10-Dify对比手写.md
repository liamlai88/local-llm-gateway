# Dify 实战报告 — 拖拽式 vs 代码手写的真实对比

> **核心问题**：JD 明确要求会用 Dify，但我已经手写了完整 RAG/Agent/Multi-Agent。Dify 还有学的必要吗？  
> **本实验**：用 Dify 在 1 小时内搭出"海外社交内容审核助手"（对应 8 周计划 Demo #1），与手写版本做架构、性能、可维护性的对比。  
> **结论**：**Dify 不是替代代码，是补充。客户演示用 Dify（拖拽 + 可视化），生产用代码（性能可控 + 灵活）。两者都要会**。  
> **实验时间**：2026.05

---

## 实验设计

### 任务：海外社交内容审核 API

输入：用户发的内容文本  
输出：JSON 判断 `{"verdict": "pass|reject|review", "category": "...", "reason": "..."}`

要求：
- 基于内部规则库做 RAG 检索增强
- 1 秒内响应
- 可对外暴露 API

### 两套实现对比

| 维度 | 手写版 (ai-gateway) | Dify 版 |
|------|---------------------|---------|
| 代码量 | ~500 行 Python | **0 行** ⭐ |
| 搭建时间 | 约 8 小时 | **1 小时** ⭐ |
| 学习成本 | LangChain + FastAPI + Chroma | **零代码** |
| 可视化 | 命令行 + Grafana | **画布编辑器** |
| 调 Prompt | 改代码 + 重启 | **网页保存即生效** |

---

## Dify 部署架构

```
            ┌──────────────────────────────────────┐
            │   Dify Workflow (拖拽编辑)            │
            │                                      │
            │  Start → 知识检索 → LLM → IF/ELSE     │
            │                              ↓       │
            │                   ┌──────────┴───┐   │
            │                   ↓              ↓   │
            │              拒绝输出        通过输出  │
            └──────┬───────────────────────────────┘
                   │
            ┌──────▼──────┐  ┌──────────────┐  ┌──────────┐
            │  Postgres   │  │   Weaviate   │  │  Redis   │
            │  (元数据)    │  │   (向量库)    │  │  (缓存)   │
            └─────────────┘  └──────────────┘  └──────────┘

   外部接入:
            ↑
       Ollama 本地 (qwen2.5-1.5b)  ← LLM
       百炼 text-embedding-v2     ← Embedding
```

**11 个 Docker 容器**协作，全部本地部署（数据合规）。

---

## 完整工作流

```
  ┌──────────────────────┐
  │  1. Start (用户输入)  │
  │     query: 待审核内容 │
  └──────────┬───────────┘
             │
  ┌──────────▼───────────┐
  │  2. 知识检索          │
  │     KB: moderation_kb │
  │     Top-K: 3          │
  └──────────┬───────────┘
             │
  ┌──────────▼───────────┐
  │  3. LLM (qwen2.5-1.5b)│
  │     System: 内容审核员 │
  │     输出: JSON verdict │
  └──────────┬───────────┘
             │
  ┌──────────▼───────────┐
  │  4. IF/ELSE          │
  │  contains "reject"?   │
  └────┬───────────┬─────┘
       │TRUE       │FALSE
       ↓           ↓
  ┌────────┐  ┌────────┐
  │拒绝输出 │  │通过输出 │
  └────────┘  └────────┘
```

---

## 实验结果

### 4 道测试题（对照之前实验同样数据）

| 测试 | 内容 | 预期 | Dify 实际 | 准确率 |
|------|------|------|-----------|--------|
| T1 | "我要杀了那个人" | reject | reject ✅ | 100% |
| T2 | "今天天气很好" | pass | pass ✅ | 100% |
| T3 | "某些种族就是天生比其他种族优越" | review | reject ⚠️ | 严了一档 |
| T4 | "如何开发一个 RAG 应用" | pass | pass ✅ | 100% |

### API 性能（生产级）

| 指标 | 值 |
|------|-----|
| 平均延迟 | **1.0 秒** |
| Token 消耗 | 平均 160 tokens/次 |
| 节点执行 | 5 步（Start → KB → LLM → IF → End）|
| LLM 调用次数 | 1 |
| API 协议 | OpenAPI 标准 + Bearer Token 认证 |

### API 调用示例

```bash
curl -X POST 'http://localhost/v1/workflows/run' \
  -H "Authorization: Bearer app-xxx" \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": {"query": "我要杀了那个人"},
    "response_mode": "blocking",
    "user": "test-001"
  }'
```

返回：
```json
{
  "data": {
    "status": "succeeded",
    "outputs": {
      "query": "我要杀了那个人",
      "reasoning_content": "{\"verdict\":\"reject\",\"category\":\"暴力\",\"reason\":\"威胁他人生命安全\"}"
    },
    "elapsed_time": 1.10,
    "total_tokens": 167
  }
}
```

---

## Dify vs 手写 ai-gateway：实测对比

### 同任务的延迟对比

| 实现 | 延迟 | 备注 |
|------|------|------|
| **Dify Workflow** | **1.0s** | Ollama 1.5B + 百炼 Embedding |
| ai-gateway 多 Agent 混合 | 1.2s 平均 | 同样模型 |
| ai-gateway Plan-Execute Turbo | 5-10s | Qwen-Turbo 云端 |

**结论**：Dify 性能跟手写差不多（因为底层模型一样），**不是性能瓶颈**。

### 灵活度对比

| 场景 | Dify | 手写代码 |
|------|------|----------|
| 改 Prompt | 网页保存即生效 ⭐ | 改代码 + 重启 |
| 新增工具 | 拖一个节点 | 改 agent.py 加函数 |
| 路由判断 | IF/ELSE 拖拽 | 写 if-else |
| **复杂 Critic 逻辑** | ❌ 难做 | ✅ 任意 Python |
| **错误重试 / 兜底** | ❌ 受框架限制 | ✅ 自由 |
| **可观测性** | Dify 自带 Trace | 自实现 Prom/Grafana |

### 商业模式对比

| 维度 | Dify | 手写代码 |
|------|------|----------|
| 客户演示 | ⭐⭐⭐⭐⭐ 可视化拖拽 | ⭐⭐ 命令行截图 |
| 客户改 Prompt 自服务 | ⭐⭐⭐⭐⭐ 业务直接改 | ❌ 必须找开发 |
| 多租户 SaaS | ⭐⭐⭐ Dify Cloud 自带 | ❌ 自实现 |
| **生产可控性** | ⭐⭐ 黑盒 | ⭐⭐⭐⭐⭐ 完全控制 |
| **复杂 Agent 逻辑** | ⭐⭐ 受限 | ⭐⭐⭐⭐⭐ 任意 |

---

## 三个核心洞察

### 洞察 1: Dify 的真实定位是"PoC + 业务自服务"

```
PoC 阶段（客户验证概念）→ 用 Dify
  → 1 小时拖出来，业务部门能自己改 Prompt
  → 客户拍板"这个方向 OK，给我做生产版"

生产阶段（高 SLA + 复杂逻辑）→ 用代码重写
  → 拿 Dify 验证好的逻辑作为蓝图
  → 用 LangChain / 自研框架重做
  → 加 Critic、Fallback、监控、限流
```

### 洞察 2: Dify 在国内 SA 场景被严重低估

```
表面看: 它就是个开源 LLM Workflow 工具
真实价值: 客户业务部门 (非工程师) 能自己用!

阿里云客户场景 90% 是这样:
  - 业务方: 我有个 idea, 让 IT 实现
  - IT 排期: 3 周
  - 业务方等不及, idea 黄了

用 Dify 后:
  - 业务方: 我自己拖一个出来试试
  - 1 小时验证, 立刻知道是否有价值
  - 有效就交给 IT 做生产, 无效就放弃
  
ROI 不是性能, 是 "尝试成本" 降低 10 倍
```

### 洞察 3: 不会 Dify 的 SA 是"半个 SA"

```
JD 明确要求: 熟练使用 Dify
真实考察: 你能不能给客户业务方做演示?

技术轮: 问你 LangChain Agent (你手写过, 稳过)
方案轮: 问你 "我们业务想做内容审核, 怎么验证?"
       → 答案应该是: "我用 Dify 在 1 小时内拖给您看"
       → 不是: "我写代码 3 天给您看"

不会 Dify 的候选人在方案轮直接 -10 分
```

---

## 给客户的真实话术

> **客户问**：我已经投了几千万做 LangChain Agent 平台，为什么还要 Dify？
>
> **答**：不冲突，是分层。
>
> | 层级 | 工具 | 谁用 |
> |------|------|------|
> | 业务验证层 | **Dify** | 业务部门、产品经理（拖拽 + 改 Prompt）|
> | 生产实现层 | LangChain / 自研 | 开发团队（高 SLA + 监控）|
> | 基础设施层 | Ollama / 百炼 / vLLM | 平台团队（GPU 调度）|
>
> 没有 Dify，业务部门只能"提需求等开发"，**创新速度受 IT 排期限制**。  
> 有了 Dify，业务部门**自己验证 + 自己迭代**，IT 只做生产化。  
>
> 这就是为什么 Dify 国内估值 10 亿美金 —— 它解决的不是"代码效率"，是"**组织效率**"。

---

## 完整 10 份实验报告认知曲线

```
1. Prompt Engineering ROI    → 模型容量是地板
2. RAG vs 纯 LLM            → 闭域 0% → 100%
3. Hybrid Search 失败        → 检索没有银弹
4. Rerank 突破              → 三层架构是标配
5. ReAct Agent 边界          → Agent 不是万能
6. Plan-Execute 突破         → 范式 > 模型规模
7. MCP Server 实现           → 工具集 USB 化
8. LoRA 微调突破            → 100 条样本 = 小模型完胜大模型
9. Multi-Agent 混合架构      → 准确率上限 × 延迟下限
10. Dify 实战 ⭐            → 拖拽 ≠ 替代代码，而是组织效率工具
```

完整覆盖 **Prompt → RAG → Agent → 协议 → 微调 → 架构 → PaaS** 的 GenAI 工程全栈。

---

## 实验脚本

完整 Dify 部署文档参考 [Dify 官方文档](https://docs.dify.ai/)。

测试 API：

```bash
# 拒绝类
curl -X POST 'http://localhost/v1/workflows/run' \
  -H "Authorization: Bearer $DIFY_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"inputs":{"query":"我要杀了那个人"},"response_mode":"blocking","user":"test"}'

# 通过类  
curl -X POST 'http://localhost/v1/workflows/run' \
  -H "Authorization: Bearer $DIFY_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"inputs":{"query":"今天天气很好"},"response_mode":"blocking","user":"test"}'
```

---

## 后续可做的进阶（按 ROI）

- [ ] **Dify 接入 ai-gateway**：让 ai-gateway 作为 Dify 的下游 LLM Provider，组合两者优势
- [ ] **多个 Workflow 编排**：建一个"路由 Workflow"，根据问题类型分发到不同子 Workflow
- [ ] **Dify Tool 接入**：把 ai-gateway 的工具（calculator/kb_search）注册成 Dify Tool
- [ ] **Workflow 版本管理**：用 Dify 的版本对比功能 A/B 测试不同 Prompt
- [ ] **接入 Slack/钉钉**：把 Dify Workflow 包装成机器人

---

## 核心金句（面试用）

1. **"Dify 不是替代代码，是补充。客户演示用 Dify，生产部署用代码"**
2. **"Dify 真实价值不是性能，是让业务方自己验证创新——尝试成本降低 10 倍"**
3. **"不会 Dify 的 SA 是半个 SA，因为方案轮的'快速验证'问题答不上"**
4. **"我 1 小时搭的 Dify Workflow API 延迟 1 秒，与手写多 Agent 系统持平"**
