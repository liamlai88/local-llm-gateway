# AI Gateway · 本地大模型 MaaS 网关原型

> 一个面向企业场景的 AI 推理网关 Demo，模拟阿里云百炼平台核心架构。基于 FastAPI + Ollama 在 Apple M5 (24GB) 上部署 Qwen2.5-1.5B，集成多租户、智能路由、计费、监控等生产级能力。

---

## 🏗 系统架构

```
                ┌──────────────────────────────────┐
                │         Client (curl/SDK)        │
                └────────────────┬─────────────────┘
                                 │ OpenAI 兼容 API
                ┌────────────────▼─────────────────┐
                │     FastAPI Gateway (:8000)      │
                │  ┌──────────────────────────┐    │
                │  │ Auth (API Key)           │    │
                │  │ Rate Limit (滑动窗口)     │    │
                │  │ Content Moderation       │    │
                │  │ Response Cache           │    │
                │  │ Cost Calculator          │    │
                │  │ Prometheus Metrics       │    │
                │  └──────────────────────────┘    │
                │           智能路由                 │
                └─────┬──────────┬──────────┬──────┘
                      │          │          │
                ┌─────▼───┐ ┌────▼────┐ ┌───▼────┐
                │ Q4-fast │ │ Q8-qual │ │ 32K-long│
                │  1.5B   │ │  1.5B   │ │  1.5B   │
                └─────────┘ └─────────┘ └─────────┘
                      │
                ┌─────▼──────────────────┐
                │  Ollama Runtime        │
                │  (Metal GPU 加速)       │
                └────────────────────────┘
                          │
                ┌─────────▼──────────────┐
                │  Prometheus + Grafana  │
                │  (实时监控大盘)          │
                └────────────────────────┘
```

---

## ✨ 核心功能

| 模块 | 实现 |
|------|------|
| **OpenAI 兼容 API** | `/v1/chat/completions`, `/v1/models` |
| **多模型路由** | `fast`(Q4) / `quality`(Q8) / `long`(32K上下文) |
| **认证授权** | API Key + 多租户 (free/enterprise) |
| **限流** | 滑动窗口算法，按 tier 分级 |
| **内容审核** | 敏感词中间件（可对接阿里云内容安全） |
| **响应缓存** | MD5 哈希内存缓存，命中率 28.9% |
| **流式输出** | SSE 协议，OpenAI 兼容 |
| **Token 计费** | 按输入/输出分别计费，对标百炼定价 |
| **可观测性** | Prometheus 8 个指标 + Grafana 5 个仪表盘 |
| **RAG 检索增强** | ChromaDB + 百炼 Embedding，让小模型答对闭域问题 |
| **ReAct Agent** | 工具调用循环 + Tool Calling 抽象，集成 calculator/weather/RAG 三类工具 |
| **MCP Server** | 按 Anthropic MCP 标准协议暴露 4 工具+1 资源+1 Prompt，可被 Claude Desktop/Cursor/任意 MCP Client 调用 |

---

## 📊 性能基线 (Apple M5 / 24GB)

部署模型: **Qwen2.5-1.5B-Instruct (Q4_K_M)**, 100% Metal GPU

| 并发 | 总 RPS | P50 | P95 | P99 | 失败率 |
|------|--------|-----|-----|-----|--------|
| 1 用户 | 0.6 | 280ms | 480ms | 940ms | 0% |
| 5 用户 | 3.2 | 330ms | 460ms | 690ms | 0% |
| 20 用户 | **8.8** | 1600ms | 2400ms | 2500ms | 0% |

### 关键发现

- **缓存层带来 70 倍提速**：热点请求 P50 从 280ms 降至 4ms
- **量化的真实代价**：Q4 → Q8 显存增加 30%，速度降 50%
- **KV Cache 显存暴涨**：上下文 4K → 32K，显存 +900MB
- **Q8 高并发下不可用**：20 用户时 P99 达 7.8s，必须独立部署

### 业务容量推算

```
单 M5 节点稳定承载: ~8.8 RPS = 760K 请求/天
50 万 DAU 内容审核场景需要: ~7 个实例 (含 30% 冗余)
启用缓存后 (命中率 30%): 实例需求降至 5 个，节省 28%
```

---

## 🚀 快速开始

### 1. 部署模型

```bash
# 从 ModelScope 下载 (国内 CDN 快)
pip install modelscope
modelscope download \
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --local_dir ./models

# 导入 Ollama
ollama create qwen2.5-1.5b -f Modelfile
```

### 2. 启动网关

```bash
pip install fastapi uvicorn httpx prometheus_client
uvicorn gateway:app --port 8000 --reload
```

### 3. 启动监控栈

```bash
cd monitoring
docker compose up -d
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000
```

### 4. 测试调用

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-demo-001" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fast",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

### 5. 压测

```bash
locust -f locustfile.py --host http://localhost:8000
# 浏览器: http://localhost:8089
```

---

## 📈 监控指标 (PromQL)

```promql
# 实时 QPS（按模型）
sum by (model) (rate(ai_gateway_requests_total[1m]))

# P99 延迟
histogram_quantile(0.99, sum by (le, model) (rate(ai_gateway_latency_seconds_bucket[1m])))

# 缓存命中率
sum(rate(ai_gateway_cache_total{result="hit"}[1m])) 
/ sum(rate(ai_gateway_cache_total[1m])) * 100

# 累计成本（CNY）
sum by (model) (ai_gateway_cost_cny_total)

# Token 消耗速率
sum by (direction, model) (rate(ai_gateway_tokens_total[1m]))
```

---

## 🛠 技术栈

- **推理引擎**: Ollama (基于 llama.cpp), Metal GPU 加速
- **模型**: Qwen2.5-1.5B (Q4_K_M / Q8_0 / 32K context 三个版本)
- **网关**: FastAPI + httpx (异步)
- **监控**: prometheus_client → Prometheus → Grafana
- **压测**: Locust
- **部署**: Docker Compose

---

## 📁 项目结构

```
ai-gateway/
├── gateway.py              # 网关主程序
├── locustfile.py           # 压测脚本
├── Modelfile               # Ollama 模型配置 (Q4)
├── Modelfile-q8            # Q8 版本
├── Modelfile-32k           # 32K 上下文版本
├── monitoring/
│   ├── docker-compose.yml  # Prom + Grafana
│   ├── prometheus.yml      # 抓取配置
│   └── grafana-datasource.yml
└── models/
    └── *.gguf              # 模型权重文件
```

---

## 🎓 学习收获

完成本项目后，对以下问题有了量化的回答：

1. **客户问"我要部署 Qwen，需要多大显存？"** → 用 KV Cache 公式精算
2. **客户问"为什么 ChatGPT 那么贵？"** → 用 Token 计费 + 量化对比解释
3. **客户问"高并发下怎么保证体验？"** → 用 P99 数据 + 缓存策略说明
4. **客户问"自建 vs 用百炼怎么选？"** → 用真实压测数据算 TCO

---

## 🧪 实验报告

基于本网关做的小型实证研究（持续更新）：

- [Prompt Engineering ROI 实证报告](experiments/prompt-engineering-roi.md) — 用同一道数学题对比本地 1.5B / 百炼 Turbo / 百炼 Max 的准确率，揭示"模型容量是地板，Prompt 是天花板"的核心规律
- [RAG vs 纯 LLM 实证报告](experiments/rag-vs-pure-llm.md) — 验证 RAG 让 1.5B 小模型在闭域知识问答上准确率从 0% 提升到 100%，揭示"拒答 ≠ 安全，幻觉才是杀手"
- [Hybrid Search 实证报告](experiments/hybrid-search-reality.md) — 一次"失败"实验：Vector / BM25 / Hybrid 三种方法准确率均仅 50%，揭示 Tokenization、Chunking、Rerank 三大被低估的 RAG 工程难点
- [Rerank 突破实验](experiments/rerank-breakthrough.md) — 加上百炼 gte-rerank 精排层，准确率从 50% 跃升至 **100%**，验证生产级 RAG 三层架构（召回+精排+生成）
- [ReAct Agent 能力边界实证](experiments/agent-react-boundaries.md) — 12 次对照实验找到 Agent 真实边界：单工具 90%+、多工具组合即使 Qwen-Turbo 也只有 25%。揭示 Few-shot 双刃剑 + Tool Use Laziness + ReAct 天花板三个反直觉真相
- [Plan-Execute 突破实验](experiments/agent-plan-execute-breakthrough.md) — 切换到 Plan-and-Execute 范式，准确率从 25% 跃升至 **75%**，证明"范式选择比模型选择重要 3 倍"。同时揭示新挑战：延迟暴涨 6×、跨 Step 数据传递难题
- [MCP Server 实现报告](experiments/mcp-server-implementation.md) — 把 Agent 工具集按 Anthropic MCP 标准协议包装成跨平台 Server（4 工具+1 资源+1 Prompt），实现"一次开发，Claude Desktop/Cursor/自建 Agent 都能用"。踩过两个真坑：env=None 陷阱、相对路径失效

---

## 📚 参考资料

- [Ollama 官方文档](https://ollama.ai/)
- [Qwen2.5 模型文档](https://qwenlm.github.io/)
- [阿里云百炼平台](https://bailian.console.aliyun.com)
- [Prometheus + FastAPI 集成](https://github.com/prometheus/client_python)
