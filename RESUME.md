# 简历段落 · AI Gateway 项目

> 对标阿里云 GenAI Solution Architect (海外 AI MaaS 方向) JD 关键词

---

## 中文版（详细，用于中文简历）

**项目名称**：本地大模型 MaaS 网关原型 — 模拟阿里云百炼平台架构  
**项目时间**：2026.04 (个人项目)  
**技术栈**：Ollama / FastAPI / Qwen2.5 / Prometheus / Grafana / Locust / Docker

**项目背景**：  
为深度理解 GenAI 基础设施与 MaaS 商业化逻辑，独立设计并实现企业级 AI 推理网关，对标阿里云百炼平台核心能力，可作为客户 POC 演示与技术方案设计的基础原型。

**核心成果**：

- 在 Apple M5 (24GB) 本地部署 **Qwen2.5-1.5B** 三个量化版本（Q4_K_M / Q8_0 / 32K 长上下文），通过对比实验量化了**量化-精度-显存**三角权衡：Q4 相比 Q8 显存节省 30%、推理速度提升 50%；上下文从 4K 扩展到 32K 仅多消耗 900MB 显存

- 设计并实现 **OpenAI 兼容的 FastAPI 网关**，集成 9 个生产级模块：API Key 多租户认证、滑动窗口限流（free 10/min, enterprise 1000/min）、敏感词内容审核、响应缓存（MD5 哈希）、智能路由（fast/quality/long）、SSE 流式输出、Token 计费引擎、结构化日志、实时监控面板

- 集成 **Prometheus + Grafana 可观测性栈**，自定义 8 个核心指标（请求量、P50/P95/P99 延迟、Token 消耗、成本累计、缓存命中率、内容拦截、在飞请求数），通过 PromQL + Grafana 仪表盘实现 SRE 级别的实时监控

- 使用 **Locust 完成多轮压测**，建立完整性能基线：单实例稳定承载 8.8 RPS（混合流量），P99 延迟 < 2.5s；缓存层使热点请求 P99 从 690ms 降至 89ms（**11 倍优化**）；通过 1→5→20 用户阶梯压测，发现并量化 Q8 模型在高并发下的尾延迟问题（P99 达 7.8s，建议独立部署）

- 基于真实压测数据，建立**客户业务容量评估模型**：50 万 DAU 内容审核场景需 7 个 M5 等价实例，启用 30% 缓存命中后实例需求降至 5 个，节省 28% 算力成本，可直接用于售前 TCO 测算

---

## 英文版（精简，用于英文简历 / LinkedIn）

**Local LLM MaaS Gateway Prototype** | *Personal Project, 2026.04*  
**Stack**: Ollama, FastAPI, Qwen2.5, Prometheus, Grafana, Locust, Docker

- Deployed **Qwen2.5-1.5B** with three quantization variants (Q4_K_M / Q8_0 / 32K context) on Apple M5 24GB; benchmarked the trade-offs across memory, throughput, and quality — Q4 saves 30% VRAM and runs 50% faster than Q8

- Built an **OpenAI-compatible FastAPI gateway** with 9 production-grade modules: multi-tenant auth, sliding-window rate limiting, content moderation, response caching, smart model routing, SSE streaming, token-based billing, structured logging, and a real-time dashboard

- Integrated **Prometheus + Grafana observability** with 8 custom metrics (QPS, P50/P95/P99 latency, token usage, cost, cache hit rate); designed PromQL queries and Grafana dashboards for SRE-level monitoring

- Conducted **Locust load tests** at 1/5/20 concurrent users; established a performance baseline of **8.8 RPS sustained** with P99 < 2.5s; cache layer reduced hot-path P99 latency from 690ms to 89ms (**11× improvement**)

- Translated benchmark data into a **customer capacity model**: a 500K-DAU content moderation use case requires 7 M5-equivalent instances; with 30% cache hits, the requirement drops to 5 instances (28% cost savings) — directly usable for TCO calculations in pre-sales scenarios

---

## 面试 STAR 故事 · 中文

### Q: "讲讲你最近做的一个 AI 项目"

**Situation（背景）**：  
"我想深入理解阿里云百炼这类 MaaS 平台的底层架构和商业化逻辑，所以利用我的 Mac M5 在本地复刻了一个简化版的企业 AI 推理网关。"

**Task（任务）**：  
"目标是端到端跑通**部署、网关、监控、压测**全链路，并且通过真实数据回答企业客户最关心的几个问题：要多少显存、能扛多少并发、单次推理多少钱。"

**Action（行动）**：

1. **部署层**：用 Ollama 在 M5 上部署了 Qwen2.5-1.5B 的 Q4 / Q8 / 32K 三个版本，验证 Metal GPU 加速生效，对比量化对显存和推理速度的影响

2. **网关层**：用 FastAPI 设计了 OpenAI 兼容的 API，加了认证、限流、缓存、内容审核、智能路由 5 个企业必备中间件，还自己写了一个 Token 计费引擎

3. **监控层**：集成 Prometheus + Grafana，导出 8 个核心指标，做了 5 个仪表盘面板：QPS、P99 延迟、缓存命中率、Token 速率、累计成本

4. **压测层**：用 Locust 做了 1/5/20 用户三轮阶梯压测，得到完整性能基线

**Result（结果）**：

- 量化结论：M5 单机 Q4 模型稳定承载 8.8 RPS，P99 < 2.5s
- 业务洞察：缓存层让热点请求快 11 倍，是企业必上的优化点
- 工程发现：Q8 模型在高并发下 P99 飙到 7.8s，必须独立部署 + 单独限流
- **可以基于这些数据，给任何客户场景算出实例数和成本** —— 比如 50 万 DAU 的内容审核，需要 7 个实例，加缓存能降到 5 个

---

## 关键词对照表（JD → 项目能力）

| JD 关键词 | 项目对应模块 |
|-----------|--------------|
| 阿里云百炼平台 | 整体架构对标，模型路由命名一致 |
| Qwen 模型家族 | 部署 Qwen2.5-1.5B 多版本 |
| LangChain / Dify (后续 Demo) | 本项目网关层，Demo #1 会接入 |
| MCP (后续 Demo) | Demo #2 实现 |
| LoRA / QLoRA (后续 Demo) | Demo #3 实现 |
| Vibe Coding | 用 Claude Code 协作完成 |
| AI 基础设施 | Ollama / GPU / 监控栈 |
| Token 成本精算 | 自实现计费引擎 |
| MaaS 商业化 | 多租户 / 限流 / 计费 |
| AI 解决方案设计 | 容量评估模型 |
| 可观测性 | Prometheus + Grafana |
| 海外社交内容审核 (Demo #1) | 内容审核中间件已具备 |
| 英文沟通 | 英文版简历 + 后续 Demo 视频 |

---

## 后续可加的能力（按 8 周计划继续推进）

- [ ] 接入 Dify，把 Demo 包装成可视化 RAG 应用 (第3周)
- [ ] 加 MCP Server，支持工具调用 (第4周)
- [ ] LoRA 微调实验，对比微调前后效果 (第5周)
- [ ] 接入 PAI / 百炼 API 做 Hybrid 部署演示 (第6周)
- [ ] 录制英文 Demo 视频 (第7周)
