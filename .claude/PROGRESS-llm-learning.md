# LLM 学习项目进度文档（接力交接）

> **用途**：下一个会话直接读这份文档，能立刻接着干。无需重述历史。
> **最后更新**：2026-05-08
> **总耗时**：~3 周（4/21 - 5/8）

---

## 0. 一句话现状

用户在准备**阿里云海外 GenAI SA P7 岗位**（100-130 万包）面试。已完成 8 周计划的 1-5 周（技术部分）+ 加分项 Multi-Agent + Dify + LangChain/LangGraph，**剩第 6-8 周的商业化/英文/简历**。GitHub 仓库 [`liamlai88/local-llm-gateway`](https://github.com/liamlai88/local-llm-gateway) 有 11 份实证报告。

---

## 1. 用户基础信息

- **机器**：MacBook Air, Apple M5, 24 GB 统一内存
- **目标岗位**：阿里云海外 GenAI Solution Architect (P7)，海外 AI MaaS 方向
- **背景**：英本硕 + 6 年英国经验（要写进简历）
- **Python 水平**：从"一点点基础"到现在能跟着指令做完整工程改造
- **学习风格偏好**：
  - **要先讲原理再写代码**（之前抱怨过"只跑脚本不深刻"）
  - 每个学习点必须有**对照实验+数据+商业洞察**
  - 拒绝纯教程式学习
- **已知踩坑反馈**：
  - "跟脚本跑，手会了脑子没会" → 现在每个新概念先讲 5 分钟原理
  - "tokens 用量大" → 已沉淀到 memory 文件减少重复

---

## 2. 项目仓库

- **本地**：`~/Calude-Learning/ai-gateway/`
- **GitHub**：https://github.com/liamlai88/local-llm-gateway
- **大小**：项目代码 46MB，加 venv 989MB
- **Ollama 模型**：`~/.ollama/models/` 5.4GB（仅保留 1.5B + 7B）

---

## 3. 项目目录结构

```
ai-gateway/
├── README.md                    # 项目说明，含 11 份实验链接
├── RESUME.md                    # 中英简历段落（每周更新）
├── gateway.py                   # FastAPI 网关主程序（18 KB）
├── rag.py                       # RAG 模块（Vector/BM25/Hybrid/Rerank）
├── agent.py                     # ReAct + Plan-Execute 双范式
├── multi_agent.py               # 6 Agent 混合架构（含 LLMFallback）
├── mcp_server.py                # FastMCP Server（4 Tools+1 Resource+1 Prompt）
├── locustfile.py                # Locust 压测脚本
├── Modelfile / Modelfile-7b     # Ollama 模型配置
├── .gitignore
├── .claude/
│   └── PROGRESS-llm-learning.md  # 本文件
├── chroma_data/                 # ChromaDB 持久化（运行时生成）
├── finetune/                    # LoRA 微调
│   ├── generate_data.py         # 生成 100 条工具调用样本
│   ├── train.sh                 # MLX-LM LoRA 训练
│   ├── compare.py               # 微调前后对比
│   ├── data/{train,valid,test}.jsonl
│   └── lora_adapter/
│       ├── adapter_config.json
│       └── adapters.safetensors  # 10MB 训练成果
├── experiments/
│   ├── 01-Prompt工程ROI.md       # 报告 1
│   ├── 02-RAG对比纯LLM.md              # 报告 2
│   ├── 03-Hybrid检索失败实验.md        # 报告 3
│   ├── 04-Rerank突破.md          # 报告 4
│   ├── 05-ReAct智能体能力边界.md       # 报告 5
│   ├── 06-PlanExecute范式突破.md  # 报告 6
│   ├── 07-MCP协议实现.md    # 报告 7
│   ├── 08-LoRA微调突破.md   # 报告 8
│   ├── 09-MultiAgent混合架构.md  # 报告 9（含 3 模型矩阵）
│   ├── 10-Dify对比手写.md            # 报告 10
│   ├── 11-LangChain与LangGraph对比.md  # 报告 11
│   └── *.py                            # 各报告对应实验脚本
└── monitoring/
    ├── docker-compose.yml       # Prometheus + Grafana
    ├── prometheus.yml
    └── grafana-datasource.yml

外部目录:
~/Calude-Learning/notes/
  ├── Token_成本计算器.xlsx       # 5 sheet 完整 Token TCO 工具
  ├── token_cost_calculator_builder.py  # 生成器源码
  ├── week1/qwen-family.md       # Qwen 家族选型笔记
  ├── week2/prompt_compare.py    # 早期实验
  └── week2/prompt_experiment.py
```

---

## 4. 11 份实证报告速查表

| # | 报告 | 核心数字 | 商业洞察金句 |
|---|------|----------|--------------|
| 1 | Prompt Engineering ROI | 1.5B+CoT 反退化 | 模型容量是地板，Prompt 是天花板 |
| 2 | RAG vs 纯 LLM | 闭域 0% → 100%, +300ms +¥0.00006 | 拒答 ≠ 安全，幻觉才是杀手 |
| 3 | Hybrid Search 失败 | Vector/BM25/Hybrid 都 50% | Hybrid 不是银弹（同质化错误）|
| 4 | Rerank 突破 | 50% → 100%, +252ms | 三层架构（召回+精排+生成）是标配 |
| 5 | ReAct Agent 边界 | 25% 天花板（Tool Use Laziness）| Few-shot 双刃剑、大模型 Tool Lazy |
| 6 | Plan-Execute 突破 | 25% → 75%, 范式胜模型 | 范式 > 模型规模 3 倍 |
| 7 | MCP Server | 4 Tools+1 Resource+1 Prompt | 工具集 USB 化跨平台 |
| 8 | LoRA 微调突破 | 100 条样本 + 25 分钟, 40%→100% | 100 条样本让 1.5B 完胜 Turbo |
| 9 | Multi-Agent 混合架构 | 4/4 准确率, 1.2s 延迟, 1/20 成本 | **架构创新 > 模型升级** |
| 10 | Dify 实战 | 1 小时拖出审核 API, 4/4 | Dify ≠ 替代代码，是组织效率 |
| 11 | LangChain/LangGraph | StateGraph 4/4 / 533ms / -62% 代码 | 框架做 80% + 创新 20% |

**核心叙事**：
```
模型 → 检索 → Agent → 协议 → 微调 → 架构 → PaaS → 框架
每份报告都有反直觉发现 + 实测数据 + 客户话术
```

---

## 5. 报告 9 (Multi-Agent) 最新数据 ⭐

### 模型规模 × 范式 完整矩阵（最重要的面试数据）

| 范式 | 1.5B | 7B | Turbo |
|------|------|------|------|
| ReAct | 25% | ❌ Ollama 不支持 | 25% |
| Plan-Execute | 75%（含逻辑错的"假通过"50%）| 75% | 75% |
| **Multi-Agent 混合** | **100%** ⚠️ 伪通过陷阱 | **100%** ✅ | **100%** ✅ |

### 反直觉发现（最值钱的洞察）

**1.5B 兜底"伪通过"陷阱**：
```
盲区题："AI 推理服务，10 万次/天，选什么 GPU"

1.5B 答: "推荐 ecs.gn7e-c12g1.3xlarge (A100 80GB)"
       ↑ 答错产品（A100 是训练卡，推理用 A10 即可）
       ↑ 评判脚本检测"推理"关键词 → 假通过
       ↑ 客户照做月成本多 4 倍（¥35000 vs ¥9800）

7B 答: "推荐 ecs.gn7i-c8g1.2xlarge (A10 24GB)" ✅
Turbo 答: 同样 A10 ✅
```

→ **生产兜底必须 7B 起步**

---

## 6. 已完成进度（按 8 周计划）

| 周次 | 任务 | 状态 | 产出 |
|------|------|------|------|
| 1 | 阿里云生态 + LLM 基础 | ✅ 超额 | Gateway + 量化对比 + KV Cache 实验 |
| 2 | Prompt + RAG + Agent 理论 | ✅ 超额 | 实操报告 1-6 |
| 3 | Demo #1: RAG 应用 | ✅ | Dify 内容审核 Workflow + ai-gateway RAG 接口 |
| 4 | Demo #2: Agent + MCP | ✅ | Multi-Agent 混合架构 + MCP Server + 接入 Claude Desktop |
| 5 | Demo #3: LoRA 微调 | ✅ | MLX-LM 训练 + 100% 工具规划准确率 |
| 加分 | LangChain/LangGraph | ✅ | 报告 11 |
| 加分 | Multi-Agent 模型矩阵 | ✅ | 1.5B/7B/Turbo 三模型对照 |

---

## 7. 还没做的（按优先级）

### 🔴 第 6 周：商业化（最影响 SA 面试通过率）

| 任务 | 状态 | 说明 |
|------|------|------|
| Token 成本计算器 (Excel) | ✅ 已生成 | `~/Calude-Learning/notes/Token_成本计算器.xlsx`，5 sheet 完整 |
| 海外 MaaS 客户价值模型 | ❌ 未做 | 需要写一份客户 TCO + ROI 完整文档 |
| **4 份海外场景方案文档**（最重要）| ❌ 未做 | 社交平台/视频/MaaS/AI 数据 各 1-2 页 |
| MaaS 商业模式分析 | ❌ 未做 | Bedrock/Vertex/百炼对比 |

**优先做哪个**：4 份海外场景方案（这是 SA 面试方案轮的核心）

### 🔴 第 7 周：英文沟通

| 任务 | 状态 | 说明 |
|------|------|------|
| 英文技术术语 200 个 | ❌ | AI/Cloud 常用 |
| 海外客户 FAQ 10 个 | ❌ | 数据主权/合规/延迟/成本 |
| 东南亚/中东/欧洲市场差异 | ❌ | 区域客户画像 |
| **英文 5 分钟 Demo 视频** | ❌ | 录制 + 上传 |

### 🔴 第 8 周：求职冲刺

| 任务 | 状态 | 说明 |
|------|------|------|
| 简历重写中英版 | ⚠️ 部分 | RESUME.md 已有素材，需打磨成 ATS 友好版 |
| 20 题面试答案 | ❌ | 含 STAR + 系统设计 + BQ |
| 模拟面试 × 2 场 | ❌ | 找朋友或 AI |
| 内推渠道挖掘 | ❌ | 脉脉/LinkedIn 找阿里 P7 |

### 🟡 中优先级（JD 加分项）

| 任务 | 状态 | 说明 |
|------|------|------|
| 阿里云 PAI 平台微调 | ❌ | LoRA 用的是 MLX-LM 本地，没用 PAI |
| 百炼控制台拖一个应用 | ❌ | 显示"会用客户的产品" |
| 阿里云 ACK GPU 调度 | ❌ | 云架构理论 |
| Wan 视觉模型实战 | ❌ | JD 提到 |

### 🟢 低优先级（可选）

- LangSmith 监控集成
- LangGraph Checkpointer 持久化
- DPO 偏好对齐微调
- 多模态（VLM）实战

---

## 8. 关键技术决策（避免重新讨论）

| 决策点 | 当前选择 | 理由 |
|--------|----------|------|
| 本地部署框架 | Ollama | M5 上 Metal 加速，开箱即用 |
| 主力本地模型 | Qwen2.5-1.5B Q4_K_M | M5 上 110 tok/s，准确率够 |
| 大模型对照 | Qwen2.5-7B Q4 | M5 4.9GB 显存能跑，29 tok/s |
| 云端 LLM | 百炼 Qwen-Turbo | JD 对标，便宜 |
| Embedding | 百炼 text-embedding-v2 (1536 维) | 中文场景最优 |
| Rerank | 百炼 gte-rerank | 跟 Embedding 同生态 |
| 向量库 | ChromaDB（嵌入式）| 无需独立服务，PersistentClient |
| 中文分词（BM25）| jieba | 默认词典 |
| 微调框架 | MLX-LM（苹果原生） | M5 上最快，对比方案是 PAI |
| Agent 框架 | 手写 + LangGraph 双轨 | 手写懂底层，LangGraph 生产用 |
| 监控 | Prometheus + Grafana | 行业标准 |
| 压测 | Locust | Python 生态 |

---

## 9. 已修复的工程踩坑（重要！）

| 踩坑 | 触发场景 | 修复 |
|------|----------|------|
| `uvicorn --reload` 不会 reload pip install | 装新包后 Gateway 没生效 | 必须手动 `pkill -f uvicorn` 重启 |
| MCP `env=None` 不是继承父进程 | mcp_test.py 调 server 没 API key | 必须 `env=os.environ.copy()` |
| ChromaDB 相对路径在 stdio 子进程失效 | MCP Server 看不到 KB 数据 | 用 `__file__` 推绝对路径（已改 rag.py）|
| Ollama 默认监听 127.0.0.1 | Docker 容器（Dify）连不上 | `OLLAMA_HOST=0.0.0.0 ollama serve` |
| Dify 容器访问宿主机 Ollama | localhost 连不通 | 用 `host.docker.internal` |
| Ollama 1.5B 不支持 OpenAI tool_calls | LangGraph create_agent 直接报错 | 本地小模型只能手写 ReAct 或 LoRA |
| LangChain create_react_agent 迁移 | V1.0 后报错 | 改用 `from langchain.agents import create_agent` |
| **dashscope SDK 编码 bug**（最近）| `latin-1 codec can't encode...` | 改用 OpenAI 兼容模式 + requests 直调（已改 rag.py 的 embed/rerank）|
| API Key 复制粘贴混入中文字符 | export 后 latin-1 报错 | 用 `read -s` 读取，验证 `key.isascii()` |
| Gateway 进程内 chroma 客户端缓存陈旧 | 外部脚本上传后 Gateway 看不到 | 重启 Gateway 让 chroma 重新加载 |

---

## 10. 环境快速恢复（每次会话开头）

### 检查依赖
```bash
which python3                    # 应该是 /opt/homebrew/bin/python3 (3.14)
ls ~/Calude-Learning/venv/bin/python3  # 应该存在
which ollama                     # /usr/local/bin/ollama 或类似
docker --version                 # 28+ 即可
```

### 启动服务（按需）
```bash
# 1. Ollama（必启）
OLLAMA_HOST=0.0.0.0 ollama serve > /tmp/ollama.log 2>&1 &

# 2. Gateway（做实验时）
cd ~/Calude-Learning/ai-gateway
source ~/Calude-Learning/venv/bin/activate
DASHSCOPE_API_KEY=sk-9d2b470... uvicorn gateway:app --port 8000 --reload &

# 3. Dify（仅做 Demo 演示时）
cd ~/Calude-Learning/dify/docker
docker compose start

# 4. Prometheus/Grafana（监控）
cd ~/Calude-Learning/ai-gateway/monitoring
docker compose up -d
```

### 验证
```bash
curl -s http://localhost:11434/api/tags          # Ollama
curl -s http://localhost:8000/v1/health          # Gateway
curl -s http://localhost:8000/v1/rag/stats       # KB 是否有数据
```

---

## 11. 上传 KB 标准命令（复制即用）

```bash
curl -X POST http://localhost:8000/v1/rag/documents \
  -H "Authorization: Bearer sk-demo-002" \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"product_b","content":"产品名称: 阿里云 ecs.gn7e-c12g1.3xlarge\nGPU: NVIDIA A100, 显存 80GB\n定价: 按量付费 ¥68/小时, 包月 ¥35000\n适用场景: 大模型训练、推理"}'

curl -X POST http://localhost:8000/v1/rag/documents \
  -H "Authorization: Bearer sk-demo-002" \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"product_a","content":"产品名称: 阿里云 ecs.gn7i-c8g1.2xlarge\nGPU: NVIDIA A10, 显存 24GB\n定价: 按量付费 ¥18/小时, 包月 ¥9800\n适用场景: AI 推理、轻量微调"}'
```

---

## 12. API Key 信息

- **DASHSCOPE_API_KEY 前缀**：`sk-9d2b470...`（35 位，纯 ASCII）
- **存放位置**：用户 shell 环境变量，不在代码里
- **使用方式**：每次启动 Gateway 必须带 `DASHSCOPE_API_KEY=...` 前缀

---

## 13. 性能基线（M5 24GB 实测）

| 指标 | 1.5B Q4 | 7B Q4 |
|------|---------|-------|
| 显存占用 | 1.4 GB | 4.9 GB |
| 单用户速度 | 110 tok/s | 29 tok/s |
| 单用户 P99 | 940ms | ~2400ms |
| 5 用户 RPS | 3.2 | ~1.5（估）|
| 20 用户 P99 | 2.5s | 5s+（估）|
| 数学题准确率 | 50% | 95%+ |

---

## 14. 接下来最该做的事（按优先级）

### 🥇 强烈推荐：4 份海外场景方案文档

最影响 SA 面试方案轮通过率，**比 Token 计算器、英文 Demo 更优先**。

四个场景对应 JD：
1. **海外社交平台内容审核**（已有 Demo #1 雏形，扩成完整方案文档）
2. **AI 视频内容分析**（Multi-Agent 架构改造 + 视频 OCR）
3. **AI 数据标注服务平台**（参考 Scale AI 模式）
4. **海外 MaaS 模型平台**（参考 Replicate/Together）

每份文档结构：
- 客户痛点 + 业务规模
- 技术架构图（用项目里现有的 ai-gateway/Multi-Agent 改造）
- 核心模块（RAG/Agent/MCP/LoRA 怎么用）
- TCO 成本估算（用 Token 计算器算）
- 实施路线图（PoC → 生产）
- 风险/合规

### 🥈 备选：英文 Demo 视频

如果用户更想推进语言能力，可以选这个。挑选 1-2 个最强 Demo（Multi-Agent 混合架构 + LoRA），录 5 分钟英文讲解。

### 🥉 备选：补 PAI 微调

把 LoRA 实验在阿里云 PAI 上重跑一遍，对标 JD"PAI/HuggingFace 微调经验"。

---

## 15. 用户偏好的"最佳学习模式"

经过试错，用户认可的模式：

```
1. 我讲核心原理（5-10 分钟概念）
2. 我给"半成品"代码（关键函数留空 / 让用户思考）
3. 用户尝试自己写 / 提问
4. 卡住时给提示，不直接给答案
5. 用户跑通
6. 用户自己总结报告（我只 review）
```

**不要的模式**（用户明确反对）：
- 直接给完整脚本让用户跑（"手会了脑子没会"）
- 不讲原理直接动手
- 一次给太多技术名词不讲清楚

---

## 16. 用户的"自测 5 题"（可用来检验深度）

下次会话开头可以问用户能不能脱稿答（之前问过，他诚实承认大部分答不上）：

1. RRF 融合公式怎么写？为什么 k=60 是经验值？
2. Multi-Agent Critic 用了哪 3 个检查规则？
3. LoRA rank=8 训练 1.5B 模型，A、B 矩阵多少参数？
4. Tool Use Laziness 在大模型上更严重还是小模型更严重？为什么？
5. Hybrid Search 在失败实验里为什么失败？（3 个真理）

这些问题答得出，才是"真懂"。答不出说明还在"跟脚本"层。

---

## 17. 接力提示：下一个会话开头怎么开场

推荐这样开：

```
读 ~/Calude-Learning/ai-gateway/.claude/PROGRESS-llm-learning.md

然后问用户:
1. 上次清理完环境了，今天想做什么？
2. 推荐方向（按优先级）：
   A. 4 份海外场景方案（SA 方案轮核心）
   B. 英文 Demo 视频（5 分钟，对应第 7 周）
   C. PAI 平台微调（JD 加分项）
   D. 补技术深度（自测 5 题查漏补缺）

3. 等用户选完，按"先讲原理 + 引导思考"模式推进
4. 不要重述已完成的进度
5. 完成后必须 push GitHub + 更新本文件
```

---

## 18. 重要文件检查表（快速验证项目完整）

```bash
# 应该全部存在
test -f ~/Calude-Learning/ai-gateway/gateway.py && echo "✓ gateway"
test -f ~/Calude-Learning/ai-gateway/rag.py && echo "✓ rag"
test -f ~/Calude-Learning/ai-gateway/agent.py && echo "✓ agent"
test -f ~/Calude-Learning/ai-gateway/multi_agent.py && echo "✓ multi_agent"
test -f ~/Calude-Learning/ai-gateway/mcp_server.py && echo "✓ mcp_server"
test -f ~/Calude-Learning/ai-gateway/finetune/lora_adapter/adapters.safetensors && echo "✓ LoRA adapter"
test -f ~/Calude-Learning/notes/Token_成本计算器.xlsx && echo "✓ Token 计算器"
ls ~/Calude-Learning/ai-gateway/experiments/*.md | wc -l   # 应该是 11
```

---

## 19. GitHub 提交规范（用户已习惯）

每次完成实验后 commit + push：

```bash
cd ~/Calude-Learning/ai-gateway
git add <files>
git commit -m "feat: <主题>

<3-5 行描述>

实证结论:
\"<面试金句>\"
"
git push
```

---

**END** — 下次会话从读这份文件开始。
