# MCP Server 实现报告 — 把 Agent 工具集标准化

> **核心问题**：每个 LLM 厂商都有自己的 Tool Calling 协议（OpenAI Function Calling / Claude Tool Use / Qwen Tool Use），同一份工具要写 N 套代码。Anthropic 推出的 MCP（Model Context Protocol）能否解决这个问题？  
> **本实验**：把现有 Agent 的 4 类能力包装成标准 MCP Server，验证可被任意 MCP Client（Claude Desktop / Cursor / 自建 Agent）调用。  
> **结论**：**4 工具 + 1 资源 + 1 Prompt 全部通过 stdio 协议成功暴露，写一次到处用**。  
> **实验时间**：2026.04

---

## 为什么需要 MCP

### 问题：N×M 复杂度爆炸

```
没有 MCP 的世界:

  OpenAI         Claude          Qwen
   │               │              │
   ▼               ▼              ▼
 Function       Tool Use      Tool Calling
   │               │              │
   ▼               ▼              ▼
 客户写工具    客户写工具    客户写工具
 (适配 N 个 LLM × M 个工具 = N×M 套代码)
```

### MCP 的解：USB 化

```
有 MCP 的世界:

  Claude Desktop / Cursor / 自建 Agent
              │
              ▼  统一 MCP 协议
       MCP Server (写一次)
              │
              ▼
   Tools / Resources / Prompts
```

**SA 价值**：客户的内部工具/数据源**写一个 MCP Server**就能被所有支持 MCP 的 AI 应用复用，避免厂商锁定。

---

## MCP 三大核心抽象

| 抽象 | 含义 | 例子 |
|------|------|------|
| **Tools** | LLM 可调用的函数 | calculator、API call、数据库查询 |
| **Resources** | 只读数据源 | 文件、知识库、配置 |
| **Prompts** | 可复用的 Prompt 模板 | "代码 review"、"成本计算" |

---

## 实现：把 AI Gateway 工具集 MCP 化

### 项目结构
```
ai-gateway/
├── agent.py          # 原 Agent 实现（ReAct + Plan-Execute）
├── rag.py            # RAG 模块
├── mcp_server.py     # ⭐ 新增 - MCP Server 入口
└── experiments/
    └── mcp_test.py   # MCP Client 测试脚本
```

### 暴露的能力

| 类型 | 名字 | 来源 | 价值 |
|------|------|------|------|
| Tool | `calculator` | 复用 Agent 工具 | 数学计算 |
| Tool | `get_weather` | 复用 Agent 工具 | 城市天气 |
| Tool | `extract_number` | 复用 Agent 工具 | 文本→数字提取 |
| Tool | `kb_search` | **复用 RAG 模块** ⭐ | Hybrid 检索 + Rerank |
| Resource | `kb://documents` | 调 RAG stats | 知识库元数据 |
| Prompt | `cost_calc_prompt` | 自定义 | GPU 成本计算模板 |

### 关键代码（FastMCP，~80 行）

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ai-gateway-tools")

@mcp.tool()
def kb_search(query: str) -> str:
    """企业知识库 Hybrid 检索 + Rerank"""
    import rag
    chunks = rag.search(query, top_k=2, mode="hybrid", use_rerank=True)
    return "\n---\n".join([c["content"][:300] for c in chunks])

@mcp.resource("kb://documents")
def list_documents() -> str:
    """知识库统计"""
    return json.dumps(rag.stats())

@mcp.prompt()
def cost_calc_prompt(product: str, hours: str = "24") -> str:
    """GPU 成本计算 Prompt 模板"""
    return f"...用 kb_search 查{product}价格→extract_number→calculator * {hours}..."

if __name__ == "__main__":
    mcp.run()  # stdio 模式
```

---

## 实验结果

### 完整测试输出

```
✅ MCP Server 连接成功

📦 工具列表 (4 个):
  - calculator: 数学表达式计算
  - get_weather: 查询城市当前天气
  - extract_number: 从文本中提取数字
  - kb_search: Hybrid 检索 + Rerank

📚 资源列表 (1 个):
  - kb://documents

📝 Prompts 列表 (1 个):
  - cost_calc_prompt

🧪 工具测试:
  calculator(15*0.6) → 9.0 ✅
  get_weather(杭州) → 晴, 22°C ✅
  extract_number(¥68/小时, hint=价格) → 68 ✅
  kb_search(A100 价格) → 阿里云 ecs.gn7e-c12g1.3xlarge / NVIDIA A100 ... ✅

📖 资源: kb://documents → 总分块数: 2 ✅
💡 Prompt: cost_calc_prompt(A100, 24) → 完整模板返回 ✅
```

---

## 实施过程踩到的两个真坑

### 坑 1: `env=None` 不是"继承父进程环境"

```python
# 直觉认为这样就能继承父进程 env
server_params = StdioServerParameters(
    command="python3",
    args=["mcp_server.py"],
    env=None,  # ← 实际：清空 env！子进程拿不到 DASHSCOPE_API_KEY
)
```

**真相**：MCP SDK 的 `env=None` 等于"完全空的环境变量"，不是 Python `subprocess` 的"继承"语义。

**解决**：

```python
import os
server_params = StdioServerParameters(
    ...
    env=os.environ.copy(),  # 显式拷贝
)
```

**SA 视角**：API Spec 里"None"的语义在不同 SDK 不一样。生产环境必须**显式传 env**，避免环境隔离 bug。

---

### 坑 2: 相对路径在 stdio 子进程下失效

```python
# rag.py 原始代码
_chroma_client = chromadb.PersistentClient(path="./chroma_data")
                                                  ↑
                                          相对路径，看 cwd
```

主进程 cwd 是 `ai-gateway/`，子进程 cwd 可能不一样 → 读到不同的 chroma 文件 → 看到 0 chunks。

**解决**：

```python
chroma_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "chroma_data")
)
_chroma_client = chromadb.PersistentClient(path=chroma_path)  # 绝对路径
```

**SA 视角**：MCP Server 经常被各种 Client 启动，**所有路径必须用 `__file__` 推导绝对路径**，不能依赖 cwd。

---

## 三个生产价值

### 价值 1: 一次开发，多端复用 ⭐

写完 mcp_server.py 后：
- ✅ Claude Desktop 能直接调（在 `claude_desktop_config.json` 加几行）
- ✅ Cursor 能调（IDE 直接接入 MCP）
- ✅ 自建 Agent 能调（用 mcp Python SDK）
- ✅ 任何支持 MCP 的工具都能调

**对比 Agent 内置工具的代价**：
- 之前 `agent.py` 写的 4 个工具只能本项目用
- 现在 MCP 化后，**全网任何 MCP 客户端都能用**

### 价值 2: 工具集"产品化"

把工具集做成 MCP Server 后：
- 可以**单独发布**到 [MCP 生态](https://github.com/modelcontextprotocol/servers)
- 可以**独立版本管理**（Server 版本和业务代码解耦）
- 可以**多团队共享**（一个团队维护，所有团队接入）

### 价值 3: 安全边界清晰

MCP 协议天然支持：
- **Resources** 是只读的（不会被 LLM 改写数据）
- **Tools** 有明确 schema（参数可校验）
- **stdio 模式**完全本地执行（不暴露网络端口）

对比 OpenAI Function Calling：MCP 的安全模型更适合企业内部数据敏感场景。

---

## 给客户讲 MCP 的话术

> "客户经常问：我已经有 LangChain Agent 了，为什么还要 MCP？
>
> 答：你的 Agent 工具**只能你的 Agent 用**。MCP 化之后：
>
> 1. **客户的内部 IT 团队**写一个 MCP Server 暴露 ERP/CRM 数据
> 2. **市场部**用 Claude Desktop 接入这个 Server，问"上季度华东大区销售前10客户"
> 3. **开发部**用 Cursor 接入同一个 Server，AI 写代码时自动查公司用户画像
> 4. **运营部**接百炼 Agent 应用，做客户数据分析
>
> **同一个 MCP Server，4 个部门 4 种使用方式**。这就是 USB 化 vs 私有协议的差别。"

---

## 完整 7 份实验报告的认知曲线

```
报告1: Prompt Engineering ROI
  → 模型容量是地板

报告2: RAG vs 纯 LLM
  → RAG 让小模型在闭域问答 0% → 100%

报告3: Hybrid Search 失败
  → 检索没有银弹

报告4: Rerank 突破
  → 三层架构是生产标配

报告5: ReAct Agent 边界
  → Agent 不是万能，能力边界更重要

报告6: Plan-Execute 突破
  → 范式 + 工具设计 > 模型规模

报告7: MCP Server 实现 ⭐ (本报告)
  → 工具集 USB 化，跨平台复用
```

---

## 后续优化方向

- [ ] **接入 Claude Desktop**：在 `~/Library/Application Support/Claude/claude_desktop_config.json` 添加这个 Server，实测在 Claude 里调用
- [ ] **HTTP+SSE 模式**：让 MCP Server 部署在云上，所有客户端共享访问
- [ ] **加 Auth 中间件**：MCP 协议支持 OAuth，企业部署必备
- [ ] **更多工具**：把 RAG 三种检索模式（vector / bm25 / hybrid）独立暴露成工具，让客户端按需选择
- [ ] **Resource 模板**：用 URI 参数支持 `kb://documents/{doc_id}` 读取特定文档

---

## 实验脚本

完整 MCP Server：[`../mcp_server.py`](../mcp_server.py)  
测试客户端：[`mcp_test.py`](mcp_test.py)  
一键运行（含 KB 准备）：[`mcp_full_test.sh`](mcp_full_test.sh)

```bash
# 完整测试（一行命令）
DASHSCOPE_API_KEY=sk-xxx bash experiments/mcp_full_test.sh

# 在 Claude Desktop 中接入（生产用法）
# 编辑 ~/Library/Application\ Support/Claude/claude_desktop_config.json:
{
  "mcpServers": {
    "ai-gateway": {
      "command": "python3",
      "args": ["/Users/liam/Calude-Learning/ai-gateway/mcp_server.py"],
      "env": {"DASHSCOPE_API_KEY": "sk-xxx"}
    }
  }
}
```

---

## 核心金句（面试用）

1. **"MCP 是 AI 工具的 USB 标准——一次开发，所有支持 MCP 的客户端都能用"**
2. **"区分 LangChain Tool 和 MCP Server：前者是单 Agent 内复用，后者是跨 Agent 跨厂商复用"**
3. **"`env=None` 不是继承父进程环境，是清空 env——这种 SDK 默认值陷阱是生产 bug 高发区"**
4. **"MCP 三大抽象（Tools/Resources/Prompts）天然契合企业安全模型，比 Function Calling 的安全边界更清晰"**
