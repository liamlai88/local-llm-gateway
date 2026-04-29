"""
AI Gateway MCP Server
把 Agent 的 4 个工具暴露成 MCP 标准协议，让任何 MCP Client（Claude Desktop / Cursor / 你的 Agent）都能用。

工具列表:
  - calculator:      数学计算
  - get_weather:     天气查询 (mock)
  - kb_search:       知识库检索 (调本项目 RAG 模块)
  - extract_number:  从文本提取数字

Resources:
  - kb://documents:  知识库当前所有文档（只读）

运行方式:
  # stdio 模式（被 Claude Desktop 等启动）
  python mcp_server.py

  # HTTP 模式（独立服务）
  mcp dev mcp_server.py
"""
import re
from typing import Any
from mcp.server.fastmcp import FastMCP

# 创建 MCP Server
mcp = FastMCP("ai-gateway-tools")


# ========== Tools ==========

@mcp.tool()
def calculator(expression: str) -> str:
    """
    数学表达式计算，支持 +-*/()% 和常见数字

    Args:
        expression: 数学表达式如 "15 * 0.6"
    """
    if not re.match(r"^[\d\s\+\-\*\/\.\(\)\%]+$", expression):
        return f"Error: 表达式包含不允许字符: {expression}"
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"


@mcp.tool()
def get_weather(city: str) -> str:
    """
    查询城市当前天气（mock 数据用于演示）

    Args:
        city: 城市中文名
    """
    mock_db = {
        "杭州": "晴, 22°C, 湿度 60%",
        "北京": "多云, 15°C, 湿度 45%",
        "上海": "小雨, 19°C, 湿度 75%",
        "新加坡": "雷阵雨, 30°C, 湿度 85%",
        "迪拜": "晴, 35°C, 湿度 30%",
    }
    return mock_db.get(city, f"未找到 {city} 的天气数据")


@mcp.tool()
def extract_number(text: str, hint: str = "") -> str:
    """
    从一段文本中提取指定字段的数字（用于把检索文本转成可计算的纯数字）

    Args:
        text: 包含数字的文本片段
        hint: 提示要提取什么数字，如 "按量付费每小时价格"、"温度"
    """
    if "价" in hint or "钱" in hint or "费" in hint:
        m = re.search(r"[¥$￥]\s*(\d+(?:\.\d+)?)", text)
        if m:
            return m.group(1)
    if "温度" in hint or "气温" in hint or "°C" in hint:
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*°?[Cc]?", text)
        if m:
            return m.group(1)
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not nums:
        return f"未在文本中找到数字"
    if len(nums) == 1:
        return nums[0]
    return f"找到多个数字 {nums}，请用更精确的 hint"


@mcp.tool()
def kb_search(query: str) -> str:
    """
    在企业知识库中检索相关文档（基于 Hybrid 检索 + Rerank）

    Args:
        query: 检索问题
    """
    try:
        import rag
        chunks = rag.search(query, top_k=2, mode="hybrid", use_rerank=True)
        if not chunks:
            return "知识库中未找到相关信息"
        snippets = [c["content"][:300] for c in chunks]
        return "\n---\n".join(snippets)
    except Exception as e:
        return f"知识库检索失败: {e}"


# ========== Resources ==========

@mcp.resource("kb://documents")
def list_documents() -> str:
    """列出知识库中所有已索引的文档"""
    try:
        import rag
        stats = rag.stats()
        return (
            f"知识库统计:\n"
            f"  总分块数: {stats.get('total_chunks', 0)}\n"
            f"  BM25 索引: {stats.get('bm25_indexed', 0)}\n"
            f"  状态: {'已初始化' if stats.get('initialized') else '空'}"
        )
    except Exception as e:
        return f"读取失败: {e}"


# ========== Prompts ==========

@mcp.prompt()
def cost_calc_prompt(product: str, hours: str = "24") -> str:
    """
    生成 GPU 实例成本计算的标准提示词模板（MCP 协议要求参数都是 string）

    Args:
        product: GPU 产品名（如 "A100"）
        hours: 计算时长（小时数，字符串形式，如 "24"）
    """
    return f"""请按以下步骤回答：
1. 用 kb_search 工具查询 "{product} 的按量付费每小时价格"
2. 用 extract_number 工具从结果中提取价格
3. 用 calculator 工具计算 价格 × {hours}
4. 给出最终答案，包含计算过程和总价"""


# ========== 启动 ==========
if __name__ == "__main__":
    # stdio 模式（默认）
    mcp.run()
