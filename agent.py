"""
Agent 模块 - ReAct 模式实现
- 工具注册表（可扩展）
- ReAct 循环（Thought → Action → Observation）
- 解析 + 错误兜底
"""
import os
import re
import json
import time
import logging
import requests
from typing import Dict, List, Callable, Any

log = logging.getLogger("agent")

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"  # 本地 Ollama
BAILIAN_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_API_KEY = "sk-demo-002"  # 调自己 Gateway 的 RAG 工具时用


# ========== 工具注册表 ==========
TOOLS: Dict[str, Dict] = {}


def register_tool(name: str, description: str, parameters: Dict):
    """装饰器：注册一个工具"""
    def decorator(func: Callable):
        TOOLS[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "func": func,
        }
        return func
    return decorator


# ========== 内置工具 ==========
@register_tool(
    name="calculator",
    description="数学表达式计算，支持 +-*/()% 和常见数学函数",
    parameters={"expression": "string, 数学表达式如 '15 * 0.6'"},
)
def calculator(expression: str) -> str:
    try:
        # 安全 eval：只允许数字和基本运算
        if not re.match(r"^[\d\s\+\-\*\/\.\(\)\%]+$", expression):
            return f"Error: 表达式包含不允许字符: {expression}"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


@register_tool(
    name="get_weather",
    description="查询城市当前天气（mock 数据用于演示）",
    parameters={"city": "string, 城市名"},
)
def get_weather(city: str) -> str:
    # mock 数据
    mock_db = {
        "杭州": "晴, 22°C, 湿度 60%",
        "北京": "多云, 15°C, 湿度 45%",
        "上海": "小雨, 19°C, 湿度 75%",
        "新加坡": "雷阵雨, 30°C, 湿度 85%",
        "迪拜": "晴, 35°C, 湿度 30%",
    }
    return mock_db.get(city, f"未找到 {city} 的天气数据")


@register_tool(
    name="kb_search",
    description="在企业知识库中检索相关文档（用于回答闭域问题）",
    parameters={"query": "string, 检索问题"},
)
def kb_search(query: str) -> str:
    """调自己 Gateway 的 RAG 接口"""
    try:
        resp = requests.post(
            "http://localhost:8000/v1/rag/query",
            headers={"Authorization": f"Bearer {DEFAULT_API_KEY}",
                     "Content-Type": "application/json"},
            json={"question": query, "mode": "hybrid", "rerank": True, "top_k": 2, "model": "fast"},
            timeout=30,
        )
        data = resp.json()
        if "sources" in data and data["sources"]:
            # 返回检索到的内容片段（精炼）
            snippets = [s["content"][:200] for s in data["sources"][:2]]
            return "\n---\n".join(snippets)
        return "知识库中未找到相关信息"
    except Exception as e:
        return f"知识库调用失败: {e}"


# ========== ReAct Prompt ==========
def build_system_prompt(few_shot: bool = True) -> str:
    tools_desc = "\n".join([
        f"- {t['name']}({', '.join(f'{k}: {v}' for k, v in t['parameters'].items())}): {t['description']}"
        for t in TOOLS.values()
    ])

    base = f"""你是一个严格遵守纪律的 AI 助手。

【铁律 - 违反任何一条都是严重错误】
1. **禁止心算任何数字**：哪怕 "1+1"，必须用 calculator
2. **禁止凭记忆回答天气、价格、规格**：必须用 get_weather / kb_search
3. **禁止跳过工具直接给 Final Answer**：除非问题完全不涉及外部信息（如"你好"）
4. **Args 只能含工具定义的参数**：不能添加 query/filter 等不存在的字段
5. **工具失败时必须重试或换工具**，不允许直接放弃

【可用工具】
{tools_desc}

【响应格式】（每次只能选一种）

格式 A（调用工具）：
Thought: <思考>
Action: <工具名>
Args: <JSON 参数>

格式 B（最终回答）：
Thought: <最终思考>
Final Answer: <答案>"""

    if not few_shot:
        return base

    examples = """

【完整示例 1: 简单计算 - 也必须用工具】

User: 3 加 5 等于多少？

Thought: 涉及数字计算，按铁律必须用 calculator
Action: calculator
Args: {"expression": "3 + 5"}

Observation: 8

Thought: 已得到结果
Final Answer: 3 加 5 等于 8

【完整示例 2: 多工具组合】

User: 上海气温减去北京气温是多少？

Thought: 我需要分别查询两个城市气温
Action: get_weather
Args: {"city": "上海"}

Observation: 小雨, 19°C, 湿度 75%

Thought: 上海 19°C，再查北京
Action: get_weather
Args: {"city": "北京"}

Observation: 多云, 15°C, 湿度 45%

Thought: 现在用计算器算 19 - 15
Action: calculator
Args: {"expression": "19 - 15"}

Observation: 4

Thought: 计算完成
Final Answer: 上海比北京高 4°C

【完整示例 2: RAG + 计算】

User: 知识库里 X 产品包月多少钱，按 12 个月算总价

Thought: 先查知识库
Action: kb_search
Args: {"query": "X 产品包月价格"}

Observation: X 产品包月 ¥1000

Thought: 用计算器算 1000 × 12
Action: calculator
Args: {"expression": "1000 * 12"}

Observation: 12000

Thought: 已得出答案
Final Answer: 12 个月总价 ¥12000"""

    return base + examples


# ========== ReAct 解析 ==========
def parse_response(text: str) -> Dict:
    """解析 LLM 输出，返回 {type, ...}"""
    # 尝试匹配 Final Answer
    m = re.search(r"Final Answer:\s*(.+?)(?:\n\n|$)", text, re.DOTALL)
    if m:
        return {"type": "final", "answer": m.group(1).strip()}

    # 尝试匹配 Action + Args
    action_m = re.search(r"Action:\s*(\w+)", text)
    args_m = re.search(r"Args:\s*(\{.+?\})", text, re.DOTALL)

    if action_m and args_m:
        try:
            args = json.loads(args_m.group(1))
            return {
                "type": "action",
                "tool": action_m.group(1),
                "args": args,
            }
        except json.JSONDecodeError as e:
            return {"type": "error", "error": f"Args JSON 解析失败: {e}"}

    return {"type": "error", "error": "无法解析 LLM 输出格式"}


# ========== 调用 LLM ==========
def call_llm(messages: List[Dict], model: str = "qwen2.5-1.5b", provider: str = "local") -> str:
    """provider: 'local' (Ollama) 或 'bailian' (百炼)"""
    if provider == "bailian":
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY 未设置")
        resp = requests.post(
            BAILIAN_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "max_tokens": 500},
            timeout=60,
        )
    else:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": model, "messages": messages, "max_tokens": 500},
            timeout=60,
        )
    return resp.json()["choices"][0]["message"]["content"]


# ========== 执行工具 ==========
def execute_tool(name: str, args: Dict) -> str:
    if name not in TOOLS:
        return f"Error: 未知工具 '{name}'，可用工具: {list(TOOLS.keys())}"
    try:
        return TOOLS[name]["func"](**args)
    except Exception as e:
        return f"Tool {name} 执行错误: {e}"


# ========== ReAct 主循环 ==========
def run_agent(question: str, max_iterations: int = 5, model: str = "qwen2.5-1.5b",
              provider: str = "local", few_shot: bool = True) -> Dict:
    """
    返回 {answer, trace, iterations, latency_ms}
    """
    messages = [
        {"role": "system", "content": build_system_prompt(few_shot=few_shot)},
        {"role": "user", "content": question},
    ]
    trace = []
    start = time.time()

    for i in range(max_iterations):
        # 调用 LLM
        llm_output = call_llm(messages, model=model, provider=provider)
        log.info(f"[Iter {i+1}] LLM output: {llm_output[:200]}...")

        # 解析
        parsed = parse_response(llm_output)

        if parsed["type"] == "final":
            trace.append({
                "step": i + 1,
                "type": "final",
                "raw_output": llm_output,
                "answer": parsed["answer"],
            })
            return {
                "answer": parsed["answer"],
                "trace": trace,
                "iterations": i + 1,
                "latency_ms": round((time.time() - start) * 1000, 1),
                "status": "success",
            }

        if parsed["type"] == "error":
            trace.append({
                "step": i + 1,
                "type": "error",
                "raw_output": llm_output,
                "error": parsed["error"],
            })
            # 让模型再试一次（把错误反馈回去）
            messages.append({"role": "assistant", "content": llm_output})
            messages.append({"role": "user", "content": f"格式错误: {parsed['error']}\n请严格按 Thought/Action/Args 或 Final Answer 格式重新响应。"})
            continue

        # parsed["type"] == "action"
        tool_name = parsed["tool"]
        tool_args = parsed["args"]
        observation = execute_tool(tool_name, tool_args)

        trace.append({
            "step": i + 1,
            "type": "action",
            "tool": tool_name,
            "args": tool_args,
            "observation": observation,
            "raw_output": llm_output,
        })

        # 拼接对话历史
        messages.append({"role": "assistant", "content": llm_output})
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    # 超过最大循环
    return {
        "answer": "达到最大循环次数，未能得出答案",
        "trace": trace,
        "iterations": max_iterations,
        "latency_ms": round((time.time() - start) * 1000, 1),
        "status": "max_iterations_exceeded",
    }


def list_tools() -> List[Dict]:
    return [
        {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}
        for t in TOOLS.values()
    ]
