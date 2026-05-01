"""
实验 2 v2: LangGraph ReAct Agent (修复版)

修复:
  1. create_react_agent 已迁移到 langchain.agents (LangGraph V1.0)
  2. 改用百炼 qwen-turbo (原生支持 tool_calls), 同时保留 Ollama 对照
"""
import os
import time
from langchain_core.tools import tool


# ========== Step 1: 定义工具 ==========

@tool
def calculator(expression: str) -> str:
    """数学表达式计算，支持 +-*/() 和数字。例如: '15 * 0.6'"""
    import re
    if not re.match(r"^[\d\s\+\-\*\/\.\(\)\%]+$", expression):
        return f"Error: 表达式包含不允许字符"
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"


@tool
def get_weather(city: str) -> str:
    """查询城市当前天气。支持: 杭州/北京/上海/迪拜/纽约/东京"""
    mock_db = {
        "杭州": "晴, 22°C, 湿度 60%",
        "北京": "多云, 15°C, 湿度 45%",
        "上海": "小雨, 19°C, 湿度 75%",
        "迪拜": "晴, 35°C, 湿度 30%",
    }
    return mock_db.get(city, f"未找到 {city} 的天气数据")


@tool
def kb_search(query: str) -> str:
    """在阿里云 GPU 产品知识库中检索"""
    mock_kb = {
        "A100": "ecs.gn7e-c12g1.3xlarge / NVIDIA A100 / 80GB HBM2e / ¥68/小时, 包月 ¥35000",
        "A10": "ecs.gn7i-c8g1.2xlarge / NVIDIA A10 / 24GB GDDR6 / ¥18/小时, 包月 ¥9800",
        "H100": "灵骏 H100 集群 / 80GB HBM3 / 项目制定价",
    }
    for k, v in mock_kb.items():
        if k in query:
            return v
    return "未找到相关产品"


tools = [calculator, get_weather, kb_search]


# ========== Step 2: 创建两个 Agent (本地 vs 百炼) ==========

# 用新 API: from langchain.agents import create_agent
try:
    from langchain.agents import create_agent
except ImportError:
    # 降级到旧 API
    from langgraph.prebuilt import create_react_agent as create_agent


def build_local_agent():
    """本地 Ollama 1.5B (Tool Use Laziness 高发)"""
    from langchain_ollama import ChatOllama
    llm = ChatOllama(
        model="qwen2.5-1.5b",
        base_url="http://localhost:11434",
        temperature=0.0,
    )
    return create_agent(model=llm, tools=tools)


def build_bailian_agent():
    """百炼 qwen-turbo (原生 tool_calls 支持)"""
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model="qwen-turbo",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        temperature=0.0,
    )
    return create_agent(model=llm, tools=tools)


# ========== Step 3: 测试函数 ==========
def run_test(agent, question, label):
    print(f"\n--- {label} ---")
    try:
        start = time.time()
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        elapsed = (time.time() - start) * 1000

        final_msg = result["messages"][-1]
        answer = final_msg.content if hasattr(final_msg, "content") else str(final_msg)

        tool_calls = []
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(f"{tc['name']}({tc.get('args', {})})")

        print(f"答案: {answer[:200]}")
        print(f"延迟: {elapsed:.0f}ms / {len(result['messages'])} 条消息")
        print(f"工具链: {' → '.join(tool_calls) if tool_calls else '❌ 无 (Tool Use Laziness)'}")
        return {"ok": bool(tool_calls), "tools": tool_calls, "answer": answer, "elapsed": elapsed}
    except Exception as e:
        print(f"❌ 错误: {type(e).__name__}: {str(e)[:200]}")
        return {"ok": False, "error": str(e)}


# ========== Step 4: 跑测试 ==========
TESTS = [
    {"id": "Q1", "question": "15 乘以 0.6 等于多少？"},
    {"id": "Q2", "question": "杭州现在天气怎么样？"},
    {"id": "Q3", "question": "杭州的气温减去北京的气温是多少度？"},
    {"id": "Q4", "question": "阿里云 A100 实例按量付费一天大概多少钱？"},
]

results = {"local": [], "bailian": []}

# 检查百炼 API Key
has_bailian = bool(os.getenv("DASHSCOPE_API_KEY"))
print("=" * 75)
print(f"环境检查: 百炼 API Key {'✓ 已设置' if has_bailian else '❌ 未设置 (跳过 bailian 测试)'}")
print("=" * 75)

print("\n初始化 Agent...")
local_agent = build_local_agent()
print("✓ 本地 Ollama 1.5B Agent")
if has_bailian:
    bailian_agent = build_bailian_agent()
    print("✓ 百炼 Qwen-Turbo Agent")

# 跑测试
for t in TESTS:
    print("\n" + "=" * 75)
    print(f"[{t['id']}] {t['question']}")
    print("=" * 75)

    r1 = run_test(local_agent, t["question"], "本地 Ollama 1.5B")
    results["local"].append(r1)

    if has_bailian:
        r2 = run_test(bailian_agent, t["question"], "百炼 Qwen-Turbo")
        results["bailian"].append(r2)


# ========== Step 5: 总结 ==========
print("\n" + "=" * 75)
print("📊 LangGraph ReAct - 模型对比")
print("=" * 75)
for label, key in [("本地 Ollama 1.5B", "local"), ("百炼 Qwen-Turbo", "bailian")]:
    if not results[key]:
        continue
    used_tools = sum(1 for r in results[key] if r.get("ok"))
    avg_lat = sum(r.get("elapsed", 0) for r in results[key]) / len(results[key])
    print(f"\n  {label}:")
    print(f"    工具调用率: {used_tools}/{len(TESTS)} = {used_tools/len(TESTS)*100:.0f}%")
    print(f"    平均延迟:   {avg_lat:.0f}ms")
