"""
实验 3: LangGraph StateGraph 重写 Multi-Agent 混合架构 (vs 你的 multi_agent.py)

对比目标:
  - 你的 multi_agent.py: ~400 行, 6 Agent + 自定义 Critic + Fallback
  - LangGraph 版本: ~150 行, 用 StateGraph 表达同样逻辑

LangGraph 关键概念实操:
  1. State (TypedDict) - 强类型状态板
  2. Node - 处理 State 的函数
  3. add_conditional_edges - 条件分支 (你的 Coordinator 路由)
  4. END - 终止节点
"""
import os
import re
import time
import json
from typing import TypedDict, List, Optional, Annotated
from operator import add

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END


# ========== Step 1: 定义 State (这就是 LangGraph 的灵魂) ==========
# 对比你的 multi_agent.py 用 dict[str, Any], LangGraph 强制 schema
class AgentState(TypedDict):
    question: str
    product: Optional[str]          # 识别到的产品名
    hours: Optional[float]          # 时长
    cities: List[str]               # 城市列表
    retrieval: Optional[str]        # KB 检索结果
    weather: dict                   # 天气数据
    calculation: Optional[str]      # 计算结果
    needs_fallback: bool            # 是否需要 LLM 兜底
    final_answer: str               # 最终答案
    trace: Annotated[List[str], add]  # 执行轨迹 (Annotated[..., add] 表示自动累加)


# ========== Step 2: 工具实现 (复用 mock 数据) ==========
WEATHER_DB = {
    "杭州": "晴, 22°C", "北京": "多云, 15°C", "上海": "小雨, 19°C",
    "迪拜": "晴, 35°C", "新加坡": "雷阵雨, 30°C",
}
KB = {
    "A100": "ecs.gn7e-c12g1.3xlarge / NVIDIA A100 / 80GB / ¥68/小时",
    "A10": "ecs.gn7i-c8g1.2xlarge / NVIDIA A10 / 24GB / ¥18/小时",
    "H100": "灵骏 H100 集群 / 项目制定价",
}


def extract_number(text: str) -> Optional[float]:
    # 优先匹配 "¥XX" 价格 (避免抓到产品型号里的数字如 ecs.gn7e)
    m = re.search(r"¥\s*(\d+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1))
    # 其次匹配 "XX°C" 温度
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*°", text)
    if m:
        return float(m.group(1))
    # 最后兜底: 最后一个数字 (一般是关键值)
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(nums[-1]) if nums else None


# ========== Step 3: 定义 Node (每个 Agent 是一个 Node) ==========

def coordinator_node(state: AgentState) -> dict:
    """规则路由 (对应你的 CoordinatorAgent)"""
    q = state["question"]
    product = next((p for p in ["A100", "A10", "H100"] if p in q), None)
    cities = [c for c in WEATHER_DB.keys() if c in q]

    hours_match = re.search(r"(\d+)\s*(小时|天|月)", q)
    hours = None
    if "一天" in q or "一日" in q:
        hours = 24
    elif hours_match:
        n = float(hours_match.group(1))
        unit = hours_match.group(2)
        hours = n * (1 if unit == "小时" else 24 if unit == "天" else 720)

    return {
        "product": product, "hours": hours, "cities": cities,
        "trace": [f"coordinator: product={product}, cities={cities}, hours={hours}"],
    }


def retriever_node(state: AgentState) -> dict:
    """RAG 检索 (对应 RetrieverAgent)"""
    if not state["product"]:
        return {"retrieval": None, "trace": ["retriever: skip (no product)"]}
    result = KB.get(state["product"], "未找到")
    return {"retrieval": result, "trace": [f"retriever: {result[:50]}..."]}


def weather_node(state: AgentState) -> dict:
    """天气查询 (对应 WeatherAgent)"""
    if not state["cities"]:
        return {"weather": {}, "trace": ["weather: skip"]}
    obs = {c: WEATHER_DB.get(c, "未知") for c in state["cities"]}
    return {"weather": obs, "trace": [f"weather: {obs}"]}


def calculator_node(state: AgentState) -> dict:
    """计算节点 (对应 CalculatorAgent)"""
    # 场景 1: RAG + 时长 → 算成本
    if state.get("retrieval") and state.get("hours"):
        price = extract_number(state["retrieval"])
        if price:
            total = price * state["hours"]
            return {"calculation": f"¥{int(total)}",
                    "trace": [f"calculator: {price} * {state['hours']} = ¥{int(total)}"]}
    # 场景 2: 两城气温差
    if len(state.get("weather", {})) >= 2:
        cities = list(state["weather"].keys())[:2]
        temps = [extract_number(state["weather"][c]) for c in cities]
        if all(temps):
            diff = temps[0] - temps[1]
            return {"calculation": f"{int(diff)}°C",
                    "trace": [f"calculator: {temps[0]}-{temps[1]} = {diff}°C ({cities[0]} 与 {cities[1]})"]}
    return {"calculation": None, "trace": ["calculator: skip"]}


def critic_node(state: AgentState) -> dict:
    """质量裁判 (对应 CriticAgent，你的关键创新!)"""
    issues = []
    # 检测 1: 有 product 但没拿到 retrieval
    if state.get("product") and not state.get("retrieval"):
        issues.append("retrieval 未产出")
    # 检测 2: 有 hours/cities 但没算结果
    if (state.get("hours") or len(state.get("weather", {})) >= 2) and not state.get("calculation"):
        issues.append("calculator 未产出")
    # 检测 3: 既没 product 也没 cities 也没 calculation = 盲区
    if not state.get("product") and not state.get("cities") and not state.get("calculation"):
        issues.append("规则路径无法处理 (盲区)")

    needs_fallback = bool(issues)
    return {
        "needs_fallback": needs_fallback,
        "trace": [f"critic: {'警告 - ' + '; '.join(issues) if issues else 'OK'}"],
    }


def llm_fallback_node(state: AgentState) -> dict:
    """LLM 兜底 (对应 LLMFallbackAgent)"""
    llm = ChatOpenAI(
        model="qwen-turbo",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        temperature=0.0,
    )
    context = f"知识库: {state.get('retrieval', '无')}\n天气: {state.get('weather', {})}\n计算: {state.get('calculation', '无')}"
    prompt = f"""你是 GPU 选型顾问。基于以下上下文回答用户问题，必要时给出具体产品推荐 (A100/A10/H100)。

上下文: {context}

问题: {state['question']}

简洁专业回答 (4-6 句):"""
    answer = llm.invoke(prompt).content
    return {"final_answer": answer.strip(), "trace": [f"llm_fallback: {answer[:80]}..."]}


def finalizer_node(state: AgentState) -> dict:
    """合成最终答案 (对应 FinalizerAgent template 模式)"""
    parts = []
    if state.get("retrieval"):
        parts.append(f"检索: {state['retrieval']}")
    if state.get("weather"):
        parts.append(f"天气: {state['weather']}")
    if state.get("calculation"):
        parts.append(f"计算结果: {state['calculation']}")
    return {"final_answer": " | ".join(parts) if parts else "无结果",
            "trace": ["finalizer: template 模式合成"]}


# ========== Step 4: 路由函数 (条件分支) ==========
def route_after_critic(state: AgentState) -> str:
    """Critic 后的路由: 警告 → fallback, 否则 → finalizer"""
    return "llm_fallback" if state["needs_fallback"] else "finalizer"


# ========== Step 5: 构建 Graph ==========
def build_graph():
    workflow = StateGraph(AgentState)

    # 加节点
    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("weather", weather_node)
    workflow.add_node("calculator", calculator_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("llm_fallback", llm_fallback_node)
    workflow.add_node("finalizer", finalizer_node)

    # 加边 (流程)
    workflow.set_entry_point("coordinator")
    workflow.add_edge("coordinator", "retriever")
    workflow.add_edge("retriever", "weather")
    workflow.add_edge("weather", "calculator")
    workflow.add_edge("calculator", "critic")

    # 条件分支 (Critic 后的路由)
    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {"llm_fallback": "llm_fallback", "finalizer": "finalizer"},
    )

    workflow.add_edge("llm_fallback", END)
    workflow.add_edge("finalizer", END)

    return workflow.compile()


# ========== Step 6: 测试 (与你之前 Multi-Agent 实验同 4 题) ==========
TESTS = [
    {"id": "命中-1", "question": "杭州和北京的气温差是多少？", "expected": "7"},
    {"id": "命中-2", "question": "阿里云 A100 按量付费一天多少钱？", "expected": "1632"},
    {"id": "盲区-1", "question": "我想训练 70B 大模型，预算 100 万，给我推荐用什么 GPU？", "expected_kw": ["A100", "H100"]},
    {"id": "盲区-2", "question": "我们公司刚开始做 AI 推理服务，每天 10 万次请求，应该怎么选 GPU？", "expected_kw": ["A10", "推理"]},
]

if __name__ == "__main__":
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("❌ DASHSCOPE_API_KEY 未设置")
        exit(1)

    print("=" * 75)
    print("LangGraph StateGraph Multi-Agent 测试")
    print("=" * 75)
    app = build_graph()

    summary = []
    for t in TESTS:
        print("\n" + "=" * 75)
        print(f"[{t['id']}] {t['question']}")

        initial_state = {
            "question": t["question"],
            "product": None, "hours": None, "cities": [],
            "retrieval": None, "weather": {}, "calculation": None,
            "needs_fallback": False, "final_answer": "", "trace": [],
        }

        start = time.time()
        result = app.invoke(initial_state)
        elapsed = (time.time() - start) * 1000

        # 评判
        answer = result["final_answer"]
        if "expected" in t:
            ok = t["expected"] in answer
        else:
            ok = any(kw in answer for kw in t["expected_kw"])

        # 输出
        print(f"答: {answer[:200]}")
        print(f"延迟: {elapsed:.0f}ms")
        print(f"路径: {' → '.join([s.split(':')[0] for s in result['trace']])}")
        print(f"评判: {'✅' if ok else '❌'}")
        summary.append({"id": t["id"], "ok": ok, "elapsed": elapsed,
                        "fallback": result.get("needs_fallback", False)})

    # 汇总
    print("\n" + "=" * 75)
    print("📊 总结")
    print("=" * 75)
    correct = sum(1 for s in summary if s["ok"])
    fallback_count = sum(1 for s in summary if s["fallback"])
    avg_lat = sum(s["elapsed"] for s in summary) / len(summary)
    print(f"  准确率:    {correct}/{len(TESTS)} = {correct/len(TESTS)*100:.0f}%")
    print(f"  Fallback:  {fallback_count}/{len(TESTS)} (走 LLM 兜底)")
    print(f"  平均延迟:  {avg_lat:.0f}ms")
    print()
    print("  对比你的 multi_agent.py:")
    print("    手写版本: 4/4 准确率, 平均 1.2s")
    print("    LangGraph 版: ~150 行 (含完整 Critic + Fallback)")
    print("    手写版本:    ~400 行")
