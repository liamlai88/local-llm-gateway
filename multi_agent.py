"""
Multi-Agent orchestration demo.
"""
import re
import time
from typing import Any, Dict, List, Optional

import agent as agent_mod

KNOWN_CITIES = [
    "杭州", "北京", "上海", "深圳", "广州", "成都", "武汉", "西安",
    "新加坡", "迪拜", "伦敦", "纽约", "东京", "首尔", "曼谷",
]


def _step(role: str, action: str, output: Any, inputs: Optional[Dict] = None,
          status: str = "ok") -> Dict:
    return {"role": role, "action": action, "inputs": inputs or {}, "output": output, "status": status}


def _clean_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None


def _fmt_number(value: Optional[float]) -> str:
    if value is None:
        return "未知"
    return str(int(value)) if float(value).is_integer() else f"{value:.2f}".rstrip("0").rstrip(".")


def _extract_product(question: str) -> Optional[str]:
    match = re.search(r"\b(A10|A100|H100|L20|V100|T4)\b", question, re.IGNORECASE)
    return match.group(1).upper() if match else None


def _extract_hours(question: str) -> Optional[float]:
    fixed = {"半天": 12, "一天": 24, "一日": 24, "一周": 168, "一个星期": 168, "一个月": 720, "一月": 720, "一年": 8760}
    for word, hours in fixed.items():
        if word in question:
            return float(hours)
    units = [
        (r"(\d+(?:\.\d+)?)\s*(小时|个小时|h|H)", 1),
        (r"(\d+(?:\.\d+)?)\s*(天|日)", 24),
        (r"(\d+(?:\.\d+)?)\s*(周|星期)", 168),
        (r"(\d+(?:\.\d+)?)\s*(个月|月)", 720),
        (r"(\d+(?:\.\d+)?)\s*年", 8760),
    ]
    for pattern, multiplier in units:
        match = re.search(pattern, question)
        if match:
            return float(match.group(1)) * multiplier
    return None


def _extract_cities(question: str) -> List[str]:
    return [city for city in KNOWN_CITIES if city in question]


def _extract_simple_expression(question: str) -> Optional[str]:
    text = question
    for src, dst in {"乘以": "*", "乘": "*", "×": "*", "x": "*", "X": "*", "除以": "/", "除": "/", "加上": "+", "加": "+", "减去": "-", "减": "-"}.items():
        text = text.replace(src, dst)
    match = re.search(r"(\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    return f"{match.group(1)} {match.group(2)} {match.group(3)}"


class CoordinatorAgent:
    name = "CoordinatorAgent"

    def run(self, question: str) -> Dict:
        product = _extract_product(question)
        hours = _extract_hours(question)
        cities = _extract_cities(question)
        expression = _extract_simple_expression(question)
        knowledge_terms = ["知识库", "价格", "定价", "规格", "显存", "实例", "GPU", "产品", "SLA"]
        calc_terms = ["多少钱", "成本", "总价", "计算", "算", "差", "减去", "乘以", "乘", "加", "除"]
        solution_terms = ["适合", "建议", "方案", "训练", "推理", "判断", "选型"]
        needs_retriever = bool(product or any(term in question for term in knowledge_terms))
        needs_weather = bool(cities and any(term in question for term in ["天气", "气温", "冷", "热"]))
        needs_calculator = bool(expression or hours or any(term in question for term in calc_terms))
        needs_solution = bool(any(term in question for term in solution_terms))
        if product and ("多少钱" in question or "成本" in question or "总价" in question):
            query = f"{product} 按量付费每小时价格"
        elif product:
            query = f"{product} 规格 显存 适用场景"
        else:
            query = question
        plan = {
            "question": question,
            "product": product,
            "hours": hours,
            "cities": cities,
            "expression": expression,
            "query": query,
            "needs": {"retriever": needs_retriever, "weather": needs_weather, "calculator": needs_calculator, "solution": needs_solution},
            "agents": ["CoordinatorAgent"],
        }
        if needs_retriever:
            plan["agents"].append("RetrieverAgent")
        if needs_weather:
            plan["agents"].append("WeatherAgent")
        if needs_calculator:
            plan["agents"].append("CalculatorAgent")
        if needs_solution:
            plan["agents"].append("SolutionAgent")
        plan["agents"].extend(["CriticAgent", "FinalizerAgent"])
        return plan


class RetrieverAgent:
    name = "RetrieverAgent"

    def run(self, query: str) -> Dict:
        observation = agent_mod.execute_tool("kb_search", {"query": query})
        return {"query": query, "observation": observation}


class WeatherAgent:
    name = "WeatherAgent"

    def run(self, cities: List[str]) -> Dict:
        return {"observations": {city: agent_mod.execute_tool("get_weather", {"city": city}) for city in cities}}


class CalculatorAgent:
    name = "CalculatorAgent"

    def run(self, plan: Dict, artifacts: Dict) -> Dict:
        calculations = []
        if plan.get("expression"):
            expression = plan["expression"]
            result = agent_mod.execute_tool("calculator", {"expression": expression})
            calculations.append({"type": "direct_math", "expression": expression, "result": result})
        retrieval = artifacts.get("retrieval", {})
        if retrieval.get("observation") and plan.get("hours"):
            price_text = agent_mod.execute_tool("extract_number", {"text": retrieval["observation"], "hint": "按量付费每小时价格"})
            unit_price = _clean_number(price_text)
            hours = plan["hours"]
            if unit_price is not None:
                expression = f"{_fmt_number(unit_price)} * {_fmt_number(hours)}"
                total = agent_mod.execute_tool("calculator", {"expression": expression})
                calculations.append({"type": "rag_cost", "unit_price": unit_price, "hours": hours, "expression": expression, "result": total})
            else:
                calculations.append({"type": "rag_cost", "error": f"未能从知识库结果提取价格: {price_text}"})
        weather = artifacts.get("weather", {}).get("observations", {})
        if len(weather) >= 2 and any(term in plan["question"] for term in ["差", "减去", "冷", "热"]):
            cities = list(weather.keys())[:2]
            first_temp = agent_mod.execute_tool("extract_number", {"text": weather[cities[0]], "hint": "气温"})
            second_temp = agent_mod.execute_tool("extract_number", {"text": weather[cities[1]], "hint": "气温"})
            a = _clean_number(first_temp)
            b = _clean_number(second_temp)
            if a is not None and b is not None:
                expression = f"{_fmt_number(a)} - {_fmt_number(b)}"
                result = agent_mod.execute_tool("calculator", {"expression": expression})
                calculations.append({"type": "weather_diff", "cities": cities, "expression": expression, "result": result})
        return {"calculations": calculations}


class SolutionAgent:
    name = "SolutionAgent"

    def run(self, plan: Dict, artifacts: Dict) -> Dict:
        product = plan.get("product")
        context = artifacts.get("retrieval", {}).get("observation", "")
        if not product:
            return {"advice": "未识别到具体产品，建议先补充产品型号或业务约束。"}
        if product == "A10":
            advice = "A10 更适合推理、轻量微调和成本敏感场景；大规模训练通常不作为首选。"
        elif product == "A100":
            advice = "A100 适合中大型训练、微调和高吞吐推理；如果是超大规模训练，建议评估集群网络和调度能力。"
        elif product == "H100":
            advice = "H100 更适合高性能训练和大模型场景，但成本通常更高，需要结合利用率评估 ROI。"
        else:
            advice = f"{product} 的适用性需要结合显存、吞吐、价格和任务类型判断。"
        if "知识库检索失败" in context or "未找到" in context:
            advice += " 当前知识库证据不足，建议先补齐产品规格和价格文档。"
        return {"advice": advice}


class CriticAgent:
    name = "CriticAgent"

    # 空话兜底关键词（SolutionAgent 给的"未识别"类回答）
    EMPTY_PHRASES = ["未识别到", "建议先补充", "业务约束", "无法确定", "需要更多信息"]

    def run(self, plan: Dict, artifacts: Dict) -> Dict:
        issues = []
        checks = []
        if plan["needs"]["retriever"]:
            ok = bool(artifacts.get("retrieval", {}).get("observation"))
            checks.append({"name": "retrieval_ran", "passed": ok})
            if not ok:
                issues.append("RetrieverAgent 未产出知识库结果")
        if plan["needs"]["weather"]:
            ok = bool(artifacts.get("weather", {}).get("observations"))
            checks.append({"name": "weather_ran", "passed": ok})
            if not ok:
                issues.append("WeatherAgent 未产出天气结果")
        if plan["needs"]["calculator"]:
            ok = bool(artifacts.get("calculation", {}).get("calculations", []))
            checks.append({"name": "calculator_ran", "passed": ok})
            if not ok:
                issues.append("CalculatorAgent 未产出计算结果")

        # 新增: 检查 Solution 是否给了空话
        if plan["needs"]["solution"]:
            advice = artifacts.get("solution", {}).get("advice", "")
            is_empty = any(p in advice for p in self.EMPTY_PHRASES)
            checks.append({"name": "solution_substantive", "passed": not is_empty})
            if is_empty:
                issues.append(f"SolutionAgent 输出为空话/兜底回复: {advice[:50]}")

        # 新增: 知识库检索到内容但 Solution 没用上 (未识别产品但 KB 有相关文档)
        if plan["needs"]["retriever"] and not plan.get("product"):
            kb_text = artifacts.get("retrieval", {}).get("observation", "")
            has_kb_content = "产品名称" in kb_text or "GPU" in kb_text
            checks.append({"name": "kb_content_utilized", "passed": False if has_kb_content else True})
            if has_kb_content:
                issues.append("知识库检索到产品文档但规则路径未识别到产品 (建议 LLM 兜底)")

        for key in ("retrieval", "weather"):
            text = str(artifacts.get(key, ""))
            if "失败" in text or "Error" in text:
                issues.append(f"{key} 阶段包含失败信息")
        return {"passed": not issues, "issues": issues, "checks": checks}


class LLMFallbackAgent:
    """
    LLM 兜底 Agent
    当规则路径的 Critic 警告时启用。基于已有 artifacts（特别是 retrieval 结果）
    用 LLM 直接生成开放式答案，弥补规则路由对未知问题模式的盲区。
    """
    name = "LLMFallbackAgent"

    # 兜底用强模型：硬编码百炼 Qwen-Turbo（开放问题需要好推理）
    FALLBACK_MODEL = "qwen-turbo"
    FALLBACK_PROVIDER = "bailian"

    def run(self, question: str, artifacts: Dict, model: str, provider: str) -> Dict:
        retrieval = artifacts.get("retrieval", {}).get("observation", "")
        weather = artifacts.get("weather", {}).get("observations", {})
        critic_issues = artifacts.get("critic", {}).get("issues", [])

        context_parts = []
        if retrieval:
            context_parts.append(f"【知识库检索结果】\n{retrieval}")
        if weather:
            context_parts.append(f"【天气信息】\n{weather}")
        if critic_issues:
            context_parts.append(f"【规则路径未能完成的环节】\n{'; '.join(critic_issues)}")

        context = "\n\n".join(context_parts) if context_parts else "（无前置数据）"

        system_prompt = """你是企业 AI 解决方案顾问。基于已检索的资料回答用户问题。

要求：
1. 必须基于【知识库检索结果】给出建议，不要编造文档里没有的产品/价格/规格
2. 如果用户问推荐型/选型问题，应该结合检索结果做对比并给出明确推荐（要提到具体产品名）
3. 如果检索结果不足以完整回答，明确告知客户哪些信息需要补充
4. 回答简洁专业，4-8 句话，不要废话"""

        user_prompt = f"""用户问题：{question}

可用上下文：
{context}

请给出专业回答："""

        # 优先用百炼 Qwen-Turbo 兜底；失败则降级到调用方的 model/provider
        try:
            answer = agent_mod.call_llm(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.FALLBACK_MODEL, provider=self.FALLBACK_PROVIDER,
            )
            return {"answer": answer.strip(), "mode": "llm_fallback_bailian", "context_used": bool(context_parts)}
        except Exception as exc_bailian:
            # 百炼不可用时降级到调用方传入的模型
            try:
                answer = agent_mod.call_llm(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    model=model, provider=provider,
                )
                return {"answer": answer.strip(), "mode": f"llm_fallback_{provider}", "context_used": bool(context_parts)}
            except Exception as exc_local:
                return {
                    "answer": f"LLM 兜底失败 (bailian: {exc_bailian}; local: {exc_local})",
                    "mode": "llm_fallback_error",
                }


class FinalizerAgent:
    name = "FinalizerAgent"

    def run(self, plan: Dict, artifacts: Dict, model: str, provider: str, use_llm_final: bool = False) -> Dict:
        deterministic = self._build_answer(plan, artifacts)
        if not use_llm_final:
            return {"answer": deterministic, "mode": "template"}
        prompt = f"""请把下面 multi-agent 执行结果整理成简洁中文答案，不要编造缺失信息。

用户问题：{plan['question']}

执行结果：
{artifacts}

模板答案：
{deterministic}
"""
        try:
            polished = agent_mod.call_llm([
                {"role": "system", "content": "你是严谨的企业 AI 方案总结助手。"},
                {"role": "user", "content": prompt},
            ], model=model, provider=provider)
            return {"answer": polished.strip(), "mode": "llm"}
        except Exception as exc:
            return {"answer": deterministic, "mode": "template_fallback", "error": str(exc)}

    def _build_answer(self, plan: Dict, artifacts: Dict) -> str:
        parts = []
        product = plan.get("product")
        retrieval = artifacts.get("retrieval", {})
        if retrieval.get("observation"):
            parts.append(f"知识库检索 query：{retrieval['query']}")
        for item in artifacts.get("calculation", {}).get("calculations", []):
            if item.get("type") == "rag_cost" and "result" in item:
                parts.append(f"按知识库提取的单价 ¥{_fmt_number(item['unit_price'])}/小时，运行 {_fmt_number(item['hours'])} 小时，总价约 ¥{item['result']}。")
            elif item.get("type") == "weather_diff" and "result" in item:
                parts.append(f"{item['cities'][0]} 与 {item['cities'][1]} 的气温差计算为 {item['expression']} = {item['result']}°C。")
            elif item.get("type") == "direct_math" and "result" in item:
                parts.append(f"计算结果：{item['expression']} = {item['result']}。")
            elif item.get("error"):
                parts.append(item["error"])
        solution = artifacts.get("solution", {})
        if solution.get("advice"):
            parts.append(solution["advice"])
        critic = artifacts.get("critic", {})
        if critic.get("issues"):
            parts.append("审查提示：" + "；".join(critic["issues"]))
        if not parts:
            if product:
                return f"已识别到产品 {product}，但当前问题没有触发可执行的专门 Agent。"
            return "当前问题没有触发知识库、天气或计算类 Agent。"
        return "\n".join(parts)


def run_multi_agent(
    question: str,
    model: str = "qwen2.5-1.5b",
    provider: str = "local",
    use_llm_final: bool = False,
    enable_fallback: bool = True,
) -> Dict:
    """
    Multi-Agent 主流程（混合架构）：
    - 快路径：规则路由 + 工具调用，毫秒级返回
    - 慢路径（自动）：当 Critic 警告时，启用 LLM Fallback Agent 基于已有 artifacts 兜底
    """
    start = time.time()
    trace = []
    artifacts: Dict[str, Any] = {}
    coordinator = CoordinatorAgent()
    plan = coordinator.run(question)
    trace.append(_step(coordinator.name, "plan", plan, {"question": question}))

    if plan["needs"]["retriever"]:
        retriever = RetrieverAgent()
        artifacts["retrieval"] = retriever.run(plan["query"])
        trace.append(_step(retriever.name, "kb_search", artifacts["retrieval"], {"query": plan["query"]}))
    if plan["needs"]["weather"]:
        weather = WeatherAgent()
        artifacts["weather"] = weather.run(plan["cities"])
        trace.append(_step(weather.name, "get_weather", artifacts["weather"], {"cities": plan["cities"]}))
    if plan["needs"]["calculator"]:
        calculator = CalculatorAgent()
        artifacts["calculation"] = calculator.run(plan, artifacts)
        trace.append(_step(calculator.name, "calculate", artifacts["calculation"]))
    if plan["needs"]["solution"]:
        solution = SolutionAgent()
        artifacts["solution"] = solution.run(plan, artifacts)
        trace.append(_step(solution.name, "advise", artifacts["solution"]))

    critic = CriticAgent()
    artifacts["critic"] = critic.run(plan, artifacts)
    trace.append(_step(critic.name, "audit", artifacts["critic"], status="ok" if artifacts["critic"]["passed"] else "warning"))

    rule_path_passed = artifacts["critic"]["passed"]

    # 慢路径：LLM 兜底
    if not rule_path_passed and enable_fallback:
        fallback = LLMFallbackAgent()
        fb_result = fallback.run(question, artifacts, model=model, provider=provider)
        artifacts["fallback"] = fb_result
        trace.append(_step(fallback.name, "llm_fallback", fb_result, {"reason": "critic_warning"}))
        plan["agents"].append("LLMFallbackAgent")

        return {
            "answer": fb_result["answer"],
            "status": "success" if "error" not in fb_result["mode"] else "fallback_failed",
            "mode": "multi_agent+llm_fallback",
            "agents": plan["agents"],
            "plan": plan,
            "artifacts": artifacts,
            "trace": trace,
            "latency_ms": round((time.time() - start) * 1000, 1),
            "path": "slow",
        }

    finalizer = FinalizerAgent()
    final = finalizer.run(plan, artifacts, model=model, provider=provider, use_llm_final=use_llm_final)
    trace.append(_step(finalizer.name, "finalize", final, {"use_llm_final": use_llm_final}))

    return {
        "answer": final["answer"],
        "status": "success" if rule_path_passed else "needs_review",
        "mode": "multi_agent",
        "agents": plan["agents"],
        "plan": plan,
        "artifacts": artifacts,
        "trace": trace,
        "latency_ms": round((time.time() - start) * 1000, 1),
        "path": "fast",
    }
