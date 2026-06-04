"""
实验 #21 分层 Swarm：顶层规则路由 + 按子任务结构嵌套不同 orchestrator

命题：最优 agent 架构 = 顶层最便宜的路由 + 按子任务结构分发到不同 orchestrator。
用三臂对照证明"分层 = 按需分配算力"：C 的准确率逼近"全程最强"B，
但 LLM 调用次数(成本代理)/延迟接近"全程最便宜"A。

复用既有实验，不重写：
  #06 run_plan_execute_agent  (DAG / 条件路由)
  #12 run_reflection_agent    (带环 / 改到达标)
  单 agent run_agent          (单步)
  #20 swarm_demo 的 LLM 路由   (规则 miss 兜底)

度量
  主指标 = LLM 调用次数(provider 无关、完全公平的成本代理)
  次指标 = 平均延迟(标注 provider 构成)、准确率(关键词命中)、路由命中率

Provider 策略（混合，贴近生产）
  简单意图(FAQ/计费) → 本地 Ollama 1.5B（免费、快）
  复杂意图(技术/合规) → 云端 qwen-turbo（能力强）
运行：
  ollama serve & ; export DASHSCOPE_API_KEY=... ; python experiments/hierarchical_swarm.py
"""

import os
import re
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agent as agent_mod  # noqa: E402

LOCAL = ("qwen2.5-1.5b", "local")
CLOUD = ("qwen-turbo", "bailian")
RESULTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hierarchical_swarm_results.json"
)

# ========== LLM 调用计数器（成本代理，provider 无关）==========
_LLM_CALLS = 0
_orig_call_llm = agent_mod.call_llm


def _counting_call_llm(*args, **kwargs):
    global _LLM_CALLS
    _LLM_CALLS += 1
    return _orig_call_llm(*args, **kwargs)


agent_mod.call_llm = _counting_call_llm


def _reset_calls():
    global _LLM_CALLS
    _LLM_CALLS = 0


# ========== L0 顶层：规则路由（零 LLM，μs 级）==========
FAQ = {
    "营业时间": "我们 7×24 小时提供服务。",
    "怎么联系": "可通过工单系统或 support@ 邮箱联系我们。",
    "退款": "退款请在控制台「订单」页发起，3 个工作日内到账。",
}
_BILLING = re.compile(r"(价格|多少钱|计费|收费|包月|按量|费用|报价)")
_TECH = re.compile(r"(报错|错误|为什么|如何排查|连不上|超时|失败|429|500|怎么调)")
_POLICY = re.compile(r"(审核|合规|能不能发|违规|政策|敏感|过审|举报)")


def rule_route(query: str):
    """规则优先。返回 (intent, provider_tuple) 或 ('miss', None)。"""
    for k in FAQ:
        if k in query:
            return "faq", None
    if _BILLING.search(query):
        return "billing", LOCAL  # 单步、结构化 → 本地单 agent
    if _TECH.search(query):
        return "tech", CLOUD  # 多步、有依赖 → 云端 DAG
    if _POLICY.search(query):
        return "policy", CLOUD  # 要改到达标 → 云端 Reflection
    return "miss", None


# ========== L0b 兜底：LLM 路由（仅规则 miss 时触发，#20 思路）==========
_ROUTER_PROMPT = (
    "你是调度器，把问题归到一类，只输出 JSON："
    '{"intent": "<billing|tech|policy|faq>"}。'
    "billing=计费价格, tech=技术报错, policy=合规审核, faq=常见问题。"
)


def llm_route(query: str):
    out = agent_mod.call_llm(
        [
            {"role": "system", "content": _ROUTER_PROMPT},
            {"role": "user", "content": query},
        ],
        model=CLOUD[0],
        provider=CLOUD[1],
    )
    try:
        intent = json.loads(re.search(r"\{.*\}", out, re.S).group())["intent"]
    except Exception:
        intent = "tech"
    prov = {"faq": LOCAL, "billing": LOCAL, "tech": CLOUD, "policy": CLOUD}.get(
        intent, CLOUD
    )
    return intent, prov


# ========== 子 orchestrator dispatch ==========
def dispatch(intent: str, query: str, prov) -> str:
    if intent == "faq":
        for k, v in FAQ.items():
            if k in query:
                return v  # 0 LLM
        return "（FAQ 未命中）"
    model, provider = prov
    if intent == "billing":
        r = agent_mod.run_agent(query, max_iterations=3, model=model, provider=provider)
    elif intent == "tech":
        r = agent_mod.run_plan_execute_agent(
            query, max_iterations=6, model=model, provider=provider
        )
    elif intent == "policy":
        r = agent_mod.run_reflection_agent(
            query, max_retries=1, max_iterations=6, model=model, provider=provider
        )
    else:
        r = agent_mod.run_agent(query, max_iterations=3, model=model, provider=provider)
    return r.get("answer", "")


# ========== 三臂 ==========
def arm_C_hierarchical(query: str, force_local: bool = False) -> dict:
    """分层：规则路由 → miss 才 LLM 兜底 → dispatch。
    force_local=True 时所有子 orchestrator 强制走本地（剥离实验：隔离纯架构功劳）。"""
    _reset_calls()
    start = time.time()
    intent, prov = rule_route(query)
    routed_by = "rule"
    if intent == "miss":
        intent, prov = llm_route(query)  # 这一步本身算 1 次 LLM
        routed_by = "llm"
    if force_local and prov is not None:
        prov = LOCAL  # 同样的路由决策，但底座统一为本地 1.5B
    ans = dispatch(intent, query, prov)
    return {
        "answer": ans,
        "intent": intent,
        "routed_by": routed_by,
        "llm_calls": _LLM_CALLS,
        "latency_ms": round((time.time() - start) * 1000, 1),
        "provider": prov[1] if prov else "none",
    }


def arm_A_single(query: str) -> dict:
    """全程单 agent（最便宜基线），本地。"""
    _reset_calls()
    start = time.time()
    r = agent_mod.run_agent(query, max_iterations=3, model=LOCAL[0], provider=LOCAL[1])
    return {
        "answer": r.get("answer", ""),
        "llm_calls": _LLM_CALLS,
        "latency_ms": round((time.time() - start) * 1000, 1),
    }


def arm_B_reflection(query: str) -> dict:
    """全程 Reflection（最强也最重），本地。"""
    _reset_calls()
    start = time.time()
    r = agent_mod.run_reflection_agent(
        query, max_retries=1, max_iterations=6, model=LOCAL[0], provider=LOCAL[1]
    )
    return {
        "answer": r.get("answer", ""),
        "llm_calls": _LLM_CALLS,
        "latency_ms": round((time.time() - start) * 1000, 1),
    }


# ========== 测试集：5 类意图 + 关键词正确性检查 ==========
# must_any: 答案命中任一关键词即算"答到点上"（粗粒度但三臂一致，可比）
TESTSET = [
    {"q": "你们营业时间是几点？", "intent": "faq", "must_any": ["24", "全天", "小时"]},
    {
        "q": "怎么联系你们客服？",
        "intent": "faq",
        "must_any": ["工单", "邮箱", "support"],
    },
    {
        "q": "A100 按量付费一小时多少钱？",
        "intent": "billing",
        "must_any": ["元", "¥", "价", "小时", "美元"],
    },
    {
        "q": "包月套餐的费用是多少？",
        "intent": "billing",
        "must_any": ["元", "¥", "月", "价"],
    },
    {
        "q": "调用 gateway 报 429 是什么原因，如何排查？",
        "intent": "tech",
        "must_any": ["限流", "请求", "频率", "重试", "429"],
    },
    {
        "q": "模型连不上为什么会超时，怎么定位？",
        "intent": "tech",
        "must_any": ["网络", "超时", "连接", "排查", "日志", "端口"],
    },
    {
        "q": "用户生成的图片要过哪些内容审核才能发？",
        "intent": "policy",
        "must_any": ["审核", "合规", "敏感", "政策", "法律"],
    },
    {
        "q": "这条广告文案能不能直接发，有没有合规风险？",
        "intent": "policy",
        "must_any": ["合规", "风险", "审核", "建议", "法律"],
    },
    {
        "q": "帮我看看这个实例适合做什么场景",
        "intent": "miss",
        "must_any": ["推理", "训练", "场景", "适合", "渲染"],
    },
]


def accuracy(ans: str, must_any) -> int:
    return int(any(k.lower() in (ans or "").lower() for k in must_any))


def run_arm(name, fn):
    print(f"\n=== {name} ===")
    rows, t_lat, t_calls, t_acc = [], 0, 0, 0
    for case in TESTSET:
        try:
            r = fn(case["q"])
        except Exception as e:
            # worker 失败隔离：单条崩溃不拖垮整轮（生产 swarm 铁律），记 0 分继续
            r = {
                "answer": "",
                "llm_calls": _LLM_CALLS,
                "latency_ms": 0.0,
                "error": f"{type(e).__name__}: {e}",
            }
        acc = accuracy(r["answer"], case["must_any"])
        t_lat += r["latency_ms"]
        t_calls += r["llm_calls"]
        t_acc += acc
        rows.append({**case, **r, "correct": acc})
        tag = r.get("intent", "")
        print(
            f"  [{acc}] {case['q'][:20]:<20} calls={r['llm_calls']} "
            f"{r['latency_ms']:.0f}ms {tag}"
        )
    n = len(TESTSET)
    summary = {
        "avg_latency_ms": round(t_lat / n, 1),
        "total_llm_calls": t_calls,
        "accuracy": round(t_acc / n, 3),
    }
    print(
        f"  >>> 准确率 {summary['accuracy']} | 总调用 {t_calls} | 均延迟 {summary['avg_latency_ms']}ms"
    )
    return {"summary": summary, "rows": rows}


def main():
    if not os.getenv("DASHSCOPE_API_KEY"):
        sys.exit("DASHSCOPE_API_KEY 未设置，先 source ~/.zshrc")

    out = {"provider_strategy": "hybrid (simple=local-1.5B, complex=cloud-turbo)"}
    out["A_single_local"] = run_arm("臂A 全程单agent(本地)", arm_A_single)
    out["B_reflection_local"] = run_arm("臂B 全程Reflection(本地)", arm_B_reflection)
    out["C_hierarchical"] = run_arm("臂C 分层Swarm(混合)", arm_C_hierarchical)
    # 剥离实验：同样的分层路由，但所有子 orchestrator 强制本地 → 隔离"纯架构"功劳
    out["D_hierarchical_local"] = run_arm(
        "臂D 分层Swarm(全本地·剥离)", lambda q: arm_C_hierarchical(q, force_local=True)
    )

    # 路由命中率
    rule_hits = sum(
        1 for r in out["C_hierarchical"]["rows"] if r["routed_by"] == "rule"
    )
    out["C_hierarchical"]["rule_hit_rate"] = round(rule_hits / len(TESTSET), 3)

    print("\n========== 四臂对照 ==========")
    print(f"{'臂':<22}{'准确率':<8}{'总调用':<8}{'均延迟ms'}")
    for k, label in [
        ("A_single_local", "A 全程单agent(本地)"),
        ("B_reflection_local", "B 全程Reflection(本地)"),
        ("D_hierarchical_local", "D 分层(全本地)"),
        ("C_hierarchical", "C 分层(混合)"),
    ]:
        s = out[k]["summary"]
        print(
            f"{label:<20}{s['accuracy']:<8}{s['total_llm_calls']:<8}{s['avg_latency_ms']}"
        )
    print(f"C 规则路由命中率: {out['C_hierarchical']['rule_hit_rate']}")
    # 功劳拆解：D-A = 纯架构功劳；C-D = 云端容量功劳
    a = out["A_single_local"]["summary"]["accuracy"]
    d = out["D_hierarchical_local"]["summary"]["accuracy"]
    c = out["C_hierarchical"]["summary"]["accuracy"]
    print(
        f"\n功劳拆解: 纯架构(D-A)=+{round(d - a, 3)} | 云端容量(C-D)=+{round(c - d, 3)} | 合计(C-A)=+{round(c - a, 3)}"
    )

    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果已落盘 {RESULTS}")


if __name__ == "__main__":
    main()
