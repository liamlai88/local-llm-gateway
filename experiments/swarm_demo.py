"""
实验 #20 Agent Swarm：串行编排 vs 并行 fan-out/gather + LLM handoff 路由

设计要点
- 复用 agent.call_llm（不重复造轮子），每个 agent = system prompt + 一次无状态 LLM 调用
- 并行实验用云端 bailian/qwen-turbo：本地 Ollama 默认 num_parallel=1 会串行处理，
  用它测"并行"会得到假结论（延迟测不出收益）。云端 HTTP 真并发。
- 全程记 trace（agent / 墙钟延迟 / 输出字符数），落盘 swarm_results.json

运行：
  export DASHSCOPE_API_KEY=...   # 已在 ~/.zshrc
  python experiments/swarm_demo.py
"""

import os
import sys
import json
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agent as agent_mod  # noqa: E402

PROVIDER = "bailian"
MODEL = "qwen-turbo"  # 汇总/路由用它；worker 也用它保持可比
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "swarm_results.json")


# ========== 一个 agent = 角色 system prompt + 一次无状态调用 ==========
def run_agent(role_prompt: str, task: str) -> dict:
    """无状态 worker。返回 {output, latency_ms, chars}。状态全靠外部显式传入。"""
    start = time.time()
    out = agent_mod.call_llm(
        [{"role": "system", "content": role_prompt}, {"role": "user", "content": task}],
        model=MODEL,
        provider=PROVIDER,
    )
    return {
        "output": out.strip(),
        "latency_ms": round((time.time() - start) * 1000, 1),
        "chars": len(out),
    }


# ========== 实验 A/B：同一组独立子任务，串行 vs 并行 ==========
# 出海 SLG 手游本地化文案：三语彼此独立 → 天然适合 fan-out
LOCALIZE = [
    ("英文文案", "你是英文游戏营销文案。只输出 1 句不超过 25 词的 slogan，不要解释。"),
    (
        "阿拉伯语文案",
        "你是阿拉伯语游戏营销文案。只输出 1 句阿拉伯语 slogan，不要解释。",
    ),
    ("泰语文案", "你是泰语游戏营销文案。只输出 1 句泰语 slogan，不要解释。"),
]
GAME_BRIEF = "一款出海中东/东南亚的策略手游《Sand Empire》，卖点：真实沙漠战争、跨服联盟、零氪也能上分。"


def serial_run() -> dict:
    """串行编排：worker 一个接一个。"""
    start = time.time()
    steps = []
    for name, prompt in LOCALIZE:
        r = run_agent(prompt, GAME_BRIEF)
        steps.append({"agent": name, **r})
    return {
        "mode": "serial",
        "wall_ms": round((time.time() - start) * 1000, 1),
        "steps": steps,
    }


def swarm_run() -> dict:
    """并行 swarm：fan-out 同时发，gather 阶段再用一次 LLM 综合。"""
    start = time.time()
    with ThreadPoolExecutor(max_workers=len(LOCALIZE)) as pool:
        futures = {
            pool.submit(run_agent, prompt, GAME_BRIEF): name
            for name, prompt in LOCALIZE
        }
        steps = []
        for fut in futures:
            name = futures[fut]
            steps.append({"agent": name, **fut.result()})
    fan_out_ms = round((time.time() - start) * 1000, 1)

    # gather：把三份结果交给 finalizer 合成一段统一稿
    joined = "\n".join(f"[{s['agent']}] {s['output']}" for s in steps)
    g = run_agent(
        "你是营销主编，把多语 slogan 整理成一份对外发布清单，每行一种语言。", joined
    )
    return {
        "mode": "swarm_parallel",
        "wall_ms": round((time.time() - start) * 1000, 1),
        "fan_out_ms": fan_out_ms,
        "gather_ms": g["latency_ms"],
        "steps": steps,
        "final": g["output"],
    }


# ========== 实验 C：LLM handoff 路由（动态分发给专家）==========
ROUTER_PROMPT = (
    "你是调度器。把用户问题分给最合适的一个专家，只输出 JSON："
    '{"agent": "<billing|tech|policy>", "reason": "<10字内>"}。'
    "billing=计费/价格，tech=技术/报错，policy=合规/审核政策。"
)
SPECIALISTS = {
    "billing": "你是计费专家，一句话回答价格相关问题。",
    "tech": "你是技术支持，一句话给出排查方向。",
    "policy": "你是合规专家，一句话回答内容审核政策。",
}
ROUTE_QUERIES = [
    "A100 按量付费一小时多少钱？",
    "调用 gateway 报 429 是什么原因？",
    "中东地区用户生成内容要过哪些审核？",
]


def route_one(query: str) -> dict:
    r = run_agent(ROUTER_PROMPT, query)
    try:
        import re

        d = json.loads(re.search(r"\{.*\}", r["output"], re.S).group())
        agent = d.get("agent", "tech")
        reason = d.get("reason", "")
    except Exception:
        agent, reason = "tech", "(解析失败兜底)"
    if agent not in SPECIALISTS:
        agent = "tech"
    ans = run_agent(SPECIALISTS[agent], query)
    return {
        "query": query,
        "routed_to": agent,
        "reason": reason,
        "answer": ans["output"],
        "route_ms": r["latency_ms"],
        "answer_ms": ans["latency_ms"],
    }


def main():
    if not os.getenv("DASHSCOPE_API_KEY"):
        sys.exit("DASHSCOPE_API_KEY 未设置，先 source ~/.zshrc")

    print("=== 实验 A：串行编排 ===")
    serial = serial_run()
    print(f"  墙钟 {serial['wall_ms']} ms")

    print("=== 实验 B：并行 swarm (fan-out/gather) ===")
    swarm = swarm_run()
    print(
        f"  墙钟 {swarm['wall_ms']} ms（fan-out {swarm['fan_out_ms']} + gather {swarm['gather_ms']}）"
    )

    serial_workers = sum(s["latency_ms"] for s in serial["steps"])
    speedup = round(serial["wall_ms"] / swarm["fan_out_ms"], 2)
    print(
        f"  fan-out 加速比 ≈ {speedup}x（串行三 worker 累计 {round(serial_workers, 1)} ms）"
    )

    print("=== 实验 C：LLM handoff 路由 ===")
    routes = [route_one(q) for q in ROUTE_QUERIES]
    for r in routes:
        print(f"  {r['query'][:18]}… → {r['routed_to']} ({r['reason']})")

    out = {
        "provider": PROVIDER,
        "model": MODEL,
        "A_serial": serial,
        "B_swarm": swarm,
        "fanout_speedup": speedup,
        "C_routing": routes,
    }
    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果已落盘 {RESULTS}")


if __name__ == "__main__":
    main()
