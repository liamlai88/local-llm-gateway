"""
实验 #12: Reflection 范式
对照 ReAct / Plan-Execute / Plan-Execute+Reflection 三种模式
8 道题 = 4 道 ReAct 容易失败 + 4 道 Plan-Execute 容易失败 (混合任务)

用法:
  python experiments/agent_reflection_demo.py            # 用 qwen-turbo
  python experiments/agent_reflection_demo.py --local    # 用本地 1.5B
"""

import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agent

# 8 道测试题: (question, expected_keywords[必须出现一个])
TESTS = [
    # 简单数学 (ReAct 一般能过)
    ("计算 (15 + 27) * 3", ["126"]),
    # 多工具组合 (ReAct 容易偷懒)
    ("上海气温减去北京气温是多少摄氏度？", ["4"]),
    # RAG + 计算 (ReAct 易跳步)
    ("迪拜的湿度比新加坡低多少个百分点？", ["55"]),
    # 三步链 (典型 Plan-Execute 强项)
    ("杭州气温加上北京气温再乘以 2 等于多少？", ["74"]),
    # 易产生幻觉的 (Reflection 该拦住)
    ("从知识库查 X 产品包月价格并算 6 个月总价", ["6000"]),
    # 单步 (不该过度规划)
    ("3 加 5 等于多少？", ["8"]),
    # 涉及单位提取 (extract_number 工具)
    ("把 '价格 ¥198' 中的数字提出来乘以 10", ["1980"]),
    # 多跳推理 (易遗漏步骤)
    ("新加坡气温减杭州气温的结果再加 5 是多少？", ["13"]),
]


def grade(answer: str, expected: list) -> bool:
    return any(k in answer for k in expected)


def run_one(mode: str, question: str, provider: str, model: str) -> dict:
    t0 = time.time()
    try:
        if mode == "react":
            r = agent.run_agent(
                question, max_iterations=6, model=model, provider=provider
            )
        elif mode == "plan_execute":
            r = agent.run_plan_execute_agent(question, model=model, provider=provider)
        elif mode == "reflection":
            r = agent.run_reflection_agent(
                question, max_retries=1, model=model, provider=provider
            )
        return r
    except Exception as e:
        return {
            "answer": f"ERROR: {e}",
            "latency_ms": (time.time() - t0) * 1000,
            "status": "error",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local", action="store_true", help="用本地 1.5B 替代百炼 Turbo"
    )
    args = parser.parse_args()

    provider = "local" if args.local else "bailian"
    model = "qwen2.5-1.5b" if args.local else "qwen-turbo"

    print(f"\n{'=' * 70}")
    print("实验 #12: Reflection 范式对照")
    print(f"Provider: {provider}, Model: {model}, 测试题: {len(TESTS)}")
    print(f"{'=' * 70}\n")

    results = {"react": [], "plan_execute": [], "reflection": []}

    for i, (q, expected) in enumerate(TESTS, 1):
        print(f"\n--- Q{i}: {q}")
        print(f"    预期含: {expected}")
        for mode in ["react", "plan_execute", "reflection"]:
            r = run_one(mode, q, provider, model)
            ok = grade(r.get("answer", ""), expected)
            results[mode].append(
                {
                    "q": q,
                    "answer": r.get("answer", "")[:120],
                    "ok": ok,
                    "latency_ms": r.get("latency_ms", 0),
                    "iterations": r.get("iterations", 0),
                    "attempts": r.get("attempts", 1),
                }
            )
            mark = "✓" if ok else "✗"
            print(
                f"    [{mode:14s}] {mark}  {r.get('latency_ms', 0):.0f}ms  answer={r.get('answer', '')[:60]}"
            )

    # 汇总
    print(f"\n\n{'=' * 70}")
    print("汇总")
    print(f"{'=' * 70}")
    print(f"{'模式':<20}{'准确率':<12}{'平均延迟(ms)':<16}{'平均 LLM 调用':<14}")
    for mode in ["react", "plan_execute", "reflection"]:
        rs = results[mode]
        acc = sum(1 for r in rs if r["ok"]) / len(rs) * 100
        avg_lat = sum(r["latency_ms"] for r in rs) / len(rs)
        avg_iter = sum(r["iterations"] for r in rs) / len(rs)
        print(f"{mode:<20}{acc:<12.1f}{avg_lat:<16.0f}{avg_iter:<14.1f}")

    # Reflection 专属: 计算"被 Critic 拦住的题目"
    refl_retried = sum(1 for r in results["reflection"] if r.get("attempts", 1) > 1)
    print(f"\nReflection 触发重试的题目: {refl_retried}/{len(TESTS)}")

    # 保存详细结果
    out_path = os.path.join(os.path.dirname(__file__), "reflection_results.json")
    with open(out_path, "w") as f:
        json.dump(
            {"provider": provider, "model": model, "results": results},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n详细结果保存到: {out_path}")


if __name__ == "__main__":
    main()
