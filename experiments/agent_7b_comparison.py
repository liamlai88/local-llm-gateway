"""
补全模型规模 × Plan-Execute 准确率曲线
测试本地 7B vs 1.5B vs 百炼 Turbo, 同 4 题, 同范式 (Plan-Execute)
"""
import requests
import json

GATEWAY = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer sk-demo-002", "Content-Type": "application/json"}


def run_agent(question, model, provider, mode="plan_execute"):
    return requests.post(
        f"{GATEWAY}/v1/agent/run",
        headers=HEADERS,
        json={"question": question, "mode": mode, "provider": provider,
              "model": model, "max_iterations": 8},
        timeout=240,
    ).json()


def evaluate(result, expected_keywords, must_use_tools, any_kw=False):
    if result.get("status") not in ("success",):
        return False, f"状态: {result.get('status')}"
    answer = result.get("answer", "")
    kw_pass = any(kw in answer for kw in expected_keywords) if any_kw else \
              all(kw in answer for kw in expected_keywords)
    if not kw_pass:
        return False, f"答案缺少 {expected_keywords}"
    used = [s.get("tool") for s in result.get("trace", []) if s["type"] in ("action", "execute")]
    for must in must_use_tools:
        if must not in used:
            return False, f"未调用 {must}"
    return True, "✅"


TESTS = [
    {"id": "Q1", "question": "15 乘以 0.6 等于多少？",
     "expected": ["9"], "must": ["calculator"]},
    {"id": "Q2", "question": "杭州现在天气怎么样？",
     "expected": ["22"], "must": ["get_weather"]},
    {"id": "Q3", "question": "杭州现在的气温减去北京气温是多少度？",
     "expected": ["7"], "must": ["get_weather", "calculator"]},
    {"id": "Q4", "question": "阿里云 A100 实例按量付费一天大概多少钱？请基于知识库数据计算（按 24 小时算）。",
     "expected": ["1632"], "must": ["kb_search", "calculator"]},
]

# 两组配置（本地 1.5B vs 7B）
CONFIGS = [
    ("Plan-Execute 1.5B (本地)", "qwen2.5-1.5b",  "local"),
    ("Plan-Execute 7B (本地)",   "qwen2.5-7b",    "local"),
]


if __name__ == "__main__":
    summary = {name: [] for name, _, _ in CONFIGS}

    for t in TESTS:
        print("\n" + "=" * 80)
        print(f"❓ [{t['id']}] {t['question']}")
        print(f"   期望: {t['expected']}  必用: {t['must']}")
        print("=" * 80)

        for name, model, provider in CONFIGS:
            try:
                result = run_agent(t["question"], model, provider)
                ok, reason = evaluate(result, t["expected"], t["must"])
                lat = result.get("latency_ms", 0)
                tools = [s.get("tool") for s in result.get("trace", [])
                         if s["type"] in ("action", "execute")]
                print(f"\n  [{name:30}] {'✅' if ok else '❌'} {reason}")
                print(f"     答: {result.get('answer','')[:80]}")
                print(f"     {result.get('iterations',0)} 步 / {lat:.0f}ms")
                if tools:
                    print(f"     工具链: {' → '.join(tools)}")
                summary[name].append({"ok": ok, "lat": lat})
            except Exception as e:
                print(f"\n  [{name}] ❌ 错误: {e}")
                summary[name].append({"ok": False, "lat": 0})

    # 总结
    print("\n" + "=" * 80)
    print("📊 模型规模 × Plan-Execute 准确率曲线")
    print("=" * 80)
    print(f"\n{'配置':<35} {'准确率':<15} {'平均延迟':<12}")
    print("-" * 65)
    for name, results in summary.items():
        if not results:
            continue
        ok_count = sum(1 for r in results if r["ok"])
        avg_lat = sum(r["lat"] for r in results) / len(results)
        rate = f"{ok_count}/{len(TESTS)} ({ok_count/len(TESTS)*100:.0f}%)"
        print(f"{name:<35} {rate:<15} {avg_lat:>8.0f}ms")
