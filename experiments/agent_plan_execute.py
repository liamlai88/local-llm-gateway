"""
Plan-and-Execute vs ReAct 对比实验
4 道题 × 2 模式 × 2 模型 = 16 次实验
重点看：Plan-Execute 能否突破 ReAct 在 Q3/Q4 上的天花板
"""
import requests
import json

GATEWAY = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer sk-demo-002", "Content-Type": "application/json"}


def upload_kb():
    print("=== 准备知识库 ===")
    requests.delete(f"{GATEWAY}/v1/rag/documents", headers=HEADERS)
    docs = {
        "product_b": """
        产品名称: 阿里云 GPU 计算实例 ecs.gn7e-c12g1.3xlarge
        GPU 型号: NVIDIA A100
        显存: 80GB HBM2e
        定价: 按量付费 ¥68/小时，包月 ¥35000
        """,
        "product_a": """
        产品名称: 阿里云 GPU 计算实例 ecs.gn7i-c8g1.2xlarge
        GPU 型号: NVIDIA A10
        显存: 24GB GDDR6
        定价: 按量付费 ¥18/小时，包月 ¥9800
        """,
    }
    for doc_id, content in docs.items():
        requests.post(f"{GATEWAY}/v1/rag/documents", headers=HEADERS,
                      json={"doc_id": doc_id, "content": content})
    print("✓ 知识库准备完成\n")


def run_agent(question, **kwargs):
    return requests.post(
        f"{GATEWAY}/v1/agent/run",
        headers=HEADERS,
        json={"question": question, "max_iterations": 8, **kwargs},
        timeout=240,
    ).json()


def evaluate(result, expected_keywords, must_use_tools):
    if result.get("status") not in ("success",):
        return False, f"状态: {result.get('status')}"
    answer = result.get("answer", "")
    if not all(kw in answer for kw in expected_keywords):
        return False, f"答案缺少 {expected_keywords}"

    used = []
    for s in result.get("trace", []):
        if s["type"] in ("action", "execute"):
            used.append(s.get("tool"))
    for must in must_use_tools:
        if must not in used:
            return False, f"未调用 {must} (实际 {used})"
    return True, "✅ 真实成功"


TESTS = [
    {"id": "Q1", "question": "15 乘以 0.6 等于多少？",
     "expected": ["9"], "must_use": ["calculator"]},
    {"id": "Q2", "question": "杭州现在天气怎么样？",
     "expected": ["22"], "must_use": ["get_weather"]},
    {"id": "Q3", "question": "杭州现在的气温减去北京气温是多少度？",
     "expected": ["7"], "must_use": ["get_weather", "calculator"]},
    {"id": "Q4", "question": "阿里云 A100 实例按量付费一天大概多少钱？请基于知识库数据计算（按 24 小时算）。",
     "expected": ["1632"], "must_use": ["kb_search", "calculator"]},
]

# 4 组对照
CONFIGS = [
    ("ReAct 1.5B",          {"mode": "react",        "provider": "local",   "model": "fast",       "few_shot": True}),
    ("ReAct Turbo",         {"mode": "react",        "provider": "bailian", "model": "qwen-turbo", "few_shot": True}),
    ("Plan-Execute 1.5B",   {"mode": "plan_execute", "provider": "local",   "model": "fast"}),
    ("Plan-Execute Turbo",  {"mode": "plan_execute", "provider": "bailian", "model": "qwen-turbo"}),
]


if __name__ == "__main__":
    upload_kb()

    summary = {name: [] for name, _ in CONFIGS}

    for t in TESTS:
        print("\n" + "=" * 80)
        print(f"❓ [{t['id']}] {t['question']}")
        print(f"   期望: {t['expected']}  必用: {t['must_use']}")
        print("=" * 80)

        for name, params in CONFIGS:
            result = run_agent(t["question"], **params)
            success, reason = evaluate(result, t["expected"], t["must_use"])
            summary[name].append({"id": t["id"], "success": success, "reason": reason,
                                  "latency_ms": result.get("latency_ms")})
            mark = "✅" if success else "❌"
            print(f"\n  [{name:22}] {mark} {reason}")
            print(f"     答案: {result.get('answer', '')[:80]}")
            print(f"     {result.get('iterations', 0)} 步 / {result.get('latency_ms', 0)}ms")
            tool_calls = [s.get("tool") for s in result.get("trace", []) if s["type"] in ("action", "execute")]
            if tool_calls:
                print(f"     工具链: {' → '.join(tool_calls)}")

    print("\n" + "=" * 80)
    print("📊 4 组对比统计")
    print("=" * 80)
    for name, results in summary.items():
        succ = sum(1 for r in results if r["success"])
        avg_lat = sum(r["latency_ms"] for r in results) / len(results)
        print(f"\n  {name:22}: {succ}/{len(TESTS)} = {succ/len(TESTS)*100:3.0f}%   avg {avg_lat:.0f}ms")
        for r in results:
            mark = "✅" if r["success"] else "❌"
            print(f"     {r['id']}: {mark} {r['reason']}")
