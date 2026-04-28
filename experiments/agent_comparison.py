"""
Agent ReAct 三组对比实验
1. 1.5B + 无 Few-shot (基线)
2. 1.5B + Few-shot
3. Qwen-Turbo (百炼，约 7B)
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
        json={"question": question, "max_iterations": 6, **kwargs},
        timeout=180,
    ).json()


def evaluate(result, expected_keywords, must_use_tools):
    """
    评判真实成功：
    - 答案包含期望关键词
    - 必须用过指定工具（防止跳过工具的假成功）
    """
    if result.get("status") != "success":
        return False, "状态非 success"
    answer = result.get("answer", "")
    if not all(kw in answer for kw in expected_keywords):
        return False, f"答案缺少关键词 {expected_keywords}"
    used_tools = [s.get("tool") for s in result.get("trace", []) if s["type"] == "action"]
    for must in must_use_tools:
        if must not in used_tools:
            return False, f"未调用必需工具 {must} (实际用了 {used_tools})"
    return True, "✅ 真实成功"


TESTS = [
    {
        "id": "Q1", "question": "15 乘以 0.6 等于多少？",
        "expected": ["9"], "must_use": ["calculator"],
    },
    {
        "id": "Q2", "question": "杭州现在天气怎么样？",
        "expected": ["22"], "must_use": ["get_weather"],
    },
    {
        "id": "Q3", "question": "杭州现在的气温减去北京气温是多少度？",
        "expected": ["7"], "must_use": ["get_weather", "calculator"],
    },
    {
        "id": "Q4", "question": "阿里云 A100 实例按量付费一天大概多少钱？请基于知识库数据计算（按 24 小时算）。",
        "expected": ["1632"], "must_use": ["kb_search", "calculator"],
    },
]

# 三组配置
CONFIGS = [
    ("1.5B 无 Few-shot",  {"provider": "local",   "model": "fast",        "few_shot": False}),
    ("1.5B + Few-shot",   {"provider": "local",   "model": "fast",        "few_shot": True}),
    ("Qwen-Turbo (百炼)", {"provider": "bailian", "model": "qwen-turbo",  "few_shot": True}),
]


if __name__ == "__main__":
    upload_kb()

    summary = {name: [] for name, _ in CONFIGS}

    for t in TESTS:
        print("\n" + "=" * 80)
        print(f"❓ [{t['id']}] {t['question']}")
        print(f"   期望关键词: {t['expected']}  必用工具: {t['must_use']}")
        print("=" * 80)

        for name, params in CONFIGS:
            result = run_agent(t["question"], **params)
            success, reason = evaluate(result, t["expected"], t["must_use"])
            summary[name].append({
                "id": t["id"],
                "success": success,
                "reason": reason,
                "iterations": result.get("iterations"),
                "latency_ms": result.get("latency_ms"),
            })
            mark = "✅" if success else "❌"
            print(f"\n  [{name:18}] {mark} {reason}")
            print(f"     答案: {result.get('answer', '')[:80]}")
            print(f"     循环 {result.get('iterations')} 轮 / {result.get('latency_ms')}ms")
            tool_calls = [f"{s['tool']}({list(s['args'].keys())})"
                          for s in result.get("trace", []) if s["type"] == "action"]
            if tool_calls:
                print(f"     工具调用: {' → '.join(tool_calls)}")

    # 汇总
    print("\n" + "=" * 80)
    print("📊 三组对比 - 真实成功率统计")
    print("=" * 80)
    for name, results in summary.items():
        succ = sum(1 for r in results if r["success"])
        avg_lat = sum(r["latency_ms"] for r in results) / len(results)
        avg_iter = sum(r["iterations"] for r in results) / len(results)
        print(f"\n  {name:18}: {succ}/{len(TESTS)} = {succ/len(TESTS)*100:3.0f}%   "
              f"平均 {avg_iter:.1f} 轮 / {avg_lat:.0f}ms")
        for r in results:
            mark = "✅" if r["success"] else "❌"
            print(f"     {r['id']}: {mark} {r['reason']}")
