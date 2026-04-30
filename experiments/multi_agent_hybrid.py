"""
Multi-Agent 混合架构对比实验

对比 3 种配置:
  A: Plan-Execute Turbo (基线)
  B: Multi-Agent 纯规则路径 (enable_fallback=False)
  C: Multi-Agent 混合架构 (enable_fallback=True) ⭐

测试题分两类:
  - 命中区: 规则路由能搞定的（气温差/价格×时长）
  - 盲区:   规则识别不到的开放性问题（70B 推荐/选型对比）
"""
import json
import requests
import time

GATEWAY = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer sk-demo-002", "Content-Type": "application/json"}


def upload_kb():
    docs = {
        "product_a100": """
        产品名称: 阿里云 GPU 计算实例 ecs.gn7e-c12g1.3xlarge
        GPU 型号: NVIDIA A100
        显存: 80GB HBM2e
        定价: 按量付费 ¥68/小时，包月 ¥35000
        适用场景: 大模型训练、微调、高吞吐推理
        """,
        "product_a10": """
        产品名称: 阿里云 GPU 计算实例 ecs.gn7i-c8g1.2xlarge
        GPU 型号: NVIDIA A10
        显存: 24GB GDDR6
        定价: 按量付费 ¥18/小时，包月 ¥9800
        适用场景: AI 推理、轻量微调、图形渲染
        """,
        "product_h100": """
        产品名称: 阿里云灵骏 H100 集群
        GPU 型号: NVIDIA H100
        显存: 80GB HBM3
        定价: 项目制，需联系销售
        适用场景: 万亿参数大模型训练、万卡级分布式训练
        """,
    }
    requests.delete(f"{GATEWAY}/v1/rag/documents", headers=HEADERS, timeout=30)
    for doc_id, content in docs.items():
        requests.post(f"{GATEWAY}/v1/rag/documents", headers=HEADERS,
                      json={"doc_id": doc_id, "content": content}, timeout=60)


def run_plan_execute(question):
    r = requests.post(
        f"{GATEWAY}/v1/agent/run",
        headers=HEADERS,
        json={"question": question, "mode": "plan_execute", "provider": "bailian", "model": "qwen-turbo"},
        timeout=180,
    )
    return r.json()


def run_multi_agent(question, enable_fallback):
    r = requests.post(
        f"{GATEWAY}/v1/multi-agent/run",
        headers=HEADERS,
        json={
            "question": question, "model": "fast", "provider": "bailian",
            "llm_final": False, "enable_fallback": enable_fallback,
        },
        timeout=180,
    )
    return r.json()


TESTS = [
    {
        "id": "命中-1", "category": "🎯 命中区: 单工具",
        "question": "杭州和北京的气温差是多少？",
        "must_have": ["7"],
    },
    {
        "id": "命中-2", "category": "🎯 命中区: RAG+计算",
        "question": "阿里云 A100 按量付费一天大概多少钱？",
        "must_have": ["1632"],
    },
    {
        "id": "盲区-1", "category": "🌫 盲区: 开放选型",
        "question": "我想训练 70B 大模型，预算 100 万，给我推荐用什么 GPU？",
        "must_have": ["A100", "H100"],  # 至少提到一个产品名
        "any_of": True,
    },
    {
        "id": "盲区-2", "category": "🌫 盲区: 业务咨询",
        "question": "我们公司刚开始做 AI 推理服务，每天 10 万次请求，应该怎么选 GPU？",
        "must_have": ["A10", "推理"],
        "any_of": True,
    },
]


def evaluate(answer, must_have, any_of=False):
    if any_of:
        return any(kw in answer for kw in must_have)
    return all(kw in answer for kw in must_have)


def main():
    print("=== 准备知识库 ===")
    upload_kb()
    print("✓ 准备完成\n")

    summary = {
        "Plan-Execute Turbo":         [],
        "Multi-Agent 纯规则":          [],
        "Multi-Agent + LLM Fallback": [],
    }

    for t in TESTS:
        print("=" * 80)
        print(f"[{t['id']}] {t['category']}")
        print(f"Q: {t['question']}")
        print(f"期望含: {t['must_have']} ({'任一' if t.get('any_of') else '全部'})")
        print("=" * 80)

        # Plan-Execute Turbo
        try:
            r1 = run_plan_execute(t["question"])
            ans1 = r1.get("answer", "")
            ok1 = evaluate(ans1, t["must_have"], t.get("any_of", False))
            lat1 = r1.get("latency_ms", 0)
            print(f"\n[Plan-Execute Turbo]   {'✅' if ok1 else '❌'}  {lat1:.0f}ms")
            print(f"  答: {ans1[:120]}")
        except Exception as e:
            ok1, lat1 = False, 0
            print(f"\n[Plan-Execute Turbo]   ❌ 错误: {e}")
        summary["Plan-Execute Turbo"].append({"ok": ok1, "lat": lat1})

        # Multi-Agent 纯规则
        r2 = run_multi_agent(t["question"], enable_fallback=False)
        ans2 = r2.get("answer", "")
        ok2 = evaluate(ans2, t["must_have"], t.get("any_of", False))
        lat2 = r2.get("latency_ms", 0)
        path2 = r2.get("path", "?")
        print(f"\n[Multi-Agent 纯规则]   {'✅' if ok2 else '❌'}  {lat2:.1f}ms ({path2})")
        print(f"  答: {ans2[:120]}")
        summary["Multi-Agent 纯规则"].append({"ok": ok2, "lat": lat2})

        # Multi-Agent + LLM Fallback
        r3 = run_multi_agent(t["question"], enable_fallback=True)
        ans3 = r3.get("answer", "")
        ok3 = evaluate(ans3, t["must_have"], t.get("any_of", False))
        lat3 = r3.get("latency_ms", 0)
        path3 = r3.get("path", "?")
        print(f"\n[Multi-Agent+Fallback] {'✅' if ok3 else '❌'}  {lat3:.1f}ms ({path3}) ⭐")
        print(f"  答: {ans3[:200]}")
        summary["Multi-Agent + LLM Fallback"].append({"ok": ok3, "lat": lat3})

        print()

    # 汇总
    n = len(TESTS)
    print("=" * 80)
    print("📊 三种架构对比")
    print("=" * 80)
    print(f"\n{'架构':<35} {'准确率':<15} {'平均延迟':<15}")
    print("-" * 70)
    for name, results in summary.items():
        succ = sum(1 for r in results if r["ok"])
        avg_lat = sum(r["lat"] for r in results) / len(results) if results else 0
        rate = f"{succ}/{n} ({succ/n*100:.0f}%)"
        print(f"{name:<35} {rate:<15} {avg_lat:>10.1f}ms")


if __name__ == "__main__":
    main()
