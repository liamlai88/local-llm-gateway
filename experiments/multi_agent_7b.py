"""
Multi-Agent 混合架构在不同模型上的表现
验证假设: 混合架构对模型规模不敏感 (因为 Critic + 规则路径承担了大部分工作)

3 组配置:
  A: Multi-Agent + 1.5B 兜底 (本地)
  B: Multi-Agent + 7B 兜底  (本地)
  C: Multi-Agent + Turbo 兜底 (百炼, 已知 4/4)

预期: 三组都接近 100%, 因为 Critic 兜底机制和模型大小关系不大
"""
import requests

GATEWAY = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer sk-demo-002", "Content-Type": "application/json"}


def upload_kb():
    docs = {
        "product_b": """产品名称: 阿里云 ecs.gn7e-c12g1.3xlarge
GPU: NVIDIA A100, 显存 80GB
定价: 按量付费 ¥68/小时, 包月 ¥35000
适用场景: 大模型训练、推理""",
        "product_a": """产品名称: 阿里云 ecs.gn7i-c8g1.2xlarge
GPU: NVIDIA A10, 显存 24GB
定价: 按量付费 ¥18/小时, 包月 ¥9800
适用场景: AI 推理、轻量微调""",
    }
    requests.delete(f"{GATEWAY}/v1/rag/documents", headers=HEADERS, timeout=30)
    for doc_id, content in docs.items():
        requests.post(f"{GATEWAY}/v1/rag/documents", headers=HEADERS,
                      json={"doc_id": doc_id, "content": content}, timeout=60)


def run(question, model, provider, force_caller_model=True):
    """force_caller_model=True 让 fallback 使用我们指定的 model"""
    return requests.post(
        f"{GATEWAY}/v1/multi-agent/run",
        headers=HEADERS,
        json={
            "question": question,
            "model": model,
            "provider": provider,
            "llm_final": False,
            "enable_fallback": True,
            "force_caller_model": force_caller_model,
        },
        timeout=240,
    ).json()


TESTS = [
    {"id": "命中-1", "q": "杭州和北京的气温差是多少？", "kw": ["7"]},
    {"id": "命中-2", "q": "阿里云 A100 按量付费一天多少钱？", "kw": ["1632"]},
    {"id": "盲区-1", "q": "我想训练 70B 大模型，预算 100 万，给我推荐用什么 GPU？",
     "kw": ["A100", "H100"], "any_of": True},
    {"id": "盲区-2", "q": "我们公司刚开始做 AI 推理服务，每天 10 万次请求，应该怎么选 GPU？",
     "kw": ["A10", "推理"], "any_of": True},
]

CONFIGS = [
    ("Multi-Agent + 1.5B 兜底", "qwen2.5-1.5b", "local", True),
    ("Multi-Agent + 7B 兜底",   "qwen2.5-7b",   "local", True),
    # 注: Turbo 因 API key 不在 force_caller_model=False 时会先尝试 turbo 默认硬编码
    # 这里不测 turbo, 之前已知 4/4
]


def evaluate(answer, kw_list, any_of=False):
    if any_of:
        return any(kw in answer for kw in kw_list)
    return all(kw in answer for kw in kw_list)


if __name__ == "__main__":
    print("=== 准备知识库 ===")
    upload_kb()
    print("✓ KB ready\n")

    summary = {name: [] for name, _, _, _ in CONFIGS}

    for t in TESTS:
        print("=" * 80)
        print(f"[{t['id']}] {t['q']}")
        print("=" * 80)

        for name, model, provider, force in CONFIGS:
            result = run(t["q"], model, provider, force)
            ans = result.get("answer", "")
            ok = evaluate(ans, t["kw"], t.get("any_of", False))
            lat = result.get("latency_ms", 0)
            path = result.get("path", "?")
            print(f"\n  [{name:30}] {'✅' if ok else '❌'}  {lat:.0f}ms ({path})")
            print(f"     答: {ans[:120]}")
            summary[name].append({"id": t["id"], "ok": ok, "lat": lat, "path": path})

    print("\n" + "=" * 80)
    print("📊 混合架构 × 模型规模 准确率")
    print("=" * 80)
    print(f"\n{'配置':<32} {'准确率':<15} {'平均延迟':<12} {'快/慢':<10}")
    print("-" * 70)
    for name, results in summary.items():
        ok_count = sum(1 for r in results if r["ok"])
        avg_lat = sum(r["lat"] for r in results) / len(results)
        fast_count = sum(1 for r in results if r["path"] == "fast")
        slow_count = len(results) - fast_count
        rate = f"{ok_count}/{len(TESTS)} ({ok_count/len(TESTS)*100:.0f}%)"
        path_summary = f"{fast_count}快/{slow_count}慢"
        print(f"{name:<32} {rate:<15} {avg_lat:>8.0f}ms  {path_summary}")
    print()
    print("已知对照: Multi-Agent + Turbo = 4/4 (100%), 平均 1.2s")
