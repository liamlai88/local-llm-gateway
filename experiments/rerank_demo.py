"""
Rerank 实验：对比 4 种检索模式
- vector / bm25 / hybrid (上次的 50% 基线)
- hybrid + rerank (今天的目标：突破 50%)
"""
import requests

GATEWAY = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer sk-demo-002", "Content-Type": "application/json"}


# 复用上一个实验的文档（确保对比公平）
DOCS = {
    "product_a": """
    产品名称: 阿里云 GPU 计算实例 ecs.gn7i-c8g1.2xlarge
    GPU 型号: NVIDIA A10
    显存: 24GB GDDR6
    适用场景: 中小规模 AI 推理、视频渲染、图形工作站
    定价: 按量付费 ¥18/小时，包月 ¥9800
    上市时间: 2024 年 3 月
    """,
    "product_b": """
    产品名称: 阿里云 GPU 计算实例 ecs.gn7e-c12g1.3xlarge
    GPU 型号: NVIDIA A100
    显存: 80GB HBM2e
    适用场景: 大模型训练、千亿参数推理、HPC 高性能计算
    定价: 按量付费 ¥68/小时，包月 ¥35000
    上市时间: 2023 年 8 月
    """,
    "product_c": """
    产品名称: 阿里云灵骏智算集群 PAI-Lingjun
    GPU 型号: NVIDIA H100 集群
    显存: 单卡 80GB HBM3，集群规模 64 卡起
    适用场景: 万亿参数大模型训练、万卡级分布式训练
    定价: 项目制，需联系销售
    上市时间: 2024 年 11 月
    """,
    "policy": """
    阿里云 GPU 实例使用规范:
    - 中小客户开发测试可选用入门级 GPU 实例（如 A10 系列）
    - 生产级 AI 推理建议使用 A100 实例，享受全天候 SLA 保障
    - 超大规模训练场景必须使用灵骏智算集群，单笔订单需走 SA 申请流程
    - 所有 GPU 实例都支持企业级镜像和统一身份认证
    """,
}


def upload_all():
    print("=== 清空知识库 ===")
    requests.delete(f"{GATEWAY}/v1/rag/documents", headers=HEADERS)
    print("=== 上传 4 份文档 ===")
    for doc_id, content in DOCS.items():
        r = requests.post(f"{GATEWAY}/v1/rag/documents", headers=HEADERS,
                          json={"doc_id": doc_id, "content": content})
        print(f"  {doc_id}: {r.json()}")


def query(question, mode, rerank=False):
    r = requests.post(
        f"{GATEWAY}/v1/rag/query",
        headers=HEADERS,
        json={
            "question": question,
            "mode": mode,
            "rerank": rerank,
            "top_k": 2,
            "model": "fast",
        },
    )
    return r.json()


TESTS = [
    {"q": "A100 实例多少钱一小时？",            "exp": "product_b", "cat": "🔢 短英数字串"},
    {"q": "我想做大模型推理，用哪个产品比较合适？", "exp": "product_b", "cat": "🧠 语义匹配"},
    {"q": "ecs.gn7i-c8g1.2xlarge 是什么型号？",   "exp": "product_a", "cat": "🔣 长 ID 串"},
    {"q": "训练超大规模模型应该走什么流程？",      "exp": "policy",    "cat": "🌐 概念性"},
]

# 4 种对照模式
MODES = [
    ("vector",         {"mode": "vector", "rerank": False}),
    ("bm25",           {"mode": "bm25",   "rerank": False}),
    ("hybrid",         {"mode": "hybrid", "rerank": False}),
    ("hybrid+rerank",  {"mode": "hybrid", "rerank": True}),  # ⭐ 新增
]


if __name__ == "__main__":
    upload_all()

    summary = {name: 0 for name, _ in MODES}
    latency = {name: [] for name, _ in MODES}

    for t in TESTS:
        print("\n" + "=" * 75)
        print(f"❓ {t['q']}")
        print(f"   类别: {t['cat']}  期望: {t['exp']}")
        print("=" * 75)

        for name, params in MODES:
            res = query(t["q"], **params)
            top1 = res["sources"][0] if res["sources"] else None
            top1_doc = top1["metadata"]["doc_id"] if top1 else "无"
            hit = "✅" if top1_doc == t["exp"] else "❌"
            if top1_doc == t["exp"]:
                summary[name] += 1
            latency[name].append(res["stats"]["retrieval_ms"])

            score_label = "rerank_score" if "rerank" in name else "score"
            score_val = top1.get("rerank_score" if "rerank" in name else "score", 0) if top1 else 0
            print(f"  [{name:14}] Top-1: {top1_doc:20} {hit}  "
                  f"延迟 {res['stats']['retrieval_ms']:6.1f}ms  "
                  f"{score_label}={score_val:.4f}")

    print("\n" + "=" * 75)
    print("📊 准确率统计 (Top-1 命中)")
    print("=" * 75)
    for name in summary:
        avg_lat = sum(latency[name]) / len(latency[name])
        print(f"  {name:14}: {summary[name]}/{len(TESTS)} = {summary[name]/len(TESTS)*100:3.0f}%   "
              f"平均延迟 {avg_lat:6.1f}ms")
