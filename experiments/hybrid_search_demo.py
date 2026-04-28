"""
Hybrid Search 对比实验
设计精巧的文档+问题，让 Vector / BM25 / Hybrid 三种模式各自暴露强弱
"""
import requests

GATEWAY = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer sk-demo-002", "Content-Type": "application/json"}


# ========== 文档库：包含专有名词、编号、近义词 ==========
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


def query(question, mode):
    """只检索不调LLM，对比检索质量"""
    r = requests.post(
        f"{GATEWAY}/v1/rag/query",
        headers=HEADERS,
        json={"question": question, "mode": mode, "top_k": 2, "model": "fast"},
    )
    return r.json()


# ========== 精心设计的 4 道题 ==========
TESTS = [
    {
        "question": "A100 实例多少钱一小时？",
        "expected_doc": "product_b",
        "category": "🔢 编号精确匹配（BM25 优势）",
    },
    {
        "question": "我想做大模型推理，用哪个产品比较合适？",
        "expected_doc": "product_b",
        "category": "🧠 语义相关（Vector 优势）",
    },
    {
        "question": "ecs.gn7i-c8g1.2xlarge 是什么型号？",
        "expected_doc": "product_a",
        "category": "🔣 长串编号（BM25 强）",
    },
    {
        "question": "训练超大规模模型应该走什么流程？",
        "expected_doc": "policy",
        "category": "🌐 概念性问法（Vector + BM25 都需要）",
    },
]


if __name__ == "__main__":
    upload_all()

    summary = {"vector": 0, "bm25": 0, "hybrid": 0}

    for t in TESTS:
        print("\n" + "=" * 75)
        print(f"❓ {t['question']}")
        print(f"   类别: {t['category']}")
        print(f"   期望命中文档: {t['expected_doc']}")
        print("=" * 75)

        for mode in ["vector", "bm25", "hybrid"]:
            res = query(t["question"], mode)
            top1 = res["sources"][0] if res["sources"] else None
            top1_doc = top1["metadata"]["doc_id"] if top1 else "无"
            hit = "✅" if top1_doc == t["expected_doc"] else "❌"
            if top1_doc == t["expected_doc"]:
                summary[mode] += 1
            print(f"\n  [{mode.upper():6}] Top-1: {top1_doc} {hit}  "
                  f"延迟 {res['stats']['retrieval_ms']}ms  "
                  f"score={top1['score']:.4f}" if top1 else "")
            if res["sources"]:
                print(f"           内容片段: {res['sources'][0]['content'][:60]}...")

    # 总分
    print("\n" + "=" * 75)
    print("📊 检索准确率统计 (Top-1 命中)")
    print("=" * 75)
    for mode, score in summary.items():
        print(f"  {mode:8}: {score}/{len(TESTS)} = {score / len(TESTS) * 100:.0f}%")
