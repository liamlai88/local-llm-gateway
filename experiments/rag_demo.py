"""
RAG 实验：纯 LLM vs RAG 增强
场景：让小模型回答一个它"不可能知道"的事情
"""
import requests

GATEWAY = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer sk-demo-002", "Content-Type": "application/json"}

# ========== 知识库内容（一份虚构的内部文档）==========
INTERNAL_DOC = """
# 阿里云海外 GenAI SA 团队工作手册 (2026版)

## 团队简介
阿里云海外 GenAI SA 团队成立于 2025 年 3 月，专注海外 AI MaaS 业务拓展。
团队总监是张三，团队当前共有 12 位 Solution Architect，分布在新加坡、迪拜和伦敦三地。

## 重点客户场景
2026 年团队聚焦 4 大场景：海外社交平台内容审核、AI 视频内容分析、AI 数据标注服务平台、模型即服务 (MaaS)。
其中社交平台是 Q1 重点，签约目标 5 家头部客户。

## 技术栈规范
所有 Demo 必须基于阿里云百炼平台开发，禁止使用 AWS Bedrock。
模型选型默认 Qwen-Plus，复杂场景才用 Qwen-Max。
RAG 应用使用 Dify 部署，Agent 应用使用百炼 Agent 编排或 LangGraph。

## OKR
2026 Q1 KPI：完成 20 个客户 PoC，签约金额 ¥3000 万。
团队整体目标客单价 ¥150 万 / 年。

## 团队文化
每周三下午是 Demo Day，全员展示一周的客户案例。
新人入职 30 天内必须完成 3 个 Demo 提交。
"""


def upload_doc():
    """上传内部文档到 RAG 知识库"""
    print("\n=== 上传内部文档到知识库 ===")
    resp = requests.post(
        f"{GATEWAY}/v1/rag/documents",
        headers=HEADERS,
        json={
            "doc_id": "internal_handbook_2026",
            "content": INTERNAL_DOC,
            "metadata": {"source": "team_handbook", "year": 2026},
        },
    )
    print(resp.json())


def query_pure_llm(question):
    """纯 LLM 回答（不带知识）"""
    resp = requests.post(
        f"{GATEWAY}/v1/chat/completions",
        headers=HEADERS,
        json={
            "model": "fast",
            "messages": [{"role": "user", "content": question}],
            "max_tokens": 200,
        },
    )
    data = resp.json()
    return {
        "answer": data["choices"][0]["message"]["content"],
        "tokens": data["usage"]["total_tokens"],
        "cost": data["x_gateway"]["cost_cny"],
    }


def query_rag(question):
    """RAG 增强回答"""
    resp = requests.post(
        f"{GATEWAY}/v1/rag/query",
        headers=HEADERS,
        json={"question": question, "model": "fast", "top_k": 2},
    )
    return resp.json()


# ========== 测试问题 ==========
QUESTIONS = [
    "阿里云海外 GenAI SA 团队的总监是谁？",
    "2026 年 Q1 团队的 KPI 目标是多少？",
    "团队规定 RAG 应用应该用什么工具部署？",
    "新人入职多久必须提交 Demo？",
]


if __name__ == "__main__":
    upload_doc()

    for q in QUESTIONS:
        print("\n" + "=" * 70)
        print(f"❓ 问题: {q}")
        print("=" * 70)

        print("\n[A] 纯 LLM（小模型瞎猜）：")
        a = query_pure_llm(q)
        print(f"答: {a['answer']}")
        print(f"成本: ¥{a['cost']}")

        print("\n[B] RAG 增强（小模型 + 检索）：")
        b = query_rag(q)
        print(f"答: {b['answer']}")
        print(f"检索到 {len(b['sources'])} 个文档片段")
        print(f"延迟: 检索 {b['stats']['retrieval_ms']}ms + 生成 {b['stats']['generation_ms']}ms")
        print(f"成本: ¥{b['stats']['cost_cny']}")
