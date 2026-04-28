"""
Agent ReAct 实验
4 道题，每道题需要不同工具组合：
1. 纯计算 (calculator)
2. 单工具 (get_weather)
3. 多工具组合 (weather + calculator)
4. RAG + 计算 (kb_search + calculator) ⭐ 最复杂
"""
import requests
import json

GATEWAY = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer sk-demo-002", "Content-Type": "application/json"}


def upload_kb():
    """先把 GPU 产品文档上传到 RAG，给 Agent 准备数据"""
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


def run_agent(question, model="fast"):
    return requests.post(
        f"{GATEWAY}/v1/agent/run",
        headers=HEADERS,
        json={"question": question, "max_iterations": 6, "model": model},
        timeout=180,
    ).json()


def print_trace(result):
    print(f"\n💡 最终答案: {result['answer']}")
    print(f"⏱  耗时: {result['latency_ms']}ms / {result['iterations']} 轮")
    print(f"📊 状态: {result['status']}")
    print("\n📜 推理过程:")
    for step in result.get("trace", []):
        print(f"\n  [Step {step['step']}] type={step['type']}")
        if step["type"] == "action":
            print(f"    Tool: {step['tool']}({json.dumps(step['args'], ensure_ascii=False)})")
            print(f"    Observation: {step['observation'][:150]}")
        elif step["type"] == "final":
            print(f"    答案: {step['answer']}")
        elif step["type"] == "error":
            print(f"    错误: {step['error']}")


TESTS = [
    {
        "id": "Q1",
        "question": "15 乘以 0.6 等于多少？",
        "category": "🧮 纯计算 (1工具)",
        "expected_answer": "9",
    },
    {
        "id": "Q2",
        "question": "杭州现在天气怎么样？",
        "category": "🌤  单工具 (1工具)",
        "expected_answer": "晴, 22°C",
    },
    {
        "id": "Q3",
        "question": "杭州现在的气温减去北京气温是多少度？",
        "category": "🔗 多工具组合 (3工具)",
        "expected_answer": "7",
    },
    {
        "id": "Q4",
        "question": "阿里云 A100 实例按量付费一天大概多少钱？请基于知识库数据计算（按 24 小时算）。",
        "category": "🎯 RAG + 计算 (2工具) ⭐",
        "expected_answer": "1632",
    },
]


if __name__ == "__main__":
    upload_kb()

    for t in TESTS:
        print("\n" + "=" * 75)
        print(f"❓ [{t['id']}] {t['question']}")
        print(f"   类别: {t['category']}")
        print(f"   期望答案包含: {t['expected_answer']}")
        print("=" * 75)
        result = run_agent(t["question"])
        print_trace(result)
