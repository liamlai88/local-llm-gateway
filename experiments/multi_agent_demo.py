"""
Multi-Agent demo runner.

Prerequisite:
  uvicorn gateway:app --port 8000 --reload
"""
import json
import requests

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
    }
    requests.delete(f"{GATEWAY}/v1/rag/documents", headers=HEADERS, timeout=30)
    for doc_id, content in docs.items():
        requests.post(f"{GATEWAY}/v1/rag/documents", headers=HEADERS, json={"doc_id": doc_id, "content": content}, timeout=60)


def run(question):
    resp = requests.post(f"{GATEWAY}/v1/multi-agent/run", headers=HEADERS, json={"question": question, "model": "fast", "llm_final": False}, timeout=180)
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    upload_kb()
    result = run("阿里云 A100 按量付费跑一天大概多少钱？这个方案适合训练大模型吗？")
    print(json.dumps({
        "answer": result["answer"],
        "agents": result["agents"],
        "status": result["status"],
        "trace_roles": [s["role"] for s in result["trace"]],
    }, ensure_ascii=False, indent=2))
