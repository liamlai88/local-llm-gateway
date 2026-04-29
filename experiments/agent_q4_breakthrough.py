"""
Q4 突破实验 - 验证 extract_number 工具能否破解跨 step 数据传递
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


def print_trace(result):
    print(f"\n💡 答案: {result.get('answer', '')[:120]}")
    print(f"⏱  {result.get('iterations', 0)} 步 / {result.get('latency_ms', 0)}ms")
    print(f"📊 状态: {result.get('status')}")
    print("\n📜 执行轨迹:")
    for s in result.get("trace", []):
        if s["type"] == "plan":
            print(f"  📋 计划:")
            for step in s["plan"]:
                print(f"     Step {step['step']}: {step['tool']}({step.get('args', {})})  // {step.get('purpose', '')}")
        elif s["type"] == "execute":
            print(f"  ⚙️  Step {s['step']} 执行: {s['tool']}({s.get('actual_args', {})})")
            print(f"     → {s.get('observation', '')[:120]}")
        elif s["type"] == "final":
            print(f"  ✅ Final: {s.get('answer', '')[:120]}")
        elif s["type"] == "action":
            print(f"  ⚙️  {s['tool']}({s.get('args', {})}) → {s.get('observation', '')[:120]}")


# ========== Q4: 关键测试 ==========
Q4 = "阿里云 A100 实例按量付费一天大概多少钱？请基于知识库数据计算（按 24 小时算）。期望答案: 1632"

print("=" * 80)
print("Q4 突破测试 - 加上 extract_number 工具后，Plan-Execute 能否搞定？")
print("=" * 80)

upload_kb()

CONFIGS = [
    ("Plan-Execute Turbo (新工具)",  {"mode": "plan_execute", "provider": "bailian", "model": "qwen-turbo"}),
    ("Plan-Execute 1.5B (新工具)",   {"mode": "plan_execute", "provider": "local",   "model": "fast"}),
    ("ReAct Turbo (对照组)",        {"mode": "react",        "provider": "bailian", "model": "qwen-turbo", "few_shot": True}),
]

for name, params in CONFIGS:
    print("\n" + "=" * 80)
    print(f"🧪 配置: {name}")
    print("=" * 80)
    result = run_agent(Q4, **params)
    print_trace(result)

    # 严格评判（防假成功）
    hit_1632 = "1632" in result.get("answer", "")
    used_tools = [s.get("tool") for s in result.get("trace", []) if s["type"] in ("action", "execute")]

    # 检查每个工具是否真的成功（observation 不是错误）
    tool_observations = {}
    for s in result.get("trace", []):
        if s["type"] in ("action", "execute"):
            obs = s.get("observation", "")
            tool_observations[s["tool"]] = obs

    kb_real = "kb_search" in tool_observations and "失败" not in tool_observations["kb_search"] and "Error" not in tool_observations.get("kb_search", "")
    extract_real = "extract_number" in tool_observations and "<step" not in str(tool_observations.get("extract_number", "")) and "未找到" not in tool_observations.get("extract_number", "")
    calc_real = "calculator" in tool_observations and "1632" in str(tool_observations.get("calculator", ""))

    real_success = hit_1632 and kb_real and extract_real and calc_real

    print(f"\n🎯 严格评判:")
    print(f"   答案包含 '1632':      {hit_1632}")
    print(f"   kb_search 真实成功:    {kb_real}  obs={tool_observations.get('kb_search', '未调用')[:60]}")
    print(f"   extract_number 真实成功: {extract_real}  obs={tool_observations.get('extract_number', '未调用')[:60]}")
    print(f"   calculator 算出 1632:   {calc_real}  obs={tool_observations.get('calculator', '未调用')[:60]}")
    print(f"\n   {'✅ 真实成功' if real_success else '❌ 假成功（最终答案对，但工具链未真正工作）'}")
