"""
LoRA 微调效果对比 v2 - 用 mlx_lm Python API 直接调用
"""
import json
import time
from pathlib import Path
from mlx_lm import load, generate

BASE = Path(__file__).parent
MODEL = str(BASE / "model_base")
ADAPTER = str(BASE / "lora_adapter")

SYSTEM_PROMPT = """你是任务规划器。给定用户问题，输出 JSON 格式的工具执行计划。

可用工具：
- calculator(expression): 数学计算
- get_weather(city): 城市天气查询
- kb_search(query): 知识库检索
- extract_number(text, hint): 从文本提取数字

输出格式（严格 JSON，不要任何额外文字）：
{"plan": [{"step": 序号, "tool": "工具名", "args": {...}, "purpose": "本步目的"}]}"""

TESTS = [
    {"id": "T1", "category": "🧮 简单计算",     "question": "37 乘以 8 等于多少？",          "expected_tools": ["calculator"],                                "expected_count": 1},
    {"id": "T2", "category": "🌤 单工具天气",   "question": "纽约现在天气怎么样？",          "expected_tools": ["get_weather"],                              "expected_count": 1},
    {"id": "T3", "category": "🔗 多工具气温差",  "question": "迪拜和首尔的气温差是多少？",     "expected_tools": ["get_weather", "extract_number", "calculator"], "expected_count": 5},
    {"id": "T4", "category": "🎯 RAG + 计算",   "question": "阿里云 H100 实例跑 5 天大概多少钱？基于知识库价格算", "expected_tools": ["kb_search", "extract_number", "calculator"],   "expected_count": 3},
    {"id": "T5", "category": "💬 无工具问候",   "question": "你好",                          "expected_tools": [],                                            "expected_count": 0},
]


def build_prompt(tokenizer, system, user):
    """用 chat template 构造输入"""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def extract_json(text):
    text = text.strip()
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def evaluate(output, expected_tools, expected_count):
    parsed = extract_json(output)
    if parsed is None:
        return {"json_valid": False, "tools_correct": False, "structure_correct": False}
    plan = parsed.get("plan", [])
    if not isinstance(plan, list):
        return {"json_valid": True, "tools_correct": False, "structure_correct": False}
    used_tools = set()
    for s in plan:
        if isinstance(s, dict) and "tool" in s:
            used_tools.add(s["tool"])
    expected_set = set(expected_tools)
    return {
        "json_valid": True,
        "tools_correct": used_tools == expected_set or (not expected_set and not used_tools),
        "structure_correct": len(plan) == expected_count,
        "actual_tools": list(used_tools),
        "actual_count": len(plan),
    }


def run_test(model, tokenizer, question, max_tokens=400):
    prompt = build_prompt(tokenizer, SYSTEM_PROMPT, question)
    start = time.time()
    output = generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False,
    )
    elapsed = time.time() - start
    return output, elapsed


def main():
    print("=" * 80)
    print("加载基础模型...")
    print("=" * 80)
    base_model, base_tok = load(MODEL)
    print("✓ 基础模型已加载\n")

    print("=" * 80)
    print("加载 LoRA 微调模型...")
    print("=" * 80)
    lora_model, lora_tok = load(MODEL, adapter_path=ADAPTER)
    print("✓ LoRA 模型已加载\n")

    summary = {"base": {"json": 0, "tools": 0, "structure": 0},
               "lora": {"json": 0, "tools": 0, "structure": 0}}

    for t in TESTS:
        print("=" * 80)
        print(f"{t['id']} {t['category']}: {t['question']}")
        print(f"期望: 工具={t['expected_tools']} 步数={t['expected_count']}")
        print("=" * 80)

        for label, model, tok, key in [
            ("基础模型", base_model, base_tok, "base"),
            ("LoRA 微调", lora_model, lora_tok, "lora"),
        ]:
            output, elapsed = run_test(model, tok, t["question"])
            ev = evaluate(output, t["expected_tools"], t["expected_count"])
            preview = output[:300].replace("\n", "\\n")
            print(f"\n--- {label} ({elapsed:.1f}s) ---")
            print(f"输出: {preview}")
            print(f"  JSON 合法:  {'✅' if ev['json_valid'] else '❌'}")
            print(f"  工具正确:  {'✅' if ev['tools_correct'] else '❌'} 实际={ev.get('actual_tools', 'N/A')}")
            print(f"  步数正确:  {'✅' if ev['structure_correct'] else '❌'} 实际={ev.get('actual_count', 'N/A')}/期望={t['expected_count']}")

            if ev["json_valid"]: summary[key]["json"] += 1
            if ev["tools_correct"]: summary[key]["tools"] += 1
            if ev["structure_correct"]: summary[key]["structure"] += 1

        print()

    n = len(TESTS)
    print("=" * 80)
    print("📊 总体对比")
    print("=" * 80)
    print(f"\n{'指标':<15} {'基础模型':<15} {'LoRA 微调':<15} {'提升':<10}")
    print("-" * 60)
    for metric in ["json", "tools", "structure"]:
        b, l = summary["base"][metric], summary["lora"][metric]
        delta = l - b
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        name = {"json": "JSON 合法率", "tools": "工具正确率", "structure": "步数正确率"}[metric]
        print(f"{name:<15} {b}/{n} ({b/n*100:3.0f}%)   {l}/{n} ({l/n*100:3.0f}%)   {delta_str}")


if __name__ == "__main__":
    main()
