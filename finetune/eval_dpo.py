"""
评测 DPO 训练效果
- 对比 base Qwen2.5-1.5B-Instruct vs DPO-tuned 在测试集上的：
  1. JSON 合规率（输出能否直接 json.loads）
  2. 任务结构正确率（顶层是否有 "plan" key）
  3. 平均 token 长度（DPO 应该让答案更简洁，不带 markdown/前缀）
- 测试集: finetune/dpo_data/test.jsonl 的 prompts
"""

import os
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "dpo_data")
DPO_ADAPTER = os.path.join(HERE, "dpo_adapter")
MODEL_NAME = os.path.expanduser(
    "~/.cache/modelscope/hub/models/Qwen/Qwen2___5-1___5B-Instruct"
)


def parse_test_set(path: str):
    """从 test.jsonl 抽出 prompt（不含已加的 ChatML 模板，因为我们要用 chat template）"""
    tests = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            # prompt 是 "<|im_start|>system...<|im_start|>user...<|im_start|>assistant\n"
            # 我们从中提取 system 和 user 文本即可
            sys_m = re.search(r"system\n(.+?)<\|im_end\|>", ex["prompt"], re.DOTALL)
            usr_m = re.search(r"user\n(.+?)<\|im_end\|>", ex["prompt"], re.DOTALL)
            tests.append(
                {
                    "system": sys_m.group(1) if sys_m else "",
                    "user": usr_m.group(1) if usr_m else "",
                    "chosen": ex["chosen"],
                    "rejected": ex["rejected"],
                }
            )
    return tests


def grade_output(text: str):
    """返回 {is_json: bool, has_plan: bool, has_markdown: bool, has_preamble: bool, length: int}"""
    has_markdown = "```" in text
    # 检查前置散文（JSON 开头不是 { 就算）
    stripped = text.strip()
    has_preamble = not stripped.startswith("{") and not stripped.startswith("```")

    # 提取 JSON 部分
    json_text = text
    if has_markdown:
        m = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
        if m:
            json_text = m.group(1)
    else:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            json_text = m.group(0)

    is_json = False
    has_plan = False
    try:
        data = json.loads(json_text)
        is_json = True
        has_plan = "plan" in data
    except Exception:
        pass

    return {
        "is_json": is_json,
        "has_plan": has_plan,
        "has_markdown": has_markdown,
        "has_preamble": has_preamble,
        "length": len(text),
        "raw": text[:200],
    }


def generate(
    model, tokenizer, system: str, user: str, max_new_tokens: int = 400
) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return text.strip()


def evaluate_model(model, tokenizer, tests, name: str):
    print(f"\n=== 评测: {name} ===")
    results = []
    for i, t in enumerate(tests, 1):
        out = generate(model, tokenizer, t["system"], t["user"])
        g = grade_output(out)
        results.append({"i": i, "user": t["user"][:40], **g})
        marks = (
            ("J" if g["is_json"] else "-")
            + ("P" if g["has_plan"] else "-")
            + ("M" if g["has_markdown"] else "-")
            + ("p" if g["has_preamble"] else "-")
        )
        print(f"  Q{i:2d} [{marks}] len={g['length']:4d}  {t['user'][:30]}")

    # 汇总
    n = len(results)
    summary = {
        "json_rate": sum(1 for r in results if r["is_json"]) / n,
        "plan_rate": sum(1 for r in results if r["has_plan"]) / n,
        "markdown_rate": sum(1 for r in results if r["has_markdown"]) / n,
        "preamble_rate": sum(1 for r in results if r["has_preamble"]) / n,
        "avg_length": sum(r["length"] for r in results) / n,
    }
    print("  ---")
    print(f"  JSON 合规: {summary['json_rate']:.0%}")
    print(f"  Plan 结构正确: {summary['plan_rate']:.0%}")
    print(f"  Markdown 包裹率: {summary['markdown_rate']:.0%}（越低越好）")
    print(f"  散文前缀率: {summary['preamble_rate']:.0%}（越低越好）")
    print(f"  平均长度: {summary['avg_length']:.0f} 字符")
    return summary, results


def main():
    tests = parse_test_set(os.path.join(DATA_DIR, "test.jsonl"))
    print(f"测试集: {len(tests)} 题")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Base 模型
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="mps"
    )
    base_summary, base_results = evaluate_model(
        base, tokenizer, tests, "Base Qwen2.5-1.5B-Instruct"
    )

    # 释放 base 占的显存
    del base
    torch.mps.empty_cache()

    # 2. DPO 模型
    base2 = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="mps"
    )
    dpo_model = PeftModel.from_pretrained(base2, DPO_ADAPTER)
    dpo_summary, dpo_results = evaluate_model(dpo_model, tokenizer, tests, "DPO-tuned")

    # 对比
    print(f"\n\n{'=' * 70}")
    print("对比汇总")
    print(f"{'=' * 70}")
    print(f"{'指标':<22}{'Base':<14}{'DPO':<14}{'差值':<10}")
    for k in ["json_rate", "plan_rate", "markdown_rate", "preamble_rate", "avg_length"]:
        bv = base_summary[k]
        dv = dpo_summary[k]
        diff = dv - bv
        if k == "avg_length":
            print(f"{k:<22}{bv:<14.0f}{dv:<14.0f}{diff:+.0f}")
        else:
            print(f"{k:<22}{bv:<14.2%}{dv:<14.2%}{diff:+.2%}")

    # 保存
    out = os.path.join(HERE, "..", "experiments", "dpo_eval_results.json")
    with open(out, "w") as f:
        json.dump(
            {
                "base": {"summary": base_summary, "results": base_results},
                "dpo": {"summary": dpo_summary, "results": dpo_results},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n详细结果: {out}")


if __name__ == "__main__":
    main()
