"""
从现有 SFT 数据生成 DPO 偏好对
chosen = 原始干净 JSON
rejected = 4 种典型 corruption 之一

输出: finetune/dpo_data/{train,valid,test}.jsonl
格式: {"prompt": "...", "chosen": "...", "rejected": "..."}
"""

import json
import os
import random
import re

random.seed(42)

SRC_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "dpo_data")
os.makedirs(OUT_DIR, exist_ok=True)


# ========== 4 种 Corruption ==========
def corrupt_markdown(json_text: str) -> str:
    """包裹在 markdown 代码块里"""
    return f"```json\n{json_text}\n```"


def corrupt_preamble(json_text: str) -> str:
    """加一段散文前缀"""
    preambles = [
        "好的，以下是执行计划：\n",
        "我已经分析了你的需求，计划如下：\n",
        "明白了！我将这样执行：\n",
        "Here is the plan:\n",
        "Let me think... I'll do this:\n",
    ]
    return random.choice(preambles) + json_text


def corrupt_omit_purpose(json_text: str) -> str:
    """删掉 purpose 字段"""
    try:
        data = json.loads(json_text)
        for step in data.get("plan", []):
            step.pop("purpose", None)
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return re.sub(r',?\s*"purpose":\s*"[^"]*"', "", json_text)


def corrupt_wrong_tool(json_text: str) -> str:
    """把工具名改成错的"""
    swaps = {
        "calculator": random.choice(["calc", "compute", "math"]),
        "get_weather": random.choice(["weather", "fetch_weather"]),
        "kb_search": random.choice(["search", "knowledge_base"]),
        "extract_number": random.choice(["extract", "get_number"]),
    }
    out = json_text
    for orig, wrong in swaps.items():
        if orig in out and random.random() < 0.5:
            out = out.replace(f'"tool": "{orig}"', f'"tool": "{wrong}"', 1)
            break  # 只改一处即可
    return out


CORRUPTIONS = [
    ("markdown", corrupt_markdown, 0.30),
    ("preamble", corrupt_preamble, 0.30),
    ("omit_purpose", corrupt_omit_purpose, 0.20),
    ("wrong_tool", corrupt_wrong_tool, 0.20),
]


def pick_corruption():
    r = random.random()
    cum = 0
    for name, fn, p in CORRUPTIONS:
        cum += p
        if r < cum:
            return name, fn
    return CORRUPTIONS[-1][0], CORRUPTIONS[-1][1]


# ========== 主转换 ==========
def convert_file(src_path: str, dst_path: str):
    n_total = 0
    n_corrupted = 0
    corruption_stats = {name: 0 for name, _, _ in CORRUPTIONS}

    with open(src_path) as f_in, open(dst_path, "w") as f_out:
        for line in f_in:
            ex = json.loads(line)
            msgs = ex["messages"]
            # 找 system + user + assistant
            system = next((m["content"] for m in msgs if m["role"] == "system"), "")
            user = next((m["content"] for m in msgs if m["role"] == "user"), "")
            assistant = next(
                (m["content"] for m in msgs if m["role"] == "assistant"), ""
            )

            # 构造 DPO 格式: prompt 用 chat template 拼，chosen/rejected 是 assistant 输出
            prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
            chosen = assistant

            # 生成 rejected
            name, fn = pick_corruption()
            rejected = fn(assistant)
            if rejected == chosen:  # corruption 没生效（如 wrong_tool 找不到匹配工具）
                # fallback 强制用 markdown
                name, fn = "markdown", corrupt_markdown
                rejected = fn(assistant)

            corruption_stats[name] += 1
            n_total += 1
            n_corrupted += 1

            f_out.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return n_total, corruption_stats


def main():
    for split in ["train", "valid", "test"]:
        src = os.path.join(SRC_DIR, f"{split}.jsonl")
        dst = os.path.join(OUT_DIR, f"{split}.jsonl")
        n, stats = convert_file(src, dst)
        print(f"{split:8s}: {n} 条偏好对  | corruption 分布: {stats}")
        print(f"  → {dst}")

    print("\nDPO 数据集准备完成！")


if __name__ == "__main__":
    main()
