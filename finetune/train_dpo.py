"""
DPO 训练脚本 - 实验 #16
- 基础模型: Qwen/Qwen2.5-1.5B-Instruct (HF Hub 自动下载)
- 数据: finetune/dpo_data/{train,valid}.jsonl
- 输出: finetune/dpo_adapter/
- 设备: MPS (Apple Silicon)

预计时间: ~20-30 分钟（M5 24GB + 1.5B + LoRA）
"""

import os
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "dpo_data")
OUT_DIR = os.path.join(HERE, "dpo_adapter")
# 直接读本地（浏览器下载后手动放置），跳过 modelscope SDK 的校验重下
MODEL_NAME = os.path.expanduser(
    "~/.cache/modelscope/hub/models/Qwen/Qwen2___5-1___5B-Instruct"
)
print(f"使用本地模型: {MODEL_NAME}")


def load_jsonl_dataset(path: str) -> Dataset:
    with open(path) as f:
        data = [json.loads(l) for l in f]
    return Dataset.from_list(data)


def main():
    print(f"加载基础模型: {MODEL_NAME}")
    print("  (首次运行会从 HuggingFace Hub 下载 ~3GB，请耐心等待)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型到 MPS
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="mps",
    )
    print(
        f"模型加载完成. 参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B"
    )

    # LoRA 配置
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 加载数据
    train_ds = load_jsonl_dataset(os.path.join(DATA_DIR, "train.jsonl"))
    valid_ds = load_jsonl_dataset(os.path.join(DATA_DIR, "valid.jsonl"))
    print(f"训练数据: {len(train_ds)} 条, 验证数据: {len(valid_ds)} 条")

    # DPO 配置
    dpo_config = DPOConfig(
        output_dir=OUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        beta=0.1,  # DPO temperature
        max_length=1024,
        remove_unused_columns=False,
        report_to="none",
        fp16=False,  # MPS 不支持 fp16 训练，用 bf16/fp32
        bf16=False,  # M5 MPS bf16 也有限制
    )

    # DPOTrainer 默认会自动构建 ref_model（policy 的初始 copy）
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # 自动构建
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("\n开始训练...")
    trainer.train()

    print(f"\n保存 LoRA adapter 到 {OUT_DIR}")
    trainer.save_model(OUT_DIR)
    print("✓ 训练完成！")


if __name__ == "__main__":
    main()
