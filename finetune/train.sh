#!/bin/bash
# LoRA 微调脚本 - 让 Qwen2.5-1.5B 学会 JSON 工具调用
set -e

cd "$(dirname "$0")"

MODEL_PATH="./model_base"
DATA_DIR="./data"
ADAPTER_DIR="./lora_adapter"

echo "================================================================"
echo "MLX LoRA 微调"
echo "================================================================"
echo "基础模型: $MODEL_PATH"
echo "训练数据: $DATA_DIR"
echo "LoRA 输出: $ADAPTER_DIR"
echo ""

mkdir -p "$ADAPTER_DIR"

# MLX-LM LoRA 关键参数说明:
# --train: 训练模式
# --iters 200: 训练 200 步（80 条数据 batch=4 ≈ 10 epoch）
# --learning-rate 1e-5: 小学习率（LoRA 标准）
# --num-layers 8: 只对最后 8 层加 LoRA（M5 24GB 够用）
# --batch-size 1: M5 上保守设置
# --max-seq-length 1024: 我们的 JSON 计划不会超过这个

python3 -m mlx_lm lora \
  --model "$MODEL_PATH" \
  --train \
  --data "$DATA_DIR" \
  --iters 200 \
  --learning-rate 1e-5 \
  --num-layers 8 \
  --batch-size 1 \
  --max-seq-length 1024 \
  --val-batches 5 \
  --steps-per-eval 50 \
  --steps-per-report 10 \
  --adapter-path "$ADAPTER_DIR"

echo ""
echo "================================================================"
echo "✓ 训练完成！LoRA adapter 保存在: $ADAPTER_DIR"
echo "================================================================"
