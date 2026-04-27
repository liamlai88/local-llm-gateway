#!/bin/bash
PROMPT="用100字解释什么是Transformer架构"

for MODEL in qwen2.5-1.5b qwen2.5-1.5b-q8; do
  echo "=== $MODEL ==="
  ollama run $MODEL --verbose "$PROMPT" 2>&1 | tail -10
  echo ""
done
