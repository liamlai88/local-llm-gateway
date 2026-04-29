#!/bin/bash
# 一键完整测试 MCP Server
# 用法: DASHSCOPE_API_KEY=sk-xxx bash experiments/mcp_full_test.sh

set -e

cd "$(dirname "$0")/.."  # 切到 ai-gateway 目录

if [ -z "$DASHSCOPE_API_KEY" ]; then
    echo "❌ 错误: DASHSCOPE_API_KEY 未设置"
    echo "正确用法: DASHSCOPE_API_KEY=sk-xxx bash experiments/mcp_full_test.sh"
    exit 1
fi

echo "================================================================"
echo "Step 1: 检查 API Key"
echo "================================================================"
echo "✓ DASHSCOPE_API_KEY: ${DASHSCOPE_API_KEY:0:10}..."

echo ""
echo "================================================================"
echo "Step 2: 准备知识库（Python 直接调 rag 模块）"
echo "================================================================"
python3 << 'PYEOF'
import rag

# 清空
try:
    rag.clear_all()
    print("✓ 已清空旧数据")
except Exception as e:
    print(f"清空失败（首次运行可忽略）: {e}")

# 插入
docs = {
    "product_b": """产品名称: 阿里云 GPU 计算实例 ecs.gn7e-c12g1.3xlarge
GPU 型号: NVIDIA A100
显存: 80GB HBM2e
定价: 按量付费 ¥68/小时，包月 ¥35000""",
    "product_a": """产品名称: 阿里云 GPU 计算实例 ecs.gn7i-c8g1.2xlarge
GPU 型号: NVIDIA A10
显存: 24GB GDDR6
定价: 按量付费 ¥18/小时，包月 ¥9800""",
}

for doc_id, content in docs.items():
    res = rag.add_document(doc_id, content)
    print(f"✓ 上传 {doc_id}: {res['chunks']} chunks")

print(f"\n📊 知识库最终状态: {rag.stats()}")
PYEOF

echo ""
echo "================================================================"
echo "Step 3: 跑完整 MCP 测试"
echo "================================================================"
python3 experiments/mcp_test.py
