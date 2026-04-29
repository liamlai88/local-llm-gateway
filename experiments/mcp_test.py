"""
MCP Server 测试脚本（不依赖 Node.js）
直接通过 stdio 协议跟 mcp_server.py 通信，验证 4 个工具都正常
"""
import os
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 确保 mcp_server.py 路径稳定（不依赖 cwd）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SERVER_SCRIPT = os.path.join(PROJECT_ROOT, "mcp_server.py")


async def main():
    # 启动 mcp_server.py 子进程，通过 stdio 通信
    # 关键: env 必须显式传，否则子进程拿不到 DASHSCOPE_API_KEY
    server_params = StdioServerParameters(
        command="python3",
        args=[SERVER_SCRIPT],
        env=os.environ.copy(),  # 显式继承父进程环境变量
        cwd=PROJECT_ROOT,        # 显式指定工作目录（chroma_data 所在）
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化
            await session.initialize()

            print("=" * 70)
            print("✅ MCP Server 连接成功")
            print("=" * 70)

            # 列出所有工具
            tools = await session.list_tools()
            print(f"\n📦 工具列表 ({len(tools.tools)} 个):")
            for t in tools.tools:
                print(f"  - {t.name}: {t.description[:60] if t.description else ''}...")

            # 列出所有资源
            resources = await session.list_resources()
            print(f"\n📚 资源列表 ({len(resources.resources)} 个):")
            for r in resources.resources:
                print(f"  - {r.uri}: {r.name}")

            # 列出所有 prompts
            prompts = await session.list_prompts()
            print(f"\n📝 Prompts 列表 ({len(prompts.prompts)} 个):")
            for p in prompts.prompts:
                print(f"  - {p.name}: {p.description[:60] if p.description else ''}...")

            # 测试 4 个工具
            print("\n" + "=" * 70)
            print("🧪 工具测试")
            print("=" * 70)

            tests = [
                ("calculator", {"expression": "15 * 0.6"}),
                ("get_weather", {"city": "杭州"}),
                ("extract_number", {"text": "定价: ¥68/小时", "hint": "价格"}),
                ("kb_search", {"query": "A100 价格"}),
            ]

            for tool_name, args in tests:
                print(f"\n  → {tool_name}({args})")
                try:
                    result = await session.call_tool(tool_name, args)
                    # result.content 是 list[TextContent]
                    content = result.content[0].text if result.content else "(空)"
                    print(f"     结果: {content[:120]}")
                except Exception as e:
                    print(f"     ❌ 错误: {e}")

            # 测试资源读取
            print("\n" + "=" * 70)
            print("📖 资源读取测试")
            print("=" * 70)
            try:
                res = await session.read_resource("kb://documents")
                content = res.contents[0].text if res.contents else "(空)"
                print(f"\nkb://documents:\n{content}")
            except Exception as e:
                print(f"❌ 错误: {e}")

            # 测试 Prompt 模板
            print("\n" + "=" * 70)
            print("💡 Prompt 模板测试")
            print("=" * 70)
            try:
                prompt = await session.get_prompt("cost_calc_prompt", {"product": "A100", "hours": "24"})
                msg = prompt.messages[0].content.text if prompt.messages else ""
                print(f"\ncost_calc_prompt(A100, 24):\n{msg}")
            except Exception as e:
                print(f"❌ 错误: {e}")


if __name__ == "__main__":
    asyncio.run(main())
