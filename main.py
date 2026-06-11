"""Agent Runtime — 终端对话入口（DeepSeek 后端）

用法:
    python3 main.py              # 纯聊天（无工具/无RAG）
    python3 main.py --tools      # 启用全部工具（读写文件/执行代码/联网搜索）
    python3 main.py --tools --rag  # 全部启用（含RAG知识库）

LLM 后端：DeepSeek API（deepseek-chat）
    在 .env 中设置 DEEPSEEK_API_KEY=你的密钥

关于工具调用：
    read_file / write_file / execute_code / web_search 是注册在 ToolRegistry
    中的工具，LLM 通过 Function Calling 自动调用——不需要手动输入命令。

关于 /todo 命令：
    /todo 是终端层面的 CLI 命令（not a Tool），用于人工管理任务追踪。
    用法：/todo add <描述>  |  /todo start <序号>  |  /todo done <序号>
    这 4 个 Tool 是完全自动的，不需要手动调用。
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s [%(name)s] %(message)s",
)

from core.message import Message
from core.memory import ShortMemory, LongMemory
from core.runtime import AgentRuntime
from core.todo import TodoList
from core.project_context import ProjectContext
from llm.deepseek_client import DeepSeekClient
from tools.registry import ToolRegistry
from tools.file_tool import FileReadTool, FileWriteTool
from tools.code_executor import CodeExecutorTool
from tools.web_search import WebSearchTool
from rag.retriever import Retriever, create_retriever

SYSTEM_PROMPT = (
    "你是一个 AI Agent 助手，运行在终端对话环境中。"
    "你可以使用工具来：读取文件、写入文件、执行代码（Python/Shell）、联网搜索。"
    "当用户让你做需要这些能力的事情时，请自动调用对应的工具。"
    "回答要简洁清晰。"
    ""
    "【记忆能力 - 非常重要】"
    "你拥有短期记忆能力，本轮对话中读取过的文件内容、分析过的信息、"
    "工具调用的返回结果，你都记得。"
    "用户追问时（如'介绍一下这个功能''还有哪些模块'），"
    "请直接基于已读取的内容和之前的分析来回答，不要重新读取相同的文件！"
    "只有用户要求查看新文件、或确信需要新的工具结果才能回答时，才再次调用工具。"
    "如果你已经分析过某个项目，后续追问该项目时直接用记忆回答即可。"
)

RAG_SYSTEM_PROMPT = (
    "你是一个 AI Agent 助手，同时拥有项目知识库支持。"
    "当用户问及项目架构、设计、模块功能等问题时，"
    "请结合检索到的参考资料给出准确回答。"
    "你也可以使用工具：读文件、写文件、执行代码、联网搜索。"
    ""
    "【记忆能力 - 非常重要】"
    "你拥有短期记忆能力，本轮对话中读取过的文件内容、分析过的信息、"
    "工具调用的返回结果，你都记得。"
    "用户追问时（如'介绍一下这个功能''还有哪些模块'），"
    "请直接基于已读取的内容和之前的分析来回答，不要重新读取相同的文件！"
    "只有用户要求查看新文件、或确信需要新的工具结果才能回答时，才再次调用工具。"
    "如果你已经分析过某个项目，后续追问该项目时直接用记忆回答即可。"
)


def build_runtime(use_tools: bool = False, use_rag: bool = False) -> AgentRuntime:
    project_root = os.path.dirname(__file__)

    try:
        llm = DeepSeekClient(
            model_name="deepseek-chat",
            temperature=1,
        )
    except ValueError as e:
        print(f"❌ DeepSeek API 配置错误: {e}")
        print("请确保 .env 文件中设置了 DEEPSEEK_API_KEY=你的密钥")
        sys.exit(1)

    registry = None
    rag_fn = None

    if use_tools:
        registry = ToolRegistry()
        registry.register(FileReadTool())
        registry.register(FileWriteTool())
        registry.register(CodeExecutorTool())
        registry.register(WebSearchTool())
        print(f"🔧 已加载 {registry.tool_count} 个工具: {registry.list_names()}")

    if use_rag:
        print("📚 正在加载 RAG 模型（BGE-M3），首次需要下载...")
        retriever = create_retriever()
        docs = []
        for fname in ["promote.txt"]:
            fpath = os.path.join(project_root, fname)
            if os.path.exists(fpath):
                with open(fpath, "r", encoding="utf-8") as f:
                    docs.append(f.read())
        if docs:
            retriever.add_documents(docs)
            print(f"📚 RAG 就绪: {retriever.chunk_count} 个文本块")
            rag_fn = lambda q: retriever.query_retrieve(q, top_k=3)

    prompt = RAG_SYSTEM_PROMPT if use_rag else SYSTEM_PROMPT

    # Stage 2 Step 2 — 加载 AGENT.md
    project_ctx = ProjectContext(os.path.join(project_root, "AGENT.md"))
    project_context_str = project_ctx.get_context_str()
    if project_context_str:
        print(f"📋 AGENT.md: 已加载项目规则 ({project_ctx.size} 字符)")

    # 初始化 LongMemory（启用自动总结，每8轮触发）
    long_memory = LongMemory(storage_path="memory_long.json")

    # Stage 2 Step 1 — 初始化 TodoList
    todo = TodoList()

    runtime = AgentRuntime(
        llm_client=llm,
        system_prompt=prompt,
        tool_registry=registry,
        rag_fn=rag_fn,
        long_memory=long_memory,
        project_context=project_context_str,
        todo_list=todo,
    )
    return runtime


def main():
    use_tools = "--tools" in sys.argv
    use_rag = "--rag" in sys.argv

    project_root = os.path.dirname(__file__)

    print("=" * 55)
    print("  Agent Runtime — 终端对话 (DeepSeek)")
    print("=" * 55)
    print(f"  模型: deepseek-chat | 上下文: 128K")
    print(f"  工具: {'✅ 启用' if use_tools else '❌ 禁用'}  |  RAG: {'✅ 启用' if use_rag else '❌ 禁用'}")

    project_ctx = ProjectContext(os.path.join(project_root, "AGENT.md"))
    if project_ctx.is_loaded:
        print(f"  AGENT.md: ✅ 已加载 ({project_ctx.size} 字符)")

    print()
    print("  输入对话内容，/exit 退出，/clear 清空记忆")
    print("  /todo show | add <描述> | start <序号> | done <序号>")
    if use_rag:
        print("  💡 RAG 示例问题: '这个项目采用什么架构？'")
    print("=" * 55)
    print()

    runtime = build_runtime(use_tools=use_tools, use_rag=use_rag)

    while True:
        try:
            user_input = input("👤 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见")
            break

        if not user_input:
            continue

        if user_input.lower() == "/exit":
            print("👋 再见")
            break

        if user_input.lower() == "/clear":
            runtime.reset_memory()
            if runtime.todo_list:
                runtime.todo_list.clear()
            print("🧹 记忆已清空\n")
            continue

        if user_input.lower().startswith("/todo"):
            _handle_todo_command(runtime, user_input)
            continue

        print("🤖 Agent: ", end="", flush=True)
        try:
            reply = runtime.run(user_input)
            print(reply)
        except Exception as e:
            print(f"\n❌ 运行出错: {e}")
        print()


def _handle_todo_command(runtime: AgentRuntime, raw_input: str) -> None:
    todo = runtime.todo_list
    if todo is None:
        print("⚠️ TodoList 未启用\n")
        return

    parts = raw_input.strip().split(maxsplit=2)
    sub = parts[1].lower() if len(parts) > 1 else "show"

    if sub == "show" or sub == "list":
        print(todo.format_for_prompt())
        print(f"进度: {todo.progress():.0%}")

    elif sub == "add":
        if len(parts) < 3:
            print("用法: /todo add <任务描述>")
        else:
            task = todo.add(parts[2])
            print(f"✅ 已添加: {task.description}")
            print(todo.format_compact())

    elif sub == "done" or sub == "complete":
        if len(parts) < 3:
            print("用法: /todo done <序号>")
        else:
            try:
                idx = int(parts[2])
                if todo.complete(idx):
                    print(f"✅ 任务 {idx} 已标记完成")
                else:
                    print(f"❌ 无效的任务序号: {idx}")
            except ValueError:
                print(f"❌ 无效的序号: {parts[2]}")

    elif sub == "start":
        if len(parts) < 3:
            print("用法: /todo start <序号>")
        else:
            try:
                idx = int(parts[2])
                if todo.start(idx):
                    print(f"🔄 任务 {idx} 已开始")
                else:
                    print(f"❌ 无效的任务序号: {idx}")
            except ValueError:
                print(f"❌ 无效的序号: {parts[2]}")

    elif sub == "reset":
        if len(parts) < 3:
            print("用法: /todo reset <序号>")
        else:
            try:
                idx = int(parts[2])
                if todo.reset(idx):
                    print(f"⬜ 任务 {idx} 已重置为待处理")
                else:
                    print(f"❌ 无效的任务序号: {idx}")
            except ValueError:
                print(f"❌ 无效的序号: {parts[2]}")

    elif sub == "clear":
        todo.clear()
        print("🧹 TodoList 已清空")

    else:
        print(f"❌ 未知的 /todo 子命令: {sub}")
        print("  可用: show | add <描述> | done <序号> | start <序号> | reset <序号> | clear")

    print()


if __name__ == "__main__":
    main()