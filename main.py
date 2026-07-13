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
    工具层另外提供 update_todo，用于让 Agent 自动维护任务进度。
"""

import sys
import os
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s [%(name)s] %(message)s",
)

from core.message import Message
from core.memory import ShortMemory, LongMemory
from core.runtime import AgentRuntime
from core.todo import TodoList
from core.todo_store import TodoStore
from core.project_context import ProjectContext
from core.planning import PlanningManager
from core.permission import PermissionManager
from core.compression import ContextCompressor
from llm.deepseek_client import DeepSeekClient
from tools.registry import ToolRegistry
from tools.file_tool import FileReadTool, FileWriteTool
from tools.code_executor import CodeExecutorTool
from tools.web_search import WebSearchTool
from tools.project_tools import LSTool, GrepTool, EditTool
from tools.todo_tool import TodoUpdateTool
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
    "\n\n【任务追踪 - 重要】"
    "当用户提出需要多步骤完成的编程任务时，请先使用 update_todo 工具创建任务列表，"
    "执行过程中用 update_todo 标记 start/done/block，保持任务进度和实际工作同步。"
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
    "\n\n【任务追踪 - 重要】"
    "当用户提出需要多步骤完成的编程任务时，请先使用 update_todo 工具创建任务列表，"
    "执行过程中用 update_todo 标记 start/done/block，保持任务进度和实际工作同步。"
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
    permission_fn = None
    memory_embedder = None

    todo_store = TodoStore(os.path.join(project_root, ".agent_todo.json"))
    todo = todo_store.load()

    if use_tools:
        registry = ToolRegistry()
        registry.register(FileReadTool(workspace_root=project_root))
        registry.register(FileWriteTool(workspace_root=project_root))
        registry.register(CodeExecutorTool())
        registry.register(WebSearchTool())
        registry.register(LSTool(workspace_root=project_root))
        registry.register(GrepTool(workspace_root=project_root))
        registry.register(EditTool(workspace_root=project_root))
        registry.register(TodoUpdateTool(todo, todo_store))
        permission_manager = PermissionManager(
            tool_registry=registry,
            project_root=project_root,
        )
        permission_fn = permission_manager.approve
        print(f"🔧 已加载 {registry.tool_count} 个工具: {registry.list_names()}")
        print("🔐 权限系统: 已启用（写文件、执行代码、联网搜索等风险操作需确认）")

    if use_rag:
        print("📚 正在加载 RAG 模型（BGE-M3），首次需要下载...")
        retriever = create_retriever()
        memory_embedder = retriever.embedder
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
    today = datetime.now().strftime("%Y-%m-%d")
    prompt = (
        f"{prompt}\n\n"
        f"【当前日期】{today}。涉及今天、明天、后天等相对日期时，以该日期为基准；"
        "实时天气、新闻、价格等信息应使用联网搜索工具核验。"
    )

    # Stage 2 Step 2 — 加载 AGENT.md
    project_ctx = ProjectContext(os.path.join(project_root, "AGENT.md"))
    project_context_str = project_ctx.get_context_str()
    if project_context_str:
        print(f"📋 AGENT.md: 已加载项目规则 ({project_ctx.size} 字符)")

    # 初始化 LongMemory（启用自动总结，每8轮触发）
    long_memory = LongMemory(storage_path="memory_long.json", embedder=memory_embedder)

    # Stage 2 Step 4 — 上下文压缩
    context_compressor = ContextCompressor(
        threshold_tokens=50000,
        keep_recent=12,
    )

    # Stage 2 Step 5 — Planning 层
    planning_manager = PlanningManager(todo_list=todo, enable_llm_planning=False)

    runtime = AgentRuntime(
        llm_client=llm,
        system_prompt=prompt,
        tool_registry=registry,
        rag_fn=rag_fn,
        long_memory=long_memory,
        project_context=project_context_str,
        todo_list=todo,
        planning_manager=planning_manager,
        permission_fn=permission_fn,
        context_compressor=context_compressor,
    )
    runtime.todo_store = todo_store
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
    print("  /todo show | add <描述> | start <序号> | done <序号> [结果] | block <序号> [原因]")
    print("  复杂任务会先进入 Planning 模式，再在执行中持续重规划")
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
            if getattr(runtime, "planning_manager", None):
                runtime.planning_manager.reset()
            if hasattr(runtime, "todo_store"):
                runtime.todo_store.save(runtime.todo_list)
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
        _show_planning(runtime)

    elif sub == "add":
        if len(parts) < 3:
            print("用法: /todo add <任务描述>")
        else:
            task = todo.add(parts[2])
            _save_todo(runtime)
            print(f"✅ 已添加: {task.description}")
            print(todo.format_compact())
            _show_planning(runtime)

    elif sub == "done" or sub == "complete":
        if len(parts) < 3:
            print("用法: /todo done <序号> [结果]")
        else:
            idx, result = _parse_index_and_result(parts[2])
            if idx is None:
                print(f"❌ 无效的序号: {parts[2]}")
            elif todo.complete(idx, result):
                _save_todo(runtime)
                print(f"✅ 任务 {idx} 已标记完成")
            else:
                print(f"❌ 无效的任务序号: {idx}")
            _show_planning(runtime)

    elif sub == "start":
        if len(parts) < 3:
            print("用法: /todo start <序号>")
        else:
            try:
                idx = int(parts[2])
                if todo.start(idx):
                    _save_todo(runtime)
                    print(f"🔄 任务 {idx} 已开始")
                else:
                    print(f"❌ 无效的任务序号: {idx}")
            except ValueError:
                print(f"❌ 无效的序号: {parts[2]}")
            _show_planning(runtime)

    elif sub == "reset":
        if len(parts) < 3:
            print("用法: /todo reset <序号>")
        else:
            try:
                idx = int(parts[2])
                if todo.reset(idx):
                    _save_todo(runtime)
                    print(f"⬜ 任务 {idx} 已重置为待处理")
                else:
                    print(f"❌ 无效的任务序号: {idx}")
            except ValueError:
                print(f"❌ 无效的序号: {parts[2]}")
            _show_planning(runtime)

    elif sub == "block":
        if len(parts) < 3:
            print("用法: /todo block <序号> [原因]")
        else:
            idx, result = _parse_index_and_result(parts[2])
            if idx is None:
                print(f"❌ 无效的序号: {parts[2]}")
            elif todo.block(idx, result):
                _save_todo(runtime)
                print(f"🚫 任务 {idx} 已标记阻塞")
            else:
                print(f"❌ 无效的任务序号: {idx}")
            _show_planning(runtime)

    elif sub == "cancel":
        if len(parts) < 3:
            print("用法: /todo cancel <序号> [原因]")
        else:
            idx, result = _parse_index_and_result(parts[2])
            if idx is None:
                print(f"❌ 无效的序号: {parts[2]}")
            elif todo.cancel(idx, result):
                _save_todo(runtime)
                print(f"❌ 任务 {idx} 已取消")
            else:
                print(f"❌ 无效的任务序号: {idx}")
            _show_planning(runtime)

    elif sub == "remove":
        if len(parts) < 3:
            print("用法: /todo remove <序号>")
        else:
            try:
                idx = int(parts[2])
                removed = todo.remove(idx)
                if removed:
                    _save_todo(runtime)
                    print(f"🗑️ 已移除: {removed.description}")
                else:
                    print(f"❌ 无效的任务序号: {idx}")
            except ValueError:
                print(f"❌ 无效的序号: {parts[2]}")
            _show_planning(runtime)

    elif sub == "clear":
        todo.clear()
        if getattr(runtime, "planning_manager", None):
            runtime.planning_manager.reset()
        _save_todo(runtime)
        print("🧹 TodoList 已清空")

    else:
        print(f"❌ 未知的 /todo 子命令: {sub}")
        print("  可用: show | add <描述> | done <序号> | start <序号> | reset <序号> | block <序号> [原因] | cancel <序号> [原因] | remove <序号> | clear")

    print()


def _save_todo(runtime: AgentRuntime) -> None:
    store = getattr(runtime, "todo_store", None)
    if store is not None and runtime.todo_list is not None:
        store.save(runtime.todo_list)


def _parse_index_and_result(text: str):
    parts = text.split(maxsplit=1)
    try:
        idx = int(parts[0])
    except (ValueError, IndexError):
        return None, None
    result = parts[1] if len(parts) > 1 else None
    return idx, result


def _show_planning(runtime: AgentRuntime) -> None:
    pm = getattr(runtime, "planning_manager", None)
    if pm is None:
        return
    text = pm.format_for_prompt()
    if text:
        print(text)


if __name__ == "__main__":
    main()
