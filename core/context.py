from typing import List, Optional, Dict, Any
from core.message import Message


def build_context(
    system_prompt: str,
    messages: List[Message],
    rag_results: Optional[List[str]] = None,
    tool_results: Optional[List[str]] = None,
    long_memory_context: Optional[str] = None,
    project_context: Optional[str] = None,   
    todo_context: Optional[str] = None,       
) -> List[Message]:

    context: List[Message] = []

    # 系统指令
    context.append(Message(role="system", content=system_prompt))

    # 注意：project_context 已由 ProjectContext.get_context_str() 格式化，
    # 包含完整标题和规则文本，此处直接注入不重复包装。
    if project_context:
        context.append(Message(role="system", content=project_context))

    if todo_context:
        context.append(Message(
            role="system",
            content=f"【当前任务进度】\n{todo_context}",
        ))

    # 长期记忆注入（在系统指令后，对话历史前）
    if long_memory_context:
        context.append(Message(
            role="system",
            content=f"【长期记忆参考】\n{long_memory_context}",
        ))

    # RAG 结果注入
    if rag_results:
        rag_content = "\n\n---\n【检索到的参考资料】\n"
        for i, chunk in enumerate(rag_results, 1):
            rag_content += f"\n{i}. {chunk}\n"
        context.append(Message(role="system", content=rag_content))

    # Tool 结果注入
    if tool_results:
        tool_content = "\n\n---\n【工具执行结果】\n"
        for i, result in enumerate(tool_results, 1):
            tool_content += f"\n{i}. {result}\n"
        context.append(Message(role="tool", content=tool_content))

    # 历史对话（含当前用户输入）
    context.extend(messages)

    return context


def build_short_context(
    system_prompt: str,
    user_input: str,
    rag_results: Optional[List[str]] = None,
    project_context: Optional[str] = None,   # ✅ Stage 2
) -> List[Message]:
    context: List[Message] = [
        Message(role="system", content=system_prompt),
    ]

    if project_context:
        context.append(Message(role="system", content=project_context))

    if rag_results:
        rag_content = "\n\n---\n【检索到的参考资料】\n"
        for i, chunk in enumerate(rag_results, 1):
            rag_content += f"\n{i}. {chunk}\n"
        context.append(Message(role="system", content=rag_content))

    context.append(Message(role="user", content=user_input))
    return context