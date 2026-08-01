# Coder Agent 项目总结

> 阅读范围：项目根目录中除 `web/` 外的 Python 源码、配置规则、README、规划文档和测试。
>
> 项目定位：这是一个本地运行的 Claude Code / Codex 风格 AI 编程助手实验项目。它不是单纯把用户消息转发给大模型的聊天应用，而是围绕 Agent Runtime 构建了工具调用、任务规划、上下文治理、记忆、权限控制、会话持久化和 HTTP/SSE 服务能力。

## 1. 一分钟项目介绍

面试中可以这样介绍：

> 我实现了一个 Python 本地 Coder Agent。系统先通过意图识别决定本轮是普通问答、ReAct 工具执行、复杂任务的 Planning + ReAct、实时联网查询，还是受 Runtime 控制的项目分析。Runtime 将模型的 Native Function Calling 转成统一工具调用，处理工具结果、权限审批、上下文注入和最终回答。服务端通过 FastAPI 暴露多会话 API，并用 SSE 推送工具、计划、权限和最终回答等结构化事件。为了避免长会话和工具结果污染上下文，系统还实现了短期记忆、结构化长期记忆、上下文压缩和 Scratchpad。

一句话概括架构：

```text
CLI / FastAPI API
    -> AgentSessionManager（Web 多会话隔离）
    -> AgentRuntime（意图路由、ReAct、Planning、项目分析、事件流）
    -> DeepSeekClient（LLM Function Calling） + ToolRegistry（本地工具）
    -> Memory / Context / Permission / Todo / RAG
    -> JSON 文件持久化
```

## 2. 技术栈和工程结构

### 2.1 技术栈

- 语言与运行环境：Python 3.10+，当前虚拟环境为 Python 3.12。
- Web 框架：FastAPI + Uvicorn。
- 模型接口：DeepSeek 的 OpenAI 兼容 Chat Completions API，使用标准库 `urllib` 发请求。
- 配置：`python-dotenv` 读取 `.env` 中的 API Key、模型名、超时和重试次数。
- 测试：pytest，测试文件位于项目根目录。
- 可选 RAG：FlagEmbedding 的 BGE-M3、NumPy。
- 持久化：JSON 文件，不使用关系型数据库或向量数据库。
- 流式通信：Server-Sent Events（SSE）。

### 2.2 后端目录职责

```text
main.py                    CLI 入口和 Runtime 装配
llm/deepseek_client.py     DeepSeek/OpenAI 兼容模型客户端
core/runtime/              Agent Runtime、事件、会话、项目分析状态机
core/context/              Prompt 拼装、Scratchpad、压缩、AGENT.md 加载
core/memory/               短期记忆、结构化长期记忆、检索和去重
core/planning/             Planning + ReAct 状态与重规划
core/todo/                 Todo 领域模型和 JSON Store
core/permission/           风险分级、CLI 审批、Web 暂停恢复
tools/                     工具抽象、注册、文件、搜索、执行、Todo 工具
rag/                       文本分块、Embedding、Top-K Retriever
server/                    框架无关 Service 和 FastAPI 薄适配层
test_*.py                  单元和集成测试
AGENT.md                   注入模型的项目规则
promote.txt                架构阶段与设计说明
```

`core/message.py`、`core/session.py`、`core/events.py` 等是旧导入路径兼容层，真实实现迁移到了 `core/runtime/`、`core/context/` 等子包。这体现了对重构后向后兼容的考虑。

## 3. 核心领域对象和依赖倒置

### 3.1 Message：统一对话协议对象

`core.runtime.message.Message` 是内部消息模型，字段包括：

- `role`：`system`、`user`、`assistant`、`tool`；
- `content`：文本内容；
- `timestamp`：时间戳；
- `name`：工具名；
- `metadata`：保存 Function Calling 的调用和返回协议。

模型客户端负责把内部 Message 转为 OpenAI 格式：assistant 带 `function_calls` 时转成 `tool_calls`；tool 消息带 `function_response` 时转成 `tool_call_id` 对应的 `role=tool` 消息。这样 Runtime 不绑定某一家模型协议。

`Message.estimate_tokens()` 采用中文约 1.5 token/字、其他字符约 0.25 token/字符的粗略估计，主要用于阈值控制和无真实 usage 时的统计，不是精确 tokenizer。

### 3.2 依赖注入

Runtime 的构造函数注入 `llm_client`、Memory、LongMemory、ToolRegistry、Permission 函数、RAG 函数、PlanningManager、Todo、Compressor、Scratchpad 等依赖。

好处：

- Runtime 不直接依赖 DeepSeek、FastAPI 或真实文件系统服务。
- 测试可以注入 `FakeLLM`、临时目录、模拟权限函数。
- 未来换模型时，只要实现 `generate()`、`stream_generate()`、`count_tokens()` 等约定接口即可。

Web 端进一步使用 `llm_factory`，由 `AgentSessionManager` 为每个 session 创建隔离的 LLM Client 和 Runtime，避免多个会话共享可变状态。

## 4. Agent Runtime：项目最核心的部分

### 4.1 运行模式

Runtime 入口有两个：

- `run(user_input)`：CLI/兼容同步路径，返回最终字符串。
- `run_events(user_input)`：Web/API 主路径，产出结构化 `AgentEvent` 生成器。

`stream_run()` 被设计为 `run_events()` 的薄包装，而不是第二套 Agent Loop。这避免 CLI、Web 的 prompt、工具协议和权限行为逐渐分叉。

每轮基础流程：

```text
用户输入
  -> 写入 ShortMemory，初始化 Scratchpad 目标
  -> 判断是否需要 Context Compression
  -> 判断是否启动 Planning
  -> Intent Router 决定是否暴露工具
  -> 普通问答：直接 LLM
  -> 工具任务：Native Function Calling ReAct Loop
  -> 项目分析：Runtime 主导的 Evidence Collection
  -> 最终回答写回 Memory 和 LongMemory
```

### 4.2 ReAct 与 Native Function Calling

项目采用 ReAct 思想，但不要求模型输出自由格式的 Thought/Action 字符串，而是用模型原生 Function Calling：

```text
LLM response
  -> text + function_calls
  -> Runtime 生成 assistant tool_calls Message
  -> Registry 执行工具
  -> Runtime 生成 tool result Message
  -> 再次请求 LLM
  -> text final answer
```

优势：工具名、参数、调用 ID 都结构化，减少解析 DSML/Markdown 伪协议的不确定性。

Runtime 支持：

- 单个 `function_call` 的兼容格式；
- 一个响应中的多个 `function_calls`；
- 模型遗漏 ID 时生成回退 ID；
- 参数不是 dict、工具不存在、工具执行抛异常时，返回可读错误 Observation，而非直接让整个 Agent 崩溃；
- 默认最多 12 轮，防止模型循环调用工具无止境消耗成本。

### 4.3 Function Calling 的关键协议问题

这是本项目最值得在面试中展开的细节。

OpenAI/DeepSeek 协议要求：assistant 一旦发出含多个 `tool_calls` 的消息，下一次模型调用前必须为其中每个 `tool_call_id` 补齐一条 `role=tool` 消息。

困难场景是：一批工具的第 2 个调用需要用户审批。若审批后立即请求模型，前一个 assistant tool_calls 只得到部分 tool result，会收到 `insufficient tool messages` 一类 HTTP 400。

项目解决方案：

1. Runtime 保存完整 `function_calls` 批次、当前 index、上下文、工具声明、轮次、已发生的 Planning 变化和缓存文本。
2. 发出 `permission_request` 事件，停止当前 SSE 流。
3. Web 接口收到批准/拒绝后调用 `resume_events()`。
4. 从原 index 恢复，先为当前及后续调用补齐 tool messages；拒绝也要写入一条代表拒绝结果的 tool message。
5. 同批调用完成后，才允许继续请求模型。

这体现了对外部 LLM 协议状态机、可恢复执行和幂等边界的理解。

### 4.4 文本与工具调用同时返回

模型可能同时给出说明文本和工具调用。Runtime 不会立即把这段文本当最终答案，而是缓存为 `pending_text`，优先执行工具；若工具循环耗尽或没有新文本才回退它。

模型也可能错误输出“调用工具”“DSML”“tool_calls”草稿，却没有合法 function call。Runtime 识别这种无效工具草稿：在工具模式里追加严格纠偏提示后重试；普通问答里过滤伪协议并重试一次自然语言回复，避免把内部协议直接展示给用户。

### 4.5 工具暴露最小化和意图识别

ToolRegistry 存在不代表每轮把所有工具发送给 LLM。Runtime 先判断：

- 普通概念、机制解释、建议问题：不传工具 schema。
- 明确读项目、文件、修改、运行、测试：进入工具循环。
- 实时天气等外部信息：只传 `web_search`，不额外暴露 Shell。
- 项目/目录只读分析：只传 `ls`、`grep`、`read_file`。

英文关键词使用 `core.intent.contains_keyword()` 的词边界正则。例如 `test` 不会匹配 `latest`，`run` 不会匹配 `runtime`。这解决了简单 substring 路由导致的误调工具问题。

## 5. Planning + Todo：复杂任务可控执行

### 5.1 何时启动 Planning

`PlanningManager.should_plan()` 使用轻量规则：任务需要行动词，且文本长度超过阈值或带“全部、阶段、复杂、多步、多个、feature”等复杂性词。纯解释问题即使出现“规划”“复杂”等词，也不进入 Planning。

这是一个重要取舍：规划本身会增加 prompt、工具轮次和延迟，因此不应对每个提问过度规划。

### 5.2 PlanState 和重规划

`PlanState` 记录：

- `objective`：用户目标；
- `steps`：3 到 6 个可执行步骤；
- `revision`：计划版本；
- `mode`：`react` 或 `planning`；
- `last_observation`：最近工具观察。

默认关闭 LLM 规划，使用本地启发式生成步骤，避免为了计划额外花一次模型调用并降低稳定性。

当 Observation 包含错误、失败、Traceback 或权限拒绝时，会追加“根据最新错误调整实现方案”和“重新运行验证”；写入或编辑文件后自动追加测试/静态检查。这就是 Observation 驱动的动态重规划。

### 5.3 TodoList

Todo 不是 UI 附属功能，而是 Planning 的可见执行状态。`TaskStatus` 包括：

- `PENDING`：待执行；
- `IN_PROGRESS`：进行中；
- `DONE`：已完成；
- `BLOCKED`：被阻塞；
- `CANCELLED`：已取消。

Todo 支持 add、insert、remove、start、complete、block、cancel、reset、clear、progress 和 JSON 序列化。复杂任务创建计划后会同步 Todo；Agent 可以调用 `update_todo` 工具，CLI 也支持 `/todo` 命令人工维护。

## 6. 工具系统

### 6.1 抽象和注册

所有工具继承 `BaseTool`，必须定义：

- `name`：唯一名称；
- `description`：给 LLM 的使用说明；
- `parameters`：JSON Schema；
- `risk_level`：安全等级；
- `execute(**kwargs)`：实际执行逻辑。

`ToolRegistry` 负责注册、覆盖告警、按名称获取、生成 Function Calling 声明、必填参数校验和统一执行。这是标准插件式/策略式扩展点。

### 6.2 文件和项目工具

- `read_file`：UTF-8 读取，支持起止行；未指定行号时只给文件开头窗口，并明确提示截断，避免一次读取大文件撑爆模型上下文。
- `write_file`：UTF-8 创建或覆盖文件。
- `edit_file`：按行区间替换，支持 `dry_run`，通过 `difflib.unified_diff` 生成预览；适合小范围修改，降低整文件覆盖风险。
- `ls`：目录浏览，支持递归和最大条目数，过滤常见噪声目录。
- `grep`：使用正则检索匹配行，支持文件名 pattern、结果数量限制，跳过二进制和噪声文件。

相对路径以 session/workspace root 解析，而非依赖进程当前目录。权限系统也用同一个 root 判断工作区边界。

### 6.3 代码执行工具

`execute_code` 提供 Python 和 Shell 两种模式：

- Python：使用 `tempfile.mkstemp()` 创建临时 `.py` 文件，`subprocess.run()` 执行，finally 中清理临时文件。
- Shell：使用 `shell=True`，捕获 stdout、stderr、退出码，30 秒 timeout。
- 对 `rm -rf /`、`mkfs`、`dd if=` 等少数明显危险字符串做黑名单拦截。

面试中要坦诚：黑名单不是可靠安全边界。生产环境应使用容器/微虚拟机、命令白名单、只读挂载、CPU/内存/网络资源限制和审计。

### 6.4 联网搜索

`web_search` 通过 DuckDuckGo HTML 页面检索，不需要额外 API Key：

1. 请求 HTML 结果页；
2. 正则提取标题、摘要、URL；
3. 失败时回退 Lite 页面；
4. 清理 HTML 标签和常见实体；
5. 将结果限制在 1 到 10 条。

它适合简单实时信息和技术资料检索，但生产级实现应有来源质量筛选、超时/重试、结构化解析、缓存、限流和注入风险处理。

## 7. 权限系统和安全边界

### 7.1 风险分级

风险等级从低到高：`SAFE`、`CAUTION`、`DANGEROUS`。

- 安全工具自动执行，例如读取、列目录、Todo 更新。
- 写文件、编辑文件、Python 执行等通常需要确认。
- Shell 会动态升为 `DANGEROUS`，因为可以执行任意系统操作。
- `web_search` 动态升为 `CAUTION`，原因是查询文本可能泄露给第三方。

### 7.2 路径策略

`PermissionManager` 对文件相关工具处理：

1. 相对路径拼到 `project_root`。
2. `/dev`、`/proc`、`/sys`、`/etc`、`/bin`、`/usr/bin` 等系统路径直接拒绝。
3. 工作区外路径：只读操作升为 caution，写操作升为 dangerous，并要求审批；策略可配置为完全禁止危险操作。

CLI 使用同步 `input()` 审批。Web 使用 `WebPermissionController`，将同步权限决策转换为“挂起 -> SSE 请求 -> API 恢复”的状态机。

### 7.3 仍需改进的安全问题

- 路径判断基于 `abspath/commonpath`，未显式处理符号链接逃逸；生产环境应使用 `realpath` 后再验证。
- 工具本身若被绕过 PermissionManager 直接调用，路径控制无法保证；应把授权放在不可绕过的执行边界。
- Shell 黑名单不足以防命令注入。
- 没有用户身份认证、RBAC、操作审计、速率限制或多租户隔离。

## 8. 上下文管理和记忆

### 8.1 Prompt 分层

每次模型调用通过 `build_context()` 构建。大致顺序：

```text
System Prompt
AGENT.md 项目规则
Todo 状态
Planning 状态
Scratchpad
相关 LongMemory
RAG references
聊天历史（包括当前用户消息）
```

重点是把不同职责分层：项目规则不是聊天历史，长期记忆不是会话历史，工具 Observation 不是最终用户可见对话。

### 8.2 ProjectContext

`ProjectContext` 启动时读取 AGENT.md，并包装成 system prompt 部分。它让 Agent 了解当前仓库的构建方式、代码约束、工具策略和架构规则，相当于代码 Agent 的 repository-level instruction。

### 8.3 ShortMemory

短期记忆保存原始 `Message`：

- 消息数超过最大长度，从最旧消息删除；
- token 估算超过阈值，也从最旧消息删除；
- Web 恢复 session 时只把用户和最终 assistant 内容重建为短期记忆，不恢复 tool protocol 消息。

风险：从头裁剪若切断了一个 tool_calls/tool response 批次，发送给 LLM 会破坏 Function Calling 协议。Runtime 专门过滤孤立 tool 消息和不完整 assistant 工具调用批次。

### 8.4 LongMemory

长期记忆用结构化 `MemoryItem` 保存：

- `id`、`type`、`content`、`tags`、`source`；
- `importance`、创建/更新时间；
- `embedding`、`content_hash`、`superseded_by`、metadata。

记忆类型包括事实、摘要、决策、文件变更、测试结果、错误、用户偏好。写入时先做 SHA-256 规范化文本哈希去重，再用 embedding 余弦相似度去重。相似记录会合并重要性和标签。

检索使用加权分数：

```text
score = 0.68 * cosine_similarity
      + 0.18 * term_overlap
      + 0.10 * importance
      + 0.04 * recency
```

启用 RAG 时复用 Retriever 的 BGE embedder；未启用时用本地稀疏词频向量，避免为了长期记忆强制加载大模型。

### 8.5 Context Compression

当短期上下文消息数超过保留窗口且 token 超过阈值时：

1. 保留最近 12 条消息；
2. 旧消息优先交给 LLM 摘要；
3. LLM 失败则本地按用户目标、工具调用、文件、错误、测试、结论抽取摘要；
4. 摘要变成一条带 metadata 的 system Message；
5. 如果压缩后不够小，再二次裁短摘要。

摘要要求保留“用户目标、读取/修改文件、工具结果、错误、测试结论、未完成事项”，避免只留下泛泛对话而丢失代码任务状态。

### 8.6 Scratchpad 与隐藏推理的边界

Scratchpad 只存可验证且可展示的任务中间状态：目标、事实、文件、尝试、阻塞、下一步。它设置每个分区最大条数和单项长度，并做去重。

这不是模型的完整思维链。面试中可强调：存储隐藏推理既不必要，也容易增加隐私、可解释性和 prompt 膨胀问题；任务状态应该是简短、事实化、可审查的。

## 9. RAG：辅助能力而不是代码理解主路径

### 9.1 为什么 Tool-first

对于代码任务，`ls -> grep -> read_file` 能提供精确的文件、行号和最新源码；RAG 分块会损失结构关系，也可能召回旧片段。因此项目明确把 RAG 定位为文档/知识检索辅助，代码理解优先走项目工具。

### 9.2 TextSplitter

分块器使用递归分割策略：优先按双换行、换行、中文句末标点、英文句末标点、空格分割；超长段落继续使用更细分隔符；仍然过长时按固定长度切分，并尽量在标点附近断开。

合并阶段维护 chunk overlap，默认 `chunk_size=600`、`chunk_overlap=100`，缓解语义在分块边界丢失。

### 9.3 Embedding 和 Retriever

- `Embedding` 使用 BGE-M3，模型延迟加载，避免未启用 RAG 时占用内存和下载时间。
- 文档入库：切分 -> 批量向量化 -> 在内存中保存 chunks、embeddings、原文。
- 查询：query 向量化 -> NumPy 计算余弦相似度 -> `argsort` 取 Top-K。

当前限制：向量全部在内存，进程退出后不会持久化；无 metadata filter、hybrid search、reranker 和增量索引。

## 10. LLM Client 与成本观测

`DeepSeekClient` 的职责：

- 从 `.env` 读取 `DEEPSEEK_API_KEY`；
- 构建 OpenAI 兼容请求体；
- 可选传 temperature、top_p、max_tokens；
- 规范化工具声明为 `type=function` 格式；
- 解析 text 与多个 tool calls，JSON 参数解析失败时回退空/安全参数；
- 对网络/API 错误重试；
- 记录 requests、prompt/completion/total tokens。

服务端默认从环境变量读取：

- `DEEPSEEK_MODEL`，默认 `deepseek-chat`；
- `DEEPSEEK_TIMEOUT`，默认 30 秒；
- `DEEPSEEK_RETRY_TIMES`，默认 1 次。

优先使用 API 返回的真实 usage；若没有 usage，用请求 payload 与输出文本的估算 token 做兜底。CLI 退出和 Web 服务统计均会展示消耗。

## 11. 会话隔离、持久化和服务端

### 11.1 AgentSessionManager

每个 Web session 都有独立的：

- `AgentRuntime`；
- ShortMemory、LongMemory；
- TodoList、TodoStore；
- PlanningManager、Scratchpad；
- ToolRegistry；
- PermissionManager/WebPermissionController；
- `history.json`、`memory_long.json`、`todo.json`。

文件目录约为：

```text
.agent_sessions/<session_id>/
  history.json
  memory_long.json
  todo.json
```

这避免一个用户的记忆、Todo、待审批工具调用污染另一个会话。

### 11.2 可见历史和 Runtime 历史分离

`AgentSession.record_event()` 只把 `user_message`、`final` 和 `error` 事件写入历史文件。工具调用、工具结果、system prompt 不会进入用户可见聊天记录。

会话恢复时从磁盘加载可见 user/assistant 历史，并重建 ShortMemory。若历史文件为空但 Runtime 里还有可见消息，snapshot 会做回填，防止会话列表出现空白。

### 11.3 Service 层和 FastAPI 薄适配

`server.service.AgentWebService` 不依赖 FastAPI，负责：

- 创建、列举、获取、删除 session；
- 校验消息内容；
- 调 Runtime 的 `run_events()` 和 `resume_events()`；
- 将事件编码为 SSE；
- 处理 session/权限请求不存在等领域错误。

FastAPI 层只做路由、Pydantic 校验、HTTP 状态码转换和 `StreamingResponse`：

```text
GET    /health
POST   /sessions
GET    /sessions
GET    /sessions/{session_id}
DELETE /sessions/{session_id}
POST   /sessions/{session_id}/messages
GET    /sessions/{session_id}/pending-permission
POST   /sessions/{session_id}/permissions/{request_id}
```

这是典型的 Hexagonal/Clean Architecture 倾向：核心业务可脱离 HTTP 单测，FastAPI 只作为输入输出适配器。

### 11.4 SSE 事件模型

事件类型包括：

- `user_message`
- `assistant_text`
- `tool_call`
- `tool_result`
- `planning_update`
- `todo_update`
- `permission_request`
- `final`
- `error`
- `done`

每个事件有 `type`、`data`、`session_id`、`timestamp`，服务端编码为：

```text
event: tool_result
data: {"type":"tool_result","data":{...},"session_id":"..."}
```

注意：当前主链路是“结构化阶段事件流”，不是 LLM token delta 级真正逐 token 流式输出。`stream_generate()` 存在，但 Function Calling/权限主路径使用 `run_events()`。

## 12. Runtime 主导的项目分析

项目分析是一个独立模式，不是让模型自主决定看哪些文件。

### 12.1 动机

若直接让模型回答“总结整个项目”，模型可能只读 README 或少量文件就过早结束；也可能因为工具选择不稳定而遗漏权限、测试、会话等关键模块。

### 12.2 AnalysisSession 的组成

`AnalysisSession/ProjectExplorer` 管理：

- `Checklist`：每个分析目标的完成状态、证据数、是否必需；
- `SearchPlanner`：给定顺序与每个目标的默认工具调用；
- `EvidenceStore`：保存路径、工具、参数、内容片段、符号；
- `CoverageTracker`：记录访问模块、文件、类/函数；
- `analysis_notes` 和 `tool_results`：运行时补充信息。

默认重点目标包括 Architecture、CLI、Prompt、Runtime、Planning、Memory、Context、Permission、Tool、Server、RAG、Tests、Deployment。项目分析只暴露 `ls`、`grep`、`read_file`，禁止写文件、执行命令和联网搜索。

### 12.3 Final Gate 与本地兜底

Runtime 按 SearchPlanner 主动收集证据，checklist 未完成时不允许模型 Final。最终模型请求使用精简的 Analysis State + Evidence Digest，而非重新塞入整个聊天和工具上下文，减少超时风险。

最终答案必须有 Evidence 标识和具体文件/模块引用；模型最终总结失败、超时或没有引用证据时，Runtime 通过 EvidenceStore 生成本地兜底分析。这个设计把“完整性控制”从 LLM 迁到确定性的 Runtime。

## 13. CLI 能力

CLI 支持：

```text
python main.py
python main.py --tools
python main.py --tools --rag
```

命令包括：

- `/exit`：退出并输出 token 使用统计；
- `/clear`：清短期记忆、Todo 和 Planning；
- `/todo show|add|start|done|block|cancel|reset|remove|clear`：人工维护任务。

`build_runtime()` 负责在 CLI 下装配 ToolRegistry、PermissionManager、RAG、ProjectContext、LongMemory、Compressor 和 PlanningManager。

## 14. 测试策略和当前验证结果

测试分为：

- `test_tools.py`：BaseTool、Registry、文件读写、行范围截断、根目录解析、编辑 dry-run、Todo 工具；
- `test_stage2.py`：Todo、ProjectContext、Scratchpad、Context、Compression、LongMemory、Planning；
- `test_runtime.py`：工具循环、多个 tool calls、非法参数、协议残片过滤、伪工具文本、工具选择、项目分析、RAG/LLM 基础；
- `test_stage3_prep.py`：事件流、Web 权限暂停恢复、批调用恢复、session 隔离；
- `test_stage3_server.py`：SSE、服务层、FastAPI 路由、历史持久化恢复、异常处理。

实际验证命令：

```bash
venv/bin/python -m pytest -q
```

结果：`87 passed`，耗时约 1.14 秒。

有两条 warning：

1. FastAPI/Starlette TestClient 的依赖弃用 warning。
2. `test_tools.py::test_registry` 返回了 `ToolRegistry` 类，而 pytest 测试函数按规范应返回 `None`。

这些不是当前功能失败，但可作为代码质量收尾项处理。

## 15. 当前不足和演进方向

以下内容应主动说清，体现工程判断力：

### 15.1 工程化和部署

- 项目 README 说明依赖，但仓库没有 `requirements.txt`、`pyproject.toml`、锁文件或 Dockerfile，环境可复现性不足。
- `.gitignore` 忽略了 `test_*.py`，虽然当前测试文件已经存在于工作区，但后续新增测试可能被意外忽略，规则需要调整。
- 没有 CI/CD、格式化/lint/type-check、覆盖率阈值、发布流程或容器化部署。

改进：引入 `pyproject.toml`、uv/pip-tools/Poetry 锁依赖、ruff、mypy、pre-commit、GitHub Actions、Docker 和健康检查/结构化日志。

### 15.2 并发和可靠性

- session Manager 使用进程内字典，没有跨进程共享、锁、超时回收或分布式任务队列。
- JSON 持久化没有原子写入、文件锁和迁移机制，多进程写同一 session 有数据竞争风险。
- LLM/工具调用为同步阻塞式，网络慢或工具慢会占用 worker。
- 权限 pending 没有过期时间、取消和崩溃恢复策略。

改进：Redis/数据库保存会话状态，任务转到队列或 sandbox worker，使用 async HTTP client，增加 timeout/cancellation/idempotency key 和状态版本号。

### 15.3 模型和检索质量

- 意图与 Planning 基于关键词启发式，面对复杂自然语言会误判，后续可引入小模型分类器、结构化 LLM router 或可观测的策略配置。
- RAG 没有持久化向量索引、BM25 混合召回、重排序、metadata filter、文档版本管理和评测集。
- 上下文压缩默认保留关键字段，但摘要仍可能丢失细粒度代码事实；应加入任务状态的结构化摘要和可追溯 source 引用。
- token 估算不精确，应接入模型 tokenizer 或依赖 API usage 做闭环限额。

### 15.4 安全

- Shell `shell=True` 和黑名单不适合生产。
- 缺少认证、授权、审计、敏感信息脱敏、请求限流、网络 egress 控制。
- 文件路径策略应处理 symlink，且工具执行边界必须强制权限校验。

改进：容器/Firecracker 隔离，最小权限挂载、无网络默认、命令 allowlist、资源 quota、审计事件和策略引擎。

## 16. 高频面试问答

### Q1：为什么不每轮都把所有工具交给 LLM？

因为工具 schema 会占 prompt token，模型也会倾向过度调用工具。普通概念题若暴露 `read_file`，模型可能无意义读文件，增加成本、延迟和失败面。项目根据意图只暴露最小工具集，实时信息只给搜索，项目分析只给只读工具，符合最小权限和最小上下文原则。

### Q2：ReAct 和 Planning + ReAct 的区别？

ReAct 适合短链路任务，模型根据上一步 Observation 决定下一步工具；Planning + ReAct 适合跨文件、多步骤、需要验证的任务，先维护显式步骤和 Todo，再由 Observation 动态重规划。Planning 提升可控性和可见性，但会增加延迟与上下文，因此不适合简单问答。

### Q3：为什么还要 LongMemory，不能只保存聊天历史吗？

聊天历史用于恢复用户可见对话，包含时间顺序；LongMemory 用结构化事实、摘要、错误、测试结果和语义检索，为当前任务按相关性提取少量知识。两者目的不同，若把历史全部塞 Prompt，会导致上下文膨胀和无关信息干扰。

### Q4：上下文压缩会不会丢信息？

一定存在信息损失，所以策略不是简单截断：保留最近消息；旧消息摘要要求保留目标、文件、工具结果、错误、测试、未完成项；关键长期事实独立存储；工具协议完整性由 Runtime 过滤保障。更进一步可用结构化状态和源文件引用降低损失。

### Q5：如何保证多工具调用时协议正确？

assistant 的一个 tool_calls batch 必须完整对应 tool results。Runtime 保存批次和 index，权限挂起后从原位置恢复并补全剩余所有 tool message，随后才发下一次 LLM 请求。这是 Function Calling 状态机最容易出错的地方。

### Q6：SSE 相比 WebSocket 为什么合适？

当前交互主要是服务端向浏览器单向持续推送 Agent 生命周期事件，客户端发送消息/审批仍可走普通 POST。SSE 基于 HTTP、实现简单、浏览器原生支持重连语义，适合本项目。若需要双向协作、客户端实时中断、多人共享状态，可考虑 WebSocket。

### Q7：RAG 为什么不能替代代码工具？

RAG 是语义近似召回，分块会割裂函数定义、调用关系和最新工作区状态；代码修改前需要精确文件和行号。工具读取是确定性的，RAG 更适合文档、规范和知识库辅助。

### Q8：本项目最难的部分是什么？

不是调用一次模型 API，而是 Runtime 工作流控制：模型协议、工具批处理、权限暂停恢复、普通问答与工具意图区分、上下文膨胀、会话持久化和项目分析完整性都会相互影响。项目把这些问题分别放到 Runtime、Permission、Context、Session、AnalysisSession 中处理，降低单一 prompt 的不可控性。

## 17. 两分钟项目讲稿

> 这个项目是我实现的本地 Coder Agent。架构上我把 CLI 和 FastAPI 都放在适配层，核心是独立的 AgentRuntime。Runtime 会先判断用户请求是普通问答、代码执行任务、复杂多步骤任务还是项目分析；普通问题直接回答，代码任务使用 DeepSeek 的 Function Calling 驱动 ReAct 工具循环，复杂任务额外维护 Planning 和 Todo。\n> \n> 我重点解决了 Function Calling 的多工具批处理和权限恢复问题。模型一次可能返回多个 tool calls，而协议要求下一轮模型调用前必须补齐所有 tool result。对于需要用户确认的写文件或 Shell 操作，我会挂起 Runtime、通过 SSE 发权限请求，审批后从原工具批次继续，确保协议完整。\n> \n> 上下文方面，我把项目规则、短期记忆、结构化长期记忆、压缩摘要、Scratchpad、计划和 RAG 分层处理。长期记忆不是简单聊天记录，而是可以按相关性检索的 MemoryItem。Web 多会话通过 SessionManager 隔离，每个会话独立保存 history、todo 和长期记忆。\n> \n> 另外项目分析功能由 Runtime 主导，不依赖模型随意看文件。它维护 checklist、搜索计划、证据仓和覆盖记录，只用只读工具收集证据，最终回答必须引用具体模块，模型失败时会基于证据本地兜底。当前全量 pytest 是 87 个用例通过。后续我会优先补齐容器化、安全 sandbox、数据库/Redis 会话、异步执行和生产级 RAG。


---

# 源码级关键实现说明

本章按实际函数调用顺序说明核心模块的实现，而不是罗列功能。

## 18. AgentRuntime 的完整请求生命周期

Web/API 的主入口是 `AgentRuntime.run_events(user_input)`；CLI 的 `run()` 是无事件版本。Runtime 的运行期状态包括：

```text
llm                        DeepSeekClient 或可替换 LLM Client
memory                     ShortMemory
long_memory                LongMemory
tool_registry              ToolRegistry
permission_fn              CLI 审批函数或 WebPermissionController.approve
planning_manager           PlanningManager 与当前 PlanState
todo_list                  TodoList
context_compressor         ContextCompressor
scratchpad                 当前任务的显式工作状态
session_id                 SSE 事件归属
_pending_*                 权限暂停时保存的可恢复工具批次
```

一次 `run_events()` 的步骤：

1. 若 `_pending_tool_call` 非空，说明同一 session 上一次正在等待审批，直接返回 error/done，防止两个请求改写同一上下文。
2. 创建 `Message(role="user", content=user_input)`，调用 `ShortMemory.add_message()`；再由 `Scratchpad.set_objective()` 记录本轮目标。
3. 若满足 ContextCompressor 的条件，先压缩旧消息。
4. 调 `PlanningManager.start_or_update_plan()`。若新计划产生，Runtime 发 `planning_update` 和 `todo_update` SSE，并把计划步骤合并入 Scratchpad 的 next steps。
5. `_should_expose_tools()` 决定走纯问答、Native Function Calling、只读项目分析或实时查询。
6. 普通问答进入 `_run_with_callback_tool()`；工具任务进入 `_run_native_tools_events()`；项目总结进入 Runtime 主导的 `ProjectExplorer`。
7. 正常结束发 final/done；异常记录日志，只发简短 error/done，避免向客户端暴露内部 traceback。

隐式状态机为：

```text
IDLE -> RECEIVED -> CONTEXT_PREPARED
     -> PLAIN_ANSWER | TOOL_LOOP | PROJECT_ANALYSIS
     -> WAITING_PERMISSION（可选）
     -> FINAL | ERROR -> DONE
```

当前状态由生成器、Memory 和 `_pending_*` 字段表达，未抽为显式 Enum。要支持跨进程恢复、取消和并发，应进一步将迁移状态版本化持久化。

## 19. 简单任务、复杂任务、项目分析、实时查询如何分流

是否使用工具与是否启动 Planning 是两个独立决定：工具用于获取信息或执行动作，Planning 用于管理多步骤状态。

### 19.1 是否暴露工具 schema

`_should_expose_tools()` 的优先级：

```text
无 ToolRegistry 或工具数为 0                         -> False
显式点名已注册工具                                   -> True
天气等实时外部信息，含时间/地点/查询意图             -> True
“查吧/搜一下”且最近上下文确实是实时检索             -> True
项目、目录、文件的只读分析                           -> 只读工具可用时 True
直接问答意图明显且没有明确执行请求                   -> False
行动词，或软行动词 + 文件/项目资源提示               -> True
```

行动词包括修复、实现、修改、测试、运行、读取、搜索、调试；直接问答词包括是什么、为什么、如何、介绍、区别、是否、状态如何、存在哪里。

问题“长期记忆存在哪里”是在问机制，不应因“存/哪里/打开”等词触发读文件。`_is_explicit_tool_request()` 专门识别“位置提问 + 记忆、会话、上下文、Planning、Todo”等内部机制词。

`contains_keyword()` 对英文使用词边界正则：`test` 不会匹配 `latest`，`run` 不会匹配 `runtime`。这是关键词路由中避免 substring 误判的实现细节。

### 19.2 最小工具集

需要工具也不会把所有工具 schema 一并暴露：

- 实时信息仅给 `web_search`；
- 项目架构/总结仅给 `ls`、`grep`、`read_file`；
- 运行、测试、调试、报错、Python/Shell 才给读、检索、编辑、写入、执行等组合；
- 一般工具请求当前可能得到完整 Registry，这是可继续收紧的策略点。

这样减少 schema token、降低模型过度调用工具，并符合最小权限原则。用户查天气时不应同时获得 Shell。

### 19.3 是否启动 Planning

`PlanningManager.should_plan()` 的规则：空输入不规划；必须有行动词；纯解释问题不规划；有行动词后，输入长度达到默认 32 字符即规划，短输入还要带“全部、阶段、复杂、多步、多个、一系列、feature”等复杂性提示。

| 请求 | 工具 | Planning | 实际路径 |
| --- | --- | --- | --- |
| 什么是 RAG | 否 | 否 | 普通问答 |
| 读取 README 并说明安装 | 是 | 通常否 | 短 ReAct |
| 修复登录失败，补测试并验证部署 | 是 | 是 | Planning + ReAct |
| 完整分析仓库架构和难点 | 是，只读 | 否 | ProjectExplorer |
| 查询今天上海天气 | 是，仅搜索 | 否 | 实时查询 |

### 19.4 普通问答的真实路径

`_run_with_callback_tool()` 不会直接把原始 memory 发给模型：

1. 从 ShortMemory 获取消息；
2. `_plain_chat_history()` 删除所有 tool 消息和带 `function_call(s)` metadata 的 assistant 消息；
3. 用最后用户输入查询 RAG 和 LongMemory；
4. 获取 Planning、Todo、Scratchpad、ProjectContext；
5. 调 `build_context()`；
6. 追加 system message，明确本轮没有 tools，禁止输出 `tool_calls`、`read_file`、DSML 等伪协议；
7. 调 `llm.generate(messages=context)`，注意这里不传 tools；
8. 清理模型回复。若仍是伪工具草稿且退化为占位回答，最多严格自然语言重试一次；
9. `_finalize_response()` 保存最终 assistant 消息，并按轮次阈值写长期摘要。

之所以删除旧工具协议，是因为上一轮的 assistant tool_calls/tool results 若带到普通追问，模型可能误以为仍处于工具循环，继续输出工具草稿或形成不合法 API 消息序列。

### 19.5 复杂任务的真实路径

```text
复杂请求
  -> should_plan=True
  -> start_or_update_plan()
  -> fallback_steps() 生成计划
  -> TodoList 同步步骤
  -> 同一个 Native Function Calling ReAct Loop
  -> 每个 Observation 调 observe_tool_result()
  -> 错误/权限拒绝/编辑代码时重规划
  -> 新计划立即注入下一轮模型上下文
```

fallback 步骤固定有“分析现状 -> 定位/设计 -> 修改 -> 验证 -> 收尾”骨架。输入含 bug/修复/错误时第二步为“复现或定位根因”，否则为“设计实现方案并确定修改范围”。

默认 `enable_llm_planning=False`，因为计划本身也会增加一次 API 成本和不确定性。代码保留 LLM 规划接口，未来可按预算开启。

重规划由 `observe_tool_result()` 完成：Observation 含错误、失败、Traceback 或权限拒绝时，追加“根据最新错误调整实现方案”和“重新运行验证确认问题已解决”；`write_file`/`edit_file` 后追加“运行测试或静态检查验证修改”。PlanState revision 增加，更新后的计划在下一轮以 system Message 注入。

## 20. Native Function Calling 与工具调用如何实现

### 20.1 从工具类到模型 schema

每个工具继承 `BaseTool` 并定义 name、description、JSON Schema parameters、risk_level、execute。`ToolRegistry.register()` 检查 BaseTool 类型和 name；同名工具覆盖会 warning。Runtime 获取 declarations，DeepSeekClient 转换为 OpenAI 兼容格式：

```json
{"type":"function","function":{"name":"edit_file","description":"...","parameters":{...}}}
```

tools 非空时请求设置 `tool_choice: "auto"`。例如 `edit_file` schema 强制 path、start_line、end_line、new_content，可选 dry_run，避免从自然语言猜参数。

### 20.2 模型响应解析

DeepSeekClient 从 `choices[0].message` 读取 content/tool_calls，每条调用统一为：

```python
{"id": "call_xxx", "name": "read_file", "args": {"path": "core/runtime/agent_runtime.py"}}
```

arguments 原本是 JSON 字符串，client 用 `json.loads()`；失败则 warning 并回退空 dict。Runtime 通过 `_ensure_function_call_ids()` 给缺失 ID 的调用生成 `call_r<轮次>_<序号>`，确保结果能回填 tool_call_id。

### 20.3 一批多个工具的标准路径

```text
1. 收到 function_calls=[call_1, call_2]
2. 创建 assistant Message，metadata.function_calls 保存完整 batch
3. 写入 ShortMemory 与当前 LLM context
4. 顺序处理每个 call：
   - 发 TOOL_CALL SSE
   - 权限检查
   - Registry required 参数校验和具体工具执行
   - 结果全局截断至 8000 字符
   - 创建 role=tool Message，function_response.id=call_i.id
   - 写入 Memory/context
   - 发 TOOL_RESULT SSE
   - 更新 Scratchpad 与 Planning
5. batch 完成后追加新的 Scratchpad/Plan system Message
6. 再请求 LLM，决定继续工具或 Final
```

关键约束：同一条 assistant tool_calls 的每一个 id，必须在下一次模型请求前有且仅有一条对应 tool Message。

### 20.4 错误也是 Observation

Registry 执行前调用 `tool.validate_params(**kwargs)`；未知工具、args 非 dict、缺 required 参数、具体工具异常都会变成 `[错误] ...` 的合法 role=tool Message，而不是终止 Agent。模型可修正参数、换工具或解释阻塞。

当前 BaseTool 只做 required 校验，未完整校验 JSON Schema 类型、枚举和额外字段；生产环境应接入 Pydantic 或 jsonschema。

### 20.5 权限在何处执行

`_execute_tool(name, args, permission_override)` 的顺序：

```text
args 非 dict -> 参数错误 Observation
permission_override=False -> 权限拒绝 Observation
存在 permission_fn 且无 override -> 请求权限
  同意 -> Registry.execute
  拒绝 -> 权限拒绝 Observation
  PermissionPending -> 抛出，上层暂停 batch
无权限问题 -> Registry.execute
```

权限被放在 Runtime 的统一执行边界。文件工具的系统路径拦截属于额外防线，不能代替统一授权策略。

### 20.6 Web 权限暂停和恢复

`PermissionPending` 出现时，`_store_pending_tool_batch()` 保存：当前 call、完整 function_calls、当前 index、LLM context、tool schemas、轮次、缓存文本、计划变化。随后发 `permission_request` SSE。

审批后 `resume_events()`：取出并清空 `_pending_*` 防止重复恢复；从保存 index 继续批处理；第一条待恢复调用使用 API 决策作为 override；补齐当前及后续所有 tool Messages；只有完整 batch 完成后，才重新调用模型。

拒绝同样生成相同 call id 的 tool result，内容为权限拒绝。这样既保持协议合法，也让模型知道应换方案或说明阻塞。

### 20.7 输出控制和具体工具算法

输出有两层限制：read_file 未给范围时只展示开头窗口、行号和继续读取提示；Runtime 对任意工具结果再截断到 8000 字符。第一层让代码按范围渐进读取，第二层保护 Shell/grep 输出不撑爆 context。

- **FileReadTool**：相对路径拼 workspace root，UTF-8 读取，渲染为 `行号 | 内容`。
- **FileWriteTool**：拒绝 `/dev`、`/proc`、`/sys`、`/boot`，自动建父目录，按 append 选 `a`/`w`。
- **EditTool**：读取为行数组，校验范围后执行 `lines[start_line - 1:end_line] = replacement`，用 `difflib.unified_diff` 生成 diff；dry_run 只预览，写回前自动补末尾换行。
- **GrepTool**：编译用户正则，跳过 `.git`、venv、node_modules、build，读前 1024 字节用 NUL 判断二进制，逐行匹配并返回 `相对路径:行号:文本`，最多 1000 条。

## 21. 记忆与上下文如何协作

### 21.1 Context 分层和消息顺序

Code Agent 同时拥有用户目标、项目规则、文件内容、工具返回、计划、待办、记忆、RAG 文档和历史回答。若无差别拼接，会造成上下文超限、无关 Observation 掩盖目标、协议残片非法和伪工具文本。

`build_context()` 顺序：

```text
1. system_prompt
2. project_context（AGENT.md）
3. todo_context
4. planning_context
5. scratchpad_context
6. long_memory_context
7. rag_results
8. tool_results（仅 legacy callback path）
9. history（含当前用户消息）
```

规则和任务状态使用 system role；RAG/长期记忆是参考；用户消息在 history 尾部。Native Tool Path 的 Observation 是逐条真实 role=tool Message，保证 Function Calling 协议。

### 21.2 ProjectContext

`ProjectContext` 初始化即读取 AGENT.md，包装为项目规则 system Message。它类似 CLAUDE.md/AGENTS.md，让模型每轮知道仓库规范。当前直接注入整份文件，无独立 token 上限；规则过大时应做分段、优先级和摘要缓存。

### 21.3 ShortMemory 和协议安全历史

ShortMemory 是 List[Message]。每次 add 后先按 max_length 从头删除，再按估算 token 从头删除。Runtime 默认新建时为 60 条/100000 token；Web Session 传 `ShortMemory(max_length=60)`，但 token 上限沿用默认 4000。

一个真实实现观察：ContextCompressor 阈值为 50000 token，Web Session 常在 4000 token 已硬裁剪，因此 Web 通常触发不到压缩器；CLI Runtime 才可能触发。若要让 Web 使用摘要而不是硬裁剪，需协调两个阈值。

短期裁剪会留下孤立 tool 或不完整 tool_calls batch。`_tool_protocol_safe_history()` 会跳过孤立 tool；对 assistant function_calls 收集紧邻 tool messages；验证每个 expected id 恰好一次；只有完整 batch 才保留，不完整 batch 整体丢弃。这是防止 API 协议错误的关键。

### 21.4 LongMemory 的写入和去重

`MemoryItem` 包含 id、type、content、tags、source、importance、时间、embedding、content_hash、superseded_by、metadata。类型包括 fact、summary、decision、file_change、test_result、error、user_preference。

`add_item()` 的过程：

1. 去掉空内容，规范化 tags/importance；
2. 优先用注入 RAG embedder 的 `encode_query()`，失败回退本地稀疏词频向量；
3. 对规范化文本计算 SHA-256；
4. 同类型 active item 先 hash 去重；
5. 再以 cosine 相似度阈值 0.88 做语义去重；
6. 重复项合并 importance、tags、metadata；不重复项 append；
7. 超过 300 项后按重要性和更新时间裁剪；
8. JSON 持久化，并兼容旧 facts/summaries 格式。

### 21.5 LongMemory 的检索

Runtime query = 最后用户消息 + Planning objective + 最近最多 3 个计划步骤，避免“继续/再试一次”这种短续问语义不足。

```text
score = 0.68 * cosine_similarity
      + 0.18 * term_overlap
      + 0.10 * importance
      + 0.04 * recency
```

按分数取 Top-K，格式化为带 type、score、tags 的 system reference。它是轻量混合检索；本地中文分词只是英文标识符和单汉字正则，复杂中文语义有限。

### 21.6 自动摘要、压缩和 Scratchpad

Runtime 默认每 8 轮在 `_finalize_response()` 后把对话摘要写入 LongMemory，优先保留用户目标、工具、文件、错误、测试和结论。

ContextCompressor 要求消息数超过 12 且 token 超过 50000：保留最近 12 条；旧消息优先让 LLM 生成工作摘要；失败时本地扫描；摘要作为带 `context_compression` metadata 的 system Message；若仍不够小再裁短；随后替换 ShortMemory 并写 LongMemory。

每次工具完成后 Scratchpad 更新：文件读写编辑进入 files；ls/grep 进 facts；所有调用进 attempts；错误进 blockers 并加入修复/验证 next step；写、编辑、执行代码必加验证 next step。每分区最多 8 项、每项最多 220 字符、去重。它存可展示事实，不存隐藏思维链。

## 22. 复杂任务端到端示例

用户：`请修复用户登录失败的问题，补充单元测试，运行测试并说明根因。`

1. 写用户消息和 Scratchpad objective。
2. `should_plan=True`，生成“分析 -> 定位 -> 修改 -> 验证 -> 收尾”，同步 Todo，推 planning/todo SSE。
3. 模型返回 grep + read_file；Runtime 顺序处理，写入完整 tool batch。
4. 模型调用 execute_code 跑测试，返回 AssertionError；Planning 增加 revision，补“调整方案”“重新验证”；Scratchpad 记录 blocker。
5. 模型调用 edit_file；权限层挂起，客户端批准后 Runtime 从 batch index 恢复，而不是重新询问模型。
6. edit diff 成为 Observation，Planning 自动补验证；模型再运行测试。
7. 模型返回结论，Runtime 保存 final、可能自动摘要，Session 写 history.json，SSE 发 final/done。

当前 Runtime 不会自动严谨地把每个 Todo 标为 start/done；模型可调用 update_todo，用户也可用 CLI 更新。这是它与完整 workflow engine 的差异。

## 23. 准确评价与下一步

亮点：工具选择、批处理协议、权限恢复由 Runtime 而非纯 prompt 控制；复杂任务通过 PlanState/Todo/Observation 重规划；上下文是规则、计划、草稿、检索记忆、RAG、协议安全历史的分层组合；Service 与 FastAPI 解耦。

不要夸大：项目是单进程 JSON 持久化的本地 Agent，不是生产多租户平台；Planning 是启发式规则；参数校验基础、Shell 隔离不足；RAG 是内存 Top-K；SSE 是事件流而非 token delta 流。

优先改进：将执行状态与 pending permission 版本化持久化；以 sandbox worker 替代 `shell=True`；协调 Web Memory/Compression 阈值；接入严格 Pydantic/JSON Schema；以数据库/Redis 替换 JSON 并补认证审计；为 RAG 增加持久化索引、混合检索、reranker 与评测集。

---


---

# 基于 `liuup/claude-code-analysis` 的具体借鉴对照

> 本节已按用户指定仓库中的分析文档重写。对照依据是该仓库对 2026 年 Claude Code TypeScript 源码的静态分析，重点参考：
>
> - `analysis/04-agent-memory.md`
> - `analysis/04b-tool-call-implementation.md`
> - `analysis/04f-context-management.md`
> - `analysis/04g-prompt-management.md`
> - `analysis/04i-session-storage-resume.md`
> - `analysis/04e-sandbox-implementation.md`
> - `analysis/04c-skills-implementation.md`
> - `analysis/04d-mcp-implementation.md`
> - `analysis/04h-multi-agent.md`
>
> 下面的表述严格区分“本项目已有的同类实现”、“本项目只借鉴到设计方向”和“Claude Code 有但本项目没有”。不能把后两者说成本项目已经实现。

## 24. 总结结论：本项目借鉴的是 Claude Code 的 Runtime 分层，不是其完整平台

从指定分析仓库的源码链路看，Claude Code 的核心不是一个 `LLM -> Bash` 脚本，而是：`Query/QueryEngine` 执行内核、Tool Pipeline、Permission、Transcript/Memory、Prompt Runtime、Sandbox、MCP/Skills、多 Agent 等共同组成的平台。

本项目最直接借鉴并实现的部分是：

```text
Claude Code: Query / Tool Pipeline / Tool Result 回流
本项目：     AgentRuntime / ToolRegistry / Message(function_call, function_response)

Claude Code: Prompt Runtime 中的 project instructions、memory、scratchpad、动态上下文
本项目：     build_context() 中的 AGENT.md、LongMemory、Scratchpad、Planning、Todo、RAG

Claude Code: 会话、摘要、工具协议的状态治理
本项目：     AgentSession、ShortMemory、LongMemory、ContextCompressor、protocol-safe history

Claude Code: Permission 决策不由模型直接决定
本项目：     PermissionManager + WebPermissionController + resume_events()
```

但本项目没有实现 Claude Code 源码分析中出现的 Sandbox Runtime、append-only JSONL transcript/sidechain、memory file index、Skills/MCP、多 Agent coordinator/swarm、工具并发调度等平台能力。

## 25. 最具体的借鉴一：从 `Query -> tool_use -> tool_result` Pipeline 到 `AgentRuntime`

### 25.1 Claude Code 源码分析中的机制

指定仓库 `analysis/04b-tool-call-implementation.md` 将 Claude Code 的工具链描述为：

```text
assistant message 的 tool_use blocks
  -> query.ts 收集
  -> Streaming executor 或 runTools()
  -> toolOrchestration.ts 依据并发安全性分批
  -> toolExecution.ts：Schema -> validateInput -> PreToolUse hooks
                        -> permission(allow/ask/deny) -> tool.call()
                        -> tool_result / attachment / progress
  -> normalizeMessagesForAPI()
  -> 下一轮 API 请求
```

工具不是普通函数，而是 `Tool` 协议对象。分析文档引用的 `src/Tool.ts` 还包含：输入/输出 schema、语义校验、并发安全声明、只读/破坏性声明、权限检查、UI 渲染、取消行为、进度回调等。

### 25.2 本项目具体对应实现

本项目的等价骨架在 `core/runtime/agent_runtime.py`：

```text
DeepSeekClient._parse_response()
  -> {text, function_calls}
  -> AgentRuntime._ensure_function_call_ids()
  -> assistant Message(metadata.function_calls)
  -> _process_tool_call_batch_events()
  -> _complete_tool_call()
  -> _execute_tool()
  -> ToolRegistry.execute()
  -> tool Message(metadata.function_response)
  -> 下一轮 llm.generate()
```

具体映射如下：

| Claude Code 分析中的节点 | 本项目对应代码 | 实际行为 |
| --- | --- | --- |
| `tool_use` block | `DeepSeekClient._parse_response()` | 把 API 的 `tool_calls` 转为 `id/name/args` |
| Tool 协议 | `BaseTool` + `ToolRegistry` | 统一 name、description、JSON Schema、risk、execute |
| permission/ask/deny | `PermissionManager`、`WebPermissionController` | 允许、拒绝或挂起等待审批 |
| `tool_result` 回流 | `Message(role="tool", metadata.function_response)` | 按 call id 回填模型协议 |
| Query 下一轮 | `_continue_native_tools_events()` | 结果写入 context 后继续 generate |
| UI progress/result | `AgentEvent` | `tool_call/tool_result/planning_update/permission_request` SSE |

### 25.3 这不是泛泛的“用了工具”，而是协议层借鉴

本项目特别借鉴了“**工具调用由 Runtime 组织，模型不能直接调用宿主能力**”这一点：模型只产生结构化意图；Runtime 才做参数检查、权限、执行、错误封装、结果回流。

多工具批次完整性是本项目一个比原始简化 Agent 更严格的地方：`_process_tool_call_batch_events()` 保证 assistant 的每个 `tool_call_id` 都有对应 tool message。权限卡在 batch 中间时，`_store_pending_tool_batch()` 保存完整 calls 和当前 index，`resume_events()` 先补齐同批结果再请求模型。否则 OpenAI/DeepSeek 兼容协议会报“tool messages 不足”。

### 25.4 与 Claude Code 的明确差距

Claude Code 的工具协议更完整，本项目尚未实现：

- `inputSchema.safeParse` 之外的完整类型/枚举校验；本项目仅检查 required 字段。
- `validateInput()` 的工具级语义校验接口。
- `PreToolUse`/`PostToolUse` hooks，可修改 input 或拦截调用。
- 工具进度、attachment、多媒体结果、可取消执行。
- `isConcurrencySafe()` 和按安全性 `partitionToolCalls()` 分批并发。
- `normalizeMessagesForAPI()` 对富 UI/attachment/API 消息的复杂规范化。

因此面试中应说：本项目实现了 Claude Code Tool Pipeline 的**最小串行版**，而不是完整 Tool Runtime。

## 26. 最具体的借鉴二：项目规则不是硬编码 prompt，而是运行时上下文

### 26.1 Claude Code 源码分析中的机制

`analysis/04g-prompt-management.md` 指出 Claude Code 有一个 Prompt Runtime：

```text
getSystemPrompt()               返回多个 system prompt section
buildEffectiveSystemPrompt()    按 override/coordinator/agent/custom/default 优先级组装
getUserContext()                注入 CLAUDE.md、currentDate
getSystemContext()              注入 git status、cacheBreaker 等运行态
专项 prompt                    compact / session memory / memory extraction / hooks
```

重要细节：`getSystemPrompt()` 返回的是 `string[]`，不是单一字符串；静态 section 与动态 section 之间有 `SYSTEM_PROMPT_DYNAMIC_BOUNDARY`，用于 prompt cache 边界。`buildEffectiveSystemPrompt()` 还严格区分“覆盖默认 prompt”和“追加 append prompt”。

### 26.2 本项目具体对应实现

本项目没有 section cache 和多重 prompt 覆盖优先级，但已借鉴“上下文由多个职责明确的层在运行期装配，而不是一个巨型常量”的思路。

`core/context/builder.py::build_context()` 的固定顺序是：

```text
1. system_prompt
2. ProjectContext：AGENT.md
3. TodoList 状态
4. Planning 状态
5. Scratchpad
6. 相关 LongMemory
7. RAG 参考
8. legacy tool_results
9. 历史消息（含本轮 user）
```

`ProjectContext.reload()` 读取 `AGENT.md`，再由 `get_context_str()` 包装成项目规则。`AgentRuntime._build_context()` 在每次工具模型请求前调用 `build_context()`；普通回答路径也用它，但会追加“本轮没有工具 schema，禁止伪工具协议”的专用 system message。

### 26.3 具体借鉴点和差异

- **借鉴点**：将仓库规则从代码硬编码中移出，作为 runtime context 注入。AGENT.md 是本项目对应 CLAUDE.md 的项目规则载体。
- **关键差异**：指定分析仓库中 CLAUDE.md 由 `getUserContext()` 作为 user context 注入；本项目 AGENT.md 由 `build_context()` 作为 system Message 注入。两者在 role 与优先级上不完全相同，不能说“完全复刻 CLAUDE.md 机制”。
- **借鉴点**：将主任务 prompt 与专项 prompt 分开。本项目的普通回答纠偏、项目分析 final gate、上下文压缩摘要、Planning 状态都是单独 prompt/message，而不是全塞进初始 system prompt。
- **未实现**：section 命名、token 统计、cache boundary、`buildEffectiveSystemPrompt()` 的 override/coordinator/agent/custom/default 优先级、`/context` 级可观测性和 prompt dump。

## 27. 最具体的借鉴三：Session Memory、长期记忆和压缩必须分层

### 27.1 Claude Code 源码分析中的四层文件化 Memory

`analysis/04-agent-memory.md` 明确区分：

```text
Auto Memory       用户/项目长期记忆，memory 目录
Session Memory    当前会话摘要文件
Agent Memory      某 agent 类型绑定的长期记忆，scope=user/project/local
Team Memory       团队共享 repo 级记忆
```

其底层不是单数据库，而是目录 + `MEMORY.md` 入口索引 + 多个 topic markdown 文件。`MEMORY.md` 限制 200 行/25KB，默认注入；具体 topic 文件通过 `findRelevantMemories()` 先扫描 header，形成 manifest，再用 sideQuery 最多选 5 个。也就是说，先选“文件”，再读取正文，而不是把所有记忆直接塞 prompt。

Session Memory 还有非常具体的触发策略：初始化至少 10000 tokens；之后至少多 5000 tokens；并参考自上次更新以来的 3 次 tool call；若上一 assistant turn 仍有 tool_use，优先等自然断点。更新由后台 forked agent 做，而且 `createMemoryFileCanUseTool()` 只允许该 agent 对精确的 memoryPath 使用 `FileEditTool`，目录 0700、文件 0600。

### 27.2 本项目对应的具体实现

本项目不使用 markdown memory directory，但实现了相同的职责分层：

| Claude Code 分层 | 本项目对应 | 实现方式 |
| --- | --- | --- |
| Session transcript | `history.json` | AgentSession 仅持久化 user/final/error 可见事件 |
| 当前短期工作上下文 | `ShortMemory` | Message 列表，按数量/token 从头裁剪 |
| Session Memory 摘要 | `ContextCompressor` | 旧消息压为 system summary，保留 recent messages |
| Auto/Agent Memory 的部分职责 | `LongMemory` | `MemoryItem` JSON、type/tags/source/importance/embedding/hash |
| 当前任务工作笔记 | `Scratchpad` | objective/facts/files/attempts/blockers/next_steps |
| Team Memory | 无 | 尚未实现 |

`LongMemory.add_item()` 的具体写入链是：规范化内容 -> 生成/刷新 embedding -> SHA-256 hash 去重 -> cosine 相似度阈值 0.88 去重 -> 合并或 append -> 最多 300 项裁剪 -> 写入 JSON。

`AgentRuntime._get_long_memory_context()` 不只用最后一句用户输入做 query，还拼接当前 Planning objective 和最近 3 个计划步骤；`LongMemory.search()` 用：

```text
0.68 * cosine similarity
+ 0.18 * term overlap
+ 0.10 * importance
+ 0.04 * recency
```

进行排序，选 Top-K 注入上下文。

### 27.3 借鉴是“分层职责”，不是“实现完全相同”

本项目借鉴的是 Claude Code 将“可见 transcript、当前会话摘要、长期记忆、agent 专属记忆”拆开的思想。它没有复刻以下具体实现：

- `MEMORY.md` + topic 文件 + header manifest；
- sideQuery 选最多 5 个 memory 文件；
- user/project/local 三种 Agent Memory scope；
- Auto Memory、Team Memory、Agent Memory Snapshot；
- 使用受限 forked subagent 更新 Session Memory；
- memory 文件 0600/目录 0700 的敏感权限；
- 已展示 memory 的 `alreadySurfaced` 去重机制。

本项目当前的 LongMemory 将多个职责合并进同一 JSON store，优点是简单，缺点是 scope、透明度、文件级治理和安全隔离不如分析仓库中的设计。

## 28. 最具体的借鉴四：上下文压缩不能只截断，必须处理工具链和状态重注入

### 28.1 Claude Code 源码分析中的机制

`analysis/04f-context-management.md` 描述的 Claude Code auto-compact 包含：动态 context window、为 summary 预留 token、接近上限前 13000 token buffer 触发、连续失败 3 次的熔断器、compact request 超限时剥洋葱式降载、post-compact 状态重注入。

尤其重要的是：compact 后不能只保留摘要。分析中指出会重新注入 plan attachment、skill attachment 等状态，避免上下文缩短后丢失已加载资源和开放工具。

### 28.2 本项目对应实现

本项目 `ContextCompressor` 的触发条件是：消息数大于 `keep_recent=12` 且估算 token 大于 `threshold_tokens=50000`。它：

1. 分离 `old_messages` 与最近 12 条；
2. 优先让 LLM 输出工作摘要，要求保留目标、读/改文件、工具结果、错误、测试、未完成项；
3. LLM 失败时根据 role、function metadata、tool result 生成本地摘要；
4. 把摘要变成带 `context_compression` metadata 的 system Message；
5. 压缩后若 token 未明显降低，`_fit_summary()` 再裁短；
6. Runtime 替换 ShortMemory，并把摘要写 LongMemory。

工具协议方面，本项目额外有 `_tool_protocol_safe_history()`：ShortMemory 从头裁剪后若留下孤立 `role=tool` 或不完整 tool_calls batch，会整体丢弃该残片，只向 API 发送完整批次。这与 Claude Code “压缩后必须保持可继续运行状态”的目标一致。

### 28.3 明确差异

本项目没有模型窗口感知、真实 token 预算、summary reserve、13000 buffer、连续失败熔断、microcompact/content replacement、plan/skill attachment 重建和 prompt cache。它是“保留最近消息 + 生成摘要 + 过滤非法工具协议”的简化版本。

另外，Web Session 的 ShortMemory 传入 `max_tokens=4000`，而 Compressor 阈值是 50000，因而 Web 很可能先硬裁剪、无法触发 Compressor。这正是对照 Claude Code 后可明确提出的改进：压缩应该在硬裁剪之前被触发，并基于真实 token 使用量做调度。

## 29. 最具体的借鉴五：会话不是聊天数组，而是可恢复执行记录

### 29.1 Claude Code 源码分析中的机制

`analysis/04i-session-storage-resume.md` 说明 Claude Code 采用 append-only JSONL：主 transcript 是 `sessionId.jsonl`；subagent 是独立 sidechain JSONL；写入经 queue 批量 flush，目录 0700/文件 0600；主链按 UUID 去重，sidechain 允许父上下文重复以保证 fork 保真。

metadata（title/tag/mode/worktree/PR 等）也写在 transcript，并会 re-append 到尾部，确保轻量 session 列表只读取 tail window 时仍能找到。resume 不是简单塞回消息数组，而是加载日志、修复 compact/snip/progress/parallel tool result 链、恢复 metadata/fileHistory/context collapse/worktree/agent state，再交给 UI。

### 29.2 本项目对应实现

本项目 `AgentSession.record_event()` 将 user、final、error 三类可见事件追加到 `history_messages`，每次写入 `.agent_sessions/<session_id>/history.json`。`AgentSessionManager.get_session()` 遇到内存中不存在、磁盘 history 存在的 session 时，会：

```text
_load_history(session_id)
  -> _history_messages() 过滤 user/assistant/error
  -> _memory_from_history() 重建 user/assistant ShortMemory
  -> _build_session() 新建独立 Runtime、Todo、LongMemory、Planning、PermissionController
```

工具协议不写入可见 history；重启后历史用户仍可继续对话。`snapshot()` 还有兜底：history 为空但 Runtime 里有可见 user/assistant 时，会回填并保存 history。

### 29.3 明确差异

本项目借鉴了“会话持久化和恢复不能依赖浏览器内存”的思路，但数据模型远小于 Claude Code：

- 本项目是覆盖写 JSON snapshot，不是 append-only JSONL event log；
- 无 sidechain/subagent transcript；
- 无 UUID 去重和批量写队列；
- 无 compact/snip/parallel tool result 修复；
- 无 metadata tail re-append、远端 ingress、worktree/file history 状态；
- 权限 pending 的 `_pending_*` 仅在内存中，服务重启无法恢复。

因此面试中必须准确说“实现了轻量多会话历史恢复”，不能说“实现了 Claude Code 级 transcript/resume”。

## 30. 最具体的借鉴六：Permission 与 Sandbox 是两层，不是一层

### 30.1 Claude Code 源码分析中的机制

`analysis/04e-sandbox-implementation.md` 的关键链路为：

```text
BashTool 调用
  -> shouldUseSandbox()
  -> bashPermissions.checkSandboxAutoAllow()
  -> Shell.ts
  -> SandboxManager.wrapWithSandbox()
  -> sandbox runtime/bwrap/macOS runtime
  -> cleanupAfterCommand()
```

`shouldUseSandbox()` 同时看 sandbox 开关、单次 `dangerouslyDisableSandbox`、是否允许 unsandboxed、是否有 command、是否命中 excludedCommands。`excludedCommands` 被源码注释明确标为“便利配置，不是安全边界”。

`convertToSandboxRuntimeConfig()` 还会把 settings 中的 permissions/sandbox 语义翻译为 runtime 的 filesystem/network allow/deny。显式 deny/ask 优先于“已沙箱化可自动放行”；sandbox write allowlist 还能反哺应用层 path validation。命令结束后还有 Git bare repo 逃逸相关清理。

### 30.2 本项目对应实现

本项目只实现应用层权限：`PermissionManager.decide()` 根据 Tool 的 risk level、动态风险、路径边界返回 allowed/requires_approval。具体为：

- Shell mode 动态升为 dangerous；
- web_search 动态升为 caution；
- 系统路径拒绝；
- 工作区外读升为 caution，工作区外写升为 dangerous；
- CLI 同步确认；Web 用 PendingPermissionRequest/SSE 恢复。

这与 Claude Code 分析中的 permission 层有同类思想：模型请求不直接等于执行、allow/deny/ask 应由 Runtime 作出、拒绝需要回流给模型。

### 30.3 不能声称借鉴完成的部分

本项目没有 `shouldUseSandbox()`、SandboxManager、filesystem/network runtime config 翻译、bwrap/macOS runtime、cleanupAfterCommand、Git bare repo 逃逸清理、sandbox doctor、网络越权对话框。`execute_code` 的 Shell 仍是 `subprocess.run(..., shell=True)` 加少量字符串黑名单，安全等级远低于分析仓库描述的两层防护。

## 31. Claude Code 有而本项目未借鉴完成的三项平台机制

### 31.1 Skills

指定分析仓库的 `analysis/04c-skills-implementation.md` 显示：Skills 来自用户目录、项目向上查找目录、managed 目录、bundled 和 MCP；`SKILL.md` frontmatter 含 `allowed_tools`、`paths`、model 等。`paths` 是条件技能触发器，匹配文件变化后才注入。受信任 file skill 的 prompt 内嵌 Shell 会先过权限检查；MCP 来源 skill 被明确禁止执行内嵌 Shell，以防远程 RCE。

本项目没有 SKILL.md loader、frontmatter、按 path 激活、来源信任分级或 prompt 内嵌 Shell 机制。当前的 AGENT.md 是项目规则文件，不等于 Skills 系统。

### 31.2 MCP

`analysis/04d-mcp-implementation.md` 中 MCP 工具命名为 `mcp__<server>__<tool>`，通过 `assembleToolPool()` 与内建工具合并，内建同名优先；有描述截断 2048 字符、连接并发限制、认证缓存、session 过期重连和 IDE 白名单。

本项目的 ToolRegistry 具备统一工具抽象，因此是未来接入 MCP 的合适扩展点；但当前没有 MCP transport、server lifecycle、命名空间、认证、重连或外部工具安全治理。不能说已实现 MCP。

### 31.3 Multi-Agent

`analysis/04h-multi-agent.md` 将 Claude Code 分成普通 AgentTool subagent、Coordinator Mode、Swarm teammates。Coordinator 的主线程专职 orchestration；Swarm 有 team、lead、teammate、mailbox、共享 task list、leader permission bridge，并限制 teammate 嵌套 spawn。

本项目只有单 AgentRuntime + TodoList。PlanningManager 是单 agent 的任务计划，不是 worker 调度；Web 多 session 是用户会话隔离，不是 agent team。项目未实现 sidechain、forked worker、mailbox、共享任务平面或权限桥接。

## 32. 基于源码对照后，最准确的面试说法

> 这个项目并非复刻 Claude Code，而是参考其源码分析中“Query Runtime 统筹 Tool Pipeline、Prompt Runtime、Memory、Permission 和 Transcript”的分层方式，做了一个 Python 单进程最小实现。最直接的对应是 AgentRuntime 的 Function Calling batch、ToolRegistry、Permission pause/resume、AGENT.md 项目上下文、ShortMemory/LongMemory/Compression/Scratchpad 与 AgentSession 历史恢复。\n> \n> 对照 Claude Code 的具体源码，本项目没有 MEMORY.md 文件索引和 sideQuery recall，没有 Session Memory forked agent，没有 JSONL sidechain transcript，没有 Tool 并发/Hooks，没有 sandbox runtime，也没有 Skills、MCP 和 multi-agent。我的实现更适合作为理解 Code Agent Runtime 的原型；下一步应优先补齐严格 schema、压缩调度、Sandbox、事件日志和可扩展工具生态。
