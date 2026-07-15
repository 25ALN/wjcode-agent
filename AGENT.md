# Coder Agent 项目规则

## 项目信息
- 项目名称：Coder Agent（Claude Code 风格 AI 编程助手）
- 项目语言：Python 3.10+
- 包管理：pip
- LLM 后端：DeepSeek API（deepseek-chat / deepseek-reasoner）
- 入口文件：main.py

## 构建与运行
- 安装 CLI 依赖：`pip install python-dotenv`
- 可选 Web 依赖：`pip install fastapi uvicorn`
- 启动 CLI：`python3 main.py`
- 启用工具：`python3 main.py --tools`
- 启用 RAG：`python3 main.py --tools --rag`
- 启动 Web：`python3 -m server.main --host 127.0.0.1 --port 8000`，浏览器打开 `http://127.0.0.1:8000/`
- 测试命令：`python3 -m pytest`（pytest 未安装时用 `python3 test_*.py`）

## API 密钥配置
- 在 `.env` 中设置 `DEEPSEEK_API_KEY=你的密钥`
- 获取地址：https://platform.deepseek.com/api_keys
- 联网搜索使用免费的 DuckDuckGo HTML/Lite 兜底页面，不需要额外搜索 API key

## 代码规范
- 遵循 PEP 8
- 使用 Google 风格 docstring
- 类名 PascalCase，函数名 snake_case，常量 UPPER_SNAKE
- 类型注解必须完整（包括 Optional、List、Dict 等泛型）

## 禁止事项
- 禁止使用全局可变状态（模块级变量在运行时修改）
- 禁止裸 except（至少捕获 Exception）
- 禁止 print 调试（统一使用 logging 模块）
- 禁止在核心模块中硬编码路径/配置（统一走 .env 或参数注入）

## 项目结构
```
core/              — 核心能力包
  runtime/         — AgentRuntime、事件、消息结构、Web/API session
  context/         — Prompt 构建、Scratchpad、上下文压缩、AGENT.md 项目上下文
  memory/          — ShortMemory、LongMemory、结构化记忆检索与去重
  planning/        — PlanningManager、PlanState、重规划逻辑
  todo/            — TodoList 数据结构与持久化
  permission/      — 权限分级、Web/API 权限挂起与恢复
  intent.py        — 工具/Planning 意图词边界匹配，避免英文子串误判
  *.py             — 旧导入路径兼容层，例如 core.message/core.compression
tools/             — 工具层（BaseTool、Registry、File、Code、Web、Project、Todo）
server/            — FastAPI/SSE Web API 适配层
web/               — 静态前端页面（Markdown 回复、thinking 状态、独立权限审批卡、工具、Todo、Planning、Scratchpad）
llm/               — 模型封装层（DeepSeekClient 主力）
rag/               — RAG 检索（BGE-M3，辅助能力）
main.py            — 终端对话入口
AGENT.md           — 本文档（项目规则）
promote.txt        — 项目规划文档
```

## 关键约束
- LLM 调用统一走 Client.generate() 接口（`{"text": str|None, "function_call": dict|None, "function_calls": list}`）
- Tool 必须继承 BaseTool 并定义 name/description/parameters/risk_level
- Tool-first：代码理解优先使用 `ls` / `grep` / `read_file`，小范围修改优先使用 `edit_file`
- 只读项目/目录分析（如难点、亮点、架构、结构梳理）只能暴露 `ls` / `grep` / `read_file`，不得同时暴露 `write_file` / `edit_file` / `execute_code` / `web_search`；此类任务进入 Exploration Mode，由 Runtime 维护 Exploration Checklist、Coverage Tracking、visited paths/files/symbols 和 Information Gain；核心模块未覆盖或最终回答缺少 Evidence 时必须继续探索，达到覆盖阈值、连续低信息增益或证据充分后再关闭工具并输出最终分析；若模型仍输出工具占位或失败，再使用本地确定性兜底答案
- `read_file` 未指定 `start_line/end_line` 时只返回文件开头窗口并提示截断；需要更多内容必须显式指定行号范围，避免长文件一次性撑爆上下文
- 工具 schema 必须按本轮意图暴露：普通概念问答不传工具，明确读/改/搜/运行项目内容时才进入工具循环；普通回答模式禁止输出 DSML/tool_calls 等伪工具协议文本
- 普通回答模式的历史上下文必须去除上一轮原始 `assistant.tool_calls` 和 `tool` 协议消息，只保留用户可见对话；如果模型仍输出纯伪工具调用，只能针对通用 fallback 做一次严格自然语言重试
- “模块怎么样了/状态如何/长期记忆存在哪里/打开旧会话从哪里恢复/请问...”这类状态或机制问答默认按普通回答处理；包含“打开/恢复”等词也不能仅凭关键词暴露工具；如果模型只输出“我先检查/我先看文件”但没有实际工具调用，应基于当前用户问题回退为直接自然语言答复
- 工具/Planning 意图匹配必须使用 `core.intent.contains_keyword()` 的词边界逻辑，禁止用裸 substring 判断英文关键词，避免 `latest` 误命中 `test`、`runtime` 误命中 `run`
- 简单任务走 ReAct；复杂执行任务必须通过 PlanningManager 生成计划，再进入 ReAct 工具循环；解释/咨询类问题即使包含“复杂/规划”等词也不能启动 Planning
- Planning 状态需要注入上下文；工具 Observation 暴露错误、失败、权限拒绝或代码修改后，应触发计划更新或补充验证步骤
- Scratchpad 只记录显式中间状态（目标、已确认事实、相关文件、尝试、阻塞、下一步），禁止记录隐藏推理或完整思考链
- 多步骤任务应通过 PlanningManager 和 `update_todo` 维护任务进度
- `AgentRuntime.run_events()` 是 Web/API 和兼容流式输出的主路径；`stream_run()` 只能作为薄包装调用 `run_events()`，不能再维护第二套 prompt/tool/permission 逻辑
- Function Calling 消息顺序必须满足协议：assistant 一旦带 `tool_calls`，下一次模型请求前必须为同一批所有 `tool_call_id` 补齐对应 tool 消息；权限恢复时必须继续完成当前 tool_calls batch，不能只补当前工具就立即请求模型
- Native tool 上下文必须过滤 ShortMemory 截断留下的非法协议片段：孤立 `tool` 消息、不完整 `assistant.tool_calls` 批次不能发送给 LLM；完整 tool_calls + tool results 批次必须保留
- Native tool 模式下如果模型输出 DSML/read_file/tool_calls 草稿但没有合法 function_call，应追加纠偏系统消息后继续工具循环，不应把草稿发给前端或兜底成普通问答
- 工具草稿检测必须只匹配短前置语或协议文本，不能把“我读取了项目文件后...”这类正常最终回答误判为工具草稿；天气、新闻、价格等实时外部信息即使是“怎么样”问法也应暴露 `web_search`
- Web/CLI runtime 系统提示必须注入当前日期；涉及今天、明天、后天等相对日期时必须以注入日期为准，必要时用联网搜索核验
- 天气/实时信息这类外部查询应只暴露 `web_search`，不要同时把 `execute_code` 暴露给模型；搜索继续问答（如“查吧”）也应延续 `web_search` 语义
- Web/API 层应使用 `server.service.AgentWebService` 暴露会话、SSE 消息流和权限恢复，不直接解析 CLI 文本输出
- Web/API 默认 LLM 请求避免长时间悬挂：`DEEPSEEK_TIMEOUT` 默认 30 秒，`DEEPSEEK_RETRY_TIMES` 默认 1 次
- Web UI 当前是 SSE 事件流 + 前端逐字呈现；真正 token 级工具流需要新增 runtime delta 事件，不能绕开 Function Calling/权限链路
- FastAPI 只作为薄适配层放在 `server/app.py`，核心服务逻辑应保持可单测、不可依赖真实 HTTP 环境
- Web/API 多会话应通过 `AgentSessionManager` 创建隔离 runtime，每个 session 独立维护 Memory、TodoList、Planning 和权限状态
- 每个 Web/API session 必须把用户可见的原始对话写入 `.agent_sessions/<session_id>/history.json`；历史持久化每轮都执行，不能和 8 轮摘要或上下文压缩阈值绑定；空会话不应作为磁盘历史展示，history 为空但 runtime memory 有可见对话时应兜底回填
- 前端左侧会话列表应从后端持久化 session 列表渲染，选择历史会话时恢复聊天记录并允许继续追问
- 长期记忆必须使用结构化 `MemoryItem`，注入上下文前应按当前用户输入/Planning 目标检索相关记忆，避免无关历史挤占上下文；不要把 session history、LongMemory 和 Context Compression 混为一层
- 添加长期记忆时应进行 hash 去重和相似度去重；新摘要必须保留用户目标、文件线索、工具结果、错误和测试结论
- 启用 RAG 时，长期记忆应复用 `Retriever.embedder`，避免重复加载 BGE 模型；未启用 RAG 时保留本地轻量检索回退
- 文件类工具的相对路径必须按 session/workspace root 解析；权限系统也必须使用同一 project_root 判断边界，工作区外访问需权限确认，系统路径直接阻断
- 所有文件操作使用 UTF-8 编码
- 临时文件使用 tempfile 模块，执行后自动清理
- LLM Client 之间通过统一接口解耦，切换到其他模型只需实现 generate/stream_generate/count_tokens 即可
