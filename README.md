# Coder Agent

一个类 Claude Code / Codex CLI 的本地 Code Agent 实验项目。项目目标不是做普通聊天机器人，而是实现一个能够理解编程任务、规划、调用工具、管理上下文、处理权限并通过 Web/CLI 交互的 Agent Runtime。

当前项目已完成核心 Runtime、Planning、Todo、Permission、Memory、Context Compression、Project Analysis、Web API 和静态前端工作台。

## 核心能力

- **ReAct + Planning 混合执行**：简单问题直接回答或轻量 ReAct；复杂编程任务进入 Planning，执行过程中根据工具观察继续调整计划。
- **Runtime 主导的项目分析**：分析整个项目、架构、亮点、难点时进入 Project Analysis Session，由 Runtime 维护 checklist、coverage、evidence 和 search planner，避免模型读几个文件就提前总结。
- **Function Calling 工具循环**：支持 `ls`、`grep`、`read_file`、`write_file`、`edit_file`、`execute_code`、`web_search`、`update_todo`。
- **权限系统**：写文件、编辑文件、执行代码、联网搜索等操作按风险级别确认；Web 端支持权限挂起和恢复。
- **上下文管理**：包含 System Prompt、用户消息、历史对话、工具结果、Scratchpad、长期记忆、RAG 参考、Planning/Todo 状态和 AGENT.md 项目规则。
- **长期记忆与历史会话**：Web 会话历史逐轮保存到 `.agent_sessions/<session_id>/history.json`，页面重新打开后可恢复；长期记忆支持结构化保存、检索和去重。
- **上下文压缩**：短期上下文过大时自动摘要旧历史，并写入长期记忆。
- **Web 工作台**：三栏界面，支持历史会话、Markdown 回答、thinking/activity 状态、工具调用面板、权限审批、Planning/Todo/Scratchpad 侧栏。
- **Token 统计**：CLI 或 Web 服务 `Ctrl+C` 退出时输出本次运行的 token 消耗统计，优先使用 DeepSeek API 返回的真实 usage，没有 usage 时显示估算值。

## 快速开始

### 1. 创建虚拟环境

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. 安装依赖

项目没有强制 Node 构建流程，前端由 FastAPI 直接托管。

```bash
pip install python-dotenv fastapi uvicorn pytest
```

如果需要启用 RAG/BGE-M3，可额外安装相关依赖：

```bash
pip install FlagEmbedding torch
```

### 3. 配置 DeepSeek API Key

在项目根目录创建 `.env`：

```bash
DEEPSEEK_API_KEY=你的密钥
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_TIMEOUT=30
DEEPSEEK_RETRY_TIMES=1
```

`.env` 已在 `.gitignore` 中忽略，不要提交。

## 启动方式

### Web UI 推荐方式

```bash
python -m server.main --host 127.0.0.1 --port 8000
```

然后打开：

```text
http://127.0.0.1:8000/
```

如果端口被占用，换一个端口即可：

```bash
python -m server.main --host 127.0.0.1 --port 8001
```

停止服务时按 `Ctrl+C`，终端会输出本次 Web 服务运行期间的 token 统计。

### CLI 对话方式

纯聊天：

```bash
python main.py
```

启用工具：

```bash
python main.py --tools
```

启用工具和 RAG：

```bash
python main.py --tools --rag
```

CLI 中输入 `/exit` 或按 `Ctrl+C` 退出时，会输出本次 CLI 会话的 token 统计。

## CLI Todo 命令

`/todo` 是终端层面的人工任务管理命令，用于查看或维护当前任务进度；工具层还有 `update_todo`，可让 Agent 在复杂任务中自动同步计划状态。

常用命令：

```text
/todo show
/todo add <任务描述>
/todo start <序号>
/todo done <序号> [结果]
/todo block <序号> [原因]
/todo reset <序号>
/todo remove <序号>
/todo clear
```

Todo 数据在 CLI 模式下保存到 `.agent_todo.json`，Web/API 每个 session 独立保存到 `.agent_sessions/<session_id>/todo.json`。

## Web UI 使用说明

- 左侧是历史会话列表，可新建、切换或删除某次历史聊天记录。
- 中间是聊天窗口，Agent 回答会渲染 Markdown。
- 权限请求显示为独立审批卡，不混入聊天回答。
- Thinking、Planning、工具调用和权限等待显示在聊天区上方的 activity strip。
- 右侧显示 Planning、Todo、Scratchpad 和 Tool Calls。
- 工具详情只展示在右侧面板，避免与最终回答混在一个气泡里。

## 目录结构

```text
.
├── main.py                  # CLI 对话入口
├── AGENT.md                 # 项目级 Agent 规则
├── promote.txt              # 项目规划和阶段状态文档
├── README.md                # 项目使用说明
├── core/
│   ├── runtime/             # AgentRuntime、Project Analysis、Session、Message、Events
│   ├── context/             # Context Builder、Scratchpad、Compression、ProjectContext
│   ├── memory/              # ShortMemory、LongMemory、结构化记忆检索和去重
│   ├── planning/            # PlanningManager、PlanState、重规划逻辑
│   ├── permission/          # 权限分级与 Web 权限恢复
│   ├── todo/                # TodoList 数据结构和持久化
│   └── *.py                 # 旧导入路径兼容层
├── llm/
│   └── deepseek_client.py   # DeepSeek Chat Completions / Function Calling 封装
├── tools/
│   ├── base_tool.py         # Tool 抽象和风险等级
│   ├── registry.py          # Tool 注册中心
│   ├── file_tool.py         # read_file / write_file
│   ├── project_tools.py     # ls / grep / edit_file
│   ├── code_executor.py     # execute_code
│   ├── web_search.py        # DuckDuckGo 免费搜索
│   └── todo_tool.py         # update_todo
├── rag/
│   ├── embedding.py         # TextSplitter + BGE-M3 Embedding
│   └── retriever.py         # RAG 检索器
├── server/
│   ├── service.py           # 框架无关服务层、SSE、会话与权限恢复
│   ├── app.py               # FastAPI 路由和静态前端托管
│   └── main.py              # Web 服务启动入口
└── web/
    ├── index.html           # 静态前端
    ├── styles.css           # 页面样式
    ├── app.js               # SSE 客户端和 UI 状态管理
    └── README.md            # Web UI 简要说明
```

## 工具和权限

| 工具 | 作用 | 默认风险 |
| --- | --- | --- |
| `ls` | 列目录 | SAFE |
| `grep` | 搜索文件内容 | SAFE |
| `read_file` | 读取文件，支持行号窗口 | SAFE |
| `write_file` | 写入文件 | CAUTION |
| `edit_file` | 小范围编辑文件，支持 dry run | CAUTION |
| `execute_code` | 执行 Python/Shell | CAUTION，shell 可升级 DANGEROUS |
| `web_search` | DuckDuckGo 免费联网搜索 | 动态确认 |
| `update_todo` | 维护任务进度 | SAFE |

工作区外路径不会静默执行；写入、编辑、执行代码和联网搜索会触发权限确认。

## 上下文和记忆

当前 Runtime 的上下文主要由这些部分组成：

```text
System Prompt
Project Context / AGENT.md
User Message
Chat History
Tool Results
Scratchpad
Long Memory
RAG References
Planning / Todo State
```

持久化位置：

- Web 可见历史：`.agent_sessions/<session_id>/history.json`
- Web 长期记忆：`.agent_sessions/<session_id>/memory_long.json`
- Web Todo：`.agent_sessions/<session_id>/todo.json`
- CLI Todo：`.agent_todo.json`
- CLI 长期记忆：`memory_long.json`

这些运行状态文件已被 `.gitignore` 忽略。

## HTTP API

FastAPI 提供的主要接口：

```text
GET  /                         # Web UI
GET  /health
POST /sessions
GET  /sessions
GET  /sessions/{session_id}
DELETE /sessions/{session_id}
POST /sessions/{session_id}/messages
GET  /sessions/{session_id}/pending-permission
POST /sessions/{session_id}/permissions/{request_id}
```

`/messages` 和权限恢复接口返回 `text/event-stream`，前端通过 SSE 消费结构化事件。

## 测试

运行完整测试：

```bash
venv/bin/python -m pytest
```

检查前端 JS 语法：

```bash
node --check web/app.js
```

本地测试文件 `test_*.py` 已在 `.gitignore` 中忽略，便于你本地保留验证用例而不上传。

## 提交前清理

建议提交前保留源码、文档和 `.gitignore`，清理本地生成物：

```bash
find . -path './venv' -prune -o -name '__pycache__' -type d -print
find . -path './venv' -prune -o -name '.pytest_cache' -type d -print
```

已忽略且不应提交的常见文件：

```text
venv/
__pycache__/
.pytest_cache/
.env
.agent_sessions/
.agent_todo.json
memory_long.json
test_*.py
```

## 当前状态

项目核心功能已完成，当前主要剩余增强方向是：

- 真正 token 级别的 delta 输出，而不是当前块级 SSE + 前端逐字呈现。
- 更细粒度的 diff 视图和工具结果折叠。
- 更丰富的 Planning 步骤状态。
- 更完善的真实浏览器长任务联调体验。
