# Coder Agent 项目规则

## 项目信息
- 项目名称：Coder Agent（Claude Code 风格 AI 编程助手）
- 项目语言：Python 3.10+
- 包管理：pip
- LLM 后端：DeepSeek API（deepseek-chat / deepseek-reasoner）
- 入口文件：main.py

## 构建与运行
- 安装依赖：`pip install google-generativeai python-dotenv`（仅保留用于兼容 GeminiClient，主流程不需要）
- 启动命令：`python3 main.py`
- 启用工具：`python3 main.py --tools`
- 启用 RAG：`python3 main.py --tools --rag`
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
  context/         — Prompt 构建、上下文压缩、AGENT.md 项目上下文
  memory/          — ShortMemory、LongMemory、结构化记忆检索与去重
  planning/        — PlanningManager、PlanState、重规划逻辑
  todo/            — TodoList 数据结构与持久化
  permission/      — 权限分级、Web/API 权限挂起与恢复
  *.py             — 旧导入路径兼容层，例如 core.message/core.compression
tools/             — 工具层（BaseTool、Registry、File、Code、Web、Project、Todo）
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
- 简单任务走 ReAct；复杂任务必须通过 PlanningManager 生成计划，再进入 ReAct 工具循环
- Planning 状态需要注入上下文；工具 Observation 暴露错误、失败、权限拒绝或代码修改后，应触发计划更新或补充验证步骤
- 多步骤任务应通过 PlanningManager 和 `update_todo` 维护任务进度
- Web/API 层应使用 `AgentRuntime.run_events()` / `resume_events()` 获取结构化事件，不直接解析 CLI 文本输出
- Web/API 多会话应通过 `AgentSessionManager` 创建隔离 runtime，每个 session 独立维护 Memory、TodoList、Planning 和权限状态
- 长期记忆必须使用结构化 `MemoryItem`，注入上下文前应按当前用户输入/Planning 目标检索相关记忆，避免无关历史挤占上下文
- 添加长期记忆时应进行 hash 去重和相似度去重；新摘要必须保留用户目标、文件线索、工具结果、错误和测试结论
- 启用 RAG 时，长期记忆应复用 `Retriever.embedder`，避免重复加载 BGE 模型；未启用 RAG 时保留本地轻量检索回退
- 所有文件操作使用 UTF-8 编码
- 临时文件使用 tempfile 模块，执行后自动清理
- LLM Client 之间通过统一接口解耦，切换到其他模型只需实现 generate/stream_generate/count_tokens 即可
