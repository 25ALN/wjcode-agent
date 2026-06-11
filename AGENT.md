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
core/          — 核心调度层（Agent Loop、Memory、Context、TodoList、ProjectContext）
tools/         — 工具层（BaseTool、Registry、File、Code、Web）
llm/           — 模型封装层（DeepSeekClient 主力，GeminiClient 保留兼容）
rag/           — RAG 检索（BGE-M3，辅助能力）
main.py        — 终端对话入口
AGENT.md       — 本文档（项目规则）
promote.txt    — 项目规划文档
```

## 关键约束
- LLM 调用统一走 Client.generate() 接口（`{"text": str|None, "function_call": dict|None}`）
- Tool 必须继承 BaseTool 并定义 name/description/parameters/risk_level
- 所有文件操作使用 UTF-8 编码
- 临时文件使用 tempfile 模块，执行后自动清理
- LLM Client 之间通过统一接口解耦，切换到其他模型只需实现 generate/stream_generate/count_tokens 即可