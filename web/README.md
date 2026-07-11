# Coder Agent Web UI

静态前端页面，不需要 Node 构建。FastAPI 后端会直接托管页面和静态资源。

启动后端：

```bash
pip install fastapi uvicorn
python3 -m server.main --host 127.0.0.1 --port 8000
```

然后打开：

```text
http://127.0.0.1:8000/
```

默认 API 地址会优先使用当前后端地址；如果用独立静态服务器打开前端，也会回退到 `http://127.0.0.1:8000`，并且可以在左侧服务地址输入框修改。

权限请求显示在聊天区上方的独立审批卡中，不写入聊天记录；thinking、Planning、工具调用和权限等待状态显示在聊天区上方的 activity strip，工具调用详情、Todo、Planning、Scratchpad 分别显示在右侧状态面板。

Agent 回复会在前端进行 Markdown 渲染，支持标题、列表、代码块、引用、表格和链接。当前 Web 链路使用 SSE 事件流，模型文本事件到达后由前端逐字呈现；在首个文本/工具/权限事件到达前会显示 thinking 状态。
