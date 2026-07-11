const STORAGE_KEY = "agent.apiBase";

function defaultApiBase() {
  const protocol = window.location.protocol;
  const hostname = window.location.hostname || "127.0.0.1";
  const port = window.location.port;

  if (protocol === "http:" || protocol === "https:") {
    if (!port || port === "8000") return window.location.origin;
    if (["5173", "8001", "8080"].includes(port)) return `${protocol}//${hostname}:8000`;
    return window.location.origin;
  }
  return "http://127.0.0.1:8000";
}

const state = {
  apiBase: localStorage.getItem(STORAGE_KEY) || defaultApiBase(),
  sessionId: null,
  sessions: [],
  tools: [],
  pendingPermission: null,
  busy: false,
  resolvingPermission: false,
  lastAssistantText: "",
  pendingAssistantText: "",
  typingAnimations: new Set(),
  activityItems: [],
  activityStatus: "",
  activityMode: "idle",
  activityHideTimer: null,
};

const el = {
  apiBaseInput: document.getElementById("apiBaseInput"),
  saveApiButton: document.getElementById("saveApiButton"),
  apiBadge: document.getElementById("apiBadge"),
  connectionDot: document.getElementById("connectionDot"),
  connectionStatus: document.getElementById("connectionStatus"),
  newSessionButton: document.getElementById("newSessionButton"),
  sessionList: document.getElementById("sessionList"),
  sessionTitle: document.getElementById("sessionTitle"),
  runtimeSubtitle: document.getElementById("runtimeSubtitle"),
  streamBadge: document.getElementById("streamBadge"),
  refreshButton: document.getElementById("refreshButton"),
  deleteSessionButton: document.getElementById("deleteSessionButton"),
  chatLog: document.getElementById("chatLog"),
  messageForm: document.getElementById("messageForm"),
  messageInput: document.getElementById("messageInput"),
  composerHint: document.getElementById("composerHint"),
  sendButton: document.getElementById("sendButton"),
  permissionBanner: document.getElementById("permissionBanner"),
  permissionTool: document.getElementById("permissionTool"),
  permissionReason: document.getElementById("permissionReason"),
  permissionArgs: document.getElementById("permissionArgs"),
  activityPanel: document.getElementById("activityPanel"),
  activityStatus: document.getElementById("activityStatus"),
  activityList: document.getElementById("activityList"),
  allowPermissionButton: document.getElementById("allowPermissionButton"),
  denyPermissionButton: document.getElementById("denyPermissionButton"),
  planningPanel: document.getElementById("planningPanel"),
  planningMode: document.getElementById("planningMode"),
  todoPanel: document.getElementById("todoPanel"),
  todoProgress: document.getElementById("todoProgress"),
  scratchpadPanel: document.getElementById("scratchpadPanel"),
  toolPanel: document.getElementById("toolPanel"),
};

el.apiBaseInput.value = state.apiBase;

function normalizeBase(value) {
  return String(value || "").trim().replace(/\/+$/, "") || defaultApiBase();
}

function apiUrl(path) {
  return `${state.apiBase}${path}`;
}

async function request(path, options = {}) {
  const response = await fetch(apiUrl(path), {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch (_) {
      // keep HTTP detail
    }
    throw new Error(detail);
  }
  return response;
}

async function requestJson(path, options = {}) {
  const response = await request(path, options);
  return response.json();
}

function setBusy(value) {
  state.busy = value;
  updateControls();
}

function updateControls() {
  const hasSession = Boolean(state.sessionId);
  const waitingApproval = Boolean(state.pendingPermission);
  const canSend = hasSession && !state.busy && !waitingApproval && !state.resolvingPermission;
  const canResolve = waitingApproval && !state.resolvingPermission;

  el.sendButton.disabled = !canSend;
  el.messageInput.disabled = !canSend;
  el.newSessionButton.disabled = state.busy || state.resolvingPermission;
  el.refreshButton.disabled = state.busy || state.resolvingPermission;
  el.deleteSessionButton.disabled = !hasSession || state.busy || state.resolvingPermission;
  el.allowPermissionButton.disabled = !canResolve;
  el.denyPermissionButton.disabled = !canResolve;
  el.permissionBanner.classList.toggle("is-resolving", state.resolvingPermission);

  if (state.resolvingPermission) setStreamState("running", "resolving");
  else if (waitingApproval) setStreamState("approval", "approval");
  else if (state.busy) setStreamState("running", "running");
  else setStreamState("idle", "idle");

  if (!hasSession) el.composerHint.textContent = "No session";
  else if (waitingApproval) el.composerHint.textContent = "Waiting for approval";
  else if (state.busy) el.composerHint.textContent = "Running";
  else el.composerHint.textContent = "Ready";
}

function setStreamState(mode, text) {
  el.streamBadge.textContent = text;
  el.streamBadge.className = `status-chip ${mode}`;
}

function setStatus(text, mode = "connected") {
  el.connectionStatus.textContent = text;
  el.connectionDot.className = `connection-dot ${mode}`;
  el.apiBadge.textContent = mode === "connected" ? "online" : mode === "checking" ? "check" : "offline";
  el.apiBadge.className = `mini-badge ${mode}`;
}

function escapeText(value) {
  const div = document.createElement("div");
  div.textContent = value == null ? "" : String(value);
  return div.innerHTML;
}

function formatTime(timestamp) {
  if (!timestamp) return "";
  const date = new Date(timestamp * 1000);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function formatSessionTime(timestamp) {
  if (!timestamp) return "";
  const date = new Date(timestamp * 1000);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString([], {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function shortId(sessionId) {
  return String(sessionId || "").slice(0, 10);
}

function compactJson(value) {
  if (value == null) return "";
  try {
    return JSON.stringify(value, null, 2);
  } catch (_) {
    return String(value);
  }
}

function ensureChatReady() {
  const empty = el.chatLog.querySelector(".empty-chat");
  if (empty) empty.remove();
}

function renderEmptyChat() {
  el.chatLog.innerHTML = '<div class="empty-chat">暂无消息</div>';
}

function cancelTypingAnimations() {
  for (const animation of state.typingAnimations) animation.cancelled = true;
  state.typingAnimations.clear();
}

function clearChat() {
  cancelTypingAnimations();
  clearActivity();
  el.chatLog.innerHTML = "";
  renderEmptyChat();
}

function safeLinkUrl(url) {
  const value = String(url || "").trim();
  if (/^(https?:|mailto:)/i.test(value)) return value;
  if (value.startsWith("#") || value.startsWith("/")) return value;
  return "";
}

function renderInlineMarkdown(value) {
  const codeTokens = [];
  let text = String(value == null ? "" : value).replace(/`([^`]+)`/g, (_, code) => {
    const token = `@@CODE${codeTokens.length}@@`;
    codeTokens.push(`<code>${escapeText(code)}</code>`);
    return token;
  });

  text = escapeText(text);
  text = text.replace(/\[([^\]]+)\]\(([^\s)]+)\)/g, (_, label, url) => {
    const cleanUrl = safeLinkUrl(url.replace(/&amp;/g, "&"));
    if (!cleanUrl) return `${label} (${escapeText(url)})`;
    return `<a href="${escapeText(cleanUrl)}" target="_blank" rel="noreferrer">${label}</a>`;
  });
  text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  text = text.replace(/__([^_]+)__/g, "<strong>$1</strong>");
  text = text.replace(/~~([^~]+)~~/g, "<del>$1</del>");
  text = text.replace(/(^|\s)\*([^*\n]+)\*/g, "$1<em>$2</em>");
  text = text.replace(/(^|\s)_([^_\n]+)_/g, "$1<em>$2</em>");
  text = text.replace(/@@CODE(\d+)@@/g, (_, index) => codeTokens[Number(index)] || "");
  return text;
}

function isFence(line) {
  return /^```/.test(line.trim());
}

function isTableSeparator(line) {
  return /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);
}

function parseTableRow(line) {
  let value = line.trim();
  if (value.startsWith("|")) value = value.slice(1);
  if (value.endsWith("|")) value = value.slice(0, -1);
  return value.split("|").map((cell) => cell.trim());
}

function renderMarkdown(markdown) {
  const lines = String(markdown == null ? "" : markdown).replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];
    const trimmed = line.trim();

    if (!trimmed) {
      index += 1;
      continue;
    }

    if (isFence(line)) {
      const language = trimmed.replace(/^```/, "").trim().replace(/[^a-zA-Z0-9_+.-]/g, "").slice(0, 24);
      index += 1;
      const code = [];
      while (index < lines.length && !isFence(lines[index])) {
        code.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) index += 1;
      const lang = language ? `<div class="code-label">${escapeText(language)}</div>` : "";
      html.push(`<div class="code-block">${lang}<pre><code>${escapeText(code.join("\n"))}</code></pre></div>`);
      continue;
    }

    const heading = /^(#{1,3})\s+(.+)$/.exec(line);
    if (heading) {
      const level = heading[1].length;
      html.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
      index += 1;
      continue;
    }

    if (/^\s*[-*_]{3,}\s*$/.test(line)) {
      html.push("<hr>");
      index += 1;
      continue;
    }

    if (index + 1 < lines.length && line.includes("|") && isTableSeparator(lines[index + 1])) {
      const headers = parseTableRow(line);
      index += 2;
      const rows = [];
      while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
        rows.push(parseTableRow(lines[index]));
        index += 1;
      }
      const head = headers.map((cell) => `<th>${renderInlineMarkdown(cell)}</th>`).join("");
      const body = rows.map((row) => `<tr>${row.map((cell) => `<td>${renderInlineMarkdown(cell)}</td>`).join("")}</tr>`).join("");
      html.push(`<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`);
      continue;
    }

    if (/^\s*>\s?/.test(line)) {
      const quote = [];
      while (index < lines.length && /^\s*>\s?/.test(lines[index])) {
        quote.push(lines[index].replace(/^\s*>\s?/, ""));
        index += 1;
      }
      html.push(`<blockquote>${renderInlineMarkdown(quote.join("\n"))}</blockquote>`);
      continue;
    }

    if (/^\s*[-*+]\s+/.test(line)) {
      const items = [];
      while (index < lines.length && /^\s*[-*+]\s+/.test(lines[index])) {
        items.push(lines[index].replace(/^\s*[-*+]\s+/, ""));
        index += 1;
      }
      html.push(`<ul>${items.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</ul>`);
      continue;
    }

    if (/^\s*\d+[.)]\s+/.test(line)) {
      const items = [];
      while (index < lines.length && /^\s*\d+[.)]\s+/.test(lines[index])) {
        items.push(lines[index].replace(/^\s*\d+[.)]\s+/, ""));
        index += 1;
      }
      html.push(`<ol>${items.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</ol>`);
      continue;
    }

    const paragraph = [];
    while (
      index < lines.length &&
      lines[index].trim() &&
      !isFence(lines[index]) &&
      !/^(#{1,3})\s+/.test(lines[index]) &&
      !/^\s*[-*+]\s+/.test(lines[index]) &&
      !/^\s*\d+[.)]\s+/.test(lines[index]) &&
      !/^\s*>\s?/.test(lines[index])
    ) {
      paragraph.push(lines[index]);
      index += 1;
    }
    html.push(`<p>${renderInlineMarkdown(paragraph.join("\n"))}</p>`);
  }

  return html.join("") || "";
}

function createMessageShell(kind, label, timestamp) {
  ensureChatReady();
  const node = document.createElement("article");
  node.className = `message ${kind}`;

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = kind === "user" ? "你" : kind === "assistant" ? "AI" : "!";

  const body = document.createElement("div");
  body.className = "message-body";

  const meta = document.createElement("div");
  meta.className = "message-meta";
  const time = formatTime(timestamp);
  meta.textContent = time ? `${label} · ${time}` : label;

  const content = document.createElement("div");
  content.className = "message-content";

  body.appendChild(meta);
  body.appendChild(content);
  node.appendChild(avatar);
  node.appendChild(body);
  el.chatLog.appendChild(node);
  el.chatLog.scrollTop = el.chatLog.scrollHeight;
  return { node, content };
}

function addMessage(kind, label, content, timestamp, options = {}) {
  const message = createMessageShell(kind, label, timestamp);
  const value = content == null ? "" : String(content);
  if (options.markdown || kind === "assistant") {
    message.content.classList.add("markdown-content");
    message.content.innerHTML = renderMarkdown(value);
  } else {
    message.content.textContent = value;
  }
  return message;
}

function renderActivity() {
  if (!el.activityPanel || !el.activityStatus || !el.activityList) return;
  const hasActivity = Boolean(state.activityStatus || state.activityItems.length);
  el.activityPanel.classList.toggle("hidden", !hasActivity);
  el.activityPanel.className = `activity-panel ${hasActivity ? "" : "hidden"} ${state.activityMode || "idle"}`.trim();
  el.activityStatus.textContent = state.activityStatus || "Ready";
  el.activityList.innerHTML = "";

  for (const item of state.activityItems.slice(-8)) {
    const node = document.createElement("div");
    node.className = `activity-item ${item.kind || "event"} ${item.status || "running"}`;
    node.innerHTML = `
      <span class="activity-dot"></span>
      <span class="activity-title">${escapeText(item.title || "Activity")}</span>
      ${item.detail ? `<span class="activity-detail">${escapeText(item.detail)}</span>` : ""}
    `;
    el.activityList.appendChild(node);
  }
}

function clearActivity() {
  if (state.activityHideTimer) window.clearTimeout(state.activityHideTimer);
  state.activityHideTimer = null;
  state.activityItems = [];
  state.activityStatus = "";
  state.activityMode = "idle";
  renderActivity();
}

function showActivity(status, mode = "running") {
  if (state.activityHideTimer) window.clearTimeout(state.activityHideTimer);
  state.activityHideTimer = null;
  state.activityStatus = status || "Running";
  state.activityMode = mode;
  renderActivity();
}

function finishActivity(status = "Done") {
  state.activityStatus = status;
  state.activityMode = status === "Error" ? "error" : "done";
  renderActivity();
}

function showThinking(text = "Thinking") {
  showActivity(text, "thinking");
}

function hideThinking() {
  finishActivity("Done");
}

function upsertActivityItem(item) {
  const key = item.id || `${item.kind || "event"}:${item.title || "activity"}`;
  const index = state.activityItems.findIndex((existing) => existing.id === key);
  const next = { ...item, id: key };
  if (index >= 0) state.activityItems[index] = { ...state.activityItems[index], ...next };
  else state.activityItems.push(next);
  renderActivity();
}

function renderAssistantResponse(content, timestamp) {
  const value = content == null ? "" : String(content);
  if (!value) return;
  showActivity("Writing answer", "answer");
  const message = createMessageShell("assistant streaming", "Agent", timestamp);
  message.content.classList.add("markdown-content");

  const animation = { cancelled: false };
  state.typingAnimations.add(animation);
  animateAssistantText(message.node, message.content, value, animation);
}

function animateAssistantText(node, contentNode, text, animation) {
  const chunkSize = text.length > 900 ? 8 : text.length > 360 ? 4 : 2;
  let offset = 0;

  function step() {
    if (animation.cancelled) return;
    offset = Math.min(text.length, offset + chunkSize);
    contentNode.textContent = text.slice(0, offset);
    el.chatLog.scrollTop = el.chatLog.scrollHeight;

    if (offset < text.length) {
      window.setTimeout(step, 12);
      return;
    }

    contentNode.innerHTML = renderMarkdown(text);
    node.classList.remove("streaming");
    state.typingAnimations.delete(animation);
  }

  step();
}

function renderSessions() {
  el.sessionList.innerHTML = "";
  if (!state.sessions.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state compact-empty";
    empty.textContent = "暂无会话";
    el.sessionList.appendChild(empty);
    return;
  }

  for (const session of state.sessions) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `session-button${session.session_id === state.sessionId ? " active" : ""}`;
    button.title = session.session_id;
    const updated = formatSessionTime(session.updated_at || session.created_at);
    button.innerHTML = `
      <span class="session-id">${escapeText(shortId(session.session_id))}</span>
      <span class="session-meta">${escapeText(String(session.memory_messages || 0))} messages${updated ? ` · ${escapeText(updated)}` : ""}</span>
    `;
    button.addEventListener("click", () => selectSession(session.session_id));
    el.sessionList.appendChild(button);
  }
}

function renderSnapshot(snapshot) {
  state.sessionId = snapshot.session_id;
  el.sessionTitle.textContent = `Session ${shortId(snapshot.session_id)}`;
  el.runtimeSubtitle.textContent = snapshot.workspace_root || "Workspace";
  renderPlanning(snapshot.planning);
  renderTodo(snapshot.todo, typeof snapshot.todo_progress === "number" ? snapshot.todo_progress : null);
  renderScratchpad(snapshot.scratchpad);
  renderPermission(snapshot.pending_permission);
  renderSessions();
  updateControls();
}

function renderPlanning(payload) {
  if (!payload) {
    el.planningMode.textContent = "react";
    el.planningMode.className = "mini-badge";
    el.planningPanel.className = "state-content empty-state";
    el.planningPanel.textContent = "暂无计划";
    return;
  }

  if (typeof payload === "string") {
    const isPlanning = payload.includes("Planning") || payload.includes("计划版本");
    el.planningMode.textContent = isPlanning ? "planning" : "react";
    el.planningMode.className = `mini-badge ${isPlanning ? "planning" : ""}`;
    el.planningPanel.className = "state-content pre-state";
    el.planningPanel.textContent = payload;
    return;
  }

  const mode = payload.mode || "planning";
  const steps = Array.isArray(payload.steps) ? payload.steps : [];
  el.planningMode.textContent = mode;
  el.planningMode.className = `mini-badge ${mode === "planning" ? "planning" : ""}`;

  if (mode !== "planning" && !steps.length) {
    el.planningPanel.className = "state-content empty-state";
    el.planningPanel.textContent = "暂无计划";
    return;
  }

  const reason = payload.reason ? `<div class="plan-reason">${escapeText(payload.reason)}</div>` : "";
  const revision = payload.revision != null ? `<span class="plan-revision">v${escapeText(payload.revision)}</span>` : "";
  const list = steps.length
    ? `<ol class="step-list">${steps.map((step) => `<li>${escapeText(step)}</li>`).join("")}</ol>`
    : "";
  const observation = payload.last_observation
    ? `<div class="observation-block">${escapeText(payload.last_observation)}</div>`
    : "";

  el.planningPanel.className = "state-content";
  el.planningPanel.innerHTML = `
    <div class="plan-title"><strong>${escapeText(payload.objective || "当前计划")}</strong>${revision}</div>
    ${reason}
    ${list}
    ${observation}
  `;
}

function renderTodo(text, progress) {
  if (typeof progress === "number") {
    el.todoProgress.textContent = `${Math.round(progress * 100)}%`;
  } else if (text) {
    el.todoProgress.textContent = "active";
  } else {
    el.todoProgress.textContent = "0%";
  }

  if (!text) {
    el.todoPanel.className = "state-content empty-state";
    el.todoPanel.textContent = "暂无任务";
    return;
  }

  el.todoPanel.className = "state-content pre-state";
  el.todoPanel.textContent = text;
}

function renderScratchpad(payload) {
  if (!payload) {
    el.scratchpadPanel.className = "state-content empty-state";
    el.scratchpadPanel.textContent = "暂无草稿";
    return;
  }

  const lines = [];
  if (payload.objective) lines.push(`目标: ${payload.objective}`);
  addList(lines, "已确认事实", payload.facts);
  addList(lines, "相关文件", payload.files);
  addList(lines, "已尝试操作", payload.attempts);
  addList(lines, "阻塞/风险", payload.blockers);
  addList(lines, "下一步", payload.next_steps);

  el.scratchpadPanel.className = lines.length ? "state-content pre-state" : "state-content empty-state";
  el.scratchpadPanel.textContent = lines.length ? lines.join("\n") : "暂无草稿";
}

function addList(lines, title, values) {
  if (!Array.isArray(values) || !values.length) return;
  lines.push(`${title}:`);
  for (const value of values) lines.push(`- ${value}`);
}

function renderPermission(payload) {
  state.pendingPermission = payload && !payload.resolved ? payload : null;

  if (!state.pendingPermission) {
    el.permissionBanner.classList.add("hidden");
    el.permissionBanner.removeAttribute("data-request-id");
    el.permissionTool.textContent = "工具调用";
    el.permissionReason.textContent = "";
    el.permissionArgs.textContent = "";
    updateControls();
    return;
  }

  const request = state.pendingPermission;
  el.permissionBanner.dataset.requestId = request.request_id || "";
  el.permissionTool.textContent = request.tool_name || "工具调用";
  const risk = request.risk_level ? `风险级别: ${request.risk_level}` : "风险级别: unknown";
  el.permissionReason.textContent = request.reason ? `${risk} · ${request.reason}` : risk;
  el.permissionArgs.textContent = compactJson(request.args || {});
  el.permissionBanner.classList.remove("hidden");
  updateControls();
}

function upsertTool(item) {
  const id = item.id || "";
  const index = id ? state.tools.findIndex((tool) => tool.id === id) : -1;
  if (index >= 0) {
    state.tools[index] = { ...state.tools[index], ...item };
  } else {
    state.tools.push({ ...item, createdAt: Date.now() });
  }
  renderTools();
}

function markLatestToolForApproval(payload) {
  const id = payload.id || "";
  let item = id ? state.tools.find((tool) => tool.id === id) : null;
  if (!item) {
    item = [...state.tools].reverse().find((tool) => tool.name === payload.tool_name && tool.status === "running");
  }
  if (item) {
    item.status = "approval";
    item.content = item.content || "等待权限确认";
    renderTools();
  }
}

function renderTools() {
  el.toolPanel.innerHTML = "";
  if (!state.tools.length) {
    el.toolPanel.className = "tool-list empty-state";
    el.toolPanel.textContent = "暂无工具调用";
    return;
  }

  el.toolPanel.className = "tool-list";
  for (const item of state.tools.slice(-24).reverse()) {
    const div = document.createElement("div");
    const status = item.status || "done";
    div.className = `tool-item ${status}`;
    const args = item.args && Object.keys(item.args).length ? compactJson(item.args) : "";
    div.innerHTML = `
      <div class="tool-item-header">
        <span class="tool-name">${escapeText(item.name || "tool")}</span>
        <span class="tool-status">${escapeText(status)}</span>
      </div>
      ${args ? `<pre class="tool-args">${escapeText(args)}</pre>` : ""}
      ${item.content ? `<pre class="tool-result">${escapeText(item.content)}</pre>` : ""}
    `;
    el.toolPanel.appendChild(div);
  }
}

async function loadSessions() {
  const payload = await requestJson("/sessions");
  state.sessions = payload.sessions || [];
  renderSessions();
  if (!state.sessionId && state.sessions.length) {
    await selectSession(state.sessions[0].session_id);
  }
}

async function createSession() {
  setBusy(true);
  try {
    const snapshot = await requestJson("/sessions", { method: "POST", body: JSON.stringify({}) });
    state.sessions.unshift(snapshot);
    state.tools = [];
    clearChat();
    renderTools();
    renderSnapshot(snapshot);
    setStatus("已连接", "connected");
  } catch (error) {
    addMessage("error", "错误", error.message);
    setStatus("连接失败", "error");
  } finally {
    setBusy(false);
  }
}

async function selectSession(sessionId) {
  if (sessionId === state.sessionId && state.sessionId) return;
  setBusy(true);
  try {
    const snapshot = await requestJson(`/sessions/${encodeURIComponent(sessionId)}`);
    state.tools = [];
    clearChat();
    renderTools();
    renderSnapshot(snapshot);
    setStatus("已连接", "connected");
  } catch (error) {
    addMessage("error", "错误", error.message);
    setStatus("连接失败", "error");
  } finally {
    setBusy(false);
  }
}

async function deleteSession() {
  if (!state.sessionId) return;
  const sessionId = state.sessionId;
  setBusy(true);
  try {
    await request(`/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
    state.sessionId = null;
    state.sessions = state.sessions.filter((session) => session.session_id !== sessionId);
    state.tools = [];
    clearChat();
    el.sessionTitle.textContent = "未选择会话";
    el.runtimeSubtitle.textContent = "等待连接";
    renderPlanning(null);
    renderTodo(null, 0);
    renderScratchpad(null);
    renderPermission(null);
    renderTools();
    renderSessions();
    if (state.sessions.length) await selectSession(state.sessions[0].session_id);
  } catch (error) {
    addMessage("error", "错误", error.message);
  } finally {
    setBusy(false);
  }
}

async function sendMessage(content) {
  const message = String(content || "").trim();
  if (!state.sessionId || !message || state.pendingPermission) return;
  addMessage("user", "你", message);
  state.lastAssistantText = "";
  state.pendingAssistantText = "";
  clearActivity();
  showThinking("Thinking");
  setBusy(true);
  try {
    await consumeSse(`/sessions/${encodeURIComponent(state.sessionId)}/messages`, {
      method: "POST",
      body: JSON.stringify({ content: message }),
    });
    await refreshSnapshot();
  } catch (error) {
    finishActivity("Error");
    addMessage("error", "错误", error.message);
  } finally {
    state.busy = false;
    updateControls();
    if (!state.pendingPermission && state.sessionId) el.messageInput.focus({ preventScroll: true });
  }
}

async function resolvePermission(approved) {
  if (!state.sessionId || !state.pendingPermission) return;
  const request = state.pendingPermission;
  const requestId = request.request_id;
  state.resolvingPermission = true;
  updateControls();
  try {
    await consumeSse(`/sessions/${encodeURIComponent(state.sessionId)}/permissions/${encodeURIComponent(requestId)}`, {
      method: "POST",
      body: JSON.stringify({ approved }),
    });
    renderPermission(null);
    await refreshSnapshot();
  } catch (error) {
    state.pendingPermission = request;
    renderPermission(request);
    addMessage("error", "错误", error.message);
  } finally {
    state.resolvingPermission = false;
    state.busy = false;
    updateControls();
    if (!state.pendingPermission && state.sessionId) el.messageInput.focus({ preventScroll: true });
  }
}

async function refreshSnapshot() {
  if (!state.sessionId) return;
  const snapshot = await requestJson(`/sessions/${encodeURIComponent(state.sessionId)}`);
  const index = state.sessions.findIndex((session) => session.session_id === snapshot.session_id);
  if (index >= 0) state.sessions[index] = snapshot;
  else state.sessions.unshift(snapshot);
  renderSnapshot(snapshot);
}

async function consumeSse(path, options) {
  const response = await request(path, options);
  if (!response.body) return;
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      handleSseFrame(frame);
      boundary = buffer.indexOf("\n\n");
    }
  }
  buffer += decoder.decode();
  if (buffer.trim()) handleSseFrame(buffer);
}

function handleSseFrame(frame) {
  const data = frame
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.replace(/^data:\s?/, ""))
    .join("\n");
  if (!data) return;
  try {
    handleEvent(JSON.parse(data));
  } catch (error) {
    addMessage("error", "SSE 解析失败", error.message);
  }
}

function handleEvent(event) {
  const data = event.data || {};
  switch (event.type) {
    case "user_message":
      break;
    case "assistant_text":
      state.pendingAssistantText = data.content || "";
      state.lastAssistantText = state.pendingAssistantText;
      showActivity("Drafting answer", "answer");
      break;
    case "final": {
      const content = data.content || state.pendingAssistantText;
      if (content) renderAssistantResponse(content, event.timestamp);
      state.lastAssistantText = content || state.lastAssistantText;
      state.pendingAssistantText = "";
      break;
    }
    case "tool_call":
      showThinking(data.name ? `Using ${data.name}` : "Using tool");
      upsertActivityItem({ id: `tool:${data.id || data.name}`, kind: "tool", title: data.name || "tool", detail: "running", status: "running" });
      upsertTool({ id: data.id, name: data.name, args: data.args, status: "running" });
      break;
    case "tool_result":
      showThinking("Reading tool result");
      upsertActivityItem({ id: `tool:${data.id || data.name}`, kind: "tool", title: data.name || "tool", detail: "done", status: "done" });
      upsertTool({ id: data.id, name: data.name, content: data.content, status: "done" });
      break;
    case "planning_update":
      showThinking("Planning");
      upsertActivityItem({ id: `planning:${data.revision || Date.now()}`, kind: "planning", title: "Planning", detail: data.reason || "updated", status: "done" });
      renderPlanning(data);
      break;
    case "todo_update":
      renderTodo(data.text, data.progress);
      break;
    case "permission_request":
      showActivity("Waiting for approval", "approval");
      upsertActivityItem({ id: `permission:${data.request_id || data.tool_name}`, kind: "permission", title: data.tool_name || "permission", detail: "approval", status: "approval" });
      markLatestToolForApproval(data);
      renderPermission(data);
      break;
    case "error":
      finishActivity("Error");
      addMessage("error", "错误", data.message || "未知错误", event.timestamp);
      break;
    case "done":
      if (state.activityMode !== "error") finishActivity("Done");
      state.busy = false;
      updateControls();
      if (!state.pendingPermission && state.sessionId) el.messageInput.focus({ preventScroll: true });
      break;
    default:
      upsertActivityItem({ id: `event:${event.type || Date.now()}`, kind: "event", title: event.type || "event", detail: compactJson(data), status: "done" });
  }
}

el.saveApiButton.addEventListener("click", async () => {
  state.apiBase = normalizeBase(el.apiBaseInput.value);
  el.apiBaseInput.value = state.apiBase;
  localStorage.setItem(STORAGE_KEY, state.apiBase);
  setStatus("检查中", "checking");
  setBusy(true);
  try {
    await requestJson("/health");
    setStatus("已连接", "connected");
    await loadSessions();
    if (!state.sessionId) await createSession();
  } catch (error) {
    setStatus("连接失败", "error");
    addMessage("error", "错误", error.message);
  } finally {
    setBusy(false);
  }
});

el.newSessionButton.addEventListener("click", createSession);
el.refreshButton.addEventListener("click", async () => {
  setBusy(true);
  try {
    await loadSessions();
    await refreshSnapshot();
    setStatus("已连接", "connected");
  } catch (error) {
    setStatus("连接失败", "error");
    addMessage("error", "错误", error.message);
  } finally {
    setBusy(false);
  }
});
el.deleteSessionButton.addEventListener("click", deleteSession);
el.allowPermissionButton.addEventListener("click", () => resolvePermission(true));
el.denyPermissionButton.addEventListener("click", () => resolvePermission(false));

el.messageForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const content = el.messageInput.value;
  el.messageInput.value = "";
  await sendMessage(content);
});

el.messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    el.messageForm.requestSubmit();
  }
});

async function boot() {
  clearChat();
  renderPlanning(null);
  renderTodo(null, 0);
  renderScratchpad(null);
  renderPermission(null);
  renderTools();
  updateControls();
  setStatus("检查中", "checking");
  setBusy(true);
  try {
    await requestJson("/health");
    setStatus("已连接", "connected");
    await loadSessions();
    if (!state.sessionId) await createSession();
  } catch (error) {
    setStatus("连接失败", "error");
    addMessage("error", "无法连接 API", `${error.message}\n请启动: python3 -m server.main --host 127.0.0.1 --port 8000`);
  } finally {
    setBusy(false);
  }
}

boot();
