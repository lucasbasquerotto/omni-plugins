#!/usr/bin/env node

/**
 * workbench MCP server : a THIN fetch wrapper around the workbench HTTP API.
 *
 * It exposes exactly ONE MCP tool, `tool` (exposed to the agent as
 * `workbench__tool`) and forwards every call as a single HTTP POST:
 *
 *   POST {base_url}{tool_path}
 *   Content-Type: application/json
 *   body: {"tool": <tool>, "params": <params>}
 *
 * No business logic, no SDK, no state: the plugin never inspects, retries or
 * reshapes the workbench payload. The HTTP response body is returned to the
 * agent as pretty-printed JSON text.
 *
 * Config (config_schema in plugin.json, delivered as a `configure` JSON-RPC
 * request and/or as environment variables):
 *   base_url     default http://workbench:8080
 *   tool_path    default /api/tool/call
 *   timeout_secs default 0 (NO timeout; a positive value opts in)
 *   auth_header  optional Authorization header value (empty = header omitted)
 *
 * Runtime: Node >= 18 (global `fetch` + `AbortController`), no npm deps.
 */

const readline = require("readline");
const process = require("process");

const MCP_PROTOCOL_VERSION = "2025-03-26";
const SERVER_NAME = "workbench";
const SERVER_VERSION = "0.1.0";

const DEFAULT_BASE_URL = "http://workbench:8080";
const DEFAULT_TOOL_PATH = "/api/tool/call";
// Timeout semantics for `timeout_secs`:
//   absent / empty / 0 -> NO timeout: the request runs until the workbench
//                         answers, the connection fails, or the CLIENT cancels
//                         it (omniagent `core__cancel_task` drops the in-flight
//                         MCP call and sends `notifications/cancelled`, which
//                         aborts the HTTP request here). Workbench dispatches
//                         are legitimately long-running: a hidden clock must
//                         never kill them.
//   positive integer   -> explicit operator opt-in timeout in seconds.
const NO_TIMEOUT_SECS = 0;
const DEFAULT_TIMEOUT_SECS = NO_TIMEOUT_SECS;

// The single declared tool name. `tool_qualify("workbench", "tool")` in core
// turns it into the exposed name `workbench__tool`.
const TOOL_NAME = "tool";

let initialized = false;

// ── Configuration ──────────────────────────────────────────────────────────

function pickString(candidates, fallback) {
  for (const value of candidates) {
    if (value === undefined || value === null) continue;
    const text = String(value).trim();
    if (text !== "") return text;
  }
  return fallback;
}

function pickInt(candidates, fallback) {
  for (const value of candidates) {
    if (value === undefined || value === null) continue;
    const text = String(value).trim();
    if (text === "") continue;
    const parsed = Number.parseInt(text, 10);
    if (Number.isFinite(parsed) && parsed > 0) return parsed;
  }
  return fallback;
}

// Timeout parser: absent/empty falls back, `0` means NO timeout (an explicit
// opt-out), any positive integer is the timeout in seconds.
function pickTimeoutSecs(candidates, fallback) {
  for (const value of candidates) {
    if (value === undefined || value === null) continue;
    const text = String(value).trim();
    if (text === "") continue;
    const parsed = Number.parseInt(text, 10);
    if (!Number.isFinite(parsed)) continue;
    if (parsed <= 0) return NO_TIMEOUT_SECS;
    return parsed;
  }
  return fallback;
}

/** Human-readable timeout for the logs: "none" when no timeout is set. */
function describeTimeout() {
  return config.timeout_secs > 0 ? String(config.timeout_secs) : "none";
}

const config = {
  base_url: pickString(
    [process.env.base_url, process.env.BASE_URL, process.env.WORKBENCH_BASE_URL],
    DEFAULT_BASE_URL
  ),
  tool_path: pickString(
    [process.env.tool_path, process.env.TOOL_PATH, process.env.WORKBENCH_TOOL_PATH],
    DEFAULT_TOOL_PATH
  ),
  timeout_secs: pickTimeoutSecs(
    [
      process.env.timeout_secs,
      process.env.TIMEOUT_SECS,
      process.env.WORKBENCH_TIMEOUT_SECS,
    ],
    DEFAULT_TIMEOUT_SECS
  ),
  auth_header: pickString(
    [
      process.env.auth_header,
      process.env.AUTH_HEADER,
      process.env.WORKBENCH_AUTH_HEADER,
    ],
    ""
  ),
};

/** Merge a `configure` payload (config keys as declared in plugin.json). */
function applyConfig(payload) {
  if (!payload || typeof payload !== "object") return;
  if (payload.base_url !== undefined)
    config.base_url = pickString([payload.base_url], config.base_url);
  if (payload.tool_path !== undefined)
    config.tool_path = pickString([payload.tool_path], config.tool_path);
  if (payload.timeout_secs !== undefined)
    config.timeout_secs = pickTimeoutSecs(
      [payload.timeout_secs],
      config.timeout_secs
    );
  if (payload.auth_header !== undefined)
    config.auth_header = pickString([payload.auth_header], "");
  console.error(
    "[workbench] config: base_url=" +
      config.base_url +
      " tool_path=" +
      config.tool_path +
      " timeout_secs=" +
      describeTimeout() +
      " auth_header=" +
      (config.auth_header ? "<set>" : "<none>")
  );
}

// ── JSON-RPC plumbing ──────────────────────────────────────────────────────

function sendJson(obj) {
  process.stdout.write(JSON.stringify(obj) + "\n");
}

function makeSuccess(reqId, result) {
  return { jsonrpc: "2.0", id: reqId, result };
}

function makeError(reqId, code, message) {
  return { jsonrpc: "2.0", id: reqId, error: { code, message } };
}

function toolResult(text, isError) {
  return { content: [{ type: "text", text }], isError: isError === true };
}

function handleInitialize(reqId) {
  sendJson(
    makeSuccess(reqId, {
      protocolVersion: MCP_PROTOCOL_VERSION,
      capabilities: { tools: { listChanged: false } },
      serverInfo: { name: SERVER_NAME, version: SERVER_VERSION },
    })
  );
  console.error("[workbench] initialized: " + SERVER_NAME + " v" + SERVER_VERSION);
}

function handleToolsList(reqId) {
  const tools = [
    {
      name: TOOL_NAME,
      description:
        "[workbench] Call a workbench tool by name: POSTs {\"tool\": <tool>, " +
        "\"params\": <params>} to the workbench HTTP API (base_url + tool_path) " +
        "and returns the response body. Thin wrapper, no business logic.",
      inputSchema: {
        type: "object",
        properties: {
          tool: {
            type: "string",
            description: "workbench tool/command name, e.g. hello_world",
          },
          params: {
            type: "object",
            additionalProperties: true,
            default: {},
            description: "arguments passed to the workbench tool",
          },
        },
        required: ["tool"],
      },
    },
  ];
  sendJson(makeSuccess(reqId, { tools }));
  console.error("[workbench] tools/list returned 1 tool");
}

// ── HTTP forwarding ────────────────────────────────────────────────────────

function prettyBody(bodyText) {
  const text = bodyText === null || bodyText === undefined ? "" : String(bodyText);
  if (text.trim() === "") return "(empty response body)";
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch (e) {
    return text;
  }
}

function targetUrl() {
  const base = String(config.base_url).replace(/\/+$/, "");
  const path = String(config.tool_path);
  return base + (path.startsWith("/") ? path : "/" + path);
}

// In-flight tools/call requests: JSON-RPC request id -> AbortController, so a
// `notifications/cancelled` from the client tears the HTTP request down for
// real (core__cancel_task must abort the in-flight call, not just flag it).
const inFlight = new Map();

async function callWorkbench(toolName, params, controller) {
  const url = targetUrl();
  const headers = {
    "Content-Type": "application/json",
    Accept: "application/json",
  };
  if (config.auth_header) headers.Authorization = config.auth_header;

  const body = JSON.stringify({ tool: toolName, params: params });
  const abort = controller || new AbortController();
  const timeout_secs = config.timeout_secs;
  // NO timeout configured -> no timer at all: the request ends when the
  // workbench answers, the connection fails, or the client cancels it.
  const timer =
    timeout_secs > 0
      ? setTimeout(() => abort.abort(), timeout_secs * 1000)
      : null;

  console.error("[workbench] POST " + url + " body=" + body);

  let response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers,
      body,
      signal: abort.signal,
    });
  } catch (err) {
    if (timer) clearTimeout(timer);
    const aborted = err && err.name === "AbortError";
    const reason = aborted
      ? timeout_secs > 0
        ? "request timed out after " + timeout_secs + "s"
        : "request cancelled by the client"
      : (err && err.message) || String(err);
    console.error("[workbench] request failed: " + reason);
    return {
      isError: true,
      text: "workbench request to " + url + " failed: " + reason,
    };
  }
  if (timer) clearTimeout(timer);

  let bodyText = "";
  try {
    bodyText = await response.text();
  } catch (err) {
    const reason = (err && err.message) || String(err);
    console.error("[workbench] failed to read response body: " + reason);
    return {
      isError: true,
      text:
        "workbench " +
        url +
        " returned HTTP " +
        response.status +
        " but the response body could not be read: " +
        reason,
    };
  }

  const pretty = prettyBody(bodyText);
  if (!response.ok) {
    const statusText = response.statusText ? " " + response.statusText : "";
    console.error("[workbench] HTTP " + response.status + " from " + url);
    return {
      isError: true,
      text:
        "workbench " +
        url +
        " returned HTTP " +
        response.status +
        statusText +
        "\n" +
        pretty,
    };
  }

  console.error("[workbench] HTTP " + response.status + " from " + url);
  return { isError: false, text: pretty };
}

async function handleCall(reqId, params) {
  const toolName = params.name || "";
  const args = params.arguments || {};

  if (toolName !== TOOL_NAME && toolName !== SERVER_NAME + "__" + TOOL_NAME) {
    sendJson(makeError(reqId, -32602, "Unknown tool: " + toolName));
    return;
  }

  const workbenchTool = args.tool;
  if (typeof workbenchTool !== "string" || workbenchTool.trim() === "") {
    sendJson(
      makeError(reqId, -32602, "Invalid params: 'tool' is required (string)")
    );
    console.error("[workbench] tools/call rejected: missing 'tool' argument");
    return;
  }

  let workbenchParams = args.params;
  if (workbenchParams === undefined || workbenchParams === null) {
    workbenchParams = {};
  } else if (
    typeof workbenchParams !== "object" ||
    Array.isArray(workbenchParams)
  ) {
    sendJson(
      makeError(reqId, -32602, "Invalid params: 'params' must be an object")
    );
    console.error("[workbench] tools/call rejected: 'params' is not an object");
    return;
  }

  // NOT awaited by the readline loop: calls are concurrent (150 parallel
  // calls must all be in flight without blocking each other) - which is also
  // what lets a `notifications/cancelled` be processed WHILE this call awaits
  // the HTTP response.
  const controller = new AbortController();
  inFlight.set(reqId, controller);
  try {
    const outcome = await callWorkbench(
      workbenchTool,
      workbenchParams,
      controller
    );
    sendJson(makeSuccess(reqId, toolResult(outcome.text, outcome.isError)));
  } finally {
    inFlight.delete(reqId);
  }
}

// ── Main loop ──────────────────────────────────────────────────────────────

const rl = readline.createInterface({ input: process.stdin, terminal: false });

console.error("[workbench] MCP server starting (PID=" + process.pid + ")");
applyConfigFromEnvSummary();

function applyConfigFromEnvSummary() {
  console.error(
    "[workbench] env config: base_url=" +
      config.base_url +
      " tool_path=" +
      config.tool_path +
      " timeout_secs=" +
      describeTimeout() +
      " auth_header=" +
      (config.auth_header ? "<set>" : "<none>")
  );
}

rl.on("line", function (line) {
  const trimmed = line.trim();
  if (!trimmed) return;

  if (trimmed === "__EOF__") {
    console.error("[workbench] EOF marker received, shutting down");
    process.exit(0);
  }

  let request;
  try {
    request = JSON.parse(trimmed);
  } catch (e) {
    console.error("[workbench] failed to parse JSON-RPC: " + e.message);
    return;
  }

  const method = request.method || "";
  const reqId = request.id;
  const params = request.params || {};

  if (method === "configure") {
    applyConfig(params);
    if (reqId !== undefined && reqId !== null) {
      sendJson(makeSuccess(reqId, {}));
    }
  } else if (method === "initialize") {
    if (reqId !== undefined && reqId !== null) {
      handleInitialize(reqId);
      initialized = true;
    }
  } else if (method === "notifications/initialized") {
    console.error("[workbench] client initialized notification received");
  } else if (method === "ping") {
    if (reqId !== undefined && reqId !== null) sendJson(makeSuccess(reqId, {}));
  } else if (method === "tools/list") {
    if (!initialized) {
      if (reqId !== undefined && reqId !== null)
        sendJson(makeError(reqId, -32000, "Server not initialized"));
      return;
    }
    if (reqId !== undefined && reqId !== null) handleToolsList(reqId);
  } else if (method === "notifications/cancelled") {
    // The client aborted an in-flight request (omniagent core__cancel_task
    // dropped the MCP call): abort the matching HTTP request so it is torn
    // down for real instead of running on unnoticed.
    const cancelId =
      params && params.requestId !== undefined ? params.requestId : undefined;
    const controller =
      cancelId !== undefined ? inFlight.get(cancelId) : undefined;
    if (controller) {
      inFlight.delete(cancelId);
      controller.abort();
      console.error(
        "[workbench] notifications/cancelled: aborted in-flight request " +
          cancelId
      );
    } else {
      console.error(
        "[workbench] notifications/cancelled: no in-flight request " +
          (cancelId === undefined ? "(missing requestId)" : cancelId)
      );
    }
  } else if (method === "tools/call") {
    if (!initialized) {
      if (reqId !== undefined && reqId !== null)
        sendJson(makeError(reqId, -32000, "Server not initialized"));
      return;
    }
    if (reqId !== undefined && reqId !== null) {
      handleCall(reqId, params).catch(function (err) {
        const reason = (err && err.message) || String(err);
        console.error("[workbench] tools/call failed: " + reason);
        sendJson(
          makeSuccess(reqId, toolResult("workbench call failed: " + reason, true))
        );
      });
    }
  } else {
    console.error("[workbench] unknown method: " + method);
    if (reqId !== undefined && reqId !== null) {
      sendJson(makeError(reqId, -32601, "Method not found: " + method));
    }
  }
});

rl.on("close", function () {
  console.error("[workbench] MCP server shutting down (stdin closed)");
  // Abort anything still in flight: stdin closed = the client is gone, no
  // consumer is left for the response.
  for (const controller of inFlight.values()) {
    controller.abort();
  }
  inFlight.clear();
  process.exit(0);
});
