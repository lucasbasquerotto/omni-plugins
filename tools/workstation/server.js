#!/usr/bin/env node

/**
 * workstation MCP server : a THIN fetch wrapper around the workstation HTTP API.
 *
 * It exposes exactly ONE MCP tool, `tool` (exposed to the agent as
 * `workstation__tool`) and forwards every call as a single HTTP POST:
 *
 *   POST {base_url}{tool_path}
 *   Content-Type: application/json
 *   body: {"tool": <tool>, "params": <params>}
 *
 * No business logic, no SDK, no state: the plugin never inspects, retries or
 * reshapes the workstation payload. The HTTP response body is returned to the
 * agent as pretty-printed JSON text.
 *
 * Config (config_schema in plugin.json, delivered as a `configure` JSON-RPC
 * request and/or as environment variables):
 *   base_url             default http://workstation:8080
 *   tool_path            default /api/tool/call
 *   timeout_secs         default 0 (0 = NO timeout; positive = operator opt-in)
 *   headers_timeout_secs default 0 (seconds to wait for the response headers;
 *                        0 = disabled, no clock at all)
 *   body_timeout_secs    default 0 (seconds of response-body inactivity
 *                        before the request is aborted; 0 = disabled)
 *   auth_header          optional Authorization header value (empty = omitted)
 *
 * Runtime: Node >= 18, stdio JSON-RPC, **no npm deps**. The HTTP forward is a
 * plain node:http / node:https POST, NOT Node's global `fetch`. Global fetch is
 * undici, whose hidden `headersTimeout` / `bodyTimeout` default to 300 s and
 * abort a long workstation run client-side even when `timeout_secs` is 0 (the
 * server keeps working, the response is lost). This request has no hidden
 * clock: the only bounds are the two `*_timeout_secs` knobs above (0 =
 * disabled) plus the AbortController the caller passes.
 */

const readline = require("readline");
const process = require("process");
const http = require("http");
const https = require("https");

const MCP_PROTOCOL_VERSION = "2025-03-26";
const SERVER_NAME = "workstation";
const SERVER_VERSION = "0.1.0";

const DEFAULT_BASE_URL = "http://workstation:8080";
const DEFAULT_TOOL_PATH = "/api/tool/call";
// Timeout semantics for `timeout_secs`:
//   absent / empty / 0 -> NO timeout: the request runs until the workstation
//                         answers, the connection fails, or the CLIENT cancels
//                         it (omniagent `core__cancel_task` drops the in-flight
//                         MCP call and sends `notifications/cancelled`, which
//                         aborts the HTTP request here). Workstation dispatches
//                         are legitimately long-running: a hidden clock must
//                         never kill them.
//   positive integer   -> explicit operator opt-in timeout in seconds.
const NO_TIMEOUT_SECS = 0;
const DEFAULT_TIMEOUT_SECS = NO_TIMEOUT_SECS;

// Explicit HTTP clocks replacing undici's hidden 300 s defaults. 0 = disabled
// (the request waits until the server answers, the connection fails, or the
// client aborts). Operator requirement (2026-09-30): NO timeout at all — the
// agent manages long runs via core__wait_task / core__cancel_task only.
// Positive values are an explicit operator opt-in per deployment.
const DEFAULT_HEADERS_TIMEOUT_SECS = 0;
const DEFAULT_BODY_TIMEOUT_SECS = 0;

// The single declared tool name. `tool_qualify("workstation", "tool")` in core
// turns it into the exposed name `workstation__tool`.
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

/** Human-readable HTTP clock for the logs: "none" when the clock is disabled. */
function describeClock(secs) {
  return secs > 0 ? String(secs) : "none";
}

const config = {
  base_url: pickString(
    [process.env.base_url, process.env.BASE_URL, process.env.WORKSTATION_BASE_URL],
    DEFAULT_BASE_URL
  ),
  tool_path: pickString(
    [process.env.tool_path, process.env.TOOL_PATH, process.env.WORKSTATION_TOOL_PATH],
    DEFAULT_TOOL_PATH
  ),
  timeout_secs: pickTimeoutSecs(
    [
      process.env.timeout_secs,
      process.env.TIMEOUT_SECS,
      process.env.WORKSTATION_TIMEOUT_SECS,
    ],
    DEFAULT_TIMEOUT_SECS
  ),
  headers_timeout_secs: pickTimeoutSecs(
    [
      process.env.headers_timeout_secs,
      process.env.HEADERS_TIMEOUT_SECS,
      process.env.WORKSTATION_HEADERS_TIMEOUT_SECS,
    ],
    DEFAULT_HEADERS_TIMEOUT_SECS
  ),
  body_timeout_secs: pickTimeoutSecs(
    [
      process.env.body_timeout_secs,
      process.env.BODY_TIMEOUT_SECS,
      process.env.WORKSTATION_BODY_TIMEOUT_SECS,
    ],
    DEFAULT_BODY_TIMEOUT_SECS
  ),
  auth_header: pickString(
    [
      process.env.auth_header,
      process.env.AUTH_HEADER,
      process.env.WORKSTATION_AUTH_HEADER,
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
  if (payload.headers_timeout_secs !== undefined)
    config.headers_timeout_secs = pickTimeoutSecs(
      [payload.headers_timeout_secs],
      config.headers_timeout_secs
    );
  if (payload.body_timeout_secs !== undefined)
    config.body_timeout_secs = pickTimeoutSecs(
      [payload.body_timeout_secs],
      config.body_timeout_secs
    );
  if (payload.auth_header !== undefined)
    config.auth_header = pickString([payload.auth_header], "");
  console.error(
    "[workstation] config: base_url=" +
      config.base_url +
      " tool_path=" +
      config.tool_path +
      " timeout_secs=" +
      describeTimeout() +
      " headers_timeout_secs=" +
      describeClock(config.headers_timeout_secs) +
      " body_timeout_secs=" +
      describeClock(config.body_timeout_secs) +
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
  console.error("[workstation] initialized: " + SERVER_NAME + " v" + SERVER_VERSION);
}

function handleToolsList(reqId) {
  const tools = [
    {
      name: TOOL_NAME,
      description:
        "[workstation] Call a workstation tool by name: POSTs {\"tool\": <tool>, " +
        "\"params\": <params>} to the workstation HTTP API (base_url + tool_path) " +
        "and returns the response body. Thin wrapper, no business logic.",
      inputSchema: {
        type: "object",
        properties: {
          tool: {
            type: "string",
            description: "workstation tool/command name, e.g. hello_world",
          },
          params: {
            type: "object",
            additionalProperties: true,
            default: {},
            description: "arguments passed to the workstation tool",
          },
        },
        required: ["tool"],
      },
    },
  ];
  sendJson(makeSuccess(reqId, { tools }));
  console.error("[workstation] tools/list returned 1 tool");
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

// ── HTTP forwarding ────────────────────────────────────────────────────────
//
// Plain node:http / node:https POST. Node's global fetch is undici, which
// enforces a hidden 300 s headersTimeout/bodyTimeout independent of the
// plugin's own `timeout_secs` AbortController; a long workstation run (the
// server answers only when the worker finishes) was aborted client-side at
// exactly 300 s even with `timeout_secs: 0`. This request has NO hidden clock:
// the only bounds are `headers_timeout_secs` / `body_timeout_secs` (0 =
// disabled) plus the AbortController the caller passes.
//
// Resolves with a fetch-like response object `{ status, statusText, ok, text() }`
// as soon as the response headers arrive; `text()` resolves with the full body
// (or rejects on a body error / body-timeout / client cancel).

function requestWorkstation(url, headers, bodyText, signal) {
  return new Promise((resolve, reject) => {
    let parsed;
    try {
      parsed = new URL(url);
    } catch (err) {
      reject(err);
      return;
    }

    const headersTimeoutSecs = config.headers_timeout_secs;
    const bodyTimeoutSecs = config.body_timeout_secs;
    const lib = parsed.protocol === "https:" ? https : http;

    let headersTimer = null;
    let bodyTimer = null;
    let responseStarted = false;
    let bodyFinished = false;
    let bodyResolve = null;
    let bodyReject = null;
    const bodyPromise = new Promise((res, rej) => {
      bodyResolve = res;
      bodyReject = rej;
    });
    // The caller only awaits bodyPromise once a response exists; if the
    // request dies before that, do not surface an unhandled rejection.
    bodyPromise.catch(function () {});

    function clearHeadersTimer() {
      if (headersTimer) {
        clearTimeout(headersTimer);
        headersTimer = null;
      }
    }

    function clearBodyTimer() {
      if (bodyTimer) {
        clearTimeout(bodyTimer);
        bodyTimer = null;
      }
    }

    function finishBody(err, text) {
      if (bodyFinished) return;
      bodyFinished = true;
      clearBodyTimer();
      if (err) bodyReject(err);
      else bodyResolve(text);
    }

    function armBodyTimer(res) {
      if (bodyTimeoutSecs <= 0) return; // clock disabled
      clearBodyTimer();
      bodyTimer = setTimeout(function () {
        const err = new Error(
          "body timeout after " + bodyTimeoutSecs + "s (body_timeout_secs)"
        );
        err.name = "BodyTimeoutError";
        res.destroy(err);
        req.destroy(err);
      }, bodyTimeoutSecs * 1000);
    }

    const req = lib.request(
      {
        protocol: parsed.protocol,
        hostname: parsed.hostname,
        port: parsed.port || undefined,
        path: parsed.pathname + parsed.search,
        method: "POST",
        headers: headers,
      },
      function (res) {
        responseStarted = true;
        clearHeadersTimer();
        armBodyTimer(res);
        const chunks = [];
        res.on("data", function (chunk) {
          chunks.push(chunk);
          armBodyTimer(res);
        });
        res.on("end", function () {
          finishBody(null, Buffer.concat(chunks).toString("utf8"));
        });
        res.on("error", function (err) {
          finishBody(err);
        });
        resolve({
          status: res.statusCode,
          statusText: res.statusMessage || "",
          ok: res.statusCode >= 200 && res.statusCode < 300,
          text: function () {
            return bodyPromise;
          },
        });
      }
    );

    req.on("error", function (err) {
      clearHeadersTimer();
      if (!responseStarted) {
        reject(err);
        return;
      }
      finishBody(err);
    });

    function onAbort() {
      const err = new Error("The operation was aborted");
      err.name = "AbortError";
      req.destroy(err);
    }
    if (signal) {
      if (signal.aborted) onAbort();
      else signal.addEventListener("abort", onAbort, { once: true });
    }

    if (headersTimeoutSecs > 0) {
      headersTimer = setTimeout(function () {
        const err = new Error(
          "headers timeout after " + headersTimeoutSecs + "s (headers_timeout_secs)"
        );
        err.name = "HeadersTimeoutError";
        req.destroy(err);
      }, headersTimeoutSecs * 1000);
    }

    req.end(bodyText);
  });
}

// In-flight tools/call requests: JSON-RPC request id -> AbortController, so a
// `notifications/cancelled` from the client tears the HTTP request down for
// real (core__cancel_task must abort the in-flight call, not just flag it).
const inFlight = new Map();

async function callWorkstation(toolName, params, controller) {
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
  // workstation answers, the connection fails, or the client cancels it.
  const timer =
    timeout_secs > 0
      ? setTimeout(() => abort.abort(), timeout_secs * 1000)
      : null;

  console.error("[workstation] POST " + url + " body=" + body);

  let response;
  try {
    response = await requestWorkstation(url, headers, body, abort.signal);
  } catch (err) {
    if (timer) clearTimeout(timer);
    const aborted = err && err.name === "AbortError";
    const reason = aborted
      ? timeout_secs > 0
        ? "request timed out after " + timeout_secs + "s"
        : "request cancelled by the client"
      : (err && err.message) || String(err);
    console.error("[workstation] request failed: " + reason);
    return {
      isError: true,
      text: "workstation request to " + url + " failed: " + reason,
    };
  }
  if (timer) clearTimeout(timer);

  let bodyText = "";
  try {
    bodyText = await response.text();
  } catch (err) {
    const reason = (err && err.message) || String(err);
    console.error("[workstation] failed to read response body: " + reason);
    return {
      isError: true,
      text:
        "workstation " +
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
    console.error("[workstation] HTTP " + response.status + " from " + url);
    return {
      isError: true,
      text:
        "workstation " +
        url +
        " returned HTTP " +
        response.status +
        statusText +
        "\n" +
        pretty,
    };
  }

  console.error("[workstation] HTTP " + response.status + " from " + url);
  return { isError: false, text: pretty };
}

async function handleCall(reqId, params) {
  const toolName = params.name || "";
  const args = params.arguments || {};

  if (toolName !== TOOL_NAME && toolName !== SERVER_NAME + "__" + TOOL_NAME) {
    sendJson(makeError(reqId, -32602, "Unknown tool: " + toolName));
    return;
  }

  const workstationTool = args.tool;
  if (typeof workstationTool !== "string" || workstationTool.trim() === "") {
    sendJson(
      makeError(reqId, -32602, "Invalid params: 'tool' is required (string)")
    );
    console.error("[workstation] tools/call rejected: missing 'tool' argument");
    return;
  }

  let workstationParams = args.params;
  if (workstationParams === undefined || workstationParams === null) {
    workstationParams = {};
  } else if (
    typeof workstationParams !== "object" ||
    Array.isArray(workstationParams)
  ) {
    sendJson(
      makeError(reqId, -32602, "Invalid params: 'params' must be an object")
    );
    console.error("[workstation] tools/call rejected: 'params' is not an object");
    return;
  }

  // NOT awaited by the readline loop: calls are concurrent (150 parallel
  // calls must all be in flight without blocking each other) - which is also
  // what lets a `notifications/cancelled` be processed WHILE this call awaits
  // the HTTP response.
  const controller = new AbortController();
  inFlight.set(reqId, controller);
  try {
    const outcome = await callWorkstation(
      workstationTool,
      workstationParams,
      controller
    );
    sendJson(makeSuccess(reqId, toolResult(outcome.text, outcome.isError)));
  } finally {
    inFlight.delete(reqId);
  }
}

// ── Main loop ──────────────────────────────────────────────────────────────

const rl = readline.createInterface({ input: process.stdin, terminal: false });

console.error("[workstation] MCP server starting (PID=" + process.pid + ")");
applyConfigFromEnvSummary();

function applyConfigFromEnvSummary() {
  console.error(
    "[workstation] env config: base_url=" +
      config.base_url +
      " tool_path=" +
      config.tool_path +
      " timeout_secs=" +
      describeTimeout() +
      " headers_timeout_secs=" +
      describeClock(config.headers_timeout_secs) +
      " body_timeout_secs=" +
      describeClock(config.body_timeout_secs) +
      " auth_header=" +
      (config.auth_header ? "<set>" : "<none>")
  );
}

rl.on("line", function (line) {
  const trimmed = line.trim();
  if (!trimmed) return;

  if (trimmed === "__EOF__") {
    console.error("[workstation] EOF marker received, shutting down");
    process.exit(0);
  }

  let request;
  try {
    request = JSON.parse(trimmed);
  } catch (e) {
    console.error("[workstation] failed to parse JSON-RPC: " + e.message);
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
    console.error("[workstation] client initialized notification received");
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
        "[workstation] notifications/cancelled: aborted in-flight request " +
          cancelId
      );
    } else {
      console.error(
        "[workstation] notifications/cancelled: no in-flight request " +
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
        console.error("[workstation] tools/call failed: " + reason);
        sendJson(
          makeSuccess(reqId, toolResult("workstation call failed: " + reason, true))
        );
      });
    }
  } else {
    console.error("[workstation] unknown method: " + method);
    if (reqId !== undefined && reqId !== null) {
      sendJson(makeError(reqId, -32601, "Method not found: " + method));
    }
  }
});

rl.on("close", function () {
  console.error("[workstation] MCP server shutting down (stdin closed)");
  // Abort anything still in flight: stdin closed = the client is gone, no
  // consumer is left for the response.
  for (const controller of inFlight.values()) {
    controller.abort();
  }
  inFlight.clear();
  process.exit(0);
});
