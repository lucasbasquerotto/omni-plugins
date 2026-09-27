#!/usr/bin/env node
/*
 * workbench MCP plugin verification (zero deps, Node >= 18).
 *
 * Runs the plugin's `server.js` over stdio JSON-RPC against a local stub HTTP
 * server and asserts the thin-wrapper contract:
 *   - tools/list exposes EXACTLY one tool `tool` with the declared schema
 *   - tools/call issues exactly ONE POST {base_url}{tool_path} with body
 *     {"tool": <tool>, "params": <params>} (byte-exact) and returns the body
 *   - non-2xx / connection refused / timeout / invalid args -> clear isError
 *     or JSON-RPC -32602, never a crash or a hang
 *   - 150 concurrent calls complete
 *
 * Usage: node test/mcp-workbench.test.js      (from tools/workbench)
 * Exit code 0 = all PASS, 1 = at least one FAIL.
 */
"use strict";

const http = require("http");
const path = require("path");
const { spawn } = require("child_process");

const PLUGIN_DIR = path.join(__dirname, "..");
// The plugin is exposed as `{dirname}__tool` (workbench__tool /
// workstation__tool) and reports that same name as its serverInfo name, so the
// harness derives it from its own directory instead of hardcoding it.
const PLUGIN_NAME = path.basename(PLUGIN_DIR);
const results = [];
let failures = 0;

function check(name, cond, detail) {
  const ok = !!cond;
  if (!ok) failures++;
  const line = `${ok ? "PASS" : "FAIL"} :: ${name}${detail ? " :: " + detail : ""}`;
  results.push(line);
  console.log(line);
}

// ── stub HTTP server ───────────────────────────────────────────────────────

const stubRequests = [];
let hangMode = false;
let slowMs = 1800; // /slow delays its response by this much (no-timeout gates)
let stubAborted = false; // a client tore an in-flight request down (timeout/cancel)

function startStub() {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const chunks = [];
      res.on("error", () => {}); // the client may disappear (timeout/cancel)
      req.on("aborted", () => {
        stubAborted = true;
      });
      res.on("close", () => {
        if (!res.writableEnded) stubAborted = true;
      });
      req.on("data", (c) => chunks.push(c));
      req.on("end", () => {
        const raw = Buffer.concat(chunks).toString("utf8");
        stubRequests.push({
          method: req.method,
          url: req.url,
          headers: req.headers,
          raw,
        });
        if (hangMode) return; // never respond -> AbortController timeout
        if (req.url === "/nope") {
          res.writeHead(404, { "content-type": "application/json" });
          res.end('{"status":"not found","method":"POST","path":"/nope"}');
          return;
        }
        if (req.url === "/boom") {
          res.writeHead(500, { "content-type": "text/plain" });
          res.end("boom");
          return;
        }
        if (req.url === "/slow") {
          setTimeout(() => {
            try {
              res.writeHead(200, { "content-type": "application/json" });
              res.end(JSON.stringify({ ok: true, slow: true }));
            } catch (e) {
              /* the client is gone (timeout/cancel) */
            }
          }, slowMs);
          return;
        }
        if (req.url === "/badjson") {
          res.writeHead(200, { "content-type": "text/plain" });
          res.end("<<not json>>");
          return;
        }
        res.writeHead(200, { "content-type": "application/json" });
        let received = null;
        try {
          received = JSON.parse(raw);
        } catch (e) {
          received = { parse_error: e.message };
        }
        res.end(JSON.stringify({ ok: true, received }));
      });
    });
    server.listen(0, "127.0.0.1", () => resolve(server));
  });
}

// ── MCP stdio client ───────────────────────────────────────────────────────

function startServer(env) {
  const child = spawn(process.execPath, ["server.js"], {
    cwd: PLUGIN_DIR,
    env: Object.assign({}, process.env, env || {}),
    stdio: ["pipe", "pipe", "pipe"],
  });
  const client = { child, pending: new Map(), seq: 0, buf: "", stderr: "" };
  child.stdout.on("data", (d) => {
    client.buf += d.toString();
    let idx;
    while ((idx = client.buf.indexOf("\n")) >= 0) {
      const line = client.buf.slice(0, idx);
      client.buf = client.buf.slice(idx + 1);
      if (!line.trim()) continue;
      let msg;
      try {
        msg = JSON.parse(line);
      } catch (e) {
        continue;
      }
      const p = client.pending.get(msg.id);
      if (p) {
        client.pending.delete(msg.id);
        if (msg.error) {
          const err = new Error(msg.error.message);
          err.code = msg.error.code;
          p.reject(err);
        } else {
          p.resolve(msg.result);
        }
      }
    }
  });
  child.stderr.on("data", (d) => {
    client.stderr += d.toString();
  });
  client.call = (method, params) =>
    new Promise((resolve, reject) => {
      const id = ++client.seq;
      client.pending.set(id, { resolve, reject });
      client.write = null;
      child.stdin.write(
        JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n"
      );
    });
  client.notify = (method, params) =>
    child.stdin.write(JSON.stringify({ jsonrpc: "2.0", method, params }) + "\n");
  client.stop = () => child.kill();
  return client;
}

async function expectRpcError(promise, code, label) {
  try {
    await promise;
    check(label, false, "expected JSON-RPC error, got a result");
  } catch (err) {
    check(label, err.code === code, `code=${err.code} message=${err.message}`);
  }
}

// ── scenarios ──────────────────────────────────────────────────────────────

async function main() {
  const stub = await startStub();
  const port = stub.address().port;
  const base = `http://127.0.0.1:${port}`;

  // Gate 1: tools/list over a real stdio smoke run (provides the raw exchange).
  const rawExchange = [];
  const client = startServer({ base_url: base, tool_path: "/echo", timeout_secs: "5" });
  const init = await client.call("initialize", {
    protocolVersion: "2025-03-26",
    capabilities: {},
    clientInfo: { name: "tester", version: "1.0.0" },
  });
  rawExchange.push({ initialize: init });
  client.notify("notifications/initialized", {});

  const list = await client.call("tools/list", {});
  rawExchange.push({ "tools/list": list });
  console.log("RAW tools/list: " + JSON.stringify(list));
  check("initialize -> serverInfo.name === " + PLUGIN_NAME, init.serverInfo && init.serverInfo.name === PLUGIN_NAME, JSON.stringify(init.serverInfo));
  check("tools/list returns EXACTLY one tool", Array.isArray(list.tools) && list.tools.length === 1, "count=" + (list.tools || []).length);
  const tool = (list.tools || [])[0] || {};
  check("tool name === 'tool' (exposed as workbench__tool)", tool.name === "tool", "name=" + tool.name);
  const schema = tool.inputSchema || {};
  check("schema.required === ['tool']", Array.isArray(schema.required) && schema.required.length === 1 && schema.required[0] === "tool", JSON.stringify(schema.required));
  check("schema.properties.tool.type === string", schema.properties && schema.properties.tool && schema.properties.tool.type === "string");
  check("schema.properties.params: object, additionalProperties, default {}", !!(schema.properties && schema.properties.params && schema.properties.params.type === "object" && schema.properties.params.additionalProperties === true && schema.properties.params.default && typeof schema.properties.params.default === "object"), JSON.stringify(schema.properties && schema.properties.params));

  // Gate 2: forwarding proof (single POST, byte-exact body, text content).
  stubRequests.length = 0;
  const call1 = await client.call("tools/call", { name: "tool", arguments: { tool: "hello_world", params: {} } });
  check("tools/call ok (isError falsy)", !call1.isError, JSON.stringify(call1 && call1.isError));
  check("forwarding: exactly ONE HTTP request", stubRequests.length === 1, "count=" + stubRequests.length);
  const r0 = stubRequests[0] || {};
  check("forwarding: POST {base_url}/echo", r0.method === "POST" && r0.url === "/echo", `${r0.method} ${r0.url}`);
  check("forwarding: request body byte-exact {\"tool\":\"hello_world\",\"params\":{}}", r0.raw === '{"tool":"hello_world","params":{}}', "raw=" + r0.raw);
  check("forwarding: Content-Type application/json", String((r0.headers || {})["content-type"] || "").indexOf("application/json") === 0, String((r0.headers || {})["content-type"]));
  const expectedText = JSON.stringify({ ok: true, received: { tool: "hello_world", params: {} } }, null, 2);
  check("response body returned as pretty-printed MCP text content", call1.content && call1.content[0] && call1.content[0].type === "text" && call1.content[0].text === expectedText, JSON.stringify(call1.content));

  // params omitted -> {} ; params passed through untouched
  stubRequests.length = 0;
  await client.call("tools/call", { name: "tool", arguments: { tool: "x" } });
  check("params omitted -> body {\"tool\":\"x\",\"params\":{}}", stubRequests[0] && stubRequests[0].raw === '{"tool":"x","params":{}}', stubRequests[0] && stubRequests[0].raw);
  stubRequests.length = 0;
  await client.call("tools/call", { name: "tool", arguments: { tool: "t", params: { a: 1, b: [1, 2], c: { d: "e" } } } });
  check("params passed through untouched", stubRequests[0] && stubRequests[0].raw === '{"tool":"t","params":{"a":1,"b":[1,2],"c":{"d":"e"}}}', stubRequests[0] && stubRequests[0].raw);

  // qualified name also accepted (PLUGIN_NAME__tool)
  stubRequests.length = 0;
  const qualified = await client.call("tools/call", { name: PLUGIN_NAME + "__tool", arguments: { tool: "q", params: {} } });
  check("qualified name " + PLUGIN_NAME + "__tool accepted", !qualified.isError && stubRequests.length === 1, "isError=" + qualified.isError);

  // Gate 4a: non-2xx
  await client.call("configure", { tool_path: "/nope" });
  stubRequests.length = 0;
  const notFound = await client.call("tools/call", { name: "tool", arguments: { tool: "hello_world", params: {} } });
  check("404 -> isError true", notFound.isError === true);
  check("404 -> status + body in message", /404/.test(notFound.content[0].text) && /not found/.test(notFound.content[0].text), JSON.stringify(notFound.content[0].text).slice(0, 200));

  await client.call("configure", { tool_path: "/boom" });
  const boom = await client.call("tools/call", { name: "tool", arguments: { tool: "hello_world", params: {} } });
  check("500 -> isError true with status", boom.isError === true && /500/.test(boom.content[0].text), JSON.stringify(boom.content[0].text).slice(0, 200));

  // Gate 4b: connection refused
  await client.call("configure", { base_url: "http://127.0.0.1:1", tool_path: "/echo" });
  const refused = await client.call("tools/call", { name: "tool", arguments: { tool: "hello_world", params: {} } });
  check("connection refused -> isError true", refused.isError === true, JSON.stringify(refused.content[0].text).slice(0, 200));

  // Gate 4c: timeout (no hang)
  await client.call("configure", { base_url: base, tool_path: "/hang", timeout_secs: 1 });
  hangMode = true;
  const t0 = Date.now();
  const timedOut = await client.call("tools/call", { name: "tool", arguments: { tool: "hello_world", params: {} } });
  const elapsed = Date.now() - t0;
  hangMode = false;
  check("timeout -> isError true, no hang", timedOut.isError === true && elapsed < 10000, `elapsed=${elapsed}ms text=${JSON.stringify(timedOut.content[0].text).slice(0, 160)}`);

  // Gate 4c-2: `timeout_secs` absent / 0 = NO timeout (the workstation tool is
  // long-running by design): the SAME slow endpoint that a 1s timeout kills
  // must complete when no timeout is configured, and the plugin must report the
  // effective mode as `none`.
  await client.call("configure", { base_url: base, tool_path: "/slow", timeout_secs: 0 });
  const slowStart = Date.now();
  const slowOk = await client.call("tools/call", { name: "tool", arguments: { tool: "hello_world", params: {} } });
  const slowElapsed = Date.now() - slowStart;
  check(
    "timeout_secs=0 -> slow call completes (no timer armed)",
    !slowOk.isError && slowElapsed >= slowMs && /slow/.test(slowOk.content[0].text),
    `elapsed=${slowElapsed}ms text=${JSON.stringify(slowOk.content[0].text).slice(0, 120)}`
  );
  check(
    "timeout_secs=0 -> plugin reports effective mode 'none'",
    /timeout_secs=none/.test(client.stderr),
    client.stderr.split("\n").filter((l) => l.indexOf("timeout_secs") >= 0).slice(-1).join("")
  );
  // a positive value stays backwards compatible: 1s kills the same slow call
  await client.call("configure", { tool_path: "/slow", timeout_secs: 1 });
  const slowKilled = await client.call("tools/call", { name: "tool", arguments: { tool: "hello_world", params: {} } });
  check(
    "timeout_secs=1 -> the same slow call is still killed",
    slowKilled.isError === true && /timed out after 1s/.test(slowKilled.content[0].text),
    JSON.stringify(slowKilled.content[0].text).slice(0, 140)
  );

  // Gate 4c-3: no timeout_secs anywhere (env + config absent) -> no timer either
  const clientNoTimeout = startServer({ base_url: base, tool_path: "/slow" });
  await clientNoTimeout.call("initialize", {
    protocolVersion: "2025-03-26",
    capabilities: {},
    clientInfo: { name: "tester", version: "1.0.0" },
  });
  clientNoTimeout.notify("notifications/initialized", {});
  const noTimeoutStart = Date.now();
  const noTimeoutCall = await clientNoTimeout.call("tools/call", { name: "tool", arguments: { tool: "hello_world", params: {} } });
  const noTimeoutElapsed = Date.now() - noTimeoutStart;
  check(
    "no timeout_secs configured -> slow call completes (default is no timer)",
    !noTimeoutCall.isError && noTimeoutElapsed >= slowMs,
    `elapsed=${noTimeoutElapsed}ms text=${JSON.stringify(noTimeoutCall.content[0].text).slice(0, 120)}`
  );
  clientNoTimeout.stop();

  // Gate 4c-4: `notifications/cancelled` tears the in-flight HTTP request down
  // for real (this is the path core__cancel_task drives): the stub must observe
  // the connection closing, and the call must answer a cancelled error instead
  // of hanging the turn.
  await client.call("configure", { base_url: base, tool_path: "/hang", timeout_secs: 0 });
  hangMode = true;
  stubAborted = false;
  const cancelledCall = client.call("tools/call", { name: "tool", arguments: { tool: "hello_world", params: {} } });
  const cancelledId = client.seq; // client.call assigns the id synchronously
  await new Promise((r) => setTimeout(r, 200));
  client.notify("notifications/cancelled", { requestId: cancelledId, reason: "tester cancel" });
  const cancelResult = await Promise.race([
    cancelledCall,
    new Promise((r) => setTimeout(() => r({ isError: true, content: [{ type: "text", text: "NO ANSWER within 5s" }] }), 5000)),
  ]);
  await new Promise((r) => setTimeout(r, 100));
  hangMode = false;
  check("notifications/cancelled -> in-flight HTTP request torn down at the server", stubAborted === true, "stubAborted=" + stubAborted);
  check(
    "notifications/cancelled -> call answers an explicit cancelled error (never hangs)",
    cancelResult.isError === true && /cancel/i.test(cancelResult.content[0].text),
    JSON.stringify(cancelResult.content[0].text).slice(0, 140)
  );
  const aliveAfterCancel = await client.call("tools/list", {});
  check("server alive after a cancelled call", !!(aliveAfterCancel.tools && aliveAfterCancel.tools.length === 1));

  // Gate 4d: invalid args -> JSON-RPC -32602
  await client.call("configure", { tool_path: "/echo", timeout_secs: 5 });
  await expectRpcError(client.call("tools/call", { name: "tool", arguments: {} }), -32602, "missing 'tool' -> JSON-RPC -32602");
  await expectRpcError(client.call("tools/call", { name: "tool", arguments: { tool: "   " } }), -32602, "blank 'tool' -> JSON-RPC -32602");
  await expectRpcError(client.call("tools/call", { name: "tool", arguments: { tool: "x", params: [1, 2] } }), -32602, "array 'params' -> JSON-RPC -32602");
  await expectRpcError(client.call("tools/call", { name: "unknown_tool", arguments: { tool: "x" } }), -32602, "unknown MCP tool -> JSON-RPC -32602");

  // non-JSON body still returns text, not a crash
  await client.call("configure", { tool_path: "/badjson" });
  const badJson = await client.call("tools/call", { name: "tool", arguments: { tool: "x", params: {} } });
  check("non-JSON 2xx body -> raw text, isError falsy", badJson.isError === false && badJson.content[0].text === "<<not json>>", JSON.stringify(badJson.content[0].text));

  // Gate 5: 150 concurrent calls
  await client.call("configure", { tool_path: "/echo", timeout_secs: 30 });
  stubRequests.length = 0;
  const t1 = Date.now();
  const calls = [];
  for (let i = 0; i < 150; i++) {
    calls.push(client.call("tools/call", { name: "tool", arguments: { tool: "tool_" + i, params: { i } } }));
  }
  const settled = await Promise.allSettled(calls);
  const concurrencyMs = Date.now() - t1;
  const okCount = settled.filter((s) => s.status === "fulfilled" && !s.value.isError).length;
  check("150 concurrent calls all ok", okCount === 150, `ok=${okCount}/150 elapsed=${concurrencyMs}ms`);
  check("150 concurrent calls -> 150 HTTP requests", stubRequests.length === 150, "requests=" + stubRequests.length);

  // server is still alive after all error paths
  const alive = await client.call("tools/list", {});
  check("server alive after error paths + concurrency", alive.tools && alive.tools.length === 1);
  check("no unhandled crash in stderr", !/UnhandledPromiseRejection|ERR_UNHANDLED|Cannot read propert/.test(client.stderr), client.stderr.split("\n").slice(-3).join(" | "));

  client.stop();
  stub.close();

  console.log("\n=== RAW JSON-RPC EXCHANGE (gate 1) ===");
  console.log(JSON.stringify(rawExchange, null, 2));
  console.log(`\nRESULT: ${results.length - failures}/${results.length} PASS`);
  console.log(failures === 0 ? PLUGIN_NAME.toUpperCase() + "_PLUGIN_TEST: ALL PASS" : PLUGIN_NAME.toUpperCase() + `_PLUGIN_TEST: ${failures} FAILED`);
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((err) => {
  console.error("harness error: " + (err && err.stack ? err.stack : err));
  process.exit(2);
});
