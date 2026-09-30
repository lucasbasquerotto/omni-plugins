# workstation (MCP tool plugin)

A **thin fetch wrapper** over the [workstation](https://github.com/nexuslbs/workstation)
HTTP API. It exposes exactly **one** MCP tool, `tool` - exposed to the agent as
**`workstation__tool`** - and forwards every call as a single HTTP request. There
is no business logic, no SDK, no state and no retry/backoff: the plugin never
inspects or reshapes the workstation payload.

## Tool

`workstation__tool`

| arg | type | required | description |
| --- | ---- | -------- | ----------- |
| `tool` | string | yes | workstation tool/command name, e.g. `hello_world` |
| `params` | object | no (default `{}`) | arguments passed to the workstation tool |

Agent call example:

```json
{"name": "workstation__tool", "arguments": {"tool": "hello_world", "params": {}}}
```

## HTTP contract

```
POST {base_url}{tool_path}
Content-Type: application/json
Accept: application/json

{"tool": "<tool>", "params": {<params>}}
```

Defaults: `base_url = http://workstation:8080` (the workstation compose service DNS
name inside the omni docker network; the dev overlay publishes host port
`12347`), `tool_path = /api/tool/call`.

> **Workstation-side endpoint status (2026-09-19).** The workstation core web seam
> currently registers only `GET /api/web/pages`
> (`src/web/providers/http.ts`) plus the shell assets and the `/health` status
> endpoint, and the workstation UI plugins register their own UI routes
> (`settings`, `plugin-inventory`, `plugin-manager`, `cordis-ui`). There is
> **no by-name tool/command invocation endpoint** in workstation today: commands
> are invoked through the CLI registry (`workstation <command> [args...]`).
> This wrapper therefore implements the agreed contract above with
> `tool_path = /api/tool/call`; if/when workstation exposes a real invocation
> route, only `tool_path` changes (a config value, no code change). Closing the
> gap is workstation-side work, tracked as `task_workstation_workstation_expose_a_by_name_tool`
> on the `workstation` board (channel `workstation`).

### Observed live in omnidev (2026-09-19)

- `GET http://workstation:12347/health` -> `200`. In the dev stack the workstation
  answers on `12347` while `http://workstation:8080` does not answer from the
  omniagent container, so the **dev plugin config overrides**
  `base_url = http://workstation:12347`; the shipped default stays
  `http://workstation:8080` (the compose `WORKSTATION_PORT` default).
- `GET http://workstation:12347/api/web/pages` -> `200` (the `web@1` contract and
  its route list).
- `POST http://workstation:12347/api/tool/call` -> `404
  {"status":"not found","method":"POST","path":"/api/tool/call"}` - the
  workstation router's own not-found envelope, which the wrapper forwards
  verbatim as `isError: true` (`GET /api/tools`, `/api/commands`, `/api/routes`
  and `/openapi.json` are `404` as well).

## Response handling

- 2xx: the response body is returned as MCP text content, pretty-printed when
  it is JSON.
- Non-2xx: `isError: true` with the status code, status text and body.
- Connection refused / DNS failure / timeout (`AbortController`,
  `timeout_secs`) / unreadable body: a clear `isError` message. Never a crash,
  never a hang.
- Missing/invalid `tool` argument: JSON-RPC error `-32602`.

Calls are **concurrent**: the stdio readline loop never awaits a call, so 150
parallel `workstation__tool` calls run in flight together.

## Config (`plugin.json` `config_schema`)

| key | type | default | description |
| --- | ---- | ------- | ----------- |
| `base_url` | string | `http://workstation:8080` | workstation HTTP API base URL |
| `tool_path` | string | `/api/tool/call` | path appended to `base_url` |
| `timeout_secs` | integer | `60` | HTTP timeout per call |
| `headers_timeout_secs` | integer | `900` | seconds to wait for the response headers; `0` = disabled (no clock) |
| `body_timeout_secs` | integer | `900` | seconds of response-body inactivity before abort; `0` = disabled (no clock) |
| `auth_header` | string | `""` | optional `Authorization` header value (e.g. `Bearer $secret:WORKSTATION_TOKEN`) |

Values reach the plugin through the `configure` JSON-RPC request and/or as
environment variables (`base_url` / `BASE_URL` / `WORKSTATION_BASE_URL`). Secret
references use `$env:VAR` / `$secret:NAME` (never a literal `${VAR}`).

## Runtime

Node >= 18 (`node:http` / `node:https` + `AbortController`), stdio JSON-RPC like
`tools/test-js-tool/server.js`. **No npm dependencies.**

The forward is deliberately **not** Node's global `fetch`: global fetch is
undici, whose hidden `headersTimeout` / `bodyTimeout` default to 300 s and abort
a long workstation run client-side even when `timeout_secs` is 0 (the server
keeps working, the response is lost). The request has no hidden clock: the only
bounds are `headers_timeout_secs` / `body_timeout_secs` (0 = disabled) and the
caller's `AbortController` (`timeout_secs` / `notifications/cancelled`).

## Test harness

`test/mcp-workstation.test.js` (zero deps, Node >= 18) runs `server.js` over stdio
against a local stub HTTP server and asserts the whole contract: one tool with
the declared schema, exactly one byte-exact `POST {base_url}{tool_path}`, the
pretty-printed text content, the error paths (404/500, connection refused,
`AbortController` timeout, JSON-RPC `-32602`) and 150 concurrent calls.

```bash
node test/mcp-workstation.test.js    # exit 0 = all PASS
```

## Local smoke test

```bash
node --check server.js
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
  | node server.js
```
