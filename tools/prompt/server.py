#!/usr/bin/env python3
"""prompt MCP server : Python equivalent of the builtin Rust prompt plugin.

Tools:
  - prompt_generate: Build the complete LLM prompt (system prompt + thread
              context + summaries + skills + subtasks + planning). Same
              output as the builtin Rust mcp-server-prompt.
  - prompt_compact-messages: Compact a conversation to stay within token
              budgets (frozen '=== Compaction Summary ===' block, progressive
              passes, durable context dumps + auto-notes). Same output as the
              builtin Rust mcp-server-prompt.

Exact-parity port of /opt/workspace/omniagent/plugins/tools/prompt/
(plugin.json + src/main.rs + src/compact.rs + src/prompt_builder.rs +
src/memory_store.rs + src/chat_message.rs + src/notes.rs + src/dump.rs).
The ONLY intended difference is the implementation language (Python vs Rust).

MCP JSON-RPC over stdio. Requires DATABASE_URL and OMNI_DIR env vars
(plugin config may inject them; $env: placeholders resolved by the
framework).
"""

import json
import os
import re
import sys
import logging
import hashlib
import psycopg2
import psycopg2.extras
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency bootstrap for the REMOTE layout. The builtin Rust prompt plugin
# links tiktoken_rs at BUILD time, so real BPE counts are always available to
# it; the Python port needs the equivalent package importable. The installer
# (omniagent src/server/plugins_install.rs install_python_deps) installs
# requirements.txt into a hermetic venv {plugin_dir}/.venv, or into
# {plugin_dir}/pylib when venv creation is unavailable, but the runtime spawns
# the plugin with the system python3 (mcp-config.json "python3 server.py") and
# puts neither on sys.path - so do it here. Adding site-packages (not the venv
# interpreter) keeps the image-provided psycopg2 importable.
# No effect on output when tiktoken is absent: the documented chars/4 fallback
# then applies, exactly like the Rust fallback on load failure.
# ---------------------------------------------------------------------------
_PLUGIN_DIR = Path(__file__).resolve().parent
_pylib_dir = _PLUGIN_DIR / "pylib"
if _pylib_dir.is_dir():
    sys.path.insert(0, str(_pylib_dir))
_venv_lib = _PLUGIN_DIR / ".venv" / "lib"
if _venv_lib.is_dir():
    for _site_packages in sorted(_venv_lib.glob("python*/site-packages")):
        if _site_packages.is_dir():
            sys.path.insert(0, str(_site_packages))

# Optional real-token measurement (parity with tiktoken_rs).
try:
    import tiktoken  # type: ignore
except ImportError:
    tiktoken = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [prompt-python] %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("mcp")

MCP_PROTOCOL_VERSION = "2025-03-26"
initialized = False
conn = None

# ---------------------------------------------------------------------------
# Plugin config - mirrors the builtin Rust PluginConfig (plugin.json
# config_schema, 15 keys). Values arrive as env vars injected by the
# framework (key name or UPPER_SNAKE variant); defaults match the Rust
# PluginConfig::default() exactly.
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "planning_complexity_max_chars": 60,
    "planning_complexity_keywords": (
        "implement,refactor,redesign,architecture,create,build,design,develop,"
        "migrate,restructure,overhaul,rewrite,configure,set up,deploy,integrate,"
        "add feature,fix bug,resolve issue,multi-step,complex"
    ),
    "prompt_plan_max_tokens": 2048,
    "memory_max_chars": 5000,
    "tokenizer_encoding": "",
    "condense_keep_turns": 4,
    "database_url": "",
    "omni_dir": "",
    "tool_excerpt_chars": 800,
    "total_excerpt_cap": 4000,
    "read_excerpt_chars": 2000,
    "compact_keep_recent": 3,
    "compact_max_passes": 3,
    "compact_keep_step": 1,
    "max_summary_chars": 50000,
    "compact_headroom_tokens": 2000,
}

# Values sent via an MCP "configure" message (the builtin receives config
# through run_server_with_config's configure callback; the framework may also
# do this for python servers). Takes precedence over env vars.
CONFIG_OVERRIDES = {}

INT_KEYS = {
    "planning_complexity_max_chars",
    "prompt_plan_max_tokens",
    "memory_max_chars",
    "condense_keep_turns",
    "tool_excerpt_chars",
    "total_excerpt_cap",
    "read_excerpt_chars",
    "compact_keep_recent",
    "compact_max_passes",
    "compact_keep_step",
    "max_summary_chars",
    "compact_headroom_tokens",
}
STR_KEYS = {
    "planning_complexity_keywords",
    "tokenizer_encoding",
    "database_url",
    "omni_dir",
}

# Clamps mirroring PluginConfig::from_json (max(1) guards).
CLAMP_MIN_ONE = {"compact_max_passes", "compact_keep_step", "max_summary_chars", "condense_keep_turns"}


def _env_names(key):
    """Candidate env var names for a config key (framework injects config as
    env; try the key as-is and the UPPER_SNAKE convention)."""
    upper = key.upper()
    names = [key, upper]
    if "-" in key:
        names.append(key.replace("-", "_").upper())
    return names


def get_config():
    """Resolve the effective plugin config: configure-message overrides, then
    env vars (empty string = unset), then builtin defaults."""
    cfg = dict(DEFAULT_CONFIG)
    for key, default in DEFAULT_CONFIG.items():
        value = None
        for name in _env_names(key):
            if name in CONFIG_OVERRIDES:
                value = CONFIG_OVERRIDES[name]
                break
        if value is None:
            for name in _env_names(key):
                v = os.environ.get(name)
                if v is not None and v.strip() != "":
                    value = v.strip()
                    break
        if value is None:
            continue
        if key in INT_KEYS:
            try:
                cfg[key] = int(value)
            except (TypeError, ValueError):
                continue
            if key in CLAMP_MIN_ONE:
                cfg[key] = max(1, cfg[key])
        else:
            cfg[key] = str(value)
    return cfg


def cfg_env(key):
    """Read a single resolved config value (string form)."""
    cfg = get_config()
    v = cfg.get(key, "")
    return "" if v is None else str(v)


def _fail_omni_dir():
    raise RuntimeError(
        "OMNI_DIR is not set: set the OMNI_DIR environment variable or configure the "
        "'omni_dir' plugin config field (default '$env:OMNI_DIR')"
    )


# ---------------------------------------------------------------------------
# MCP protocol helpers
# ---------------------------------------------------------------------------

def send_json(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def make_success(req_id, result):
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def make_error(req_id, code, message):
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def make_tool_result(text, is_error=False):
    return {
        "content": [{"type": "text", "text": text}],
        "isError": is_error,
    }


# ---------------------------------------------------------------------------
# File / memory helpers (mirror Rust memory_store.rs + helpers)
# ---------------------------------------------------------------------------

def read_file(path):
    """Read a file, return content or empty string."""
    try:
        with open(path, "r") as f:
            return f.read()
    except (FileNotFoundError, IsADirectoryError, PermissionError, OSError):
        return ""


def truncate_str(s, max_chars):
    """Rust truncate_str: chars-count based, appends '...' (no note)."""
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "..."


def load_memory_raw(data_dir, profile_name):
    """Memory = {data_dir}/profiles/{profile}/MEMORY.md (profile ROOT).
    USER.md is never read (SOUL.md/USER.md support removed upstream)."""
    base = Path(data_dir) / "profiles" / profile_name
    return read_file(base / "MEMORY.md")


def build_memory_section(memory_raw, memory_max_chars):
    """Rust read_memory_section + truncate_content, exact header/format."""
    if not memory_raw:
        return ""
    raw_len = len(memory_raw)
    if raw_len > memory_max_chars:
        header = (
            "## MEMORY (your personal notes) "
            f"[TRUNCATED: showing first {memory_max_chars} of {raw_len} chars]"
        )
        body = memory_raw[:memory_max_chars] + "..." + \
            f"\n\n[... truncated from {raw_len} to ~{memory_max_chars} chars]"
    else:
        header = f"## MEMORY (your personal notes) [{raw_len} chars]"
        body = memory_raw
    return f"{header}\n{body}"


# ---------------------------------------------------------------------------
# Stable identity / guidance texts (exact copies of prompt_builder.rs)
# ---------------------------------------------------------------------------

_IDENTITY_TEMPLATE = (
    "You are OmniAgent: precise, efficient, autonomous. Your tools: {tool_list}. Use minimum roundtrips. If a tool fails, move on: don't retry more than twice. HONESTY RULE: never claim a success you did not verify. If you cannot complete the task, do NOT end with a normal final summary - call the builtin `core__fail_thread` tool and pass the COMPLETE final summary (what was done, what remains undone, why it is blocked) as its `reason` argument: the reason becomes the thread's last Error-type message, so fail = last message = summary. A plain final summary must never wrap an incomplete task - it looks like success and leaves the thread `completed`, while fail-thread ends the thread FAILED and lets the kanban workflow route the failure. Never write a summary message AFTER the fail call (the thread is already terminated and it is lost). CLEAR/DELETE DIRECTIVES: for an explicit clear/delete/set request, never report done or 'no change applied' until you have EXECUTED the change and VERIFIED the observable end state on the target environment the request names (the item is gone there, via its own API/DB/UI); 'no change applied' is valid only when you can prove the requested end state already holds. NEVER end a turn with only thinking and no action: a response with no tool call is treated as the end of the task, so every turn MUST end with either tool calls or a final answer. If you have finished thinking, immediately emit your next tool call or your final answer - never stop after reasoning alone."
)

def build_dynamic_identity(tool_names):
    """Rust prompt_builder.rs build_dynamic_identity (HEAD).

    Generic tool listing: the names come from the plugin registry ALREADY
    fully qualified; this function must NOT know any specific tool or
    plugin name. An empty list renders "no tools available".
    """
    tool_list = "no tools available" if not tool_names else ", ".join(tool_names)
    return _IDENTITY_TEMPLATE.format(tool_list=tool_list)

TOOL_GUIDANCE = (
    "TOOL USE RULES (fail the task if you violate these):\n"
    "1. CALL TOOLS DIRECTLY: Do NOT search the filesystem, read plugin config files, "
    "or inspect server configuration to discover what tools exist or how to call them. "
    "Available tools are listed with their name, description, and parameters in the "
    "function-calling API. Reading config files to find tools is always wrong and wastes turns.\n"
    "2. SEARCH FIRST: before exploring a repo or asking the operator, run "
    "`search_wiki` + `search_messages` (at most 2 retrieval calls, then act). Use "
    "search tools before querying databases for text or vector searches; use direct "
    "data queries only for structured aggregations (counts, sums, averages, groupings).\n"
    "3. WRITE COMPLETE FILES: When writing a file, write the entire content in a single "
    "operation. Do NOT write placeholder content expecting to fill in values afterward. "
    "EXCEPTION - LARGE OUTPUTS: if the file content is too large to fit in a single "
    "response (approaching your output token limit), split it across multiple "
    "filesystem__write calls: first with append=false, then append=true for each "
    "subsequent chunk. Never abandon a large write - chunk it. Never let an output "
    "length limit cause task failure.\n"
    "4. RENAME INSTEAD OF RECREATE: When a file or directory already exists and you "
    "need to change its name, use the rename tool. Do NOT delete and recreate.\n"
    "5. NO POLLING: Do NOT repeatedly check the same condition. If you're waiting "
    "for something, make a single request and wait for the result.\n"
    "6. SET DIRECTLY: For configuration values, set the new value directly. Do NOT "
    "read the current value, flip it, and write it back.\n"
    "7. COMPLETE WORK: Before presenting results, finish ALL steps. Do not interrupt "
    "your work to show intermediate progress unless asked.\n"
    "8. CONFIRM DESTRUCTIVE ACTIONS: Before delete, overwrite, or stop operations, "
    "present what you will do and wait for confirmation.\n"
    "9. SKIP ON FAILURE: If an operation fails (network error, not found, bad request), "
    "try once more with a different approach, then move on. Do NOT retry the same "
    "failing call more than once. There is no hidden state that changes between retries.\n"
    "10. TAKE NOTES: maintain a durable working memory with the note_* tools "
    "(notes__note_write/notes__note_append/notes__note_read/notes__note_list/notes__note_rm) after every non-trivial "
    "discovery (paths, line numbers, commands, root causes, decisions). Notes "
    "survive compaction and thread death - the retry thread starts with them.\n"
    "11. VERIFY-ONCE: read a file ONCE with `filesystem__read` (offset/limit paging - ONE\n"
    "call per page) and write the facts you need into your working notes; never re-read the\n"
    "same file or line range. NEVER use `docker__compose exec ... sed -n` / `grep -n` to read\n"
    "file contents: docker__compose is for RUNNING commands/builds, not reading files.\n"
    "Re-reading overlapping line ranges of the same file is the #1 budget killer (threads have\n"
    "died at 120/120 after 100+ sed windows with zero commits). Consult your notes, not the\n"
    "disk, when you need content again.\n"
    "12. NEVER RE-READ CONTEXT DUMPS: a context-*.json dump is read ONCE per "
    "thread - a second read returns a '[duplicate read ...]' marker, not content. "
    "Trust the injected '=== Context Compacted ===' summary and your notes instead; "
    "re-reading dumps is a forbidden anti-loop that wastes iterations.\n"
    "13. SUBTASKS: the thread's subtasks are listed in the '## Subtasks' block of "
    "your context WITH their ids ([#<id>]). In plan mode the engine has ALREADY "
    "created one subtask per parsed plan step: REUSE those rows - mark the one you "
    "are CURRENTLY working on as processing (action=\"update\", subtask_id=<id>, "
    "status=\"processing\"), mark each finished step completed (action=\"update\", "
    "subtask_id=<id>, status=\"completed\") and cancel rows that are no longer "
    "needed (status=\"cancelled\"). Do NOT create a second set for the plan; call "
    "action=\"add\" only for steps discovered MID-RUN, in the order you will execute "
    "them. At least one subtask processing at a time, more allowed for "
    "interdependent subtasks. Before your final answer, complete or cancel ALL "
    "subtasks so none remain pending.\n"
    "14. NO-REPETITION + VERIFY-ONCE + NO-PROGRESS STOP: never re-issue a tool call "
    "(same tool + same effective arguments/scope) whose result is already in your "
    "context or notes when nothing relevant changed in between - including read-only "
    "verification commands (git log/status/rev-parse, search, list, info, status). "
    "After a state-changing operation (commit+push, file write), verify ONCE (e.g. one "
    "git rev-parse showing local == origin/main) and move on; never re-verify an "
    "unchanged state. If you catch yourself repeating the same checks with no state "
    "change and no progress, STOP exploring and produce your final report of what is "
    "done and what remains. Repeated no-progress read-only calls are blocked by the "
    "engine and will not re-execute.\n"
    "15. GIVE UP LOUDLY: if you cannot complete the task, do NOT finish with a normal "
    "final summary - call the builtin `core__fail_thread` tool with the COMPLETE final "
    "summary as its `reason` argument (what was done, what remains undone, why it is "
    "blocked); the reason becomes the thread's last Error-type message, so fail = last "
    "message = summary. A plain final summary must never wrap an incomplete task: it "
    "looks like success and leaves the thread `completed`, while fail-thread ends the "
    "thread FAILED and lets the kanban workflow route the failure."
)


# Audit V-5: the formatting hint is OWNED by the platform plugin, which
# advertises it as `capabilities.prompt_hint` in its `initialize` result; the
# core forwards it to this tool as the `platform_hint` argument. This module
# keeps NO platform-name to hint table (parity with the Rust copy).
GENERIC_PLATFORM_HINT = (
    "You are on a messaging platform. Use standard markdown formatting: "
    "**bold**, *italic*, `code`, ```code blocks```, [links](url), headings, "
    "lists, blockquotes. Avoid platform-specific syntax that may not be "
    "supported."
)


def build_platform_hint(platform, declared):
    """Return the `platform` section text, or None when there is no section.

    1. a declared (non-empty) hint wins (the plugin owns its platform rules),
    2. otherwise a named platform gets GENERIC_PLATFORM_HINT so an unknown
       platform degrades gracefully instead of silently losing guidance,
    3. an unnamed platform gets no section at all (historical behaviour).
    """
    if declared is not None:
        hint = str(declared).strip()
        if hint:
            return hint
    if not str(platform or "").strip():
        return None
    return GENERIC_PLATFORM_HINT


# ---------------------------------------------------------------------------
# Skills (Rust get_skills: frontmatter-aware, flat + Hermes layouts)
# ---------------------------------------------------------------------------

def extract_frontmatter_field(content, field):
    """Rust extract_frontmatter_field: YAML frontmatter between --- markers."""
    content = content.strip()
    if not content.startswith("---"):
        return None
    after_first = content[len("---"):].lstrip()
    end = after_first.find("\n---")
    if end == -1:
        return None
    frontmatter = after_first[:end]
    prefix = field + ":"
    for line in frontmatter.splitlines():
        line = line.strip()
        if line.startswith(prefix):
            value = line[len(prefix):].strip()
            if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
                return value[1:-1]
            if len(value) >= 2 and value[0] == "'" and value[-1] == "'":
                return value[1:-1]
            return value
    return None


def extract_skill_description(content):
    """Rust extract_skill_description: frontmatter description:, else first
    meaningful line with leading '#' stripped."""
    desc = extract_frontmatter_field(content, "description")
    if desc is not None and desc.strip():
        return desc
    for line in content.splitlines():
        line = line.strip()
        if line == "" or line == "---":
            continue
        return line.lstrip("#").strip()
    return "No description"


def skill_display_name(content, fallback):
    """Rust skill_display_name: frontmatter name:, else fallback."""
    name = extract_frontmatter_field(content, "name")
    if name is not None and name.strip():
        return name
    return fallback


def get_skills(data_dir, profile_name):
    """Rust get_skills: profile-scoped skills root; flat <name>.md and
    Hermes-style <category>/<name>/SKILL.md; sorted."""
    skills_dir = Path(data_dir) / "profiles" / profile_name / "skills"
    skills = []
    if skills_dir.exists():
        for entry in sorted(skills_dir.iterdir()):
            path = entry
            if path.is_file() and path.suffix == ".md":
                content = read_file(path)
                if content:
                    stem = path.stem
                    name = skill_display_name(content, stem)
                    desc = extract_skill_description(content)
                    skills.append(f"- {name}: {desc}")
            if path.is_dir():
                try:
                    cat_entries = sorted(path.iterdir())
                except OSError:
                    cat_entries = []
                for cat_entry in cat_entries:
                    skill_file = cat_entry / "SKILL.md"
                    if not skill_file.is_file():
                        continue
                    content = read_file(skill_file)
                    if content:
                        dir_name = cat_entry.name
                        name = skill_display_name(content, dir_name)
                        desc = extract_skill_description(content)
                        skills.append(f"- {name}: {desc}")
    skills.sort()
    return skills

# ---------------------------------------------------------------------------
# DB helpers (mirror main.rs queries)
# ---------------------------------------------------------------------------

def get_db():
    """Get or create a database connection (configure-message overrides first)."""
    global conn
    if conn is None or conn.closed:
        cfg = get_config()
        database_url = cfg.get("database_url") or os.environ.get("DATABASE_URL", "")
        if not database_url:
            raise RuntimeError("DATABASE_URL must be set")
        conn = psycopg2.connect(database_url)
        conn.autocommit = True
    return conn


# ---------------------------------------------------------------------------
# Conversation history selection (2026-10-01, task: omni profile "main" loses
# operator statements 2-3 turns later).
#
# The OPERATOR-VISIBLE turns of a thread are:
#   role 'cause'      = the operator prompt that opened the thread
#   role 'sub_cause'  = an operator prompt merged into a RUNNING thread
#   msg_type 'summary'= the agent's final answer (what the operator reads)
#   msg_type 'message'/'reasoning' = intermediate agent text (legacy + live)
# Everything else is TOOL TRAFFIC ('tool', 'tool-result', 'multi-tool'), the
# prompt dump ('prompt') or the internal plan ('plan') and must NOT be
# selected: it would swamp the window with machine output.
#
# The previous filter `role = 'cause' OR msg_type IN ('message','reasoning')`
# matched the cause plus ONE reasoning blob in a 73-message thread (thread
# 3745), so the agent could not see ANY of its own previous answers or the
# operator prompts merged into the thread: the rendered prompt carried a
# 2-line "conversation" for a 73-message conversation.
CONVERSATION_MSG_TYPES = ("message", "reasoning", "summary")
CONVERSATION_ROLES = ("cause", "sub_cause")
# Per-message cap and total cap for the rendered history block. The newest
# turns are kept first; older ones are dropped when the total cap is reached.
CONVERSATION_MESSAGE_MAX_CHARS = 1200
CONVERSATION_HISTORY_MAX_CHARS = 8000
# Verbatim operator messages of the most recent OTHER threads of the channel.
# A new operator message normally starts a NEW thread (the previous one is
# terminal), so the current thread's own history holds only the new message;
# without this block the operator's earlier words reach the agent only through
# the lossy, agent-generated channel summary.
CHANNEL_OPERATOR_MESSAGES_LIMIT = 5
CHANNEL_OPERATOR_MESSAGE_MAX_CHARS = 1200


def get_thread_messages(cursor, thread_id, limit=10):
    cursor.execute(
        """SELECT id, thread_id, role, content, msg_type, msg_subtype,
                  COALESCE(TO_CHAR(created_at, 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'), '') AS created_at
           FROM messages
           WHERE thread_id = %s
             AND (role IN ('cause', 'sub_cause')
                  OR msg_type IN ('message', 'reasoning', 'summary'))
           ORDER BY created_at DESC
           LIMIT %s""",
        (thread_id, limit),
    )
    rows = cursor.fetchall()
    rows.reverse()  # oldest first
    return rows


def get_recent_channel_operator_messages(cursor, channel_id, before_thread_id, limit=5):
    """Verbatim operator prompts from the most recent OTHER threads of the
    same channel (newest last). Purely additive context: it lets a follow-up
    thread answer from the operator's ACTUAL words instead of the paraphrased
    channel summary."""
    cursor.execute(
        """SELECT t.id, m.content
           FROM messages m
           JOIN threads t ON t.id = m.thread_id
           WHERE t.channel_id = %s
             AND m.role = 'cause'
             AND t.id < %s
           ORDER BY m.id DESC
           LIMIT %s""",
        (str(channel_id), int(before_thread_id), limit),
    )
    rows = cursor.fetchall()
    rows.reverse()  # oldest first
    return rows


def get_latest_summary(cursor, channel_id):
    cursor.execute(
        """SELECT id, channel_id, next_thread_id, content
           FROM summaries
           WHERE channel_id = %s
           ORDER BY id DESC
           LIMIT 1""",
        (channel_id,),
    )
    return cursor.fetchone()


def get_threads_since(cursor, channel_id, since_id, limit=5):
    cursor.execute(
        """SELECT id, status, cause
           FROM threads
           WHERE channel_id = %s
             AND status = 'completed'
             AND id > %s
           ORDER BY id ASC
           LIMIT %s""",
        (channel_id, since_id, limit),
    )
    return cursor.fetchall()


def get_subtasks(cursor, thread_id):
    cursor.execute(
        """SELECT id, description, status, thread_id
           FROM thread_subtasks
           WHERE thread_id = %s
           ORDER BY id ASC""",
        (thread_id,),
    )
    return cursor.fetchall()


def get_thread_task_ref(cursor, thread_id):
    cursor.execute(
        "SELECT task_id, schedule_task_id FROM threads WHERE id = %s",
        (thread_id,),
    )
    return cursor.fetchone()


# ---------------------------------------------------------------------------
# R8-J: prior attempts of the SAME task
# ---------------------------------------------------------------------------

PRIOR_ATTEMPTS_MAX_ENTRIES = 5
PRIOR_ATTEMPTS_MAX_SUMMARY_CHARS = 800


def get_prior_threads_by_task(cursor, task_id, current_thread_id, limit):
    cursor.execute(
        """SELECT id, status, iterations,
                  TO_CHAR(ended_at, 'YYYY-MM-DD HH24:MI') AS ended_at
           FROM threads
           WHERE task_id = %s
             AND (task_type = 'kanban' OR task_type IS NULL)
             AND id < %s
           ORDER BY id DESC
           LIMIT %s""",
        (task_id, current_thread_id, limit),
    )
    return cursor.fetchall()


def get_thread_summary(cursor, thread_id):
    cursor.execute(
        "SELECT content FROM messages WHERE thread_id = %s AND msg_type = 'summary' ORDER BY id DESC LIMIT 1",
        (thread_id,),
    )
    row = cursor.fetchone()
    return row[0] if row else None


def render_prior_attempts_block(rows, summaries):
    """Rust render_prior_attempts_block: newest first, capped at 5."""
    rows = sorted(rows, key=lambda r: r[0], reverse=True)[:PRIOR_ATTEMPTS_MAX_ENTRIES]
    if not rows:
        return None
    lines = ["=== Previous attempts of this task (READ - do NOT repeat what they did) ==="]
    for r in rows:
        t_id, status, iterations, ended_at = r
        iter_s = str(iterations) if iterations is not None else "?"
        ended_s = ended_at or "?"
        summary = summaries.get(t_id)
        if summary is not None:
            lines.append(
                f"- thread {t_id} | status {status} | iterations {iter_s} | "
                f"ended_at {ended_s} | summary: {truncate_str(summary, PRIOR_ATTEMPTS_MAX_SUMMARY_CHARS)}"
            )
        else:
            lines.append(
                f"- thread {t_id} | status {status} | iterations {iter_s} | "
                f"ended_at {ended_s} | (no summary message)"
            )
    return "\n".join(lines)


def build_prior_attempts_block(cursor, thread_id):
    """Rust build_prior_attempts_block: earlier threads of the SAME kanban
    task (all statuses) with their final summaries. Plain threads get none."""
    row = get_thread_task_ref(cursor, thread_id)
    if not row or not row[0]:
        return None
    task_id = row[0]
    rows = get_prior_threads_by_task(cursor, task_id, thread_id, PRIOR_ATTEMPTS_MAX_ENTRIES)
    if not rows:
        return None
    summaries = {}
    for r in rows:
        try:
            summaries[r[0]] = get_thread_summary(cursor, r[0])
        except Exception:
            summaries[r[0]] = None
    return render_prior_attempts_block(rows, summaries)


# ---------------------------------------------------------------------------
# R8-K: learned knowledge (promoted memories read-back)
# ---------------------------------------------------------------------------

LEARNED_KNOWLEDGE_MAX_ENTRY_CHARS = 600
LEARNED_KNOWLEDGE_MAX_TOTAL_CHARS = 2000


def strip_frontmatter(content):
    """Rust strip_frontmatter: drop leading ---...--- YAML block."""
    trimmed = content.lstrip()
    if trimmed.startswith("---"):
        after_open = trimmed[len("---"):]
        idx = after_open.find("\n---")
        if idx != -1:
            return after_open[idx + 4:].lstrip()
        return trimmed  # unterminated fence - treat whole doc as body
    return trimmed


def load_promoted_memories(data_dir, profile_name):
    """Read promoted memories from the SHARED wiki (<OMNI_DIR>/wiki).

    The wiki is instance-level at the omni-dir root, not profile-scoped;
    profile_name is kept for API compatibility only.
    """
    base = Path(data_dir) / "wiki" / "Memory" / "Promoted"
    memories = []
    if base.is_dir():
        for path in base.glob("*.md"):
            content = read_file(path)
            if not content:
                continue
            title = path.stem
            body = truncate_str(strip_frontmatter(content).strip(), LEARNED_KNOWLEDGE_MAX_ENTRY_CHARS)
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0.0
            memories.append({"title": title, "body": body, "mtime": mtime})
    memories.sort(key=lambda m: m["mtime"], reverse=True)
    return memories


def render_learned_knowledge_block(memories):
    # Bodies live in wiki Memory/Promoted pages and already start with their
    # own "# Memory: <title>" heading, so the "- **<title>**:" bullet would
    # only duplicate the title.
    header = "=== Learned Knowledge (validated memories promoted from prior threads) ==="
    parts = [header]
    total = 0
    for m in memories:
        if m["body"].startswith("# Memory: " + m["title"]):
            entry = m["body"]
        else:
            entry = f"- **{m['title']}**: {m['body']}"
        if total + len(entry) > LEARNED_KNOWLEDGE_MAX_TOTAL_CHARS and total > 0:
            break
        total += len(entry)
        parts.append(entry)
    return "\n\n".join(parts)


def build_learned_knowledge_block(data_dir, profile_name):
    """Rust build_learned_knowledge_block: never fails the prompt."""
    memories = load_promoted_memories(data_dir, profile_name)
    if not memories:
        return ("=== Learned Knowledge === (none yet - after completing this task, promote what "
                "you learned via memory_promote-to-memory so future threads benefit)")
    return render_learned_knowledge_block(memories)


# ---------------------------------------------------------------------------
# R8-K: interrupted-attempt warning
# ---------------------------------------------------------------------------

def count_interrupted_attempts(cursor, thread_id):
    row = get_thread_task_ref(cursor, thread_id)
    if not row or not row[0]:
        return None
    cursor.execute(
        "SELECT COUNT(*) FROM threads WHERE task_id = %s AND status = 'interrupted'",
        (row[0],),
    )
    return cursor.fetchone()[0]


# ---------------------------------------------------------------------------
# Continuation self-orientation (Phase 3b): role block, prior step-threads,
# kanban history, resume ledger
# ---------------------------------------------------------------------------

def step_to_role(workflow_step):
    if workflow_step == "running":
        return "executor"
    if workflow_step == "testing":
        return "tester"
    if workflow_step == "review":
        return "reviewer"
    return None


def build_role_block(workflow_step):
    """Rust build_role_block: exact role instructions."""
    if workflow_step == "running":
        return ("You are the EXECUTOR of this workflow step: implement/execute the task described "
                "in the task description. Before acting, read the prior step-threads of this task "
                "listed below - resume from where the previous attempt ended, avoid repeating work "
                "that already succeeded, and fix the work if the last testing/review step failed or "
                "requested changes.")
    if workflow_step == "testing":
        return ("You are the TESTER of this workflow step: run the tests for the executed task (you "
                "may create automated tests), but you must NOT implement the task itself. Read the "
                "executor's thread and all recent threads of this task before testing.")
    if workflow_step == "review":
        return ("You are the REVIEWER of this workflow step: perform a comprehensive review of the "
                "execution AND the tests. You must NOT implement the task. Read the executor and "
                "tester threads plus all recent threads of this task. If everything passes: report "
                "a successful status with a normal summary. If you find issues: call the fail tool "
                "with workflow_step 'running', 'testing', or 'blocked' (never 'review') so the "
                "right role re-runs.")
    return None


def _mini_yaml_parse(text):
    """Minimal indentation-based YAML-subset parser for workflows.yml
    (nested maps of scalar values). Falls back to this when pyyaml is
    unavailable; the expected shape is
    workflows: {id}: roles: {role}: template: <name>."""
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text)
    except ImportError:
        pass
    root = {}
    stack = [(-1, root)]
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        content = line.strip()
        key = None
        value = None
        if ":" in content:
            k, _, v = content.partition(":")
            key = k.strip().strip("\"'")
            value = v.strip().strip("\"'") if v.strip() else None
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value is None:
            nxt = {}
            parent[key] = nxt
            stack.append((indent, nxt))
        else:
            parent[key] = value
    return root


class TemplateError(Exception):
    """A template reference is not a valid PROFILE template (operator
    directive 2026-09-24: templates must be profile templates - resolving
    anything else is a bug). Raised instead of silently degrading."""


def validate_template_name(profile_name, template_name):
    """Reject template names that escape the profile templates dir: path
    separators, '..', absolute paths, empty names."""
    name = str(template_name).strip()
    if not name:
        raise TemplateError(
            f"template name is empty for profile '{profile_name}': expected a file name in profiles/{profile_name}/templates/"
        )
    p = Path(name)
    if (
        p.is_absolute()
        or "/" in name
        or "\\" in name
        or ".." in name
    ):
        raise TemplateError(
            f"template name '{template_name}' must be a plain file name inside profiles/{profile_name}/templates/ (no path separators, no '..', no absolute paths)"
        )
    return name


def load_template(data_dir, profile_name, template_name):
    """Rust memory_store::load_template: profiles/<p>/templates/<name>(.md)."""
    if not template_name:
        return None
    name = validate_template_name(profile_name, template_name)
    path = Path(data_dir) / "profiles" / profile_name / "templates" / name
    if path.suffix == "":
        path = path.with_suffix(".md")
    if not path.exists():
        return None
    content = read_file(path).strip()
    if not content:
        return None
    return content


def load_role_template(data_dir, profile_name, workflow_id, role):
    """Rust load_role_template: workflows.yml role template FILE NAME ->
    content from profiles/<p>/templates/<name>.md.

    Returns None when the workflow/role is absent or the template field is
    empty (no template configured). Raises TemplateError when the template
    field IS set but the name is invalid or the file is missing - templates
    must be PROFILE templates, never a silent fallback (operator directive
    2026-09-24)."""
    try:
        text = read_file(Path(data_dir) / "workflows.yml")
        if not text:
            return None
        parsed = _mini_yaml_parse(text)
        workflows = parsed.get("workflows", {}) if isinstance(parsed, dict) else {}
        wf = workflows.get(workflow_id, {}) if isinstance(workflows, dict) else {}
        roles = wf.get("roles", {}) if isinstance(wf, dict) else {}
        role_entry = roles.get(role, {}) if isinstance(roles, dict) else {}
        template_name = role_entry.get("template") if isinstance(role_entry, dict) else None
        if not template_name or not str(template_name).strip():
            return None
        name = validate_template_name(profile_name, str(template_name).strip())
        content = load_template(data_dir, profile_name, name)
        if content is None:
            raise TemplateError(
                f"workflow template '{name}' not found for profile '{profile_name}': expected profiles/{profile_name}/templates/{name}.md - templates must be profile templates"
            )
        return content
    except TemplateError:
        raise
    except Exception as e:
        log.warning("load_role_template failed: %s", e)
        return None


def apply_workflow_mapping(system, user, user_message, workflow_step, template):
    """Rust apply_workflow_mapping: executor keeps task as USER + template as
    SYSTEM; tester/reviewer are INVERSE."""
    role = step_to_role(workflow_step)
    if role is None:
        return system, user
    if template is None:
        if workflow_step in ("testing", "review"):
            log.warning("workflow template required for %s step but missing in workflows.yml", workflow_step)
        return system, user
    if workflow_step == "running":
        system = system + f"\n\n## Workflow instructions ({role})\n{template}"
    elif workflow_step in ("testing", "review"):
        user = template
        description = user_message.strip()
        if description:
            system = system + f"\n\n## Task under {role} - context only, do not implement\n{description}"
    return system, user


def extract_tracking_path(body):
    """Rust extract_tracking_path: pull a data/tasks/<name> path out of a task body."""
    pos = body.find("data/tasks/")
    if pos == -1:
        return None
    start = pos
    head = body[:pos]
    for i in range(len(head) - 1, -1, -1):
        c = head[i]
        if c.isspace() or c in '`"\'([' ',':
            start = i + 1
            break
        if i == 0:
            start = 0
    after = body[pos + len("data/tasks/"):]
    end = len(after)
    for i, c in enumerate(after):
        if c.isspace() or c in '`"\'):],':
            end = i
            break
    if end == 0:
        return None
    return body[start:pos + len("data/tasks/")] + after[:end]


def last_message_info(cursor, thread_id):
    cursor.execute(
        "SELECT content, msg_type FROM messages WHERE thread_id = %s ORDER BY id DESC LIMIT 1",
        (thread_id,),
    )
    row = cursor.fetchone()
    if row is None:
        return "<no messages>", None
    return row[0], row[1]


def build_continuation_block(cursor, thread_id):
    """Rust build_continuation_block: role block + prior step-threads + kanban
    history + resume ledger. None for plain threads; never fails the prompt."""
    row = get_thread_task_ref(cursor, thread_id)
    if not row:
        return None
    task_id, schedule_task_id = row
    blocks = []

    # Role instructions for the current workflow step.
    cursor.execute(
        "SELECT workflow_id, workflow_step FROM threads WHERE id = %s",
        (thread_id,),
    )
    step_row = cursor.fetchone()
    if step_row:
        wf_id, step = step_row
        if step:
            role_block = build_role_block(step)
            if role_block:
                blocks.append(role_block)

    # Prior step-threads of this task (chronological).
    prior = []
    if task_id:
        cursor.execute(
            "SELECT id, status, workflow_step FROM threads "
            "WHERE task_id = %s AND (task_type = 'kanban' OR task_type IS NULL) "
            "AND id != %s ORDER BY id DESC LIMIT 8",
            (task_id, thread_id),
        )
        prior.extend(cursor.fetchall())
    if schedule_task_id:
        cursor.execute(
            "SELECT id, status, workflow_step FROM threads "
            "WHERE schedule_task_id = %s AND id != %s ORDER BY id DESC LIMIT 8",
            (schedule_task_id, thread_id),
        )
        prior.extend(cursor.fetchall())
        cursor.execute(
            "SELECT id, status, workflow_step FROM threads "
            "WHERE task_id = %s AND task_type = 'cron' AND id != %s ORDER BY id DESC LIMIT 8",
            (schedule_task_id, thread_id),
        )
        prior.extend(cursor.fetchall())
    prior = sorted(prior, key=lambda t: t[0])
    if prior:
        parts = [
            "Prior step-threads of this task (thread, step, terminal status, last message) - "
            "resume from where the previous attempt ended; do not re-do completed work or "
            "repeat its mistakes:"
        ]
        for t in prior:
            t_id, status, step = t
            step_s = step or "-"
            content, msg_type = last_message_info(cursor, t_id)
            parts.append(
                f"thread {t_id} [step {step_s}] status {status} | "
                f"last message ({msg_type or 'text'}): {truncate_str(content, 180)}"
            )
        blocks.append("\n".join(parts))

    # Recent kanban history.
    if task_id:
        cursor.execute(
            """SELECT action, initial_board, final_board, comment,
                      TO_CHAR(created_at, 'YYYY-MM-DD HH24:MI') AS created_at
               FROM kanban_history
               WHERE kanban_task_id = %s
               ORDER BY id DESC
               LIMIT 5""",
            (task_id,),
        )
        history = cursor.fetchall()
        if history:
            parts = ["Recent kanban history (why this task is being run again):"]
            for h in history:
                action, initial_board, final_board, comment, created_at = h
                suffix = ""
                if comment:
                    suffix = f' - "{truncate_str(comment, 120)}"'
                parts.append(
                    f"{initial_board or '?'} -> {final_board or '?'}: {action} ({created_at or '?'}){suffix}"
                )
            blocks.append("\n".join(parts))

    # Resume ledger referenced by the task body.
    if task_id:
        cursor.execute("SELECT body FROM kanban_tasks WHERE id = %s", (task_id,))
        body_row = cursor.fetchone()
        if body_row and body_row[0]:
            ledger = extract_tracking_path(body_row[0])
            if ledger:
                blocks.append(
                    "Task tracking file (resume ledger): read " + ledger + " first - it records "
                    "what has been done, verified, or failed across attempts of this task."
                )

    if not blocks:
        return None
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Cross-task channel context (Phase 3b-ext): other tasks' terminal threads
# ---------------------------------------------------------------------------

CROSS_TASK_MAX_ENTRIES = 3
CROSS_TASK_MAX_MESSAGE_CHARS = 400
CROSS_TASK_MAX_TITLE_CHARS = 60


def render_cross_task_block(own_task_id, own_schedule_task_id, rows):
    """Rust render_cross_task_block: drop own-task threads, newest first,
    capped at 3."""
    def excludes(other_task_id):
        return other_task_id == own_task_id or other_task_id == own_schedule_task_id

    entries = [r for r in rows if r[1] is not None and not excludes(r[1])]
    entries.sort(key=lambda r: r[0], reverse=True)
    entries = entries[:CROSS_TASK_MAX_ENTRIES]
    if not entries:
        return None
    lines = [
        "Recent threads from other tasks on this channel (background context from sibling tasks, "
        "not your own history): a prior phase on this channel often already solved the exact "
        "problem you are investigating - trust its final result instead of re-deriving it."
    ]
    for r in entries:
        t_id, t_task_id, t_title, t_step, t_status, t_content, t_msg_type = r
        if t_task_id and t_title:
            task_label = f'task {t_task_id} "{truncate_str(t_title, CROSS_TASK_MAX_TITLE_CHARS)}"'
        elif t_task_id:
            task_label = f"task {t_task_id}"
        else:
            task_label = "plain thread"
        step = t_step or "-"
        msg_type = t_msg_type or "text"
        content = t_content or "<no messages>"
        lines.append(
            f"- thread {t_id} | {task_label} | step {step} | status {t_status} | "
            f"last message ({msg_type}): {truncate_str(content, CROSS_TASK_MAX_MESSAGE_CHARS)}"
        )
    return "\n".join(lines)


def build_cross_task_block(cursor, thread_id):
    """Rust build_cross_task_block: same channel, terminal threads of OTHER
    tasks; None for plain threads / no matches."""
    row = get_thread_task_ref(cursor, thread_id)
    if not row:
        return None
    task_id, schedule_task_id = row
    if not task_id and not schedule_task_id:
        return None
    cursor.execute(
        "SELECT channel_id FROM threads WHERE id = %s",
        (thread_id,),
    )
    cid_row = cursor.fetchone()
    if not cid_row or not cid_row[0]:
        return None
    channel_id = cid_row[0]

    cursor.execute(
        """
        SELECT t.id,
               t.task_id,
               k.title        AS task_title,
               t.workflow_step,
               t.status,
               m.content      AS last_content,
               m.msg_type     AS last_msg_type
        FROM threads t
        LEFT JOIN kanban_tasks k ON k.id = t.task_id
        LEFT JOIN LATERAL (
            SELECT content, msg_type
            FROM messages
            WHERE thread_id = t.id
            ORDER BY thread_sequence DESC NULLS LAST,
                     iteration_number DESC NULLS LAST,
                     id DESC
            LIMIT 1
        ) m ON true
        WHERE t.channel_id = %s
          AND t.id != %s
          AND t.task_id IS NOT NULL
          AND (NULLIF(%s::text, '') IS NULL OR t.task_id != NULLIF(%s::text, '')::text)
          AND (NULLIF(%s::text, '') IS NULL OR t.task_id != NULLIF(%s::text, '')::text)
          AND t.status IN ('completed', 'review', 'failed', 'interrupted', 'skipped')
        ORDER BY t.id DESC
        LIMIT %s
        """,
        (channel_id, thread_id,
         task_id or "", task_id or "",
         schedule_task_id or "", schedule_task_id or "",
         CROSS_TASK_MAX_ENTRIES),
    )
    rows = cursor.fetchall()
    return render_cross_task_block(task_id, schedule_task_id, rows)

# ---------------------------------------------------------------------------
# Tool: prompt_generate (mirror Rust handle_generate_full)
# ---------------------------------------------------------------------------

def handle_generate(req_id, arguments, meta):
    """Generate the full LLM prompt as JSON: system/memory/context/user/plan."""
    try:
        args = arguments or {}
        profile_name = args.get("profile_name") or (meta or {}).get("profile_name") or "omni"
        platform = args.get("platform") or (meta or {}).get("platform") or ""
        # V-5: hint declared by the platform plugin (forwarded by the core).
        platform_hint = args.get("platform_hint")
        if platform_hint is None and meta:
            platform_hint = meta.get("platform_hint")
        system_message = args.get("system_message")
        user_message = args.get("user_message") or ""
        tool_names = args.get("tool_names") or []
        if not isinstance(tool_names, list):
            tool_names = []
        tool_names = [str(t) for t in tool_names]
        thread_id = args.get("thread_id")
        if thread_id is None and meta:
            thread_id = meta.get("thread_id")
        channel_id = args.get("channel_id")
        if channel_id is None and meta:
            channel_id = meta.get("channel_name") or meta.get("channel_id")

        cfg = get_config()
        data_dir = cfg_env("omni_dir") or os.environ.get("OMNI_DIR") or _fail_omni_dir()

        # 1. System prompt parts (identity + guidance + profile + optional
        # system message + platform hint) and memory section (MEMORY.md).
        memory_raw = load_memory_raw(data_dir, profile_name)
        parts = []
        parts.append(build_dynamic_identity(tool_names))
        parts.append(TOOL_GUIDANCE)
        parts.append(f"Active profile: {profile_name}.")
        if system_message:
            parts.append(system_message)
        hint = build_platform_hint(platform, platform_hint)
        if hint:
            parts.append(hint)
        memory_section = build_memory_section(memory_raw, cfg.get("memory_max_chars", 5000))
        if memory_section:
            parts.append(memory_section)

        # Split into system (non-memory) vs memory (parts starting with
        # "## MEMORY"), exactly like the Rust split loop.
        system_parts = []
        memory_text = ""
        for part in parts:
            if part.startswith("## MEMORY"):
                memory_text += part + "\n"
            else:
                system_parts.append(part)
        system = "\n\n".join(system_parts)
        memory = memory_text.strip()

        # 2. Context blocks.
        context_blocks = []
        db = get_db()
        cursor = db.cursor()

        # 2a. Recent thread messages (the thread's own conversational turns)
        if thread_id is not None:
            try:
                tid = int(thread_id)
                msgs = get_thread_messages(cursor, tid, 10)
                if msgs:
                    # Newest turns first while the total budget lasts: the turns
                    # an operator follow-up refers to ("I said 2 messages ago")
                    # are the recent ones, never the beginning of the thread.
                    remaining = CONVERSATION_HISTORY_MAX_CHARS
                    formatted = []
                    for m in reversed(msgs):
                        text = truncate_str(m[3], CONVERSATION_MESSAGE_MAX_CHARS)
                        if formatted and len(text) > remaining:
                            break
                        formatted.append(f"[{m[2]}]: {text}")
                        remaining -= len(text)
                    formatted.reverse()
                    context_blocks.append(
                        "Recent conversation history (current thread, oldest first; "
                        "each entry is one operator-visible turn):\n"
                        + "\n".join(formatted)
                    )
            except Exception as e:
                log.warning("Failed to get thread messages: %s", e)

        # 2b. Latest summary and threads since
        if channel_id is not None:
            try:
                summary = get_latest_summary(cursor, str(channel_id))
                if summary:
                    context_blocks.append(
                        f"Previous channel summary (covers threads up to id={summary[2]}):\n"
                        f"{truncate_str(summary[3], 3000)}"
                    )
                    threads = get_threads_since(cursor, str(channel_id), summary[2], 5)
                    if threads:
                        thread_info = [f"[Thread #{t[0]} by {t[2]}]: completed" for t in threads]
                        context_blocks.append(
                            "Recent threads (after last summary):\n" + "\n---\n".join(thread_info)
                        )
            except Exception as e:
                log.warning("Failed to get summary: %s", e)

        # 2c. Skills
        skills = get_skills(data_dir, profile_name)
        if skills:
            context_blocks.append(
                "Available skills (read one with view_skill before acting when it matches the task):\n"
                + "\n".join(skills)
            )

        # 2c-ext2. Previous attempts of the SAME task (R8-J)
        if thread_id is not None:
            try:
                prior = build_prior_attempts_block(cursor, int(thread_id))
                if prior:
                    context_blocks.append(prior)
            except Exception as e:
                log.warning("Prior attempts context unavailable: %s", e)

        # 2c-ext3. Learned Knowledge (R8-K)
        try:
            learned = build_learned_knowledge_block(data_dir, profile_name)
            if learned:
                context_blocks.append(learned)
        except Exception as e:
            log.warning("Learned knowledge context unavailable: %s", e)

        # 2b-2. Verbatim operator prompts from the most recent other threads
        # of this channel. Operator statements made 2-3 messages ago normally
        # live in a PREVIOUS thread (each operator message opens its own
        # thread once the previous one is terminal); without this block they
        # reach the agent only as the lossy channel summary paraphrase.
        if channel_id is not None and thread_id is not None:
            try:
                ops = get_recent_channel_operator_messages(
                    cursor, channel_id, int(thread_id), CHANNEL_OPERATOR_MESSAGES_LIMIT
                )
                if ops:
                    lines = [
                        f"[Thread #{r[0]} operator]: "
                        f"{truncate_str(r[1], CHANNEL_OPERATOR_MESSAGE_MAX_CHARS)}"
                        for r in ops
                    ]
                    context_blocks.append(
                        "Recent operator messages in this channel (VERBATIM operator "
                        "words, threads just before this one, newest last; when they "
                        "disagree with the channel summary above, THESE win):\n"
                        + "\n".join(lines)
                    )
            except Exception as e:
                log.warning("Failed to get channel operator messages: %s", e)
        # 2d. Subtasks
        if thread_id is not None:
            try:
                subtask_rows = get_subtasks(cursor, int(thread_id))
                if subtask_rows:
                    lines = [f"## Subtasks (Thread #{thread_id})"]
                    lines.append(
                        "These rows are the thread's plan/execution subtasks, listed in "
                        "creation order. Each row is addressable by its id: update it with "
                        'subtasks__manage_subtasks(action="update", subtask_id=<id>, '
                        'status="processing"|"completed"|"cancelled"). Reuse these rows - '
                        "do NOT create a duplicate set; only add a row for a step discovered mid-run."
                    )
                    for i, s in enumerate(subtask_rows):
                        icon = {
                            "completed": "✅",
                            "cancelled": "❌",
                            "processing": "🔄",
                            "in_progress": "🔄",
                            "error": "⚠️",
                        }.get(s[2], "⬜")
                        lines.append(f"{i + 1}. [#{s[0]}] {icon} {s[1]}")
                    context_blocks.append("\n".join(lines))
            except Exception as e:
                log.warning("Failed to get subtasks: %s", e)

        # 2e. Continuation self-orientation
        if thread_id is not None:
            try:
                cont = build_continuation_block(cursor, int(thread_id))
                if cont:
                    context_blocks.append(cont)
            except Exception as e:
                log.warning("continuation context unavailable: %s", e)

        # 2e-ext0. Interrupted-attempt warning
        if thread_id is not None:
            try:
                n = count_interrupted_attempts(cursor, int(thread_id))
                if n is not None and n > 0:
                    context_blocks.append(
                        f"WARNING: {n} prior attempt(s) of this task were INTERRUPTED (iteration "
                        "limit). Read their summaries above. If you are about to do what they "
                        "did, you are repeating a mistake."
                    )
            except Exception as e:
                log.warning("interrupted-attempt count unavailable: %s", e)

        # 2e-ext. Cross-task channel context
        if thread_id is not None:
            try:
                cross = build_cross_task_block(cursor, int(thread_id))
                if cross:
                    context_blocks.append(cross)
            except Exception as e:
                log.warning("cross-task context unavailable: %s", e)

        context = "\n\n---\n\n".join(context_blocks)
        user = user_message

        # 2f. Workflow step prompt mapping (inverse for tester/reviewer)
        if thread_id is not None:
            try:
                cursor.execute(
                    "SELECT workflow_id, workflow_step FROM threads WHERE id = %s",
                    (int(thread_id),),
                )
                wf = cursor.fetchone()
                if wf and wf[0] and wf[1]:
                    role = step_to_role(wf[1])
                    template = None
                    if role:
                        template = load_role_template(data_dir, profile_name, wf[0], role)
                    system, user = apply_workflow_mapping(
                        system, user, user_message, wf[1], template
                    )
            except TemplateError:
                # A template reference that is not a valid PROFILE template is
                # a hard error (operator directive 2026-09-24) - never swallow.
                raise
            except Exception as e:
                log.warning("workflow step lookup unavailable: %s", e)

        cursor.close()

        # Plan resolution (mirrors Rust): true=plan, false=no plan,
        # null/absent = let plugin-level config decide.
        plan_input = args.get("plan")
        if plan_input is not None:
            plan = bool(plan_input)
        else:
            max_chars = cfg.get("planning_complexity_max_chars", 60)
            keywords_str = cfg.get("planning_complexity_keywords", "")
            keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]
            lower_user = user_message.lower()
            has_keyword = False
            if keywords:
                has_keyword = any(k in lower_user for k in keywords)
            plan = len(user_message) > max_chars or has_keyword

        result = json.dumps({
            "system": system,
            "memory": memory,
            "context": context,
            "user": user,
            "plan": plan,
        }, indent=2, ensure_ascii=False)

        send_json(make_success(req_id, make_tool_result(result)))

    except Exception as e:
        log.error("generate tool failed: %s", e, exc_info=True)
        send_json(make_success(req_id, make_tool_result(f"Error: {e}", True)))


# ---------------------------------------------------------------------------
# Token budget measurement (mirror Rust measure_size: chars/4 fallback, real
# BPE when tokenizer_encoding configured and tiktoken available)
# ---------------------------------------------------------------------------

def serialize_messages_rust(items):
    """Serialize ChatMessage dicts exactly like Rust serde_json::to_string:
    field order role, content, tool_call_id?, tool_calls?, name?; compact,
    unicode raw (ensure_ascii=False)."""
    out = []
    for m in items:
        d = {"role": m.get("role", ""), "content": m.get("content", "")}
        if m.get("tool_call_id") is not None:
            d["tool_call_id"] = m["tool_call_id"]
        if m.get("tool_calls") is not None:
            calls = []
            for tc in m["tool_calls"]:
                calls.append({
                    "id": tc.get("id", ""),
                    "type": tc.get("type", "function"),
                    "function": {
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": tc.get("function", {}).get("arguments", ""),
                    },
                })
            d["tool_calls"] = calls
        if m.get("name") is not None:
            d["name"] = m["name"]
        out.append(d)
    return json.dumps(out, separators=(",", ":"), ensure_ascii=False)


def measure_size(messages, tokenizer_encoding):
    """Rust measure_size: chars/4 fallback, tiktoken BPE when configured."""
    chars = 0
    for m in messages:
        chars += len(m.get("content", ""))
        calls = m.get("tool_calls") or []
        for tc in calls:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            chars += len(fn.get("name", "")) + len(fn.get("arguments", ""))
    if not tokenizer_encoding:
        return chars // 4
    if tiktoken is None:
        return chars // 4
    try:
        text = serialize_messages_rust(messages)
        enc = None
        try:
            enc = tiktoken.encoding_for_model(tokenizer_encoding)
        except Exception:
            enc = tiktoken.get_encoding(tokenizer_encoding)
        return len(enc.encode(text, disallowed_special=()))
    except Exception as e:
        log.warning("[prompt] Failed to load BPE encoding '%s': %s: falling back to chars/4",
                    tokenizer_encoding, e)
        return chars // 4


# ---------------------------------------------------------------------------
# Compaction (mirror compact.rs: frozen summary block + dump.rs + notes.rs)
# ---------------------------------------------------------------------------

COMPACTION_SUMMARY_MARKER = "=== Compaction Summary ==="

# --- Deterministic shrink fallback (mirrors Rust main.rs) -------------------

# Minimum content size (chars) the truncation fallback leaves in a message:
# below this the truncation marker alone would dominate, so the message is
# skipped and the next-largest candidate is tried.
SHRINK_MIN_CHARS = 1000

# Absolute bound on truncation rounds. Each round reduces the largest eligible
# message by at least the marker cost, so the loop always terminates.
SHRINK_MAX_ROUNDS = 24

# Last-resort floor (chars) per message: when the per-message floors of the
# readable stage sum above the target, eligible messages are cut to a stub.
SHRINK_FLOOR_CHARS = 200


def truncation_marker(removed_chars):
    """Marker inserted where content was cut (byte-identical to Rust)."""
    return (
        "\n[... {} chars truncated by the context compactor to fit the prompt "
        "token budget; the full text is in the thread context-*.json dump / "
        "auto-notes.md ...]\n"
    ).format(removed_chars)


def truncate_middle(content, keep_head, keep_tail):
    """Keep head+tail chars, cut the middle. Returns (new_content, dropped)."""
    total = len(content)
    if total <= keep_head + keep_tail + len(truncation_marker(0)):
        return content, 0
    removed = total - keep_head - keep_tail
    return content[:keep_head] + truncation_marker(removed) + content[total - keep_tail:], removed


def message_chars(m):
    """Size of a message as the compactor sees it (content + tool-call args)."""
    calls = 0
    for tc in (m.get("tool_calls") or []):
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        calls += len(str(fn.get("name", ""))) + len(str(fn.get("arguments", "")))
    return len(str(m.get("content", ""))) + calls


def is_shrinkable(m, idx, newest_idx, last_user_idx, first_system_idx):
    """May this message's CONTENT be truncated by the budget fallback?

    Tool results are the primary target (full text survives in the
    context-*.json dump / auto-notes.md); assistant text is truncatable while
    its tool_calls payload is preserved verbatim; the frozen summary block and
    the per-iteration "=== " injections are digests. NEVER touched: the system
    prompt (first system message), the newest message and the current (last)
    user turn.
    """
    if idx == newest_idx or idx == last_user_idx or idx == first_system_idx:
        return False
    role = m.get("role", "")
    if role == "tool":
        return True
    if role == "assistant":
        return m.get("tool_calls") is not None
    if role == "system":
        content = str(m.get("content", ""))
        return content.startswith(COMPACTION_SUMMARY_MARKER) or content.startswith("=== ")
    return False


def shrink_messages_to_target(messages, target, tokenizer_encoding):
    """Deterministically shrink messages until measure_size <= target.

    Bounded last resort behind the progressive drain: each round removes the
    estimated token deficit from the LARGEST eligible message (head+tail kept,
    middle replaced by a marker), so it always terminates and never touches the
    system prompt, the current user turn, the newest message or the tool-call
    structure. Returns the number of content chars dropped.
    """
    marker_cost = len(truncation_marker(0))
    dropped_total = 0
    # Two stages: readable digests first, short stubs only when the array is
    # STILL over target (many eligible messages whose floors sum above it).
    for floor in (SHRINK_MIN_CHARS, SHRINK_FLOOR_CHARS):
        for _ in range(SHRINK_MAX_ROUNDS):
            size = measure_size(messages, tokenizer_encoding)
            if size <= target:
                break
            newest_idx = len(messages) - 1
            first_system_idx = next(
                (i for i, m in enumerate(messages) if m.get("role") == "system"), None)
            last_user_idx = next(
                (i for i in range(len(messages) - 1, -1, -1)
                 if messages[i].get("role") == "user"), None)
            best = None
            for i, m in enumerate(messages):
                if not is_shrinkable(m, i, newest_idx, last_user_idx, first_system_idx):
                    continue
                chars = len(str(m.get("content", "")))
                # 3x margin: every round must make NET progress (the marker we
                # insert must cost less than what we remove), otherwise a
                # deficit smaller than the marker reshapes content while the
                # array stays over the target.
                if chars - floor <= 3 * marker_cost:
                    continue
                if best is None or chars > best[0]:
                    best = (chars, i)
            if best is None:
                break
            chars, idx = best
            removable = max(0, chars - floor)
            total_chars = sum(message_chars(m) for m in messages)
            chars_per_token = max(1, total_chars // size) if size > 0 else 4
            deficit = max(0, size - target)
            to_remove = max(deficit * chars_per_token, 3 * marker_cost)
            if to_remove > removable:
                to_remove = removable
            if to_remove <= marker_cost:
                break
            keep = chars - to_remove
            keep_head = keep * 2 // 3
            keep_tail = keep - keep_head
            new_content, dropped = truncate_middle(
                str(messages[idx].get("content", "")), keep_head, keep_tail)
            if dropped == 0:
                break
            messages[idx]["content"] = new_content
            dropped_total += dropped
    return dropped_total

READ_TOOL_PREFIXES = (
    # Current grammar `{plugin}__{tool}`.
    "filesystem__",
    "search__",
    "skills__view",
    "git__status",
    "git__run_command",
    # Pre-separator-flip names: the one-release alias window.
    "filesystem_read",
    "filesystem_list",
    "filesystem_search",
    "filesystem_info",
    "search_database",
    "search_messages",
    "search_wiki",
    "skills_view",
    "git_status",
    "git_run-command",
)


def legacy_read_type_tool(name):
    return any(name.startswith(p) for p in READ_TOOL_PREFIXES)


def is_read_type_tool(name, settings):
    """Descriptor-driven read-type check (audit V-2): when the core supplies
    the declared read-only tools, THAT set decides; otherwise the legacy
    prefix list is used as a fallback."""
    read_only = settings.get("read_only_tools") or []
    if read_only:
        return name in read_only
    return legacy_read_type_tool(name)


# --- dump.rs: context-<iter>.json digests ---

DUMP_MAX_BYTES = 200 * 1024
DUMP_MAX_FILES = 3
DIGEST_HEAD_CHARS = 400
# Process-lifetime dedupe set (mirrors Rust static Mutex<HashSet>).
_APPENDED = set()


def _stable_hash(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def append_dump(dir_path, iter_num, tool, args, content):
    """Rust append_dump: JSON-lines digest, deduped by (file, tool+args)."""
    if tool == "" and content == "":
        return False
    try:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    file_path = str(Path(dir_path) / f"context-{iter_num}.json")
    dedupe_key = (file_path, _stable_hash(f"{tool}\u0000{args}"))
    if dedupe_key in _APPENDED:
        return False

    chars = len(content)
    head = content[:DIGEST_HEAD_CHARS]
    tail = content[-DIGEST_HEAD_CHARS:] if chars > DIGEST_HEAD_CHARS else content
    args_s = args if args.strip() else f"[content:{_stable_hash(content)}]"
    line = json.dumps({
        "tool": tool,
        "args": args_s,
        "chars": chars,
        "head": head,
        "tail": tail,
    }, separators=(",", ":"), ensure_ascii=False)

    try:
        with open(file_path, "a") as f:
            f.write(line + "\n")
    except OSError:
        return False
    _APPENDED.add(dedupe_key)
    enforce_caps(dir_path, iter_num)
    return True


def enforce_caps(dir_path, iter_num):
    """Rust enforce_caps: per-file 200KB (keep newest lines) + keep last 3
    context-*.json files."""
    file_path = Path(dir_path) / f"context-{iter_num}.json"
    try:
        if file_path.stat().st_size > DUMP_MAX_BYTES:
            content = read_file(file_path)
            lines = content.splitlines()
            kept = []
            bytes_used = 0
            for l in reversed(lines):
                lb = len(l.encode("utf-8")) + 1
                if bytes_used + lb > DUMP_MAX_BYTES:
                    break
                bytes_used += lb
                kept.append(l)
            kept.reverse()
            with open(file_path, "w") as f:
                f.write("\n".join(kept) + "\n")
    except OSError:
        pass

    files = []
    try:
        for e in Path(dir_path).iterdir():
            name = e.name
            if name.startswith("context-") and name.endswith(".json"):
                rest = name[len("context-"):-len(".json")]
                if rest.isdigit():
                    files.append((name, int(rest)))
    except OSError:
        return
    if len(files) > DUMP_MAX_FILES:
        files.sort(key=lambda x: x[1])
        for name, _ in files[:len(files) - DUMP_MAX_FILES]:
            try:
                Path(dir_path, name).unlink()
            except OSError:
                pass


def note_append(dir_path, name, content):
    """Rust notes::note_append: append content.trim_end() + '\\n'."""
    try:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    try:
        with open(Path(dir_path) / name, "a") as f:
            if content.strip():
                f.write(content.rstrip() + "\n")
    except OSError:
        pass


def prune_summary_block(content, max_chars):
    """Rust prune_summary_block: bound the frozen block; newest entries kept,
    oldest dropped; header preserved byte-for-byte."""
    if len(content) <= max_chars:
        return content
    idx = content.find("\n- [")
    header_end = idx + 1 if idx != -1 else 0
    header = content[:header_end]
    entries_text = content[header_end:]
    budget = max(0, max_chars - len(header) - 64)

    entry_list = []
    cur = ""
    for line in entries_text.splitlines():
        if line.startswith("- [") and cur:
            entry_list.append(cur)
            cur = line
        elif cur:
            cur = cur + "\n" + line
        else:
            cur = line
    if cur:
        entry_list.append(cur)

    kept = []
    used = 0
    for entry in reversed(entry_list):
        cost = len(entry) + 1
        if used + cost > budget:
            break
        used += cost
        kept.append(entry)
    kept.reverse()
    if not kept:
        kept = [entry_list[-1]] if entry_list else []
    return header + "\n".join(kept) + "\n[older entries pruned - see context-*.json dumps]"


def compact_old_assistant_messages(messages, keep_recent, thread_dir, current_iteration, settings):
    """Rust compact::compact_old_assistant_messages: drain the oldest
    tool-call turns into ONE frozen summary block. Returns
    (removed, dump_file, dump_entries). Null-contract: no drain when
    tool-call turns <= keep_recent (returns zeroes, list untouched)."""
    removed = 0
    dump_file = None
    dump_entries = 0

    summary_idx = None
    for i, m in enumerate(messages):
        if m.get("role") == "system" and m.get("content", "").startswith(COMPACTION_SUMMARY_MARKER):
            summary_idx = i
            break

    tool_indices = [
        i for i, m in enumerate(messages)
        if m.get("role") == "assistant" and m.get("tool_calls") is not None
    ]

    if len(tool_indices) <= keep_recent:
        return removed, dump_file, dump_entries

    n_drain = len(tool_indices) - keep_recent
    drain_start = tool_indices[0]
    drain_end = tool_indices[n_drain - 1] + 1
    while drain_end < len(messages) and messages[drain_end].get("role") == "tool":
        drain_end += 1

    entries = []
    i = drain_start
    while i < drain_end:
        m = messages[i]
        if m.get("role") == "assistant" and m.get("tool_calls") is not None:
            calls = m.get("tool_calls") or []
            names = [tc.get("function", {}).get("name", "") for tc in calls]

            tool_end = i + 1
            while tool_end < drain_end and messages[tool_end].get("role") == "tool":
                tool_end += 1
            tool_count = tool_end - i - 1

            # WS-2: durable dump + auto-notes of the tool results about to
            # be drained.
            if thread_dir:
                for tm in messages[i + 1:tool_end]:
                    if tm.get("role") == "tool" and tm.get("content"):
                        tool_name = tm.get("name") or ""
                        args = ""
                        for tc in calls:
                            if tc.get("function", {}).get("name") == tool_name:
                                args = tc.get("function", {}).get("arguments", "")
                                break
                        if append_dump(thread_dir, current_iteration, tool_name, args, tm.get("content", "")):
                            dump_entries += 1
                        if is_read_type_tool(tool_name, settings):
                            note_append(
                                thread_dir,
                                "auto-notes.md",
                                f"## [engine:auto-note {tool_name}]\n"
                                f"{tm['content'][:settings['read_excerpt_chars']]}\n",
                            )
                if tool_count > 0:
                    dump_file = f"context-{current_iteration}.json"

            # Excerpt for the summary entry (same caps).
            excerpt = ""
            total_excerpt = 0
            for tm in messages[i + 1:tool_end]:
                tool_name = tm.get("name") or ""
                is_read = is_read_type_tool(tool_name, settings)
                excerpt_chars = settings["read_excerpt_chars"] if is_read else settings["tool_excerpt_chars"]
                content_preview = tm.get("content", "")[:excerpt_chars]
                chunk_len = len(content_preview)
                if total_excerpt + chunk_len > settings["total_excerpt_cap"]:
                    break
                total_excerpt += chunk_len
                excerpt += content_preview + "\n"

            if excerpt.strip() == "":
                entries.append(
                    f"- [iter {current_iteration}] {', '.join(names)} → (results drained)"
                )
            else:
                entries.append(
                    f"- [iter {current_iteration}] {', '.join(names)} →\n{excerpt.rstrip()}"
                )
            removed += tool_count
            i = tool_end
        elif m.get("role") == "system" and m.get("content", "").startswith(COMPACTION_SUMMARY_MARKER):
            # Defensive: never fold the summary block into itself.
            i += 1
        else:
            # Non-tool-call message inside the drained span: preserve a
            # preview so the information is not silently lost.
            total = len(m.get("content", ""))
            preview = m.get("content", "")[:800]
            suffix = "…" if total > 800 else ""
            entries.append(f"- [{m.get('role', '')}] {preview}{suffix}")
            i += 1

    joined_entries = "\n".join(entries)
    if summary_idx is not None:
        messages[summary_idx]["content"] = messages[summary_idx]["content"] + "\n" + joined_entries
        messages[summary_idx]["content"] = prune_summary_block(
            messages[summary_idx]["content"], settings["max_summary_chars"]
        )
        del messages[drain_start:drain_end]
    else:
        block_content = (
            f"{COMPACTION_SUMMARY_MARKER}\n"
            "Frozen prefix block: older conversation turns were compacted into this summary, "
            "oldest first. Everything before this block is the fixed preamble; everything "
            "after is the live conversation. Recover destroyed read results from auto-notes.md "
            "/ context-*.json dumps.\n"
            + joined_entries
        )
        block_content = prune_summary_block(block_content, settings["max_summary_chars"])
        block = {"role": "system", "content": block_content}
        messages.insert(drain_start, block)
        del messages[drain_start + 1:drain_end + 1]

    return removed, dump_file, dump_entries


def read_only_tool_names(args):
    """Read-only tool names declared by the plugin descriptors (audit V-2).

    The core passes the registry-derived set; a missing/empty argument means
    an older core, in which case the legacy prefix list is used as fallback.
    """
    raw = args.get("read_only_tools")
    if not isinstance(raw, list):
        return []
    return [v for v in raw if isinstance(v, str)]


def handle_compact_messages(req_id, arguments):
    """Rust handle_compact_messages: budget-gated, progressive passes, frozen
    summary block; returns the compacted array or null."""
    try:
        args = arguments or {}
        messages_arr = args.get("messages")
        if not isinstance(messages_arr, list):
            send_json(make_success(req_id, make_tool_result(
                "Missing required argument: 'messages' (array of ChatMessage)", True)))
            return

        cfg = get_config()
        keep_recent = args.get("keep_recent")
        keep_recent = int(keep_recent) if keep_recent is not None else cfg.get("compact_keep_recent", 3)

        # Parse into ChatMessage dicts (mirror serde parse; missing role or
        # content is a parse error).
        messages = []
        try:
            for m in messages_arr:
                if not isinstance(m, dict):
                    raise ValueError("message is not an object")
                if "role" not in m or "content" not in m:
                    raise ValueError("missing field `role` or `content`")
                parsed = {"role": str(m["role"]), "content": str(m["content"])}
                for key in ("tool_call_id", "name"):
                    if m.get(key) is not None:
                        parsed[key] = m[key]
                if m.get("tool_calls") is not None:
                    calls = []
                    for tc in m["tool_calls"]:
                        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                        calls.append({
                            "id": tc.get("id", "") if isinstance(tc, dict) else "",
                            "type": tc.get("type", "function") if isinstance(tc, dict) else "function",
                            "function": {
                                "name": fn.get("name", "") if isinstance(fn, dict) else "",
                                "arguments": fn.get("arguments", "") if isinstance(fn, dict) else "",
                            },
                        })
                    parsed["tool_calls"] = calls
                messages.append(parsed)
        except Exception as e:
            send_json(make_success(req_id, make_tool_result(f"Failed to parse messages: {e}", True)))
            return

        hard_budget = args.get("hard_budget")
        if hard_budget is None:
            send_json(make_success(req_id, make_tool_result(
                "Missing required argument: 'hard_budget' (token budget; the omniagent passes the "
                "resolved per-thread hard token budget)", True)))
            return
        hard_budget = int(hard_budget)
        soft_budget = args.get("soft_budget")
        if soft_budget is None:
            send_json(make_success(req_id, make_tool_result(
                "Missing required argument: 'soft_budget' (token budget; the omniagent passes the "
                "resolved per-thread soft token budget)", True)))
            return
        soft_budget = int(soft_budget)

        thread_dir = args.get("thread_dir")
        if thread_dir is not None:
            thread_dir = str(thread_dir)
        current_iteration = int(args.get("current_iteration") or 0)

        tokenizer_encoding = cfg.get("tokenizer_encoding", "")
        # Engine override (Rust main.rs L1785-1800): when the core's post-call
        # provider usage shows the context is still over the hard budget (the
        # local measure can under-count versus the provider tokenizer), the
        # core re-invokes us with force_compact=true so the gate cannot
        # silently under-trigger (0-compaction incident, threads 1139/1140).
        # Force skips the threshold check and compacts toward the soft budget
        # whenever drainable turns exist; null-contract otherwise preserved.
        # Boolean-only, exactly like args["force_compact"].as_bool().
        force_compact = args.get("force_compact") is True
        before = len(messages)
        dump_file = None
        entries = 0

        current_size = measure_size(messages, tokenizer_encoding)
        # Accounting alignment with the provider (mirror Rust main.rs): this
        # plugin measures MESSAGES ONLY, while the provider also bills the tool
        # schemas, the chat template and its own tokenizer. `billed_prompt_tokens`
        # is the provider's ground-truth size for the array it measured as
        # `measured_tokens`, so their difference is exactly the invisible
        # overhead. Compaction must reach hard_budget - overhead (- headroom),
        # not merely the soft budget, or the provider keeps billing over the
        # hard budget and the agent force-compacts forever (chronic overshoot
        # error: 279/24h on 2026-09-20).
        billed_prompt_tokens = int(args.get("billed_prompt_tokens") or 0)
        measured_tokens = int(args.get("measured_tokens") or 0)
        # A billing baseline exists when the core paired a provider-billed prompt
        # token count with the plugin's measured size of the SAME request. The
        # difference is the invisible overhead (tool schemas, chat template, the
        # provider's own tokenizer, cached-token accounting) that the plugin's
        # message-only measure cannot see. It is applied to the gate AND targets
        # whenever a baseline exists - NOT only after an overshoot - so compaction
        # fires BEFORE the provider bills over the hard budget (chronic 279/24h
        # "condensation did not reduce it" fired once per growth span because the
        # gate compared the raw measured size against the hard budget while the
        # provider billed measured + overhead).
        if measured_tokens > 0 and billed_prompt_tokens > measured_tokens:
            # Clamp: a provider number can be a cumulative/aggregate total, and
            # tokenizers differ; never let the derived overhead drive the
            # reduction target below half the hard budget.
            overhead = min(billed_prompt_tokens - measured_tokens, hard_budget // 2)
        else:
            overhead = 0
        # Headroom absorbs tokenizer drift and the per-iteration system injections
        # the compaction gate cannot measure; applied whenever a billing baseline
        # exists (the estimate comes from the PREVIOUS request, the next one drifts).
        # Observed provider overshoot of the PREVIOUS request (mirror Rust
        # main.rs, thread 3998): the provider billed MORE than this plugin's
        # target allowed by adding tokens after our measure. The billed/measured
        # pair is inherently one request old, so the excess is exactly how much
        # the overhead estimate was short - carrying it into the headroom makes
        # the NEXT reduction account for it instead of repeating the overshoot.
        # One-directional: it can only TIGHTEN the internal reduction target,
        # never relax an operator-configured budget.
        observed_overshoot = int(args.get("observed_overshoot_tokens") or 0)
        headroom = (
            (cfg.get("compact_headroom_tokens", 2000) if overhead > 0 else 0)
            + observed_overshoot
        )
        if observed_overshoot > 0:
            log.info(
                "[prompt] observed provider overshoot %s tokens fed back: "
                "reduction target tightened to fit the hard budget",
                observed_overshoot,
            )
        # Size our own measure must reach so the provider bills under the hard
        # budget. Never above the hard budget; the soft budget stays the stricter
        # reduction target whenever it already sits below it.
        hard_target = max(0, hard_budget - overhead - headroom)
        # The size the provider needs: whenever a billing baseline exists, the
        # unmeasurable overhead has to come out of our own measure as well -
        # otherwise the provider bills over while the plugin still sees "under".
        must_fit_target = (
            hard_target if (overhead > 0 or observed_overshoot > 0) else hard_budget
        )
        # Progressive-drain target: the soft budget (the historical reduction
        # target), tightened to hard_target whenever a billing baseline exists.
        reduce_target = min(soft_budget, hard_target) if overhead > 0 else soft_budget
        # Truncation-fallback target: the drain target, but NEVER above what fits
        # under the hard budget (a soft budget configured above the hard budget
        # must not disable the reduction).
        effective_target = min(reduce_target, must_fit_target)
        # Threshold gate: compact ONLY when the provider-fit HARD budget is
        # exceeded (or the core forces it because the provider billed over the
        # hard budget). With a billing baseline the gate is hard_budget -
        # overhead (- headroom): the provider bills measured + overhead, so the
        # raw measured size must sit BELOW the hard budget by the known overhead
        # or the provider bills over and the overshoot error fires (chronic
        # 279/24h). The SOFT budget is the REDUCTION TARGET consumed below, never
        # a trigger: a prompt whose size sits between soft and hard stays
        # completely UNTOUCHED (no draining, no truncation, null-contract).
        if force_compact or current_size > must_fit_target:
            settings = {
                "read_only_tools": read_only_tool_names(args),
                "tool_excerpt_chars": cfg.get("tool_excerpt_chars", 800),
                "total_excerpt_cap": cfg.get("total_excerpt_cap", 4000),
                "read_excerpt_chars": cfg.get("read_excerpt_chars", 2000),
                "max_summary_chars": cfg.get("max_summary_chars", 50000),
            }
            compact_max_passes = cfg.get("compact_max_passes", 3)
            compact_keep_step = cfg.get("compact_keep_step", 1)
            # keep floors at 1 so the in-flight tool-call chain is never drained.
            keep = max(1, keep_recent)
            for pass_num in range(compact_max_passes):
                removed, df, de = compact_old_assistant_messages(
                    messages, keep, thread_dir, current_iteration, settings
                )
                if df:
                    dump_file = df
                entries += de
                after_size = measure_size(messages, tokenizer_encoding)
                if after_size <= reduce_target or keep <= 1:
                    break
                if pass_num + 1 == compact_max_passes:
                    break
                keep = max(1, keep - compact_keep_step)

        # Deterministic fallback (mirror Rust main.rs): truncate the largest
        # eligible content, head+tail kept, until the measured size is under the
        # HARD fit target or nothing is left to shrink. The fallback exists to
        # FIT THE HARD BUDGET (the provider's limit), never to reach the soft
        # budget: the soft budget is the compaction DRAIN target, and truncating
        # results down to it while the hard budget is larger destroys results the
        # agent still needs (the v0.3.2 "dumb and slow" regression, thread 2812).
        # Never touches the system prompt, the current user turn, the newest
        # message or the tool-call structure.
        truncated_chars = 0
        if measure_size(messages, tokenizer_encoding) > must_fit_target:
            truncated_chars = shrink_messages_to_target(
                messages, must_fit_target, tokenizer_encoding)

        after = len(messages)
        changed = before != after or truncated_chars > 0
        after_size = measure_size(messages, tokenizer_encoding)
        # `over_budget` tells the core whether the prompt STILL exceeds the size
        # that fits under the hard budget; while true the core keeps forcing
        # compaction.
        still_over = after_size > must_fit_target

        result = {
            "messages": [
                {
                    "role": m.get("role", ""),
                    "content": m.get("content", ""),
                    **({"tool_call_id": m["tool_call_id"]} if m.get("tool_call_id") is not None else {}),
                    **({"tool_calls": m["tool_calls"]} if m.get("tool_calls") is not None else {}),
                    **({"name": m["name"]} if m.get("name") is not None else {}),
                }
                for m in messages
            ] if changed else None,
            "was_compacted": changed,
            "iteration": current_iteration,
            "dump_file": dump_file,
            "entries": entries,
            "before_count": before,
            "after_count": after,
            "measured_tokens": after_size,
            "observed_overshoot_tokens": observed_overshoot,
            "effective_target": effective_target,
            "truncate_target": must_fit_target,
            "over_budget": still_over,
            "truncated_chars": truncated_chars,
        }
        text = json.dumps(result, indent=2, ensure_ascii=False)
        send_json(make_success(req_id, make_tool_result(text)))

    except Exception as e:
        log.error("compact-messages tool failed: %s", e, exc_info=True)
        send_json(make_success(req_id, make_tool_result(f"Error: {e}", True)))

# ---------------------------------------------------------------------------
# MCP lifecycle (mirror the Rust server: same tool names, descriptions,
# input schemas, and serverInfo)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "prompt_generate",
        "description": (
            "Generate the complete LLM prompt for a conversation, including system prompt "
            "(identity, tool guidance, memory, user profile), thread context (recent messages, "
            "summaries, skills, subtasks), and optional planning instructions. Returns the full "
            "prompt as a JSON string. This is the single source of truth for prompt building: "
            "no other prompt assembly is needed."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "profile_name": {
                    "type": "string",
                    "description": "Profile name (default: omni)"
                },
                "platform": {
                    "type": "string",
                    "description": "Platform identifier (e.g. 'telegram', 'mattermost')"
                },
                "platform_hint": {
                    "type": "string",
                    "description": "Formatting hint declared by the platform plugin's initialize capabilities (audit V-5). Optional: when absent a named platform falls back to a generic markdown note."
                },
                "system_message": {
                    "type": "string",
                    "description": "Optional system message override"
                },
                "user_message": {
                    "type": "string",
                    "description": "User's message to include in the prompt"
                },
                "tool_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of available tool names"
                },
                "thread_id": {
                    "type": "integer",
                    "description": "Thread ID for context assembly (recent messages, subtasks)"
                },
                "channel_id": {
                    "type": "string",
                    "description": "Channel name for context assembly (summaries)"
                },
                "plan": {
                    "type": "boolean",
                    "description": "Plan mode suggestion: true=plan, false=no plan, null=let plugin decide based on config"
                }
            },
            "required": []
        }
    },
    {
        "name": "prompt_compact-messages",
        "description": (
            "Compact and prune a conversation to stay within token budgets. REQUIRES "
            "'soft_budget'/'hard_budget' token params (resolved per-thread by the "
            "omniagent; chars/4 fallback when no tokenizer). Compacts ONLY when the "
            "hard budget is exceeded (null-contract otherwise): drains the oldest "
            "tool-call turns into ONE frozen '=== Compaction Summary ===' block right "
            "after the system prompt, keeps recent turns verbatim, excerpts older "
            "read-type results and auto-notes them (auto-notes.md in thread_dir). The "
            "returned messages array must keep the prefix byte-stable; only the tail "
            "may change. Returns the compacted message array or null."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "description": "Array of ChatMessage objects to compact"
                },
                "keep_recent": {
                    "type": "integer",
                    "description": "Number of most recent messages to always keep (default: 3)"
                },
                "soft_budget": {
                    "type": "integer",
                    "description": "Required soft token budget: the reduction target when the hard budget is exceeded"
                },
                "hard_budget": {
                    "type": "integer",
                    "description": "Required hard token budget: compaction/pruning triggers when the context exceeds it"
                }
            },
            "required": ["messages", "soft_budget", "hard_budget"]
        }
    },
]


def handle_initialize(req_id):
    result = {
        "protocolVersion": MCP_PROTOCOL_VERSION,
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": {"name": "mcp-server-prompt", "version": "0.1.0"},
    }
    send_json(make_success(req_id, result))
    log.info("Initialized: mcp-server-prompt (python) v0.1.0")


def handle_tools_list(req_id):
    send_json(make_success(req_id, {"tools": TOOLS}))
    log.info("tools/list returned 2 tools")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global initialized

    log.info("prompt-python MCP server starting (PID=%d)", os.getpid())
    log.info("OMNI_DIR=%s", os.environ.get("OMNI_DIR", "(not set)"))
    log.info("DATABASE_URL=%s", "set" if os.environ.get("DATABASE_URL") else "(not set)")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            log.error("Failed to parse JSON-RPC: %s", e)
            continue

        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params", {})
        meta = params.get("_meta") if isinstance(params, dict) else None

        if method == "initialize":
            if req_id is not None:
                handle_initialize(req_id)
                initialized = True

        elif method == "notifications/initialized":
            log.info("Client initialized notification received")

        elif method == "configure":
            # The builtin receives plugin config via run_server_with_config's
            # configure callback; accept the same message when the framework
            # sends it to python servers too.
            if isinstance(params, dict):
                for k, v in params.items():
                    if v is None:
                        continue
                    if isinstance(v, (int, float, bool)):
                        CONFIG_OVERRIDES[k] = str(v)
                    elif isinstance(v, str):
                        CONFIG_OVERRIDES[k] = v
                log.info("configure received: %d keys", len(CONFIG_OVERRIDES))
            if req_id is not None:
                send_json(make_success(req_id, {"configured": True}))

        elif method == "tools/list":
            if not initialized:
                if req_id is not None:
                    send_json(make_error(req_id, -32000, "Server not initialized"))
                continue
            if req_id is not None:
                handle_tools_list(req_id)

        elif method == "tools/call":
            if not initialized:
                if req_id is not None:
                    send_json(make_error(req_id, -32000, "Server not initialized"))
                continue
            if req_id is not None:
                tool_name = params.get("name", "")
                arguments = params.get("arguments", {}) if isinstance(params, dict) else {}

                if tool_name == "prompt_generate":
                    handle_generate(req_id, arguments, meta)
                elif tool_name == "prompt_compact-messages":
                    handle_compact_messages(req_id, arguments)
                else:
                    send_json(make_error(req_id, -32602, f"Unknown tool: {tool_name}"))

        else:
            log.warning("Unknown method: %s", method)
            if req_id is not None:
                send_json(make_error(req_id, -32601, f"Method not found: {method}"))

    log.info("prompt-python MCP server shutting down (stdin closed)")


if __name__ == "__main__":
    main()
