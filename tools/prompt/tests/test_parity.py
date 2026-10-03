#!/usr/bin/env python3
"""Parity guards for the remote Python prompt plugin (tools/prompt/server.py).

These tests pin the parity surface against the builtin Rust prompt plugin
(omniagent plugins/tools/prompt). They need NO database:

  * the pure parity surface is asserted by importing server.py as a module
    (read-type detection, descriptor precedence, token-budget measurement,
    manifest schema, guidance text);
  * the tool registry and the compaction path are driven end to end over
    stdio JSON-RPC by spawning `python3 server.py`, exactly like the runtime
    does (same transport the builtin exposes through run_server_with_config).

Run (inside the omniagent runtime image / dev container, where psycopg2 is
installed, like the plugin itself):

    python3 tools/prompt/tests/test_parity.py

Regenerate the golden tool list after an intentional, Rust-verified change
(the differential harness must show the Rust side matching it):

    PROMPT_PARITY_UPDATE=1 python3 tools/prompt/tests/test_parity.py

Byte-identical OUTPUT parity (prompt_generate / prompt_compact-messages over
real DB state) is proven by tools/prompt/tests/differential_harness.py; the
procedure is documented in tools/prompt/PARITY.md.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLUGIN_DIR = HERE.parent
FIXTURE_DIR = HERE / "fixtures"
TOOLS_LIST_FIXTURE = FIXTURE_DIR / "tools_list.json"

sys.path.insert(0, str(PLUGIN_DIR))
import server  # noqa: E402  (the plugin module under test)

# Builtin Rust plugin manifest: compared when present (workspace checkouts /
# the dev container expose it at the default path).
RUST_PLUGIN_JSON = Path(os.environ.get(
    "PROMPT_RUST_PLUGIN_JSON",
    "/opt/workspace/omniagent/plugins/tools/prompt/plugin.json",
))

# Reference manifest: the builtin Rust plugin's config_schema, snapshotted at
# tests/fixtures/rust_plugin_config_schema.json and, when the omniagent
# checkout is present, compared live.
RUST_SCHEMA_FIXTURE = FIXTURE_DIR / "rust_plugin_config_schema.json"

# Exact-parity contract: NO label may differ. The omni_dir label was the
# last hold-out (omni-plugins manifest convention "OMNI_DIR"); it is now
# pinned to the builtin Rust value "Omni Dir" so gate 1 is a literal
# zero-diff manifest comparison.
ALLOWED_LABEL_DIFFS = {}

EXPECTED_TOOL_NAMES = ["prompt_generate", "prompt_compact-messages"]

# Tool-naming grammar c78e874: current `{plugin}__{tool}` names.
GRAMMAR_CURRENT = [
    "filesystem__write", "filesystem__read",
    "notes__note_write", "notes__note_append", "notes__note_read",
    "notes__note_list", "notes__note_rm",
    "docker__compose", "subtasks__manage_subtasks",
]
# Pre-flip (one-release alias window) names that must no longer appear in the
# user-visible guidance text.
GRAMMAR_LEGACY_IN_GUIDANCE = [
    "filesystem_write", "filesystem_read", "filesystem__exec",
    "notes_note-write", "notes_note-write", "notes_note_append",
    "docker_compose", "subtasks_manage-subtasks", "subtasks_manage_subtasks",
]

READ_TOOLS_LEGACY = [
    "filesystem__read", "filesystem__list", "filesystem__search",
    "filesystem__info", "search__database", "search__messages", "search__wiki",
    "skills__view", "git__status", "git__run_command",
    "filesystem_read", "filesystem_list", "filesystem_search",
    "filesystem_info", "search_database", "search_messages", "search_wiki",
    "skills_view", "git_status", "git_run-command",
]
# Tools the legacy prefix list does NOT match. NOTE: that list is
# deliberately coarse (`filesystem__` matches filesystem__write too) - exact
# Rust parity; only the descriptor set (1d29f3b) narrows writers out.
NON_READ_TOOLS = [
    "docker__compose", "subtasks__manage_subtasks", "prompt__generate",
]


class Session:
    """Line-delimited JSON-RPC session against `python3 server.py`."""

    def __init__(self, initialize=True, **env_overrides):
        env = dict(os.environ)
        env.setdefault("OMNI_DIR", str(PLUGIN_DIR))
        for key, value in env_overrides.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        self.proc = subprocess.Popen(
            [sys.executable, "server.py"],
            cwd=str(PLUGIN_DIR),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._id = 0
        if initialize:
            self.rpc("initialize", {"protocolVersion": server.MCP_PROTOCOL_VERSION})

    def rpc(self, method, params=None):
        self._id += 1
        req_id = self._id
        msg = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}}
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise AssertionError(
                f"server closed stdout while handling {method} (id={req_id})")
        return json.loads(line)

    def call_tool(self, name, arguments):
        resp = self.rpc("tools/call", {"name": name, "arguments": arguments})
        result = resp["result"]
        text = result["content"][0]["text"]
        return result.get("isError", False), text

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def load_tool_list():
    session = Session()
    try:
        return session.rpc("tools/list")["result"]["tools"]
    finally:
        session.close()


def compact_corpus(read_tool="filesystem__read", non_read_tool="docker__compose"):
    """Deterministic over-budget conversation with two tool-call turns."""
    messages = [{"role": "system", "content": "S" * 200}]
    for i in range(40):
        messages.append({"role": "user", "content": f"question {i} " + "u" * 200})
        messages.append({"role": "assistant", "content": f"answer {i} " + "a" * 200})
    # Read-type tool turn (generous excerpt) and a non-read turn (generic cap).
    messages.append({
        "role": "assistant", "content": "",
        "tool_calls": [{"id": "c1", "type": "function",
                        "function": {"name": read_tool, "arguments": "{}"}}],
    })
    messages.append({"role": "tool", "name": read_tool, "tool_call_id": "c1",
                     "content": "R" * 6000})
    messages.append({
        "role": "assistant", "content": "",
        "tool_calls": [{"id": "c2", "type": "function",
                        "function": {"name": non_read_tool, "arguments": "{}"}}],
    })
    messages.append({"role": "tool", "name": non_read_tool, "tool_call_id": "c2",
                     "content": "D" * 6000})
    messages.append({"role": "user", "content": "latest question"})
    messages.append({"role": "assistant", "content": "latest answer"})
    return messages


class ToolRegistryParity(unittest.TestCase):
    """Gate 2: same tools, same names, same input schemas."""

    @classmethod
    def setUpClass(cls):
        cls.tools = load_tool_list()

    def test_tool_names_and_order(self):
        self.assertEqual([t["name"] for t in self.tools], EXPECTED_TOOL_NAMES)

    def test_schemas_match_golden(self):
        """Golden snapshot of the registry (regenerate with PROMPT_PARITY_UPDATE=1)."""
        if os.environ.get("PROMPT_PARITY_UPDATE") == "1":
            FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
            TOOLS_LIST_FIXTURE.write_text(
                json.dumps(self.tools, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8")
            self.skipTest("fixture regenerated")
        self.assertTrue(TOOLS_LIST_FIXTURE.is_file(),
                        f"missing golden fixture {TOOLS_LIST_FIXTURE}")
        golden = json.loads(TOOLS_LIST_FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(self.tools, golden)

    def test_uninitialized_session_is_rejected(self):
        session = Session(initialize=False)
        try:
            resp = session.rpc("tools/list")
            self.assertIn("error", resp)
        finally:
            session.close()

    def test_unknown_tool_is_rejected(self):
        session = Session()
        try:
            resp = session.rpc("tools/call", {"name": "prompt_nope", "arguments": {}})
            self.assertIn("error", resp)
            self.assertIn("Unknown tool", resp["error"]["message"])
        finally:
            session.close()


class ManifestParity(unittest.TestCase):
    """Gate 1: same plugin.json config schema (keys/types/defaults/labels)."""

    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(
            (PLUGIN_DIR / "plugin.json").read_text(encoding="utf-8"))["config_schema"]
        source = RUST_PLUGIN_JSON if RUST_PLUGIN_JSON.is_file() else RUST_SCHEMA_FIXTURE
        cls.reference_source = str(source)
        cls.rust = json.loads(source.read_text(encoding="utf-8"))["config_schema"]

    def test_entry_count(self):
        self.assertEqual(len(self.schema), 16)
        self.assertEqual(len(self.rust), 16)

    def test_schema_matches_builtin(self):
        rust = self.rust
        self.assertEqual(len(rust), len(self.schema))
        label_diffs = {}
        for mine, theirs in zip(self.schema, rust):
            self.assertEqual(mine["key"], theirs["key"])
            self.assertEqual(mine["type"], theirs["type"])
            self.assertEqual(mine["default"], theirs["default"])
            self.assertEqual(mine.get("description"), theirs.get("description"))
            if mine.get("label") != theirs.get("label"):
                label_diffs[mine["key"]] = (theirs["label"], mine["label"])
        self.assertEqual(label_diffs, ALLOWED_LABEL_DIFFS,
                         "labels must equal the builtin plugin.json exactly")


class GuidanceGrammarParity(unittest.TestCase):
    """c78e874: the guidance text must use the `{plugin}__{tool}` grammar."""

    def test_current_grammar_names_present(self):
        missing = [n for n in GRAMMAR_CURRENT if n not in server.TOOL_GUIDANCE]
        self.assertEqual(missing, [], "guidance lost current-grammar tool names")

    def test_legacy_names_absent(self):
        for name in GRAMMAR_LEGACY_IN_GUIDANCE:
            self.assertNotIn(name, server.TOOL_GUIDANCE,
                             f"legacy tool name {name!r} still in TOOL_GUIDANCE")


class GiveUpLoudlyParity(unittest.TestCase):
    """2026-09-18 (task_omnidev_honesty_rule_in_the_system_prompt): a give-up
    MUST call core__fail_thread - a plain final summary must never wrap an
    incomplete task (it looks like success and leaves the thread `completed`).
    Guards the Python copy against silently drifting away from the Rust one."""

    def test_identity_mandates_fail_thread(self):
        identity = server.build_dynamic_identity([])
        self.assertIn(
            "HONESTY RULE: never claim a success you did not verify", identity,
            "identity lost the new HONESTY RULE opening")
        self.assertIn(
            "core__fail_thread", identity,
            "identity must name the core__fail_thread tool a give-up has to call")
        self.assertIn(
            "Never write a summary message AFTER the fail call", identity,
            "identity must forbid a summary message after the fail call")

    def test_old_give_up_wording_is_gone(self):
        identity = server.build_dynamic_identity([])
        self.assertNotIn(
            "your final summary MUST clearly state", identity,
            "the old give-up wording (plain final summary) is still present")

    def test_guidance_carries_rule_15_give_up_loudly(self):
        self.assertIn("15. GIVE UP LOUDLY", server.TOOL_GUIDANCE,
                      "TOOL_GUIDANCE lost rule 15 (give up loudly)")
        self.assertIn("core__fail_thread", server.TOOL_GUIDANCE,
                      "rule 15 must name core__fail_thread")


class ReadToolDetectionParity(unittest.TestCase):
    """c78e874 (prefix list) + 1d29f3b (descriptor-driven, fail-closed)."""

    def test_legacy_prefix_list_covers_both_grammars(self):
        for name in READ_TOOLS_LEGACY:
            self.assertTrue(server.legacy_read_type_tool(name),
                            f"{name} should be a read-type tool")
        for name in NON_READ_TOOLS:
            self.assertFalse(server.legacy_read_type_tool(name),
                             f"{name} should not be a read-type tool")

    def test_empty_descriptors_fall_back_to_prefixes(self):
        for settings in ({}, {"read_only_tools": None}, {"read_only_tools": []}):
            self.assertTrue(server.is_read_type_tool("filesystem__read", settings))
            self.assertFalse(server.is_read_type_tool("docker__compose", settings))

    def test_descriptors_take_precedence(self):
        settings = {"read_only_tools": ["my_plugin__read_thing"]}
        self.assertTrue(server.is_read_type_tool("my_plugin__read_thing", settings))
        # Declared set is authoritative: a prefix match outside it is NOT read.
        self.assertFalse(server.is_read_type_tool("filesystem__read", settings))

    def test_read_only_tool_names_parsing(self):
        self.assertEqual(server.read_only_tool_names({"read_only_tools": ["a", "b"]}),
                         ["a", "b"])
        self.assertEqual(server.read_only_tool_names({}), [])
        self.assertEqual(server.read_only_tool_names({"read_only_tools": "a"}), [])
        self.assertEqual(server.read_only_tool_names({"read_only_tools": [1, "a"]}),
                         ["a"])


class MeasureSizeParity(unittest.TestCase):
    """Rust measure_size: chars/4 fallback, BPE when a tokenizer is set."""

    def test_empty_encoding_uses_chars_over_four(self):
        messages = [{"role": "user", "content": "x" * 40}]
        self.assertEqual(server.measure_size(messages, ""), 10)

    def test_invalid_encoding_falls_back(self):
        messages = [{"role": "user", "content": "x" * 40}]

        class Boom:
            @staticmethod
            def encoding_for_model(_):
                raise ValueError("unknown model")

            @staticmethod
            def get_encoding(_):
                raise ValueError("unknown encoding")

        original = server.tiktoken
        server.tiktoken = Boom
        try:
            self.assertEqual(server.measure_size(messages, "not-a-real-encoding"), 10)
        finally:
            server.tiktoken = original

    def test_valid_encoding_counts_serialized_messages(self):
        messages = [{"role": "user", "content": "abcd"}]

        class Half:
            @staticmethod
            def encode(text, disallowed_special=()):
                return list(range(len(text) // 2))

        class FakeTiktoken:
            @staticmethod
            def encoding_for_model(_):
                return Half()

        original = server.tiktoken
        server.tiktoken = FakeTiktoken
        try:
            expected = len(server.serialize_messages_rust(messages)) // 2
            self.assertEqual(server.measure_size(messages, "gpt-4"), expected)
        finally:
            server.tiktoken = original


class CompactMessagesE2E(unittest.TestCase):
    """Compaction over stdio: budgets, excerpts, descriptor override, errors."""

    def setUp(self):
        self.session = Session()

    def tearDown(self):
        self.session.close()

    def test_over_budget_compaction_uses_budget_and_excerpts(self):
        # hard=1000 forces progressive draining (keep_recent shrinks), so the
        # read result is excerpted to read_excerpt_chars and the generic one to
        # tool_excerpt_chars. The deterministic budget fallback may then shrink
        # the read result FURTHER so the prompt still fits the hard budget (see
        # ProviderBillingAlignment) - the caps below are upper bounds.
        messages = compact_corpus()
        is_error, text = self.session.call_tool("prompt_compact-messages", {
            "messages": messages,
            "hard_budget": 1000,
            "soft_budget": 500,
            "keep_recent": 2,
        })
        self.assertFalse(is_error, text)
        envelope = json.loads(text)
        # Rust main.rs L1845-1862: the tool ALWAYS returns the envelope; the
        # messages array is present only when something was compacted.
        self.assertTrue(envelope["was_compacted"], text)
        self.assertIsInstance(envelope["messages"], list)
        self.assertLess(len(envelope["messages"]), len(messages))
        joined = json.dumps(envelope, ensure_ascii=False)
        self.assertIn(server.COMPACTION_SUMMARY_MARKER, joined)
        read_run = max((len(m.group(0)) for m in re.finditer(r"R+", joined)), default=0)
        generic_run = max((len(m.group(0)) for m in re.finditer(r"D+", joined)), default=0)
        # The excerpt caps are UPPER bounds: hard=1000 is far below this
        # corpus's irreducible floor (40 q/a pairs), so the deterministic
        # fallback shrinks the two tool results further - never ABOVE the caps.
        self.assertLessEqual(read_run, 2000,
                             "read-type tool keeps read_excerpt_chars at most")
        self.assertLessEqual(generic_run, 800,
                             "other tools keep tool_excerpt_chars at most")
        # Honest contract for a target the fixed corpus floor cannot reach: the
        # envelope's over_budget flag MUST agree with its own measurement against
        # the HARD fit target (truncate_target), so the core suppresses the
        # overshoot error instead of re-logging it. The strong "fits under the
        # target in one cycle" assertion lives in ProviderBillingAlignment, where
        # the reduction IS achievable.
        self.assertEqual(envelope["over_budget"],
                         envelope["measured_tokens"] > envelope["truncate_target"],
                         text)

    def test_descriptor_supplied_read_tool_gets_generous_excerpt(self):
        # Same budgets as the excerpt test above.
        messages = compact_corpus(read_tool="my_plugin__read_file")
        is_error, text = self.session.call_tool("prompt_compact-messages", {
            "messages": messages,
            "hard_budget": 1000,
            "soft_budget": 500,
            "keep_recent": 2,
            "read_only_tools": ["my_plugin__read_file"],
        })
        self.assertFalse(is_error, text)
        joined = json.dumps(json.loads(text), ensure_ascii=False)
        run = max((len(m.group(0)) for m in re.finditer(r"R+", joined)), default=0)
        self.assertLessEqual(
            run, 2000,
            "descriptor-declared read tool keeps the generous excerpt")
        envelope = json.loads(text)
        # Same honest contract as the excerpt test above: with hard=1000 this
        # corpus floor cannot fit, but the flag must agree with the measure the
        # envelope reports against the HARD fit target (the core keys the
        # overshoot error on it).
        self.assertEqual(envelope["over_budget"],
                         envelope["measured_tokens"] > envelope["truncate_target"],
                         text)

    def test_force_compact_bypasses_gate(self):
        """Rust main.rs L1785-1800: the core re-invokes with force_compact=true
        when provider usage shows the context is still over the hard budget."""
        messages = compact_corpus()
        # Negative control (Rust main.rs L2875-2885): budgets above the local
        # estimate, NO force -> null-contract, no compaction.
        is_error, text = self.session.call_tool("prompt_compact-messages", {
            "messages": messages,
            "hard_budget": 10_000_000,
            "soft_budget": 5_000_000,
            "keep_recent": 0,
        })
        self.assertFalse(is_error, text)
        control = json.loads(text)
        self.assertFalse(control["was_compacted"], text)
        self.assertIsNone(control["messages"])
        # Same corpus WITH force_compact=true must compact (engine override).
        is_error, text = self.session.call_tool("prompt_compact-messages", {
            "messages": messages,
            "hard_budget": 10_000_000,
            "soft_budget": 5_000_000,
            "keep_recent": 0,
            "force_compact": True,
        })
        self.assertFalse(is_error, text)
        envelope = json.loads(text)
        self.assertTrue(envelope["was_compacted"], text)
        self.assertIsInstance(envelope["messages"], list)

    def test_within_budget_null_contract(self):
        """Rust main.rs L1845-1862: always the envelope; `messages` is null and
        `was_compacted` false when nothing was compacted."""
        is_error, text = self.session.call_tool("prompt_compact-messages", {
            "messages": [{"role": "user", "content": "hi"}],
            "hard_budget": 100000,
            "soft_budget": 50000,
        })
        self.assertFalse(is_error, text)
        envelope = json.loads(text)
        self.assertIsNone(envelope["messages"])
        self.assertFalse(envelope["was_compacted"])
        self.assertEqual(envelope["before_count"], envelope["after_count"])
        self.assertEqual(
            sorted(envelope), ["after_count", "before_count", "dump_file",
                               "effective_target", "entries", "iteration",
                               "measured_tokens", "messages",
                               "observed_overshoot_tokens", "over_budget",
                               "truncate_target", "truncated_chars",
                               "was_compacted"])

    def test_soft_budget_never_triggers_and_truncation_targets_hard_budget(self):
        """Hard design rule (operator, 2026-09-24): compaction is triggered ONLY
        by the HARD budget; the soft budget is the REDUCTION TARGET.

        A prompt whose size sits between soft and hard must be left completely
        UNTOUCHED (null-contract: no draining, no truncation, no dump / no
        compaction event), and the deterministic truncation fallback must never
        drag content down to the SOFT budget while the array still fits the HARD
        budget (the v0.3.2 "dumb and slow" regression, thread 2812).
        """
        # ~250k proxy tokens (chars/4): strictly between soft=100k and hard=400k.
        messages = [{"role": "system", "content": "SYSTEM PROMPT"}]
        for _ in range(5):
            messages.append({
                "role": "assistant", "content": "",
                "tool_calls": [{"id": "c1", "type": "function",
                                "function": {"name": "filesystem__read",
                                             "arguments": "{}"}}],
            })
            messages.append({"role": "tool", "name": "filesystem__read",
                             "tool_call_id": "c1", "content": "X" * 200000})
        messages.append({"role": "user", "content": "CURRENT USER TURN"})
        measured = server.measure_size(messages, "")
        self.assertGreater(measured, 100_000, measured)
        self.assertLess(measured, 400_000, measured)

        with tempfile.TemporaryDirectory() as thread_dir:
            is_error, text = self.session.call_tool("prompt_compact-messages", {
                "messages": messages,
                "hard_budget": 400_000,
                "soft_budget": 100_000,
                "keep_recent": 3,
                "thread_dir": thread_dir,
                "current_iteration": 7,
            })
            self.assertFalse(is_error, text)
            envelope = json.loads(text)
            self.assertFalse(envelope["was_compacted"], text)
            self.assertIsNone(envelope["messages"], text)
            self.assertEqual(envelope["truncated_chars"], 0, text)
            self.assertEqual(envelope["entries"], 0, text)
            self.assertIsNone(envelope["dump_file"], text)
            # No compaction event at all: the durable thread dir stays empty.
            self.assertEqual(os.listdir(thread_dir), [], text)
            # Soft is reported as the reduction TARGET, hard as the fit target.
            self.assertEqual(envelope["effective_target"], 100_000, text)
            self.assertEqual(envelope["truncate_target"], 400_000, text)

        # Same shape over the HARD budget: compaction fires, and the deterministic
        # fallback reduces only to the HARD fit target - never down to the soft
        # budget. One single tool turn is not drainable, so the fallback does it.
        over_corpus = [
            {"role": "system", "content": "SYSTEM PROMPT"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "filesystem__read",
                                          "arguments": "{}"}}]},
            {"role": "tool", "name": "filesystem__read",
             "tool_call_id": "c1", "content": "X" * 2_000_000},
            {"role": "user", "content": "CURRENT USER TURN"},
        ]
        over_size = server.measure_size(over_corpus, "")
        self.assertGreater(over_size, 200_000, over_size)
        is_error, text = self.session.call_tool("prompt_compact-messages", {
            "messages": over_corpus,
            "hard_budget": 200_000,
            "soft_budget": 100_000,
            "keep_recent": 3,
        })
        self.assertFalse(is_error, text)
        envelope = json.loads(text)
        self.assertTrue(envelope["was_compacted"], text)
        self.assertEqual(envelope["truncate_target"], 200_000, text)
        self.assertGreater(envelope["truncated_chars"], 0, text)
        self.assertLessEqual(envelope["measured_tokens"], 200_000, text)
        # NOT truncated down to the soft budget: the fallback only has to fit the
        # HARD budget, the soft budget stays the drain target.
        self.assertGreater(envelope["measured_tokens"], 100_000, text)

    def test_legacy_prefix_list_is_coarse_like_rust(self):
        # `filesystem__` matches writers too; only the declared descriptor set
        # narrows them out - exact Rust legacy_read_type_tool semantics.
        self.assertTrue(server.legacy_read_type_tool("filesystem__write"))
        self.assertFalse(server.is_read_type_tool(
            "filesystem__write", {"read_only_tools": ["filesystem__read"]}))

    def test_missing_arguments_error_messages(self):
        for arguments, needle in (
            ({}, "Missing required argument: 'messages'"),
            ({"messages": []}, "Missing required argument: 'hard_budget'"),
            ({"messages": [], "hard_budget": 10}, "Missing required argument: 'soft_budget'"),
        ):
            with self.subTest(needle=needle):
                is_error, text = self.session.call_tool(
                    "prompt_compact-messages", arguments)
                self.assertTrue(is_error, text)
                self.assertIn(needle, text)


class ProviderBillingAlignment(unittest.TestCase):
    """Chronic overshoot fix (279 ERROR-level compactions/24h, 2026-09-20).

    The plugin measures MESSAGES ONLY; the provider also bills the tool schemas
    it sends alongside and its own chat template. The core passes back the pair
    (billed_prompt_tokens, measured_tokens) from the SAME request, so the plugin
    can subtract that invisible overhead and reduce until the PROVIDER fits
    under the hard budget - instead of stopping at the soft budget while the
    provider keeps billing over the hard budget and the agent force-compacts
    (and re-logs the overshoot error) forever.
    """

    def setUp(self):
        self.session = Session()

    def tearDown(self):
        self.session.close()

    def test_billed_overhead_drives_target_and_clears_over_budget(self):
        messages = [{"role": "system", "content": "SYSTEM PROMPT MUST SURVIVE"}]
        for _ in range(5):
            messages.append({
                "role": "assistant", "content": "",
                "tool_calls": [{"id": "c1", "type": "function",
                                "function": {"name": "filesystem__read",
                                             "arguments": "{}"}}],
            })
            messages.append({"role": "tool", "name": "filesystem__read",
                             "tool_call_id": "c1", "content": "X" * 200000})
        messages.append({"role": "user", "content": "CURRENT USER TURN MUST SURVIVE"})
        messages.append({"role": "assistant", "content": "working"})

        measured = server.measure_size(messages, "")
        hard, soft = 20000, 10000
        billed = measured + 5000
        self.assertGreater(measured, hard)

        is_error, text = self.session.call_tool("prompt_compact-messages", {
            "messages": messages,
            "keep_recent": 3,
            "soft_budget": soft,
            "hard_budget": hard,
            "force_compact": True,
            "billed_prompt_tokens": billed,
            "measured_tokens": measured,
        })
        self.assertFalse(is_error, text)
        envelope = json.loads(text)

        # The reduction target accounts for the provider overhead + headroom and
        # never exceeds the hard budget.
        self.assertLessEqual(envelope["effective_target"], hard - 5000 - 2000)
        # The provider fits from now on: the core stops force-compacting.
        self.assertFalse(envelope["over_budget"], text)
        # The reduction (drain + deterministic fallback) only has to fit the HARD
        # budget: `truncate_target` is hard_budget - overhead - headroom, while
        # `effective_target` stays the soft-budget DRAIN aim.
        self.assertLessEqual(envelope["measured_tokens"], envelope["truncate_target"])
        self.assertEqual(envelope["truncate_target"], hard - 5000 - 2000, text)
        self.assertEqual(envelope["effective_target"], soft, text)
        # The deterministic fallback (not just the summary drain) did the work.
        self.assertGreater(envelope["truncated_chars"], 0)

        out = envelope["messages"]
        self.assertIsInstance(out, list, text)
        # The system prompt and the CURRENT user turn survive verbatim.
        self.assertEqual(out[0]["content"], "SYSTEM PROMPT MUST SURVIVE")
        self.assertEqual(out[-2]["content"], "CURRENT USER TURN MUST SURVIVE")
        # The tool-call STRUCTURE is preserved; only content was truncated.
        with_calls = [m for m in out if m.get("tool_calls")]
        self.assertTrue(with_calls, text)
        self.assertEqual(with_calls[0]["tool_calls"][0]["function"]["name"],
                         "filesystem__read")
        self.assertTrue(all(m.get("tool_call_id")
                            for m in out if m.get("role") == "tool"), text)


    def test_observed_overshoot_feedback_tightens_the_gate(self):
        """Thread 3998 (thread 3978's 201637-token call): the core feeds the
        observed provider overshoot back, so a prompt this plugin measured as
        fitting IS reduced once the previous request proved the estimate was
        short. Without the feedback the gate stays open and the provider keeps
        billing over the hard budget - the state that produced the chronic
        'condensation did not reduce it' error."""
        messages = [{"role": "system", "content": "SYSTEM PROMPT MUST SURVIVE"}]
        for _ in range(5):
            messages.append({
                "role": "assistant", "content": "",
                "tool_calls": [{"id": "c1", "type": "function",
                                "function": {"name": "filesystem__read",
                                             "arguments": "{}"}}],
            })
            messages.append({"role": "tool", "name": "filesystem__read",
                             "tool_call_id": "c1", "content": "X" * 200000})
        messages.append({"role": "user", "content": "CURRENT USER TURN MUST SURVIVE"})
        messages.append({"role": "assistant", "content": "working"})

        measured = server.measure_size(messages, "")
        overhead, headroom, overshoot = 5000, 2000, 2000
        hard = measured + overhead + headroom + 1000
        soft = 50000
        self.assertLess(measured, hard)

        def call(observed):
            is_error, text = self.session.call_tool("prompt_compact-messages", {
                "messages": messages,
                "keep_recent": 3,
                "soft_budget": soft,
                "hard_budget": hard,
                "billed_prompt_tokens": hard + overshoot,
                "measured_tokens": hard + overshoot - overhead,
                "observed_overshoot_tokens": observed,
            })
            self.assertFalse(is_error, text)
            return json.loads(text)

        # CONTROL: no feedback -> the gate stays open even though the provider
        # really billed over the hard budget.
        control = call(0)
        self.assertFalse(control["was_compacted"], control)
        self.assertFalse(control["over_budget"], control)
        self.assertEqual(control["truncate_target"], measured + 1000, control)

        # WITH the observed overshoot the target is tightened by exactly that
        # much: the gate fires and the prompt is reduced under the target.
        envelope = call(overshoot)
        self.assertEqual(envelope["observed_overshoot_tokens"], overshoot, envelope)
        self.assertTrue(envelope["was_compacted"], envelope)
        self.assertFalse(envelope["over_budget"], envelope)
        self.assertEqual(envelope["truncate_target"], measured - 1000, envelope)
        self.assertLessEqual(envelope["measured_tokens"],
                             envelope["truncate_target"])
        self.assertLess(envelope["after_count"], envelope["before_count"])

if __name__ == "__main__":
    unittest.main(verbosity=2)