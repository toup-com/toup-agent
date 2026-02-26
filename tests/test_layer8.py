"""
Layer 8 Tests — Chat Commands, Session Tools, Process Manager, FS Tools,
Access Control, Model Per-Session, Idempotency, Usage Tracking
"""

import asyncio
import os
import tempfile
import time
import unittest


# ── Chat Commands ──

class TestCommandRegistry(unittest.TestCase):
    def test_import(self):
        from app.agent.chat_commands import CommandRegistry, CommandDef, CommandScope
        self.assertTrue(CommandRegistry)

    def test_builtins_registered(self):
        from app.agent.chat_commands import CommandRegistry
        reg = CommandRegistry()
        cmds = reg.list_commands()
        names = [c.name for c in cmds]
        for expected in ["status", "new", "reset", "compact", "model", "usage", "think", "help"]:
            self.assertIn(expected, names)

    def test_get_by_name(self):
        from app.agent.chat_commands import CommandRegistry
        reg = CommandRegistry()
        cmd = reg.get("status")
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.name, "status")

    def test_get_by_alias(self):
        from app.agent.chat_commands import CommandRegistry
        reg = CommandRegistry()
        cmd = reg.get("m")
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.name, "model")

    def test_parse_command(self):
        from app.agent.chat_commands import CommandRegistry
        reg = CommandRegistry()
        result = reg.parse("/status")
        self.assertEqual(result, ("status", ""))

    def test_parse_command_with_args(self):
        from app.agent.chat_commands import CommandRegistry
        reg = CommandRegistry()
        result = reg.parse("/model gpt-4o")
        self.assertEqual(result, ("model", "gpt-4o"))

    def test_parse_not_command(self):
        from app.agent.chat_commands import CommandRegistry
        reg = CommandRegistry()
        result = reg.parse("hello world")
        self.assertIsNone(result)

    def test_parse_bot_mention(self):
        from app.agent.chat_commands import CommandRegistry
        reg = CommandRegistry()
        result = reg.parse("/help@mybot")
        self.assertEqual(result[0], "help")

    def test_register_custom(self):
        from app.agent.chat_commands import CommandRegistry, CommandDef
        reg = CommandRegistry()
        reg.register(CommandDef("mytest", "Test command"))
        cmd = reg.get("mytest")
        self.assertIsNotNone(cmd)

    def test_unregister(self):
        from app.agent.chat_commands import CommandRegistry, CommandDef
        reg = CommandRegistry()
        reg.register(CommandDef("temp", "Temp"))
        self.assertTrue(reg.unregister("temp"))
        self.assertIsNone(reg.get("temp"))

    def test_command_to_dict(self):
        from app.agent.chat_commands import CommandDef, CommandScope
        cmd = CommandDef("test", "A test", scope=CommandScope.DM, aliases=["t"])
        d = cmd.to_dict()
        self.assertEqual(d["name"], "test")
        self.assertEqual(d["scope"], "dm")
        self.assertEqual(d["aliases"], ["t"])

    def test_singleton(self):
        from app.agent.chat_commands import get_command_registry
        r1 = get_command_registry()
        r2 = get_command_registry()
        self.assertIs(r1, r2)


class TestCommandExecution(unittest.TestCase):
    def test_execute_help(self):
        from app.agent.chat_commands import CommandRegistry
        reg = CommandRegistry()
        result = asyncio.get_event_loop().run_until_complete(reg.execute("/help"))
        self.assertIn("text", result)
        self.assertIn("/status", result["text"])

    def test_execute_new(self):
        from app.agent.chat_commands import CommandRegistry
        reg = CommandRegistry()
        result = asyncio.get_event_loop().run_until_complete(reg.execute("/new"))
        self.assertEqual(result["action"], "new_session")

    def test_execute_think(self):
        from app.agent.chat_commands import CommandRegistry
        reg = CommandRegistry()
        result = asyncio.get_event_loop().run_until_complete(reg.execute("/think high"))
        self.assertEqual(result["action"], "set_thinking")
        self.assertEqual(result["level"], "high")
        self.assertEqual(result["budget"], 10000)

    def test_execute_think_invalid(self):
        from app.agent.chat_commands import CommandRegistry
        reg = CommandRegistry()
        result = asyncio.get_event_loop().run_until_complete(reg.execute("/think ultra"))
        self.assertIn("Invalid", result["text"])

    def test_execute_model_no_args(self):
        from app.agent.chat_commands import CommandRegistry
        reg = CommandRegistry()
        result = asyncio.get_event_loop().run_until_complete(reg.execute("/model"))
        self.assertIn("Current model", result["text"])

    def test_execute_model_switch(self):
        from app.agent.chat_commands import CommandRegistry
        reg = CommandRegistry()
        result = asyncio.get_event_loop().run_until_complete(reg.execute("/model gpt-4o"))
        self.assertEqual(result["action"], "set_model")
        self.assertEqual(result["model"], "gpt-4o")

    def test_execute_config_get(self):
        from app.agent.chat_commands import CommandRegistry
        reg = CommandRegistry()
        result = asyncio.get_event_loop().run_until_complete(
            reg.execute("/config get mykey", context={"mykey": "myval"})
        )
        self.assertIn("myval", result["text"])

    def test_execute_config_set(self):
        from app.agent.chat_commands import CommandRegistry
        reg = CommandRegistry()
        result = asyncio.get_event_loop().run_until_complete(
            reg.execute("/config set mykey newval")
        )
        self.assertEqual(result["action"], "set_config")


# ── Session Tools ──

class TestSessionManager(unittest.TestCase):
    def test_import(self):
        from app.agent.session_tools import SessionManager, SessionKind, SessionStatus, AgentInfo
        self.assertTrue(SessionManager)

    def test_create_session(self):
        from app.agent.session_tools import SessionManager, SessionKind
        mgr = SessionManager()
        session = mgr.create_session(kind=SessionKind.MAIN, model="gpt-4o", channel="telegram")
        self.assertIsNotNone(session.session_id)
        self.assertEqual(session.kind, SessionKind.MAIN)
        self.assertEqual(session.model, "gpt-4o")

    def test_end_session(self):
        from app.agent.session_tools import SessionManager
        mgr = SessionManager()
        session = mgr.create_session()
        self.assertTrue(mgr.end_session(session.session_id))
        self.assertFalse(session.is_active)

    def test_list_sessions(self):
        from app.agent.session_tools import SessionManager, SessionKind
        mgr = SessionManager()
        mgr.create_session(kind=SessionKind.MAIN)
        mgr.create_session(kind=SessionKind.SUBAGENT)
        all_sessions = mgr.list_sessions()
        self.assertEqual(len(all_sessions), 2)
        main_only = mgr.list_sessions(kind=SessionKind.MAIN)
        self.assertEqual(len(main_only), 1)

    def test_session_status(self):
        from app.agent.session_tools import SessionManager
        mgr = SessionManager()
        session = mgr.create_session()
        status = mgr.session_status(session.session_id)
        self.assertIsNotNone(status)
        self.assertIn("uptime", status)

    def test_sessions_send(self):
        from app.agent.session_tools import SessionManager
        mgr = SessionManager()
        s1 = mgr.create_session()
        s2 = mgr.create_session()
        result = asyncio.get_event_loop().run_until_complete(
            mgr.sessions_send(s1.session_id, s2.session_id, "hello from s1")
        )
        self.assertTrue(result["sent"])

    def test_sessions_send_to_inactive(self):
        from app.agent.session_tools import SessionManager
        mgr = SessionManager()
        s1 = mgr.create_session()
        s2 = mgr.create_session()
        mgr.end_session(s2.session_id)
        result = asyncio.get_event_loop().run_until_complete(
            mgr.sessions_send(s1.session_id, s2.session_id, "hello")
        )
        self.assertIn("error", result)

    def test_agents_list(self):
        from app.agent.session_tools import SessionManager
        mgr = SessionManager()
        agents = mgr.agents_list()
        self.assertEqual(len(agents), 1)
        self.assertEqual(agents[0].agent_id, "default")

    def test_register_agent(self):
        from app.agent.session_tools import SessionManager, AgentInfo
        mgr = SessionManager()
        mgr.register_agent(AgentInfo(agent_id="code", name="Coder", model="gpt-4o"))
        agents = mgr.agents_list()
        self.assertEqual(len(agents), 2)

    def test_track_usage(self):
        from app.agent.session_tools import SessionManager
        mgr = SessionManager()
        session = mgr.create_session()
        mgr.track_usage(session.session_id, tokens_in=100, tokens_out=50, cost=0.01)
        self.assertEqual(session.tokens_in, 100)
        self.assertEqual(session.tokens_out, 50)

    def test_message_log(self):
        from app.agent.session_tools import SessionManager
        mgr = SessionManager()
        s1 = mgr.create_session()
        s2 = mgr.create_session()
        asyncio.get_event_loop().run_until_complete(
            mgr.sessions_send(s1.session_id, s2.session_id, "msg1")
        )
        log = mgr.get_message_log(session_id=s1.session_id)
        self.assertEqual(len(log), 1)

    def test_singleton(self):
        from app.agent.session_tools import get_session_manager
        m1 = get_session_manager()
        m2 = get_session_manager()
        self.assertIs(m1, m2)


# ── Process Manager ──

class TestProcessManager(unittest.TestCase):
    def test_import(self):
        from app.agent.process_manager import ProcessManager, ProcessState, ManagedProcess
        self.assertTrue(ProcessManager)

    def test_process_state_enum(self):
        from app.agent.process_manager import ProcessState
        self.assertEqual(ProcessState.RUNNING.value, "running")
        self.assertEqual(ProcessState.COMPLETED.value, "completed")

    def test_managed_process_to_dict(self):
        from app.agent.process_manager import ManagedProcess, ProcessState
        proc = ManagedProcess(process_id="abc", command="echo hi", state=ProcessState.RUNNING)
        d = proc.to_dict()
        self.assertEqual(d["process_id"], "abc")
        self.assertEqual(d["state"], "running")

    def test_start_and_wait(self):
        from app.agent.process_manager import ProcessManager, ProcessState
        mgr = ProcessManager()
        proc = asyncio.get_event_loop().run_until_complete(
            mgr.start("echo hello_world")
        )
        self.assertIsNotNone(proc.pid)
        # Wait for completion
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.5))
        status = mgr.status(proc.process_id)
        self.assertIn(status["state"], ["completed", "running"])

    def test_list_processes(self):
        from app.agent.process_manager import ProcessManager
        mgr = ProcessManager()
        asyncio.get_event_loop().run_until_complete(mgr.start("echo one"))
        asyncio.get_event_loop().run_until_complete(mgr.start("echo two"))
        procs = mgr.list_processes()
        self.assertEqual(len(procs), 2)

    def test_output(self):
        from app.agent.process_manager import ProcessManager
        mgr = ProcessManager()
        proc = asyncio.get_event_loop().run_until_complete(mgr.start("echo test_output"))
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.5))
        output = mgr.output(proc.process_id, stream="stdout")
        self.assertIsNotNone(output)
        if output:
            self.assertTrue(any("test_output" in line for line in output))

    def test_stop_all(self):
        from app.agent.process_manager import ProcessManager
        mgr = ProcessManager()
        count = asyncio.get_event_loop().run_until_complete(mgr.stop_all())
        self.assertIsInstance(count, int)

    def test_cleanup(self):
        from app.agent.process_manager import ProcessManager
        mgr = ProcessManager()
        removed = mgr.cleanup()
        self.assertIsInstance(removed, int)

    def test_singleton(self):
        from app.agent.process_manager import get_process_manager
        m1 = get_process_manager()
        m2 = get_process_manager()
        self.assertIs(m1, m2)


# ── File System Tools ──

class TestFSTools(unittest.TestCase):
    def test_import(self):
        from app.agent.fs_tools import grep, find, ls, apply_patch, GrepMatch, FileInfo
        self.assertTrue(grep)

    def test_ls_tmpdir(self):
        from app.agent.fs_tools import ls
        results = ls("/tmp")
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) > 0)

    def test_ls_file_info(self):
        from app.agent.fs_tools import ls
        results = ls("/tmp")
        if results:
            fi = results[0]
            self.assertIn("name", fi.to_dict())
            self.assertIn("size", fi.to_dict())

    def test_find_py_files(self):
        from app.agent.fs_tools import find
        results = find(".", pattern="*.py", max_depth=2, max_results=10)
        self.assertIsInstance(results, list)

    def test_find_with_type(self):
        from app.agent.fs_tools import find
        dirs = find(".", file_type="d", max_depth=1, max_results=10)
        for d in dirs:
            self.assertTrue(os.path.isdir(d))

    def test_grep_in_file(self):
        from app.agent.fs_tools import grep
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("line one\nline two\ngrep target here\nline four\n")
            f.flush()
            results = grep("grep target", f.name)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].line_number, 3)
            os.unlink(f.name)

    def test_grep_case_insensitive(self):
        from app.agent.fs_tools import grep
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello World\nhello world\nHELLO WORLD\n")
            f.flush()
            results = grep("hello", f.name, ignore_case=True)
            self.assertEqual(len(results), 3)
            os.unlink(f.name)

    def test_grep_match_to_dict(self):
        from app.agent.fs_tools import GrepMatch
        m = GrepMatch(file="test.py", line_number=5, line="found it")
        d = m.to_dict()
        self.assertEqual(d["file"], "test.py")
        self.assertEqual(d["line_number"], 5)

    def test_apply_patch(self):
        from app.agent.fs_tools import apply_patch
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("line 1\nline 2\nline 3\n")
            f.flush()
            patch = "@@ -1,3 +1,3 @@\n line 1\n-line 2\n+line 2 modified\n line 3\n"
            result = apply_patch(f.name, patch)
            self.assertTrue(result["success"])
            with open(f.name) as rf:
                content = rf.read()
            self.assertIn("modified", content)
            os.unlink(f.name)

    def test_apply_patch_no_file(self):
        from app.agent.fs_tools import apply_patch
        result = apply_patch("/nonexistent/file.txt", "@@...")
        self.assertFalse(result["success"])


# ── Access Control ──

class TestAccessControl(unittest.TestCase):
    def test_import(self):
        from app.agent.access_control import AccessController, DMPolicy, GroupPolicy, ToolPolicy
        self.assertTrue(AccessController)

    def test_dm_policy_enum(self):
        from app.agent.access_control import DMPolicy
        self.assertEqual(DMPolicy.PAIRING.value, "pairing")
        self.assertEqual(DMPolicy.OPEN.value, "open")

    def test_open_dm(self):
        from app.agent.access_control import AccessController, DMPolicy
        ac = AccessController()
        ac.set_dm_policy("telegram", DMPolicy.OPEN)
        result = ac.check_dm_access("telegram", "user123")
        self.assertTrue(result["allowed"])

    def test_disabled_dm(self):
        from app.agent.access_control import AccessController, DMPolicy
        ac = AccessController()
        ac.set_dm_policy("telegram", DMPolicy.DISABLED)
        result = ac.check_dm_access("telegram", "user123")
        self.assertFalse(result["allowed"])

    def test_allowlist_dm(self):
        from app.agent.access_control import AccessController, DMPolicy
        ac = AccessController()
        ac.set_dm_policy("telegram", DMPolicy.ALLOWLIST)
        ac.add_to_allowlist("telegram", "user1")
        self.assertTrue(ac.check_dm_access("telegram", "user1")["allowed"])
        self.assertFalse(ac.check_dm_access("telegram", "user2")["allowed"])

    def test_pairing_flow(self):
        from app.agent.access_control import AccessController, DMPolicy
        ac = AccessController()
        ac.set_dm_policy("telegram", DMPolicy.PAIRING)
        # User not paired
        self.assertFalse(ac.check_dm_access("telegram", "newuser")["allowed"])
        # Generate code
        code = ac.generate_pairing_code("newuser", "telegram", username="John")
        self.assertEqual(len(code.code), 6)
        # Approve
        result = ac.approve_pairing(code.code)
        self.assertTrue(result["success"])
        # Now user is allowed
        self.assertTrue(ac.check_dm_access("telegram", "newuser")["allowed"])

    def test_pairing_expired(self):
        from app.agent.access_control import AccessController
        ac = AccessController()
        ac._pairing_ttl = 0  # Expire immediately
        code = ac.generate_pairing_code("user", "tg")
        import time; time.sleep(0.01)
        result = ac.approve_pairing(code.code)
        self.assertFalse(result["success"])

    def test_block_user(self):
        from app.agent.access_control import AccessController, DMPolicy
        ac = AccessController()
        ac.set_dm_policy("telegram", DMPolicy.OPEN)
        ac.block_user("telegram", "baduser")
        result = ac.check_dm_access("telegram", "baduser")
        self.assertFalse(result["allowed"])
        self.assertEqual(result["reason"], "blocked")

    def test_owner_bypass(self):
        from app.agent.access_control import AccessController, DMPolicy
        ac = AccessController()
        ac.set_dm_policy("telegram", DMPolicy.DISABLED)
        ac.set_owner("admin1")
        result = ac.check_dm_access("telegram", "admin1")
        self.assertTrue(result["allowed"])
        self.assertEqual(result["reason"], "owner")

    def test_group_policy(self):
        from app.agent.access_control import AccessController, GroupPolicy
        ac = AccessController()
        ac.set_group_policy("discord", GroupPolicy.ALLOWLIST)
        ac.add_to_allowlist("discord", "server1", is_group=True)
        self.assertTrue(ac.check_group_access("discord", "server1")["allowed"])
        self.assertFalse(ac.check_group_access("discord", "server2")["allowed"])

    def test_tool_policy(self):
        from app.agent.access_control import AccessController
        ac = AccessController()
        ac.set_tool_denied(["exec", "write_file"])
        ac.set_tool_elevated(["exec"])
        self.assertFalse(ac.check_tool_access("exec")["allowed"])
        self.assertTrue(ac.check_tool_access("read_file")["allowed"])

    def test_tool_elevation(self):
        from app.agent.access_control import ToolPolicy
        tp = ToolPolicy(elevated_tools={"exec", "delete_file"})
        self.assertTrue(tp.is_elevated("exec"))
        self.assertFalse(tp.is_elevated("read_file"))

    def test_pending_pairings(self):
        from app.agent.access_control import AccessController
        ac = AccessController()
        ac.generate_pairing_code("u1", "tg")
        ac.generate_pairing_code("u2", "tg")
        pending = ac.list_pending_pairings()
        self.assertEqual(len(pending), 2)

    def test_singleton(self):
        from app.agent.access_control import get_access_controller
        a1 = get_access_controller()
        a2 = get_access_controller()
        self.assertIs(a1, a2)


# ── Model Session Manager ──

class TestModelSessionManager(unittest.TestCase):
    def test_import(self):
        from app.agent.model_session import ModelSessionManager, AVAILABLE_MODELS
        self.assertTrue(ModelSessionManager)
        self.assertIn("claude-opus-4-6", AVAILABLE_MODELS)

    def test_get_config(self):
        from app.agent.model_session import ModelSessionManager
        mgr = ModelSessionManager()
        config = mgr.get_config("session1")
        self.assertEqual(config.model, "claude-opus-4-6")

    def test_set_model(self):
        from app.agent.model_session import ModelSessionManager
        mgr = ModelSessionManager()
        result = mgr.set_model("s1", "gpt-4o")
        self.assertTrue(result["success"])
        self.assertEqual(mgr.get_config("s1").model, "gpt-4o")

    def test_set_model_invalid(self):
        from app.agent.model_session import ModelSessionManager
        mgr = ModelSessionManager()
        result = mgr.set_model("s1", "nonexistent-model")
        self.assertFalse(result["success"])
        self.assertIn("available", result)

    def test_set_thinking(self):
        from app.agent.model_session import ModelSessionManager
        mgr = ModelSessionManager()
        result = mgr.set_thinking("s1", "high")
        self.assertTrue(result["success"])
        self.assertEqual(result["budget"], 10000)
        self.assertEqual(mgr.get_config("s1").thinking_level, "high")

    def test_track_usage(self):
        from app.agent.model_session import ModelSessionManager
        mgr = ModelSessionManager()
        record = mgr.track_usage("s1", "gpt-4o", tokens_in=1000, tokens_out=500)
        self.assertGreater(record.cost, 0)

    def test_usage_summary(self):
        from app.agent.model_session import ModelSessionManager
        mgr = ModelSessionManager()
        mgr.track_usage("s1", "gpt-4o", tokens_in=1000, tokens_out=500)
        mgr.track_usage("s1", "gpt-4o-mini", tokens_in=2000, tokens_out=1000)
        summary = mgr.get_usage_summary("s1")
        self.assertEqual(summary["total_tokens_in"], 3000)
        self.assertEqual(summary["total_requests"], 2)
        self.assertIn("gpt-4o", summary["by_model"])

    def test_usage_summary_all(self):
        from app.agent.model_session import ModelSessionManager
        mgr = ModelSessionManager()
        mgr.track_usage("s1", "gpt-4o", tokens_in=100, tokens_out=50)
        mgr.track_usage("s2", "gpt-4o", tokens_in=200, tokens_out=100)
        summary = mgr.get_usage_summary()
        self.assertEqual(summary["total_tokens_in"], 300)

    def test_idempotency(self):
        from app.agent.model_session import ModelSessionManager
        mgr = ModelSessionManager()
        key = mgr.generate_idempotency_key("s1", "do something")
        self.assertIsNone(mgr.check_idempotency(key))
        mgr.set_idempotency(key, {"result": "done"})
        cached = mgr.check_idempotency(key)
        self.assertEqual(cached["result"], "done")

    def test_idempotency_expiry(self):
        from app.agent.model_session import ModelSessionManager
        mgr = ModelSessionManager()
        mgr._idempotency_ttl = 0
        key = "test-key"
        mgr.set_idempotency(key, {"r": 1})
        import time; time.sleep(0.01)
        self.assertIsNone(mgr.check_idempotency(key))

    def test_cleanup_idempotency(self):
        from app.agent.model_session import ModelSessionManager
        mgr = ModelSessionManager()
        mgr._idempotency_ttl = 0
        mgr.set_idempotency("k1", {})
        mgr.set_idempotency("k2", {})
        import time; time.sleep(0.01)
        removed = mgr.cleanup_idempotency()
        self.assertEqual(removed, 2)

    def test_list_models(self):
        from app.agent.model_session import ModelSessionManager
        mgr = ModelSessionManager()
        models = mgr.list_models()
        self.assertTrue(len(models) >= 4)
        names = [m["model"] for m in models]
        self.assertIn("gpt-4o", names)

    def test_config_to_dict(self):
        from app.agent.model_session import SessionModelConfig
        cfg = SessionModelConfig(session_id="s1", model="gpt-4o", thinking_level="high", thinking_budget=10000)
        d = cfg.to_dict()
        self.assertEqual(d["model"], "gpt-4o")
        self.assertEqual(d["thinking_budget"], 10000)

    def test_singleton(self):
        from app.agent.model_session import get_model_session_manager
        m1 = get_model_session_manager()
        m2 = get_model_session_manager()
        self.assertIs(m1, m2)


if __name__ == "__main__":
    unittest.main()
