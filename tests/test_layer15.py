"""
Layer 15 Tests — Final sweep: 4 channels + multi-account + delivery modes +
Gmail Pub/Sub + agent policies + TUI chat + Tailscale + SSH tunnels.
"""

import asyncio
import base64
import json
import pytest


# ── Zalo Channel ──────────────────────────────────────

class TestZaloChannel:
    def test_import(self):
        from app.agent.channels.zalo_channel import ZaloChannel
        assert ZaloChannel is not None

    def test_connect_missing_creds(self):
        from app.agent.channels.zalo_channel import ZaloChannel
        ch = ZaloChannel()
        loop = asyncio.get_event_loop()
        assert loop.run_until_complete(ch.connect()) == False

    def test_connect_with_creds(self):
        from app.agent.channels.zalo_channel import ZaloChannel
        ch = ZaloChannel({"oa_id": "123", "access_token": "abc"})
        loop = asyncio.get_event_loop()
        assert loop.run_until_complete(ch.connect()) == True

    def test_send_message(self):
        from app.agent.channels.zalo_channel import ZaloChannel
        ch = ZaloChannel({"oa_id": "123", "access_token": "abc"})
        loop = asyncio.get_event_loop()
        loop.run_until_complete(ch.connect())
        msg_id = loop.run_until_complete(ch.send_message("user1", "hello"))
        assert msg_id is not None

    def test_webhook(self):
        from app.agent.channels.zalo_channel import ZaloChannel
        ch = ZaloChannel({"oa_id": "123", "access_token": "abc"})
        loop = asyncio.get_event_loop()
        msg = loop.run_until_complete(ch.handle_webhook({
            "event_name": "user_send_text",
            "sender": {"id": "u1"},
            "message": {"text": "xin chao"}
        }))
        assert msg is not None
        assert msg["text"] == "xin chao"


# ── Mattermost Channel ───────────────────────────────

class TestMattermostChannel:
    def test_import(self):
        from app.agent.channels.mattermost_channel import MattermostChannel
        assert MattermostChannel is not None

    def test_connect(self):
        from app.agent.channels.mattermost_channel import MattermostChannel
        ch = MattermostChannel({"server_url": "https://mm.example.com", "bot_token": "tok"})
        loop = asyncio.get_event_loop()
        assert loop.run_until_complete(ch.connect()) == True

    def test_webhook(self):
        from app.agent.channels.mattermost_channel import MattermostChannel
        ch = MattermostChannel()
        loop = asyncio.get_event_loop()
        msg = loop.run_until_complete(ch.handle_webhook({
            "event": "posted",
            "data": {"post": {"channel_id": "c1", "user_id": "u1", "message": "hello"}}
        }))
        assert msg is not None
        assert msg["text"] == "hello"

    def test_add_reaction(self):
        from app.agent.channels.mattermost_channel import MattermostChannel
        ch = MattermostChannel({"server_url": "https://mm.example.com", "bot_token": "tok"})
        loop = asyncio.get_event_loop()
        loop.run_until_complete(ch.connect())
        assert loop.run_until_complete(ch.add_reaction("p1", "thumbsup")) == True


# ── Feishu/Lark Channel ──────────────────────────────

class TestFeishuChannel:
    def test_import(self):
        from app.agent.channels.feishu_channel import FeishuChannel
        assert FeishuChannel is not None

    def test_connect(self):
        from app.agent.channels.feishu_channel import FeishuChannel
        ch = FeishuChannel({"app_id": "cli_xxx", "app_secret": "secret"})
        loop = asyncio.get_event_loop()
        assert loop.run_until_complete(ch.connect()) == True

    def test_webhook(self):
        from app.agent.channels.feishu_channel import FeishuChannel
        ch = FeishuChannel()
        loop = asyncio.get_event_loop()
        msg = loop.run_until_complete(ch.handle_webhook({
            "header": {"event_type": "im.message.receive_v1"},
            "event": {
                "message": {"chat_id": "oc_xxx", "content": "hello feishu"},
                "sender": {"sender_id": {"open_id": "ou_xxx"}}
            }
        }))
        assert msg is not None
        assert msg["text"] == "hello feishu"


# ── Nostr Channel ────────────────────────────────────

class TestNostrChannel:
    def test_import(self):
        from app.agent.channels.nostr_channel import NostrChannel
        assert NostrChannel is not None

    def test_connect(self):
        from app.agent.channels.nostr_channel import NostrChannel
        ch = NostrChannel({"private_key": "nsec1testkey"})
        loop = asyncio.get_event_loop()
        assert loop.run_until_complete(ch.connect()) == True

    def test_dm_webhook(self):
        from app.agent.channels.nostr_channel import NostrChannel
        ch = NostrChannel()
        loop = asyncio.get_event_loop()
        msg = loop.run_until_complete(ch.handle_webhook({
            "kind": 4,
            "pubkey": "npub1abc",
            "content": "encrypted dm"
        }))
        assert msg is not None
        assert msg["text"] == "encrypted dm"

    def test_note_webhook(self):
        from app.agent.channels.nostr_channel import NostrChannel
        ch = NostrChannel()
        loop = asyncio.get_event_loop()
        msg = loop.run_until_complete(ch.handle_webhook({
            "kind": 1,
            "pubkey": "npub1abc",
            "content": "public note"
        }))
        assert msg is not None
        assert msg["channel_id"] == "public"


# ── Multi-Account ────────────────────────────────────

class TestMultiAccount:
    def test_import(self):
        from app.agent.multi_account import MultiAccountManager
        assert MultiAccountManager is not None

    def test_add_account(self):
        from app.agent.multi_account import MultiAccountManager
        mgr = MultiAccountManager()
        acct = mgr.add_account("telegram", "bot_1", credentials={"token": "abc"})
        assert acct.account_id == "bot_1"
        assert acct.is_primary == True  # First account becomes primary

    def test_multiple_accounts(self):
        from app.agent.multi_account import MultiAccountManager
        mgr = MultiAccountManager()
        mgr.add_account("telegram", "bot_1")
        mgr.add_account("telegram", "bot_2")
        accounts = mgr.list_accounts("telegram")
        assert len(accounts) == 2

    def test_set_primary(self):
        from app.agent.multi_account import MultiAccountManager
        mgr = MultiAccountManager()
        mgr.add_account("telegram", "bot_1")
        mgr.add_account("telegram", "bot_2")
        mgr.set_primary("telegram", "bot_2")
        primary = mgr.get_primary("telegram")
        assert primary.account_id == "bot_2"

    def test_remove_primary_reassigns(self):
        from app.agent.multi_account import MultiAccountManager
        mgr = MultiAccountManager()
        mgr.add_account("telegram", "bot_1")
        mgr.add_account("telegram", "bot_2")
        mgr.remove_account("telegram", "bot_1")
        primary = mgr.get_primary("telegram")
        assert primary is not None  # bot_2 became primary

    def test_connect_account(self):
        from app.agent.multi_account import MultiAccountManager, AccountState
        mgr = MultiAccountManager()
        mgr.add_account("telegram", "bot_1")
        loop = asyncio.get_event_loop()
        assert loop.run_until_complete(mgr.connect_account("telegram", "bot_1")) == True
        acct = mgr.get_account("telegram", "bot_1")
        assert acct.state == AccountState.CONNECTED

    def test_list_platforms(self):
        from app.agent.multi_account import MultiAccountManager
        mgr = MultiAccountManager()
        mgr.add_account("telegram", "bot_1")
        mgr.add_account("discord", "guild_1")
        assert len(mgr.list_platforms()) == 2

    def test_record_message(self):
        from app.agent.multi_account import MultiAccountManager
        mgr = MultiAccountManager()
        mgr.add_account("telegram", "bot_1")
        mgr.record_message("telegram", "bot_1")
        mgr.record_message("telegram", "bot_1")
        acct = mgr.get_account("telegram", "bot_1")
        assert acct.message_count == 2

    def test_stats(self):
        from app.agent.multi_account import MultiAccountManager
        mgr = MultiAccountManager()
        mgr.add_account("telegram", "bot_1")
        s = mgr.stats()
        assert s["total_accounts"] == 1

    def test_singleton(self):
        from app.agent.multi_account import get_account_manager
        m1 = get_account_manager()
        m2 = get_account_manager()
        assert m1 is m2


# ── Delivery Modes ───────────────────────────────────

class TestDeliveryModes:
    def test_import(self):
        from app.agent.delivery_modes import DeliveryManager, DeliveryMode
        assert DeliveryManager is not None

    def test_set_and_get_mode(self):
        from app.agent.delivery_modes import DeliveryManager, DeliveryMode
        mgr = DeliveryManager()
        mgr.set_mode("telegram", "group_1", DeliveryMode.DIRECT)
        assert mgr.get_mode("telegram", "group_1") == DeliveryMode.DIRECT

    def test_default_gateway(self):
        from app.agent.delivery_modes import DeliveryManager, DeliveryMode
        mgr = DeliveryManager()
        assert mgr.get_mode("telegram", "unknown") == DeliveryMode.GATEWAY

    def test_platform_default(self):
        from app.agent.delivery_modes import DeliveryManager, DeliveryMode
        mgr = DeliveryManager()
        mgr.set_platform_default("discord", DeliveryMode.DIRECT)
        assert mgr.get_mode("discord", "any_channel") == DeliveryMode.DIRECT

    def test_should_deliver(self):
        from app.agent.delivery_modes import DeliveryManager, DeliveryMode
        mgr = DeliveryManager()
        mgr.set_mode("telegram", "muted", DeliveryMode.NONE)
        assert mgr.should_deliver("telegram", "muted") == False
        assert mgr.should_deliver("telegram", "active") == True

    def test_announce_only(self):
        from app.agent.delivery_modes import DeliveryManager, DeliveryMode
        mgr = DeliveryManager()
        mgr.set_mode("telegram", "broadcast", DeliveryMode.ANNOUNCE)
        assert mgr.is_announce_only("telegram", "broadcast") == True

    def test_remove_config(self):
        from app.agent.delivery_modes import DeliveryManager, DeliveryMode
        mgr = DeliveryManager()
        mgr.set_mode("telegram", "temp", DeliveryMode.DIRECT)
        assert mgr.remove_config("telegram", "temp") == True
        assert mgr.get_mode("telegram", "temp") == DeliveryMode.GATEWAY

    def test_list_configs(self):
        from app.agent.delivery_modes import DeliveryManager, DeliveryMode
        mgr = DeliveryManager()
        mgr.set_mode("telegram", "g1", DeliveryMode.DIRECT)
        mgr.set_mode("telegram", "g2", DeliveryMode.ANNOUNCE)
        assert len(mgr.list_configs()) == 2

    def test_stats(self):
        from app.agent.delivery_modes import DeliveryManager, DeliveryMode
        mgr = DeliveryManager()
        mgr.set_mode("telegram", "g1", DeliveryMode.DIRECT)
        s = mgr.stats()
        assert s["total_configs"] == 1

    def test_singleton(self):
        from app.agent.delivery_modes import get_delivery_manager
        m1 = get_delivery_manager()
        m2 = get_delivery_manager()
        assert m1 is m2


# ── Gmail Pub/Sub ────────────────────────────────────

class TestGmailPubSub:
    def test_import(self):
        from app.agent.gmail_pubsub import GmailPubSubWatcher
        assert GmailPubSubWatcher is not None

    def test_add_watch(self):
        from app.agent.gmail_pubsub import GmailPubSubWatcher
        w = GmailPubSubWatcher()
        watch = w.add_watch("user@gmail.com")
        assert watch.email == "user@gmail.com"

    def test_remove_watch(self):
        from app.agent.gmail_pubsub import GmailPubSubWatcher
        w = GmailPubSubWatcher()
        w.add_watch("user@gmail.com")
        assert w.remove_watch("user@gmail.com") == True
        assert w.get_watch("user@gmail.com") is None

    def test_process_notification(self):
        from app.agent.gmail_pubsub import GmailPubSubWatcher
        w = GmailPubSubWatcher()
        w.add_watch("user@gmail.com")
        data = base64.b64encode(json.dumps({
            "emailAddress": "user@gmail.com",
            "historyId": 12345,
        }).encode()).decode()
        loop = asyncio.get_event_loop()
        event = loop.run_until_complete(w.process_notification({
            "message": {"data": data}
        }))
        assert event is not None
        assert event.email == "user@gmail.com"

    def test_ignore_unwatched_email(self):
        from app.agent.gmail_pubsub import GmailPubSubWatcher
        w = GmailPubSubWatcher()
        data = base64.b64encode(json.dumps({
            "emailAddress": "unknown@gmail.com",
            "historyId": 1,
        }).encode()).decode()
        loop = asyncio.get_event_loop()
        event = loop.run_until_complete(w.process_notification({
            "message": {"data": data}
        }))
        assert event is None

    def test_pause_resume(self):
        from app.agent.gmail_pubsub import GmailPubSubWatcher, WatchState
        w = GmailPubSubWatcher()
        w.add_watch("user@gmail.com")
        w.pause_watch("user@gmail.com")
        assert w.get_watch("user@gmail.com").state == WatchState.PAUSED
        w.resume_watch("user@gmail.com")
        assert w.get_watch("user@gmail.com").state == WatchState.ACTIVE

    def test_list_watches(self):
        from app.agent.gmail_pubsub import GmailPubSubWatcher
        w = GmailPubSubWatcher()
        w.add_watch("a@gmail.com")
        w.add_watch("b@gmail.com")
        assert len(w.list_watches()) == 2

    def test_stats(self):
        from app.agent.gmail_pubsub import GmailPubSubWatcher
        w = GmailPubSubWatcher()
        w.add_watch("user@gmail.com")
        s = w.stats()
        assert s["total_watches"] == 1

    def test_singleton(self):
        from app.agent.gmail_pubsub import get_gmail_watcher
        w1 = get_gmail_watcher()
        w2 = get_gmail_watcher()
        assert w1 is w2


# ── Agent-to-Agent Policies ──────────────────────────

class TestAgentPolicies:
    def test_import(self):
        from app.agent.agent_policies import AgentPolicyManager
        assert AgentPolicyManager is not None

    def test_allow_all_default(self):
        from app.agent.agent_policies import AgentPolicyManager, SpawnPermission
        mgr = AgentPolicyManager()
        assert mgr.can_spawn("a1", "a2") == SpawnPermission.ALLOWED

    def test_allow_list(self):
        from app.agent.agent_policies import AgentPolicyManager, PolicyMode, SpawnPermission
        mgr = AgentPolicyManager()
        mgr.set_policy("a1", mode=PolicyMode.ALLOW_LIST, allowed_agents=["a2"])
        assert mgr.can_spawn("a1", "a2") == SpawnPermission.ALLOWED
        assert mgr.can_spawn("a1", "a3") == SpawnPermission.DENIED

    def test_deny_list(self):
        from app.agent.agent_policies import AgentPolicyManager, PolicyMode, SpawnPermission
        mgr = AgentPolicyManager()
        mgr.set_policy("a1", mode=PolicyMode.DENY_LIST, denied_agents=["a2"])
        assert mgr.can_spawn("a1", "a2") == SpawnPermission.DENIED
        assert mgr.can_spawn("a1", "a3") == SpawnPermission.ALLOWED

    def test_deny_all(self):
        from app.agent.agent_policies import AgentPolicyManager, PolicyMode, SpawnPermission
        mgr = AgentPolicyManager()
        mgr.set_policy("a1", mode=PolicyMode.DENY_ALL)
        assert mgr.can_spawn("a1", "a2") == SpawnPermission.DENIED

    def test_target_not_spawnable(self):
        from app.agent.agent_policies import AgentPolicyManager, SpawnPermission
        mgr = AgentPolicyManager()
        mgr.set_policy("a2", can_be_spawned=False)
        assert mgr.can_spawn("a1", "a2") == SpawnPermission.DENIED

    def test_concurrent_limit(self):
        from app.agent.agent_policies import AgentPolicyManager, SpawnPermission
        mgr = AgentPolicyManager()
        mgr.set_policy("a1", max_concurrent=1)
        mgr.record_spawn("a1", "a2")
        assert mgr.can_spawn("a1", "a3") == SpawnPermission.DENIED

    def test_record_despawn(self):
        from app.agent.agent_policies import AgentPolicyManager, SpawnPermission
        mgr = AgentPolicyManager()
        mgr.set_policy("a1", max_concurrent=1)
        mgr.record_spawn("a1", "a2")
        mgr.record_despawn("a1", "a2")
        assert mgr.can_spawn("a1", "a3") == SpawnPermission.ALLOWED

    def test_add_allowed(self):
        from app.agent.agent_policies import AgentPolicyManager, SpawnPermission
        mgr = AgentPolicyManager()
        mgr.add_allowed("a1", "a2")
        # Mode switches to ALLOW_LIST
        assert mgr.can_spawn("a1", "a2") == SpawnPermission.ALLOWED
        assert mgr.can_spawn("a1", "a3") == SpawnPermission.DENIED

    def test_get_active_spawns(self):
        from app.agent.agent_policies import AgentPolicyManager
        mgr = AgentPolicyManager()
        mgr.record_spawn("a1", "a2")
        mgr.record_spawn("a1", "a3")
        assert len(mgr.get_active_spawns("a1")) == 2

    def test_stats(self):
        from app.agent.agent_policies import AgentPolicyManager
        mgr = AgentPolicyManager()
        mgr.set_policy("a1")
        s = mgr.stats()
        assert s["total_policies"] == 1

    def test_singleton(self):
        from app.agent.agent_policies import get_policy_manager
        m1 = get_policy_manager()
        m2 = get_policy_manager()
        assert m1 is m2


# ── TUI Chat ─────────────────────────────────────────

class TestTUIChat:
    def test_import(self):
        from app.agent.tui_chat import TUIChat
        assert TUIChat is not None

    def test_display_message(self):
        from app.agent.tui_chat import TUIChat
        tui = TUIChat()
        tui.display_message("user", "Hello!")
        assert tui.message_count == 1

    def test_render(self):
        from app.agent.tui_chat import TUIChat
        tui = TUIChat(agent_id="test-agent")
        tui.display_message("user", "Hello!")
        tui.display_message("assistant", "Hi!")
        output = tui.render()
        assert "test-agent" in output
        assert "Hello!" in output
        assert "Hi!" in output

    def test_parse_command(self):
        from app.agent.tui_chat import TUIChat
        tui = TUIChat()
        result = tui.parse_input("/status")
        assert result["type"] == "command"
        assert result["command"] == "status"

    def test_parse_message(self):
        from app.agent.tui_chat import TUIChat
        tui = TUIChat()
        result = tui.parse_input("just a message")
        assert result["type"] == "message"

    def test_set_status(self):
        from app.agent.tui_chat import TUIChat
        tui = TUIChat()
        tui.set_status("Connected")
        tui.display_message("system", "x")
        output = tui.render()
        assert "Connected" in output

    def test_set_option(self):
        from app.agent.tui_chat import TUIChat
        tui = TUIChat()
        assert tui.set_option("timestamps", True) == True
        assert tui.set_option("invalid", True) == False

    def test_clear(self):
        from app.agent.tui_chat import TUIChat
        tui = TUIChat()
        tui.display_message("user", "test")
        tui.clear()
        assert tui.message_count == 0

    def test_history(self):
        from app.agent.tui_chat import TUIChat
        tui = TUIChat()
        tui.parse_input("/status")
        tui.parse_input("hello")
        h = tui.get_history()
        assert len(h) == 2

    def test_stats(self):
        from app.agent.tui_chat import TUIChat
        tui = TUIChat()
        tui.display_message("user", "x")
        tui.display_message("assistant", "y")
        s = tui.stats()
        assert s["total_messages"] == 2


# ── Tailscale ────────────────────────────────────────

class TestTailscale:
    def test_import(self):
        from app.agent.tailscale import TailscaleManager
        assert TailscaleManager is not None

    def test_configure(self):
        from app.agent.tailscale import TailscaleManager
        mgr = TailscaleManager()
        config = mgr.configure(auth_key="tskey-xxx", hostname="hexbrain")
        assert config.hostname == "hexbrain"

    def test_connect(self):
        from app.agent.tailscale import TailscaleManager
        mgr = TailscaleManager()
        mgr.configure(hostname="hexbrain", tailnet="mynet")
        loop = asyncio.get_event_loop()
        assert loop.run_until_complete(mgr.connect()) == True
        assert mgr.is_connected == True

    def test_serve(self):
        from app.agent.tailscale import TailscaleManager
        mgr = TailscaleManager()
        mgr.configure(hostname="hexbrain")
        loop = asyncio.get_event_loop()
        loop.run_until_complete(mgr.connect())
        config = loop.run_until_complete(mgr.serve(port=8000, path="/api"))
        assert config.active == True
        assert config.port == 8000

    def test_funnel(self):
        from app.agent.tailscale import TailscaleManager
        mgr = TailscaleManager()
        mgr.configure(hostname="hexbrain")
        loop = asyncio.get_event_loop()
        loop.run_until_complete(mgr.connect())
        config = loop.run_until_complete(mgr.funnel(port=8000))
        assert config.funnel == True

    def test_get_url(self):
        from app.agent.tailscale import TailscaleManager
        mgr = TailscaleManager()
        mgr.configure(hostname="hexbrain", tailnet="mynet")
        loop = asyncio.get_event_loop()
        loop.run_until_complete(mgr.connect())
        loop.run_until_complete(mgr.serve(port=8000))
        url = mgr.get_url(8000)
        assert "hexbrain" in url

    def test_stop_serve(self):
        from app.agent.tailscale import TailscaleManager
        mgr = TailscaleManager()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(mgr.connect())
        loop.run_until_complete(mgr.serve(port=8000))
        assert loop.run_until_complete(mgr.stop_serve(8000)) == True

    def test_disconnect(self):
        from app.agent.tailscale import TailscaleManager
        mgr = TailscaleManager()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(mgr.connect())
        loop.run_until_complete(mgr.disconnect())
        assert mgr.is_connected == False

    def test_status(self):
        from app.agent.tailscale import TailscaleManager
        mgr = TailscaleManager()
        s = mgr.status()
        assert s["state"] == "disconnected"

    def test_singleton(self):
        from app.agent.tailscale import get_tailscale_manager
        m1 = get_tailscale_manager()
        m2 = get_tailscale_manager()
        assert m1 is m2


# ── SSH Tunnels ──────────────────────────────────────

class TestSSHTunnels:
    def test_import(self):
        from app.agent.ssh_tunnels import SSHTunnelManager
        assert SSHTunnelManager is not None

    def test_create_tunnel(self):
        from app.agent.ssh_tunnels import SSHTunnelManager, TunnelType
        mgr = SSHTunnelManager()
        tunnel = mgr.create_tunnel("api", "vps.example.com", local_port=8000, remote_bind_port=9000)
        assert tunnel.name == "api"
        assert tunnel.tunnel_type == TunnelType.REMOTE

    def test_open_tunnel(self):
        from app.agent.ssh_tunnels import SSHTunnelManager, TunnelState
        mgr = SSHTunnelManager()
        mgr.create_tunnel("api", "vps.example.com")
        loop = asyncio.get_event_loop()
        assert loop.run_until_complete(mgr.open_tunnel("api")) == True
        assert mgr.get_tunnel("api").state == TunnelState.OPEN

    def test_close_tunnel(self):
        from app.agent.ssh_tunnels import SSHTunnelManager, TunnelState
        mgr = SSHTunnelManager()
        mgr.create_tunnel("api", "vps.example.com")
        loop = asyncio.get_event_loop()
        loop.run_until_complete(mgr.open_tunnel("api"))
        loop.run_until_complete(mgr.close_tunnel("api"))
        assert mgr.get_tunnel("api").state == TunnelState.CLOSED

    def test_reconnect(self):
        from app.agent.ssh_tunnels import SSHTunnelManager, TunnelState
        mgr = SSHTunnelManager()
        mgr.create_tunnel("api", "vps.example.com")
        loop = asyncio.get_event_loop()
        loop.run_until_complete(mgr.open_tunnel("api"))
        loop.run_until_complete(mgr.reconnect("api"))
        t = mgr.get_tunnel("api")
        assert t.reconnect_count == 1
        assert t.state == TunnelState.OPEN

    def test_remove_open_tunnel_blocked(self):
        from app.agent.ssh_tunnels import SSHTunnelManager
        mgr = SSHTunnelManager()
        mgr.create_tunnel("api", "vps.example.com")
        loop = asyncio.get_event_loop()
        loop.run_until_complete(mgr.open_tunnel("api"))
        assert mgr.remove_tunnel("api") == False  # Can't remove while open

    def test_remove_closed_tunnel(self):
        from app.agent.ssh_tunnels import SSHTunnelManager
        mgr = SSHTunnelManager()
        mgr.create_tunnel("api", "vps.example.com")
        assert mgr.remove_tunnel("api") == True

    def test_list_tunnels(self):
        from app.agent.ssh_tunnels import SSHTunnelManager
        mgr = SSHTunnelManager()
        mgr.create_tunnel("t1", "h1.com")
        mgr.create_tunnel("t2", "h2.com")
        assert len(mgr.list_tunnels()) == 2

    def test_open_count(self):
        from app.agent.ssh_tunnels import SSHTunnelManager
        mgr = SSHTunnelManager()
        mgr.create_tunnel("t1", "h1.com")
        mgr.create_tunnel("t2", "h2.com")
        loop = asyncio.get_event_loop()
        loop.run_until_complete(mgr.open_tunnel("t1"))
        assert mgr.get_open_count() == 1

    def test_stats(self):
        from app.agent.ssh_tunnels import SSHTunnelManager
        mgr = SSHTunnelManager()
        mgr.create_tunnel("t1", "h1.com")
        s = mgr.stats()
        assert s["total_tunnels"] == 1

    def test_singleton(self):
        from app.agent.ssh_tunnels import get_tunnel_manager
        m1 = get_tunnel_manager()
        m2 = get_tunnel_manager()
        assert m1 is m2
