"""
Plugin Registry — Plugins can register custom tools, commands, and channels.

Extends the hook system to allow plugins to contribute:
- Custom agent tools (appear in tool list)
- Chat commands (respond to /command in chat)
- Channel adapters (new message channels)

Usage:
    from app.agent.plugin_registry import get_plugin_registry

    reg = get_plugin_registry()

    # Register a custom tool
    reg.register_tool("my_plugin", ToolDefinition(
        name="my_tool",
        description="Does something cool",
        handler=my_tool_handler,
    ))

    # Register a custom command
    reg.register_command("my_plugin", CommandDefinition(
        name="my_command",
        description="A custom command",
        handler=my_command_handler,
    ))
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)

# Handler types
ToolHandler = Callable[..., Coroutine[Any, Any, Any]]
CommandHandler = Callable[..., Coroutine[Any, Any, str]]


class PluginState(str, Enum):
    ACTIVE = "active"
    DISABLED = "disabled"
    ERROR = "error"
    LOADING = "loading"


@dataclass
class ToolDefinition:
    """Definition of a plugin-provided tool."""
    name: str
    description: str
    handler: Optional[ToolHandler] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    plugin_name: str = ""
    elevated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "plugin": self.plugin_name,
            "elevated": self.elevated,
            "has_handler": self.handler is not None,
        }


@dataclass
class CommandDefinition:
    """Definition of a plugin-provided chat command."""
    name: str
    description: str
    handler: Optional[CommandHandler] = None
    usage: str = ""
    plugin_name: str = ""
    admin_only: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "usage": self.usage,
            "plugin": self.plugin_name,
            "admin_only": self.admin_only,
        }


@dataclass
class ChannelDefinition:
    """Definition of a plugin-provided channel."""
    name: str
    description: str
    channel_class: Optional[Any] = None
    config_schema: Dict[str, Any] = field(default_factory=dict)
    plugin_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "plugin": self.plugin_name,
            "has_class": self.channel_class is not None,
        }


@dataclass
class PluginInfo:
    """Information about a registered plugin."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    state: PluginState = PluginState.ACTIVE
    registered_at: float = 0.0
    tools: List[str] = field(default_factory=list)
    commands: List[str] = field(default_factory=list)
    channels: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.registered_at == 0.0:
            self.registered_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "state": self.state.value,
            "tools": self.tools,
            "commands": self.commands,
            "channels": self.channels,
        }


class PluginRegistry:
    """
    Central registry for plugin-contributed tools, commands, and channels.

    Each plugin registers itself and then contributes tools/commands/channels
    that are merged into the agent's capabilities.
    """

    def __init__(self):
        self._plugins: Dict[str, PluginInfo] = {}
        self._tools: Dict[str, ToolDefinition] = {}
        self._commands: Dict[str, CommandDefinition] = {}
        self._channels: Dict[str, ChannelDefinition] = {}

    def register_plugin(
        self,
        name: str,
        *,
        version: str = "1.0.0",
        description: str = "",
        author: str = "",
    ) -> PluginInfo:
        """Register a plugin."""
        info = PluginInfo(
            name=name,
            version=version,
            description=description,
            author=author,
        )
        self._plugins[name] = info
        logger.info(f"[PLUGIN] Registered plugin: {name} v{version}")
        return info

    def unregister_plugin(self, name: str) -> bool:
        """Unregister a plugin and all its contributions."""
        if name not in self._plugins:
            return False

        # Remove all contributions
        for tool_name in list(self._tools):
            if self._tools[tool_name].plugin_name == name:
                del self._tools[tool_name]

        for cmd_name in list(self._commands):
            if self._commands[cmd_name].plugin_name == name:
                del self._commands[cmd_name]

        for ch_name in list(self._channels):
            if self._channels[ch_name].plugin_name == name:
                del self._channels[ch_name]

        del self._plugins[name]
        logger.info(f"[PLUGIN] Unregistered plugin: {name}")
        return True

    def register_tool(self, plugin_name: str, tool: ToolDefinition) -> ToolDefinition:
        """Register a plugin-provided tool."""
        if plugin_name not in self._plugins:
            self.register_plugin(plugin_name)

        tool.plugin_name = plugin_name
        self._tools[tool.name] = tool
        self._plugins[plugin_name].tools.append(tool.name)
        logger.info(f"[PLUGIN] {plugin_name} registered tool: {tool.name}")
        return tool

    def register_command(self, plugin_name: str, command: CommandDefinition) -> CommandDefinition:
        """Register a plugin-provided chat command."""
        if plugin_name not in self._plugins:
            self.register_plugin(plugin_name)

        command.plugin_name = plugin_name
        self._commands[command.name] = command
        self._plugins[plugin_name].commands.append(command.name)
        logger.info(f"[PLUGIN] {plugin_name} registered command: /{command.name}")
        return command

    def register_channel(self, plugin_name: str, channel: ChannelDefinition) -> ChannelDefinition:
        """Register a plugin-provided channel."""
        if plugin_name not in self._plugins:
            self.register_plugin(plugin_name)

        channel.plugin_name = plugin_name
        self._channels[channel.name] = channel
        self._plugins[plugin_name].channels.append(channel.name)
        logger.info(f"[PLUGIN] {plugin_name} registered channel: {channel.name}")
        return channel

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a registered tool by name."""
        return self._tools.get(name)

    def get_command(self, name: str) -> Optional[CommandDefinition]:
        """Get a registered command by name."""
        return self._commands.get(name)

    def get_channel(self, name: str) -> Optional[ChannelDefinition]:
        """Get a registered channel by name."""
        return self._channels.get(name)

    def list_tools(self, plugin_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all registered tools."""
        tools = list(self._tools.values())
        if plugin_name:
            tools = [t for t in tools if t.plugin_name == plugin_name]
        return [t.to_dict() for t in tools]

    def list_commands(self, plugin_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all registered commands."""
        cmds = list(self._commands.values())
        if plugin_name:
            cmds = [c for c in cmds if c.plugin_name == plugin_name]
        return [c.to_dict() for c in cmds]

    def list_channels(self) -> List[Dict[str, Any]]:
        """List all registered channels."""
        return [c.to_dict() for c in self._channels.values()]

    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all registered plugins."""
        return [p.to_dict() for p in self._plugins.values()]

    def get_plugin(self, name: str) -> Optional[PluginInfo]:
        """Get plugin info."""
        return self._plugins.get(name)

    def disable_plugin(self, name: str) -> bool:
        """Disable a plugin (keeps registered but inactive)."""
        plugin = self._plugins.get(name)
        if not plugin:
            return False
        plugin.state = PluginState.DISABLED
        return True

    def enable_plugin(self, name: str) -> bool:
        """Enable a previously disabled plugin."""
        plugin = self._plugins.get(name)
        if not plugin:
            return False
        plugin.state = PluginState.ACTIVE
        return True

    def stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        active = sum(1 for p in self._plugins.values() if p.state == PluginState.ACTIVE)
        return {
            "total_plugins": len(self._plugins),
            "active_plugins": active,
            "total_tools": len(self._tools),
            "total_commands": len(self._commands),
            "total_channels": len(self._channels),
        }


# ── Singleton ────────────────────────────────────────────
_registry: Optional[PluginRegistry] = None


def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
    return _registry
