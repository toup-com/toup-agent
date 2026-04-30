"""Shared infrastructure for BaseChannel adapters."""

from app.agent.channels.shared.message_handler import make_channel_handler

__all__ = ["make_channel_handler"]
