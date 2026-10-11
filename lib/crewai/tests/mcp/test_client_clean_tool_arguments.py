"""Tests for MCPClient tool argument cleaning of container values."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai.events.event_bus import crewai_event_bus
from crewai.mcp.client import MCPClient, _MCPToolResult
from crewai.mcp.transports.base import BaseTransport, TransportType


class MockTransport(BaseTransport):
    @property
    def transport_type(self) -> TransportType:
        return TransportType.STREAMABLE_HTTP

    async def connect(self) -> "MockTransport":
        self._read_stream = MagicMock()
        self._write_stream = MagicMock()
        self._connected = True
        return self

    async def disconnect(self) -> None:
        self._connected = False

    async def __aenter__(self) -> "MockTransport":
        return await self.connect()

    async def __aexit__(self, *args: object) -> None:
        await self.disconnect()


def test_clean_tool_arguments_keeps_empty_containers():
    client = MCPClient(MockTransport())

    cleaned = client._clean_tool_arguments(
        {"query": "cats", "tags": [], "filters": {}}
    )

    assert cleaned == {"query": "cats", "tags": [], "filters": {}}


def test_clean_tool_arguments_keeps_list_items_that_clean_to_empty():
    client = MCPClient(MockTransport())

    cleaned = client._clean_tool_arguments({"mixed": [{"opt": None}, 1]})

    assert cleaned == {"mixed": [{}, 1]}


def test_clean_tool_arguments_removes_none_and_keeps_falsy_scalars():
    client = MCPClient(MockTransport())

    cleaned = client._clean_tool_arguments(
        {
            "skip": None,
            "label": "",
            "count": 0,
            "nested": {"drop": None, "keep": 0},
            "items": [None, "a"],
        }
    )

    assert cleaned == {
        "label": "",
        "count": 0,
        "nested": {"keep": 0},
        "items": ["a"],
    }


@pytest.mark.asyncio
async def test_call_tool_result_sends_empty_containers_to_server():
    transport = MockTransport()
    transport._connected = True
    client = MCPClient(transport)
    client._initialized = True

    with (
        patch.object(
            client,
            "_call_tool_impl",
            AsyncMock(return_value=_MCPToolResult("ok", False)),
        ) as call_impl,
        crewai_event_bus.scoped_handlers(),
    ):
        await client.call_tool_result("search", {"query": "cats", "tags": []})

    call_impl.assert_awaited_once_with("search", {"query": "cats", "tags": []})
