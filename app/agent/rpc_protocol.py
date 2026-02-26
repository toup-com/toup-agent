"""
RPC Protocol — JSON-RPC 2.0 over WebSocket for gateway commands.

Implements a JSON-RPC 2.0 server that allows clients to invoke
agent operations, send messages, manage sessions, and control
configuration through a standardized protocol.

Usage:
    from app.agent.rpc_protocol import RPCServer, RPCMethod

    server = RPCServer()
    server.register("agent.send", handle_send)
    server.register("sessions.list", handle_sessions_list)

    # Handle incoming request
    response = await server.handle_request(json_str)
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# JSON-RPC 2.0 error codes
class RPCErrorCode(IntEnum):
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    # Custom codes
    UNAUTHORIZED = -32001
    RATE_LIMITED = -32002
    TIMEOUT = -32003


@dataclass
class RPCRequest:
    """A JSON-RPC 2.0 request."""
    method: str
    params: Union[Dict[str, Any], List[Any], None] = None
    id: Optional[Union[str, int]] = None
    jsonrpc: str = "2.0"

    @property
    def is_notification(self) -> bool:
        """Notifications have no id and expect no response."""
        return self.id is None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"jsonrpc": self.jsonrpc, "method": self.method}
        if self.params is not None:
            d["params"] = self.params
        if self.id is not None:
            d["id"] = self.id
        return d


@dataclass
class RPCResponse:
    """A JSON-RPC 2.0 response."""
    id: Optional[Union[str, int]] = None
    result: Any = None
    error: Optional[Dict[str, Any]] = None
    jsonrpc: str = "2.0"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.error is not None:
            d["error"] = self.error
        else:
            d["result"] = self.result
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @staticmethod
    def success(id: Optional[Union[str, int]], result: Any) -> "RPCResponse":
        return RPCResponse(id=id, result=result)

    @staticmethod
    def error_response(
        id: Optional[Union[str, int]],
        code: int,
        message: str,
        data: Any = None,
    ) -> "RPCResponse":
        err: Dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        return RPCResponse(id=id, error=err)


# Method handler type
RPCHandler = Callable[..., Coroutine[Any, Any, Any]]


@dataclass
class RPCMethod:
    """A registered RPC method."""
    name: str
    handler: RPCHandler
    description: str = ""
    requires_auth: bool = False
    params_schema: Optional[Dict[str, Any]] = None
    registered_at: float = 0.0

    def __post_init__(self):
        if self.registered_at == 0.0:
            self.registered_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "requires_auth": self.requires_auth,
        }


class RPCServer:
    """
    JSON-RPC 2.0 server for HexBrain gateway.

    Handles parsing, routing, and response formatting for
    JSON-RPC requests. Methods are registered with handlers.
    """

    def __init__(self):
        self._methods: Dict[str, RPCMethod] = {}
        self._request_count: int = 0
        self._error_count: int = 0

    def register(
        self,
        method_name: str,
        handler: RPCHandler,
        *,
        description: str = "",
        requires_auth: bool = False,
        params_schema: Optional[Dict[str, Any]] = None,
    ) -> RPCMethod:
        """Register an RPC method."""
        method = RPCMethod(
            name=method_name,
            handler=handler,
            description=description,
            requires_auth=requires_auth,
            params_schema=params_schema,
        )
        self._methods[method_name] = method
        logger.info(f"[RPC] Registered method: {method_name}")
        return method

    def unregister(self, method_name: str) -> bool:
        """Unregister an RPC method."""
        return self._methods.pop(method_name, None) is not None

    def parse_request(self, raw: str) -> Union[RPCRequest, List[RPCRequest], RPCResponse]:
        """
        Parse a raw JSON-RPC request string.

        Returns parsed request(s) or error response if invalid.
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            return RPCResponse.error_response(
                None, RPCErrorCode.PARSE_ERROR, f"Parse error: {e}"
            )

        if isinstance(data, list):
            # Batch request
            if not data:
                return RPCResponse.error_response(
                    None, RPCErrorCode.INVALID_REQUEST, "Empty batch"
                )
            requests = []
            for item in data:
                req = self._parse_single(item)
                if isinstance(req, RPCResponse):
                    return req  # Return first error
                requests.append(req)
            return requests

        return self._parse_single(data)

    def _parse_single(self, data: Any) -> Union[RPCRequest, RPCResponse]:
        """Parse a single request object."""
        if not isinstance(data, dict):
            return RPCResponse.error_response(
                None, RPCErrorCode.INVALID_REQUEST, "Request must be an object"
            )

        if data.get("jsonrpc") != "2.0":
            return RPCResponse.error_response(
                data.get("id"), RPCErrorCode.INVALID_REQUEST,
                "Missing or invalid jsonrpc version"
            )

        method = data.get("method")
        if not isinstance(method, str):
            return RPCResponse.error_response(
                data.get("id"), RPCErrorCode.INVALID_REQUEST,
                "Method must be a string"
            )

        return RPCRequest(
            method=method,
            params=data.get("params"),
            id=data.get("id"),
        )

    async def handle_request(self, raw: str) -> Optional[str]:
        """
        Handle a raw JSON-RPC request string.

        Returns JSON response string, or None for notifications.
        """
        self._request_count += 1
        parsed = self.parse_request(raw)

        if isinstance(parsed, RPCResponse):
            self._error_count += 1
            return parsed.to_json()

        if isinstance(parsed, list):
            # Batch
            responses = []
            for req in parsed:
                resp = await self._execute(req)
                if resp is not None:
                    responses.append(resp.to_dict())
            return json.dumps(responses) if responses else None

        resp = await self._execute(parsed)
        if resp is None:
            return None  # Notification
        return resp.to_json()

    async def _execute(self, request: RPCRequest) -> Optional[RPCResponse]:
        """Execute a single RPC request."""
        method = self._methods.get(request.method)
        if not method:
            if request.is_notification:
                return None
            self._error_count += 1
            return RPCResponse.error_response(
                request.id, RPCErrorCode.METHOD_NOT_FOUND,
                f"Method not found: {request.method}"
            )

        try:
            params = request.params or {}
            if isinstance(params, dict):
                result = await method.handler(**params)
            elif isinstance(params, list):
                result = await method.handler(*params)
            else:
                result = await method.handler(params)

            if request.is_notification:
                return None

            return RPCResponse.success(request.id, result)

        except TypeError as e:
            self._error_count += 1
            return RPCResponse.error_response(
                request.id, RPCErrorCode.INVALID_PARAMS, str(e)
            )
        except Exception as e:
            self._error_count += 1
            logger.error(f"[RPC] Error in {request.method}: {e}")
            return RPCResponse.error_response(
                request.id, RPCErrorCode.INTERNAL_ERROR, str(e)
            )

    def list_methods(self) -> List[Dict[str, Any]]:
        """List all registered methods."""
        return [m.to_dict() for m in self._methods.values()]

    def get_method(self, name: str) -> Optional[RPCMethod]:
        """Get a method by name."""
        return self._methods.get(name)

    def stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            "registered_methods": len(self._methods),
            "total_requests": self._request_count,
            "total_errors": self._error_count,
            "error_rate": round(
                self._error_count / self._request_count * 100, 1
            ) if self._request_count else 0.0,
        }


# ── Singleton ────────────────────────────────────────────
_server: Optional[RPCServer] = None


def get_rpc_server() -> RPCServer:
    """Get the global RPC server."""
    global _server
    if _server is None:
        _server = RPCServer()
    return _server
