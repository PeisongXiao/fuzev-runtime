"""Shared request validation and response helpers for the HTTP adapters."""

import asyncio
import hmac
import json
import re
import time
from uuid import uuid4

from pydantic import ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse


class APIError(Exception):
    def __init__(self, status, detail):
        super().__init__(str(detail))
        self.status, self.detail = status, detail


def validation_error(field, message):
    loc = field if isinstance(field, list) else [field]
    if not loc or loc[0] not in ("body", "header", "query", "path"):
        loc = ["body", *loc]
    return APIError(422, [{"loc": loc, "msg": message, "type": "value_error"}])


def api_error(error):
    while isinstance(error, BaseExceptionGroup):
        error = error.exceptions[0]
    if isinstance(error, APIError):
        if isinstance(error.detail, dict) and error.detail.get("type") == "invalid_request_error":
            return error.status, {"error": error.detail}
        return error.status, {"detail": error.detail}
    if isinstance(error, ValidationError):
        detail = error.errors(include_input=False, include_context=False, include_url=False)
        for item in detail:
            item["loc"] = ["body", *item["loc"]]
        return 422, {"detail": detail}
    if isinstance(error, RecursionError):
        return 422, {"detail": [{"loc": ["body"], "msg": "JSON nesting is too deep", "type": "json_invalid"}]}
    if isinstance(error, (json.JSONDecodeError, UnicodeDecodeError)):
        return 422, {"detail": [{"loc": ["body"], "msg": "Expected valid JSON", "type": "json_invalid"}]}
    if isinstance(error, asyncio.CancelledError):
        return 499, {"detail": "Request cancelled"}
    return 500, {"detail": "Internal server error"}


def request_id(headers):
    values = headers.getlist("x-typesafe-request-id") if hasattr(headers, "getlist") else [
        value for key, value in headers.items() if key.lower() == "x-typesafe-request-id"]
    if not values:
        return "req_" + uuid4().hex
    if len(values) != 1 or re.fullmatch(r"[A-Za-z0-9._:-]+", values[0]) is None:
        raise validation_error(["header", "x-typesafe-request-id"],
                               "Use letters, digits, periods, underscores, colons or hyphens")
    return values[0]


def validate_decision(request, head):
    if len(request.candidates) > head.max_options:
        raise validation_error("candidates", f"This head supports at most {head.max_options} options")
    if request.primitive not in head.primitives:
        raise validation_error("primitive", "This head does not support the requested primitive")


async def parse_body(http, schema):
    return schema.model_validate(await http.json())


def add_decision_route(app, respond, submit):
    from .service_contracts import ServiceDecisionRequest

    async def decide(http: Request):
        request = await parse_body(http, ServiceDecisionRequest)

        async def operation(service, key):
            if service.head is None:
                raise APIError(404, "No decision head is loaded")
            validate_decision(request, service.head)
            return await submit(service, request, key)
        return await respond(http, operation)

    app.add_api_route("/v1/decisions", decide, methods=["POST"])


def decision_overflow(needed, context_window, field="candidates"):
    if needed > context_window:
        raise validation_error(field, f"Decision input needs {needed} tokens; the context window is {context_window}")


def health_response(model_name, context_window, *, decisions, images, streaming=True):
    return {"status": "ok", "model": model_name, "context_window": context_window,
            "capabilities": {"generation": True, "decisions": decisions,
                             "images": images, "streaming": streaming}}


async def finished_before_disconnect(http, task):
    """Wait for task after the request body is read; return False if the client disconnects first."""
    async def disconnect():
        while (await http.receive())["type"] != "http.disconnect":
            pass

    listener = asyncio.create_task(disconnect())
    try:
        await asyncio.wait({task, listener}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        listener.cancel()
        await asyncio.gather(listener, return_exceptions=True)
    return task.done()


def sse_event(value):
    return "data: " + (value if isinstance(value, str) else json.dumps(value)) + "\n\n"


def stream_error(error):
    """Final event for a started stream that is cancelled or fails."""
    status, body = api_error(error)
    return sse_event({"error": {"message": body.get("detail", "Request failed"),
                                "type": "request_cancelled" if status == 499 else "server_error",
                                "code": status}})


class APIBoundary:
    """Request IDs, body size, authentication and errors for both adapters."""

    def __init__(self, app, api_key=None, max_request_bytes=8 * 1024 * 1024):
        self.app, self.api_key = app, api_key
        self.max_request_bytes = max_request_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        key, started, received = "req_" + uuid4().hex, False, 0

        async def bounded_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_request_bytes:
                    raise APIError(413, "Request body exceeds the configured limit")
            return message

        async def respond(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                message["headers"] = [(name, value) for name, value in message.get("headers", [])
                                      if name.lower() != b"x-typesafe-request-id"]
                message["headers"].append((b"x-typesafe-request-id", key.encode("ascii")))
            await send(message)

        try:
            http = Request(scope)
            key = request_id(http.headers)
            scope.setdefault("state", {})["request_id"] = key
            if self.api_key is not None and scope["path"].startswith("/v1/"):
                supplied = http.headers.get("authorization", "")
                if not hmac.compare_digest(supplied.encode(), ("Bearer " + self.api_key).encode()):
                    raise APIError(401, "Missing or invalid bearer token")
            await self.app(scope, bounded_receive, respond)
        except Exception as error:
            if started:
                raise
            status, body = api_error(error)
            await JSONResponse(body, status_code=status)(scope, receive, respond)


def models_response(model_name, head=None):
    models = [] if head is None else [{
        "name": model_name, "description": f"FuZev decisions with {model_name}",
        "release_date": head.metadata.get("release_date", ""),
    }]
    return {"object": "list", "data": [{"id": model_name, "object": "model",
            "created": int(time.time()), "owned_by": "fuzev"}], "models": models}
