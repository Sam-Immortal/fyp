from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class SanitizationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Only process requests with a body that we can safely re-read
        if request.method in ("POST", "PUT", "PATCH"):
            content_type = request.headers.get("content-type", "")
            if content_type.startswith("application/json") or content_type.startswith("text/"):
                # Read the raw body
                body = await request.body()
                # Check for NUL bytes
                if b"\x00" in body:
                    # Strip NUL bytes
                    cleaned_body = body.replace(b"\x00", b"")
                    # Replace the request body so downstream handlers see the cleaned version
                    request._body = cleaned_body

        try:
            response = await call_next(request)
        finally:
            pass
        return response