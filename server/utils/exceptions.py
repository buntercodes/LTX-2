"""Exception handlers and middleware."""

from __future__ import annotations

import logging
import time
import uuid
from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class AppException(Exception):
    """Base application exception."""

    def __init__(self, message: str, status_code: int = 500, detail: dict | None = None):
        self.message = message
        self.status_code = status_code
        self.detail = detail
        super().__init__(message)


class ModelNotLoadedError(AppException):
    """Model not loaded error."""

    def __init__(self, message: str = "Model not loaded"):
        super().__init__(message, status_code=503)


class GenerationError(AppException):
    """Video generation error."""

    def __init__(self, message: str, detail: dict | None = None):
        super().__init__(message, status_code=500, detail=detail)


class InvalidInputError(AppException):
    """Invalid input error."""

    def __init__(self, message: str, detail: dict | None = None):
        super().__init__(message, status_code=400, detail=detail)


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Handle application exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": type(exc).__name__,
            "message": exc.message,
            "detail": exc.detail,
        },
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "detail": {"exception": str(exc)},
        },
    )


class RequestLoggingMiddleware:
    """Middleware for logging API requests."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request_id = str(uuid.uuid4())[:8]
            start_time = time.time()

            # Log request
            logger.info(
                f"[{request_id}] {scope['method']} {scope['path']}"
            )

            # Add request ID to scope
            scope["state"] = getattr(scope, "state", {})
            scope["state"]["request_id"] = request_id

            # Process request
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    # Log response
                    elapsed = time.time() - start_time
                    status = message.get("status", 200)
                    logger.info(
                        f"[{request_id}] Response: {status} ({elapsed:.3f}s)"
                    )
                await send(message)

            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)
