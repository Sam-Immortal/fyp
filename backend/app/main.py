"""FastAPI application entrypoint for ScamGuard backend."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import auth, nlp, ws_audio
from app.audio.inference import get_inference_service
from app.core.config import Settings, get_settings
from app.core.middleware import SanitizationMiddleware
from app.db.session import init_db
from app.workers.broker import shutdown_broker, startup_broker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("scamguard.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info("Starting %s in %s mode", settings.app_name, settings.environment)
    await init_db()
    await startup_broker()
    # Initialize audio inference service
    get_inference_service()
    logger.info("Audio inference service initialized")
    yield
    await shutdown_broker()
    logger.info("Shutting down %s; purging in-memory buffers", settings.app_name)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Deepfake Audio Detection & AI Scam Alert System API.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(SanitizationMiddleware)

    app.include_router(auth.router)
    app.include_router(ws_audio.router)
    app.include_router(nlp.router)

    @app.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": settings.app_name, "version": "0.1.0"}

    return app


app = create_app()


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "development",
    )


if __name__ == "__main__":
    run()
