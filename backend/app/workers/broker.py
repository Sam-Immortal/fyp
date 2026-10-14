from __future__ import annotations

from taskiq_redis import RedisStreamBroker, RedisAsyncResultBackend

from app.core.config import get_settings

settings = get_settings()

result_backend = RedisAsyncResultBackend(redis_url=settings.redis_url)
broker = RedisStreamBroker(url=settings.redis_url).with_result_backend(result_backend)


async def startup_broker() -> None:
    if not broker.is_worker_process:
        await broker.startup()


async def shutdown_broker() -> None:
    if not broker.is_worker_process:
        await broker.shutdown()
