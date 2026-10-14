from __future__ import annotations

from app.workers.broker import broker


@broker.task(task_name="echo")
async def echo_task(message: str) -> str:
    return f"echo: {message}"
