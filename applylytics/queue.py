"""Redis Queue connection for background jobs."""

from __future__ import annotations

import os

from redis import Redis
from rq import Queue

_QUEUE: Queue | None = None


def get_redis_connection() -> Redis:
    url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    return Redis.from_url(
        url,
        socket_connect_timeout=1,
        socket_timeout=1,
    )


def ping_redis(timeout_seconds: float = 1.0) -> bool:
    """Return True if Redis responds to PING within the timeout."""
    try:
        conn = get_redis_connection()
        return bool(conn.ping())
    except Exception:
        return False


def get_queue() -> Queue:
    global _QUEUE
    if _QUEUE is None:
        _QUEUE = Queue("default", connection=get_redis_connection(), default_timeout=600)
    return _QUEUE
