"""Atomically increments a per-process request counter in worker-state.json.

The photo-analysis worker reads this file to compute req/min as one of its
"is the server idle?" signals. Counter is cumulative; the worker computes
deltas between polls.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
STATE_PATH = BACKEND_ROOT / "workers" / "worker-state.json"


class RequestCounterMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._lock = asyncio.Lock()
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

    async def dispatch(self, request: Request, call_next):
        try:
            await self._increment()
        except Exception:
            pass
        return await call_next(request)

    async def _increment(self) -> None:
        async with self._lock:
            try:
                with open(STATE_PATH, "r", encoding="utf-8") as f:
                    state = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                state = {"request_count": 0}
            state["request_count"] = int(state.get("request_count", 0)) + 1
            tmp = STATE_PATH.with_suffix(STATE_PATH.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, STATE_PATH)
