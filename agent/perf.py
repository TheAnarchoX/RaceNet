"""Performance logging utilities."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class PerfLogger:
    """Lightweight JSONL performance logger."""

    def __init__(self, path: Path | str | None, agent_id: str = ""):
        self._path = Path(path) if path else None
        self._agent_id = agent_id
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._path is not None

    def log(self, event: str, **data: Any) -> None:
        if not self._path:
            return
        payload = {
            "ts": time.time(),
            "event": event,
            "agent_id": self._agent_id,
            **data,
        }
        line = json.dumps(payload, ensure_ascii=False)
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
