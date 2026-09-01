"""Файловый кэш поиска, загрузок и ответов модели.

Без него разработка и прогон eval-набора сжирают бесплатную квоту и упираются в throttling
поисковика на первом же часе работы.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path
from typing import Any

from .config import CACHE_DIR


class Cache:
    def __init__(self, namespace: str, enabled: bool = True) -> None:
        self.enabled = enabled
        self.dir = CACHE_DIR / namespace
        if enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return self.dir / f"{digest}.json"

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        with contextlib.suppress(OSError, TypeError):
            self._path(key).write_text(json.dumps(value, ensure_ascii=False), "utf-8")
