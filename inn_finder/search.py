"""Веб-поиск через Exa.ai.

Провайдер спрятан за протоколом SearchProvider: пайплайн знает только про метод search(),
поэтому подключение другого поисковика -- это ещё один класс с тем же методом.

Выбор в пользу Exa сделан замером против DuckDuckGo (ddgs): по ключевому запросу с ИНН
из задания ddgs не отдавал нужный домен ни ссылкой, ни в тексте сниппета, тогда как Exa
отдаёт его прямой ссылкой. Подробности -- в README.
"""
from __future__ import annotations

import asyncio
import os
import random
from typing import Protocol

import httpx

from .cache import Cache
from .models import SearchHit


class SearchProvider(Protocol):
    name: str

    async def search(self, query: str, max_results: int = 10) -> list[SearchHit]:
        ...


EXA_URL = "https://api.exa.ai/search"


class ExaSearch:
    """Exa.ai.

    Отличается от обычного поисковика тем, что возвращает не сниппет, а текст страницы.
    Для этого пайплайна это потенциально важно: домен компании часто лежит не в ссылке,
    а в тексте -- в поле «Сайт» справочника или в корпоративной почте. Обратная сторона --
    из длинного текста добывается и много постороннего.
    """

    name = "exa"

    def __init__(
        self,
        api_key: str = "",
        use_cache: bool = True,
        search_type: str = "",
        max_chars: int = 1200,
        concurrency: int = 3,
    ) -> None:
        self.api_key = api_key or os.getenv("EXA_API_KEY", "")
        self.search_type = search_type or os.getenv("EXA_SEARCH_TYPE", "auto")
        self.max_chars = max_chars
        self.cache = Cache(f"search-exa-{self.search_type}", use_cache)
        self._sem = asyncio.Semaphore(max(1, concurrency))
        self.calls = 0
        self.errors: list[str] = []

    def _payload(self, query: str, max_results: int) -> dict:
        return {
            "query": query,
            "numResults": max_results,
            "type": self.search_type,
            "contents": {"text": {"maxCharacters": self.max_chars}},
        }

    async def raw_search(self, query: str, max_results: int = 10) -> list[dict]:
        """Запрос в обход кэша и ограничителя -- только для замера параллели."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                EXA_URL,
                json=self._payload(query, max_results),
                headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json().get("results") or []

    async def search(self, query: str, max_results: int = 10) -> list[SearchHit]:
        if not self.api_key:
            raise RuntimeError("EXA_API_KEY не задан")

        key = f"{self.search_type}|{max_results}|{query}"
        cached = self.cache.get(key)
        if cached is not None:
            return [SearchHit(**hit) for hit in cached]

        rows: list[dict] = []
        async with self._sem, httpx.AsyncClient(timeout=30.0) as client:
            for attempt in range(3):
                try:
                    response = await client.post(
                        EXA_URL,
                        json=self._payload(query, max_results),
                        headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
                    )
                except httpx.HTTPError as exc:
                    self.errors.append(type(exc).__name__)
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue

                if response.status_code in (429, 500, 502, 503):
                    self.errors.append(f"HTTP {response.status_code}")
                    await asyncio.sleep(2.0 * (attempt + 1) + random.random())
                    continue
                if response.status_code >= 400:
                    self.errors.append(f"HTTP {response.status_code}: {response.text[:140]}")
                    break

                rows = response.json().get("results") or []
                self.calls += 1
                break

        hits = []
        for i, row in enumerate(rows):
            text = row.get("text") or ""
            highlights = " ".join(row.get("highlights") or [])
            hits.append(
                SearchHit(
                    query=query,
                    title=row.get("title") or "",
                    url=row.get("url") or "",
                    snippet=(highlights + " " + text).strip()[: self.max_chars],
                    rank=i + 1,
                )
            )
        if hits:
            self.cache.set(key, [hit.model_dump() for hit in hits])
        return hits

    async def search_many(self, queries: list[str], max_results: int = 10) -> list[SearchHit]:
        batches = await asyncio.gather(*(self.search(q, max_results) for q in queries))
        return [hit for batch in batches for hit in batch]
