"""Клиент LLM поверх OpenAI-совместимого /chat/completions.

Один и тот же код обслуживает облачный Mistral и локальную модель на vLLM/SGLang/Ollama:
меняется только конфиг. Промпты, парсер и телеметрия общие -- иначе сравнивать модели
между собой бессмысленно.
"""
from __future__ import annotations

import asyncio
import json
import random
import re
import time
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .cache import Cache
from .config import LLMConfig
from .models import LLMCall
from .ratelimit import RateLimiter

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict | None:
    """Достаёт JSON-объект из ответа модели.

    Маленькие модели любят обрамить ответ пояснениями и markdown-заборами, поэтому
    полагаться на чистый вывод нельзя даже при включённом json-режиме сервера.
    """
    if not text:
        return None
    candidates: list[str] = []
    fence = _FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1))
    candidates.append(text)

    for chunk in candidates:
        chunk = chunk.strip()
        start = chunk.find("{")
        if start == -1:
            continue
        depth, in_string, escaped = 0, False, False
        for i, ch in enumerate(chunk[start:], start=start):
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(chunk[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


class LLMClient:
    def __init__(self, config: LLMConfig, use_cache: bool = True) -> None:
        self.config = config
        self.cache = Cache(f"llm-{config.name}", use_cache)
        self.limiter = RateLimiter(config.rps)
        self.calls: list[LLMCall] = []
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> LLMClient:
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url.rstrip("/"),
            timeout=self.config.timeout,
            headers={
                "Authorization": f"Bearer {self.config.api_key or 'EMPTY'}",
                "Content-Type": "application/json",
            },
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client:
            await self._client.aclose()

    def _build_messages(self, system: str, user: str) -> list[dict]:
        """gemma-шаблоны часто не принимают role=system -- тогда склеиваем в user."""
        if self.config.supports_system_role:
            return [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return [{"role": "user", "content": f"{system}\n\n---\n\n{user}"}]

    def _response_format(self, schema: type[BaseModel] | None) -> dict | None:
        mode = self.config.json_mode
        if mode == "json_object":
            return {"type": "json_object"}
        if mode == "json_schema" and schema is not None:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__.lower(),
                    "schema": schema.model_json_schema(),
                    "strict": True,
                },
            }
        return None

    async def _raw_call(self, messages: list[dict], schema: type[BaseModel] | None) -> tuple[str, dict]:
        assert self._client is not None, "LLMClient используется вне контекстного менеджера"
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        response_format = self._response_format(schema)
        if response_format:
            payload["response_format"] = response_format

        last_error = ""
        for attempt in range(5):
            await self.limiter.acquire()
            try:
                response = await self._client.post("/chat/completions", json=payload)
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(1.5 * (attempt + 1))
                continue

            if response.status_code == 429 or response.status_code >= 500:
                last_error = f"HTTP {response.status_code}"
                await asyncio.sleep(2.5 * (attempt + 1) + random.random())
                continue
            if response.status_code == 400 and response_format is not None:
                # Сервер не умеет заявленный json-режим -- повторяем без него.
                payload.pop("response_format", None)
                response_format = None
                last_error = f"HTTP 400: {response.text[:200]}"
                continue
            if response.status_code >= 400:
                raise RuntimeError(f"LLM {self.config.name} HTTP {response.status_code}: {response.text[:300]}")

            data = response.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            if isinstance(content, list):  # некоторые серверы отдают список частей
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            return content, data.get("usage") or {}

        raise RuntimeError(f"LLM {self.config.name} недоступна: {last_error}")

    async def complete_json(
        self,
        system: str,
        user: str,
        schema: type[T],
        tag: str = "call",
    ) -> T | None:
        """Один структурированный вызов: JSON по схеме, с одной попыткой починки формата."""
        cache_key = json.dumps(
            [self.config.model, self.config.temperature, system, user, schema.__name__],
            ensure_ascii=False,
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.calls.append(LLMCall(**cached["stats"]))
            try:
                return schema.model_validate(cached["value"]) if cached["value"] is not None else None
            except ValidationError:
                pass

        stats = LLMCall(tag=tag, provider=self.config.name, model=self.config.model)
        started = time.monotonic()
        messages = self._build_messages(system, user)
        parsed: T | None = None

        for attempt in (1, 2):
            stats.attempts = attempt
            try:
                content, usage = await self._raw_call(messages, schema)
            except RuntimeError as exc:
                stats.failed = True
                stats.error = str(exc)[:300]
                break

            stats.prompt_tokens = usage.get("prompt_tokens")
            stats.completion_tokens = usage.get("completion_tokens")
            raw = extract_json(content)
            if raw is not None:
                try:
                    parsed = schema.model_validate(raw)
                    break
                except ValidationError as exc:
                    error = str(exc)[:400]
            else:
                error = "в ответе нет валидного JSON-объекта"

            if attempt == 1:
                stats.json_ok_first_try = False
                stats.repaired = True
                messages = messages + [
                    {"role": "assistant", "content": content[:1500]},
                    {
                        "role": "user",
                        "content": (
                            f"Your previous reply was rejected: {error}\n"
                            "Reply again with ONE valid JSON object matching the schema. "
                            "No markdown, no commentary, JSON only."
                        ),
                    },
                ]
            else:
                stats.failed = True

        stats.latency_s = round(time.monotonic() - started, 3)
        self.calls.append(stats)
        if not stats.failed:  # сбой кэшировать нельзя -- он не является ответом модели
            self.cache.set(
                cache_key,
                {"value": parsed.model_dump() if parsed else None, "stats": stats.model_dump()},
            )
        return parsed

    async def complete_text(self, system: str, user: str, tag: str = "call") -> str:
        """Свободный текст -- нужен агентному циклу, который сам печатает JSON-действие.

        Кэшируется по полному транскрипту: транскрипт на каждом шаге разный, поэтому ключи
        уникальны, а повторный прогон eval воспроизводит ту же траекторию бесплатно.
        """
        cache_key = json.dumps([self.config.model, self.config.temperature, system, user], ensure_ascii=False)
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.calls.append(LLMCall(**cached["stats"]))
            return cached["value"]

        stats = LLMCall(tag=tag, provider=self.config.name, model=self.config.model)
        started = time.monotonic()
        try:
            content, usage = await self._raw_call(self._build_messages(system, user), None)
            stats.prompt_tokens = usage.get("prompt_tokens")
            stats.completion_tokens = usage.get("completion_tokens")
        except RuntimeError as exc:
            stats.failed = True
            stats.error = str(exc)[:300]
            content = ""
        stats.latency_s = round(time.monotonic() - started, 3)
        self.calls.append(stats)
        if not stats.failed:
            self.cache.set(cache_key, {"value": content, "stats": stats.model_dump()})
        return content


class NullLLMClient(LLMClient):
    """Заглушка без модели: пайплайн работает на одних детерминированных сигналах.

    Нужна как база отсчёта -- показывает, сколько именно добавляет LLM поверх кода.
    """

    def __init__(self) -> None:
        from .config import LLMConfig

        super().__init__(LLMConfig(name="none", base_url="", model="none", api_key=""), use_cache=False)

    async def __aenter__(self) -> NullLLMClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def complete_json(self, system: str, user: str, schema: type[T], tag: str = "call") -> T | None:
        return None

    async def complete_text(self, system: str, user: str, tag: str = "call") -> str:
        return ""
