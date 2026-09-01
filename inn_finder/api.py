"""Программный интерфейс: один объект обслуживает много ИНН.

Нужен и для пакетного режима, и для HTTP-сервиса: поисковый клиент с его кэшем и блокировкой,
а также клиент модели с его лимитером живут между запросами, а не создаются заново на каждый.
"""
from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

from .agent import run_agent
from .config import PipelineConfig, get_llm_config
from .llm import LLMClient, NullLLMClient
from .models import Result
from .pipeline import run_pipeline
from .search import ExaSearch


def build_llm(provider: str, use_cache: bool = True) -> LLMClient:
    if provider == "none":
        return NullLLMClient()
    return LLMClient(get_llm_config(provider), use_cache=use_cache)


class Resolver:
    """ИНН -> домен. Использовать как асинхронный контекстный менеджер.

        async with Resolver(provider="local", mode="graph") as resolver:
            result = await resolver.resolve("7721581040")
            print(result.domain)
    """

    def __init__(
        self,
        provider: str = "local",
        mode: str = "graph",
        config: PipelineConfig | None = None,
    ) -> None:
        self.provider = provider
        self.mode = mode
        self.config = config or PipelineConfig()
        self.search = ExaSearch(use_cache=self.config.use_cache)
        self.llm = build_llm(provider, self.config.use_cache)
        self._entered = False

    async def __aenter__(self) -> Resolver:
        await self.llm.__aenter__()
        self._entered = True
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self._entered = False
        await self.llm.__aexit__(*exc)

    async def resolve(self, inn: str) -> Result:
        if not self._entered:
            raise RuntimeError("Resolver используется вне 'async with'")
        # Счётчики вызовов модели -- на запрос, а не на весь процесс.
        self.llm.calls = []
        runner = run_agent if self.mode == "agent" else run_pipeline
        return await runner(inn, self.llm, self.config, self.search)

    async def resolve_many(self, inns: Iterable[str], concurrency: int = 2) -> list[Result]:
        """Пакетная обработка.

        Параллелизм умеренный: поиск и модель внутри и так сериализованы своими ограничителями,
        а десяток одновременных обходов сайтов быстро упирается в троттлинг поисковика.
        """
        semaphore = asyncio.Semaphore(max(1, concurrency))
        order = list(inns)

        async def one(inn: str) -> Result:
            async with semaphore:
                try:
                    return await self.resolve(inn)
                except Exception as exc:  # пакет не должен падать из-за одного ИНН
                    failed = Result(inn=inn, mode=self.mode, provider=self.provider)
                    failed.reason = f"исключение: {type(exc).__name__}: {exc}"
                    return failed

        return list(await asyncio.gather(*(one(inn) for inn in order)))


async def resolve_inn(inn: str, provider: str = "local", mode: str = "graph") -> Result:
    """Разовый вызов без ручного управления жизненным циклом."""
    async with Resolver(provider=provider, mode=mode) as resolver:
        return await resolver.resolve(inn)
