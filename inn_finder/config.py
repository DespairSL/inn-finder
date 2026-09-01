"""Конфигурация: провайдеры LLM, лимиты, пути кэша.

Провайдер LLM сменный: облачный Mistral и локальная модель говорят по одному и тому же
OpenAI-совместимому протоколу, поэтому промпты и парсер для них общие -- это условие
честного сравнения моделей.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
# Явный путь: find_dotenv() обходит стек вызовов и падает, когда модуль импортируют из stdin.
load_dotenv(ROOT / '.env')
CACHE_DIR = Path(os.getenv("INN_FINDER_CACHE", ROOT / ".cache"))


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LLMConfig:
    """Всё, чем один бэкенд отличается от другого.

    Конфигурация вынесена отдельно от клиента: клиент говорит по OpenAI-совместимому
    протоколу, поэтому подключить любой другой сервер -- это добавить сюда ещё одну функцию.
    """

    name: str
    base_url: str
    model: str
    api_key: str
    # Как просить JSON: нативный режим сервера или только промптом.
    json_mode: str = "json_object"  # json_object | json_schema | none
    # Некоторые chat-шаблоны (в частности gemma) не принимают role=system.
    supports_system_role: bool = True
    temperature: float = 0.0
    max_tokens: int = 900
    # Ограничение частоты запросов: у бесплатного Mistral ~1 rps, у локальной модели предела нет.
    rps: float = 1.0
    timeout: float = 120.0


def local_config() -> LLMConfig:
    return LLMConfig(
        name="local",
        base_url=os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1"),
        model=os.getenv("LOCAL_LLM_MODEL", "gemma-4-12b"),
        api_key=os.getenv("LOCAL_LLM_API_KEY", "EMPTY"),
        json_mode=os.getenv("LOCAL_LLM_JSON_MODE", "none"),
        supports_system_role=_flag("LOCAL_LLM_SYSTEM_ROLE", False),
        rps=float(os.getenv("LOCAL_LLM_RPS", "1000")),
        timeout=float(os.getenv("LOCAL_LLM_TIMEOUT", "300")),
    )


PROVIDERS = {"local": local_config}


def get_llm_config(provider: str) -> LLMConfig:
    try:
        return PROVIDERS[provider]()
    except KeyError:
        raise SystemExit(
            f"неизвестный провайдер LLM: {provider!r}; доступны: {', '.join(PROVIDERS)}"
        ) from None


@dataclass(frozen=True)
class PipelineConfig:
    max_candidates_to_verify: int = 6
    max_pages_per_candidate: int = 5
    search_results_per_query: int = 10
    http_timeout: float = 12.0
    fetch_concurrency: int = 8
    agent_max_steps: int = 10
    # Сколько кандидатов показать модели. Это не порог приёмки: решение принимает модель.
    max_candidates_to_judge: int = 5
    use_cache: bool = True
