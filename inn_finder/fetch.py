"""Загрузка страниц кандидата и извлечение из них текста.

В LLM уходит только вычищенный текст: сырой HTML забивает контекст мусором, а маленькая
модель от мусора страдает сильнее большой.
"""
from __future__ import annotations

import asyncio
import os
import re
from urllib.parse import urljoin

import httpx
from selectolax.parser import HTMLParser

from .cache import Cache

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Страницы, где по закону и по обычаю печатают реквизиты.
COMMON_PATHS = [
    "/contacts", "/contact", "/kontakty", "/about", "/o-kompanii", "/o-nas", "/company",
    "/rekvizity", "/requisites", "/privacy", "/policy", "/oferta", "/offer",
]
_LINK_KEYWORDS = (
    "контакт", "contact", "о компании", "о нас", "about", "реквизит", "requisite",
    "оферт", "offer", "политик", "privacy", "юридическ", "info",
)


class PageFetcher:
    def __init__(self, timeout: float = 12.0, concurrency: int = 8, use_cache: bool = True) -> None:
        self.cache = Cache("pages", use_cache)
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(concurrency)
        self.calls = 0

    async def get(self, client: httpx.AsyncClient, url: str) -> dict:
        cached = self.cache.get(url)
        if cached is not None:
            return cached

        result = {"url": url, "final_url": url, "status": None, "html": "", "error": None}
        async with self.semaphore:
            try:
                response = await client.get(url)
                self.calls += 1
                result["status"] = response.status_code
                result["final_url"] = str(response.url)
                content_type = response.headers.get("content-type", "")
                if "html" in content_type or not content_type:
                    result["html"] = response.text[:400_000]
            except (httpx.HTTPError, UnicodeDecodeError, ValueError) as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"[:200]
        # Сетевой сбой кэшировать нельзя: одна икота иначе навсегда превращает живой сайт в мёртвый.
        if result["status"] is not None:
            self.cache.set(url, result)
        return result

    @staticmethod
    def make_client(timeout: float) -> httpx.AsyncClient:
        """Клиент для обхода сайтов-кандидатов.

        Проверка TLS по умолчанию выключена: заметная часть российских сайтов работает на
        сертификатах НУЦ РФ, которых нет в системном хранилище, и с проверкой они просто
        недоступны -- то есть теряются правильные ответы. Мы читаем публичные страницы и
        ничего никуда не отправляем, так что риск ограничен подменой читаемого контента.
        Включается обратно через INN_FINDER_VERIFY_TLS=1.
        """
        verify = os.getenv("INN_FINDER_VERIFY_TLS", "0") == "1"
        return httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT, "Accept-Language": "ru,en;q=0.8"},
            verify=verify,
        )


def html_to_text(html: str) -> str:
    if not html:
        return ""
    tree = HTMLParser(html)
    for tag in tree.css("script, style, noscript, svg"):
        tag.decompose()
    text = tree.body.text(separator=" ", strip=True) if tree.body else tree.text(separator=" ", strip=True)
    return re.sub(r"[\s   ]+", " ", text).strip()


def html_title(html: str) -> str:
    if not html:
        return ""
    tree = HTMLParser(html)
    node = tree.css_first("title")
    return (node.text(strip=True) if node else "")[:200]


def interesting_links(html: str, base_url: str, limit: int = 6) -> list[str]:
    """Ссылки на страницы, где вероятны реквизиты (обычно это футер)."""
    if not html:
        return []
    tree = HTMLParser(html)
    scored: list[tuple[int, str]] = []
    seen: set[str] = set()
    for node in tree.css("a[href]"):
        href = node.attributes.get("href") or ""
        if href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        text = (node.text(strip=True) or "").lower()
        haystack = f"{text} {href.lower()}"
        weight = sum(2 if kw in text else 1 for kw in _LINK_KEYWORDS if kw in haystack)
        if not weight:
            continue
        absolute = urljoin(base_url, href).split("#")[0]
        if absolute in seen:
            continue
        seen.add(absolute)
        scored.append((weight, absolute))
    scored.sort(key=lambda item: -item[0])
    return [url for _, url in scored[:limit]]
