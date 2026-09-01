"""ИНН -> карточка организации, без реестров и платных источников.

Реквизиты восстанавливаются из сниппетов справочников в поисковой выдаче: сначала
регуляркой, а что не берётся регуляркой -- маленькой моделью.
"""
from __future__ import annotations

import re

from pydantic import BaseModel

from .inn import is_entrepreneur
from .llm import LLMClient
from .models import OrgCard, SearchHit
from .search import ExaSearch

_OGRN_RE = re.compile(r"ОГРН(?:ИП)?\s*:?\s*([15]\d{12}|3\d{14})")
# ИП зовётся не «ООО «Ромашка»», а «ИП Фамилия Имя Отчество» -- отдельный разбор.
_IP_NAME_RE = re.compile(r"ИП\s+([А-ЯЁ][а-яё\-]+(?:\s+[А-ЯЁ][а-яё\-]+){1,2})")

_NAME_RE = re.compile(
    r"((?:ООО|ОАО|ЗАО|ПАО|АО|НАО|АНО|НКО|ФГУП|ГУП|МУП|ИП)\s*[«\"'][^»\"']{2,80}[»\"'])"
)
# Без кавычек пишут не реже: «ПАО Сбербанк», «АО Альфа-Банк». Запасной разбор.
_NAME_PLAIN_RE = re.compile(
    r"\b(ООО|ОАО|ЗАО|ПАО|АО|НАО|АНО|НКО)\s+"
    r"([А-ЯЁ][А-Яа-яЁё\-]{2,30}(?:\s+[А-ЯЁ][А-Яа-яЁё\-]{2,30}){0,2})"
)

_NAME_LONG_RE = re.compile(
    r"(ОБЩЕСТВО С ОГРАНИЧЕННОЙ ОТВЕТСТВЕННОСТЬЮ\s*[«\"'][^»\"']{2,80}[»\"'])", re.IGNORECASE
)
_CITY_RE = re.compile(r"\b(?:г\.|город)\s*([А-ЯЁ][а-яё\-]{2,25})")


class OrgExtraction(BaseModel):
    """То, что модель извлекает из сниппетов, когда ключа к реестру нет."""

    found: bool
    name_short: str | None = None
    name_full: str | None = None
    ogrn: str | None = None
    city: str | None = None


EXTRACT_SYSTEM = (
    "You extract Russian company registry data from search-engine snippets. "
    "The snippets come from business-registry aggregator sites. "
    "Return ONLY a JSON object with keys: found (boolean), name_short (string or null), "
    "name_full (string or null), ogrn (string or null), city (string or null). "
    "Keep the company name exactly as written in the snippets, in Russian, including the legal form "
    "(ООО, АО, ПАО) and quotes. Never invent a name: if the snippets do not clearly state the company "
    "for the requested INN, return {\"found\": false}."
)


def _windows_near(text: str, inn: str, radius: int = 220) -> list[str]:
    """Фрагменты текста вокруг каждого упоминания ИНН.

    Провайдер отдаёт не короткий сниппет, а текст страницы, и в один такой текст попадает
    сразу несколько организаций: справочник перечисляет десяток банков, наш ИНН стоит у
    одного из них, а первое встреченное название принадлежит совсем другому. Поэтому имя
    ищется рядом с ИНН, а не по всему тексту.
    """
    return [
        text[max(0, m.start() - radius) : m.end() + radius]
        for m in re.finditer(rf"(?<!\d){re.escape(inn)}(?!\d)", text)
    ]


def relevant_hits(inn: str, hits: list[SearchHit]) -> list[SearchHit]:
    """Только те результаты, где ИНН упомянут буквально.

    Иначе реквизиты собираются из сниппетов о посторонних организациях, и дальше весь
    пайплайн уверенно ищет сайт не той компании. Лучше не узнать название, чем узнать чужое.
    """
    return [hit for hit in hits if inn in f"{hit.title} {hit.snippet}"]


def _extract_by_regex(inn: str, hits: list[SearchHit]) -> OrgCard:
    """Дешёвая ветка: реквизиты из сниппетов регуляркой, без вызова модели."""
    # Заголовок берём целиком (он почти всегда «Название -- ИНН ...»), из текста -- только
    # окрестности ИНН.
    parts: list[str] = []
    for hit in relevant_hits(inn, hits):
        parts.append(hit.title)
        parts.extend(_windows_near(f"{hit.title} {hit.snippet}", inn))
    blob = " ".join(parts)
    card = OrgCard(inn=inn, source="search_fallback")

    ogrn = _OGRN_RE.search(blob)
    if ogrn:
        card.ogrn = ogrn.group(1)

    # Длина ИНН уже сказала, кого искать: у ИП имя человека, у организации -- название с формой.
    # Без этого разбора в карточку ИП попадает название юрлица из соседнего сниппета.
    if is_entrepreneur(inn):
        ip_name = _IP_NAME_RE.search(blob)
        if ip_name:
            person = re.sub(r"\s+", " ", ip_name.group(1)).strip()
            card.name_short = f"ИП {person}"
            card.name_full = card.name_short
    else:
        long_name = _NAME_LONG_RE.search(blob)
        short_name = _NAME_RE.search(blob)
        if long_name:
            card.name_full = re.sub(r"\s+", " ", long_name.group(1)).strip()
        if short_name:
            card.name_short = re.sub(r"\s+", " ", short_name.group(1)).strip()
        if not card.name_short:  # форма без кавычек: ПАО Сбербанк
            plain = _NAME_PLAIN_RE.search(blob)
            if plain:
                card.name_short = f"{plain.group(1)} {plain.group(2)}".strip()
        if not card.name_short and card.name_full:
            card.name_short = card.name_full

    city = _CITY_RE.search(blob)
    if city:
        card.city = city.group(1)
    return card


async def resolve_from_search(
    inn: str,
    search: ExaSearch,
    llm: LLMClient | None = None,
    results_per_query: int = 10,
) -> OrgCard:
    hits = await search.search(f'"{inn}"', results_per_query)
    if len(hits) < 3:
        hits += await search.search(f"ИНН {inn} реквизиты организации", results_per_query)

    card = _extract_by_regex(inn, hits)
    if card.name_short and card.ogrn:
        return card  # регулярка справилась, модель не нужна

    usable = relevant_hits(inn, hits)
    if llm is not None and usable:
        snippets = "\n".join(f"- {hit.title} :: {hit.snippet}" for hit in usable[:8])[:4000]
        extracted = await llm.complete_json(
            EXTRACT_SYSTEM,
            f"Requested INN: {inn}\n\nSnippets:\n{snippets}",
            OrgExtraction,
            tag="extract_org",
        )
        if extracted and extracted.found:
            card.name_short = card.name_short or extracted.name_short
            card.name_full = card.name_full or extracted.name_full
            card.ogrn = card.ogrn or extracted.ogrn
            card.city = card.city or extracted.city
    return card


async def resolve_org(inn: str, search: ExaSearch, llm: LLMClient | None = None) -> OrgCard:
    """ИНН -> карточка организации.

    Реестров и платных источников не используем: наименование восстанавливается из поисковой
    выдачи. Функция оставлена отдельной точкой входа -- подключить сюда реестр значит добавить
    одну ветку, ничего не меняя выше по пайплайну.
    """
    return await resolve_from_search(inn, search, llm)
