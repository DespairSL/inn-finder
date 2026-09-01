"""Детерминированная проверка кандидата: сбор улик и их взвешивание.

Идея: ИНН на сайте организации почти всегда есть (футер, контакты, оферта, политика ПДн).
Найденный на домене ИНН -- это доказательство, а не догадка, и оно не зависит от того,
насколько умна используемая модель.
"""
from __future__ import annotations

import asyncio
import re

import httpx

from .domains import brand_in_domain, name_similarity, registrable
from .fetch import COMMON_PATHS, PageFetcher, html_title, html_to_text, interesting_links
from .models import Candidate, OrgCard, PageEvidence

# ИНН берём только рядом со словом "ИНН": иначе в улики попадут телефоны и номера статей.
_PUBLISHED_INN_RE = re.compile(r"ИНН[:\s]*(\d{10}|\d{12})", re.IGNORECASE)

_PARKED_MARKERS = (
    "домен продается", "домен продаётся", "domain is for sale", "срок регистрации домена истек",
    "this domain is parked", "купить домен", "домен припаркован", "sedoparking",
)

WEIGHTS = {
    "inn_on_site": 100.0,
    "ogrn_on_site": 55.0,
    "site_field": 38.0,
    "corporate_email": 28.0,
    "registry_mention": 12.0,
    "snippet_mention": 4.0,
    "brand_in_domain": 22.0,
    "name_similarity": 25.0,
    "mention": 6.0,
    "rank": 2.0,
    "foreign_inn": -60.0,
    "unreachable": -40.0,
    "parked": -35.0,
    "ru_zone": 6.0,
    "ru_sibling": 15.0,
}


def _normalize(text: str) -> str:
    return re.sub(r"[\s   ]+", " ", text or "")


def contains_code(text: str, code: str | None) -> bool:
    """Ищет ИНН/ОГРН как отдельное число, чтобы не ловить его внутри длинных цифр."""
    if not code:
        return False
    return re.search(rf"(?<!\d){re.escape(code)}(?!\d)", _normalize(text)) is not None


def excerpt_around(text: str, code: str, width: int = 160) -> str:
    match = re.search(rf"(?<!\d){re.escape(code)}(?!\d)", _normalize(text))
    if not match:
        return ""
    start = max(0, match.start() - width // 2)
    return _normalize(text)[start : match.end() + width // 2].strip()


def published_inns(text: str) -> set[str]:
    """ИНН, объявленные на странице. Чужой ИНН на сайте -- сильный довод против кандидата."""
    return set(_PUBLISHED_INN_RE.findall(_normalize(text)))


def looks_parked(text: str) -> bool:
    low = (text or "").lower()
    return any(marker in low for marker in _PARKED_MARKERS)


# --------------------------------------------------------------------------- проверка кандидата


async def verify_candidate(
    candidate: Candidate,
    card: OrgCard,
    fetcher: PageFetcher,
    client: httpx.AsyncClient,
    max_pages: int = 5,
) -> Candidate:
    """Скачивает главную и страницы с реквизитами и ищет в них ИНН и ОГРН."""
    home_url = f"https://{candidate.domain}/"
    home = await fetcher.get(client, home_url)
    if not home.get("html") and not home.get("status"):
        home = await fetcher.get(client, f"http://{candidate.domain}/")

    candidate.reachable = bool(home.get("status") and 200 <= home["status"] < 400 and home.get("html"))
    candidate.final_url = home.get("final_url")

    home_text = html_to_text(home.get("html", ""))
    home_title = html_title(home.get("html", ""))
    seen_inns: set[str] = set(published_inns(home_text))
    if home_title:
        candidate.titles.insert(0, home_title)

    pages: list[PageEvidence] = []
    if candidate.reachable:
        pages.append(
            PageEvidence(
                url=home.get("final_url") or home_url,
                status=home.get("status"),
                inn_found=contains_code(home_text, card.inn),
                ogrn_found=contains_code(home_text, card.ogrn),
                title=home_title,
                excerpt=excerpt_around(home_text, card.inn) or home_text[:200],
            )
        )

    # Если главная редиректит на другой домен -- настоящий сайт, скорее всего, там.
    if candidate.final_url:
        redirected = registrable(candidate.final_url)
        if redirected and redirected != candidate.domain:
            candidate.redirects_to = redirected

    found = pages and pages[0].inn_found
    if candidate.reachable and not found:
        links = interesting_links(home.get("html", ""), home.get("final_url") or home_url)
        fallback = [f"https://{candidate.domain}{path}" for path in COMMON_PATHS[:6]]
        targets, seen = [], {home.get("final_url"), home_url}
        for url in links + fallback:
            if url not in seen:
                seen.add(url)
                targets.append(url)
            if len(targets) >= max_pages:
                break

        results = await asyncio.gather(*(fetcher.get(client, url) for url in targets))
        for page in results:
            if not page.get("html"):
                continue
            text = html_to_text(page["html"])
            seen_inns |= published_inns(text)
            inn_found = contains_code(text, card.inn)
            ogrn_found = contains_code(text, card.ogrn)
            if inn_found or ogrn_found:
                pages.append(
                    PageEvidence(
                        url=page.get("final_url") or page["url"],
                        status=page.get("status"),
                        inn_found=inn_found,
                        ogrn_found=ogrn_found,
                        title=html_title(page["html"]),
                        excerpt=excerpt_around(text, card.inn if inn_found else (card.ogrn or "")),
                    )
                )
            if inn_found:
                break

    candidate.pages = pages
    foreign = sorted(seen_inns - {card.inn})
    candidate.foreign_inn = foreign[0] if foreign else None
    candidate.inn_on_site = any(page.inn_found for page in pages)
    candidate.ogrn_on_site = any(page.ogrn_found for page in pages)
    if home_text and looks_parked(home_text):
        candidate.reachable = False

    names = [n for n in (card.name_short, card.name_full) if n]
    candidate.name_similarity = name_similarity(candidate.domain, names)
    return candidate


def score_candidate(candidate: Candidate, card: OrgCard, brand_variants_list: list[str]) -> float:
    """Скор по собранным уликам.

    Он НЕ решает исход: используется только для порядка, в котором кандидаты показываются
    модели, и для признака blacklisted_without_proof. Решение принимает модель.
    """
    score = 0.0
    if candidate.inn_on_site:
        score += WEIGHTS["inn_on_site"]
    if candidate.ogrn_on_site:
        score += WEIGHTS["ogrn_on_site"]
    if candidate.site_field_hits:
        score += WEIGHTS["site_field"]
    if candidate.email_hits:
        score += WEIGHTS["corporate_email"]
    if candidate.mentioned_by_registry:
        score += WEIGHTS["registry_mention"]
    score += WEIGHTS["snippet_mention"] * min(candidate.snippet_hits, 3)

    candidate.brand_in_domain = brand_in_domain(candidate.domain, brand_variants_list)
    if candidate.brand_in_domain:
        score += WEIGHTS["brand_in_domain"]
    score += WEIGHTS["name_similarity"] * candidate.name_similarity
    score += WEIGHTS["mention"] * min(candidate.mention_count, 4)
    if candidate.best_rank <= 10:
        score += WEIGHTS["rank"] * (10 - candidate.best_rank) / 10

    if candidate.domain.rsplit(".", 1)[-1] in {"ru", "su", "xn--p1ai", "рф"}:
        score += WEIGHTS["ru_zone"]
    # Чужой ИНН -- довод против, но только когда своего на сайте нет: на сайте группы компаний
    # рядом с нужным юрлицом законно соседствуют другие (так устроен dadata.ru).
    if candidate.foreign_inn and not (candidate.inn_on_site or candidate.ogrn_on_site):
        score += WEIGHTS["foreign_inn"]
    if candidate.reachable is False:
        score += WEIGHTS["unreachable"]
    proven = candidate.has_proof
    # Домен, который и есть бренд компании (yandex.ru для ООО «Яндекс»), снимает запрет:
    # чёрный список защищает от чужих ответов, а не от собственного сайта площадки.
    is_own_brand = candidate.brand_in_domain or candidate.name_similarity >= 0.8
    candidate.blacklisted_without_proof = candidate.requires_proof and not (proven or is_own_brand)
    if candidate.blacklisted_without_proof:
        score = -100.0

    candidate.score = round(score, 2)
    return candidate.score


def prefer_ru_siblings(candidates: list[Candidate]) -> None:
    """yandex.com / yandex.by / yandex.ru -- у российского юрлица основной сайт в зоне .ru.

    Правило применяется только к однокоренным доменам: сравнивать так разные бренды нельзя.
    """
    groups: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        groups.setdefault(candidate.domain.split(".")[0], []).append(candidate)
    for siblings in groups.values():
        if len(siblings) < 2:
            continue
        for candidate in siblings:
            if candidate.domain.rsplit(".", 1)[-1] in {"ru", "xn--p1ai", "рф"}:
                candidate.score = round(candidate.score + WEIGHTS["ru_sibling"], 2)


def evidence_grade(candidate: Candidate) -> tuple[float, str | None]:
    """Уверенность по КЛАССУ улик, а не по сумме баллов.

    Сумма слабых признаков может перевалить любой порог и выглядеть как доказательство --
    в отчётах это превращается в ложный эталон. Поэтому уверенность привязана к тому, что
    именно найдено.
    """
    if candidate.inn_on_site:
        return 0.97, "inn_on_site"
    if candidate.ogrn_on_site:
        return 0.93, "ogrn_on_site"
    if candidate.site_field_hits:
        return 0.75, "directory_site_field"
    if candidate.email_hits:
        return 0.7, "corporate_email"
    return 0.5, None
