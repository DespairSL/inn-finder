"""Режим graph: фиксированный граф узлов, LLM -- узкий классификатор на подготовленных уликах.

Порядок такой, что модель вообще не вызывается, если ИНН нашёлся на сайте: это одновременно
самая точная ветка и экономия бесплатной квоты.
"""
from __future__ import annotations

import asyncio
import time
from typing import Literal

from pydantic import BaseModel

from .config import PipelineConfig
from .domains import (
    brand_variants,
    is_aggregator,
    is_blocked,
    looks_like_registry_hit,
    mine_domains,
    registrable,
)
from .fetch import PageFetcher
from .inn import is_entrepreneur, is_valid_inn, normalize_inn
from .llm import LLMClient
from .models import Candidate, OrgCard, Result, SearchHit
from .prompts import DECIDE_EXAMPLES, DECIDE_SYSTEM, QUERY_SYSTEM
from .registry import resolve_org
from .search import ExaSearch
from .verify import evidence_grade, prefer_ru_siblings, score_candidate, verify_candidate


class QueryPlan(BaseModel):
    queries: list[str] = []


class Decision(BaseModel):
    """Финальный ответ модели. Уверенность -- перечисление: числа маленькие модели калибруют плохо."""

    domain: str | None = None
    confidence: Literal["high", "medium", "low"] = "low"
    reason: str = ""


def build_queries(card: OrgCard, llm_queries: list[str] | None = None) -> list[str]:
    """Шаблонные запросы + то, что предложила модель. Шаблоны работают и без LLM."""
    queries = [f'"{card.inn}"', f"ИНН {card.inn} официальный сайт"]
    name = card.name_short or card.name_full
    if name:
        queries.append(f'{name} официальный сайт')
        if card.city:
            queries.append(f"{name} {card.city}")
    for query in llm_queries or []:
        if query and query not in queries:
            queries.append(query)
    return queries[:6]


def collect_candidates(
    hits: list[SearchHit], inn: str, ogrn: str | None, variants: list[str]
) -> dict[str, Candidate]:
    """Домены из выдачи плюс домены, добытые из текста сниппетов.

    Сайт компании часто вообще не попадает в выдачу ссылкой: в топе стоят справочники, а сам
    домен лежит у них в сниппете полем "Сайт" или корпоративной почтой. Игнорировать этот
    текст -- значит терять правильный ответ на ровном месте.
    """
    candidates: dict[str, Candidate] = {}

    def touch(domain: str) -> Candidate:
        return candidates.setdefault(domain, Candidate(domain=domain))

    for hit in hits:
        hit_domain = registrable(hit.url)
        is_registry = bool(hit_domain) and (
            is_aggregator(hit_domain) or looks_like_registry_hit(hit.url, hit.title, inn, ogrn, variants)
        )

        # 1. Сам домен из выдачи. Справочники и чёрный список не выбрасываем совсем:
        #    для собственного юрлица такой домен -- правильный ответ, но только по доказательству.
        if hit_domain:
            candidate = touch(hit_domain)
            candidate.requires_proof |= is_registry or is_blocked(hit_domain)
            candidate.mention_count += 1
            candidate.best_rank = min(candidate.best_rank, hit.rank)
            if hit.title:
                candidate.titles.append(hit.title)
            if hit.snippet:
                candidate.snippets.append(hit.snippet)

        # 2. Домены из текста сниппета -- с учётом того, чем именно они подтверждены.
        for domain, kind in mine_domains(f"{hit.title} {hit.snippet}"):
            if domain == hit_domain:
                continue
            candidate = touch(domain)
            candidate.mentioned_by_registry |= is_registry
            if kind == "site_field":
                candidate.site_field_hits += 1
            elif kind == "email":
                candidate.email_hits += 1
            else:
                candidate.snippet_hits += 1
            if hit.snippet:
                candidate.snippets.append(hit.snippet)

    return candidates


def prior_rank(candidate: Candidate, variants: list[str]) -> float:
    """Дешёвый предварительный порядок: кого проверять сетевыми запросами в первую очередь."""
    from .domains import brand_in_domain, name_similarity

    score = candidate.mention_count * 3.0 + max(0, 10 - candidate.best_rank) * 0.5
    score -= 6.0 * candidate.requires_proof
    score += 20.0 * bool(candidate.site_field_hits) + 14.0 * bool(candidate.email_hits)
    score += 4.0 * min(candidate.snippet_hits, 3) + 5.0 * candidate.mentioned_by_registry
    if brand_in_domain(candidate.domain, variants):
        score += 8.0
    score += 6.0 * name_similarity(candidate.domain, variants)
    return score


def render_candidates(candidates: list[Candidate]) -> str:
    """Нумерованный список кандидатов с уликами. Сырой HTML в модель не уходит никогда."""
    blocks = []
    for i, candidate in enumerate(candidates, start=1):
        lines = "\n   - ".join(candidate.evidence_lines())
        blocks.append(f"{i}. {candidate.domain}\n   - {lines}")
    return "\n".join(blocks)


def resolve_choice(raw: str | None, candidates: list[Candidate]) -> Candidate | None:
    """Сопоставляет ответ модели со списком кандидатов.

    Модель может ответить доменом, доменом с протоколом или номером пункта. Всё, что не
    сопоставилось, считается отказом: выдумать домен, которого не было в списке, нельзя.
    """
    if raw is None:
        return None
    value = str(raw).strip().strip('"').lower()
    if value in {"", "null", "none", "нет", "-"}:
        return None

    by_domain = {c.domain: c for c in candidates}
    if value in by_domain:
        return by_domain[value]
    normalized = registrable(value)
    if normalized and normalized in by_domain:
        return by_domain[normalized]
    if value.isdigit() and 1 <= int(value) <= len(candidates):
        return candidates[int(value) - 1]
    return None


async def decide(llm: LLMClient, card: OrgCard, candidates: list[Candidate]) -> Decision | None:
    """Единственная точка принятия решения: модель смотрит на все улики сразу."""
    user = (
        f"{DECIDE_EXAMPLES}\n\n"
        f"Now the real task.\n"
        f"Company: {card.display_name}, INN {card.inn}"
        f"{', OGRN ' + card.ogrn if card.ogrn else ''}"
        f"{', ' + card.city if card.city else ''}\n"
        f"Full legal name: {card.name_full or 'unknown'}\n"
        f"Type: {'individual entrepreneur (often has no website at all)' if is_entrepreneur(card.inn) else 'legal entity'}\n\n"
        f"Candidates:\n{render_candidates(candidates)}\n\nAnswer:"
    )
    return await llm.complete_json(DECIDE_SYSTEM, user, Decision, tag="decide")


async def run_pipeline(
    inn_raw: str,
    llm: LLMClient,
    config: PipelineConfig | None = None,
    search: ExaSearch | None = None,
) -> Result:
    config = config or PipelineConfig()
    started = time.monotonic()
    inn = normalize_inn(inn_raw)
    result = Result(inn=inn, mode="graph", provider=llm.config.name, model=llm.config.model)

    if not is_valid_inn(inn):
        result.reason = "ИНН не проходит проверку контрольной суммы"
        result.elapsed_s = round(time.monotonic() - started, 2)
        return result

    search = search or ExaSearch(use_cache=config.use_cache)
    fetcher = PageFetcher(config.http_timeout, config.fetch_concurrency, config.use_cache)

    # 1. Карточка организации: наименование и ОГРН восстанавливаются из поисковой выдачи.
    card = await resolve_org(inn, search, llm)
    result.org = card
    if not (card.name_short or card.name_full):
        result.reason = "не удалось установить наименование организации по ИНН"
        result.elapsed_s = round(time.monotonic() - started, 2)
        result.llm_calls = list(llm.calls)
        return result

    # 2. Запросы: шаблоны + вариант от модели (бренд обычно не равен юридическому названию).
    plan = await llm.complete_json(
        QUERY_SYSTEM,
        f"Company: {card.name_full or card.name_short}\nCity: {card.city or 'unknown'}",
        QueryPlan,
        tag="queries",
    )
    queries = build_queries(card, plan.queries if plan else None)

    # 3. Поиск и сбор кандидатов.
    hits = await search.search_many(queries, config.search_results_per_query)
    variants = brand_variants(card.name_short or "", card.name_full or "")
    candidates = collect_candidates(hits, inn, card.ogrn, variants)
    ordered = sorted(candidates.values(), key=lambda c: -prior_rank(c, variants))
    shortlist = ordered[: config.max_candidates_to_verify]

    if not shortlist:
        result.reason = "поиск не дал ни одного домена вне агрегаторов и соцсетей"
        result.searches, result.fetches = search.calls, fetcher.calls
        result.llm_calls = list(llm.calls)
        result.elapsed_s = round(time.monotonic() - started, 2)
        return result

    # 4. Проверка уликами: ИНН/ОГРН на страницах сайта, доступность, редиректы.
    async with PageFetcher.make_client(config.http_timeout) as client:
        shortlist = list(
            await asyncio.gather(
                *(verify_candidate(c, card, fetcher, client, config.max_pages_per_candidate) for c in shortlist)
            )
        )

        # Редирект на другой домен -- это, как правило, и есть настоящий сайт.
        extra = []
        known = {c.domain for c in shortlist}
        for candidate in shortlist:
            target = candidate.redirects_to
            if target and target not in known and not is_blocked(target):
                known.add(target)
                extra.append(Candidate(domain=target, mention_count=candidate.mention_count, best_rank=candidate.best_rank))
        if extra:
            extra = list(
                await asyncio.gather(
                    *(verify_candidate(c, card, fetcher, client, config.max_pages_per_candidate) for c in extra)
                )
            )
            shortlist.extend(extra)

    # Скор используется только для порядка показа кандидатов модели -- это мягкое решение:
    # плохой порядок стоит пропущенного ответа, но не может породить ложный.
    for candidate in shortlist:
        score_candidate(candidate, card, variants)
    prefer_ru_siblings(shortlist)

    # 5. Решение принимает модель, глядя на все улики сразу. Кода-арбитра с порогами здесь
    #    нет намеренно: пороги, подобранные на нескольких примерах, ломаются на незнакомых
    #    данных, а обобщать -- работа модели.
    ranked = sorted(shortlist, key=lambda c: -c.score)[: config.max_candidates_to_judge]
    decision = await decide(llm, card, ranked)
    if decision is None:  # модель не ответила -- отвечаем только фактами
        proven = [c for c in ranked if c.has_proof]
        chosen = max(proven, key=lambda c: c.score) if proven else None
        reason_prefix = "модель не ответила; "
    else:
        chosen = resolve_choice(decision.domain, ranked)
        reason_prefix = ""

    # 6. Проверки ответа модели. Все -- о непротиворечивости, а не подобранные пороги:
    #    выбрать можно только из показанного списка, и площадка из чёрного списка проходит
    #    лишь тогда, когда она совпадает с брендом или подтверждена реквизитами.
    if chosen is not None and chosen.blacklisted_without_proof:
        result.reason = (
            f"модель выбрала {chosen.domain} -- справочник или площадка, не связанная с этим юрлицом"
        )
        chosen = None
    elif chosen is not None and chosen.reachable is False:
        result.reason = f"модель выбрала {chosen.domain}, но сайт не отвечает"
        chosen = None
    elif chosen is not None and chosen.foreign_inn and not chosen.has_proof:
        result.reason = (
            f"модель выбрала {chosen.domain}, но на сайте опубликован ИНН {chosen.foreign_inn} -- "
            "это другое юрлицо"
        )
        chosen = None
    elif chosen is not None:
        result.confidence, result.proof = evidence_grade(chosen)
        result.domain = chosen.domain
        result.reason = reason_prefix + (
            decision.reason if decision and decision.reason else "выбрано по совокупности улик"
        )
    elif not result.reason:
        result.reason = reason_prefix + (
            decision.reason if decision and decision.reason
            else "ни один кандидат не связан с этим юрлицом"
        )

    result.candidates = sorted(shortlist, key=lambda c: -c.score)
    result.searches, result.fetches = search.calls, fetcher.calls
    result.llm_calls = list(llm.calls)
    result.elapsed_s = round(time.monotonic() - started, 2)
    return result
