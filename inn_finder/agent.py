"""Режим agent: настоящий ReAct-цикл, в котором решения принимает модель.

Протокол намеренно не использует нативный function calling: у gemma-совместимых сборок его
может не быть вовсе, а сравнение моделей честно только при побайтово одинаковом протоколе.
Действие -- один JSON-объект в тексте ответа.

Транскрипт пересобирается в одно пользовательское сообщение на каждом шаге: для небольших
моделей это стабильнее длинной многоходовой истории и одинаково ведёт себя на обоих бэкендах.
"""
from __future__ import annotations

import json
import time

import httpx

from .config import PipelineConfig
from .domains import brand_variants, is_aggregator, is_blocked, mine_domains, registrable
from .fetch import PageFetcher, html_title, html_to_text
from .inn import is_valid_inn, normalize_inn
from .llm import LLMClient, extract_json
from .models import AgentStep, Candidate, OrgCard, Result
from .prompts import AGENT_SYSTEM
from .registry import resolve_org
from .search import ExaSearch
from .verify import evidence_grade, score_candidate, verify_candidate

MAX_OBS = 1400


class AgentTools:
    """Инструменты агента. Тот же детерминированный код, что и в режиме graph."""

    def __init__(self, card: OrgCard, search: ExaSearch, fetcher: PageFetcher, config: PipelineConfig) -> None:
        self.card = card
        self.search = search
        self.fetcher = fetcher
        self.config = config
        self.checked: dict[str, Candidate] = {}

    async def run(self, action: str, args: dict, client: httpx.AsyncClient) -> str:
        if action == "search":
            return await self._search(str(args.get("query") or ""))
        if action == "fetch":
            return await self._fetch(str(args.get("url") or ""), client)
        if action == "check_company_codes":
            return await self._check(str(args.get("domain") or ""), client)
        return f"Unknown action {action!r}. Use search, fetch, check_company_codes or finish."

    async def _search(self, query: str) -> str:
        if not query:
            return "Empty query."
        hits = await self.search.search(query, self.config.search_results_per_query)
        if not hits:
            return "No results."
        lines = []
        for hit in hits[:6]:
            domain = registrable(hit.url) or "?"
            mark = " [aggregator]" if is_aggregator(domain) else (" [never-answer]" if is_blocked(domain) else "")
            lines.append(f"{domain}{mark} | {hit.title[:90]} | {hit.snippet[:220]}")

        # Домены, названные в ТЕКСТЕ сниппетов: поле "Сайт", корпоративная почта, упоминание.
        # Без этого агент видел бы меньше, чем детерминированный режим, и сравнение моделей
        # измеряло бы качество инструмента, а не решений модели.
        mined: dict[str, str] = {}
        for hit in hits:
            for candidate_domain, kind in mine_domains(f"{hit.title} {hit.snippet}"):
                mined.setdefault(candidate_domain, kind)
        if mined:
            listed = ", ".join(f"{d} ({k})" for d, k in list(mined.items())[:8])
            lines.append(f"\ndomains named inside the snippets: {listed}")
            lines.append("A domain listed as 'site_field' or 'email' is a strong lead -- check it.")
        return "\n".join(lines)

    async def _fetch(self, url: str, client: httpx.AsyncClient) -> str:
        if not url:
            return "Empty url."
        if "://" not in url:
            url = "https://" + url
        page = await self.fetcher.get(client, url)
        if not page.get("html"):
            return f"Could not read the page (status {page.get('status')}, error {page.get('error')})."
        text = html_to_text(page["html"])
        return f"title: {html_title(page['html'])}\ntext: {text[:MAX_OBS]}"

    async def _check(self, domain: str, client: httpx.AsyncClient) -> str:
        domain = registrable(domain) or domain
        if not domain:
            return "Empty domain."
        if domain in self.checked:
            candidate = self.checked[domain]
        else:
            candidate = Candidate(domain=domain, requires_proof=is_blocked(domain))
            await verify_candidate(candidate, self.card, self.fetcher, client, self.config.max_pages_per_candidate)
            self.checked[domain] = candidate

        pages = ", ".join(page.url for page in candidate.pages if page.inn_found or page.ogrn_found) or "none"
        warning = (
            "\nNOTE: this domain is a directory/marketplace/social network. It can only be the answer "
            "if it belongs to this exact legal entity, which requires the INN or OGRN to be published there."
            if is_blocked(domain)
            else ""
        )
        return (
            f"domain: {domain}\n"
            f"reachable: {candidate.reachable}\n"
            f"INN {self.card.inn} published on the site: {candidate.inn_on_site}\n"
            f"OGRN {self.card.ogrn or '-'} published on the site: {candidate.ogrn_on_site}\n"
            f"pages with the codes: {pages}\n"
            f"site title: {candidate.titles[0][:120] if candidate.titles else '-'}" + warning
        )


def _render_transcript(card: OrgCard, steps: list[AgentStep], max_steps: int) -> str:
    header = (
        f"Company to find the website for:\n"
        f"- INN: {card.inn}\n"
        f"- OGRN: {card.ogrn or 'unknown'}\n"
        f"- Legal name: {card.name_full or card.name_short or 'unknown (find it yourself)'}\n"
        f"- City: {card.city or 'unknown'}\n"
    )
    if not steps:
        return header + f"\nYou have {max_steps} steps. Output your first action as one JSON object."

    lines = [header, "\nWhat you did so far:"]
    for step in steps:
        if step.invalid:
            lines.append(f"[step {step.index}] INVALID OUTPUT -- {step.observation}")
            continue
        lines.append(f"[step {step.index}] action={step.action} args={json.dumps(step.args, ensure_ascii=False)}")
        lines.append(f"observation: {step.observation[:MAX_OBS]}")
    left = max_steps - len(steps)
    lines.append(
        f"\nYou have {left} step(s) left. Output the next action as one JSON object."
        + (" This is your LAST step: you must use finish." if left <= 1 else "")
    )
    return "\n".join(lines)


async def run_agent(
    inn_raw: str,
    llm: LLMClient,
    config: PipelineConfig | None = None,
    search: ExaSearch | None = None,
) -> Result:
    config = config or PipelineConfig()
    started = time.monotonic()
    inn = normalize_inn(inn_raw)
    result = Result(inn=inn, mode="agent", provider=llm.config.name, model=llm.config.model)

    if not is_valid_inn(inn):
        result.reason = "ИНН не проходит проверку контрольной суммы"
        result.elapsed_s = round(time.monotonic() - started, 2)
        return result

    search = search or ExaSearch(use_cache=config.use_cache)
    fetcher = PageFetcher(config.http_timeout, config.fetch_concurrency, config.use_cache)

    # Карточку добываем тем же кодом, что и в graph: сравниваем агентность, а не доступ к реестру.
    card = await resolve_org(inn, search, llm)
    result.org = card

    tools = AgentTools(card, search, fetcher, config)
    steps: list[AgentStep] = []
    seen_actions: set[str] = set()
    consecutive_invalid = 0
    finished_domain: str | None = None
    finish_reason = ""

    async with PageFetcher.make_client(config.http_timeout) as client:
        for index in range(1, config.agent_max_steps + 1):
            raw = await llm.complete_text(
                AGENT_SYSTEM, _render_transcript(card, steps, config.agent_max_steps), tag=f"agent_step_{index}"
            )
            step = AgentStep(index=index, raw=raw[:1500])
            parsed = extract_json(raw)

            if not parsed or not isinstance(parsed.get("action"), str):
                step.invalid = True
                step.observation = "Your output was not a single JSON object with an 'action' key. Try again."
                steps.append(step)
                consecutive_invalid += 1
                # Модель, которая трижды подряд не смогла выдать JSON, не выдаст его и на десятый
                # раз -- дальше это просто трата квоты.
                if consecutive_invalid >= 3:
                    break
                continue
            consecutive_invalid = 0

            step.action = parsed["action"].strip()
            step.args = parsed.get("args") if isinstance(parsed.get("args"), dict) else {}

            if step.action == "finish":
                finished_domain = step.args.get("domain")
                finish_reason = str(step.args.get("reason") or "")[:300]
                step.observation = "finished"
                steps.append(step)
                break

            signature = f"{step.action}:{json.dumps(step.args, sort_keys=True, ensure_ascii=False)}"
            if signature in seen_actions:
                step.repeated = True
                step.observation = "You already ran this exact action. Do something different or finish."
                steps.append(step)
                continue
            seen_actions.add(signature)

            step.observation = await tools.run(step.action, step.args, client)
            steps.append(step)

        # Ответ модели проходит ту же детерминированную проверку, что и в режиме graph.
        raw_answer = None
        if isinstance(finished_domain, str) and finished_domain.strip().lower() not in {"", "null", "none"}:
            raw_answer = registrable(finished_domain) or finished_domain.strip().lower()

        result.raw_llm_answer = raw_answer
        verified: Candidate | None = None
        if raw_answer:
            verified = tools.checked.get(raw_answer)
            if verified is None:
                verified = Candidate(domain=raw_answer, requires_proof=is_blocked(raw_answer))
                await verify_candidate(verified, card, fetcher, client, config.max_pages_per_candidate)

    variants = brand_variants(card.name_short or "", card.name_full or "")
    checked = list(tools.checked.values())
    if verified and verified.domain not in tools.checked:
        checked.append(verified)
    for candidate in checked:
        score_candidate(candidate, card, variants)

    if verified is not None:
        if verified.has_proof:
            result.domain = verified.domain
            result.confidence, result.proof = evidence_grade(verified)
            result.reason = f"агент назвал {verified.domain}; принадлежность подтверждена реквизитами"
        elif verified.blacklisted_without_proof:
            result.reason = (
                f"агент назвал {verified.domain} -- справочник/маркетплейс/соцсеть, "
                "не совпадающий с брендом и не подтверждённый реквизитами"
            )
        elif verified.reachable:
            result.domain = verified.domain
            result.confidence, result.proof = evidence_grade(verified)
            result.reason = f"агент назвал {verified.domain}; прямых подтверждений нет: {finish_reason}"
        else:
            result.reason = f"агент назвал {verified.domain}, но сайт недоступен"
    else:
        result.reason = finish_reason or "агент не назвал домен за отведённые шаги"

    result.agent_steps = steps
    result.candidates = sorted(checked, key=lambda c: -c.score)
    result.llm_calls = list(llm.calls)
    result.searches, result.fetches = search.calls, fetcher.calls
    result.elapsed_s = round(time.monotonic() - started, 2)
    return result
