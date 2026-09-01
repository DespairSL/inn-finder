"""Модели данных пайплайна. Всё, что уходит в LLM и приходит из неё, проходит через них."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class OrgCard(BaseModel):
    """Карточка организации по ИНН."""

    inn: str
    ogrn: str | None = None
    name_full: str | None = None
    name_short: str | None = None
    city: str | None = None
    source: Literal["search_fallback", "none"] = "none"

    @property
    def display_name(self) -> str:
        return self.name_short or self.name_full or f"ИНН {self.inn}"


class SearchHit(BaseModel):
    query: str
    title: str = ""
    url: str = ""
    snippet: str = ""
    rank: int = 0


class PageEvidence(BaseModel):
    url: str
    status: int | None = None
    inn_found: bool = False
    ogrn_found: bool = False
    title: str = ""
    excerpt: str = ""


class Candidate(BaseModel):
    """Домен-кандидат со всеми собранными о нём уликами."""

    domain: str
    titles: list[str] = Field(default_factory=list)
    snippets: list[str] = Field(default_factory=list)
    mention_count: int = 0
    best_rank: int = 99
    # Как домен попал в кандидаты: явное поле "Сайт", корпоративная почта, просто упоминание.
    site_field_hits: int = 0
    email_hits: int = 0
    snippet_hits: int = 0
    mentioned_by_registry: bool = False
    # Домен из чёрного списка (соцсеть, маркетплейс, справочник). Для СВОЕГО юрлица это
    # законный ответ (ozon.ru для ООО "Интернет решения"), поэтому не выбрасываем его,
    # а требуем прямого доказательства: ИНН или ОГРН на страницах самого сайта.
    requires_proof: bool = False
    # Итог правила requires_proof после скоринга: домен из чёрного списка, у которого
    # нет ни доказательства, ни совпадения с брендом компании. Ответом быть не может.
    blacklisted_without_proof: bool = False

    reachable: bool | None = None
    final_url: str | None = None
    redirects_to: str | None = None
    inn_on_site: bool = False
    ogrn_on_site: bool = False
    # ИНН другой организации, опубликованный на этом сайте: довод против кандидата.
    foreign_inn: str | None = None
    pages: list[PageEvidence] = Field(default_factory=list)
    name_similarity: float = 0.0
    brand_in_domain: bool = False

    score: float = 0.0

    @property
    def has_proof(self) -> bool:
        """Улики, прямо связывающие домен с этим юрлицом через реестровые коды."""
        return bool(self.inn_on_site or self.ogrn_on_site)

    @property
    def has_entity_evidence(self) -> bool:
        return bool(
            self.has_proof or self.site_field_hits or self.email_hits
        )

    def evidence_lines(self) -> list[str]:
        """Компактная карточка улик для LLM -- сырой HTML в модель не уходит никогда."""
        lines = [f"domain: {self.domain}"]
        if self.titles:
            lines.append(f"page title: {self.titles[0][:160]}")
        lines.append(f"reachable: {self.reachable}")
        lines.append(f"INN found on site: {self.inn_on_site}")
        # Показываем чужой ИНН только если своего на сайте нет: иначе это шум, который
        # маленькую модель только сбивает (на сайте группы компаний ИНН несколько).
        if self.foreign_inn and not (self.inn_on_site or self.ogrn_on_site):
            lines.append(
                f"the site publishes INN {self.foreign_inn}, which belongs to a DIFFERENT company"
            )
        lines.append(f"OGRN found on site: {self.ogrn_on_site}")
        lines.append(f"brand-like domain name: {self.brand_in_domain}")
        lines.append(f"name similarity: {self.name_similarity:.2f}")
        lines.append(f"appeared in {self.mention_count} search result(s), best rank {self.best_rank}")
        if self.site_field_hits:
            lines.append("a directory lists this domain in the company's \"website\" field")
        if self.email_hits:
            lines.append("the company's contact e-mail is on this domain")
        if self.mentioned_by_registry:
            lines.append("mentioned in a business-registry page about this company")
        for page in self.pages[:3]:
            if page.excerpt:
                lines.append(f"[{page.url}] {page.excerpt[:300]}")
        for snippet in self.snippets[:2]:
            lines.append(f"[search snippet] {snippet[:220]}")
        return lines


class LLMCall(BaseModel):
    """Телеметрия одного вызова модели -- на ней потом строится сравнение бэкендов."""

    tag: str
    provider: str
    model: str
    attempts: int = 1
    json_ok_first_try: bool = True
    repaired: bool = False
    failed: bool = False
    error: str | None = None
    latency_s: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class AgentStep(BaseModel):
    index: int
    raw: str = ""
    action: str | None = None
    args: dict = Field(default_factory=dict)
    observation: str = ""
    invalid: bool = False
    repeated: bool = False


class Result(BaseModel):
    """Итог работы пайплайна. Ответ на задачу -- поле domain."""

    inn: str
    domain: str | None = None
    confidence: float = 0.0
    reason: str = ""
    # Чем именно подтверждён ответ, если подтверждён: inn_on_site / ogrn_on_site /
    # directory_site_field / corporate_email.
    proof: str | None = None
    mode: str = "graph"
    provider: str = ""
    model: str = ""
    org: OrgCard | None = None
    candidates: list[Candidate] = Field(default_factory=list)
    llm_calls: list[LLMCall] = Field(default_factory=list)
    agent_steps: list[AgentStep] = Field(default_factory=list)
    # Что модель назвала сама, до детерминированной страховки (нужно для сравнения моделей).
    raw_llm_answer: str | None = None
    elapsed_s: float = 0.0
    searches: int = 0
    fetches: int = 0

    def answer(self) -> dict:
        return {"domain": self.domain}
