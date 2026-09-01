"""CLI: python -m inn_finder.cli 7721581040 [--mode graph|agent] [--provider local|none]"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .api import Resolver, build_llm  # noqa: F401  (build_llm переиспользуется в eval)
from .config import PipelineConfig
from .models import Result


async def resolve(inn: str, provider: str, mode: str, config: PipelineConfig) -> Result:
    async with Resolver(provider=provider, mode=mode, config=config) as resolver:
        return await resolver.resolve(inn)


async def resolve_batch(inns: list[str], provider: str, mode: str, config: PipelineConfig,
                        concurrency: int) -> list[Result]:
    async with Resolver(provider=provider, mode=mode, config=config) as resolver:
        return await resolver.resolve_many(inns, concurrency)


def read_inns(source: str) -> list[str]:
    """Список ИНН из файла или из stdin (по одному в строке; '#' -- комментарий)."""
    raw = sys.stdin.read() if source == "-" else Path(source).read_text("utf-8")
    return [line.strip() for line in raw.splitlines() if line.strip() and not line.startswith("#")]


def format_trace(result) -> str:
    lines = [
        f"ИНН        : {result.inn}",
        f"режим      : {result.mode} | модель: {result.provider}/{result.model}",
        f"организация: {result.org.display_name if result.org else '-'} "
        f"(источник карточки: {result.org.source if result.org else '-'})",
        f"время      : {result.elapsed_s} с | поисков: {result.searches} | загрузок страниц: {result.fetches}",
        f"причина    : {result.reason}",
    ]
    if result.llm_calls:
        failed = sum(1 for call in result.llm_calls if call.failed)
        repaired = sum(1 for call in result.llm_calls if call.repaired)
        tokens = sum((call.completion_tokens or 0) for call in result.llm_calls)
        lines.append(
            f"вызовов LLM: {len(result.llm_calls)} | сбоев: {failed} | починок JSON: {repaired} | "
            f"токенов на выходе: {tokens}"
        )
    if result.agent_steps:
        lines.append("шаги агента:")
        for step in result.agent_steps:
            mark = "!" if step.invalid else ("=" if step.repeated else " ")
            lines.append(f"  {mark}{step.index}. {step.action or 'INVALID'} {json.dumps(step.args, ensure_ascii=False)}")
        if result.raw_llm_answer is not None:
            lines.append(f"  ответ модели до проверки: {result.raw_llm_answer}")
    if result.candidates:
        lines.append("кандидаты:")
        for candidate in result.candidates[:6]:
            flags = []
            if candidate.inn_on_site:
                flags.append("ИНН на сайте")
            if candidate.ogrn_on_site:
                flags.append("ОГРН на сайте")
            if candidate.site_field_hits or candidate.email_hits:
                flags.append("указан в справочнике как сайт")
            if candidate.reachable is False:
                flags.append("недоступен")
            lines.append(f"  {candidate.score:7.2f}  {candidate.domain:<28} {', '.join(flags)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Поиск официального сайта организации по ИНН")
    parser.add_argument("inn", nargs="?", help="ИНН организации (10 или 12 цифр)")
    parser.add_argument("--batch", metavar="ФАЙЛ",
                        help="файл со списком ИНН (по одному в строке), '-' -- читать stdin; "
                             "результат печатается построчно в JSONL")
    parser.add_argument("--concurrency", type=int, default=2,
                        help="сколько ИНН обрабатывать параллельно в пакетном режиме")
    parser.add_argument("--mode", choices=["graph", "agent"], default="graph",
                        help="graph -- порядок шагов задаёт код, решение принимает модель; agent -- ReAct-цикл")
    parser.add_argument("--provider", choices=["local", "none"], default="local",
                        help="local -- модель на OpenAI-совместимом сервере; "
                             "none -- без модели, только детерминированные сигналы (база отсчёта)")
    parser.add_argument("--no-cache", action="store_true", help="игнорировать кэш поиска/страниц/LLM")
    parser.add_argument("--verbose", "-v", action="store_true", help="показать трассировку")
    parser.add_argument("--full", action="store_true", help="выдать весь Result в JSON")
    args = parser.parse_args(argv)

    config = PipelineConfig(use_cache=not args.no_cache)

    if args.batch:
        inns = read_inns(args.batch)
        results = asyncio.run(resolve_batch(inns, args.provider, args.mode, config, args.concurrency))
        # Эхо исходной строки, а не нормализованного ИНН: иначе две мусорные строки
        # схлопываются в одинаковый пустой ключ и результат не с чем сопоставить.
        for source, result in zip(inns, results, strict=True):
            if args.verbose:
                print(format_trace(result), file=sys.stderr)
            payload = result.model_dump(exclude_none=True) if args.full else {
                "inn": source, "domain": result.domain, "proof": result.proof,
            }
            print(json.dumps(payload, ensure_ascii=False), flush=True)
        return 0

    if not args.inn:
        parser.error("укажите ИНН или --batch ФАЙЛ")
    result = asyncio.run(resolve(args.inn, args.provider, args.mode, config))

    # Молчаливая деградация вводила бы в заблуждение: если модель не отвечала, это надо сказать.
    failed_calls = sum(1 for call in result.llm_calls if call.failed)
    if failed_calls and not args.verbose:
        print(
            f"предупреждение: {failed_calls} вызов(ов) LLM не удался, ответ получен только на "
            f"детерминированных сигналах (провайдер {args.provider})",
            file=sys.stderr,
        )
    if args.verbose:
        print(format_trace(result), file=sys.stderr)
    if args.full:
        print(json.dumps(result.model_dump(exclude_none=True), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result.answer(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
