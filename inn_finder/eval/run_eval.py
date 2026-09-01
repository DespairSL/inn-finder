"""Прогон набора ИНН по нескольким конфигурациям и сравнительная таблица.

Ради этого и разделены режимы: одна и та же детерминированная обвязка, меняется только
модель и способ принятия решений -- значит разница в цифрах относится к модели, а не к обвязке.

    python -m inn_finder.eval.run_eval --config none:graph --config local:graph --config local:agent
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path

from ..agent import run_agent
from ..api import build_llm
from ..config import PipelineConfig
from ..models import Result
from ..pipeline import run_pipeline
from ..search import ExaSearch

DATASET = Path(__file__).parent / "dataset.json"


async def run_config(provider: str, mode: str, rows: list[dict], config: PipelineConfig) -> list[Result]:
    search = ExaSearch(use_cache=config.use_cache)  # общий кэш поиска на все конфигурации
    results: list[Result] = []
    async with build_llm(provider, config.use_cache) as llm:
        runner = run_agent if mode == "agent" else run_pipeline
        for row in rows:
            try:
                results.append(await runner(row["inn"], llm, config, search))
            except Exception as exc:  # прогон не должен падать из-за одного ИНН
                failed = Result(inn=row["inn"], mode=mode, provider=provider, model=llm.config.model)
                failed.reason = f"исключение: {type(exc).__name__}: {exc}"
                results.append(failed)
    return results


def is_correct(row: dict, result: Result) -> bool:
    """Эталон может допускать несколько равнозначных доменов (ya.ru и yandex.ru)."""
    expected = row.get("expected")
    if expected is None:  # размеченный негатив: у организации сайта нет
        return result.domain is None
    return result.domain == expected or result.domain in (row.get("alt") or [])


def summarize(rows: list[dict], results: list[Result]) -> dict:
    # Размечено = ключ "expected" присутствует. Его значение null означает "сайта нет"
    # (подтверждённый негатив), а не "ещё не размечено" -- иначе негативы не выразить,
    # а именно на них измеряется главная ошибка: выданный домен там, где его быть не должно.
    labelled = [(row, res) for row, res in zip(rows, results, strict=True) if "expected" in row]
    positives = [(row, res) for row, res in labelled if row.get("expected") is not None]
    negatives = [(row, res) for row, res in labelled if row.get("expected") is None]

    answered = [res for res in results if res.domain]
    correct = sum(1 for row, res in labelled if is_correct(row, res))
    wrong = sum(1 for row, res in positives if res.domain and not is_correct(row, res))
    missed = sum(1 for row, res in positives if not res.domain)
    false_positives = sum(1 for _, res in negatives if res.domain)

    llm_calls = [call for res in results for call in res.llm_calls]
    agent_steps = [len(res.agent_steps) for res in results if res.agent_steps]
    invalid = sum(1 for res in results for step in res.agent_steps if step.invalid)
    repeated = sum(1 for res in results for step in res.agent_steps if step.repeated)
    # Насколько сама модель угадала домен до детерминированной страховки.
    raw_ok = sum(
        1
        for row, res in positives
        if res.raw_llm_answer is not None and res.raw_llm_answer == row["expected"]
    )

    return {
        "n": len(results),
        "labelled": len(labelled),
        "correct": correct,
        "wrong": wrong,
        "missed": missed,
        "positives": len(positives),
        "negatives": len(negatives),
        "false_positives": false_positives,
        "accuracy": correct / len(labelled) if labelled else None,
        "precision": correct / (correct + wrong + false_positives)
        if (correct + wrong + false_positives)
        else None,
        "answered": len(answered),
        "null_rate": 1 - len(answered) / len(results) if results else 0.0,
        "proven": sum(1 for res in results if res.proof in {"inn_on_site", "ogrn_on_site"}),
        "raw_llm_correct": raw_ok,
        "llm_calls": len(llm_calls),
        "llm_failed": sum(1 for call in llm_calls if call.failed),
        "json_repairs": sum(1 for call in llm_calls if call.repaired),
        "json_first_try": (
            sum(1 for call in llm_calls if call.json_ok_first_try) / len(llm_calls) if llm_calls else None
        ),
        "out_tokens": sum(call.completion_tokens or 0 for call in llm_calls),
        "agent_steps_avg": statistics.mean(agent_steps) if agent_steps else None,
        "agent_invalid": invalid,
        "agent_repeated": repeated,
        "time_avg": statistics.mean([res.elapsed_s for res in results]) if results else 0.0,
    }


def fmt(value, spec: str = "") -> str:
    if value is None:
        return "-"
    return format(value, spec) if spec else str(value)


def print_table(summaries: dict[str, dict]) -> None:
    rows = [
        ("верных / размечено", lambda s: f"{s['correct']}/{s['labelled']}"),
        ("ошибочный домен", lambda s: fmt(s["wrong"])),
        ("домен там, где сайта нет", lambda s: fmt(s["false_positives"])),
        ("null при известном сайте", lambda s: fmt(s["missed"])),
        ("precision (по непустым)", lambda s: fmt(s["precision"], ".2f")),
        ("доля null", lambda s: fmt(s["null_rate"], ".2f")),
        ("подтверждено реестровым кодом", lambda s: fmt(s["proven"])),
        ("угадала сама модель (agent)", lambda s: fmt(s["raw_llm_correct"])),
        ("вызовов LLM", lambda s: fmt(s["llm_calls"])),
        ("сбоев LLM", lambda s: fmt(s["llm_failed"])),
        ("валидный JSON с 1-й попытки", lambda s: fmt(s["json_first_try"], ".2f")),
        ("починок JSON", lambda s: fmt(s["json_repairs"])),
        ("шагов агента (среднее)", lambda s: fmt(s["agent_steps_avg"], ".1f")),
        ("невалидных действий", lambda s: fmt(s["agent_invalid"])),
        ("повторов действий", lambda s: fmt(s["agent_repeated"])),
        ("токенов на выходе", lambda s: fmt(s["out_tokens"])),
        ("секунд на ИНН", lambda s: fmt(s["time_avg"], ".1f")),
    ]
    names = list(summaries)
    width = max(len(name) for name, _ in rows) + 2
    header = "метрика".ljust(width) + "".join(name.rjust(18) for name in names)
    print("\n" + header)
    print("-" * len(header))
    for label, getter in rows:
        print(label.ljust(width) + "".join(getter(summaries[name]).rjust(18) for name in names))


def print_by_tag(rows: list[dict], dump: dict[str, list]) -> None:
    """Точность по классам случаев. Средняя цифра прячет то, что интересно: где именно ломается."""
    tags = sorted({tag for row in rows if "expected" in row for tag in row.get("tags", [])})
    if not tags:
        return
    names = list(dump)
    width = max((len(t) for t in tags), default=10) + 2
    print("\n\nточность по классам случаев")
    print("класс".ljust(width) + "".join(name.rjust(18) for name in names))
    print("-" * (width + 18 * len(names)))
    for tag in tags:
        cells = []
        for name in names:
            pairs = [
                (row, res)
                for row, res in zip(rows, dump[name], strict=True)
                if "expected" in row and tag in row.get("tags", [])
            ]
            ok = sum(1 for row, res in pairs if is_correct(row, Result(**res)))
            cells.append(f"{ok}/{len(pairs)}".rjust(18))
        print(tag.ljust(width) + "".join(cells))


async def main_async(args) -> None:
    rows = json.loads(Path(args.dataset).read_text("utf-8"))
    if args.limit:
        rows = rows[: args.limit]
    config = PipelineConfig(use_cache=not args.no_cache)

    summaries: dict[str, dict] = {}
    dump: dict[str, list] = {}
    for spec in args.config:
        provider, _, mode = spec.partition(":")
        mode = mode or "graph"
        print(f"\n>>> конфигурация {provider}:{mode} -- {len(rows)} ИНН")
        results = await run_config(provider, mode, rows, config)
        for row, res in zip(rows, results, strict=True):
            mark = "?" if "expected" not in row else ("OK " if is_correct(row, res) else "БЕДА")
            proof = f" [{res.proof}]" if res.proof else ""
            print(f"  {mark} {row['inn']} -> {res.domain!s:<28}{proof} ({res.elapsed_s}s) {res.reason[:70]}")
        summaries[f"{provider}:{mode}"] = summarize(rows, results)
        dump[f"{provider}:{mode}"] = [res.model_dump(exclude_none=True) for res in results]

    print_table(summaries)
    print_by_tag(rows, dump)

    unlabelled = [
        (row["inn"], res["domain"])
        for key, results in dump.items()
        for row, res in zip(rows, results, strict=True)
        if "expected" not in row and res.get("proof") in {"inn_on_site", "ogrn_on_site"}
        and res.get("domain")
    ]
    if unlabelled:
        print("\nНайдено с доказательством для неразмеченных ИНН (можно перенести в dataset.json):")
        for inn, domain in dict(unlabelled).items():
            print(f'  {{"inn": "{inn}", "expected": "{domain}"}}')

    if args.out:
        Path(args.out).write_text(json.dumps({"summaries": summaries, "runs": dump}, ensure_ascii=False, indent=2), "utf-8")
        print(f"\nподробности: {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнение конфигураций пайплайна")
    parser.add_argument("--config", action="append", default=None,
                        help="провайдер:режим, например local:agent. Можно указывать несколько раз")
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--out", default="eval_results/last.json")
    args = parser.parse_args()
    args.config = args.config or ["local:graph"]
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
