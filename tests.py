"""Тесты чистых функций: сеть и LLM не нужны. Запуск: .venv/bin/python tests.py"""
from inn_finder.domains import (
    brand_variants,
    is_blocked,
    looks_like_registry_hit,
    mine_domains,
    registrable,
)
from inn_finder.inn import is_entrepreneur, is_valid_inn
from inn_finder.llm import extract_json
from inn_finder.models import Candidate, OrgCard, SearchHit
from inn_finder.pipeline import collect_candidates, resolve_choice
from inn_finder.registry import _extract_by_regex, relevant_hits
from inn_finder.verify import contains_code, published_inns, score_candidate

CARD = OrgCard(inn="7721581040", ogrn="5077746329876", name_short='ООО «Дейта Кью»',
               name_full='ОБЩЕСТВО С ОГРАНИЧЕННОЙ ОТВЕТСТВЕННОСТЬЮ "ДЕЙТА КЬЮ"', city="Москва")
checks: list[tuple[str, bool]] = []


def check(name: str, condition: bool) -> None:
    checks.append((name, bool(condition)))


# --- ИНН
check("валидный ИНН юрлица", is_valid_inn("7721581040"))
check("битая контрольная сумма", not is_valid_inn("7721581041"))
check("валидный ИНН ИП", is_valid_inn("500100732259"))
check("ИП определяется по длине", is_entrepreneur("500100732259") and not is_entrepreneur("7721581040"))
check("мусор отсекается", not is_valid_inn("abc") and not is_valid_inn("123"))

# --- домены
check("eTLD+1 из URL", registrable("https://www.dadata.ru/api/x") == "dadata.ru")
check("поддомен сворачивается", registrable("http://a.b.example.co.uk/x") == "example.co.uk")
check("мусор -> None", registrable("не url") is None)
check("чёрный список", is_blocked("rusprofile.ru") and is_blocked("hh.ru") and not is_blocked("dadata.ru"))

# --- добыча доменов из сниппетов
check("поле Сайт", mine_domains("Сайт: https://dadata.ru ·") == [("dadata.ru", "site_field")])
check("корпоративная почта", ("dadata.ru", "email") in mine_domains("почта info@dadata.ru"))
check("бесплатная почта игнорируется", mine_domains("пишите на ivan@mail.ru") == [])
check("упоминание в тексте", ("dadata.ru", "mention") in mine_domains("торговая марка DADATA.RU"))

# --- определение справочных страниц
V = brand_variants('ООО «Дейта Кью»', "")
check("ИНН в URL -> справочник", looks_like_registry_hit("https://fbc.ru/inn/7721581040/", "", "7721581040", None, V))
check("ОГРН в URL -> справочник",
      looks_like_registry_hit("https://x.ru/c/5077746329876", "", "7721581040", "5077746329876", V))
check("сайт компании справочником не считается",
      not looks_like_registry_hit("https://dadata.ru/", "DaData.ru — подсказки", "7721581040", None, V))
SBER_V = brand_variants("ПАО «Сбербанк»", "")
check("страница 'Реквизиты' на своём сайте не справочник",
      not looks_like_registry_hit("https://sberbank.ru/rekvizity", "Реквизиты ПАО Сбербанк ИНН 7707083893",
                                  "7707083893", None, SBER_V))

# --- поиск кода на странице
check("ИНН как отдельное число", contains_code("ИНН 7721581040, КПП", "7721581040"))
check("ИНН внутри длинного числа не считается", not contains_code("12377215810401", "7721581040"))
check("неразрывные пробелы", contains_code("ИНН 7721581040", "7721581040"))

# --- разбор JSON из ответа модели
check("json в markdown-заборе", extract_json('текст ```json {"a": 1} ``` хвост') == {"a": 1})
check("скобка внутри строки", extract_json('{"d": "a.ru", "s": "}"}')["s"] == "}")
check("нет json", extract_json("просто текст") is None)

# --- сбор кандидатов
HITS = [
    SearchHit(query="q", title='ООО "ДЕЙТА КЬЮ" ИНН 7721581040', url="https://fbc.ru/inn/7721581040/",
              snippet="Телефон +7 495 220-19-62 почта info@dadata.ru", rank=1),
    SearchHit(query="q", title="DaData.ru", url="https://dadata.ru/", snippet="API подсказок", rank=2),
]
CANDS = collect_candidates(HITS, "7721581040", "5077746329876", V)
check("сайт добыт из сниппета справочника", "dadata.ru" in CANDS)
check("справочник помечен requires_proof", CANDS["fbc.ru"].requires_proof)
check("корпоративная почта засчитана", CANDS["dadata.ru"].email_hits == 1)

# --- правила скоринга
proven = Candidate(domain="x.ru", inn_on_site=True, reachable=True)
score_candidate(proven, CARD, V)
check("ИНН на сайте даёт высокий скор", proven.score >= 100 and proven.has_proof)

name_only = Candidate(domain="deytakyu.ru", reachable=True, mention_count=3)
score_candidate(name_only, CARD, V)
check("одно имя -- не улика уровня юрлица", not name_only.has_entity_evidence)

blocked = Candidate(domain="ozon.ru", requires_proof=True, reachable=True, mention_count=5)
score_candidate(blocked, CARD, V)
check("чёрный список без доказательства обнулён", blocked.score == -100)

blocked_proven = Candidate(domain="ozon.ru", requires_proof=True, reachable=True, inn_on_site=True)
score_candidate(blocked_proven, CARD, V)
check("чёрный список с доказательством допускается", blocked_proven.score >= 100)

# Домен, совпадающий с брендом юрлица, -- это его собственный сайт, даже если он в чёрном списке.
YANDEX = OrgCard(inn="7736207543", name_short='ООО "Яндекс"')
YV = brand_variants('ООО "Яндекс"', "")
own = Candidate(domain="yandex.ru", requires_proof=True, reachable=True, name_similarity=1.0)
score_candidate(own, YANDEX, YV)
check("свой бренд снимает запрет чёрного списка",
      not own.blacklisted_without_proof and own.score > 0)
foreign = Candidate(domain="ozon.ru", requires_proof=True, reachable=True, mention_count=3)
score_candidate(foreign, YANDEX, YV)
check("чужая площадка остаётся под запретом", foreign.blacklisted_without_proof)

# --- карточка организации восстанавливается только из релевантных сниппетов
MIXED = [
    SearchHit(query="q", title='ООО "ЧУЖАЯ" ИНН 7804582055', snippet="ОГРН 1027804582055", rank=1),
    SearchHit(query="q", title='ООО "НАША" ИНН 7721581040', snippet="ОГРН 5077746329876", rank=2),
]
check("сниппеты про чужой ИНН отбрасываются", len(relevant_hits("7721581040", MIXED)) == 1)
CARD_MIXED = _extract_by_regex("7721581040", MIXED)
check("название берётся из своего сниппета", CARD_MIXED.name_short == 'ООО «НАША»'
      or "НАША" in (CARD_MIXED.name_short or ""))
check("чужой ОГРН не подхватывается", CARD_MIXED.ogrn == "5077746329876")
# Длинный текст справочника: наш ИНН у одной компании, рядом перечислены другие.
CROWDED = [SearchHit(
    query="q",
    title="Рейтинг банков",
    snippet=("АО «Альфа-Банк» — крупный частный банк, ИНН 7728168971. "
             + "прочий текст " * 20
             + "ПАО Сбербанк, ИНН 7707083893, ОГРН 1027700132195, Москва."),
    rank=1)]
CROWDED_CARD = _extract_by_regex("7707083893", CROWDED)
check("имя берётся рядом с ИНН, а не первое в тексте",
      "Сбербанк" in (CROWDED_CARD.name_short or "") and "Альфа" not in (CROWDED_CARD.name_short or ""))

IP_HITS = [
    SearchHit(query="q", title="ИП Мясина Елена Анатольевна (ИНН 500100732259)",
              snippet="ОГРНИП 304500116000157, ИНН 500100732259, Московская область", rank=1),
    SearchHit(query="q", title="Правовая информация",
              snippet='Учредитель: ООО "Мультисервисные системы". ИНН 500100732259', rank=2),
]
IP_CARD = _extract_by_regex("500100732259", IP_HITS)
check("у ИП берётся имя человека, а не соседнее юрлицо", IP_CARD.name_short == "ИП Мясина Елена Анатольевна")
check("ОГРНИП распознан", IP_CARD.ogrn == "304500116000157")

EMPTY = _extract_by_regex("7721581040", MIXED[:1])
check("нет своих сниппетов -> пустая карточка, а не чужая", EMPTY.name_short is None)

# --- чужой ИНН на сайте
check("ИНН рядом со словом ИНН находится", published_inns("ИНН: 7804582055, КПП 780401001") == {"7804582055"})
check("телефон не принимается за ИНН", published_inns("тел. 8 812 4263626") == set())
foreign = Candidate(domain="ms-s.pro", reachable=True, foreign_inn="7804582055", mention_count=3)
score_candidate(foreign, CARD, V)
check("чужой ИНН штрафует кандидата", foreign.score < 0)
check("чужой ИНН попадает в улики для модели",
      any("DIFFERENT company" in line for line in foreign.evidence_lines()))

# Сайт группы компаний: рядом с нужным юрлицом законно стоят чужие ИНН (случай dadata.ru).
both = Candidate(domain="dadata.ru", reachable=True, inn_on_site=True, foreign_inn="7725861463")
score_candidate(both, CARD, V)
check("свой ИНН перевешивает соседний чужой", both.score >= 100 and both.has_proof)
check("чужой ИНН не показывается модели, если свой найден",
      not any("DIFFERENT company" in line for line in both.evidence_lines()))

# --- сопоставление ответа модели со списком кандидатов
CANDS_LIST = [Candidate(domain="dadata.ru"), Candidate(domain="dp.ru")]
check("выбор по домену", resolve_choice("dadata.ru", CANDS_LIST).domain == "dadata.ru")
check("выбор по URL", resolve_choice("https://www.dadata.ru/", CANDS_LIST).domain == "dadata.ru")
check("выбор по номеру пункта", resolve_choice("2", CANDS_LIST).domain == "dp.ru")
check("null во всех написаниях",
      all(resolve_choice(v, CANDS_LIST) is None for v in (None, "null", "none", "", "-")))
check("домен вне списка не принимается", resolve_choice("вымышленный.ru", CANDS_LIST) is None)

failed = [name for name, ok in checks if not ok]
print("\n".join(f"  {'OK  ' if ok else 'FAIL'} {name}" for name, ok in checks))
print(f"\n{len(checks) - len(failed)}/{len(checks)} тестов пройдено")
raise SystemExit(1 if failed else 0)
