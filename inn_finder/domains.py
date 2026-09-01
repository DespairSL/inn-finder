"""Нормализация доменов, чёрные списки и матчинг названия организации с доменом."""
from __future__ import annotations

import difflib
import re
from urllib.parse import urlparse

import tldextract

# Берём встроенный снапшот списка публичных суффиксов: без сетевого похода при старте.
_extract = tldextract.TLDExtract(suffix_list_urls=(), fallback_to_snapshot=True)

# Агрегаторы реквизитов. Ответом быть не могут, но их сниппеты -- ценный источник:
# они часто прямо публикуют поле "Сайт" искомой организации.
AGGREGATORS = {
    "rusprofile.ru", "checko.ru", "list-org.com", "zachestnyibiznes.ru", "sbis.ru", "saby.ru",
    "audit-it.ru", "kartoteka.ru", "spark-interfax.ru", "sparkinterfax.ru", "seldon.ru",
    "e-ecolog.ru", "synapsenet.ru", "vypiska-nalog.com", "fbc.ru", "b2bpoisk.ru", "outstaff.ru",
    "vbankcenter.ru", "k-agent.ru", "companies.rbc.ru", "sbis.com", "delovoy-profil.ru",
    "egrul.nalog.ru", "nalog.ru", "nalog.gov.ru", "bo.nalog.ru", "pb.nalog.ru", "fedresurs.ru",
    "rusprofile.com", "gks.ru", "rosstat.gov.ru", "damia.ru", "ofdata.ru", "kontur.ru",
    "focus.kontur.ru", "zakupki.gov.ru", "clarspb.ru", "primecompany.ru", "sudact.ru",
    "casebook.ru", "arbitr.ru", "kad.arbitr.ru", "reformagkh.ru", "myseldon.com",
}

# Сюда ответ не может уехать ни при каких уликах.
NEVER_ANSWER = AGGREGATORS | {
    "vk.com", "vk.ru", "ok.ru", "facebook.com", "instagram.com", "twitter.com", "x.com",
    "t.me", "telegram.me", "youtube.com", "rutube.ru", "dzen.ru", "zen.yandex.ru",
    "linkedin.com", "pinterest.com", "tiktok.com", "wa.me", "whatsapp.com",
    "hh.ru", "superjob.ru", "rabota.ru", "zarplata.ru", "avito.ru", "youla.ru",
    "ozon.ru", "wildberries.ru", "market.yandex.ru", "aliexpress.ru", "sbermegamarket.ru",
    "yandex.ru", "ya.ru", "google.com", "google.ru", "mail.ru", "bing.com", "duckduckgo.com",
    "2gis.ru", "yell.ru", "zoon.ru", "flamp.ru", "orgpage.ru", "spr.ru", "bizly.ru",
    "wikipedia.org", "ru.wikipedia.org", "wikimapia.org", "livejournal.com",
    "github.com", "gitlab.com", "medium.com", "habr.com", "vc.ru", "rbc.ru", "kommersant.ru",
    "interfax.ru", "tass.ru", "ria.ru", "forbes.ru", "cnews.ru", "cbr.ru", "gosuslugi.ru",
    "archive.org", "web.archive.org", "reg.ru", "nic.ru", "timeweb.com", "beget.com",
    "tilda.cc", "wixsite.com", "narod.ru", "ucoz.ru", "blogspot.com", "wordpress.com",
}

# Формы собственности и служебные слова, которые не несут бренда.
_LEGAL_FORMS = re.compile(
    r"\b(ООО|ОАО|ЗАО|ПАО|АО|НАО|ИП|НКО|ФГУП|ГУП|МУП|АНО|НОУ|ЧОУ|ФГБУ|ФГБОУ|ГБУ|МБУ|ТСЖ|НП|СНТ|"
    r"ОБЩЕСТВО|ОГРАНИЧЕННОЙ|ОТВЕТСТВЕННОСТЬЮ|АКЦИОНЕРНОЕ|ПУБЛИЧНОЕ|НЕПУБЛИЧНОЕ|ЗАКРЫТОЕ|ОТКРЫТОЕ|"
    r"КОМПАНИЯ|ФИРМА|ГРУППА|ХОЛДИНГ|ТОРГОВЫЙ|ДОМ|ЦЕНТР|С|LLC|LTD|INC|JSC|CO)\b",
    re.IGNORECASE,
)

_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z",
    "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
    "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}
_TRANSLIT_ALT = {**_TRANSLIT, "х": "kh", "ц": "ts", "й": "i", "ю": "ju", "я": "ja"}

_DOMAIN_RE = re.compile(
    r"\b((?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:ru|рф|com|net|org|su|io|pro|info|biz|"
    r"moscow|store|online|site|shop|tech|group|company|team|dev|app|me|cc|ai|by|kz))\b",
    re.IGNORECASE,
)


def registrable(url_or_domain: str) -> str | None:
    """URL -> регистрируемый домен (eTLD+1), в нижнем регистре, без www."""
    if not url_or_domain:
        return None
    value = url_or_domain.strip()
    if "://" not in value:
        value = "http://" + value
    host = urlparse(value).hostname or ""
    parts = _extract(host)
    if not parts.domain or not parts.suffix:
        return None
    return f"{parts.domain}.{parts.suffix}".lower()


def is_aggregator(domain: str) -> bool:
    return domain in AGGREGATORS


def is_blocked(domain: str) -> bool:
    return domain in NEVER_ANSWER


def transliterate(text: str, alt: bool = False) -> str:
    table = _TRANSLIT_ALT if alt else _TRANSLIT
    return "".join(table.get(ch, ch) for ch in text.lower())


def clean_name(name: str) -> str:
    """Убирает форму собственности и кавычки, оставляя бренд."""
    without_form = _LEGAL_FORMS.sub(" ", name or "")
    return re.sub(r"[\"«»'`()]+", " ", without_form).strip()


def brand_tokens(name: str) -> list[str]:
    cleaned = clean_name(name)
    tokens = [t for t in re.split(r"[\s\-_,.]+", cleaned) if len(t) > 2]
    return tokens[:6]


def _spelling_variants(value: str) -> list[str]:
    """Устоявшиеся написания, которых не даёт побуквенная транслитерация.

    Яндекс -> yandex, а не yandeks; Феникс -> phoenix/fenix. Без этого домен компании
    не опознаётся как её собственный бренд.
    """
    out = [value]
    if "ks" in value:
        out.append(value.replace("ks", "x"))
    if "iy" in value:
        out.append(value.replace("iy", "y"))
    return out


def brand_variants(*names: str) -> list[str]:
    """Варианты написания бренда для поисковых запросов и матчинга домена."""
    variants: list[str] = []
    for name in names:
        if not name:
            continue
        for token in brand_tokens(name):
            low = token.lower()
            for base in (low, transliterate(low), transliterate(low, alt=True)):
                for value in _spelling_variants(base):
                    if value and value not in variants:
                        variants.append(value)
        joined = "".join(brand_tokens(name)).lower()
        for base in (joined, transliterate(joined), transliterate(joined, alt=True)):
            for value in _spelling_variants(base):
                if len(value) > 3 and value not in variants:
                    variants.append(value)
    return variants


def brand_in_domain(domain: str, variants: list[str]) -> bool:
    label = domain.split(".")[0].replace("-", "")
    if len(label) < 4:  # ya.ru не является совпадением с брендом "Яндекс"
        return False
    for variant in variants:
        cleaned = variant.replace("-", "")
        if len(cleaned) < 4:
            continue
        if cleaned in label or label in cleaned:
            return True
    return False


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def name_similarity(domain: str, names: list[str]) -> float:
    """Насколько доменное имя похоже на бренд (с учётом транслитерации)."""
    label = domain.split(".")[0].replace("-", "")
    best = 0.0
    for name in names:
        if not name:
            continue
        joined = "".join(brand_tokens(name)).lower()
        for base in (joined, transliterate(joined), transliterate(joined, alt=True)):
            for candidate in _spelling_variants(base):
                best = max(best, similarity(label, candidate))
    return best


def extract_domains_from_text(text: str) -> list[str]:
    """Вытаскивает домены из сниппета агрегатора (там часто прямо указан сайт компании)."""
    found: list[str] = []
    for match in _DOMAIN_RE.finditer(text or ""):
        domain = registrable(match.group(1))
        if domain and domain not in found:
            found.append(domain)
    return found


# --------------------------------------------------------------------------- добыча доменов из сниппетов

# Домен официального сайта часто не является ссылкой в выдаче, а лежит текстом внутри сниппета
# справочника: "Сайт: https://dadata.ru", "info@dadata.ru", "торговая марка DADATA.RU".
_SITE_FIELD_RE = re.compile(
    r"(?:веб[-\s]?сайт|официальный\s+сайт|сайт компании|сайт|website|home\s?page)\s*[:\-–—]?\s+"
    r"((?:https?://)?(?:www\.)?[a-z0-9][a-z0-9.-]*\.[a-z]{2,})",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"[a-z0-9._%+-]+@([a-z0-9.-]+\.[a-z]{2,})", re.IGNORECASE)

# Почтовые сервисы: адрес на них о сайте организации ничего не говорит.
FREE_MAIL = {
    "mail.ru", "yandex.ru", "ya.ru", "gmail.com", "bk.ru", "list.ru", "inbox.ru", "internet.ru",
    "rambler.ru", "outlook.com", "hotmail.com", "icloud.com", "yahoo.com", "protonmail.com",
    "vk.com", "mail.com", "narod.ru", "bigmir.net", "ukr.net",
}

# Путь страницы справочника почти всегда содержит эти маркеры вместе с ИНН/ОГРН.
_REGISTRY_TITLE_MARKERS = ("инн", "огрн", "реквизит", "выписка", "егрюл", "контрагент", "проверка контрагента")


def mine_domains(text: str) -> list[tuple[str, str]]:
    """Домены из текста сниппета с указанием, чем именно они подтверждены.

    Возвращает пары (домен, вид): site_field -- явное поле "Сайт", email -- корпоративная
    почта, mention -- просто упоминание в тексте.
    """
    found: dict[str, str] = {}
    priority = {"site_field": 3, "email": 2, "mention": 1}

    def add(raw: str, kind: str) -> None:
        domain = registrable(raw)
        if not domain or is_blocked(domain):
            return
        if kind == "email" and domain in FREE_MAIL:
            return
        if priority[kind] > priority.get(found.get(domain, ""), 0):
            found[domain] = kind

    for match in _SITE_FIELD_RE.finditer(text or ""):
        add(match.group(1), "site_field")
    for match in _EMAIL_RE.finditer(text or ""):
        add(match.group(1), "email")
    for domain in extract_domains_from_text(text):
        add(domain, "mention")
    return list(found.items())


def looks_like_registry_hit(
    url: str, title: str, inn: str, ogrn: str | None, variants: list[str] | None = None
) -> bool:
    """Страница справочника про эту организацию: ответом быть не может, но её сниппет ценен.

    Определяется структурно, а не списком доменов -- списка на все справочники рунета не хватит.
    Признак сильный: реквизиты организации в самом URL. Одного слова "реквизиты" в заголовке
    мало: у компаний есть собственные страницы с реквизитами, и терять их нельзя.
    """
    low_url = (url or "").lower()
    if inn and inn in low_url:
        return True
    if ogrn and ogrn in low_url:
        return True

    low_title = (title or "").lower()
    marker_in_title = any(marker in low_title for marker in _REGISTRY_TITLE_MARKERS)
    if not (marker_in_title and inn and inn in low_title):
        return False

    # Домен, совпадающий с брендом, -- это сайт самой компании, как бы ни выглядел заголовок.
    domain = registrable(url)
    return not (domain and variants and brand_in_domain(domain, variants))
