"""Валидация ИНН по контрольным суммам -- отсекает мусор до первого сетевого запроса."""
from __future__ import annotations

import re

_C10 = (2, 4, 10, 3, 5, 9, 4, 6, 8)
_C11 = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
_C12 = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)


def normalize_inn(raw: str) -> str:
    return re.sub(r"\D", "", str(raw or ""))


def _checksum(digits: list[int], coefficients: tuple[int, ...]) -> int:
    return sum(c * d for c, d in zip(coefficients, digits, strict=True)) % 11 % 10


def is_valid_inn(raw: str) -> bool:
    inn = normalize_inn(raw)
    digits = [int(ch) for ch in inn]
    if len(inn) == 10:  # юридическое лицо
        return _checksum(digits[:9], _C10) == digits[9]
    if len(inn) == 12:  # ИП / физлицо
        return _checksum(digits[:10], _C11) == digits[10] and _checksum(digits[:11], _C12) == digits[11]
    return False


def is_entrepreneur(raw: str) -> bool:
    """12 знаков -> ИП. У ИП сайт бывает заметно реже, это влияет на априорную уверенность."""
    return len(normalize_inn(raw)) == 12
