"""Context-independent bounded encodings for live-safety identities."""

from __future__ import annotations

from decimal import Decimal

_MAX_DECIMAL_DIGITS = 1_024
_MAX_DECIMAL_EXPONENT = 4_096
_MAX_DECIMAL_TEXT = 4_096


class CanonicalEncodingError(ValueError):
    """A value cannot be represented by the bounded exact safety encoding."""


def canonical_decimal(value: object) -> str:
    """Encode an exact finite Decimal without consulting the ambient Decimal context."""
    if type(value) is not Decimal or not value.is_finite():
        raise CanonicalEncodingError("numeric field is malformed")
    sign, raw_digits, exponent = value.as_tuple()
    if type(exponent) is not int:
        raise CanonicalEncodingError("numeric field is malformed")
    if len(raw_digits) > _MAX_DECIMAL_DIGITS or abs(exponent) > _MAX_DECIMAL_EXPONENT:
        raise CanonicalEncodingError("numeric field exceeds bounded encoding")
    if not any(raw_digits):
        return "0"
    digits = list(raw_digits)
    while digits[-1] == 0:
        digits.pop()
        exponent += 1
    required = len(digits) + max(exponent, 0) + max(-(len(digits) + exponent), 0) + 2
    if required > _MAX_DECIMAL_TEXT or abs(exponent) > _MAX_DECIMAL_EXPONENT:
        raise CanonicalEncodingError("numeric field exceeds bounded encoding")
    body = "".join(str(digit) for digit in digits)
    if exponent >= 0:
        body += "0" * exponent
    else:
        point = len(body) + exponent
        body = body[:point] + "." + body[point:] if point > 0 else "0." + "0" * -point + body
    return ("-" if sign else "") + body
