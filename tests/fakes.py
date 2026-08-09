"""Small deterministic fakes shared by adapter tests."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

FIXTURES = Path(__file__).parent / "fixtures"


class FakeTauricRunner:
    """Return a copied mapping without importing or invoking TradingAgents."""

    def __init__(self, result: Mapping[str, Any] | BaseException) -> None:
        self._result = result

    @classmethod
    def from_fixture(cls, filename: str) -> Self:
        payload = json.loads((FIXTURES / filename).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("Tauric fixture must contain a JSON object")
        return cls(payload)

    @classmethod
    def from_mapping(cls, result: Mapping[str, Any]) -> Self:
        return cls(result)

    @classmethod
    def with_future_evidence(cls) -> Self:
        runner = cls.from_fixture("tauric_decision.json")
        assert isinstance(runner._result, Mapping)
        payload = copy.deepcopy(dict(runner._result))
        payload["evidence"][0]["published_at"] = "2026-08-08T20:00:01Z"
        return cls(payload)

    def propagate(self, symbol: str, date: str) -> Mapping[str, Any]:
        del symbol, date
        if isinstance(self._result, BaseException):
            raise self._result
        return copy.deepcopy(dict(self._result))
