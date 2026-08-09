"""Small deterministic fakes shared by adapter tests."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

FIXTURES = Path(__file__).parent / "fixtures"

FIXTURE_CONFIGURATION: dict[str, Any] = {
    "profile": "tauric-market-only-audited-v3",
    "upstream_commit": "a33fd4c0f134485a43553a2c23a63cb14adbd88f",
    "llm_provider": "fixture",
    "deep_think_llm": "fixture-model",
    "quick_think_llm": "fixture-model",
    "backend_url": "fixture://offline",
    "google_thinking_level": None,
    "openai_reasoning_effort": None,
    "anthropic_effort": None,
    "temperature": 0.0,
    "llm_max_retries": 0,
    "checkpoint_enabled": False,
    "memory_log_max_entries": 0,
    "output_language": "English",
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    "news_article_limit": 0,
    "global_news_article_limit": 0,
    "global_news_lookback_days": 0,
    "global_news_queries": [],
    "data_vendors": {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance"
    },
    "tool_vendors": {},
    "benchmark_ticker": None,
    "benchmark_map": {"": "SPY"},
    "selected_analysts": ["market"],
    "storage_scope": "isolated-per-run",
    "confidence_method": "unit_interval:evidence=0.50,thesis=0.25,bear_case=0.25",
    "evidence_availability_method": "completed_daily_bar_next_utc_day_v1",
    "analysis_cutoff_method": "previous_utc_date_v1",
    "tool_trace_policy": "pinned_market_tools_complete_ordered_v1",
    "llm_temporal_scope": "not_point_in_time_historical"
}


def tauric_state(filename: str = "tauric_decision.json") -> dict[str, Any]:
    payload = json.loads((FIXTURES / filename).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Tauric fixture must contain a JSON object")
    return payload


def tauric_trace(filename: str = "tauric_market_trace.json") -> list[dict[str, Any]]:
    payload = json.loads((FIXTURES / filename).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise TypeError("Tauric trace fixture must contain a list of JSON objects")
    return payload


class FakeTauricRunner:
    """Return a copied actual-state envelope without importing TradingAgents."""

    def __init__(self, result: Mapping[str, Any] | BaseException) -> None:
        self._result = result

    @classmethod
    def from_fixture(cls, filename: str) -> Self:
        return cls.from_state(tauric_state(filename))

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> Self:
        return cls(
            {
                "state": copy.deepcopy(dict(state)),
                "tool_trace": copy.deepcopy(tauric_trace()),
                "model_id": "fixture-model",
                "configuration": copy.deepcopy(FIXTURE_CONFIGURATION),
            }
        )

    @classmethod
    def from_result(cls, result: Mapping[str, Any]) -> Self:
        return cls(result)

    def propagate(self, symbol: str, date: str) -> Mapping[str, Any]:
        del symbol, date
        if isinstance(self._result, BaseException):
            raise self._result
        return copy.deepcopy(dict(self._result))
