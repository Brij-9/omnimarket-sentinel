from __future__ import annotations

import copy
import importlib
import re
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from market_sentinel.research.base import ResearchProvider
from market_sentinel.research.tauric import (
    PINNED_TRADINGAGENTS_COMMIT,
    ResearchUnavailable,
    TauricResearchProvider,
    TauricUpstreamRunner,
)
from tests.factories import instrument
from tests.fakes import FakeTauricRunner

AS_OF = datetime(2026, 8, 8, 20, tzinfo=UTC)
INSTALL_COMMAND = 'python -m pip install -e ".[research]"'


def _payload(**updates: Any) -> dict[str, Any]:
    runner = FakeTauricRunner.from_fixture("tauric_decision.json")
    result = runner.propagate("AAPL", "2026-08-08")
    payload = copy.deepcopy(dict(result))
    payload.update(updates)
    return payload


def _provider(payload: dict[str, Any]) -> TauricResearchProvider:
    return TauricResearchProvider(
        runner=FakeTauricRunner.from_mapping(payload), prompt_version="tauric-v1"
    )


def test_tauric_decision_becomes_timestamped_research_packet() -> None:
    provider: ResearchProvider = TauricResearchProvider(
        runner=FakeTauricRunner.from_fixture("tauric_decision.json"),
        prompt_version="tauric-v1",
    )

    packet = provider.analyze(instrument(symbol="AAPL"), AS_OF)

    assert packet.instrument_id == "AAPL@alpaca"
    assert packet.as_of == AS_OF
    assert packet.thesis.startswith("Durable services growth")
    assert packet.bear_case.startswith("Premium valuation")
    assert packet.catalysts == ("Next quarterly earnings release", "New product cycle")
    assert packet.risks == ("Multiple compression", "Regulatory pressure")
    assert packet.confidence == Decimal("0.72")
    assert packet.model_id == "fixture-model"
    assert packet.prompt_version == "tauric-v1"
    assert re.fullmatch(r"[0-9a-f]{64}", packet.configuration_hash)
    assert packet.evidence
    assert all(item.published_at <= AS_OF for item in packet.evidence)


def test_aware_as_of_is_normalized_to_utc() -> None:
    offset_as_of = AS_OF.astimezone(timezone(timedelta(hours=5, minutes=30)))
    packet = _provider(_payload()).analyze(instrument(), offset_as_of)
    assert packet.as_of == AS_OF


def test_naive_as_of_is_rejected_before_research_runs() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _provider(_payload()).analyze(instrument(), datetime(2026, 8, 8, 20))


def test_future_dated_evidence_is_rejected() -> None:
    provider = TauricResearchProvider(
        runner=FakeTauricRunner.with_future_evidence(), prompt_version="tauric-v1"
    )
    with pytest.raises(ValueError, match="look-ahead evidence"):
        provider.analyze(instrument(symbol="AAPL"), AS_OF)


@pytest.mark.parametrize(
    ("confidence", "scale", "expected"),
    [("0.375", "unit_interval", Decimal("0.375")), (37.5, "percent", Decimal("0.375"))],
)
def test_supported_confidence_scales_are_explicitly_normalized(
    confidence: str | float, scale: str, expected: Decimal
) -> None:
    packet = _provider(_payload(confidence=confidence, confidence_scale=scale)).analyze(
        instrument(), AS_OF
    )
    assert packet.confidence == expected


@pytest.mark.parametrize(
    ("confidence", "scale"),
    [
        (0.5, None),
        (50, "basis_points"),
        (float("nan"), "unit_interval"),
        (float("inf"), "unit_interval"),
        (-0.1, "unit_interval"),
        (101, "percent"),
        (True, "unit_interval"),
    ],
)
def test_malformed_or_ambiguous_confidence_is_rejected(
    confidence: Any, scale: str | None
) -> None:
    payload = _payload(confidence=confidence)
    if scale is None:
        payload.pop("confidence_scale")
    else:
        payload["confidence_scale"] = scale
    with pytest.raises(ValueError, match="confidence"):
        _provider(payload).analyze(instrument(), AS_OF)


def test_decision_requires_a_thesis_or_bear_case() -> None:
    with pytest.raises(ValueError, match="thesis or bear case"):
        _provider(_payload(thesis="  ", bear_case="")).analyze(instrument(), AS_OF)


@pytest.mark.parametrize(
    "evidence",
    [
        [],
        [{"uri": "https://example.test/item", "title": "Source"}],
        [
            {
                "uri": "javascript:alert(1)",
                "title": "Source",
                "published_at": "2026-08-08T19:00:00Z",
            }
        ],
        [
            {
                "uri": "https://example.test/item",
                "title": "  ",
                "published_at": "2026-08-08T19:00:00Z",
            }
        ],
        [
            {
                "uri": "https://example.test/item",
                "title": "Source",
                "published_at": "2026-08-08T19:00:00",
            }
        ],
    ],
)
def test_evidence_requires_valid_uri_title_and_source_time(evidence: list[dict[str, str]]) -> None:
    with pytest.raises(ValueError, match="evidence"):
        _provider(_payload(evidence=evidence)).analyze(instrument(), AS_OF)


def test_duplicate_evidence_uri_is_removed_deterministically() -> None:
    evidence = _payload()["evidence"]
    evidence.append(
        {
            "uri": "https://EXAMPLE.test/aapl/products#different-fragment",
            "title": "Duplicate product announcement",
            "published_at": "2026-08-08T19:16:00Z",
        }
    )
    packet = _provider(_payload(evidence=evidence)).analyze(instrument(), AS_OF)
    assert tuple(item.uri for item in packet.evidence) == (
        "https://example.test/aapl/earnings",
        "https://example.test/aapl/products",
    )


def test_configuration_hash_is_stable_and_excludes_secret_fields() -> None:
    first = _payload()
    first["configuration"]["api_key"] = "secret-value-one"
    second = _payload()
    second["configuration"]["api_key"] = "secret-value-two"

    first_packet = _provider(first).analyze(instrument(), AS_OF)
    second_packet = _provider(second).analyze(instrument(), AS_OF)

    assert first_packet.configuration_hash == second_packet.configuration_hash
    assert "secret-value" not in repr(first_packet)


def test_order_or_sizing_fields_are_rejected_without_echoing_values() -> None:
    token = "do-not-serialize-this-secret"
    with pytest.raises(ValueError, match="cannot contain trading instructions") as error:
        _provider(_payload(position_size=token)).analyze(instrument(), AS_OF)
    assert token not in str(error.value)


@pytest.mark.parametrize("failure", [TimeoutError("token-123"), RuntimeError("token-123")])
def test_upstream_failures_are_closed_without_leaking_error_details(failure: BaseException) -> None:
    provider = TauricResearchProvider(
        runner=FakeTauricRunner(failure), prompt_version="tauric-v1"
    )
    with pytest.raises(ResearchUnavailable, match="failed closed") as error:
        provider.analyze(instrument(), AS_OF)
    assert "token-123" not in str(error.value)


def test_lazy_runner_import_failure_has_exact_install_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = TauricUpstreamRunner(
        llm_provider="ollama", model_id="qwen3:latest", checkpoint_enabled=False
    )

    def unavailable(_: str) -> Any:
        raise ModuleNotFoundError("No module named 'tradingagents'")

    monkeypatch.setattr(importlib, "import_module", unavailable)
    with pytest.raises(ResearchUnavailable, match=re.escape(INSTALL_COMMAND)):
        runner.propagate("AAPL", "2026-08-08")


def test_lazy_runner_copies_config_and_sanitizes_pinned_upstream_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default_config: dict[str, Any] = {
        "llm_provider": "openai",
        "deep_think_llm": "default-deep",
        "quick_think_llm": "default-quick",
        "checkpoint_enabled": True,
        "nested": {"unchanged": True},
        "api_key": "must-not-escape",
    }
    captured: dict[str, Any] = {}

    class FakeGraph:
        def __init__(self, selected_analysts: tuple[str, ...], config: dict[str, Any]) -> None:
            captured["selected_analysts"] = selected_analysts
            captured["config"] = config
            config["nested"]["unchanged"] = False

        def propagate(self, symbol: str, date: str) -> tuple[dict[str, Any], str]:
            captured["call"] = (symbol, date)
            research = _payload()
            research["untrusted_extra"] = object()
            return {"research_packet": research, "messages": [object()]}, "BUY"

    graph_module = SimpleNamespace(TradingAgentsGraph=FakeGraph)
    config_module = SimpleNamespace(DEFAULT_CONFIG=default_config)

    def import_fake(name: str) -> Any:
        modules = {
            "tradingagents.graph.trading_graph": graph_module,
            "tradingagents.default_config": config_module,
        }
        return modules[name]

    monkeypatch.setattr(importlib, "import_module", import_fake)
    runner = TauricUpstreamRunner(
        llm_provider="ollama",
        model_id="qwen3:latest",
        checkpoint_enabled=False,
        selected_analysts=("market", "news"),
    )
    result = runner.propagate("AAPL", "2026-08-08")

    assert default_config["nested"]["unchanged"] is True
    assert captured["config"]["llm_provider"] == "ollama"
    assert captured["config"]["deep_think_llm"] == "qwen3:latest"
    assert captured["config"]["quick_think_llm"] == "qwen3:latest"
    assert captured["config"]["checkpoint_enabled"] is False
    assert captured["selected_analysts"] == ("market", "news")
    assert captured["call"] == ("AAPL", "2026-08-08")
    assert result["model_id"] == "qwen3:latest"
    assert result["configuration"]["upstream_commit"] == PINNED_TRADINGAGENTS_COMMIT
    assert "untrusted_extra" not in result
    assert "api_key" not in result["configuration"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"llm_provider": "unknown", "model_id": "model", "checkpoint_enabled": False},
        {"llm_provider": "ollama", "model_id": "", "checkpoint_enabled": False},
        {"llm_provider": "ollama", "model_id": "model", "checkpoint_enabled": "false"},
        {
            "llm_provider": "ollama",
            "model_id": "model",
            "checkpoint_enabled": False,
            "selected_analysts": ("trader",),
        },
    ],
)
def test_lazy_runner_rejects_unvalidated_provider_model_and_checkpoint_values(
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="Tauric runner configuration is invalid"):
        TauricUpstreamRunner(**kwargs)
