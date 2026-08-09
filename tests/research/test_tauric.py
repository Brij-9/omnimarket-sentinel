from __future__ import annotations

import copy
import json
import multiprocessing
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from market_sentinel.research.base import ResearchProvider
from market_sentinel.research.tauric import (
    INVALID_RESEARCH_STATE_REASON,
    NO_TRUSTWORTHY_EVIDENCE_REASON,
    PINNED_TRADINGAGENTS_COMMIT,
    RESEARCH_INSTALL_COMMAND,
    ResearchDependencyUnavailable,
    ResearchUnavailable,
    TauricResearchProvider,
    TauricUpstreamRunner,
    _canonical_evidence_uri,
    _execute_instrumented_graph,
    _reset_pinned_dataflow_config,
    _ToolTraceRecorder,
)
from tests.factories import instrument
from tests.fakes import (
    FIXTURE_CONFIGURATION,
    FakeTauricRunner,
    tauric_state,
    tauric_trace,
)

AS_OF = datetime(2026, 8, 8, 20, tzinfo=UTC)


def _provider(
    state: dict[str, Any], trace: list[dict[str, Any]] | None = None
) -> TauricResearchProvider:
    result = {
        "state": copy.deepcopy(state),
        "tool_trace": copy.deepcopy(trace if trace is not None else tauric_trace()),
        "model_id": "fixture-model",
        "configuration": copy.deepcopy(FIXTURE_CONFIGURATION),
    }
    return TauricResearchProvider(
        runner=FakeTauricRunner.from_result(result), prompt_version="tauric-v1"
    )


def _return_state_worker(
    request: dict[str, Any], *, state: dict[str, Any], trace: list[dict[str, Any]]
) -> dict[str, Any]:
    del request
    return {"state": copy.deepcopy(state), "tool_trace": copy.deepcopy(trace)}


def _capture_config_worker(
    request: dict[str, Any],
    *,
    state: dict[str, Any],
    trace: list[dict[str, Any]],
    capture_path: str,
) -> dict[str, Any]:
    config = request["config"]
    path_fields = ("project_dir", "results_dir", "data_cache_dir", "memory_log_path")
    captured = {
        "config": config,
        "paths_exist": {key: Path(config[key]).parent.exists() for key in path_fields},
    }
    Path(capture_path).write_text(json.dumps(captured, sort_keys=True), encoding="utf-8")
    return {"state": copy.deepcopy(state), "tool_trace": copy.deepcopy(trace)}


def _dependency_missing_worker(request: dict[str, Any]) -> dict[str, Any]:
    del request
    raise ModuleNotFoundError("arbitrary injected dependency text token-123")


def _arbitrary_failure_worker(request: dict[str, Any]) -> dict[str, Any]:
    del request
    raise RuntimeError("arbitrary injected failure token-123")


def _delayed_side_effect_worker(
    request: dict[str, Any], *, marker_path: str, started: Any
) -> dict[str, Any]:
    del request
    started.set()
    time.sleep(5)
    Path(marker_path).write_text("continued", encoding="utf-8")
    return {"state": {}, "tool_trace": []}


def _runner(
    worker: Any,
    *,
    llm_provider: str = "openai",
    checkpoint_enabled: bool = False,
    timeout_seconds: float = 10,
    worker_started: Any = None,
) -> TauricUpstreamRunner:
    return TauricUpstreamRunner(
        llm_provider=llm_provider,
        model_id="fixture-model",
        checkpoint_enabled=checkpoint_enabled,
        selected_analysts=("market",),
        timeout_seconds=timeout_seconds,
        worker=worker,
        worker_started=worker_started,
    )


class _FakeToolMessage:
    def __init__(self, entry: dict[str, Any]) -> None:
        self.content = entry["content"]
        self.name = entry["name"]
        self.tool_call_id = entry["id"]
        self.status = "success"


class _PinnedApiFakePropagator:
    def __init__(self) -> None:
        self.max_recur_limit = 100

    def get_graph_args(self, callbacks: list[Any] | None = None) -> dict[str, Any]:
        return {"config": {"callbacks": callbacks or []}, "stream_mode": "values"}


class _PinnedApiFakeGraph:
    trace = tauric_trace()
    state = tauric_state()
    omit_last_result = False

    def __init__(self, selected_analysts: tuple[str, ...], config: dict[str, Any]) -> None:
        assert selected_analysts == ("market",)
        self.config = config
        self.propagator = _PinnedApiFakePropagator()

    def resolve_instrument_context(self, ticker: str, asset_type: str = "stock") -> str:
        raise AssertionError(f"live identity lookup was not disabled: {ticker}/{asset_type}")

    def propagate(
        self, symbol: str, cutoff: str, asset_type: str = "stock"
    ) -> tuple[dict[str, Any], str]:
        context = self.resolve_instrument_context(symbol, asset_type)
        assert "AAPL" in context
        callbacks = self.propagator.get_graph_args()["config"]["callbacks"]
        assert len(callbacks) == 1
        callback = callbacks[0]
        for entry in self.trace:
            run_id = uuid4()
            callback.on_tool_start(
                {"name": entry["name"], "description": "pinned tool"},
                json.dumps(entry["args"], sort_keys=True),
                run_id=run_id,
                inputs=entry["args"],
                tool_call_id=entry["id"],
            )
            if not self.omit_last_result or entry is not self.trace[-1]:
                callback.on_tool_end(_FakeToolMessage(entry), run_id=run_id)
        state = copy.deepcopy(self.state)
        state["trade_date"] = cutoff
        state["instrument_context"] = context
        return state, "HOLD"


class _MissingToolResultFakeGraph(_PinnedApiFakeGraph):
    omit_last_result = True


class _MissingPinnedCallbackSeamFakeGraph:
    def __init__(self, selected_analysts: tuple[str, ...], config: dict[str, Any]) -> None:
        del selected_analysts, config

    def resolve_instrument_context(self, ticker: str, asset_type: str = "stock") -> str:
        return f"{ticker}/{asset_type}"


class _RecordingRunner:
    def __init__(self) -> None:
        self.request: tuple[str, str] | None = None

    def propagate(self, symbol: str, date: str) -> dict[str, Any]:
        self.request = (symbol, date)
        return dict(FakeTauricRunner.from_state(tauric_state()).propagate(symbol, date))


def test_actual_agent_state_becomes_sourced_research_packet() -> None:
    provider: ResearchProvider = TauricResearchProvider(
        runner=FakeTauricRunner.from_fixture("tauric_decision.json"),
        prompt_version="tauric-v1",
    )

    packet = provider.analyze(instrument(symbol="AAPL"), AS_OF)

    assert packet.instrument_id == "AAPL@alpaca"
    assert packet.as_of == AS_OF
    assert packet.thesis == "Services growth and margin resilience support the constructive case."
    assert packet.bear_case == "Valuation and demand sensitivity support the cautious case."
    assert packet.catalysts == ()
    assert packet.risks == ()
    assert packet.confidence == Decimal("1")
    assert packet.model_id == "fixture-model"
    assert packet.prompt_version == "tauric-v1"
    assert re.fullmatch(r"[0-9a-f]{64}", packet.configuration_hash)
    assert len(packet.evidence) == 1
    assert packet.evidence[0].uri == "https://finance.yahoo.com/quote/AAPL/history"
    assert packet.evidence[0].published_at == datetime(2026, 8, 8, tzinfo=UTC)


def test_final_state_fixture_has_cleared_messages_and_separate_trace() -> None:
    state = tauric_state()
    assert [message["type"] for message in state["messages"]] == ["human", "ai"]
    assert not any(message.get("tool_calls") for message in state["messages"])
    assert {entry["name"] for entry in tauric_trace()} == {
        "get_stock_data",
        "get_indicators",
        "get_verified_market_snapshot",
    }


def test_exact_pinned_api_shaped_fake_captures_trace_and_disables_identity_lookup() -> None:
    result = _execute_instrumented_graph(
        _PinnedApiFakeGraph,
        {
            "symbol": "AAPL",
            "date": "2026-08-07",
            "selected_analysts": ["market"],
            "config": {"checkpoint_enabled": False},
        },
    )
    assert result["state"]["messages"] == tauric_state()["messages"]
    assert result["tool_trace"] == tauric_trace()
    assert result["state"]["instrument_context"].startswith(
        "The instrument to analyze is `AAPL`"
    )


def test_instrumented_graph_fails_closed_when_started_tool_has_no_result() -> None:
    with pytest.raises(RuntimeError, match="pinned tool trace unavailable"):
        _execute_instrumented_graph(
            _MissingToolResultFakeGraph,
            {
                "symbol": "AAPL",
                "date": "2026-08-07",
                "selected_analysts": ["market"],
                "config": {"checkpoint_enabled": False},
            },
        )


def test_instrumented_graph_fails_closed_if_pinned_callback_seam_changes() -> None:
    with pytest.raises(RuntimeError, match="pinned tool trace unavailable"):
        _execute_instrumented_graph(
            _MissingPinnedCallbackSeamFakeGraph,
            {
                "symbol": "AAPL",
                "date": "2026-08-07",
                "selected_analysts": ["market"],
                "config": {"checkpoint_enabled": False},
            },
        )


def test_trace_recorder_pairs_run_ids_and_preserves_start_order_on_inverted_completion() -> None:
    recorder = _ToolTraceRecorder()
    first, second = tauric_trace()[:2]
    first_run, second_run = uuid4(), uuid4()
    for entry, run_id in ((first, first_run), (second, second_run)):
        recorder.on_tool_start(
            {"name": entry["name"]},
            "{}",
            run_id=run_id,
            inputs=entry["args"],
            tool_call_id=entry["id"],
        )
    recorder.on_tool_end(_FakeToolMessage(second), run_id=second_run)
    recorder.on_tool_end(_FakeToolMessage(first), run_id=first_run)
    assert [entry["id"] for entry in recorder.export()] == [first["id"], second["id"]]


def test_pinned_merge_on_update_config_is_reset_before_audited_graph_setup() -> None:
    module = SimpleNamespace(
        _config={"data_vendors": {"prediction_markets": "polymarket"}},
        set_config=lambda _value: None,
    )
    _reset_pinned_dataflow_config(module)
    assert module._config == {}


def test_pinned_private_config_shape_change_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="pinned tool trace unavailable"):
        _reset_pinned_dataflow_config(SimpleNamespace(set_config=lambda _value: None))


def test_prior_completed_bar_is_available_at_next_utc_day_boundary() -> None:
    boundary = datetime(2026, 8, 8, tzinfo=UTC)
    packet = _provider(tauric_state()).analyze(instrument(), boundary)
    assert packet.evidence[0].published_at == boundary


def test_provider_uses_previous_utc_date_as_graph_analysis_cutoff() -> None:
    runner = _RecordingRunner()
    provider = TauricResearchProvider(runner=runner, prompt_version="tauric-v1")
    provider.analyze(instrument(), AS_OF)
    assert runner.request == ("AAPL", "2026-08-07")


@pytest.mark.parametrize("hour,minute", [(0, 1), (13, 0), (23, 59)])
def test_public_same_day_bar_is_rejected_by_previous_utc_date_cutoff(
    hour: int, minute: int
) -> None:
    trace = tauric_trace()
    trace[0]["args"]["end_date"] = "2026-08-08"
    trace[0]["content"] = trace[0]["content"].replace(
        "to 2026-08-07", "to 2026-08-08"
    ).replace(
        "2026-08-07,220.00,224.00,219.00,223.00,1000",
        "2026-08-08,220.00,224.00,219.00,223.00,1000",
    )
    as_of = datetime(2026, 8, 8, hour, minute, tzinfo=UTC)
    with pytest.raises(ResearchUnavailable):
        _provider(tauric_state(), trace).analyze(instrument(), as_of)


def test_naive_retrieval_clock_is_validated_but_not_used_as_publication_time() -> None:
    trace = tauric_trace()
    trace[0]["content"] = trace[0]["content"].replace(
        "Data retrieved on: 2026-08-08 19:20:00",
        "Data retrieved on: 2026-08-09 23:59:59",
    )
    packet = _provider(tauric_state(), trace).analyze(instrument(), AS_OF)
    assert packet.evidence[0].published_at == datetime(2026, 8, 8, tzinfo=UTC)


def test_trade_fields_and_reports_are_not_converted_to_research_content() -> None:
    state = tauric_state()
    state["investment_debate_state"]["bull_history"] = ""
    state["investment_debate_state"]["bear_history"] = ""
    state["market_report"] = "A fabricated thesis in arbitrary prose."
    state["final_trade_decision"] = "BUY because this contains a bullish thesis."
    with pytest.raises(ResearchUnavailable, match=re.escape(INVALID_RESEARCH_STATE_REASON)):
        _provider(state).analyze(instrument(), AS_OF)


def test_fictional_direct_research_packet_shape_is_rejected() -> None:
    fictional = {
        "thesis": "invented",
        "bear_case": "invented",
        "confidence": 90,
        "confidence_scale": "percent",
        "evidence": [],
        "model_id": "fixture-model",
        "configuration": FIXTURE_CONFIGURATION,
    }
    provider = TauricResearchProvider(
        runner=FakeTauricRunner.from_result(fictional), prompt_version="tauric-v1"
    )
    with pytest.raises(ResearchUnavailable, match=re.escape(INVALID_RESEARCH_STATE_REASON)):
        provider.analyze(instrument(), AS_OF)


def test_confidence_is_evidence_completeness_not_directional_certainty() -> None:
    state = tauric_state()
    state["investment_debate_state"]["bear_history"] = ""
    state["final_trade_decision"] = "STRONG BUY WITH CERTAINTY"
    packet = _provider(state).analyze(instrument(), AS_OF)
    assert packet.confidence == Decimal("0.75")


def test_aware_as_of_is_normalized_to_utc() -> None:
    offset_as_of = AS_OF.astimezone(timezone(timedelta(hours=5, minutes=30)))
    packet = _provider(tauric_state()).analyze(instrument(), offset_as_of)
    assert packet.as_of == AS_OF


def test_naive_as_of_is_rejected_before_research_runs() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _provider(tauric_state()).analyze(instrument(), datetime(2026, 8, 8, 20))


def test_agent_state_trade_date_must_match_requested_utc_date() -> None:
    state = tauric_state()
    state["trade_date"] = "2026-08-08"
    with pytest.raises(ResearchUnavailable, match=re.escape(INVALID_RESEARCH_STATE_REASON)):
        _provider(state).analyze(instrument(), AS_OF)


def test_future_row_in_structured_tool_result_is_rejected() -> None:
    trace = tauric_trace()
    trace[0]["content"] = trace[0]["content"].replace(
        "# Total records: 1", "# Total records: 2"
    ) + (
        "2026-08-08,226.00,227.00,225.00,226.00,1200\n"
    )
    with pytest.raises(ResearchUnavailable, match="look-ahead evidence"):
        _provider(tauric_state(), trace).analyze(instrument(), AS_OF)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("from 2026-08-01 to 2026-08-07", "from 2026-08-01 to 2026-08-06"),
        ("Data retrieved on: 2026-08-08 19:20:00", "Data retrieved on: someday"),
    ],
)
def test_pinned_stock_output_rejects_mismatched_headers_and_ambiguous_rows(
    old: str, new: str
) -> None:
    trace = tauric_trace()
    trace[0]["content"] = trace[0]["content"].replace(old, new)
    with pytest.raises(ResearchUnavailable, match=re.escape(NO_TRUSTWORTHY_EVIDENCE_REASON)):
        _provider(tauric_state(), trace).analyze(instrument(), AS_OF)


def test_duplicate_csv_observation_is_rejected() -> None:
    trace = tauric_trace()
    trace[0]["content"] = trace[0]["content"].replace(
        "# Total records: 1", "# Total records: 2"
    )
    trace[0]["content"] += (
        "2026-08-07,221.00,225.00,220.00,224.00,1001\n"
    )
    with pytest.raises(ResearchUnavailable, match=re.escape(NO_TRUSTWORTHY_EVIDENCE_REASON)):
        _provider(tauric_state(), trace).analyze(instrument(), AS_OF)


def test_observation_after_paired_end_date_is_rejected() -> None:
    trace = tauric_trace()
    trace[0]["args"]["end_date"] = "2026-08-06"
    trace[0]["content"] = trace[0]["content"].replace(
        "to 2026-08-07", "to 2026-08-06"
    )
    with pytest.raises(ResearchUnavailable, match=re.escape(INVALID_RESEARCH_STATE_REASON)):
        _provider(tauric_state(), trace).analyze(instrument(), AS_OF)


def test_pathological_total_record_count_fails_closed_without_raw_parse_error() -> None:
    trace = tauric_trace()
    secret = "7" * 5000
    trace[0]["content"] = trace[0]["content"].replace(
        "# Total records: 1", f"# Total records: {secret}"
    )
    with pytest.raises(ResearchUnavailable, match=re.escape(NO_TRUSTWORTHY_EVIDENCE_REASON)) as exc:
        _provider(tauric_state(), trace).analyze(instrument(), AS_OF)
    assert secret not in str(exc.value)


def test_missing_tool_trace_fails_closed() -> None:
    with pytest.raises(ResearchUnavailable, match=re.escape(INVALID_RESEARCH_STATE_REASON)):
        _provider(tauric_state(), []).analyze(instrument(), AS_OF)


def test_duplicate_trace_id_is_rejected() -> None:
    trace = tauric_trace()
    trace[1]["id"] = trace[0]["id"]
    with pytest.raises(ResearchUnavailable, match=re.escape(INVALID_RESEARCH_STATE_REASON)):
        _provider(tauric_state(), trace).analyze(instrument(), AS_OF)


@pytest.mark.parametrize("name", ["get_news", "get_prediction_markets", "unknown_tool"])
def test_any_unvalidated_executed_tool_alongside_stock_is_rejected(name: str) -> None:
    trace = tauric_trace()
    trace.append(
        {"name": name, "args": {"topic": "future"}, "id": "unsafe-call", "content": "99%"}
    )
    with pytest.raises(ResearchUnavailable, match=re.escape(INVALID_RESEARCH_STATE_REASON)):
        _provider(tauric_state(), trace).analyze(instrument(), AS_OF)


def test_indicator_cutoff_and_output_dates_are_validated() -> None:
    trace = tauric_trace()
    trace[1]["content"] = trace[1]["content"].replace(
        "2026-08-07: 61.50", "2026-08-08: 61.50"
    )
    with pytest.raises(ResearchUnavailable, match=re.escape(INVALID_RESEARCH_STATE_REASON)):
        _provider(tauric_state(), trace).analyze(instrument(), AS_OF)


def test_snapshot_cutoff_and_latest_source_date_are_validated() -> None:
    trace = tauric_trace()
    trace[2]["content"] = trace[2]["content"].replace(
        "Latest trading row used: 2026-08-07", "Latest trading row used: 2026-08-08"
    )
    with pytest.raises(ResearchUnavailable, match=re.escape(INVALID_RESEARCH_STATE_REASON)):
        _provider(tauric_state(), trace).analyze(instrument(), AS_OF)


def test_normal_pinned_snapshot_accepts_two_ascending_recent_rows() -> None:
    packet = _provider(tauric_state(), tauric_trace()).analyze(instrument(), AS_OF)

    assert packet.evidence[0].published_at == datetime(2026, 8, 8, tzinfo=UTC)


def test_snapshot_reversed_recent_rows_are_rejected() -> None:
    trace = tauric_trace()
    trace[2]["content"] = trace[2]["content"].replace(
        "| 2026-08-06 | 221.00 |\n| 2026-08-07 | 223.00 |",
        "| 2026-08-07 | 223.00 |\n| 2026-08-06 | 221.00 |",
    )

    with pytest.raises(ResearchUnavailable, match=re.escape(INVALID_RESEARCH_STATE_REASON)):
        _provider(tauric_state(), trace).analyze(instrument(), AS_OF)


def test_snapshot_duplicate_recent_row_is_rejected() -> None:
    trace = tauric_trace()
    trace[2]["content"] = trace[2]["content"].replace(
        "| 2026-08-06 | 221.00 |", "| 2026-08-07 | 221.00 |"
    )

    with pytest.raises(ResearchUnavailable, match=re.escape(INVALID_RESEARCH_STATE_REASON)):
        _provider(tauric_state(), trace).analyze(instrument(), AS_OF)


def test_snapshot_last_recent_row_must_match_latest() -> None:
    trace = tauric_trace()
    trace[2]["content"] = trace[2]["content"].replace(
        "| 2026-08-06 | 221.00 |\n| 2026-08-07 | 223.00 |",
        "| 2026-08-05 | 219.00 |\n| 2026-08-06 | 221.00 |",
    )

    with pytest.raises(ResearchUnavailable, match=re.escape(INVALID_RESEARCH_STATE_REASON)):
        _provider(tauric_state(), trace).analyze(instrument(), AS_OF)


def test_snapshot_future_recent_row_is_rejected() -> None:
    trace = tauric_trace()
    trace[2]["content"] = trace[2]["content"].replace(
        "| 2026-08-07 | 223.00 |", "| 2026-08-08 | 223.00 |"
    )

    with pytest.raises(ResearchUnavailable, match=re.escape(INVALID_RESEARCH_STATE_REASON)):
        _provider(tauric_state(), trace).analyze(instrument(), AS_OF)


@pytest.mark.parametrize(
    "field", ["news_report", "sentiment_report", "fundamentals_report", "past_context"]
)
def test_market_only_state_rejects_nonempty_unselected_or_memory_content(field: str) -> None:
    state = tauric_state()
    state[field] = "unsafe current or future context"
    with pytest.raises(ResearchUnavailable, match=re.escape(INVALID_RESEARCH_STATE_REASON)):
        _provider(state).analyze(instrument(), AS_OF)


def test_final_state_tool_messages_or_calls_are_rejected() -> None:
    state = tauric_state()
    state["messages"].append(
        {"type": "tool", "name": "get_stock_data", "tool_call_id": "laundered", "content": "x"}
    )
    with pytest.raises(ResearchUnavailable, match=re.escape(INVALID_RESEARCH_STATE_REASON)):
        _provider(state).analyze(instrument(), AS_OF)


def test_tool_call_symbol_and_cutoff_must_match_request() -> None:
    trace = tauric_trace()
    trace[0]["args"]["symbol"] = "MSFT"
    with pytest.raises(ResearchUnavailable, match=re.escape(INVALID_RESEARCH_STATE_REASON)):
        _provider(tauric_state(), trace).analyze(instrument(), AS_OF)


def test_tool_call_after_requested_cutoff_is_rejected() -> None:
    trace = tauric_trace()
    trace[0]["args"]["end_date"] = "2026-08-08"
    with pytest.raises(ResearchUnavailable, match=re.escape(INVALID_RESEARCH_STATE_REASON)):
        _provider(tauric_state(), trace).analyze(instrument(), AS_OF)


def test_malicious_confidence_object_is_never_stringified_or_leaked() -> None:
    class MaliciousConfidence:
        def __str__(self) -> str:
            raise AssertionError("secret-confidence-token")

    result = FakeTauricRunner.from_state(tauric_state()).propagate("AAPL", "2026-08-08")
    result["confidence"] = MaliciousConfidence()
    provider = TauricResearchProvider(
        runner=FakeTauricRunner.from_result(result), prompt_version="tauric-v1"
    )
    with pytest.raises(ResearchUnavailable, match=re.escape(INVALID_RESEARCH_STATE_REASON)) as exc:
        provider.analyze(instrument(), AS_OF)
    assert "secret-confidence-token" not in str(exc.value)


def test_dependency_guidance_is_fixed_and_preserved_by_provider() -> None:
    provider = TauricResearchProvider(
        runner=FakeTauricRunner(ResearchDependencyUnavailable("token-123")),
        prompt_version="tauric-v1",
    )
    with pytest.raises(ResearchDependencyUnavailable) as exc:
        provider.analyze(instrument(), AS_OF)
    assert str(exc.value) == (
        "TradingAgents research dependency is unavailable; install it with: "
        f"{RESEARCH_INSTALL_COMMAND}"
    )
    assert "token-123" not in str(exc.value)


def test_arbitrary_runner_failure_is_scrubbed() -> None:
    provider = TauricResearchProvider(
        runner=FakeTauricRunner(RuntimeError("token-123")), prompt_version="tauric-v1"
    )
    with pytest.raises(ResearchUnavailable, match="failed closed") as exc:
        provider.analyze(instrument(), AS_OF)
    assert "token-123" not in str(exc.value)


def test_process_dependency_failure_uses_fixed_install_guidance() -> None:
    runner = _runner(_dependency_missing_worker)
    with pytest.raises(ResearchDependencyUnavailable) as exc:
        runner.propagate("AAPL", "2026-08-08")
    assert RESEARCH_INSTALL_COMMAND in str(exc.value)
    assert "token-123" not in str(exc.value)


def test_process_arbitrary_failure_is_scrubbed() -> None:
    runner = _runner(_arbitrary_failure_worker)
    with pytest.raises(ResearchUnavailable, match="failed closed") as exc:
        runner.propagate("AAPL", "2026-08-08")
    assert "token-123" not in str(exc.value)


def test_audited_config_ignores_env_mutated_upstream_defaults_and_isolates_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attacker = "https://attacker.invalid/v1?api_key=secret-token"
    monkeypatch.setenv("TRADINGAGENTS_LLM_BACKEND_URL", attacker)
    monkeypatch.setenv("OPENAI_API_BASE", attacker)
    monkeypatch.setenv("TRADINGAGENTS_TEMPERATURE", "9.9")
    monkeypatch.setenv("TRADINGAGENTS_MAX_DEBATE_ROUNDS", "99")
    monkeypatch.setenv("TRADINGAGENTS_CHECKPOINT_ENABLED", "true")
    captures: list[dict[str, Any]] = []

    for index in range(2):
        capture_path = tmp_path / f"capture-{index}.json"
        worker = partial(
            _capture_config_worker,
            state=tauric_state(),
            trace=tauric_trace(),
            capture_path=str(capture_path),
        )
        result = _runner(worker, checkpoint_enabled=True).propagate("AAPL", "2026-08-08")
        captures.append(json.loads(capture_path.read_text(encoding="utf-8")))
        assert attacker not in json.dumps(result, sort_keys=True)
        assert result["configuration"]["backend_url"] == "https://api.openai.com/v1"
        assert result["configuration"]["temperature"] == 0.0
        assert result["configuration"]["max_debate_rounds"] == 1
        assert result["configuration"]["confidence_method"] == (
            "unit_interval:evidence=0.50,thesis=0.25,bear_case=0.25"
        )
        assert result["configuration"]["evidence_availability_method"] == (
            "completed_daily_bar_next_utc_day_v1"
        )
        assert result["configuration"]["analysis_cutoff_method"] == "previous_utc_date_v1"
        assert result["configuration"]["tool_trace_policy"] == (
            "pinned_market_tools_complete_ordered_v1"
        )
        assert result["configuration"]["llm_temporal_scope"] == (
            "not_point_in_time_historical"
        )

    for capture in captures:
        config = capture["config"]
        assert config["backend_url"] == "https://api.openai.com/v1"
        assert config["temperature"] == 0.0
        assert config["max_debate_rounds"] == 1
        assert config["max_risk_discuss_rounds"] == 1
        assert config["llm_max_retries"] == 0
        assert config["data_vendors"]["core_stock_apis"] == "yfinance"
        assert "prediction_markets" not in config["data_vendors"]
        assert "news_data" not in config["data_vendors"]
        assert config["news_article_limit"] == 0
        assert config["global_news_queries"] == []
        assert config["tool_vendors"] == {}
        assert config["checkpoint_enabled"] is True
        assert all(capture["paths_exist"].values())
        assert not Path(config["results_dir"]).exists()
        assert not Path(config["data_cache_dir"]).exists()
        assert not Path(config["memory_log_path"]).exists()
    assert captures[0]["config"]["project_dir"] != captures[1]["config"]["project_dir"]


def test_ollama_endpoint_is_fixed_to_loopback(tmp_path: Path) -> None:
    capture_path = tmp_path / "ollama.json"
    worker = partial(
        _capture_config_worker,
        state=tauric_state(),
        trace=tauric_trace(),
        capture_path=str(capture_path),
    )
    _runner(worker, llm_provider="ollama").propagate("AAPL", "2026-08-08")
    captured = json.loads(capture_path.read_text(encoding="utf-8"))
    assert captured["config"]["backend_url"] == "http://127.0.0.1:11434/v1"


@pytest.mark.parametrize("provider", ["openai_compatible", "anthropic", "unknown"])
def test_unconstrained_provider_is_rejected(provider: str) -> None:
    worker = partial(_return_state_worker, state=tauric_state(), trace=tauric_trace())
    with pytest.raises(ValueError, match="Tauric runner configuration is invalid"):
        _runner(worker, llm_provider=provider)


@pytest.mark.parametrize(
    "analysts",
    [
        ("market", "news"),
        ("news",),
        ("social",),
        ("fundamentals",),
        ("market", "fundamentals"),
    ],
)
def test_constructor_cannot_enable_nonmarket_analysts(analysts: tuple[str, ...]) -> None:
    worker = partial(_return_state_worker, state=tauric_state(), trace=tauric_trace())
    with pytest.raises(ValueError, match="Tauric runner configuration is invalid"):
        TauricUpstreamRunner(
            llm_provider="openai",
            model_id="fixture-model",
            checkpoint_enabled=False,
            selected_analysts=analysts,
            worker=worker,
        )


def test_timeout_terminates_work_and_repeated_deadlines_do_not_accumulate_workers(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "continued.txt"
    context = multiprocessing.get_context("spawn")

    for _ in range(3):
        started = context.Event()
        worker = partial(
            _delayed_side_effect_worker,
            marker_path=str(marker),
            started=started,
        )
        runner = _runner(worker, timeout_seconds=0.2, worker_started=started)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(runner.propagate, "AAPL", "2026-08-08")
            assert started.wait(timeout=10), "worker body never started"
            with pytest.raises(ResearchUnavailable, match="timed out"):
                future.result(timeout=10)

    time.sleep(0.65)
    assert not marker.exists()
    assert not [
        child
        for child in multiprocessing.active_children()
        if child.name == "tauric-research"
    ]


@pytest.mark.parametrize(
    "uri",
    [
        "https://example.test/a?api_key=secret-token",
        "https://exa mple.test/a",
        "https://exa\tmple.test/a",
        "https://-bad.example/a",
        "https://example..test/a",
        "https://\ud800.test/a",
    ],
)
def test_evidence_uri_rejects_queries_whitespace_control_and_invalid_idna(uri: str) -> None:
    with pytest.raises(ValueError, match="research evidence URI is invalid") as exc:
        _canonical_evidence_uri(uri)
    assert "secret-token" not in str(exc.value)


def test_evidence_uri_is_canonical_and_query_free() -> None:
    assert (
        _canonical_evidence_uri("https://FINANCE.YAHOO.COM:443/quote/AAPL/history#fragment")
        == "https://finance.yahoo.com/quote/AAPL/history"
    )


def test_symbol_with_query_credentials_is_rejected_without_leak() -> None:
    token = "api_key=secret-token"
    state = tauric_state()
    state["company_of_interest"] = f"AAPL?{token}"
    trace = tauric_trace()
    trace[0]["args"]["symbol"] = f"AAPL?{token}"
    with pytest.raises(ResearchUnavailable, match=re.escape(INVALID_RESEARCH_STATE_REASON)) as exc:
        _provider(state, trace).analyze(instrument(symbol=f"AAPL?{token}"), AS_OF)
    assert token not in str(exc.value)


def test_unknown_trace_error_does_not_leak_content() -> None:
    token = "secret-future-probability-token"
    trace = tauric_trace()
    trace.append(
        {
            "name": "get_prediction_markets",
            "args": {"topic": token},
            "id": "unsafe-secret",
            "content": token,
        }
    )
    with pytest.raises(ResearchUnavailable) as exc:
        _provider(tauric_state(), trace).analyze(instrument(), AS_OF)
    assert token not in str(exc.value)


def test_configuration_hash_is_stable_and_secret_fields_are_excluded() -> None:
    first_result = FakeTauricRunner.from_state(tauric_state()).propagate("AAPL", "2026-08-08")
    second_result = copy.deepcopy(first_result)
    first_result["configuration"]["api_key"] = "secret-one"
    second_result["configuration"]["api_key"] = "secret-two"
    first = TauricResearchProvider(
        runner=FakeTauricRunner.from_result(first_result), prompt_version="tauric-v1"
    ).analyze(instrument(), AS_OF)
    second = TauricResearchProvider(
        runner=FakeTauricRunner.from_result(second_result), prompt_version="tauric-v1"
    ).analyze(instrument(), AS_OF)
    assert first.configuration_hash == second.configuration_hash
    assert "secret-" not in repr(first)


def test_runner_provenance_records_all_material_safe_settings() -> None:
    worker = partial(_return_state_worker, state=tauric_state(), trace=tauric_trace())
    result = _runner(worker).propagate("AAPL", "2026-08-08")
    provenance = result["configuration"]
    expected_fields = set(FIXTURE_CONFIGURATION)
    assert set(provenance) == expected_fields
    assert provenance["upstream_commit"] == PINNED_TRADINGAGENTS_COMMIT
    assert provenance["selected_analysts"] == ["market"]
    assert not {"project_dir", "results_dir", "data_cache_dir", "memory_log_path"} & set(
        provenance
    )
