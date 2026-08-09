"""Fail-closed adapter for the pinned Tauric TradingAgents integration."""

from __future__ import annotations

import csv
import hashlib
import importlib
import io
import ipaddress
import json
import math
import multiprocessing
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import SplitResult, quote, urlsplit

from market_sentinel.domain.models import Evidence, Instrument, ResearchPacket

PINNED_TRADINGAGENTS_COMMIT = "a33fd4c0f134485a43553a2c23a63cb14adbd88f"
RESEARCH_INSTALL_COMMAND = 'python -m pip install -e ".[research]"'
INVALID_RESEARCH_STATE_REASON = "Tauric research failed closed: invalid pinned upstream state"
NO_TRUSTWORTHY_EVIDENCE_REASON = (
    "Tauric research failed closed: no trustworthy timestamped evidence"
)
LOOK_AHEAD_EVIDENCE_REASON = "Tauric research failed closed: look-ahead evidence"
_DEPENDENCY_REASON = (
    "TradingAgents research dependency is unavailable; install it with: "
    f"{RESEARCH_INSTALL_COMMAND}"
)
_UPSTREAM_FAILURE_REASON = "Tauric research failed closed"
_TIMEOUT_REASON = "Tauric research timed out and failed closed"
_CONFIDENCE_METHOD = "unit_interval:evidence=0.50,thesis=0.25,bear_case=0.25"
_EVIDENCE_AVAILABILITY_METHOD = "completed_daily_bar_next_utc_day_v1"
_CONFIG_PROFILE = "tauric-audited-v1"
_MAX_STOCK_RECORDS = 100_000

_SUPPORTED_ENDPOINTS = {
    "openai": "https://api.openai.com/v1",
    "ollama": "http://127.0.0.1:11434/v1",
}
_SUPPORTED_ANALYSTS = frozenset({"market", "social", "news", "fundamentals"})
_SAFE_SYMBOL = re.compile(r"^[A-Za-z0-9.^=_+-]{1,32}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9._:-]{1,200}$")
_DNS_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_BAD_PERCENT_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")
_SECRET_KEY_PARTS = ("authorization", "credential", "password", "secret", "token", "key")

_AGENT_STATE_FIELDS = frozenset(
    {
        "messages",
        "company_of_interest",
        "asset_type",
        "instrument_context",
        "trade_date",
        "sender",
        "market_report",
        "sentiment_report",
        "news_report",
        "fundamentals_report",
        "investment_debate_state",
        "investment_plan",
        "trader_investment_plan",
        "risk_debate_state",
        "final_trade_decision",
        "past_context",
    }
)
_INVEST_DEBATE_FIELDS = frozenset(
    {"bull_history", "bear_history", "history", "current_response", "judge_decision", "count"}
)
_RISK_DEBATE_FIELDS = frozenset(
    {
        "aggressive_history",
        "conservative_history",
        "neutral_history",
        "history",
        "latest_speaker",
        "current_aggressive_response",
        "current_conservative_response",
        "current_neutral_response",
        "judge_decision",
        "count",
    }
)
_STATE_TEXT_FIELDS = _AGENT_STATE_FIELDS - {
    "messages",
    "investment_debate_state",
    "risk_debate_state",
}
_PUBLIC_CONFIGURATION_FIELDS = frozenset(
    {
        "profile",
        "upstream_commit",
        "llm_provider",
        "deep_think_llm",
        "quick_think_llm",
        "backend_url",
        "google_thinking_level",
        "openai_reasoning_effort",
        "anthropic_effort",
        "temperature",
        "llm_max_retries",
        "checkpoint_enabled",
        "memory_log_max_entries",
        "output_language",
        "max_debate_rounds",
        "max_risk_discuss_rounds",
        "max_recur_limit",
        "news_article_limit",
        "global_news_article_limit",
        "global_news_lookback_days",
        "global_news_queries",
        "data_vendors",
        "tool_vendors",
        "benchmark_ticker",
        "benchmark_map",
        "selected_analysts",
        "storage_scope",
        "confidence_method",
        "evidence_availability_method",
    }
)


class ResearchUnavailable(RuntimeError):
    """Research could not be produced safely, so callers must fail closed."""


class ResearchDependencyUnavailable(ResearchUnavailable):
    """The exact optional dependency is unavailable, with fixed safe guidance."""

    def __init__(self, *_untrusted: object) -> None:
        super().__init__(_DEPENDENCY_REASON)


class TauricRunner(Protocol):
    """Injected boundary that keeps unit tests independent of TradingAgents."""

    def propagate(self, symbol: str, date: str) -> Mapping[str, Any]: ...


Worker = Callable[[dict[str, Any]], Mapping[str, Any]]


def _research_text(value: Any, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ResearchUnavailable(INVALID_RESEARCH_STATE_REASON)
    normalized = value.strip()
    if not allow_empty and not normalized:
        raise ResearchUnavailable(INVALID_RESEARCH_STATE_REASON)
    if len(normalized) > 100_000 or any(
        ord(character) < 32 and character not in "\t\n\r" for character in normalized
    ):
        raise ResearchUnavailable(INVALID_RESEARCH_STATE_REASON)
    return normalized


def _is_safe_symbol(value: Any) -> bool:
    return isinstance(value, str) and _SAFE_SYMBOL.fullmatch(value) is not None


def _is_safe_id(value: Any) -> bool:
    return isinstance(value, str) and _SAFE_ID.fullmatch(value) is not None


def _exact_iso_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.isoformat() == value else None


def _canonical_host(host: str) -> str:
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        try:
            ascii_host = host.encode("idna").decode("ascii").lower()
        except (UnicodeError, ValueError):
            raise ValueError("research evidence URI is invalid") from None
        if len(ascii_host) > 253:
            raise ValueError("research evidence URI is invalid") from None
        labels = ascii_host.split(".")
        if not labels or any(not _DNS_LABEL.fullmatch(label) for label in labels):
            raise ValueError("research evidence URI is invalid") from None
        return ascii_host
    return f"[{ip.compressed}]" if ip.version == 6 else ip.compressed


def _canonical_evidence_uri(value: Any) -> str:
    """Validate a source URI without ever echoing untrusted input."""

    if (
        not isinstance(value, str)
        or value != value.strip()
        or not value
        or any(character.isspace() or ord(character) < 32 for character in value)
        or _BAD_PERCENT_ESCAPE.search(value)
    ):
        raise ValueError("research evidence URI is invalid")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except (UnicodeError, ValueError):
        raise ValueError("research evidence URI is invalid") from None
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
    ):
        raise ValueError("research evidence URI is invalid")
    host = _canonical_host(parsed.hostname)
    netloc = host if port in (None, 443) else f"{host}:{port}"
    return SplitResult("https", netloc, parsed.path or "/", "", "").geturl()


def _validate_debate(value: Any, expected_fields: frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ResearchUnavailable(INVALID_RESEARCH_STATE_REASON)
    result: dict[str, Any] = {}
    for key in expected_fields:
        item = value[key]
        if key == "count":
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ResearchUnavailable(INVALID_RESEARCH_STATE_REASON)
            result[key] = item
        else:
            result[key] = _research_text(item, allow_empty=True)
    return result


def _validate_state(value: Any, symbol: str, requested_date: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _AGENT_STATE_FIELDS:
        raise ResearchUnavailable(INVALID_RESEARCH_STATE_REASON)
    state: dict[str, Any] = {}
    for key in _STATE_TEXT_FIELDS:
        state[key] = _research_text(value[key], allow_empty=True)
    if state["company_of_interest"] != symbol or state["trade_date"] != requested_date:
        raise ResearchUnavailable(INVALID_RESEARCH_STATE_REASON)
    state["investment_debate_state"] = _validate_debate(
        value["investment_debate_state"], _INVEST_DEBATE_FIELDS
    )
    state["risk_debate_state"] = _validate_debate(
        value["risk_debate_state"], _RISK_DEBATE_FIELDS
    )
    messages = value["messages"]
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes, bytearray)):
        raise ResearchUnavailable(INVALID_RESEARCH_STATE_REASON)
    state["messages"] = list(messages)
    return state


def _validated_stock_call(
    call: Mapping[str, Any], symbol: str, requested: date
) -> tuple[str, date, date]:
    if set(call) != {"name", "args", "id", "type"} or call.get("type") != "tool_call":
        raise ResearchUnavailable(INVALID_RESEARCH_STATE_REASON)
    call_id = call.get("id")
    args = call.get("args")
    if not _is_safe_id(call_id) or not isinstance(args, Mapping):
        raise ResearchUnavailable(INVALID_RESEARCH_STATE_REASON)
    if set(args) != {"symbol", "start_date", "end_date"} or args.get("symbol") != symbol:
        raise ResearchUnavailable(INVALID_RESEARCH_STATE_REASON)
    start = _exact_iso_date(args.get("start_date"))
    end = _exact_iso_date(args.get("end_date"))
    if start is None or end is None or start > end or end > requested:
        raise ResearchUnavailable(INVALID_RESEARCH_STATE_REASON)
    assert isinstance(call_id, str)
    return call_id, start, end


def _parse_stock_tool_result(
    content: Any,
    symbol: str,
    start: date,
    end: date,
    requested: date,
) -> date:
    try:
        return _parse_stock_tool_result_strict(content, symbol, start, end, requested)
    except ResearchUnavailable:
        raise
    except Exception:
        raise ResearchUnavailable(NO_TRUSTWORTHY_EVIDENCE_REASON) from None


def _parse_stock_tool_result_strict(
    content: Any,
    symbol: str,
    start: date,
    end: date,
    requested: date,
) -> date:
    if not isinstance(content, str) or len(content) > 5_000_000:
        raise ResearchUnavailable(NO_TRUSTWORTHY_EVIDENCE_REASON)
    sections = content.split("\n\n")
    if len(sections) != 2:
        raise ResearchUnavailable(NO_TRUSTWORTHY_EVIDENCE_REASON)
    header_lines = sections[0].splitlines()
    expected_heading = (
        f"# Stock data for {symbol.upper()} from {start.isoformat()} to {end.isoformat()}"
    )
    if len(header_lines) != 3 or header_lines[0] != expected_heading:
        raise ResearchUnavailable(NO_TRUSTWORTHY_EVIDENCE_REASON)
    total_text = header_lines[1].removeprefix("# Total records: ")
    if (
        header_lines[1] != f"# Total records: {total_text}"
        or not total_text.isascii()
        or not total_text.isdecimal()
        or len(total_text) > 6
    ):
        raise ResearchUnavailable(NO_TRUSTWORTHY_EVIDENCE_REASON)
    total_records = int(total_text)
    if total_records > _MAX_STOCK_RECORDS:
        raise ResearchUnavailable(NO_TRUSTWORTHY_EVIDENCE_REASON)
    retrieved_text = header_lines[2].removeprefix("# Data retrieved on: ")
    try:
        retrieved_at = datetime.strptime(retrieved_text, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        raise ResearchUnavailable(NO_TRUSTWORTHY_EVIDENCE_REASON) from None
    if (
        header_lines[2] != f"# Data retrieved on: {retrieved_text}"
        or retrieved_at.strftime("%Y-%m-%d %H:%M:%S") != retrieved_text
    ):
        raise ResearchUnavailable(NO_TRUSTWORTHY_EVIDENCE_REASON)
    # The pinned tool emits a naive local retrieval clock. It is syntax-checked
    # above but cannot establish an aware source-publication instant.
    reader = csv.DictReader(io.StringIO(sections[1]))
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise ResearchUnavailable(NO_TRUSTWORTHY_EVIDENCE_REASON)
    if len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ResearchUnavailable(NO_TRUSTWORTHY_EVIDENCE_REASON)
    observed: list[date] = []
    for row in reader:
        if None in row or not all(isinstance(row.get(field), str) for field in required):
            raise ResearchUnavailable(NO_TRUSTWORTHY_EVIDENCE_REASON)
        observed_date = _exact_iso_date(row["Date"])
        if observed_date is None or observed_date < start:
            raise ResearchUnavailable(NO_TRUSTWORTHY_EVIDENCE_REASON)
        if observed_date > requested:
            raise ResearchUnavailable(LOOK_AHEAD_EVIDENCE_REASON)
        if observed_date > end or observed_date == requested:
            raise ResearchUnavailable(NO_TRUSTWORTHY_EVIDENCE_REASON)
        if observed and observed_date <= observed[-1]:
            raise ResearchUnavailable(NO_TRUSTWORTHY_EVIDENCE_REASON)
        for field in required - {"Date"}:
            try:
                number = Decimal(row[field])
            except InvalidOperation:
                raise ResearchUnavailable(NO_TRUSTWORTHY_EVIDENCE_REASON) from None
            if not number.is_finite():
                raise ResearchUnavailable(NO_TRUSTWORTHY_EVIDENCE_REASON)
        observed.append(observed_date)
    if len(observed) != total_records or not observed:
        raise ResearchUnavailable(NO_TRUSTWORTHY_EVIDENCE_REASON)
    return max(observed)


def _extract_evidence(
    messages: Sequence[Any], symbol: str, requested: date, as_of: datetime
) -> tuple[Evidence, ...]:
    pending: dict[str, tuple[date, date]] = {}
    consumed: set[str] = set()
    seen_call_ids: set[str] = set()
    seen_result_ids: set[str] = set()
    latest: date | None = None
    for message in messages:
        if not isinstance(message, Mapping):
            raise ResearchUnavailable(INVALID_RESEARCH_STATE_REASON)
        message_type = message.get("type")
        if message_type == "ai":
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, Sequence) or isinstance(
                tool_calls, (str, bytes, bytearray)
            ):
                raise ResearchUnavailable(INVALID_RESEARCH_STATE_REASON)
            for call in tool_calls:
                if not isinstance(call, Mapping):
                    raise ResearchUnavailable(INVALID_RESEARCH_STATE_REASON)
                call_id = call.get("id")
                if isinstance(call_id, str):
                    if call_id in seen_call_ids or call_id in seen_result_ids:
                        raise ResearchUnavailable(INVALID_RESEARCH_STATE_REASON)
                    seen_call_ids.add(call_id)
                if call.get("name") != "get_stock_data":
                    continue
                validated_id, start, end = _validated_stock_call(call, symbol, requested)
                pending[validated_id] = (start, end)
            continue
        if message_type != "tool":
            continue
        call_id = message.get("tool_call_id")
        if isinstance(call_id, str):
            if call_id in seen_result_ids:
                raise ResearchUnavailable(INVALID_RESEARCH_STATE_REASON)
            seen_result_ids.add(call_id)
        if message.get("name") != "get_stock_data":
            if isinstance(call_id, str) and (call_id in pending or call_id in consumed):
                raise ResearchUnavailable(INVALID_RESEARCH_STATE_REASON)
            continue
        if (
            set(message) != {"type", "content", "name", "tool_call_id"}
            or not _is_safe_id(call_id)
            or call_id not in pending
        ):
            raise ResearchUnavailable(INVALID_RESEARCH_STATE_REASON)
        assert isinstance(call_id, str)
        start, end = pending.pop(call_id)
        observed = _parse_stock_tool_result(
            message.get("content"), symbol, start, end, requested
        )
        consumed.add(call_id)
        if latest is None or observed > latest:
            latest = observed
    if pending:
        raise ResearchUnavailable(INVALID_RESEARCH_STATE_REASON)
    if latest is None:
        raise ResearchUnavailable(NO_TRUSTWORTHY_EVIDENCE_REASON)
    published_at = datetime.combine(
        latest + timedelta(days=1), datetime.min.time(), tzinfo=UTC
    )
    if published_at > as_of:
        raise ResearchUnavailable(NO_TRUSTWORTHY_EVIDENCE_REASON)
    uri = _canonical_evidence_uri(
        f"https://finance.yahoo.com/quote/{quote(symbol, safe='')}/history"
    )
    return (
        Evidence(
            uri=uri,
            title=f"{symbol} historical market data (Yahoo Finance)",
            published_at=published_at,
        ),
    )


def _is_secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(part in normalized for part in _SECRET_KEY_PARTS)


def _safe_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ResearchUnavailable(INVALID_RESEARCH_STATE_REASON)
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ResearchUnavailable(INVALID_RESEARCH_STATE_REASON)
            if _is_secret_key(key):
                continue
            result[key] = _safe_json_value(item)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_safe_json_value(item) for item in value]
    raise ResearchUnavailable(INVALID_RESEARCH_STATE_REASON)


def _public_configuration(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ResearchUnavailable(INVALID_RESEARCH_STATE_REASON)
    unknown = {
        key
        for key in value
        if not isinstance(key, str)
        or (key not in _PUBLIC_CONFIGURATION_FIELDS and not _is_secret_key(key))
    }
    if unknown:
        raise ResearchUnavailable(INVALID_RESEARCH_STATE_REASON)
    missing = _PUBLIC_CONFIGURATION_FIELDS - set(value)
    if missing:
        raise ResearchUnavailable(INVALID_RESEARCH_STATE_REASON)
    public = {
        key: _safe_json_value(value[key])
        for key in sorted(_PUBLIC_CONFIGURATION_FIELDS)
    }
    if (
        public["upstream_commit"] != PINNED_TRADINGAGENTS_COMMIT
        or public["confidence_method"] != _CONFIDENCE_METHOD
        or public["evidence_availability_method"] != _EVIDENCE_AVAILABILITY_METHOD
    ):
        raise ResearchUnavailable(INVALID_RESEARCH_STATE_REASON)
    return public


def _configuration_hash(
    configuration: Any, *, model_id: str, prompt_version: str
) -> str:
    provenance = {
        "configuration": _public_configuration(configuration),
        "model_id": model_id,
        "prompt_version": prompt_version,
    }
    encoded = json.dumps(
        provenance,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class TauricResearchProvider:
    """Convert a sanitized real AgentState to research-only domain data."""

    def __init__(self, *, runner: TauricRunner, prompt_version: str) -> None:
        if not _is_safe_id(prompt_version):
            raise ValueError("prompt version is invalid")
        self._prompt_version = prompt_version
        self._runner = runner

    def analyze(self, instrument: Instrument, as_of: datetime) -> ResearchPacket:
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")
        normalized_as_of = as_of.astimezone(UTC)
        if not _is_safe_symbol(instrument.symbol):
            raise ResearchUnavailable(INVALID_RESEARCH_STATE_REASON)
        requested_date = normalized_as_of.date().isoformat()
        try:
            raw = self._runner.propagate(instrument.symbol, requested_date)
        except ResearchDependencyUnavailable:
            raise ResearchDependencyUnavailable() from None
        except Exception:
            raise ResearchUnavailable(_UPSTREAM_FAILURE_REASON) from None
        if not isinstance(raw, Mapping) or set(raw) != {"state", "model_id", "configuration"}:
            raise ResearchUnavailable(INVALID_RESEARCH_STATE_REASON)
        model_id = raw["model_id"]
        if not _is_safe_id(model_id):
            raise ResearchUnavailable(INVALID_RESEARCH_STATE_REASON)
        assert isinstance(model_id, str)
        state = _validate_state(raw["state"], instrument.symbol, requested_date)
        debate = state["investment_debate_state"]
        assert isinstance(debate, dict)
        thesis = debate["bull_history"]
        bear_case = debate["bear_history"]
        assert isinstance(thesis, str)
        assert isinstance(bear_case, str)
        if not thesis and not bear_case:
            raise ResearchUnavailable(INVALID_RESEARCH_STATE_REASON)
        messages = state["messages"]
        assert isinstance(messages, list)
        evidence = _extract_evidence(
            messages,
            instrument.symbol,
            normalized_as_of.date(),
            normalized_as_of,
        )
        confidence = Decimal("0.50")
        if thesis:
            confidence += Decimal("0.25")
        if bear_case:
            confidence += Decimal("0.25")
        configuration_hash = _configuration_hash(
            raw["configuration"],
            model_id=model_id,
            prompt_version=self._prompt_version,
        )
        return ResearchPacket(
            instrument_id=f"{instrument.symbol}@{instrument.venue}",
            as_of=normalized_as_of,
            thesis=thesis,
            bear_case=bear_case,
            catalysts=(),
            risks=(),
            evidence=evidence,
            confidence=confidence,
            model_id=model_id,
            prompt_version=self._prompt_version,
            configuration_hash=configuration_hash,
        )


def _message_mapping(message: Any) -> dict[str, Any]:
    if isinstance(message, Mapping):
        message_type = message.get("type")
        content = message.get("content")
        tool_calls = message.get("tool_calls", [])
        name = message.get("name")
        tool_call_id = message.get("tool_call_id")
    else:
        message_type = getattr(message, "type", None)
        content = getattr(message, "content", None)
        tool_calls = getattr(message, "tool_calls", [])
        name = getattr(message, "name", None)
        tool_call_id = getattr(message, "tool_call_id", None)
    if not isinstance(message_type, str) or not isinstance(content, str):
        raise RuntimeError("invalid upstream message")
    if message_type == "ai":
        if not isinstance(tool_calls, Sequence):
            raise RuntimeError("invalid upstream message")
        sanitized_calls: list[dict[str, Any]] = []
        for call in tool_calls:
            if not isinstance(call, Mapping):
                raise RuntimeError("invalid upstream message")
            sanitized_calls.append(
                {
                    "name": call.get("name"),
                    "args": call.get("args"),
                    "id": call.get("id"),
                    "type": call.get("type"),
                }
            )
        return {"type": "ai", "content": content, "tool_calls": sanitized_calls}
    if message_type == "tool":
        return {
            "type": "tool",
            "content": content,
            "name": name,
            "tool_call_id": tool_call_id,
        }
    return {"type": message_type, "content": content}


def _serialize_agent_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("invalid upstream state")
    result = {key: value.get(key) for key in _AGENT_STATE_FIELDS if key != "messages"}
    messages = value.get("messages")
    if not isinstance(messages, Sequence):
        raise RuntimeError("invalid upstream state")
    result["messages"] = [_message_mapping(message) for message in messages]
    return result


def _invoke_pinned_graph(request: dict[str, Any]) -> Mapping[str, Any]:
    """Child-process worker that imports and calls only the pinned public API."""

    graph_module = importlib.import_module("tradingagents.graph.trading_graph")
    graph_class = getattr(graph_module, "TradingAgentsGraph", None)
    if not callable(graph_class):
        raise ImportError("pinned TradingAgents API unavailable")
    graph = graph_class(
        selected_analysts=tuple(request["selected_analysts"]),
        config=request["config"],
    )
    upstream = graph.propagate(request["symbol"], request["date"])
    if not isinstance(upstream, tuple) or len(upstream) != 2:
        raise RuntimeError("invalid pinned TradingAgents result")
    return _serialize_agent_state(upstream[0])


def _worker_entry(
    connection: Connection,
    worker: Worker,
    request: dict[str, Any],
) -> None:
    """Return fixed status codes so child errors and secrets never cross IPC."""

    try:
        result = worker(request)
        if not isinstance(result, Mapping):
            connection.send(("invalid", None))
        else:
            connection.send(("ok", dict(result)))
    except (ImportError, ModuleNotFoundError):
        connection.send(("dependency_unavailable", None))
    except BaseException:
        with suppress(BaseException):
            connection.send(("upstream_unavailable", None))
    finally:
        connection.close()


def _terminate_process(process: BaseProcess) -> None:
    if not process.is_alive():
        process.join(timeout=1)
        return
    process.terminate()
    process.join(timeout=2)
    if process.is_alive():
        process.kill()
        process.join(timeout=2)


def _audited_config(
    run_root: Path,
    *,
    llm_provider: str,
    model_id: str,
    checkpoint_enabled: bool,
) -> dict[str, Any]:
    return {
        "project_dir": str(run_root),
        "results_dir": str(run_root / "results"),
        "data_cache_dir": str(run_root / "cache"),
        "memory_log_path": str(run_root / "memory" / "trading_memory.md"),
        "memory_log_max_entries": 0,
        "llm_provider": llm_provider,
        "deep_think_llm": model_id,
        "quick_think_llm": model_id,
        "backend_url": _SUPPORTED_ENDPOINTS[llm_provider],
        "google_thinking_level": None,
        "openai_reasoning_effort": None,
        "anthropic_effort": None,
        "temperature": 0.0,
        "llm_max_retries": 0,
        "checkpoint_enabled": checkpoint_enabled,
        "output_language": "English",
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
        "max_recur_limit": 100,
        "news_article_limit": 20,
        "global_news_article_limit": 10,
        "global_news_lookback_days": 7,
        "global_news_queries": [
            "Federal Reserve interest rates inflation",
            "S&P 500 earnings GDP economic outlook",
            "geopolitical risk trade war sanctions",
            "ECB Bank of England BOJ central bank policy",
            "oil commodities supply chain energy",
        ],
        "data_vendors": {
            "core_stock_apis": "yfinance",
            "technical_indicators": "yfinance",
            "fundamental_data": "yfinance",
            "news_data": "yfinance",
            "macro_data": "fred",
            "prediction_markets": "polymarket",
        },
        "tool_vendors": {},
        "benchmark_ticker": None,
        "benchmark_map": {
            ".NS": "^NSEI",
            ".BO": "^BSESN",
            ".T": "^N225",
            ".HK": "^HSI",
            ".L": "^FTSE",
            ".TO": "^GSPTSE",
            ".AX": "^AXJO",
            ".SS": "000001.SS",
            ".SZ": "399001.SZ",
            "": "SPY",
        },
    }


def _provenance(config: Mapping[str, Any], selected_analysts: tuple[str, ...]) -> dict[str, Any]:
    return {
        "profile": _CONFIG_PROFILE,
        "upstream_commit": PINNED_TRADINGAGENTS_COMMIT,
        **{
            key: config[key]
            for key in _PUBLIC_CONFIGURATION_FIELDS
            if key in config
        },
        "selected_analysts": list(selected_analysts),
        "storage_scope": "isolated-per-run",
        "confidence_method": _CONFIDENCE_METHOD,
        "evidence_availability_method": _EVIDENCE_AVAILABILITY_METHOD,
    }


class TauricUpstreamRunner:
    """Run the pinned graph in a killable process and return a real-state envelope."""

    def __init__(
        self,
        *,
        llm_provider: str,
        model_id: str,
        checkpoint_enabled: bool,
        selected_analysts: tuple[str, ...] = ("market", "news", "fundamentals"),
        timeout_seconds: float = 300.0,
        worker: Worker = _invoke_pinned_graph,
    ) -> None:
        normalized_provider = llm_provider.strip().lower() if isinstance(llm_provider, str) else ""
        valid_analysts = (
            isinstance(selected_analysts, tuple)
            and bool(selected_analysts)
            and len(set(selected_analysts)) == len(selected_analysts)
            and set(selected_analysts).issubset(_SUPPORTED_ANALYSTS)
        )
        valid_timeout = (
            not isinstance(timeout_seconds, bool)
            and isinstance(timeout_seconds, (int, float))
            and math.isfinite(float(timeout_seconds))
            and timeout_seconds > 0
        )
        if (
            normalized_provider not in _SUPPORTED_ENDPOINTS
            or not _is_safe_id(model_id)
            or type(checkpoint_enabled) is not bool
            or not valid_analysts
            or not valid_timeout
            or not callable(worker)
        ):
            raise ValueError("Tauric runner configuration is invalid")
        self._llm_provider = normalized_provider
        self._model_id = model_id
        self._checkpoint_enabled = checkpoint_enabled
        self._selected_analysts = selected_analysts
        self._timeout_seconds = float(timeout_seconds)
        self._worker = worker

    def propagate(self, symbol: str, date: str) -> Mapping[str, Any]:
        requested = _exact_iso_date(date)
        if not _is_safe_symbol(symbol) or requested is None:
            raise ValueError("Tauric research request is invalid")
        with tempfile.TemporaryDirectory(prefix="market-sentinel-tauric-") as temporary:
            config = _audited_config(
                Path(temporary),
                llm_provider=self._llm_provider,
                model_id=self._model_id,
                checkpoint_enabled=self._checkpoint_enabled,
            )
            Path(config["results_dir"]).mkdir(parents=True, exist_ok=True)
            Path(config["data_cache_dir"]).mkdir(parents=True, exist_ok=True)
            Path(config["memory_log_path"]).parent.mkdir(parents=True, exist_ok=True)
            state = self._run_process(
                {
                    "symbol": symbol,
                    "date": date,
                    "selected_analysts": list(self._selected_analysts),
                    "config": config,
                }
            )
            return {
                "state": state,
                "model_id": self._model_id,
                "configuration": _provenance(config, self._selected_analysts),
            }

    def _run_process(self, request: dict[str, Any]) -> Mapping[str, Any]:
        context = multiprocessing.get_context("spawn")
        receive, send = context.Pipe(duplex=False)
        process = context.Process(
            target=_worker_entry,
            args=(send, self._worker, request),
            name="tauric-research",
        )
        started = False
        try:
            try:
                process.start()
                started = True
                send.close()
            except Exception:
                raise ResearchUnavailable(_UPSTREAM_FAILURE_REASON) from None
            if not receive.poll(self._timeout_seconds):
                if process.is_alive():
                    _terminate_process(process)
                    raise ResearchUnavailable(_TIMEOUT_REASON)
                raise ResearchUnavailable(_UPSTREAM_FAILURE_REASON)
            try:
                response = receive.recv()
            except (EOFError, OSError):
                raise ResearchUnavailable(_UPSTREAM_FAILURE_REASON) from None
            process.join(timeout=2)
            if process.is_alive():
                _terminate_process(process)
                raise ResearchUnavailable(_UPSTREAM_FAILURE_REASON)
            if (
                not isinstance(response, tuple)
                or len(response) != 2
                or not isinstance(response[0], str)
            ):
                raise ResearchUnavailable(_UPSTREAM_FAILURE_REASON)
            status, payload = response
            if status == "dependency_unavailable":
                raise ResearchDependencyUnavailable()
            if status != "ok" or not isinstance(payload, Mapping):
                raise ResearchUnavailable(_UPSTREAM_FAILURE_REASON)
            return dict(payload)
        finally:
            receive.close()
            send.close()
            if started and process.is_alive():
                _terminate_process(process)


__all__ = [
    "INVALID_RESEARCH_STATE_REASON",
    "NO_TRUSTWORTHY_EVIDENCE_REASON",
    "PINNED_TRADINGAGENTS_COMMIT",
    "RESEARCH_INSTALL_COMMAND",
    "ResearchDependencyUnavailable",
    "ResearchUnavailable",
    "TauricResearchProvider",
    "TauricRunner",
    "TauricUpstreamRunner",
]
