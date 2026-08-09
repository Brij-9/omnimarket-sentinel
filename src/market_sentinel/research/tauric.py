"""Fail-closed adapter for the pinned Tauric TradingAgents integration."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol
from urllib.parse import SplitResult, urlsplit

from market_sentinel.domain.models import Evidence, Instrument, ResearchPacket

PINNED_TRADINGAGENTS_COMMIT = "a33fd4c0f134485a43553a2c23a63cb14adbd88f"
RESEARCH_INSTALL_COMMAND = 'python -m pip install -e ".[research]"'

_SUPPORTED_LLM_PROVIDERS = frozenset(
    {
        "anthropic",
        "azure",
        "bedrock",
        "deepseek",
        "glm",
        "glm-cn",
        "google",
        "groq",
        "kimi",
        "minimax",
        "minimax-cn",
        "mistral",
        "nvidia",
        "ollama",
        "openai",
        "openai_compatible",
        "openrouter",
        "qwen",
        "qwen-cn",
        "xai",
    }
)
_SUPPORTED_ANALYSTS = frozenset({"market", "social", "news", "fundamentals"})
_PAYLOAD_FIELDS = frozenset(
    {
        "thesis",
        "bear_case",
        "catalysts",
        "risks",
        "confidence",
        "confidence_scale",
        "model_id",
        "configuration",
        "evidence",
    }
)
_TRADE_FIELD_FRAGMENTS = (
    "action",
    "allocation",
    "buy",
    "notional",
    "order",
    "position",
    "quantity",
    "sell",
    "side",
    "size",
    "stop_loss",
    "take_profit",
    "trade",
)
_PUBLIC_CONFIGURATION_FIELDS = frozenset(
    {
        "checkpoint_enabled",
        "deep_think_llm",
        "llm_provider",
        "quick_think_llm",
        "selected_analysts",
        "upstream_commit",
    }
)


class ResearchUnavailable(RuntimeError):
    """Research could not be produced safely, so callers must fail closed."""


class TauricRunner(Protocol):
    """Injected boundary that keeps unit tests independent of TradingAgents."""

    def propagate(self, symbol: str, date: str) -> Mapping[str, Any]: ...


def _valid_plain_text(value: Any, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError("upstream research text is invalid")
    normalized = value.strip()
    if (not normalized and not allow_empty) or any(ord(character) < 32 for character in normalized):
        raise ValueError("upstream research text is invalid")
    return normalized


def _text_list(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"research {field_name} must be a list of nonempty text")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        try:
            text = _valid_plain_text(item)
        except ValueError:
            raise ValueError(f"research {field_name} must be a list of nonempty text") from None
        if text not in seen:
            normalized.append(text)
            seen.add(text)
    return tuple(normalized)


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError("research evidence has an invalid source timestamp") from None
    else:
        raise ValueError("research evidence requires a source timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("research evidence source timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


def _canonical_evidence_uri(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("research evidence URI is invalid")
    parsed = urlsplit(value.strip())
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ValueError("research evidence URI must be an absolute HTTPS URI")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("research evidence URI cannot contain credentials")
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("research evidence URI is invalid") from None
    host = parsed.hostname.lower()
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = host if port in (None, 443) else f"{host}:{port}"
    return SplitResult("https", netloc, parsed.path or "/", parsed.query, "").geturl()


def _evidence(value: Any, as_of: datetime) -> tuple[Evidence, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)) or not value:
        raise ValueError("research evidence requires at least one timestamped source")
    result: list[Evidence] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"uri", "title", "published_at"}:
            raise ValueError("research evidence entries require URI, title, and source timestamp")
        uri = _canonical_evidence_uri(item["uri"])
        try:
            title = _valid_plain_text(item["title"])
        except ValueError:
            raise ValueError("research evidence title is invalid") from None
        published_at = _parse_timestamp(item["published_at"])
        if published_at > as_of:
            raise ValueError("look-ahead evidence is not permitted")
        if uri not in seen:
            result.append(Evidence(uri=uri, title=title, published_at=published_at))
            seen.add(uri)
    return tuple(result)


def _normalized_confidence(value: Any, scale: Any) -> Decimal:
    if isinstance(value, bool) or not isinstance(scale, str):
        raise ValueError("research confidence and its scale are invalid")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("research confidence must be finite")
    try:
        confidence = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError("research confidence is invalid") from None
    if not confidence.is_finite():
        raise ValueError("research confidence must be finite")
    if scale == "unit_interval":
        normalized = confidence
        upper_bound = Decimal("1")
    elif scale == "percent":
        normalized = confidence / Decimal("100")
        upper_bound = Decimal("100")
    else:
        raise ValueError("research confidence uses an unsupported scale")
    if confidence < 0 or confidence > upper_bound:
        raise ValueError("research confidence is outside its supported scale")
    return normalized


def _contains_trading_instruction(payload: Mapping[str, Any]) -> bool:
    for key in payload:
        if not isinstance(key, str):
            return True
        normalized = key.lower().replace("-", "_")
        if any(fragment in normalized for fragment in _TRADE_FIELD_FRAGMENTS):
            return True
    return False


def _public_configuration(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("research configuration provenance is invalid")
    public: dict[str, Any] = {}
    for key in sorted(_PUBLIC_CONFIGURATION_FIELDS):
        if key not in value:
            continue
        item = value[key]
        if isinstance(item, (str, bool, int)) or item is None:
            public[key] = item
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            if not all(isinstance(element, str) for element in item):
                raise ValueError("research configuration provenance is invalid")
            public[key] = list(item)
        else:
            raise ValueError("research configuration provenance is invalid")
    public["upstream_commit"] = PINNED_TRADINGAGENTS_COMMIT
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
    """Convert sanitized Tauric output to research-only domain data."""

    def __init__(self, *, runner: TauricRunner, prompt_version: str) -> None:
        try:
            self._prompt_version = _valid_plain_text(prompt_version)
        except ValueError:
            raise ValueError("prompt version is invalid") from None
        self._runner = runner

    def analyze(self, instrument: Instrument, as_of: datetime) -> ResearchPacket:
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")
        normalized_as_of = as_of.astimezone(UTC)
        try:
            raw_payload = self._runner.propagate(
                instrument.symbol, normalized_as_of.date().isoformat()
            )
        except Exception:
            raise ResearchUnavailable("Tauric research failed closed") from None
        if not isinstance(raw_payload, Mapping) or not all(
            isinstance(key, str) for key in raw_payload
        ):
            raise ValueError("upstream research must be a sanitized mapping")
        if _contains_trading_instruction(raw_payload):
            raise ValueError("research cannot contain trading instructions")
        if not set(raw_payload).issubset(_PAYLOAD_FIELDS):
            raise ValueError("upstream research mapping contains unsupported fields")

        thesis = _valid_plain_text(raw_payload.get("thesis", ""), allow_empty=True)
        bear_case = _valid_plain_text(raw_payload.get("bear_case", ""), allow_empty=True)
        if not thesis and not bear_case:
            raise ValueError("research requires a nonempty thesis or bear case")
        catalysts = _text_list(raw_payload.get("catalysts", ()), "catalysts")
        risks = _text_list(raw_payload.get("risks", ()), "risks")
        confidence = _normalized_confidence(
            raw_payload.get("confidence"), raw_payload.get("confidence_scale")
        )
        model_id = _valid_plain_text(raw_payload.get("model_id"))
        evidence = _evidence(raw_payload.get("evidence"), normalized_as_of)
        configuration_hash = _configuration_hash(
            raw_payload.get("configuration"),
            model_id=model_id,
            prompt_version=self._prompt_version,
        )
        return ResearchPacket(
            instrument_id=f"{instrument.symbol}@{instrument.venue}",
            as_of=normalized_as_of,
            thesis=thesis,
            bear_case=bear_case,
            catalysts=catalysts,
            risks=risks,
            evidence=evidence,
            confidence=confidence,
            model_id=model_id,
            prompt_version=self._prompt_version,
            configuration_hash=configuration_hash,
        )


class TauricUpstreamRunner:
    """Lazily invoke the exact reviewed TradingAgents API and sanitize its output."""

    def __init__(
        self,
        *,
        llm_provider: str,
        model_id: str,
        checkpoint_enabled: bool,
        selected_analysts: tuple[str, ...] = ("market", "news", "fundamentals"),
        timeout_seconds: float = 300.0,
    ) -> None:
        normalized_provider = llm_provider.strip().lower() if isinstance(llm_provider, str) else ""
        normalized_model = model_id.strip() if isinstance(model_id, str) else ""
        valid_model = (
            bool(normalized_model)
            and len(normalized_model) <= 200
            and all(character.isprintable() for character in normalized_model)
        )
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
            normalized_provider not in _SUPPORTED_LLM_PROVIDERS
            or not valid_model
            or type(checkpoint_enabled) is not bool
            or not valid_analysts
            or not valid_timeout
        ):
            raise ValueError("Tauric runner configuration is invalid")
        self._llm_provider = normalized_provider
        self._model_id = normalized_model
        self._checkpoint_enabled = checkpoint_enabled
        self._selected_analysts = selected_analysts
        self._timeout_seconds = float(timeout_seconds)

    def propagate(self, symbol: str, date: str) -> Mapping[str, Any]:
        if not self._valid_symbol_and_date(symbol, date):
            raise ValueError("Tauric research request is invalid")
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tauric-research")
        future = executor.submit(self._propagate, symbol, date)
        try:
            return future.result(timeout=self._timeout_seconds)
        except FutureTimeoutError:
            future.cancel()
            raise ResearchUnavailable("Tauric research timed out and failed closed") from None
        except ResearchUnavailable:
            raise
        except Exception:
            raise ResearchUnavailable("Tauric research failed closed") from None
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _valid_symbol_and_date(symbol: str, requested_date: str) -> bool:
        if (
            not isinstance(symbol, str)
            or not symbol.strip()
            or len(symbol) > 64
            or not all(character.isprintable() for character in symbol)
        ):
            return False
        try:
            parsed_date = date.fromisoformat(requested_date)
        except (TypeError, ValueError):
            return False
        return parsed_date.isoformat() == requested_date

    def _propagate(self, symbol: str, requested_date: str) -> Mapping[str, Any]:
        try:
            graph_module = importlib.import_module("tradingagents.graph.trading_graph")
            config_module = importlib.import_module("tradingagents.default_config")
        except (ImportError, ModuleNotFoundError):
            raise ResearchUnavailable(
                "TradingAgents research is unavailable; install it with: "
                f"{RESEARCH_INSTALL_COMMAND}"
            ) from None

        default_config = getattr(config_module, "DEFAULT_CONFIG", None)
        graph_class = getattr(graph_module, "TradingAgentsGraph", None)
        if not isinstance(default_config, Mapping) or not callable(graph_class):
            raise ResearchUnavailable("Pinned TradingAgents API is unavailable")
        config = copy.deepcopy(dict(default_config))
        config["llm_provider"] = self._llm_provider
        config["deep_think_llm"] = self._model_id
        config["quick_think_llm"] = self._model_id
        config["checkpoint_enabled"] = self._checkpoint_enabled

        graph = graph_class(selected_analysts=self._selected_analysts, config=config)
        upstream_result = graph.propagate(symbol, requested_date)
        if (
            not isinstance(upstream_result, tuple)
            or len(upstream_result) != 2
            or not isinstance(upstream_result[0], Mapping)
        ):
            raise ResearchUnavailable("Pinned TradingAgents returned an invalid result")
        state = upstream_result[0]
        research = state.get("research_packet", state)
        if not isinstance(research, Mapping):
            raise ResearchUnavailable("Pinned TradingAgents returned invalid research")
        sanitized = {
            key: copy.deepcopy(research[key]) for key in _PAYLOAD_FIELDS if key in research
        }
        sanitized["model_id"] = self._model_id
        sanitized["configuration"] = {
            "llm_provider": self._llm_provider,
            "deep_think_llm": self._model_id,
            "quick_think_llm": self._model_id,
            "checkpoint_enabled": self._checkpoint_enabled,
            "selected_analysts": list(self._selected_analysts),
            "upstream_commit": PINNED_TRADINGAGENTS_COMMIT,
        }
        return sanitized


__all__ = [
    "PINNED_TRADINGAGENTS_COMMIT",
    "RESEARCH_INSTALL_COMMAND",
    "ResearchUnavailable",
    "TauricResearchProvider",
    "TauricRunner",
    "TauricUpstreamRunner",
]
