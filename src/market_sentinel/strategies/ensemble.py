"""Deterministic weighted ensemble with fail-closed research adjustment."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from types import MappingProxyType

from market_sentinel.domain.enums import Horizon, SignalDirection
from market_sentinel.domain.models import Evidence, ResearchPacket, Signal

_ZERO = Decimal("0")
_ONE = Decimal("1")
_NEGATIVE_ONE = Decimal("-1")


@dataclass(frozen=True, slots=True)
class EnsembleWeights:
    """Immutable strategy weight set whose version is recorded on every output."""

    version: str
    values: Mapping[str, Decimal]

    def __post_init__(self) -> None:
        checked = dict(self.values)
        if not self.version.strip() or not checked:
            raise ValueError("versioned ensemble weights must not be empty")
        if any(
            not isinstance(value, Decimal) or not value.is_finite() or value <= _ZERO
            for value in checked.values()
        ):
            raise ValueError("ensemble weights must be finite positive Decimals")
        object.__setattr__(self, "values", MappingProxyType(checked))


class SignalEnsemble:
    """Combine coherent strategy signals without sizing or execution intent."""

    HIGH_STRENGTH_CONFLICT = Decimal("0.75")
    NORMALIZED_COST = Decimal("0.05")
    MIN_AFTER_COST_STRENGTH = Decimal("0.25")
    MAX_RESEARCH_ADJUSTMENT = Decimal("0.10")
    DEFAULT_WEIGHTS = EnsembleWeights(
        version="ensemble-weights-v1",
        values={
            "swing-breakout": Decimal("0.45"),
            "opening-range-vwap": Decimal("0.35"),
            "crypto-volatility-breakout": Decimal("0.20"),
        },
    )

    def __init__(self, weights: EnsembleWeights | None = None) -> None:
        self.weights = self.DEFAULT_WEIGHTS if weights is None else weights

    def combine(
        self,
        signals: Sequence[Signal],
        research: ResearchPacket | None,
    ) -> Signal | None:
        """Return one bounded signal or fail closed for any incoherent input."""
        try:
            checked = self._validated_signals(signals)
            if not checked:
                return None
            if research is not None and not _research_is_valid(research, checked[0].instrument_id):
                return None

            normalized = tuple((item, _clamp(item.strength)) for item in checked)
            strong_longs = any(
                item.direction is SignalDirection.LONG
                and strength >= self.HIGH_STRENGTH_CONFLICT
                for item, strength in normalized
            )
            strong_shorts = any(
                item.direction is SignalDirection.SHORT
                and strength <= -self.HIGH_STRENGTH_CONFLICT
                for item, strength in normalized
            )
            if strong_longs and strong_shorts:
                return None

            active_weight = sum(
                (self.weights.values[item.strategy_id] for item, _ in normalized), _ZERO
            )
            if active_weight <= _ZERO:
                return None
            composite = sum(
                (
                    self.weights.values[item.strategy_id] * strength
                    for item, strength in normalized
                ),
                _ZERO,
            ) / active_weight
            if composite == _ZERO:
                return None
            if abs(composite) - self.NORMALIZED_COST < self.MIN_AFTER_COST_STRENGTH:
                return None
            direction = (
                SignalDirection.LONG if composite > _ZERO else SignalDirection.SHORT
            )
            aligned = tuple(
                (item, strength)
                for item, strength in normalized
                if item.direction is direction
            )
            if not aligned:
                return None
            selected, _ = max(
                aligned,
                key=lambda pair: (
                    abs(self.weights.values[pair[0].strategy_id] * pair[1]),
                    pair[0].strategy_id,
                ),
            )
            if selected.research_required and research is None:
                return None

            adjusted = composite
            if research is not None:
                adjustment = (
                    (research.confidence - Decimal("0.5")) * Decimal("2")
                    * self.MAX_RESEARCH_ADJUSTMENT
                )
                adjusted += adjustment if direction is SignalDirection.LONG else -adjustment
                adjusted = _clamp(adjusted)
                if (direction is SignalDirection.LONG and adjusted <= _ZERO) or (
                    direction is SignalDirection.SHORT and adjusted >= _ZERO
                ):
                    return None
            if abs(adjusted) - self.NORMALIZED_COST < self.MIN_AFTER_COST_STRENGTH:
                return None

            entry, invalidation, take_profit = _weighted_protective_prices(
                aligned, self.weights.values
            )
            evidence_uris = tuple(
                dict.fromkeys(
                    uri
                    for item, _ in aligned
                    for uri in item.evidence_uris
                )
            )
            if research is not None:
                evidence_uris = tuple(
                    dict.fromkeys(
                        (*evidence_uris, *(item.uri for item in research.evidence))
                    )
                )
            return Signal(
                strategy_id="signal-ensemble",
                strategy_version=self.weights.version,
                instrument_id=selected.instrument_id,
                direction=direction,
                strength=adjusted,
                horizon=selected.horizon,
                entry_price=entry,
                invalidation_price=invalidation,
                take_profit=take_profit,
                research_required=selected.research_required,
                evidence_uris=evidence_uris,
            )
        except (AttributeError, InvalidOperation, TypeError, ValueError, ZeroDivisionError):
            return None

    def _validated_signals(self, signals: Sequence[Signal]) -> tuple[Signal, ...]:
        if isinstance(signals, (str, bytes)):
            return ()
        checked = tuple(signals)
        if not checked or any(not isinstance(item, Signal) for item in checked):
            return ()
        instrument_id = checked[0].instrument_id
        horizon = checked[0].horizon
        strategy_ids: set[str] = set()
        for item in checked:
            if (
                item.strategy_id not in self.weights.values
                or item.strategy_id in strategy_ids
                or not item.strategy_version.strip()
                or not item.instrument_id.strip()
                or item.instrument_id != instrument_id
                or item.horizon is not horizon
                or not isinstance(item.horizon, Horizon)
                or item.direction not in (SignalDirection.LONG, SignalDirection.SHORT)
                or not _signal_numbers_are_valid(item)
                or (item.direction is SignalDirection.LONG and item.strength <= _ZERO)
                or (item.direction is SignalDirection.SHORT and item.strength >= _ZERO)
                or not isinstance(item.research_required, bool)
                or not isinstance(item.evidence_uris, tuple)
                or any(not isinstance(uri, str) or not uri.strip() for uri in item.evidence_uris)
            ):
                return ()
            strategy_ids.add(item.strategy_id)
        return checked


def _signal_numbers_are_valid(signal: Signal) -> bool:
    numbers = (
        signal.strength,
        signal.entry_price,
        signal.invalidation_price,
        signal.take_profit,
    )
    if any(not isinstance(value, Decimal) or not value.is_finite() for value in numbers):
        return False
    if (
        signal.entry_price <= _ZERO
        or signal.invalidation_price <= _ZERO
        or signal.take_profit <= _ZERO
    ):
        return False
    if signal.direction is SignalDirection.LONG:
        return signal.invalidation_price < signal.entry_price < signal.take_profit
    return signal.invalidation_price > signal.entry_price > signal.take_profit


def _research_is_valid(research: ResearchPacket, instrument_id: str) -> bool:
    if not isinstance(research, ResearchPacket):
        return False
    if (
        research.instrument_id != instrument_id
        or not isinstance(research.confidence, Decimal)
        or not research.confidence.is_finite()
        or not _ZERO <= research.confidence <= _ONE
        or not isinstance(research.as_of, datetime)
        or research.as_of.tzinfo is None
        or research.as_of.utcoffset() is None
        or not all(
            isinstance(value, str) and value.strip()
            for value in (
                research.instrument_id,
                research.thesis,
                research.bear_case,
                research.model_id,
                research.prompt_version,
                research.configuration_hash,
            )
        )
        or not all(isinstance(item, str) for item in (*research.catalysts, *research.risks))
    ):
        return False
    return all(
        isinstance(item, Evidence)
        and bool(item.uri.strip())
        and bool(item.title.strip())
        and item.published_at.tzinfo is not None
        and item.published_at <= research.as_of
        for item in research.evidence
    )


def _weighted_protective_prices(
    aligned: tuple[tuple[Signal, Decimal], ...],
    weights: Mapping[str, Decimal],
) -> tuple[Decimal, Decimal, Decimal]:
    contributions = tuple(
        (item, abs(weights[item.strategy_id] * strength)) for item, strength in aligned
    )
    total = sum((contribution for _, contribution in contributions), _ZERO)
    if total <= _ZERO:
        raise ValueError("protective price weights must be positive")

    def average(attribute: str) -> Decimal:
        return sum(
            (
                getattr(item, attribute) * contribution
                for item, contribution in contributions
            ),
            _ZERO,
        ) / total

    return average("entry_price"), average("invalidation_price"), average("take_profit")


def _clamp(value: Decimal) -> Decimal:
    return max(_NEGATIVE_ONE, min(_ONE, value))
