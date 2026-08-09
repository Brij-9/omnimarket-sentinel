"""Behavioral contracts for deterministic signal combination."""

from datetime import UTC, datetime
from decimal import Decimal

from market_sentinel.domain.enums import Horizon, SignalDirection
from market_sentinel.domain.models import ResearchPacket, Signal
from market_sentinel.strategies.ensemble import SignalEnsemble
from tests.factories import research_packet, signal


def _signal(
    strategy_id: str,
    strength: str,
    *,
    direction: SignalDirection | None = None,
    instrument_id: str = "AAPL@alpaca",
    horizon: Horizon = Horizon.SWING,
    entry: str = "100",
    invalidation: str | None = None,
    take_profit: str | None = None,
    research_required: bool = False,
) -> Signal:
    signed_strength = Decimal(strength)
    resolved_direction = direction or (
        SignalDirection.LONG if signed_strength >= 0 else SignalDirection.SHORT
    )
    if resolved_direction is SignalDirection.LONG:
        stop = invalidation or "95"
        target = take_profit or "110"
    else:
        stop = invalidation or "105"
        target = take_profit or "90"
    return signal(
        strategy_id=strategy_id,
        instrument_id=instrument_id,
        direction=resolved_direction,
        strength=signed_strength,
        horizon=horizon,
        entry_price=entry,
        invalidation_price=stop,
        take_profit=target,
        research_required=research_required,
    )


def test_conflicting_high_strength_signals_produce_no_trade() -> None:
    """Averaging strong opposing convictions could conceal material disagreement."""
    signals = [
        _signal("swing-breakout", "0.8"),
        _signal("crypto-volatility-breakout", "-0.8"),
    ]

    assert SignalEnsemble().combine(signals, research=None) is None


def test_ensemble_applies_immutable_versioned_weights_deterministically() -> None:
    """Ignoring configured weights would change the hand-calculated composite."""
    ensemble = SignalEnsemble()
    signals = [
        _signal("swing-breakout", "0.8"),
        _signal("opening-range-vwap", "0.4"),
    ]

    first = ensemble.combine(signals, research=None)
    second = ensemble.combine(tuple(reversed(signals)), research=None)

    assert first == second
    assert first is not None
    assert first.strength == Decimal("0.625")
    assert first.strategy_version == ensemble.weights.version
    try:
        ensemble.weights.values["swing-breakout"] = Decimal("1")  # type: ignore[index]
    except TypeError:
        pass
    else:
        raise AssertionError("ensemble weights must be immutable")


def test_ensemble_clamps_constructed_out_of_range_strengths() -> None:
    """Boundary normalization must contain signals from untrusted deserialization."""
    malformed_strength = _signal("swing-breakout", "0.8").model_copy(
        update={"strength": Decimal("9")}
    )

    combined = SignalEnsemble().combine([malformed_strength], research=None)

    assert combined is not None
    assert combined.strength == Decimal("1")


def test_ensemble_rejects_weak_after_cost_signal() -> None:
    """A pre-cost edge below the explicit cost and expectancy floor is not tradable."""
    assert (
        SignalEnsemble().combine([_signal("swing-breakout", "0.29")], research=None)
        is None
    )


def test_ensemble_only_requires_research_for_selected_strategy() -> None:
    """A non-selected strategy must not impose its research policy on the winner."""
    selected_requires = [_signal("swing-breakout", "0.8", research_required=True)]
    nonselected_requires = [
        _signal("swing-breakout", "0.8"),
        _signal("crypto-volatility-breakout", "0.4", research_required=True),
    ]

    assert SignalEnsemble().combine(selected_requires, research=None) is None
    assert SignalEnsemble().combine(nonselected_requires, research=None) is not None


def test_research_adjustment_is_bounded_and_never_reverses_direction() -> None:
    """Research confidence may modify, but never take over, deterministic direction."""
    ensemble = SignalEnsemble()
    base = _signal("swing-breakout", "0.6")
    supportive = research_packet(instrument_id="AAPL@alpaca", confidence="1")
    adverse = research_packet(instrument_id="AAPL@alpaca", confidence="0")

    raised = ensemble.combine([base], supportive)
    lowered = ensemble.combine([base], adverse)

    assert raised is not None and lowered is not None
    assert raised.strength == Decimal("0.70")
    assert lowered.strength == Decimal("0.50")
    assert raised.direction is SignalDirection.LONG
    assert lowered.direction is SignalDirection.LONG


def test_short_research_adjustment_preserves_signed_direction() -> None:
    """Support for a short must increase magnitude without flipping its sign."""
    short = _signal("swing-breakout", "-0.6")
    supportive = research_packet(instrument_id="AAPL@alpaca", confidence="1")

    combined = SignalEnsemble().combine([short], supportive)

    assert combined is not None
    assert combined.strength == Decimal("-0.70")
    assert combined.direction is SignalDirection.SHORT


def test_ensemble_rejects_incoherent_signal_sets_and_protective_prices() -> None:
    """Mixed instruments/horizons or malformed stops must fail closed as one set."""
    base = _signal("swing-breakout", "0.8")
    other_instrument = _signal("opening-range-vwap", "0.8", instrument_id="MSFT@alpaca")
    other_horizon = _signal("opening-range-vwap", "0.8", horizon=Horizon.INTRADAY)
    bad_stop = base.model_copy(update={"invalidation_price": Decimal("101")})

    ensemble = SignalEnsemble()
    assert ensemble.combine([base, other_instrument], research=None) is None
    assert ensemble.combine([base, other_horizon], research=None) is None
    assert ensemble.combine([bad_stop], research=None) is None
    assert ensemble.combine([object()], research=None) is None  # type: ignore[list-item]


def test_ensemble_rejects_direction_strength_mismatch_and_unknown_strategy() -> None:
    """Contradictory or unversioned inputs must not enter weighted scoring."""
    wrong_sign = _signal(
        "swing-breakout", "0.8", direction=SignalDirection.SHORT
    ).model_copy(update={"strength": Decimal("0.8")})
    unknown = _signal("unknown", "0.8")

    ensemble = SignalEnsemble()
    assert ensemble.combine([wrong_sign], research=None) is None
    assert ensemble.combine([unknown], research=None) is None


def test_mismatched_or_malformed_research_fails_closed() -> None:
    """Research for another instrument or with invalid confidence is unusable."""
    base = _signal("swing-breakout", "0.8")
    mismatch = research_packet(instrument_id="MSFT@alpaca")
    malformed = ResearchPacket.model_construct(
        instrument_id="AAPL@alpaca",
        as_of=datetime(2026, 8, 9, tzinfo=UTC),
        thesis="thesis",
        bear_case="bear",
        catalysts=(),
        risks=(),
        evidence=(),
        confidence=Decimal("NaN"),
        model_id="model",
        prompt_version="v1",
        configuration_hash="hash",
    )

    ensemble = SignalEnsemble()
    assert ensemble.combine([base], mismatch) is None
    assert ensemble.combine([base], malformed) is None


def test_ensemble_preserves_coherent_protective_price_ordering() -> None:
    """Combining valid long protection must still yield a valid long envelope."""
    signals = [
        _signal("swing-breakout", "0.8", entry="100", invalidation="95", take_profit="110"),
        _signal(
            "opening-range-vwap",
            "0.6",
            entry="102",
            invalidation="98",
            take_profit="112",
        ),
    ]

    combined = SignalEnsemble().combine(signals, research=None)

    assert combined is not None
    assert combined.invalidation_price < combined.entry_price < combined.take_profit
