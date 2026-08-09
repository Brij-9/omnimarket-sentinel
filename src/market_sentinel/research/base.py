"""Side-effect-free contracts for optional research providers."""

from datetime import datetime
from typing import Protocol

from market_sentinel.domain.models import Instrument, ResearchPacket


class ResearchProvider(Protocol):
    """Convert point-in-time research into a validated domain packet."""

    def analyze(self, instrument: Instrument, as_of: datetime) -> ResearchPacket: ...
