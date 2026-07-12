"""Wall-clock guard so training saves and exits cleanly before a Kaggle session cap."""
from __future__ import annotations

import time


class TimeBudget:
    def __init__(self, max_seconds: float, safety_margin: float = 0.05):
        # Stop a little early so there is time to flush the final checkpoint.
        self.deadline = time.monotonic() + max_seconds * (1.0 - safety_margin)
        self.max_seconds = max_seconds

    def remaining(self) -> float:
        return self.deadline - time.monotonic()

    def should_stop(self) -> bool:
        return self.remaining() <= 0
