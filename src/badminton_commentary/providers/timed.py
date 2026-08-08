from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .base import LLMProvider


@dataclass(frozen=True)
class ProviderTiming:
    label: str
    seconds: float
    succeeded: bool


@dataclass
class TimingStats:
    records: list[ProviderTiming] = field(default_factory=list)

    @property
    def call_count(self) -> int:
        return len(self.records)

    @property
    def total_seconds(self) -> float:
        return sum(record.seconds for record in self.records)


class TimedProvider:
    def __init__(
        self,
        provider: LLMProvider,
        *,
        label: str,
        stats: TimingStats,
        reporter: Callable[[ProviderTiming], None] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._provider = provider
        self._label = label
        self._stats = stats
        self._reporter = reporter
        self._clock = clock

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        started = self._clock()
        succeeded = False
        try:
            response = self._provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            succeeded = True
            return response
        finally:
            timing = ProviderTiming(
                label=self._label,
                seconds=self._clock() - started,
                succeeded=succeeded,
            )
            self._stats.records.append(timing)
            if self._reporter is not None:
                self._reporter(timing)
