from __future__ import annotations
from typing import ParamSpec, Concatenate, Generic
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from time import perf_counter
import numpy as np


@dataclass
class Spike:
    time: float


class SpikingNetwork(ABC):
    def __init__(self) -> None:
        self.time = 0.0
        self.spike_count = 0

        self.event_listeners = defaultdict(list)

    def simulate_for(self, duration: float) -> None:
        time_end = self.time + duration

        while self.time < time_end:
            self.simulate_next_spike()

    def simulate_while(self, test: Callable) -> None:
        while test():
            self.simulate_next_spike()

    def simulate_next_spike(self) -> None:
        spike = self._get_next_spike()
        interspike_interval = spike.time - self.time

        self._emit_event("pre_time_evolution", interspike_interval)
        self._evolve_in_time(interspike_interval)
        self._emit_event("post_time_evolution", interspike_interval)

        self._emit_event("pre_spike", spike)
        self._process_spike(spike)
        self._emit_event("post_spike", spike)

    @abstractmethod
    def _get_next_spike(self) -> Spike:
        return Spike(time=np.inf)

    @abstractmethod
    def _evolve_in_time(self, duration: float) -> None:
        self.time += duration

    @abstractmethod
    def _process_spike(self, spike: Spike) -> None:
        self.spike_count += 1

    def add_event_listener(
        self, event: Hashable, listener: NetworkEventListener
    ) -> None:
        self.event_listeners[event].append(listener)

    def remove_event_listener(
        self, event: Hashable, listener: NetworkEventListener
    ) -> None:
        if listener in self.event_listeners[event]:
            self.event_listeners[event].remove(listener)

    def _emit_event(self, event: Hashable, *args, **kwargs) -> None:
        for listener in self.event_listeners[event]:
            listener(self, *args, **kwargs)


P = ParamSpec("P")
EventListener = Callable[P, object]
NetworkEventListener = Callable[Concatenate[SpikingNetwork, P], object]


class after_n_events(Generic[P]):
    def __init__(self, n: int, callback: EventListener[P]) -> None:
        self.event_count = 0
        self.target_event_count = n
        self.has_run = False
        self.callback = callback

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> None:
        if self.has_run:
            return

        self.event_count += 1

        if self.event_count == self.target_event_count:
            self.callback(*args, **kwargs)
            self.has_run = True


class every_n_events(after_n_events[P]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> None:
        super().__call__(*args, **kwargs)

        if self.has_run:
            self.event_count = 0
            self.has_run = False


class after_interval(Generic[P]):
    def __init__(
        self,
        interval: float,
        callback: NetworkEventListener[P],
        *,
        start_time: float = 0.0,
    ) -> None:
        self.interval = interval
        self.target_time = start_time + interval
        self.has_run = False
        self.callback = callback

    def __call__(
        self, network: SpikingNetwork, *args: P.args, **kwargs: P.kwargs
    ) -> None:
        if self.has_run:
            return

        if network.time > self.target_time:
            self.callback(network, *args, **kwargs)
            self.has_run = True


class every_interval(after_interval[P]):
    def __call__(
        self, network: SpikingNetwork, *args: P.args, **kwargs: P.kwargs
    ) -> None:
        super().__call__(network, *args, **kwargs)

        if self.has_run:
            assert self.target_time is not None
            self.target_time += self.interval
            self.has_run = False


class after_real_time_interval(Generic[P]):
    def __init__(self, interval_in_seconds: float, callback: EventListener[P]) -> None:
        self.interval_in_seconds = interval_in_seconds
        self.end_time = perf_counter() + interval_in_seconds
        self.has_run = False
        self.callback = callback

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> None:
        if self.has_run:
            return

        if perf_counter() >= self.end_time:
            self.callback(*args, **kwargs)
            self.has_run = True


class every_real_time_interval(after_real_time_interval[P]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> None:
        super().__call__(*args, **kwargs)

        if self.has_run:
            self.end_time += self.interval_in_seconds
            self.has_run = False
