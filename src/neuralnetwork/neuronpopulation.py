from __future__ import annotations
from typing import NamedTuple
import enum
from myenums import OrderedEnum, HiddenValueEnum
import numpy as np
import numpy.typing as npt


class NeuronPopulation:
    def __init__(
        self,
        neuron_count: int,
        *,
        rng: np.random.Generator,
        rates_spontaneous: npt.ArrayLike = 0.0,
        rates_evoked: npt.ArrayLike = 0.0,
        time_constant_rates_evoked: float = 1.0,
    ) -> None:
        self.neuron_count = neuron_count
        self.rates_spontaneous = rates_spontaneous
        self.rates_evoked = rates_evoked
        self.time_constant_rates_evoked = time_constant_rates_evoked
        self.rng = rng

    @property
    def rates_spontaneous(self) -> npt.NDArray[np.float_]:
        return self.__rates_spontaneous

    @rates_spontaneous.setter
    def rates_spontaneous(self, value: npt.ArrayLike) -> None:
        self.__rates_spontaneous = np.full(self.neuron_count, value)

    @property
    def rates_evoked(self) -> npt.NDArray[np.float_]:
        return self.__rates_evoked

    @rates_evoked.setter
    def rates_evoked(self, value: npt.ArrayLike) -> None:
        self.__rates_evoked = np.full(self.neuron_count, value)

    def let_evoked_rates_decay_for(self, duration: float) -> None:
        self.rates_evoked *= np.exp(-duration / self.time_constant_rates_evoked)

    def draw_next_spike(self) -> NextSpike:
        interspike_interval_spontaneous = self._draw_spontaneous_interspike_interval()
        interspike_interval_evoked = self._draw_evoked_interspike_interval()

        if interspike_interval_spontaneous < interspike_interval_evoked:
            spike_type = SpikeType.SPONTANEOUS
            interspike_interval = interspike_interval_spontaneous
            spiking_probability_by_neuron = self.rates_spontaneous / np.sum(
                self.rates_spontaneous
            )
        else:
            spike_type = SpikeType.EVOKED
            interspike_interval = interspike_interval_evoked
            spiking_probability_by_neuron = self.rates_evoked / np.sum(
                self.rates_evoked
            )

        spiking_neuron = self._draw_spiking_neuron(spiking_probability_by_neuron)

        return NextSpike(interspike_interval, spiking_neuron, spike_type)

    def _draw_spontaneous_interspike_interval(self) -> float:
        return self.rng.exponential(1 / np.sum(self.rates_spontaneous))

    def _draw_evoked_interspike_interval(self) -> float:
        uniform_sample = self.rng.random()

        if uniform_sample >= self._probability_of_evoked_spike_in_finite_time():
            return np.inf

        return self._evoked_interspike_interval_percentile_function(uniform_sample)

    def _probability_of_evoked_spike_in_finite_time(self) -> float:
        return 1 - np.exp(-self.time_constant_rates_evoked * np.sum(self.rates_evoked))

    def _evoked_interspike_interval_percentile_function(
        self, probability: float
    ) -> float:
        return -self.time_constant_rates_evoked * np.log(
            1
            + np.log(1 - probability)
            / (self.time_constant_rates_evoked * np.sum(self.rates_evoked))
        )

    def _draw_spiking_neuron(
        self, spiking_probability_by_neuron: npt.NDArray[np.float_]
    ) -> int:
        return self.rng.choice(self.neuron_count, p=spiking_probability_by_neuron)


@enum.unique
class SpikeType(OrderedEnum, HiddenValueEnum):
    SPONTANEOUS = enum.auto()
    EVOKED = enum.auto()


class NextSpike(NamedTuple):
    interspike_interval: float
    spiking_neuron: int
    spike_type: SpikeType
