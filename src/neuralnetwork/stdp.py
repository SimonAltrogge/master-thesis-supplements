from __future__ import annotations
from typing import NamedTuple
from dataclasses import dataclass
from collections.abc import Sequence
import numpy as np
import numpy.typing as npt


class SpikeTimingDependentPlasticity:
    def __init__(
        self,
        neuron_count: int,
        *,
        stdp_function: STDPFunction,
        learning_rates: npt.ArrayLike = 1.0,
    ) -> None:
        self.neuron_count = neuron_count
        self.learning_rates = learning_rates

        presynaptic_time_constants, presynaptic_amplitudes = zip(
            *stdp_function.presynaptic_exponentials
        )
        postsynaptic_time_constants, postsynaptic_amplitudes = zip(
            *stdp_function.postsynaptic_exponentials
        )
        self.presynaptic_time_constants = np.array(presynaptic_time_constants)
        self.presynaptic_amplitudes = np.array(presynaptic_amplitudes)
        self.postsynaptic_time_constants = np.array(postsynaptic_time_constants)
        self.postsynaptic_amplitudes = np.array(postsynaptic_amplitudes)

        self.presynaptic_traces = 0.0
        self.postsynaptic_traces = 0.0

    @property
    def learning_rates(self) -> npt.NDArray[np.float_]:
        return self.__learning_rates

    @learning_rates.setter
    def learning_rates(self, value: npt.ArrayLike) -> None:
        self.__learning_rates = np.full(self.neuron_count, value)

    @property
    def presynaptic_traces(self) -> npt.NDArray[np.float_]:
        return self.__presynaptic_traces

    @presynaptic_traces.setter
    def presynaptic_traces(self, value: npt.ArrayLike) -> None:
        self.__presynaptic_traces = np.full(
            (self.neuron_count, len(self.presynaptic_amplitudes)), value
        )

    @property
    def postsynaptic_traces(self) -> npt.NDArray[np.float_]:
        return self.__postsynaptic_traces

    @postsynaptic_traces.setter
    def postsynaptic_traces(self, value: npt.ArrayLike) -> None:
        self.__postsynaptic_traces = np.full(
            (self.neuron_count, len(self.postsynaptic_amplitudes)), value
        )

    def get_weight_changes(self) -> WeightChanges:
        weight_changes_to_spiking_neuron = self.learning_rates * (
            self.presynaptic_traces @ self.presynaptic_amplitudes
        )
        weight_changes_from_spiking_neuron = self.learning_rates * (
            self.postsynaptic_traces @ self.postsynaptic_amplitudes
        )

        return WeightChanges(
            to_neuron=weight_changes_to_spiking_neuron,
            from_neuron=weight_changes_from_spiking_neuron,
        )

    def increase_synaptic_traces_of(self, spiking_neuron: int) -> None:
        self.presynaptic_traces[spiking_neuron, :] += 1
        self.postsynaptic_traces[spiking_neuron, :] += 1

    def let_synaptic_traces_decay_for(self, duration: float) -> None:
        self.presynaptic_traces *= np.exp(-duration / self.presynaptic_time_constants)
        self.postsynaptic_traces *= np.exp(-duration / self.postsynaptic_time_constants)


@dataclass
class STDPFunction:
    presynaptic_exponentials: Sequence[Exponential]
    postsynaptic_exponentials: Sequence[Exponential]


class WeightChanges(NamedTuple):
    to_neuron: npt.NDArray[np.float_]
    from_neuron: npt.NDArray[np.float_]


class Exponential(NamedTuple):
    time_constant: float
    amplitude: float
