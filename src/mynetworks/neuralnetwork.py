from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from neuralnetwork import *
from neuralnetwork.neuronpopulation import SpikeType
from mynetworks.util import *


STDP_INTEGRAL = -(4 / 30)
TIME_CONSTANT_STDP_POTENTIATION = 2.5
AMPLITUDE_STDP_POTENTIALTION = 0.08
TIME_CONSTANT_STDP_DEPRESSION = 5.0
AMPLITUDE_STDP_DEPRESSION = get_remaining_amplitude_for_symmetric_stdp(
    STDP_INTEGRAL,
    TIME_CONSTANT_STDP_DEPRESSION,
    TIME_CONSTANT_STDP_POTENTIATION,
    AMPLITUDE_STDP_POTENTIALTION,
)
SYNAPTIC_EXPONENTIALS = [
    Exponential(TIME_CONSTANT_STDP_POTENTIATION, AMPLITUDE_STDP_POTENTIALTION),
    Exponential(TIME_CONSTANT_STDP_DEPRESSION, AMPLITUDE_STDP_DEPRESSION),
]


class NeuralNetwork(SpikingNetwork):
    def __init__(
        self,
        *,
        weight_matrix: npt.ArrayLike,
        weight_max=0.7 / 12,
        rates_spontaneous=0.002,
        stdp_function=None,
        learning_rates=0.14,
        is_learning=True,
        rng: np.random.Generator,
    ):
        super().__init__()

        neuron_count = len(np.asarray(weight_matrix))
        if stdp_function is None:
            stdp_function = STDPFunction(SYNAPTIC_EXPONENTIALS, SYNAPTIC_EXPONENTIALS)

        self.neurons = NeuronPopulation(
            neuron_count, rates_spontaneous=rates_spontaneous, rng=rng
        )
        self.weights = Weights(weight_matrix, upper_bounds=weight_max)
        self.stdp = SpikeTimingDependentPlasticity(
            neuron_count, learning_rates=learning_rates, stdp_function=stdp_function
        )

        self.is_learning = is_learning

    def _get_next_spike(self):
        interspike_interval, spiking_neuron, spike_type = self.neurons.draw_next_spike()

        return NeuronalSpike(
            self.time + interspike_interval, spiking_neuron, spike_type
        )

    def _evolve_in_time(self, duration: float):
        super()._evolve_in_time(duration)
        self.neurons.let_evoked_rates_decay_for(duration)
        self.stdp.let_synaptic_traces_decay_for(duration)

    def _process_spike(self, spike: NeuronalSpike):
        super()._process_spike(spike)
        self.neurons.rates_evoked += self.weights.matrix[:, spike.neuron]
        self.stdp.increase_synaptic_traces_of(spike.neuron)

        if self.is_learning:
            weight_changes = self.stdp.get_weight_changes()
            self.weights.matrix[spike.neuron, :] += weight_changes.to_neuron
            self.weights.matrix[:, spike.neuron] += weight_changes.from_neuron
            self.weights.clip_to_bounds()


@dataclass
class NeuronalSpike(Spike):
    neuron: int
    spike_type: SpikeType
