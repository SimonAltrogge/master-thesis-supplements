import numpy as np
import numpy.typing as npt
from normalization.functions import NormalizationFunction
from mynetworks.neuralnetwork import NeuralNetwork, NeuronalSpike


class HardlyNormalizedNeuralNetwork(NeuralNetwork):
    def __init__(
        self,
        *,
        weight_matrix: npt.ArrayLike,
        weight_max=0.7 / 12,
        rates_spontaneous=0.002,
        stdp_function=None,
        learning_rates=0.14,
        is_learning=True,
        normalization_function: NormalizationFunction,
        rng: np.random.Generator,
    ):
        super().__init__(
            weight_matrix=weight_matrix,
            weight_max=weight_max,
            rates_spontaneous=rates_spontaneous,
            stdp_function=stdp_function,
            learning_rates=learning_rates,
            is_learning=is_learning,
            rng=rng,
        )
        self.normalization = normalization_function

    def _process_spike(self, spike: NeuronalSpike):
        super()._process_spike(spike)
        self.weights.matrix[...] = self.normalization(self.weights.matrix)
        self.weights.clip_to_bounds()
