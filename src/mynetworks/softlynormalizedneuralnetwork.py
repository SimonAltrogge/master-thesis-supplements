import numpy as np
import numpy.typing as npt
from normalization.functions import NormalizationFunction
from normalization.decorators import softly
from mynetworks.neuralnetwork import NeuralNetwork


class SoftlyNormalizedNeuralNetwork(NeuralNetwork):
    def __init__(
        self,
        *,
        weight_matrix: npt.ArrayLike,
        weight_max=0.7 / 12,
        rates_spontaneous=0.002,
        stdp_function=None,
        learning_rates=0.14,
        is_learning=True,
        time_constant_normalization=60 * 60 * 100,
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
        self.soft_normalization = softly(
            normalization_function, time_constant=time_constant_normalization
        )

    def _evolve_in_time(self, duration: float):
        super()._evolve_in_time(duration)
        self.weights.matrix[...] = self.soft_normalization(
            self.weights.matrix, duration
        )
        self.weights.clip_to_bounds()
