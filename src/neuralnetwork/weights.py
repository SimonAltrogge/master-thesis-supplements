import numpy as np
import numpy.typing as npt


class Weights:
    def __init__(
        self,
        matrix: npt.ArrayLike,
        *,
        upper_bounds: npt.ArrayLike | None = None,
        lower_bounds: npt.ArrayLike | None = 0.0,
        has_self_connections: bool = False,
    ) -> None:
        self.matrix = matrix
        self.upper_bounds = upper_bounds
        self.lower_bounds = lower_bounds
        self.has_self_connections = has_self_connections

        self.clip_to_bounds()

    @property
    def matrix(self) -> npt.NDArray[np.float_]:
        return self.__matrix

    @matrix.setter
    def matrix(self, value: npt.ArrayLike) -> None:
        self.__matrix = np.atleast_2d(value).astype(np.float_, copy=True)

    def clip_to_bounds(self) -> None:
        if self.upper_bounds is not None or self.lower_bounds is not None:
            np.clip(self.matrix, self.lower_bounds, self.upper_bounds, out=self.matrix)  # type: ignore

        if not self.has_self_connections:
            np.fill_diagonal(self.matrix, 0.0)


def get_random_weight_matrix(
    neuron_count: int,
    total_input_and_output_per_neuron: float,
    *,
    rng: np.random.Generator,
    has_self_connections: bool = False,
) -> npt.NDArray[np.float_]:
    weight_avg = total_input_and_output_per_neuron / (
        neuron_count - int(not has_self_connections)
    )
    weight_matrix = weight_avg * (2 * rng.random((neuron_count, neuron_count)))

    if not has_self_connections:
        np.fill_diagonal(weight_matrix, 0.0)

    return weight_matrix
