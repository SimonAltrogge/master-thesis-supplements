from typing import Any, Protocol
import numpy as np
import numpy.typing as npt


def normalize_additively(
    matrix: npt.ArrayLike, norm_values: npt.ArrayLike = 1.0, *, axis: int
) -> npt.NDArray[np.float_]:
    matrix = np.asarray(matrix)
    normalization_summands = _get_normalization_summands(matrix, norm_values, axis=axis)

    return matrix + normalization_summands


def _get_normalization_summands(
    matrix: npt.ArrayLike, norm_values: npt.ArrayLike = 1.0, *, axis: int
) -> npt.NDArray[np.float_]:
    matrix = np.asarray(matrix)
    sums = np.sum(matrix, axis=axis, keepdims=True)

    return (norm_values - sums) / matrix.shape[axis]


def normalize_multiplicatively(
    matrix: npt.ArrayLike, norm_values: npt.ArrayLike = 1.0, *, axis: int
) -> npt.NDArray[np.float_]:
    matrix = np.asarray(matrix)
    normalization_factors = _get_normalization_factors(matrix, norm_values, axis=axis)

    return matrix * normalization_factors


def _get_normalization_factors(
    matrix: npt.ArrayLike, norm_values: npt.ArrayLike = 1.0, *, axis: int
) -> npt.NDArray[np.float_]:
    matrix = np.asarray(matrix)
    sums = np.sum(matrix, axis=axis, keepdims=True)

    return norm_values / sums


class NormalizationFunction(Protocol):
    def __call__(
        self, matrix: npt.ArrayLike, *args: Any, **kwargs: Any
    ) -> npt.NDArray[np.float_]:
        ...
