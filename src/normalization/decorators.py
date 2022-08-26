from __future__ import annotations
from typing import TypeVar
from collections import defaultdict
from collections.abc import Sequence, Mapping
import numpy as np
import numpy.typing as npt
from normalization.functions import NormalizationFunction


class softly:
    def __init__(
        self,
        normalization_function: NormalizationFunction,
        *args,
        time_constant: float,
        **kwargs,
    ) -> None:
        self.time_constant = time_constant
        self.normalize = normalization_function
        self.args = args
        self.kwargs = kwargs

    def __call__(
        self, matrix: npt.ArrayLike, duration: float, *args, **kwargs
    ) -> npt.NDArray[np.float_]:
        matrix = np.asarray(matrix)
        target = self.normalize(matrix, *(args or self.args), **(self.kwargs | kwargs))

        return target + (matrix - target) * np.exp(-duration / self.time_constant)


class blockwise:
    def __init__(
        self,
        normalization_function: NormalizationFunction,
        *args,
        blocks: Sequence[tuple],
        args_by_block_index: ArgsByIndex | None = None,
        kwargs_by_block_index: KwargsByIndex | None = None,
        **kwargs,
    ) -> None:
        self.blocks = blocks
        self.normalize = normalization_function
        self.args = args
        self.args_by_block_index = _regularize_args_by_index(
            args_by_block_index, len(blocks)
        )
        self.kwargs_by_block_index = _regularize_kwargs_by_index(
            kwargs_by_block_index, len(blocks)
        )
        self.kwargs = kwargs

    def __call__(
        self,
        matrix: npt.ArrayLike,
        *args,
        args_by_block_index: ArgsByIndex | None = None,
        kwargs_by_block_index: KwargsByIndex | None = None,
        **kwargs,
    ) -> npt.NDArray[np.float_]:
        matrix = np.asarray(matrix)
        args_by_block_index = _regularize_args_by_index(
            args_by_block_index, len(self.blocks)
        )
        kwargs_by_block_index = _regularize_kwargs_by_index(
            kwargs_by_block_index, len(self.blocks)
        )

        result = matrix.astype(np.float_, copy=True)
        for index, block in enumerate(self.blocks):
            args_for_block = (
                args_by_block_index[index]
                or self.args_by_block_index[index]
                or args
                or self.args
            )
            kwargs_for_block = (
                self.kwargs
                | kwargs
                | self.kwargs_by_block_index[index]
                | kwargs_by_block_index[index]
            )

            result[block] = self.normalize(
                matrix[block], *args_for_block, **kwargs_for_block
            )

        return result


class columns_and_rows_iteratively:
    def __init__(
        self,
        normalization_function: NormalizationFunction,
        *args,
        order: Sequence[int] | None = None,
        rel_tolerance: float = 1e-3,
        abs_tolerance: float = 1e-9,
        max_iterations: int = 100,
        args_by_axis: ArgsByIndex | None = None,
        kwargs_by_axis: KwargsByIndex | None = None,
        **kwargs,
    ) -> None:
        self.order = order
        self.normalize = normalization_function
        self.args = args
        self.args_by_axis = args_by_axis
        self.kwargs_by_axis = kwargs_by_axis
        self.kwargs = kwargs

        self.rel_tolerance = rel_tolerance
        self.abs_tolerance = abs_tolerance
        self.max_iterations = max_iterations

    def __call__(
        self,
        matrix: npt.ArrayLike,
        *args,
        args_by_axis: ArgsByIndex | None = None,
        kwargs_by_axis: KwargsByIndex | None = None,
        **kwargs,
    ) -> npt.NDArray[np.float_]:
        normalize_once = columns_and_rows_once(
            self.normalize,
            *self.args,
            order=self.order,
            args_by_axis=self.args_by_axis,
            kwargs_by_axis=self.kwargs_by_axis,
            **self.kwargs,
        )

        current_matrix = np.asarray(matrix)
        for _ in range(self.max_iterations):
            previous_matrix, current_matrix = current_matrix, normalize_once(
                current_matrix,
                *args,
                args_by_axis=args_by_axis,
                kwargs_by_axis=kwargs_by_axis,
                **kwargs,
            )

            if np.allclose(
                current_matrix, previous_matrix, self.rel_tolerance, self.abs_tolerance
            ):
                break

        return current_matrix


class columns_and_rows_once:
    def __init__(
        self,
        normalization_function: NormalizationFunction,
        *args,
        order: Sequence[int] | None = None,
        args_by_axis: ArgsByIndex | None = None,
        kwargs_by_axis: KwargsByIndex | None = None,
        **kwargs,
    ) -> None:
        if order is None:
            order = [0, 1]

        self.order = order
        self.normalize = normalization_function
        self.args = args
        self.args_by_axis = _regularize_args_by_index(args_by_axis, len(order))
        self.kwargs_by_axis = _regularize_kwargs_by_index(kwargs_by_axis, len(order))
        self.kwargs = kwargs

    def __call__(
        self,
        matrix: npt.ArrayLike,
        *args,
        args_by_axis: ArgsByIndex | None = None,
        kwargs_by_axis: KwargsByIndex | None = None,
        **kwargs,
    ) -> npt.NDArray[np.float_]:
        args_by_axis = _regularize_args_by_index(args_by_axis, len(self.order))
        kwargs_by_axis = _regularize_kwargs_by_index(kwargs_by_axis, len(self.order))

        current_matrix = np.asarray(matrix)
        for axis in self.order:
            args_for_axis = (
                args_by_axis[axis] or self.args_by_axis[axis] or args or self.args
            )
            kwargs_for_axis = (
                self.kwargs | kwargs | self.kwargs_by_axis[axis] | kwargs_by_axis[axis]
            )

            current_matrix = self.normalize(
                current_matrix,
                axis=axis,
                *args_for_axis,
                **kwargs_for_axis,
            )

        return current_matrix


def _regularize_args_by_index(
    args_by_index: ArgsByIndex | None, length: int
) -> ArgsByIndex:
    if args_by_index is None:
        return [None] * length
    elif isinstance(args_by_index, Mapping):
        return defaultdict(None, args_by_index)

    return args_by_index


def _regularize_kwargs_by_index(
    kwargs_by_index: KwargsByIndex | None, length: int
) -> KwargsByIndex:
    if isinstance(kwargs_by_index, Mapping):
        return defaultdict(dict, kwargs_by_index)
    elif isinstance(kwargs_by_index, Sequence):
        return [{} if item is None else item for item in kwargs_by_index]

    return [{} for _ in range(length)]


T = TypeVar("T")
IntMapping = Sequence[T] | Mapping[int, T]
ArgsByIndex = IntMapping[Sequence | None]
KwargsByIndex = IntMapping[Mapping]
