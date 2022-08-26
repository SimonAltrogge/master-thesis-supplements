from typing import overload
from collections.abc import Sequence
import numpy as np
import numpy.typing as npt


class Layers:
    def __init__(
        self, neuron_count_by_layer: Sequence[int] | npt.NDArray[np.int_]
    ) -> None:
        self.neuron_count_by_layer = np.array(neuron_count_by_layer)

    @property
    def neuron_count_total(self) -> int:
        return np.sum(self.neuron_count_by_layer)

    def __len__(self) -> int:
        return len(self.neuron_count_by_layer)

    @overload
    def __getitem__(self, key: int) -> slice:
        ...

    @overload
    def __getitem__(self, key: tuple[int, int]) -> tuple[slice, slice]:
        ...

    def __getitem__(self, key: int | tuple[int, int]) -> slice | tuple[slice, slice]:
        if isinstance(key, tuple):
            return self.get_block_slices(to_layer_index=key[0], from_layer_index=key[1])

        return self.get_layer_slice(key)

    def get_block_slices(
        self, to_layer_index: int, from_layer_index: int
    ) -> tuple[slice, slice]:
        return (
            self.get_layer_slice(to_layer_index),
            self.get_layer_slice(from_layer_index),
        )

    def get_layer_slice(self, layer_index: int) -> slice:
        first_neuron = self.get_first_neuron_of(layer_index)
        neuron_count = self.neuron_count_by_layer[layer_index]

        return slice(first_neuron, first_neuron + neuron_count)

    def get_layer_indices(self, layer_index: int) -> npt.NDArray[np.int_]:
        first_neuron = self.get_first_neuron_of(layer_index)
        neuron_count = self.neuron_count_by_layer[layer_index]

        return np.arange(first_neuron, first_neuron + neuron_count)

    def get_first_neuron_of(self, layer_index: int) -> int:
        return np.sum(self.neuron_count_by_layer[:layer_index])
