import warnings
import math
import numpy as np
import xarray as xr
from clusterings import Clustering
from layers import Layers
from normalization.functions import NormalizationFunction
from normalization.decorators import blockwise, columns_and_rows_once
from scipy.stats import mannwhitneyu


def get_remaining_amplitude_for_symmetric_stdp(
    integral, time_constant, other_time_constant, other_amplitude
):
    return (integral / 2 - other_time_constant * other_amplitude) / time_constant


def get_membership_for_clustering_spread_evenly_over_layers(
    layers: Layers, cluster_count: int
):
    membership = np.empty(0, dtype=int)

    for layer_index in range(len(layers)):
        neuron_count_of_layer = layers.neuron_count_by_layer[layer_index]
        neuron_count_by_cluster_of_layer = np.full(
            cluster_count, neuron_count_of_layer // cluster_count
        )
        leftover_neuron_count = neuron_count_of_layer % cluster_count

        if leftover_neuron_count != 0:
            if layer_index == 0:
                clusters_to_add_leftover_neurons_to = np.arange(leftover_neuron_count)
            else:
                _, neuron_count_by_cluster_so_far = np.unique(
                    membership, return_counts=True
                )
                clusters_to_add_leftover_neurons_to = (
                    np.arange(leftover_neuron_count)
                    + np.argmin(neuron_count_by_cluster_so_far)
                ) % cluster_count

            neuron_count_by_cluster_of_layer[clusters_to_add_leftover_neurons_to] += 1

        partial_membership = np.repeat(
            np.arange(cluster_count), neuron_count_by_cluster_of_layer
        )
        membership = np.concatenate([membership, partial_membership])

    return membership


def construct_weight_matrix_from_clustering(
    clustering: Clustering,
    upper_bounds=1.0,
    *,
    rng: np.random.Generator | None = None,
    has_self_connections=False,
):
    are_connected = np.equal(*np.ix_(clustering.membership, clustering.membership))

    if rng:
        connected_weights = upper_bounds * rng.random(are_connected.shape)
    else:
        connected_weights = np.full(are_connected.shape, upper_bounds)

    if not has_self_connections:
        np.fill_diagonal(connected_weights, 0.0)

    return np.where(are_connected, connected_weights, 0.0)


def get_blockwise_column_and_row_normalization_from_layers(
    layers: Layers,
    total_input_and_output_per_neuron: float,
    normalization_function: NormalizationFunction,
):
    blocks = []
    kwargs_by_block_index = []

    for to_layer_index in range(len(layers)):
        neuron_count_to = layers.neuron_count_by_layer[to_layer_index]

        for from_layer_index in range(len(layers)):
            neuron_count_from = layers.neuron_count_by_layer[from_layer_index]

            block = layers[to_layer_index, from_layer_index]

            norm_value_columns = total_input_and_output_per_neuron * (
                neuron_count_to / layers.neuron_count_total
            )
            norm_value_rows = total_input_and_output_per_neuron * (
                neuron_count_from / layers.neuron_count_total
            )

            blocks.append(block)
            kwargs_by_block_index.append(
                {
                    "kwargs_by_axis": [
                        {"norm_values": norm_value_columns},
                        {"norm_values": norm_value_rows},
                    ]
                }
            )

    _check_consistency(blocks, kwargs_by_block_index)

    return blockwise(
        columns_and_rows_once(normalization_function),
        blocks=blocks,
        kwargs_by_block_index=kwargs_by_block_index,
    )


def get_weight_avg_and_std(weights):
    weights_sum = np.sum(weights.matrix)
    weight_count = weights.matrix.size
    if not weights.has_self_connections:
        weight_count -= len(weights.matrix)

    weight_avg = weights_sum / weight_count

    weight_matrix_centered = weights.matrix - weight_avg
    if not weights.has_self_connections:
        np.fill_diagonal(weight_matrix_centered, 0.0)

    weight_std = math.sqrt(np.sum(weight_matrix_centered**2) / weight_count)

    return weight_avg, weight_std


def get_weight_avg_within_clusters(clustering, weights):
    weight_avg_by_cluster = np.empty(len(clustering))

    for cluster_index, cluster in enumerate(clustering.clusters()):
        weights_sum = np.sum(weights.matrix[np.ix_(cluster, cluster)])
        weight_count = len(cluster) ** 2
        if not weights.has_self_connections:
            weight_count -= len(cluster)
        weight_avg = weights_sum / weight_count

        weight_avg_by_cluster[cluster_index] = weight_avg

    return xr.DataArray(
        weight_avg_by_cluster, coords=[("cluster", clustering.labels())]
    )


def mwu_within_clusters(clustering, weights):
    pvalue_by_cluster = np.empty(len(clustering))

    for cluster_index, cluster in enumerate(clustering.clusters()):
        cluster_mask = np.zeros_like(weights.matrix, dtype=bool)
        cluster_mask[np.ix_(cluster, cluster)] = True

        _, pvalue = mannwhitneyu(
            weights.matrix[cluster_mask],
            weights.matrix[~cluster_mask],
            alternative="greater",
        )

        pvalue_by_cluster[cluster_index] = pvalue

    return xr.DataArray(pvalue_by_cluster, coords=[("cluster", clustering.labels())])


def _check_consistency(blocks, kwargs_by_block_index):
    for block_index, block in enumerate(blocks):
        if block == ...:
            warnings.warn(
                f"consistency cannot be checked for block with index {block_index} "
                f"since the number of rows and columns is unknown for block '{block}'"
            )
            continue

        norm_values_by_axis = [
            kwargs["norm_values"]
            for kwargs in kwargs_by_block_index[block_index]["kwargs_by_axis"]
        ]
        columns_norm_values_sum = _get_norm_values_sum_for_axis(
            0, block, norm_values_by_axis
        )
        rows_norm_values_sum = _get_norm_values_sum_for_axis(
            1, block, norm_values_by_axis
        )

        if not math.isclose(columns_norm_values_sum, rows_norm_values_sum):
            warnings.warn(
                f"block {block_index} is inconsistent: "
                f"normalized columns and rows do not add up to the same value in "
                f"block '{block}' with norm values '{norm_values_by_axis}'"
            )


def _get_norm_values_sum_for_axis(axis, block_indices, norm_values_by_axis):
    if len(np.asarray(norm_values_by_axis[axis]).shape) != 0:
        return np.sum(norm_values_by_axis[axis])

    axis_positions = block_indices[(axis + 1) % 2]
    if isinstance(axis_positions, slice):
        axis_positions_count = (
            (
                (axis_positions.stop - (axis_positions.start or 0))
                / (axis_positions.step or 1)
            )
            + 1
        ) // 1
    elif isinstance(axis_positions, (tuple, list)):
        axis_positions_count = len(axis_positions)
    else:
        axis_positions_count = 1

    return axis_positions_count * norm_values_by_axis[axis]
