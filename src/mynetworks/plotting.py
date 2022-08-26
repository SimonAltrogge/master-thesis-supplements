import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from clusterings import redetect_clustering


def get_sorted_weight_matrix(weight_matrix, sorting):
    return weight_matrix[np.ix_(sorting, sorting)]


def get_sorting_from_clustering(clustering):
    return np.argsort(clustering.membership)


def get_layer_contained_sorting_from_clustering(clustering, layers):
    sorting = np.empty(clustering.element_count(), dtype=int)
    for layer_index in range(len(layers)):
        layer = layers.get_layer_indices(layer_index)
        membership_within_layer = clustering.membership[layer]
        partial_sorting = np.argsort(membership_within_layer) + layer[0]
        sorting[layer] = partial_sorting

    return sorting


def weighted_moving_average(x, y, weight_function):
    def wmavg(samples):
        distances = x - samples[:, np.newaxis]
        weights = weight_function(distances)

        return np.average(np.broadcast_to(y, weights.shape), axis=1, weights=weights)

    return wmavg


def weighted_moving_average_err(x, y_err, weight_function):
    def wmavg_err(samples):
        distances = x - samples[:, np.newaxis]
        weights = weight_function(distances)

        return np.sqrt(np.sum((y_err * weights) ** 2, axis=1)) / np.sum(weights, axis=1)

    return wmavg_err


def plot_weights(ax, weight_matrix, *, weight_max=None, normalize=False, **param_dict):
    defaults = {"interpolation": "none", "cmap": "Blues", "vmin": 0.0}
    if weight_max is not None:
        if normalize:
            weight_matrix = weight_matrix / weight_max
            defaults["vmax"] = 1.0
        else:
            defaults["vmax"] = weight_max

    image = ax.matshow(weight_matrix, **(defaults | param_dict))

    ax.xaxis.set_ticks_position("top")

    return image


def plot_colored_stem(ax, *args, color, alpha=None, **kwargs):
    from matplotlib import colors

    color = colors.to_rgba(color, alpha)

    stem_container = ax.stem(*args, **kwargs)

    markerline, stemlines, baseline = stem_container

    markerline.set_color(color)
    stemlines.set_color(color)
    baseline.set_color(color)

    return stem_container


def weight_matrix_views_plotter(
    weights, layers, clustering_prior=None, *, random_state
):
    fig, axs = plt.subplots(
        1, 3, sharex=True, sharey=True, constrained_layout=True, dpi=100, figsize=(8, 3)
    )
    weight_matrix_scaled = weights.matrix / weights.upper_bounds

    vrange_dict = {"vmin": 0.0, "vmax": max(np.amax(weight_matrix_scaled), 1.0)}

    img = axs[0].imshow(weight_matrix_scaled, **vrange_dict)
    axs[0].xaxis.set_minor_locator(AutoMinorLocator(3))
    axs[0].xaxis.set_major_locator(MultipleLocator(9))
    axs[0].yaxis.set_minor_locator(AutoMinorLocator(3))
    axs[0].yaxis.set_major_locator(MultipleLocator(9))
    axs[0].set_title("unsorted")

    clustering = redetect_clustering(
        weights.matrix, clustering_prior, rel_tol=1e-2, random_state=random_state
    )
    if len(clustering) == 0:
        axs[1].imshow(np.full_like(weight_matrix_scaled, np.nan), **vrange_dict)
        axs[2].imshow(np.full_like(weight_matrix_scaled, np.nan), **vrange_dict)
    else:
        layer_contained_sorting = get_layer_contained_sorting_from_clustering(
            clustering, layers
        )
        axs[1].imshow(
            get_sorted_weight_matrix(weight_matrix_scaled, layer_contained_sorting),
            **vrange_dict
        )
        sorting = get_sorting_from_clustering(clustering)
        axs[2].imshow(
            get_sorted_weight_matrix(weight_matrix_scaled, sorting), **vrange_dict
        )
    axs[1].set_title("sorted within layers")
    axs[2].set_title("sorted")

    fig.colorbar(img, ax=axs, shrink=0.7)

    return fig
