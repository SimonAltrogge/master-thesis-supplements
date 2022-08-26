import numpy as np
from clusterings import Clustering
from layers import Layers
from mynetworks.util import (
    get_membership_for_clustering_spread_evenly_over_layers,
    construct_weight_matrix_from_clustering,
)

NEURON_COUNT = 54
LAYER_COUNT = 2
ASSEMBLY_COUNT = 3

WEIGHT_MAX = 0.7 / 12

SEED_RNG = 42

rng = np.random.default_rng(SEED_RNG)
layers = Layers(np.full(LAYER_COUNT, NEURON_COUNT // LAYER_COUNT))
weight_matrix = construct_weight_matrix_from_clustering(
    Clustering(
        get_membership_for_clustering_spread_evenly_over_layers(layers, ASSEMBLY_COUNT)
    ),
    WEIGHT_MAX,
    rng=rng,
)

import numpy.ma as ma
import matplotlib.pyplot as plt
from mynetworks.plotting import plot_weights

plt.rcParams.update(
    {
        "font.family": "serif",
        "text.usetex": True,
        "text.latex.preamble": (
            r"\usepackage{newtxtext}"
            r"\usepackage{newtxmath}"
            r"\usepackage[T1]{fontenc}"
            r"\usepackage{siunitx}"
        ),
    }
)

fig, ax = plt.subplots(figsize=(3.3, 2.96), constrained_layout=True)

assembly_one_mask = np.full_like(weight_matrix, True, dtype=bool)
assembly_one_mask[:9, :9] = False
assembly_one_mask[:9, 27:36] = False
assembly_one_mask[27:36, :9] = False
assembly_one_mask[27:36, 27:36] = False

plot_weights(
    ax,
    ma.masked_array(weight_matrix, ~assembly_one_mask),
    weight_max=WEIGHT_MAX,
    normalize=True,
    cmap="Greys",
)
mat = plot_weights(
    ax,
    ma.masked_array(weight_matrix, assembly_one_mask),
    weight_max=WEIGHT_MAX,
    normalize=True,
)
# mat = plot_weights(ax, weight_matrix, weight_max=WEIGHT_MAX, normalize=True)

ax.set_xlim(-0.5, 53.5)
ax.set_ylim(53.5, -0.5)

ax.set_xticks([26.5], [])
ax.set_xticks([13, 40], ["layer 1", "layer 2"], minor=True)

ax.set_yticks([26.5], [])
ax.set_yticks([13, 40], ["layer 1", "layer 2"], rotation=90, va="center", minor=True)

ax.tick_params(which="major", size=7.0)
ax.tick_params(which="minor", width=0.0, size=0.0)

ax.axhline(26.5, linewidth=0.8, color="black")
ax.axvline(26.5, linewidth=0.8, color="black")

ax.set_xlabel("From")
ax.xaxis.set_label_position("top")

ax.set_ylabel("To")

cbar = fig.colorbar(mat, ax=ax, ticks=[0, 1], pad=0.03, aspect=40)
cbar_font_size = 8
cbar.ax.tick_params(labelsize=cbar_font_size)
cbar.set_label(
    r"Weight normalized by $w_\text{max}$",
    size=cbar_font_size,
    rotation=-90,
    labelpad=0.0,
)

plt.savefig("../figs/norm-explanation.pdf")
