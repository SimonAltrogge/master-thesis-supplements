import numpy as np
import matplotlib.pyplot as plt
from layers import Layers
from clusterings import Clustering
from mynetworks.plotting import get_sorted_weight_matrix, get_layer_contained_sorting_from_clustering, plot_weights

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

DATA_DIR = "../data/driftexample"
FIGS_DIR = "../figs"
WEIGHT_MAX = 0.7 / 12

layers = Layers([27, 27])
weight_matrix_final = np.load(f"{DATA_DIR}/weight_matrix_final.npy")
clustering_membership_final = np.load(f"{DATA_DIR}/clustering_membership_final.npy")
clustering_final = Clustering(clustering_membership_final)

fig, ax = plt.subplots(figsize=(3, 2.7), constrained_layout=True)

mat = plot_weights(ax, get_sorted_weight_matrix(weight_matrix_final, get_layer_contained_sorting_from_clustering(clustering_final, layers)), weight_max=WEIGHT_MAX, normalize=True)
ax.axhline(26.5, linewidth=0.8, color="black")
ax.axvline(26.5, linewidth=0.8, color="black")
ax.set_xticks([])
ax.set_yticks([])

cbar = fig.colorbar(mat, ax=ax, ticks=[0, 1], pad=0.03, aspect=40)
cbar_font_size = 8
cbar.ax.tick_params(labelsize=cbar_font_size)
cbar.set_label(
    r"Weight normalized by $w_\text{max}$",
    size=cbar_font_size,
    rotation=-90,
    labelpad=0.0,
)

plt.savefig(f"{FIGS_DIR}/finalweight-example.pdf")
