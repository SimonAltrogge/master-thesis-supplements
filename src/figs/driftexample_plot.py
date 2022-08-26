import numpy as np
import matplotlib.pyplot as plt
from clusterings import Clustering
from mynetworks.plotting import (
    get_sorted_weight_matrix,
    get_sorting_from_clustering,
    plot_weights,
)

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
FINAL_DAY = 55.5

weight_matrix_initial = np.load(f"{DATA_DIR}/weight_matrix_initial.npy")
weight_matrix_after_16h = np.load(f"{DATA_DIR}/weight_matrix_after_16h.npy")
weight_matrix_final = np.load(f"{DATA_DIR}/weight_matrix_final.npy")
clustering_membership_final = np.load(
    f"{DATA_DIR}/clustering_membership_final.npy"
)

sorting_after_16h = list(range(54))
sorting_after_16h[13] = 52
sorting_after_16h[36] = 13
sorting_after_16h[37] = 49
sorting_after_16h[49] = 36
sorting_after_16h[52] = 37

sorting_final = get_sorting_from_clustering(Clustering(clustering_membership_final))


def set_ticks(ax):
    ax.set_xticks(np.arange(17, 54, 18), []) # , np.arange(17, 54, 18) + 1)
    ax.set_xticks(np.arange(5, 54, 6), [], minor=True)
    ax.set_yticks(np.arange(17, 54, 18), []) # , np.arange(17, 54, 18) + 1)
    ax.set_yticks(np.arange(5, 54, 6), [], minor=True)
    # ax.tick_params(axis='both', which='major', labelsize="small", pad=2)


fig, axs = plt.subplots(2, 3, figsize=(6.045, 3.705), sharex="row", sharey="row", constrained_layout=True)

axs[0, 0].set_ylabel("Weights")
axs[1, 0].set_ylabel("Reordered")

axs[1, 0].set_xlabel("Initially")
axs[1, 1].set_xlabel(r"After \qty{16}{\hour}")
axs[1, 2].set_xlabel(r"After \qty{" + str(FINAL_DAY) + r"}{\day}")

mat = plot_weights(
    axs[0, 0], weight_matrix_initial, weight_max=WEIGHT_MAX, normalize=True
)
plot_weights(axs[0, 1], weight_matrix_after_16h, weight_max=WEIGHT_MAX, normalize=True)
plot_weights(
    axs[0, 2],
    weight_matrix_final,
    weight_max=WEIGHT_MAX,
    normalize=True,
)
plot_weights(axs[1, 0], weight_matrix_initial, weight_max=WEIGHT_MAX, normalize=True)
plot_weights(
    axs[1, 1],
    get_sorted_weight_matrix(
        weight_matrix_after_16h,
        sorting_after_16h,
    ),
    weight_max=WEIGHT_MAX,
    normalize=True,
)
plot_weights(
    axs[1, 2],
    get_sorted_weight_matrix(
        weight_matrix_final,
        sorting_final,
    ),
    weight_max=WEIGHT_MAX,
    normalize=True,
)

for ax in axs[0]:
    set_ticks(ax)

for ax in axs[1]:
    ax.set_xticks([])
    ax.set_yticks([])

cbar = fig.colorbar(mat, ax=axs, ticks=[0, 1], pad=0.02, aspect=40)
cbar.set_label(r"Weight normalized by $w_\text{max}$", rotation=-90)

fig.align_labels()

plt.savefig(f"{FIGS_DIR}/drift-example.pdf")
