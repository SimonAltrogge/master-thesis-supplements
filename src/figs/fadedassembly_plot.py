import numpy as np
import matplotlib.pyplot as plt
from mynetworks import plotting as plot
from mynetworks.plotting import (
    get_sorted_weight_matrix,
    get_sorting_from_clustering,
    plot_weights,
)
from clusterings import Clustering

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

# TIME (measured in units of time_constant_rates_evoked=10ms)
millisecond = 1 / 10
second = 100
minute = 60 * second
hour = 60 * minute
day = 24 * hour

ideal_start = True
tau = 10 * minute
seed = 101

DATA_DIR = "../data/fadedassembly"
FIGS_DIR = "../figs"
WEIGHT_MAX = 0.7 / 12

prefix = f"{DATA_DIR}/ideal_start={ideal_start}-tau={tau:010.1f}-seed={seed}-"

structure_npz = np.load(prefix + "structure_history.npz")
remarkable_npz = np.load(prefix + "remarkable_weight_matrices.npz")

index = 4
structure_index = np.nonzero(structure_npz["time"] == remarkable_npz["time"][index])[0][0]

fig, ax = plt.subplots(figsize=(3, 2.7), constrained_layout=True)

mat = plot_weights(
    ax,
    get_sorted_weight_matrix(
        remarkable_npz["weight_matrix"][index],
        get_sorting_from_clustering(
            Clustering(structure_npz["assemblies_membership"][structure_index])
        ),
    ),
    weight_max = WEIGHT_MAX,
    normalize = True
)

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

plt.savefig(f"{FIGS_DIR}/faded-assembly.pdf")
