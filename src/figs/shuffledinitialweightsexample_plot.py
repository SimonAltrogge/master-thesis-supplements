import numpy as np
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

DATA_DIR = "../data/scanwhichtaushuffled"
FIGS_DIR = "../figs"
WEIGHT_MAX = 0.7 / 12
seed = 2002

weight_matrix_initial = np.load(f"{DATA_DIR}/seed={seed}-initial.npz")["weight_matrix"]

fig, ax = plt.subplots(figsize=(3, 2.7), constrained_layout=True)

mat = plot_weights(ax, weight_matrix_initial, weight_max=WEIGHT_MAX, normalize=True)
# ax.axhline(26.5, linewidth=0.8, color="black")
# ax.axvline(26.5, linewidth=0.8, color="black")
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

plt.savefig(f"{FIGS_DIR}/shuffledinitialweights-example.pdf")
