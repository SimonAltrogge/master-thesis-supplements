import itertools
import numpy as np
import matplotlib.pyplot as plt
from clusterings import Clustering, redetect_clustering
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

# TIME (measured in units of time_constant_rates_evoked=10ms)
millisecond = 1 / 10
second = 100
minute = 60 * second
hour = 60 * minute
day = 24 * hour

WEIGHT_MAX = 0.7 / 12
SEED_RS = 73

DATA_DIR = "../data/robustnessdistributed"
FIGS_DIR = "../figs"

time_constants = (10 * day, 1 * hour, 1)
learning_rates = (0.10, 0.14, 0.18)

rs = np.random.RandomState(SEED_RS)

initial_clustering = Clustering.from_sizes((18, 18, 18))

fig, axs = plt.subplots(
    3,
    8,
    gridspec_kw=dict(width_ratios=[1, 1, 0.01, 1, 1, 0.01, 1, 1]),
    figsize=(6.045, 2.92),
    constrained_layout=True,
)
for row, column in itertools.product((0, 1, 2), (2, 5)):
    axs[row, column].set_visible(False)

fig.supxlabel(r"\(\tau_\text{norm}\)", x=0.502, fontsize="medium")
fig.text(0.20, 0.06, r"\qty{10}{\day}", horizontalalignment="center", fontsize="small")
fig.text(0.502, 0.06, r"\qty{1}{\hour}", horizontalalignment="center", fontsize="small")
fig.text(0.81, 0.06, r"\qty{10}{\milli\second}", horizontalalignment="center", fontsize="small")

fig.supylabel(r"\(\eta\)", y=0.52, fontsize="medium")
axs[0, 0].set_ylabel(r"\num{0.10}", fontsize="small")
axs[1, 0].set_ylabel(r"\num{0.14}", fontsize="small")
axs[2, 0].set_ylabel(r"\num{0.18}", fontsize="small")

axs[0, 0].set_xlabel("Weights", fontsize="small")
axs[0, 0].xaxis.set_label_position("top")
axs[0, 1].set_xlabel("Reordered", fontsize="small")
axs[0, 1].xaxis.set_label_position("top")
axs[0, 3].set_xlabel("Weights", fontsize="small")
axs[0, 3].xaxis.set_label_position("top")
axs[0, 4].set_xlabel("Reordered", fontsize="small")
axs[0, 4].xaxis.set_label_position("top")
axs[0, 6].set_xlabel("Weights", fontsize="small")
axs[0, 6].xaxis.set_label_position("top")
axs[0, 7].set_xlabel("Reordered", fontsize="small")
axs[0, 7].xaxis.set_label_position("top")

for index, (time_constant, learning_rate) in enumerate(
    itertools.product(time_constants, learning_rates)
):
    time = 80

    row = index % 3
    column = (index // 3) * 3
    axs_pair = (axs[row, column], axs[row, column + 1])

    if time > 1 and time_constant == 1 * hour and learning_rate == 0.18:
        continue

    matrix = np.load(
        f"{DATA_DIR}/tau={time_constant:08.0f}-eta={learning_rate}-weight_matrix_after_{time}d.npy"
    )

    plot_weights(axs_pair[0], matrix, weight_max=WEIGHT_MAX, normalize=True)
    axs_pair[0].set_xticks([])
    axs_pair[0].set_xticklabels([])
    axs_pair[0].set_yticks([])
    axs_pair[0].set_yticklabels([])

    clustering = redetect_clustering(
        matrix, initial_clustering, rel_tol=1e-2, random_state=rs
    )

    mat = plot_weights(
        axs_pair[1],
        get_sorted_weight_matrix(matrix, get_sorting_from_clustering(clustering)),
        weight_max=WEIGHT_MAX,
        normalize=True,
    )
    axs_pair[1].set_xticks([])
    axs_pair[1].set_xticklabels([])
    axs_pair[1].set_yticks([])
    axs_pair[1].set_yticklabels([])

cbar = fig.colorbar(mat, ax=axs, ticks=[0, 1], pad=0.02, aspect=40)
cbar.ax.tick_params(labelsize="small")
cbar.set_label(
    r"Weight normalized by $w_\text{max}$", size="small", rotation=-90, labelpad=0.0
)

if time > 1:
    axs[2,3].remove()
    axs[2,4].remove()

plt.savefig(f"{FIGS_DIR}/robustness-distributed{time}d.pdf")
