import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from clusterings import Clustering
from mynetworks.plotting import (
    get_sorted_weight_matrix,
    get_sorting_from_clustering,
    weighted_moving_average,
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

DATA_DIR = "../data/totalio"
FIGS_DIR = "../figs"
WEIGHT_MAX = 0.7 / 12
FINAL_TIME = 100 * hour

gaussian = norm(loc=0, scale=0.5).pdf
samples = np.linspace(0, FINAL_TIME / hour, 200)
step = 1

fig, axs = plt.subplots(
    3, 3, figsize=(6.045, 5.5575), sharey="row", constrained_layout=True
)

axs[0, 0].set_ylabel("Initially")
axs[1, 0].set_ylabel(
    r"After \qty{" + str(int(FINAL_TIME / hour)) + r"}{\hour}," + r" reordered"
)
axs[2, 0].set_ylabel("Mean total input weight")

axs[2, 0].set_xlabel(r"Time / \unit{\hour}")
axs[2, 1].set_xlabel(r"Time / \unit{\hour}")
axs[2, 2].set_xlabel(r"Time / \unit{\hour}")

SEEDS = (40, 41, 42)
for index, seed in enumerate(SEEDS):
    prefix = f"{DATA_DIR}/seed={seed}"
    weight_matrix_initial = np.load(f"{prefix}-weight_matrix_initial.npy")
    weight_matrix_final = np.load(
        f"{prefix}-weight_matrix_after_{int(FINAL_TIME / hour)}h.npy"
    )
    clustering_membership_final = np.load(
        f"{prefix}-clustering_membership_after_{int(FINAL_TIME / hour)}h.npy"
    )
    totalio_history = np.load(f"{prefix}-totalio_history.npz")

    mat = plot_weights(
        axs[0, index],
        weight_matrix_initial,
        weight_max=WEIGHT_MAX,
        normalize=True,
    )
    axs[0, index].set_xticks([])
    axs[0, index].set_xticklabels([])
    axs[0, index].set_yticks([])
    axs[0, index].set_yticklabels([])

    plot_weights(
        axs[1, index],
        get_sorted_weight_matrix(
            weight_matrix_final,
            get_sorting_from_clustering(Clustering(clustering_membership_final)),
        ),
        weight_max=WEIGHT_MAX,
        normalize=True,
    )
    axs[1, index].set_xticks([])
    axs[1, index].set_xticklabels([])
    axs[1, index].set_yticks([])
    axs[1, index].set_yticklabels([])

    time = totalio_history["time"] / hour
    mask = time <= FINAL_TIME / hour
    time = time[mask]

    print("max: ", np.amax(totalio_history["total_input_by_neuron"]))
    print("min: ", np.amin(totalio_history["total_input_by_neuron"]))

    total_input_avg = np.mean(totalio_history["total_input_by_neuron"], axis=1)[mask]
    total_input_avg_wmavg = weighted_moving_average(time, total_input_avg, gaussian)

    print("mean:", np.mean(total_input_avg)) # [time > 10]))

    total_input_std = np.std(totalio_history["total_input_by_neuron"], axis=1)[mask]
    # total_input_std_wmavg = weighted_moving_average(time, total_input_std, gaussian)

    # axs[2, index].fill_between(
    #     samples,
    #     total_input_avg_wmavg(samples) - total_input_std_wmavg(samples),
    #     total_input_avg_wmavg(samples) + total_input_std_wmavg(samples),
    #     color="tab:green",
    #     alpha=0.5,
    #     linewidth=0.0,
    # )
    axs[2, index].plot(time[::step], total_input_avg[::step], ".", markersize=1.5)
    axs[2, index].plot(samples, total_input_avg_wmavg(samples), color="tab:green")

    axs[2, index].set_xlim(0, int(FINAL_TIME / hour))
    axs[2, index].set_xticks(
        np.arange(
            int(FINAL_TIME / hour) + 1,
            None,
            24 if int(FINAL_TIME / hour) % 24 == 0 else 50,
        )
    )
    axs[2, index].set_ylim(0.0, 0.95)

    axs[2, index].annotate(
        "",
        xy=(0, total_input_avg[0]),
        xytext=(-6, total_input_avg[0]),
        arrowprops=dict(arrowstyle="->", mutation_scale=6.0, shrinkA=0.0, shrinkB=0.0),
    )

    axs[2, index].set_box_aspect(1)
    axs[2, index].spines["top"].set_visible(False)
    axs[2, index].spines["right"].set_visible(False)

cbar = fig.colorbar(mat, ax=axs[:2, :], ticks=[0, 1], pad=0.01, aspect=40, shrink=0.975)
cbar.set_label(r"Weight normalized by $w_\text{max}$", rotation=-90)

fig.align_labels()

plt.savefig(f"{FIGS_DIR}/total-io.pdf")
