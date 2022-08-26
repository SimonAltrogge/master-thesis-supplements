import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import cm
from mynetworks.plotting import plot_colored_stem, weighted_moving_average

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

blues = cm.get_cmap("Blues")
viridis = cm.get_cmap("viridis")

# TIME (measured in units of time_constant_rates_evoked=10ms)
millisecond = 1 / 10
second = 100
minute = 60 * second
hour = 60 * minute
day = 24 * hour

# PARAMETERS
WEIGHT_MAX = 0.7 / 12
SCALE = 0.017
DATA_DIR = "../data/driftexample"

similarity_npz = np.load(f"{DATA_DIR}/similarity_history.npz")
weights_npz = np.load(f"{DATA_DIR}/weights_history.npz")
history_npz = np.load(f"{DATA_DIR}/history.npz")

fig, axs = plt.subplots(3, 1, figsize=(4.8, 6.0), sharex=True, gridspec_kw=dict(hspace=0.1), constrained_layout=True)


# similarity
ax = axs[0]
times = similarity_npz["time"] / day
with_previous = similarity_npz["with_previous"]
with_initial = similarity_npz["with_initial"]

ax.axhline(
    0.2,
    color="black",
    linewidth=0.8,
    linestyle="--",
    dashes=((3.7 * 3.5 / 3.7) / 0.8, (1.6 * 3.5 / 3.7) / 0.8),
)

plot_colored_stem(
    ax, times, with_previous[:, 0, 0], color=viridis(0.40), markerfmt=" ", bottom=1.0
)
plot_colored_stem(
    ax, times, with_previous[:, 1, 1], color=viridis(0.55), markerfmt=" ", bottom=1.0
)
plot_colored_stem(
    ax, times, with_previous[:, 2, 2], color=viridis(0.70), markerfmt=" ", bottom=1.0
)

ax.step(times, with_initial[:, 1, 1], where="post", color=blues(0.8))
ax.step(times, with_initial[:, 2, 2], where="post", color=blues(0.6))
ax.step(times, with_initial[:, 0, 0], where="post", color=blues(1.0))

ax.set_xlim(0.0, np.amax(times) + 0.2)
ax.set_ylim(0.0, 1.05)
ax.set_yticks(np.linspace(0.0, 1.0, 6))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ax.set_xlabel(r"Time / \unit{\day}")
ax.set_ylabel("Similarity")


# average weight
ax = axs[1]
times = weights_npz["time"] / day
avg = weights_npz["avg"]
avg_within_current = weights_npz["avg_within_current"]
avg_within_initial = weights_npz["avg_within_initial"]

gaussian = norm(loc=0, scale=SCALE).pdf
samples = np.linspace(np.amin(times), np.amax(times), 500)

ax.plot(
    samples,
    weighted_moving_average(times, avg_within_current[:, 0], gaussian)(samples)
    / WEIGHT_MAX,
    color=viridis(0.40),
)
ax.plot(
    samples,
    weighted_moving_average(times, avg_within_current[:, 1], gaussian)(samples)
    / WEIGHT_MAX,
    color=viridis(0.55),
)
ax.plot(
    samples,
    weighted_moving_average(times, avg_within_current[:, 2], gaussian)(samples)
    / WEIGHT_MAX,
    color=viridis(0.70),
)

ax.plot(
    samples,
    weighted_moving_average(times, avg_within_initial[:, 0], gaussian)(samples)
    / WEIGHT_MAX,
    color=blues(1.0),
)
ax.plot(
    samples,
    weighted_moving_average(times, avg_within_initial[:, 1], gaussian)(samples)
    / WEIGHT_MAX,
    color=blues(0.8),
)
ax.plot(
    samples,
    weighted_moving_average(times, avg_within_initial[:, 2], gaussian)(samples)
    / WEIGHT_MAX,
    color=blues(0.6),
)

ax.plot(
    samples,
    weighted_moving_average(times, avg, gaussian)(samples) / WEIGHT_MAX,
    color="tab:gray",
)

ax.set_xlim(0.0, np.amax(times) + 0.2)
ax.set_ylim(0.0, 1.05)
ax.set_yticks(np.linspace(0.0, 1.0, 6))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ax.set_xlabel(r"Time / \unit{\day}")
ax.set_ylabel(r"Average weight / \(w_\text{max}\)")


# modularity
ax = axs[2]
times = history_npz["time"] / day
modularity = history_npz["modularity"]

gaussian = norm(loc=0, scale=SCALE).pdf
samples = np.linspace(np.amin(times), np.amax(times), 500)

ax.axhline(
    2.0 / 3.0,
    color="black",
    linewidth=0.8,
    linestyle="--",
    dashes=((3.7 * 3.5 / 3.7) / 0.8, (1.6 * 3.5 / 3.7) / 0.8),
)

ax.plot(times, modularity, ".", markersize=1.5)
ax.plot(
    samples,
    weighted_moving_average(times, modularity, gaussian)(samples),
    color="tab:orange",
)

ax.set_xlim(0.0, np.amax(times) + 0.2)
ax.set_ylim(0.0, 0.75)
ax.set_yticks(np.arange(0.0, 0.7, 0.2))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_xlabel(r"Time / \unit{\day}")
ax.set_ylabel("Modularity")

fig.align_labels()

plt.savefig("../figs/persistence-example.pdf")
