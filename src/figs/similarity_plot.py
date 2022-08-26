import numpy as np

# TIME (measured in units of time_constant_rates_evoked=10ms)
millisecond = 1 / 10
second = 100
minute = 60 * second
hour = 60 * minute
day = 24 * hour

# PARAMETERS
TAU = 4 * hour
SEED_RNG = 1002

prefix = f"../data/scanwhichtau/tau={TAU:08.0f}-seed={SEED_RNG}-"
similarity_npz = np.load(prefix + "similarity_history.npz")

import matplotlib.pyplot as plt
from matplotlib import cm
from mynetworks.plotting import plot_colored_stem

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

times = similarity_npz["time"] / day
with_previous = similarity_npz["with_previous"]
with_initial = similarity_npz["with_initial"]

blues = cm.get_cmap("Blues")
viridis = cm.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

ax.axhline(
    0.2,
    color="black",
    linewidth=0.8,
    linestyle="--",
    dashes=((3.7 * 3.5 / 3.7) / 0.8, (1.6 * 3.5 / 3.7) / 0.8),
)

plot_colored_stem(
    ax, times, with_previous[:, 0, 0], color=viridis(0.30), markerfmt=" ", bottom=1.0
)
plot_colored_stem(
    ax, times, with_previous[:, 1, 1], color=viridis(0.45), markerfmt=" ", bottom=1.0
)
plot_colored_stem(
    ax, times, with_previous[:, 2, 2], color=viridis(0.60), markerfmt=" ", bottom=1.0
)

ax.step(times, with_initial[:, 2, 2], where="post", color=blues(0.6))
ax.step(times, with_initial[:, 0, 0], where="post", color=blues(1.0))
ax.step(times, with_initial[:, 1, 1], where="post", color=blues(0.8))

ax.set_xlim(0.0, np.amax(times) + 0.2)
ax.set_ylim(0.0, 1.05)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_xlabel(r"Time / \unit{\day}")
ax.set_ylabel("Similarity")

plt.savefig("../figs/similarity.pdf")
