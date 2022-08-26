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
WEIGHT_MAX = 0.7 / 12

prefix = f"../data/scanwhichtau/tau={TAU:08.0f}-seed={SEED_RNG}-"
weights_npz = np.load(prefix + "weights_history.npz")

from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import cm
from mynetworks.plotting import weighted_moving_average

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

times = weights_npz["time"] / day
avg = weights_npz["avg"]
avg_within_current = weights_npz["avg_within_current"]
avg_within_initial = weights_npz["avg_within_initial"]

blues = cm.get_cmap("Blues")
viridis = cm.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

gaussian = norm(loc=0, scale=0.015).pdf
samples = np.linspace(np.amin(times), np.amax(times), 500)

ax.plot(
    samples,
    weighted_moving_average(times, avg_within_current[:, 0], gaussian)(samples)
    / WEIGHT_MAX,
    color=viridis(0.30),
)
ax.plot(
    samples,
    weighted_moving_average(times, avg_within_current[:, 1], gaussian)(samples)
    / WEIGHT_MAX,
    color=viridis(0.45),
)
ax.plot(
    samples,
    weighted_moving_average(times, avg_within_current[:, 2], gaussian)(samples)
    / WEIGHT_MAX,
    color=viridis(0.60),
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

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_xlabel(r"Time / \unit{\day}")
ax.set_ylabel(r"Average weight / \(w_\text{max}\)")

plt.savefig("../figs/average-weight.pdf")
