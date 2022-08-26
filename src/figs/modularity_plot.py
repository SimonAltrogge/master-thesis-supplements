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
history_npz = np.load(prefix + "history.npz")

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

times = history_npz["time"] / day
modularity = history_npz["modularity"]

blues = cm.get_cmap("Blues")

fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

gaussian = norm(loc=0, scale=0.1).pdf
samples = np.linspace(np.amin(times), np.amax(times), 500)

ax.axhline(
    2.0 / 3.0,
    color="black",
    linewidth=0.8,
    linestyle="--",
    dashes=((3.7 * 3.5 / 3.7) / 0.8, (1.6 * 3.5 / 3.7) / 0.8),
)

ax.plot(times, modularity, ".", markersize=1.5)
ax.plot(samples, weighted_moving_average(times, modularity, gaussian)(samples), color="tab:orange")

ax.set_xlim(0.0, np.amax(times) + 0.2)
ax.set_ylim(0.0, 0.75)
ax.set_yticks(np.arange(0.0, 0.7, 0.2))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_xlabel(r"Time / \unit{\day}")
ax.set_ylabel("Modularity")

plt.savefig("../figs/modularity.pdf")
