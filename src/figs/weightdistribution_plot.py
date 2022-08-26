import numpy as np
import matplotlib.pyplot as plt
from clusterings import Clustering, redetect_clustering

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

DATA_DIR = "../data/robustnessdistributed"
FIGS_DIR = "../figs"

SEED_RS = 73
WEIGHT_MAX = 0.7 / 12

rs = np.random.RandomState(SEED_RS)
initial_clustering = Clustering.from_sizes((18, 18, 18))
bins = 10

fig, ax = plt.subplots(figsize=(4.0, 2.7), constrained_layout=True)

intraweights_list = []
for index, tau in enumerate((10 * day, 10 * millisecond)):
    weight_matrix = np.load(
        f"{DATA_DIR}/tau={tau:08.0f}-eta={0.14}-weight_matrix_after_{80}d.npy"
    )
    clustering = redetect_clustering(
        weight_matrix, initial_clustering, rel_tol=1e-2, random_state=rs
    )

    intraweights = np.concatenate(
        [
            weight_matrix[np.ix_(cluster, cluster)][
                ~np.identity(len(cluster), dtype=bool)
            ]
            for cluster in clustering.clusters()
        ]
    )

    intraweights_list.append(intraweights / WEIGHT_MAX)

ax.hist(
    intraweights_list,
    bins=bins,
    weights=[np.ones(len(iw)) / len(iw) for iw in intraweights_list],
    label=[r"\(\tau_\text{norm} = \qty{10}{\day}\)", r"\(\tau_\text{norm} = \qty{10}{\milli\second}\)"],
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlabel(r"Weight / \(w_\text{max}\)")
ax.set_ylabel("Probability")
ax.set_xticks(np.linspace(0, 1, 11))
ax.legend(frameon=False)

plt.savefig(f"{FIGS_DIR}/weight-distribution.pdf")
