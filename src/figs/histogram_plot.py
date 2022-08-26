import glob
import numpy as np
from scipy.stats import hypergeom, chisquare

millisecond = 1 / 10
second = 100
minute = 60 * second
hour = 60 * minute
day = 24 * hour

DATA_DIR = "../data/scanwhichtau"
FIGS_DIR = "../figs"
LAYER_SIZE = 27

time_constants = (
    1 * minute,
    10 * minute,
    1 * hour,
    4 * hour,
    12 * hour,
    1 * day,
    3 * day,
    10 * day,
    None,
)
time_constant_strings = (
    r"\qty{1}{\minute}",
    r"\qty{10}{\minute}",
    r"\qty{1}{\hour}",
    r"\qty{4}{\hour}",
    r"\qty{12}{\hour}",
    r"\qty{1}{\day}",
    r"\qty{3}{\day}",
    r"\qty{10}{\day}",
    "control",
)

histogram_by_time_constant = {time_constant: [] for time_constant in time_constants}
pmf_by_time_constant = {
    time_constant: np.zeros(LAYER_SIZE + 1) for time_constant in time_constants
}
pvalue_by_time_constant = {}

for filepath in glob.glob(f"{DATA_DIR}/*-results.npz"):
    results = np.load(filepath, allow_pickle=True)

    is_valid = results["has_original_clusters"] and results["had_proper_clusters"]
    if not is_valid:
        continue

    if "time_constant_normalization" in results:
        time_constant = int(results["time_constant_normalization"])
    else:
        time_constant = None

    neuron_count_by_cluster_by_layer = results["neuron_count_by_cluster_by_layer"]
    histogram_by_time_constant[time_constant] += list(
        neuron_count_by_cluster_by_layer.flat
    )

    cluster_sizes = np.sum(neuron_count_by_cluster_by_layer, axis=0)
    neuron_count = np.sum(cluster_sizes)
    for cluster_size in cluster_sizes:
        pmf_by_time_constant[time_constant] += hypergeom.pmf(
            np.arange(LAYER_SIZE + 1), neuron_count, cluster_size, LAYER_SIZE
        )

for time_constant in time_constants:
    pmf = pmf_by_time_constant[time_constant]
    pmf /= np.sum(pmf)

for index, time_constant in enumerate(time_constants):
    pmf = pmf_by_time_constant[time_constant]
    histogram = histogram_by_time_constant[time_constant]

    expected_frequencies = pmf * len(histogram)
    expected_frequencies_merged = {}

    left_limit = 0
    while expected_frequencies[left_limit] < 5:
        for right_limit in range(left_limit + 1, LAYER_SIZE):
            if (
                freq := np.sum(expected_frequencies[left_limit : (right_limit + 1)])
            ) >= 5:
                break
        expected_frequencies_merged[(left_limit, right_limit)] = freq
        left_limit = right_limit + 1
    last_left_limit = left_limit

    right_limit = LAYER_SIZE
    while expected_frequencies[right_limit] < 5:
        for left_limit in range(right_limit - 1, last_left_limit - 1, -1):
            if (
                freq := np.sum(expected_frequencies[left_limit : (right_limit + 1)])
            ) >= 5:
                break
        expected_frequencies_merged[(left_limit, right_limit)] = freq
        right_limit = left_limit - 1
    last_right_limit = right_limit

    for value in range(last_left_limit, last_right_limit + 1):
        expected_frequencies_merged[value] = expected_frequencies[value]

    histogram_merged = {key: 0 for key in expected_frequencies_merged.keys()}
    for value in histogram:
        value = int(value)
        if value in histogram_merged:
            histogram_merged[value] += 1
        else:
            for key in histogram_merged.keys():
                if isinstance(key, tuple):
                    left_limit, right_limit = key
                    if left_limit <= value <= right_limit:
                        histogram_merged[key] += 1
                        break

    res = chisquare(list(histogram_merged.values()), list(expected_frequencies_merged.values()))
    pvalue_by_time_constant[time_constant] = res[1]
    print(time_constant_strings[index], len(histogram), res, res[1] <= 0.01)

import matplotlib.pyplot as plt

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

fig, axs = plt.subplots(
    3, 3, figsize=(6.045, 5.655), sharex=True, sharey=True, constrained_layout=True
)

for ax in axs.flat:
    ax.set_xlim(-0.5, 27.5)
    ax.set_xticks(np.arange(28, None, 9))

    ax.set_ylim(0.0, 0.75)
    ax.set_yticks(np.arange(0.0, 0.7, 0.2))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

for index, time_constant in enumerate(time_constants):
    histogram = histogram_by_time_constant[time_constant]
    pmf = pmf_by_time_constant[time_constant]
    pvalue = pvalue_by_time_constant[time_constant]
    ax = axs[index // 3, index % 3]

    markerline, stemlines, baseline = ax.stem(pmf[pmf > 0.0], basefmt=" ")
    markerline.set_markersize(1.5)

    hist, *_ = ax.hist(
        histogram,
        bins=np.arange(LAYER_SIZE + 1) - 0.5,
        density=True,
        histtype="step",
        color="tab:orange",
    )

    if time_constant is not None:
        title = r"\(\tau_\text{norm} = " + time_constant_strings[index] + "\)"
    else:
        title = time_constant_strings[index]

    ax.annotate(
        title,
        (0.97, 0.97),
        xycoords="axes fraction",
        size=11,
        horizontalalignment="right",
        verticalalignment="top",
    )

    significance = ""
    if pvalue < 0.05:
        significance += "*"
    if pvalue < 0.01:
        significance += "*"
    if pvalue < 0.001:
        significance += "*"

    if significance != "":
        ax.text(
            9, np.amax(hist), significance, ha="center", va="bottom"
        )

fig.supylabel("Probability", y=0.54, fontsize="medium")
fig.supxlabel("Neuron count", x=0.536, fontsize="medium")

plt.savefig(f"{FIGS_DIR}/histogram.pdf")
