import glob
import numpy as np

millisecond = 1 / 10
second = 100
minute = 60 * second
hour = 60 * minute
day = 24 * hour

DATA_DIR = "../data/scanwhichtau"
FIGS_DIR = "../figs"

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

final_times_by_time_constant = {time_constant: [] for time_constant in time_constants}
for filepath in glob.glob(f"{DATA_DIR}/*-results.npz"):
    results = np.load(filepath, allow_pickle=True)

    is_valid = results["has_original_clusters"] and results["had_proper_clusters"]
    if not is_valid:
        continue

    if "time_constant_normalization" in results:
        time_constant = results["time_constant_normalization"].item()
    else:
        time_constant = None

    final_times_by_time_constant[time_constant].append(results["final_time"].item() / day)

for value in final_times_by_time_constant.values():
    print(np.mean(value))

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

plt.boxplot(
    final_times_by_time_constant.values(),
    labels=time_constant_strings,
    patch_artist=True,
    medianprops={"color": "white", "linewidth": 0.8},
    boxprops={"facecolor": "C0", "edgecolor": "white", "linewidth": 0.8},
    whiskerprops={"color": "C0", "linewidth": 1.5},
    capprops={"color": "C0", "linewidth": 1.5},
    flierprops={"markeredgecolor": "C0", "alpha":0.5},
)
plt.savefig(f"{FIGS_DIR}/drift-speed.pdf")
