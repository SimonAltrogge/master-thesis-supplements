import numpy as np
from clusterings import Clustering
from collections import defaultdict

millisecond = 1 / 10
second = 100
minute = 60 * second
hour = 60 * minute
day = 24 * hour

DATA_DIR = "../data/scanwhichtaushuffled"
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
    r"1 minute",
    r"10 minutes",
    r"1 hours",
    r"4 hours",
    r"12 hours",
    r"1 days",
    r"3 days",
    r"10 days",
    "None",
)
seed_rng_list = 2000 + np.arange(36)

for index, time_constant in enumerate(time_constants):
    counts = defaultdict(int)
    final_number_of_clusters = []
    prefix = "tau=None" if time_constant is None else f"tau={time_constant:08.0f}"

    for seed_rng in seed_rng_list:
        if time_constant == 60000 and seed_rng == 2035:
            counts["invalid"] += 1
            counts["not_original_clusters"] += 1
            final_number_of_clusters.append(1)
            continue

        results = np.load(
            f"{DATA_DIR}/{prefix}-seed={seed_rng}-results.npz",
            allow_pickle=True,
        )
        history = np.load(
            f"{DATA_DIR}/{prefix}-seed={seed_rng}-history.npz",
            allow_pickle=True,
        )

        is_valid = results["has_original_clusters"] and results["had_proper_clusters"]

        counts["valid"] += int(is_valid)
        counts["invalid"] += int(not is_valid)
        counts["not_original_clusters"] += int(not results["has_original_clusters"])
        counts["not_proper_clusters"] += int(not results["had_proper_clusters"])
        counts["both"] += int(
            (not results["has_original_clusters"])
            and (not results["had_proper_clusters"])
        )
        final_number_of_clusters.append(len(Clustering(history["membership"][-1])))

    print(f"tau={time_constant_strings[index]}:")
    print(f"valid: {counts['valid']}")
    print(f"invalid: {counts['invalid']}")
    print(f"not_original_clusters: {counts['not_original_clusters']}")
    print(f"not_proper_clusters: {counts['not_proper_clusters']}")
    print(f"both: {counts['both']}")
    print(f"#assemblies: {np.unique(final_number_of_clusters, return_counts=True)}")
    print("----------")
