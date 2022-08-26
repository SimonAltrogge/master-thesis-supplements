#!/home/altrogge/anaconda3/envs/newneuro/bin/python
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=long
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simon.altrogge@uni-bonn.de

import os
import sys
import argparse

os.environ["OMP_NUM_THREADS"] = os.environ["SLURM_CPUS_ON_NODE"]
sys.path.append(os.getcwd())

parser = argparse.ArgumentParser(description="Simulate a Hawkes network.")
parser.add_argument(
    "learning_rate",
    metavar="eta",
    type=float,
    help="learning rate of the STDP",
)
# parser.add_argument(
#     "time_constant_normalization",
#     type=int,
#     help="time constant of the soft normalization",
# )
parser.add_argument(
    "seed_rng",
    type=int,
    help="seed for the random number generator",
)
args = parser.parse_args()

import matplotlib.pyplot as plt
from cycler import cycler

plt.rcParams["figure.dpi"] = 100
px = 1 / plt.rcParams["figure.dpi"]
plt.rcParams["figure.figsize"] = [1200 * px, 900 * px]

plt.rcParams.update(
    {
        "figure.facecolor": "#282c34",
        "axes.facecolor": "#21242b",
        "axes.edgecolor": "#5B6268",
        "axes.labelcolor": "#bbc2cf",
        "axes.prop_cycle": cycler(
            color=[
                "#2257A0",
                "#da8548",
                "#98be65",
                "#ff6c6b",
                "#a9a1e1",
                "#5699AF",
                "#c678dd",
                "#51afef",
                "#ECBE7B",
                "#46D9FF",
            ]
        ),
        "text.color": "#bbc2cf",
        "grid.color": "#3f444a",
        "xtick.color": "#5b6268",
        "xtick.labelcolor": "#5b6268",
        "ytick.color": "#5b6268",
        "ytick.labelcolor": "#5b6268",
    }
)
import enum
import numpy as np
import xarray as xr
from tqdm import tqdm
from myenums import HiddenValueEnum, OrderedEnum
from layers import Layers
import clusterings
import normalization.functions
from mynetworks import NeuralNetwork, SoftlyNormalizedNeuralNetwork, util
from neuralnetwork.network import after_interval, every_n_events
# from neuralnetwork.weights import get_random_weight_matrix
from neuralnetwork.history import History
import warnings
from datetime import timedelta
from neuralnetwork.network import every_interval, every_real_time_interval
from mynetworks.plotting import weight_matrix_views_plotter

# in tau=10ms
millisecond = 0.1
second = 100
minute = 60 * second
hour = 60 * minute
day = 24 * hour

SEED_RNG = 42
SEED_RS = 73

NEURONS_PER_LAYER = 27
# NEURON_COUNT = 54
CLUSTER_COUNT = 3

TOTAL_INPUT_AND_OUTPUT_PER_NEURON = 0.7
WEIGHT_MAX = 0.7 / 12

SPONTANEOUS_RATE = 0.002

LEARNING_RATE = 0.14

NORMALIZATION_FUNCTION_STR = "normalize_additively"
TIME_CONSTANT_NORMALIZATION = 4 * hour

SIGNIFICANCE_LEVEL = 0.05
LEARNING_RATE = args.learning_rate
# TIME_CONSTANT_NORMALIZATION = args.time_constant_normalization
SEED_RNG = args.seed_rng
# descriptor = f"tau={TIME_CONSTANT_NORMALIZATION:08.0f}-seed={SEED_RNG}"
descriptor = f"tau=None-seed={SEED_RNG}"

# directory = "scan"
directory = "scan_shuffled"

rng = np.random.default_rng(SEED_RNG)
rs = np.random.RandomState(SEED_RS)

layers = Layers(neuron_count_by_layer=[NEURONS_PER_LAYER, NEURONS_PER_LAYER])
initial_clustering = clusterings.Clustering(
    util.get_membership_for_clustering_spread_evenly_over_layers(layers, CLUSTER_COUNT)
)
rng.shuffle(initial_clustering.membership) # REVIEW: SHUFFLED
# initial_clustering = clusterings.NoClustering(NEURON_COUNT)
initial_weight_matrix = util.construct_weight_matrix_from_clustering(
    initial_clustering, upper_bounds=WEIGHT_MAX, rng=rng
)
# initial_weight_matrix = get_random_weight_matrix(
#     NEURON_COUNT, TOTAL_INPUT_AND_OUTPUT_PER_NEURON, rng=rng
# )
normalization_function = util.get_blockwise_column_and_row_normalization_from_layers(
    layers,
    TOTAL_INPUT_AND_OUTPUT_PER_NEURON,
    getattr(normalization.functions, NORMALIZATION_FUNCTION_STR),
)

# network = SoftlyNormalizedNeuralNetwork(
#     weight_matrix=initial_weight_matrix,
#     weight_max=WEIGHT_MAX,
#     rates_spontaneous=SPONTANEOUS_RATE,
#     learning_rates=LEARNING_RATE,
#     is_learning=False,
#     time_constant_normalization=TIME_CONSTANT_NORMALIZATION,
#     normalization_function=normalization_function,
#     rng=rng,
# )
network = NeuralNetwork(
    weight_matrix=initial_weight_matrix,
    weight_max=WEIGHT_MAX,
    rates_spontaneous=SPONTANEOUS_RATE,
    learning_rates=LEARNING_RATE,
    is_learning=False,
    rng=rng,
)
class Status(OrderedEnum, HiddenValueEnum):
    NORMAL = enum.auto()
    PARTLY_COLLAPSED = enum.auto()
    FULLY_COLLAPSED = enum.auto()


chance_level_of_initial = clusterings.chance_level_jaccard_similarities(initial_clustering)

is_completely_remodeled_by_cluster = xr.DataArray(
    np.full(len(initial_clustering), False),
    coords=[("cluster", initial_clustering.labels())],
)

history = History(
    {
        "time",
        "status",
        "clustering",
        "modularity",
    }
)
similarity_history = History(
    {
        "time",
        "with_initial",
        "with_previous",
        "are_previous",
    }
)
weights_history = History(
    {
        "time",
        "avg",
        "avg_within_initial",
        "avg_within_current",
        "pvalues",
        "are_significantly_larger",
    }
)

detected_initial_clustering, initial_modularity = clusterings.detect_clustering(
    initial_weight_matrix,
    initial_clustering.membership,
    random_state=rs,
    return_modularity=True,
)
if detected_initial_clustering != initial_clustering:
    initial_modularity = np.nan
history.append(
    time=network.time,
    status=Status.NORMAL,
    clustering=initial_clustering,
    modularity=initial_modularity,
)

initial_similarity = clusterings.jaccard_similarities(
    initial_clustering, initial_clustering
)
similarity_history.append(
    time=network.time,
    with_initial=initial_similarity,
    with_previous=xr.full_like(initial_similarity, np.nan),
    are_previous=True,
)

initial_weight_avg, _ = util.get_weight_avg_and_std(network.weights)
initial_weight_avg_within_clusters = util.get_weight_avg_within_clusters(
    initial_clustering, network.weights
)
initial_pvalues = util.mwu_within_clusters(initial_clustering, network.weights)
weights_history.append(
    time=network.time,
    avg=initial_weight_avg,
    avg_within_initial=initial_weight_avg_within_clusters,
    avg_within_current=initial_weight_avg_within_clusters,
    pvalues=initial_pvalues,
    are_significantly_larger=bool(np.all(initial_pvalues <= SIGNIFICANCE_LEVEL)),
)
def analyse_structure(network, *args, **kwargs):
    previous_clustering = history.clustering[-1]

    current_clustering, current_modularity = clusterings.redetect_clustering(
        network.weights.matrix,
        clustering_prior=previous_clustering,
        rel_tol=1e-2,
        random_state=rs,
        return_modularity=True,
    )

    if current_clustering == previous_clustering:
        return

    if not current_clustering:
        current_status = Status.FULLY_COLLAPSED
    elif len(current_clustering) != len(initial_clustering):
        current_status = Status.PARTLY_COLLAPSED
    else:
        current_status = Status.NORMAL

    history.append(
        time=network.time,
        status=current_status,
        clustering=current_clustering,
        modularity=current_modularity,
    )

    similarities_of_current_with_initial_clusters = clusterings.jaccard_similarities(
        current_clustering, initial_clustering
    )
    similarities_of_current_with_previous_clusters = clusterings.jaccard_similarities(
        current_clustering, previous_clustering
    )
    chance_level_of_previous = clusterings.chance_level_jaccard_similarities(
        previous_clustering
    )
    similarity_history.append(
        time=network.time,
        with_initial=similarities_of_current_with_initial_clusters,
        with_previous=similarities_of_current_with_previous_clusters,
        are_previous=(
            len(current_clustering) == len(previous_clustering)
            and bool(
                np.all(
                    np.diagonal(similarities_of_current_with_previous_clusters)
                    >= chance_level_of_previous + (1 - chance_level_of_previous) / 2
                )
            )
        ),
    )

    weight_avg, _ = util.get_weight_avg_and_std(network.weights)
    weight_avg_within_initial_clusters = util.get_weight_avg_within_clusters(
        initial_clustering, network.weights
    )
    weight_avg_within_current_clusters = util.get_weight_avg_within_clusters(
        current_clustering, network.weights
    )
    pvalues = util.mwu_within_clusters(current_clustering, network.weights)
    weights_history.append(
        time=network.time,
        avg=weight_avg,
        avg_within_initial=weight_avg_within_initial_clusters,
        avg_within_current=weight_avg_within_current_clusters,
        pvalues=pvalues,
        are_significantly_larger=bool(np.all(pvalues <= SIGNIFICANCE_LEVEL)),
    )

    if len(current_clustering) == len(initial_clustering):
        are_leq_than_chance = np.diagonal(
            similarities_of_current_with_initial_clusters
        ) <= chance_level_of_initial
    else:
        are_leq_than_chance = np.full_like(is_completely_remodeled_by_cluster, True)

    np.logical_or(
        is_completely_remodeled_by_cluster,
        are_leq_than_chance,
        out=is_completely_remodeled_by_cluster.values,
    )
ANALYSIS_PERIOD = 100
network.add_event_listener(
    "post_spike",
    every_n_events(
        ANALYSIS_PERIOD,
        analyse_structure,
    ),
)
network.add_event_listener(
    "post_spike",
    after_interval(
        1 * minute, lambda network, *_: setattr(network, "is_learning", True)
    ),
)
def take_snapshot(network, *args, **kwargs):
    try:
        np.savez_compressed(
            f"../{directory}/figs/snapshot-{descriptor}-time={network.time:.0f}",
            seed=SEED_RNG,
            # time_constant_normalization=TIME_CONSTANT_NORMALIZATION,
            time=network.time,
            weight_matrix=network.weights.matrix,
        )
    except Exception as err:
        warnings.warn(
            f"saving weight matrix failed for snapshot at time={network.time} with error: {err}"
        )

    try:
        snapshot_fig = weight_matrix_views_plotter(
            network.weights,
            layers,
            clustering_prior=history.clustering[-1],
            random_state=rs,
        )

        snapshot_fig.suptitle(
            # f"tau_norm={TIME_CONSTANT_NORMALIZATION}, "
            f"tau_norm=None, "
            f"seed={SEED_RNG}, "
            f"time={timedelta(seconds=network.time/second)}"
        )
        snapshot_fig.savefig(
            f"../{directory}/figs/snapshot-{descriptor}-time={network.time:.0f}.png"
        )
        plt.close(snapshot_fig)
    except Exception as err:
        plt.close("all")
        warnings.warn(
            f"plotting failed for snapshot at time={network.time} with error: {err}"
        )


network.add_event_listener(
    "post_spike",
    every_interval(1 * day, take_snapshot),
)
network.add_event_listener(
    "post_spike",
    every_real_time_interval(60 * 60, take_snapshot),
)


def get_neuron_count_by_cluster_by_layer(clustering, layers):
    neuron_count_by_cluster_by_layer = np.empty((len(layers), len(clustering)))

    # REVIEW: Should Layers be iterable like Assemblies?
    for layer_index in range(len(layers)):
        for cluster_index, cluster in enumerate(clustering.clusters()):
            neuron_count_by_cluster_by_layer[layer_index, cluster_index] = len(
                np.intersect1d(
                    cluster, layers.get_layer_indices(layer_index), assume_unique=True
                )
            )

    return neuron_count_by_cluster_by_layer


with tqdm(
    initial=network.time,
    mininterval=60,
    maxinterval=60,
    smoothing=0.7,
    unit="z",  # use 'zeconds' as simulation time unit to distinguish between s/z and z/s
    unit_scale=(10 * 1e-3),
    ncols=160,
) as progress_bar:

    def update_progress_bar(_, amount):
        progress_bar.update(amount)

    network.add_event_listener("post_time_evolution", update_progress_bar)
    network.simulate_while(lambda: not np.all(is_completely_remodeled_by_cluster))
    network.remove_event_listener("post_time_evolution", update_progress_bar)

np.savez_compressed(
    f"../{directory}/{descriptor}-results",
    seed_rng=SEED_RNG,
    # time_constant_normalization=TIME_CONSTANT_NORMALIZATION,
    final_time=network.time,
    final_spike_count=network.spike_count,
    final_weight_matrix=network.weights.matrix,
    has_original_clusters=np.all(similarity_history.are_previous),
    had_proper_clusters=np.all(weights_history.are_significantly_larger),
    neuron_count_by_cluster_by_layer=get_neuron_count_by_cluster_by_layer(
        history.clustering[-1], layers
    ),
)

np.savez_compressed(
    f"../{directory}/{descriptor}-history",
    seed_rng=SEED_RNG,
    # time_constant_normalization=TIME_CONSTANT_NORMALIZATION,
    time=np.array(history.time),
    status_value=np.array([status.value for status in history.status]),
    membership=np.array([clustering.membership for clustering in history.clustering]),
    modularity=np.array(history.modularity),
)

np.savez_compressed(
    f"../{directory}/{descriptor}-similarity_history",
    seed_rng=SEED_RNG,
    # time_constant_normalization=TIME_CONSTANT_NORMALIZATION,
    time=np.array(similarity_history.time),
    with_initial=np.array(similarity_history.with_initial),
    with_previous=np.array(similarity_history.with_previous),
    are_previous=np.array(similarity_history.are_previous),
)

np.savez_compressed(
    f"../{directory}/{descriptor}-weights_history",
    seed_rng=SEED_RNG,
    # time_constant_normalization=TIME_CONSTANT_NORMALIZATION,
    time=np.array(weights_history.time),
    avg=np.array(weights_history.avg),
    avg_within_initial=np.array(weights_history.avg_within_initial),
    avg_within_current=np.array(weights_history.avg_within_current),
    pvalues=np.array(weights_history.pvalues),
    are_significantly_larger=np.array(weights_history.are_significantly_larger),
)
