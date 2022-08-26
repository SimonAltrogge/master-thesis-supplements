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

TIME_END = 50 * day

with tqdm(
    initial=network.time,
    total=TIME_END,
    mininterval=1,
    maxinterval=60,
    smoothing=0.7,
    unit="z",  # use 'zeconds' as simulation time unit to distinguish between s/z and z/s
    unit_scale=(10 * 1e-3),
    ncols=160,
    bar_format="{percentage:5.1f}%|{bar:20}| {n:"
    + f"{len(str(int(TIME_END * 10 * 1e-3))) + 6}"
    + ".6f}{unit}/{total:.0f}{unit} [{elapsed}<{remaining}, {rate_fmt}]",
# with tqdm(
#     initial=network.time,
#     mininterval=1,
#     maxinterval=60,
#     smoothing=0.7,
#     unit="z",  # use 'zeconds' as simulation time unit to distinguish between s/z and z/s
#     unit_scale=(10 * 1e-3),
#     ncols=160,
) as progress_bar:

    def update_progress_bar(_, amount):
        progress_bar.update(amount)

    network.add_event_listener("post_time_evolution", update_progress_bar)

    network.simulate_for(TIME_END - network.time)
    # network.simulate_while(lambda: not np.all(is_completely_remodeled_by_cluster))

    network.remove_event_listener("post_time_evolution", update_progress_bar)

    # this is necessary, because the last spike will overshoot TIME_END by a bit
    progress_bar.total = progress_bar.n

print(f"avg. rate: {(network.spike_count / (network.time / second)) / network.neurons.neuron_count:.3f}")
