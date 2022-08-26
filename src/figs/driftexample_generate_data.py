import numpy as np
from clusterings import Clustering, redetect_clustering
from neuralnetwork.stdp import Exponential, STDPFunction
from mynetworks import NeuralNetwork
import mynetworks.util as util

# TIME (measured in units of time_constant_rates_evoked=10ms)
millisecond = 1 / 10
second = 100
minute = 60 * second
hour = 60 * minute
day = 24 * hour

# PARAMETERS
DATA_DIR = "../data/driftexample"
SEED_RNG = 29
SEED_RS = 73

ASSEMBLY_COUNT = 3
NEURON_COUNT_PER_ASSEMBLY = 18

RATE_SPONTANEOUS = 0.002
WEIGHT_MAX = 0.7 / 12

LEARNING_RATE = 0.14
STDP_INTEGRAL = -(4 / 30)
STDP_POTENTIATION_TIME_CONSTANT = 2.5
STDP_POTENTIATION_AMPLITUDE = 0.08
STDP_DEPRESSION_TIME_CONSTANT = 5.0

# DERIVED VALUES
stdp_depression_amplitude = util.get_remaining_amplitude_for_symmetric_stdp(
    STDP_INTEGRAL,
    STDP_DEPRESSION_TIME_CONSTANT,
    STDP_POTENTIATION_TIME_CONSTANT,
    STDP_POTENTIATION_AMPLITUDE,
)
stdp_potentiation_exponential = Exponential(
    STDP_POTENTIATION_TIME_CONSTANT, STDP_POTENTIATION_AMPLITUDE
)
stdp_depression_exponential = Exponential(
    STDP_DEPRESSION_TIME_CONSTANT, stdp_depression_amplitude
)
stdp_function = STDPFunction(
    presynaptic_exponentials=(
        stdp_potentiation_exponential,
        stdp_depression_exponential,
    ),
    postsynaptic_exponentials=(
        stdp_potentiation_exponential,
        stdp_depression_exponential,
    ),
)

rng = np.random.default_rng(SEED_RNG)
rs = np.random.RandomState(SEED_RS)

initial_clustering = Clustering.from_sizes(
    (NEURON_COUNT_PER_ASSEMBLY,) * ASSEMBLY_COUNT
)
initial_weight_matrix = util.construct_weight_matrix_from_clustering(
    initial_clustering, upper_bounds=WEIGHT_MAX, rng=rng
)
np.save(f"{DATA_DIR}/weight_matrix_initial", initial_weight_matrix)

# NETWORK CONSTRUCTION
network = NeuralNetwork(
    weight_matrix=initial_weight_matrix,
    weight_max=WEIGHT_MAX,
    rates_spontaneous=RATE_SPONTANEOUS,
    stdp_function=stdp_function,
    learning_rates=LEARNING_RATE,
    is_learning=False,
    rng=rng,
)

import xarray as xr
from neuralnetwork.network import every_n_events, every_interval
from neuralnetwork.history import History
import clusterings

chance_level_of_initial = clusterings.chance_level_jaccard_similarities(initial_clustering)

is_completely_remodeled_by_cluster = xr.DataArray(
    np.full(len(initial_clustering), False),
    coords=[("cluster", initial_clustering.labels())],
)

history = History(
    {
        "time",
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
SIGNIFICANCE_LEVEL = 0.05

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

    history.append(
        time=network.time,
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
        are_leq_than_chance = np.full_like(is_completely_remodeled_by_cluster, False)

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

network.simulate_for(1 * minute)
network.is_learning = True

from tqdm import tqdm

network.add_event_listener(
    "post_spike",
    every_interval(
        5 * day,
        lambda network, *args, **kwargs: np.save(
            f"{DATA_DIR}/weight_matrix_after_{int(network.time / day)}d",
            network.weights.matrix,
        ),
    ),
)

# TIME_END = 60 * day
# with tqdm(
#     initial=network.time,
#     total=TIME_END,
#     mininterval=1,
#     maxinterval=60,
#     smoothing=0.7,
#     unit="z",  # use 'zeconds' as simulation time unit to distinguish between s/z and z/s
#     unit_scale=(10 * 1e-3),
#     ncols=160,
#     bar_format="{percentage:5.1f}%|{bar:20}| {n:"
#     + f"{len(str(int(TIME_END * 10 * 1e-3))) + 6}"
#     + ".6f}{unit}/{total:.0f}{unit} [{elapsed}<{remaining}, {rate_fmt}]",
# ) as progress_bar:
with tqdm(
    initial=network.time,
    mininterval=1,
    maxinterval=60,
    smoothing=0.7,
    unit="z",  # use 'zeconds' as simulation time unit to distinguish between s/z and z/s
    unit_scale=(10 * 1e-3),
    ncols=160,
) as progress_bar:

    def update_progress_bar(_, amount):
        progress_bar.update(amount)

    network.add_event_listener("post_time_evolution", update_progress_bar)

    if network.time < 16 * hour:
        network.simulate_for(16 * hour - network.time)
        np.save(f"{DATA_DIR}/weight_matrix_after_16h", network.weights.matrix)

    # network.simulate_for(TIME_END - network.time)
    network.simulate_while(lambda: not np.all(is_completely_remodeled_by_cluster))
    np.save(
        # f"{DATA_DIR}/weight_matrix_after_{int(network.time / day)}d",
        f"{DATA_DIR}/weight_matrix_final",
        network.weights.matrix,
    )

    network.remove_event_listener("post_time_evolution", update_progress_bar)

    # this is necessary, because the last spike will overshoot TIME_END by a bit
    # progress_bar.total = progress_bar.n

np.savez_compressed(
    f"{DATA_DIR}/results",
    final_time=network.time,
    final_spike_count=network.spike_count,
    final_weight_matrix=network.weights.matrix,
    has_original_clusters=np.all(similarity_history.are_previous),
    had_proper_clusters=np.all(weights_history.are_significantly_larger),
)

np.savez_compressed(
    f"{DATA_DIR}/history",
    time=np.array(history.time),
    membership=np.array([clustering.membership for clustering in history.clustering]),
    modularity=np.array(history.modularity),
)

np.savez_compressed(
    f"{DATA_DIR}/similarity_history",
    time=np.array(similarity_history.time),
    with_initial=np.array(similarity_history.with_initial),
    with_previous=np.array(similarity_history.with_previous),
    are_previous=np.array(similarity_history.are_previous),
)

np.savez_compressed(
    f"{DATA_DIR}/weights_history",
    time=np.array(weights_history.time),
    avg=np.array(weights_history.avg),
    avg_within_initial=np.array(weights_history.avg_within_initial),
    avg_within_current=np.array(weights_history.avg_within_current),
    pvalues=np.array(weights_history.pvalues),
    are_significantly_larger=np.array(weights_history.are_significantly_larger),
)

final_clustering = redetect_clustering(
    network.weights.matrix,
    history.clustering[-1],
    rel_tol=1e-2,
    random_state=rs,
)
np.save(
    # f"{DATA_DIR}/clustering_membership_after_{int(network.time / day)}d",
    f"{DATA_DIR}/clustering_membership_final",
    final_clustering.membership,
)

import matplotlib.pyplot as plt
from layers import Layers
from mynetworks import plotting

layers = Layers([27, 27])
plt.imshow(plotting.get_sorted_weight_matrix(network.weights.matrix, plotting.get_layer_contained_sorting_from_clustering(final_clustering, layers)))
plt.show()
