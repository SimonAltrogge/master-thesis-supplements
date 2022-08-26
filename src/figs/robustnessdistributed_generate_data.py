# TIME (measured in units of time_constant_rates_evoked=10ms)
millisecond = 1 / 10
second = 100
minute = 60 * second
hour = 60 * minute
day = 24 * hour

# PARAMETERS
DATA_DIR = "../data/robustnessdistributed"
SEED_RNG = 42
SEED_RS = 73

LAYER_COUNT = 2
NEURON_COUNT_PER_LAYER = 27

ASSEMBLY_COUNT = 3
NEURON_COUNT_PER_ASSEMBLY = 18

RATE_SPONTANEOUS = 0.002
WEIGHT_MAX = 0.7 / 12

LEARNING_RATE = 0.14
STDP_INTEGRAL = -(4 / 30)
STDP_POTENTIATION_TIME_CONSTANT = 2.5
STDP_POTENTIATION_AMPLITUDE = 0.08
STDP_DEPRESSION_TIME_CONSTANT = 5.0

NORMALIZATION_FUNCTION_STR = "normalize_additively"
TIME_CONSTANT_NORMALIZATION = 4 * hour
TOTAL_INPUT_AND_OUTPUT_PER_NEURON = 0.7

import numpy as np
from clusterings import Clustering
from layers import Layers
from neuralnetwork.stdp import Exponential, STDPFunction
import normalization.functions
from mynetworks import SoftlyNormalizedNeuralNetwork
import mynetworks.util as util

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

layers = Layers((NEURON_COUNT_PER_LAYER,) * LAYER_COUNT)
normalization_function = util.get_blockwise_column_and_row_normalization_from_layers(
    layers,
    TOTAL_INPUT_AND_OUTPUT_PER_NEURON,
    getattr(normalization.functions, NORMALIZATION_FUNCTION_STR),
)

rng = np.random.default_rng(SEED_RNG)
rs = np.random.RandomState(SEED_RS)

initial_clustering = Clustering(
    util.get_membership_for_clustering_spread_evenly_over_layers(layers, ASSEMBLY_COUNT)
)
initial_weight_matrix = util.construct_weight_matrix_from_clustering(
    initial_clustering, upper_bounds=WEIGHT_MAX, rng=rng
)
# np.save(f"{DATA_DIR}/weight_matrix_initial", initial_weight_matrix)

# NETWORK CONSTRUCTION
network = SoftlyNormalizedNeuralNetwork(
    weight_matrix=initial_weight_matrix,
    weight_max=WEIGHT_MAX,
    rates_spontaneous=RATE_SPONTANEOUS,
    stdp_function=stdp_function,
    learning_rates=LEARNING_RATE,
    is_learning=False,
    time_constant_normalization=TIME_CONSTANT_NORMALIZATION,
    normalization_function=normalization_function,
    rng=rng,
)

network.simulate_for(1 * minute)
network.is_learning = True

from tqdm import tqdm
from neuralnetwork.network import every_interval

descriptor = f"tau={TIME_CONSTANT_NORMALIZATION:08.0f}-eta={LEARNING_RATE}"
np.save(f"{DATA_DIR}/{descriptor}-weight_matrix_initial", initial_weight_matrix)

network.add_event_listener(
    "post_time_evolution",
    every_interval(
        1 * day,
        lambda network, *args, **kwargs: np.save(
            f"{DATA_DIR}/{descriptor}-weight_matrix_after_{int(network.time / day)}d",
            network.weights.matrix,
        ),
    ),
)

TIME_END = 100 * day
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
) as progress_bar:

    def update_progress_bar(_, amount):
        progress_bar.update(amount)

    network.add_event_listener("post_time_evolution", update_progress_bar)

    network.simulate_for(TIME_END - network.time)
    # np.save(
    #     f"{DATA_DIR}/weight_matrix_after_{int(network.time / day)}d",
    #     network.weights.matrix,
    # )

    network.remove_event_listener("post_time_evolution", update_progress_bar)

    # this is necessary, because the last spike will overshoot TIME_END by a bit
    progress_bar.total = progress_bar.n

# final_clustering = redetect_clustering(
#     network.weights.matrix,
#     initial_clustering,
#     rel_tol=1e-2,
#     random_state=rs,
# )
# np.save(
#     f"{DATA_DIR}/clustering_membership_after_{int(network.time / day)}d",
#     final_clustering.membership,
# )
