import numpy as np
from clusterings import detect_clustering
from neuralnetwork.stdp import Exponential, STDPFunction
from neuralnetwork.weights import get_random_weight_matrix
from mynetworks import NeuralNetwork
import mynetworks.util as util

# TIME (measured in units of time_constant_rates_evoked=10ms)
millisecond = 1 / 10
second = 100
minute = 60 * second
hour = 60 * minute
day = 24 * hour

# PARAMETERS
DATA_DIR = "../data/totalio"
SEED_RNG = 42
SEED_RS = 73

NEURON_COUNT = 54

RATE_SPONTANEOUS = 0.002
WEIGHT_MAX = 0.7 / 12
TOTAL_IO_PER_NEURON = 0.9

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

initial_weight_matrix = get_random_weight_matrix(
    NEURON_COUNT, total_input_and_output_per_neuron=TOTAL_IO_PER_NEURON, rng=rng
)

prefix = f"{DATA_DIR}/seed={SEED_RNG}"
np.save(f"{prefix}-weight_matrix_initial", initial_weight_matrix)

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

from neuralnetwork.history import History
from neuralnetwork.network import every_interval

totalio_history = History({"time", "total_input_by_neuron", "total_output_by_neuron"})


def record_total_io(network, *args, **kwargs):
    total_input_by_neuron = np.sum(network.weights.matrix, axis=1)
    total_output_by_neuron = np.sum(network.weights.matrix, axis=0)

    totalio_history.append(
        time=network.time,
        total_input_by_neuron=total_input_by_neuron,
        total_output_by_neuron=total_output_by_neuron,
    )


record_total_io(network)
network.add_event_listener("post_spike", every_interval(1 * minute, record_total_io))

network.simulate_for(1 * minute)
network.is_learning = True

from tqdm import tqdm

TIME_END = (24 * hour, 48 * hour, 50 * hour, 72 * hour, 75 * hour, 96 * hour, 100 * hour)
for i in range(len(TIME_END)):
    with tqdm(
        initial=network.time,
        total=TIME_END[i],
        mininterval=1,
        maxinterval=60,
        smoothing=0.7,
        unit="z",  # use 'zeconds' as simulation time unit to distinguish between s/z and z/s
        unit_scale=(10 * 1e-3),
        ncols=160,
        bar_format="{percentage:5.1f}%|{bar:20}| {n:"
        + f"{len(str(int(TIME_END[i] * 10 * 1e-3))) + 6}"
        + ".6f}{unit}/{total:.0f}{unit} [{elapsed}<{remaining}, {rate_fmt}]",
    ) as progress_bar:

        def update_progress_bar(_, amount):
            progress_bar.update(amount)

        network.add_event_listener("post_time_evolution", update_progress_bar)

        network.simulate_for(TIME_END[i] - network.time)

        network.remove_event_listener("post_time_evolution", update_progress_bar)

        # this is necessary, because the last spike will overshoot TIME_END by a bit
        progress_bar.total = progress_bar.n

    np.save(
        f"{prefix}-weight_matrix_after_{int(network.time / hour)}h", network.weights.matrix
    )
    clustering = detect_clustering(network.weights.matrix, random_state=rs)
    np.save(
        f"{prefix}-clustering_membership_after_{int(network.time / hour)}h",
        clustering.membership,
    )
    np.savez(
        f"{prefix}-totalio_history",
        seed_rng=SEED_RNG,
        time=np.array(totalio_history.time),
        total_input_by_neuron=np.array(totalio_history.total_input_by_neuron),
        total_output_by_neuron=np.array(totalio_history.total_output_by_neuron),
    )

import matplotlib.pyplot as plt
from mynetworks.plotting import get_sorted_weight_matrix, get_sorting_from_clustering

plt.imshow(
    get_sorted_weight_matrix(
        network.weights.matrix, get_sorting_from_clustering(clustering)
    )
)
plt.show()
