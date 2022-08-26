import numpy as np
from layers import Layers
from clusterings import Clustering
from mynetworks import util

NEURONS_PER_LAYER = 27
CLUSTER_COUNT = 3
WEIGHT_MAX = 0.7 / 12
DATA_DIR = "../data/scanwhichtaushuffled"

layers = Layers(neuron_count_by_layer=(NEURONS_PER_LAYER, NEURONS_PER_LAYER))
initial_clustering = Clustering(
    util.get_membership_for_clustering_spread_evenly_over_layers(layers, CLUSTER_COUNT)
)


def get_neuron_count_by_cluster_by_layer(clustering, layers):
    neuron_count_by_cluster_by_layer = np.empty((len(layers), len(clustering)))

    for layer_index in range(len(layers)):
        for cluster_index, cluster in enumerate(clustering.clusters()):
            neuron_count_by_cluster_by_layer[layer_index, cluster_index] = len(
                np.intersect1d(
                    cluster, layers.get_layer_indices(layer_index), assume_unique=True
                )
            )

    return neuron_count_by_cluster_by_layer


for seed in 2000 + np.arange(36):
    rng = np.random.default_rng(seed)
    rng.shuffle(initial_clustering.membership)
    initial_weight_matrix = util.construct_weight_matrix_from_clustering(
        initial_clustering, upper_bounds=WEIGHT_MAX, rng=rng
    )

    np.savez(
        f"{DATA_DIR}/seed={seed}-initial",
        weight_matrix=initial_weight_matrix,
        neuron_count_by_cluster_by_layer=get_neuron_count_by_cluster_by_layer(
            initial_clustering, layers
        ),
    )
