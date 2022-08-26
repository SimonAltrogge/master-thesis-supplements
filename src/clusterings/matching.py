from __future__ import annotations
from collections.abc import Mapping
import numpy as np
from clusterings.similarities import jaccard_similarities

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clusterings.clustering import Clustering


def match_clusters(clustering: Clustering, reference_clustering: Clustering) -> dict:
    if clustering.element_count() != reference_clustering.element_count():
        raise ValueError(
            "You tried to match clusters of a Clustering to clusters of a "
            "reference Clustering with a different number of elements. "
            f"The Clustering to be matched has {clustering.element_count()} "
            "elements while the reference Clustering has "
            f"{reference_clustering.element_count()} elements. Please provide "
            "a reference Clustering with the same number of elements as the one "
            "to be matched."
        )

    if np.array_equal(
        clustering.membership, reference_clustering.membership, equal_nan=True
    ):
        labels = clustering.labels()
        return dict(zip(labels, labels))

    similarities = jaccard_similarities(clustering, reference_clustering)

    corresponding_labels = {
        coord.item(): None for coord in similarities.coords["of_cluster"]
    }
    while similarities.size != 0:
        max_similarity_indices = similarities.argmax(...)
        max_similarity_labels = {
            dimension: similarities.coords[dimension][index.item()].item()
            for dimension, index in max_similarity_indices.items()
        }

        corresponding_labels[
            max_similarity_labels["of_cluster"]
        ] = max_similarity_labels["to_cluster"]
        similarities = similarities.drop_sel(labels=max_similarity_labels)

    return corresponding_labels


def label_unmatched_clusters_with_spare_labels(cluster_matching: Mapping) -> dict:
    clustering_labels = cluster_matching.keys()
    reference_clustering_labels = cluster_matching.values()
    spare_labels = [
        label for label in clustering_labels if label not in reference_clustering_labels
    ]
    relabeling = {
        label: reference_label if reference_label is not None else spare_labels.pop()
        for label, reference_label in cluster_matching.items()
    }

    return relabeling
