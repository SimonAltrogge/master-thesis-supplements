from __future__ import annotations
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clusterings.clustering import Clustering


def are_identical(clustering_a: Clustering, clustering_b: Clustering) -> bool:
    return np.array_equal(
        clustering_a.membership,
        clustering_b.membership,
        equal_nan=(
            np.issubdtype(clustering_a.membership.dtype, np.number)
            and np.issubdtype(clustering_b.membership.dtype, np.number)
        ),
    )


def are_equivalent(clustering_a: Clustering, clustering_b: Clustering) -> bool:
    if are_identical(clustering_a, clustering_b):
        return True

    if len(clustering_a) != len(clustering_b):
        return False

    _, first_elements_of_a_clusters, a_membership_given_by_cluster_indices = np.unique(
        clustering_a.membership, return_index=True, return_inverse=True
    )
    corresponding_b_labels = clustering_b.membership[first_elements_of_a_clusters]
    relabeled_a_membership = corresponding_b_labels[
        a_membership_given_by_cluster_indices
    ]

    return np.array_equal(relabeled_a_membership, clustering_b.membership)
