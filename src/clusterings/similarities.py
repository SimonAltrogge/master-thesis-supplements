from __future__ import annotations
import numpy as np
import numpy.typing as npt
import xarray as xr

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clusterings.clustering import Clustering


def jaccard_similarity(cluster_a: npt.ArrayLike, cluster_b: npt.ArrayLike) -> float:
    shared_elements = np.intersect1d(cluster_a, cluster_b, assume_unique=True)
    all_elements = np.union1d(cluster_a, cluster_b)

    return len(shared_elements) / len(all_elements)


def jaccard_similarities(
    clustering_a: Clustering, clustering_b: Clustering
) -> xr.DataArray:
    similarities = np.empty((len(clustering_a), len(clustering_b)))

    for cluster_a_index, cluster_a in enumerate(clustering_a.clusters()):
        for cluster_b_index, cluster_b in enumerate(clustering_b.clusters()):
            similarities[cluster_a_index, cluster_b_index] = jaccard_similarity(
                cluster_a, cluster_b
            )

    return xr.DataArray(
        similarities,
        coords=[
            ("of_cluster", clustering_a.labels()),
            ("to_cluster", clustering_b.labels()),
        ],
    )


def chance_level_jaccard_similarities(clustering: Clustering) -> xr.DataArray:
    similarities = np.empty(len(clustering))

    for cluster_index, cluster in enumerate(clustering.clusters()):
        similarities[cluster_index] = 1 / (
            (len(clustering) - 1) + clustering.element_count() / len(cluster)
        )

    return xr.DataArray(similarities, coords=[("cluster", clustering.labels())])
