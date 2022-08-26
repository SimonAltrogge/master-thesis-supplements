from typing import overload, Literal, Sequence
import math
import numpy as np
import numpy.typing as npt
import bct
from clusterings.clustering import Clustering, NoClustering
from clusterings.matching import (
    match_clusters,
    label_unmatched_clusters_with_spare_labels,
)


@overload
def detect_clustering(
    weight_matrix: npt.ArrayLike,
    membership_prior: Sequence[object] | npt.NDArray | None,
    *,
    random_state: np.random.RandomState,
    gamma: float,
    return_modularity: Literal[False],
) -> Clustering:
    ...


@overload
def detect_clustering(
    weight_matrix: npt.ArrayLike,
    membership_prior: Sequence[object] | npt.NDArray | None,
    *,
    random_state: np.random.RandomState,
    gamma: float,
    return_modularity: Literal[True],
) -> tuple[Clustering, float]:
    ...


def detect_clustering(
    weight_matrix: npt.ArrayLike,
    membership_prior: Sequence[object] | npt.NDArray | None = None,
    *,
    random_state: np.random.RandomState,
    gamma: float = 1.0,
    return_modularity: bool = False,
) -> Clustering | tuple[Clustering, float]:
    weight_matrix = np.asarray(weight_matrix)

    try:
        membership, modularity = bct.community_louvain(
            weight_matrix,
            gamma=gamma,  # type: ignore  # pyright wrongly infers int instead of float
            ci=membership_prior,
            seed=random_state,
        )
    except bct.utils.BCTParamError as err:
        if str(err) != (
            "Modularity infinite loop style G. Please contact the developer."
        ):
            raise

        clustering = NoClustering(len(weight_matrix))
        modularity = np.nan
    else:
        # bct labels the clusters starting with 1, but 0 is more common in programming
        clustering = Clustering(membership - 1)

    if return_modularity:
        return (clustering, modularity)

    return clustering


@overload
def redetect_clustering(
    weight_matrix: npt.ArrayLike,
    clustering_prior: Clustering,
    *,
    rel_tol: float,
    random_state: np.random.RandomState,
    gamma: float,
    return_modularity: Literal[False],
) -> Clustering:
    ...


@overload
def redetect_clustering(
    weight_matrix: npt.ArrayLike,
    clustering_prior: Clustering,
    *,
    rel_tol: float,
    random_state: np.random.RandomState,
    gamma: float,
    return_modularity: Literal[True],
) -> tuple[Clustering, float]:
    ...


def redetect_clustering(
    weight_matrix: npt.ArrayLike,
    clustering_prior: Clustering,
    *,
    rel_tol: float,
    random_state: np.random.RandomState,
    gamma: float = 1.0,
    return_modularity: bool = False,
) -> Clustering | tuple[Clustering, float]:
    clustering, modularity = detect_clustering(
        weight_matrix,
        membership_prior=clustering_prior.membership,
        random_state=random_state,
        gamma=gamma,
        return_modularity=True,
    )
    clustering_not_using_prior, modularity_not_using_prior = detect_clustering(
        weight_matrix,
        membership_prior=None,
        random_state=random_state,
        gamma=gamma,
        return_modularity=True,
    )

    if not clustering or (
        modularity_not_using_prior > modularity
        and not math.isclose(modularity, modularity_not_using_prior, rel_tol=rel_tol)
    ):
        clustering = clustering_not_using_prior
        modularity = modularity_not_using_prior

    cluster_matching = match_clusters(clustering, clustering_prior)

    if len(clustering) > len(clustering_prior):
        relabeling = label_unmatched_clusters_with_spare_labels(cluster_matching)
    else:
        relabeling = cluster_matching

    clustering.relabel(relabeling)

    if return_modularity:
        return (clustering, modularity)

    return clustering
