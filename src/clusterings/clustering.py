from typing import TypeVar
from collections.abc import Iterator, Generator, Collection, Sequence, Mapping
import numpy as np
import numpy.typing as npt
import xarray as xr
from clusterings.comparison import are_equivalent

T = TypeVar("T")


class Clustering:
    def __init__(self, membership: Sequence[object] | npt.NDArray) -> None:
        self.membership = membership

    @property
    def membership(self) -> npt.NDArray:
        return self.__membership

    @membership.setter
    def membership(self, value: Sequence[object] | npt.NDArray) -> None:
        self.__membership = np.array(value)

    def __repr__(self) -> str:
        return f"Clustering(membership={self.membership})"

    def __len__(self) -> int:
        return len(self.labels())

    def __getitem__(self, label: object) -> npt.NDArray[np.int_]:
        cluster, *_ = np.nonzero(self.membership == label)

        if len(cluster) == 0:
            raise KeyError(
                f"You tried to access a cluster labeled '{label}' but no such cluster "
                "exists in this Clustering. This error might be caused by trying to "
                "use an index instead of a label to access a cluster.\n"
                "\n"
                f"The existing clusters are labeled {self.labels()}."
            )

        return cluster

    def __iter__(self) -> Iterator[object]:
        return iter(self.labels())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Clustering):
            return NotImplemented

        if other.element_count() != self.element_count():
            return NotImplemented

        return are_equivalent(self, other)  # type: ignore

    def get(self, label: object, default: T = None) -> npt.NDArray[np.int_] | T:
        try:
            return self[label]
        except KeyError:
            return default

    def labels(self) -> npt.NDArray:
        return np.unique(self.membership)

    def clusters(self) -> Generator[npt.NDArray[np.int_], None, None]:
        return (self[label] for label in self)

    def labeled_clusters(
        self,
    ) -> Generator[tuple[object, npt.NDArray[np.int_]], None, None]:
        return ((label, self[label]) for label in self)

    def sizes(self) -> xr.DataArray:
        labels, sizes = np.unique(self.membership, return_counts=True)

        return xr.DataArray(sizes, coords=[("cluster", labels)])

    def element_count(self) -> int:
        return len(self.membership)

    def relabel(self, new_label_by_old_label: Mapping) -> None:
        if len(new_label_by_old_label) == 0 or np.array_equal(
            tuple(new_label_by_old_label.keys()), tuple(new_label_by_old_label.values())
        ):
            return

        if has_duplicates(new_label_by_old_label.values()):
            duplicates = get_duplicates(new_label_by_old_label.values())

            raise ValueError(
                "The new labels you have provided are not unique. The following labels "
                f"occur more than once: {duplicates}. Please choose a unique new label "
                "for every label that you want to change"
            )

        old_labels, membership_given_by_cluster_indices = np.unique(
            self.membership, return_inverse=True
        )
        new_labels = np.array(
            [new_label_by_old_label.get(label, label) for label in old_labels]
        )

        if has_duplicates(new_labels):
            duplicates = get_duplicates(new_labels)

            raise ValueError(
                "You have provided new labels that are already in use without also "
                "providing new unique labels for the clusters currently labeled by "
                f"these. The labels in question are: {duplicates}.\n"
                f"The mapping given was: {new_label_by_old_label}.\n"
                f"The mapping computed was: {old_labels} to {new_labels}.\n"
                "\n"
                "Please change your selection of new labels to not contain those "
                "currently in use or also provide new labels for these very ones "
                "to prevent accidential merging of clusters."
            )

        relabeled_membership = new_labels[membership_given_by_cluster_indices]

        self.membership = relabeled_membership

    @classmethod
    def from_sizes(
        cls, sizes: Sequence[int], labels: Sequence | npt.NDArray | None = None
    ):
        if labels is None:
            labels = np.arange(len(sizes), dtype=int)

        if len(labels) != len(sizes):
            raise ValueError(
                f"You have provided {len(labels)} labels for {len(sizes)} clusters."
                "Please match the number of labels to the number of clusters."
            )
        elif has_duplicates(labels):
            duplicates = get_duplicates(labels)

            raise ValueError(
                "The labels you have provided are not unique. The following labels "
                f"occur more than once: {duplicates}. Please choose a unique label "
                "for every cluster to prevent accidential merging of clusters."
            )

        membership = np.repeat(labels, sizes)

        return cls(membership)


class NoClustering(Clustering):
    def __init__(self, element_count: int) -> None:
        super().__init__(membership=np.full(element_count, np.nan))

    def __repr__(self) -> str:
        return f"NoClustering(element_count={self.element_count()})"

    def labels(self) -> npt.NDArray:
        return np.empty(0, dtype=int)

    def sizes(self) -> xr.DataArray:
        return xr.DataArray(np.empty(0, dtype=int), dims="cluster")

    def relabel(self, _) -> None:
        pass

    @classmethod
    def from_sizes(cls, sizes: Sequence[int], _):
        return cls(np.sum(sizes, dtype=int))


# REVIEW: Should this be extracted to a different file?
def has_duplicates(sequence: Collection | npt.NDArray) -> bool:
    return len(sequence) != len(np.unique(list(sequence)))


def get_duplicates(sequence: Collection | npt.NDArray) -> npt.NDArray:
    unique_elements, counts = np.unique(list(sequence), return_counts=True)

    return unique_elements[counts > 1]
