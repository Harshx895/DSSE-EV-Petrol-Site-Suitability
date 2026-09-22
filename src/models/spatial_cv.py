"""
src/models/spatial_cv.py
--------------------------
Spatial Block Cross-Validation (k=5) via KMeans clustering on (lat, lon).

Standard random K-Fold is PROHIBITED — spatial autocorrelation within ~300m
inflates R² and causes academic rejection (per CONTEXT §9).

Usage:
  splitter = SpatialBlockCV(k=5, random_state=42)
  for fold, (train_idx, test_idx) in enumerate(splitter.split(coords)):
      ...
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


class SpatialBlockCV:
    """
    Assigns each point to one of k geographic clusters (via KMeans on lat/lon),
    then yields leave-one-cluster-out train/test splits.
    """

    def __init__(self, k: int = 5, random_state: int = 42):
        self.k = k
        self.random_state = random_state
        self.block_labels_: np.ndarray | None = None

    def fit(self, coords: np.ndarray) -> "SpatialBlockCV":
        """Cluster coordinates into k spatial blocks."""
        km = KMeans(n_clusters=self.k, random_state=self.random_state, n_init=10)
        self.block_labels_ = km.fit_predict(coords)
        return self

    def split(self, coords: np.ndarray):
        """
        Generator yielding (train_indices, test_indices) for each fold.
        coords: (N, 2) array of [lat, lon].
        """
        self.fit(coords)
        for fold in range(self.k):
            test_mask  = self.block_labels_ == fold
            train_mask = ~test_mask
            yield np.where(train_mask)[0], np.where(test_mask)[0]

    def get_n_splits(self) -> int:
        return self.k
