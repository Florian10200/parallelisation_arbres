"""Random forest séquentielle et parallèle via joblib."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed

from .tree import DecisionTreeClassifierSimple


@dataclass
class _FittedTree:
    estimator: DecisionTreeClassifierSimple


class RandomForestClassifierSimple:
    """Random forest binaire avec bootstrap et agrégation par vote majoritaire."""

    def __init__(
        self,
        n_estimators: int = 40,
        max_depth: int = 10,
        min_samples_split: int = 2,
        max_features: str | int | float | None = "sqrt",
        bootstrap: bool = True,
        n_jobs: int = 1,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.estimators_: list[DecisionTreeClassifierSimple] = []
        self.classes_: np.ndarray | None = None
        self._rng = np.random.default_rng(random_state)

    def _fit_one_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seed: int,
    ) -> DecisionTreeClassifierSimple:
        n_samples = X.shape[0]
        local_rng = np.random.default_rng(seed)
        if self.bootstrap:
            sample_indices = local_rng.integers(0, n_samples, size=n_samples)
        else:
            sample_indices = np.arange(n_samples)

        tree = DecisionTreeClassifierSimple(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            random_state=seed,
        )
        tree.fit(X[sample_indices], y[sample_indices])
        return tree

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifierSimple":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError("Cette implémentation gère uniquement la classification binaire.")

        seeds = self._rng.integers(0, 2**31 - 1, size=self.n_estimators)

        if self.n_jobs == 1:
            trees = [self._fit_one_tree(X, y, int(seed)) for seed in seeds]
        else:
            trees = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(self._fit_one_tree)(X, y, int(seed)) for seed in seeds
            )

        self.estimators_ = trees
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if not self.estimators_:
            raise RuntimeError("Le modèle doit être entraîné avant la prédiction.")
        if self.classes_ is None:
            raise RuntimeError("Les classes ne sont pas disponibles.")

        votes = np.array([tree.predict(X) for tree in self.estimators_])
        positive_class = self.classes_[1]
        negative_class = self.classes_[0]

        positive_votes = np.sum(votes == positive_class, axis=0)
        return np.where(
            positive_votes >= (len(self.estimators_) / 2.0),
            positive_class,
            negative_class,
        )

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return float(np.mean(np.asarray(y) == y_pred))
