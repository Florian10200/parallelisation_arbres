"""Arbre de décision binaire séquentiel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TreeNode:
    """Noeud d'un arbre de décision."""

    is_leaf: bool
    prediction: int
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    impurity: float = 0.0
    n_samples: int = 0


class DecisionTreeClassifierSimple:
    """Arbre de décision binaire pour la classification.

    Le critère utilisé est l'impureté de Gini.
    """

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 2,
        max_features: str | int | float | None = None,
        random_state: int | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.root_: TreeNode | None = None
        self.n_features_in_: int | None = None
        self.classes_: np.ndarray | None = None
        self._rng = np.random.default_rng(random_state)

    @staticmethod
    def _gini_from_counts(n_positive: int, n_total: int) -> float:
        if n_total == 0:
            return 0.0
        p = n_positive / n_total
        return 1.0 - p * p - (1.0 - p) * (1.0 - p)

    def _resolve_max_features(self, n_features: int) -> int:
        if self.max_features is None:
            return n_features
        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                return max(1, int(np.sqrt(n_features)))
            raise ValueError(f"Valeur max_features inconnue: {self.max_features}")
        if isinstance(self.max_features, float):
            if not 0.0 < self.max_features <= 1.0:
                raise ValueError("max_features en flottant doit être dans (0, 1].")
            return max(1, int(np.ceil(self.max_features * n_features)))
        if isinstance(self.max_features, int):
            return max(1, min(n_features, self.max_features))
        raise TypeError("Type de max_features non supporté.")

    def _best_split_for_feature(
        self, x_column: np.ndarray, y_binary: np.ndarray
    ) -> tuple[float, float | None]:
        order = np.argsort(x_column, kind="mergesort")
        x_sorted = x_column[order]
        y_sorted = y_binary[order]
        n_samples = y_sorted.size

        if n_samples <= 1 or np.all(x_sorted == x_sorted[0]):
            return float("inf"), None

        cum_pos = np.cumsum(y_sorted)
        total_pos = int(cum_pos[-1])

        best_impurity = float("inf")
        best_threshold = None

        for split_index in range(1, n_samples):
            if x_sorted[split_index - 1] == x_sorted[split_index]:
                continue

            left_count = split_index
            right_count = n_samples - split_index
            left_pos = int(cum_pos[split_index - 1])
            right_pos = total_pos - left_pos

            left_impurity = self._gini_from_counts(left_pos, left_count)
            right_impurity = self._gini_from_counts(right_pos, right_count)
            weighted_impurity = (
                left_count / n_samples * left_impurity
                + right_count / n_samples * right_impurity
            )

            if weighted_impurity < best_impurity:
                best_impurity = weighted_impurity
                best_threshold = float(0.5 * (x_sorted[split_index - 1] + x_sorted[split_index]))

        return best_impurity, best_threshold

    def _majority_class(self, y_binary: np.ndarray) -> int:
        return int(np.mean(y_binary) >= 0.5)

    def _build_node(self, X: np.ndarray, y_binary: np.ndarray, depth: int) -> TreeNode:
        n_samples, n_features = X.shape
        prediction = self._majority_class(y_binary)
        impurity = self._gini_from_counts(int(np.sum(y_binary)), n_samples)

        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or impurity == 0.0
        ):
            return TreeNode(
                is_leaf=True,
                prediction=prediction,
                impurity=impurity,
                n_samples=n_samples,
            )

        n_candidate_features = self._resolve_max_features(n_features)
        candidate_features = self._rng.choice(
            n_features,
            size=n_candidate_features,
            replace=False,
        )

        best_feature = None
        best_threshold = None
        best_impurity = float("inf")

        for feature_index in candidate_features:
            feature_impurity, threshold = self._best_split_for_feature(
                X[:, feature_index],
                y_binary,
            )
            if threshold is not None and feature_impurity < best_impurity:
                best_impurity = feature_impurity
                best_feature = int(feature_index)
                best_threshold = threshold

        if best_feature is None or best_threshold is None:
            return TreeNode(
                is_leaf=True,
                prediction=prediction,
                impurity=impurity,
                n_samples=n_samples,
            )

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        if not np.any(left_mask) or not np.any(right_mask):
            return TreeNode(
                is_leaf=True,
                prediction=prediction,
                impurity=impurity,
                n_samples=n_samples,
            )

        left_node = self._build_node(X[left_mask], y_binary[left_mask], depth + 1)
        right_node = self._build_node(X[right_mask], y_binary[right_mask], depth + 1)

        return TreeNode(
            is_leaf=False,
            prediction=prediction,
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_node,
            right=right_node,
            impurity=impurity,
            n_samples=n_samples,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifierSimple":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError("Cette implémentation gère uniquement la classification binaire.")

        y_binary = (y == self.classes_[1]).astype(np.int64)
        self.root_ = self._build_node(X, y_binary, depth=0)
        return self

    def _predict_one(self, x: np.ndarray) -> int:
        if self.root_ is None:
            raise RuntimeError("Le modèle doit être entraîné avant la prédiction.")

        node = self.root_
        while not node.is_leaf:
            assert node.feature_index is not None
            assert node.threshold is not None
            if x[node.feature_index] <= node.threshold:
                node = node.left  # type: ignore[assignment]
            else:
                node = node.right  # type: ignore[assignment]
        return node.prediction

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        y_binary_pred = np.array([self._predict_one(x) for x in X], dtype=np.int64)
        if self.classes_ is None:
            raise RuntimeError("Le modèle doit être entraîné avant la prédiction.")
        return np.where(y_binary_pred == 1, self.classes_[1], self.classes_[0])
