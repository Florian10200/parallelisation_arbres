"""Génération d'un jeu de données artificiel pour la classification."""

from __future__ import annotations

import numpy as np


def make_classification_dataset(
    n_samples: int = 3000,
    n_features: int = 12,
    n_informative: int = 6,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Construit un problème binaire non linéaire.

    Les variables informatives sont mélangées avec du bruit pour obtenir un
    problème réaliste pour un arbre de décision.
    """

    if n_informative > n_features:
        raise ValueError("n_informative doit être inférieur ou égal à n_features.")

    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float64)

    informative = X[:, :n_informative]
    latent = (
        1.2 * informative[:, 0]
        - 0.8 * informative[:, 1]
        + 0.7 * informative[:, 2] ** 2
        - 0.6 * np.sin(informative[:, 3])
        + 0.4 * informative[:, 4] * informative[:, 5]
    )
    latent += 0.7 * rng.normal(size=n_samples)

    threshold = float(np.median(latent))
    y = (latent > threshold).astype(np.int64)
    return X, y


def make_train_test_split(
    n_samples: int = 3000,
    n_features: int = 12,
    n_informative: int = 6,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Génère puis découpe le jeu de données en entraînement et test."""

    X, y = make_classification_dataset(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        random_state=random_state,
    )

    rng = np.random.default_rng(random_state)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    split_index = int((1.0 - test_size) * n_samples)
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
