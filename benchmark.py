"""Benchmark des random forests séquentielles et parallèles."""

from __future__ import annotations

from dataclasses import dataclass
import os
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd

from src.forest import RandomForestClassifierSimple


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration du benchmark."""

    worker_list: tuple[int, ...] = (1, 2, 4, 8)
    n_estimators: int = 64
    max_depth: int = 10
    min_samples_split: int = 2
    repetitions: int = 5
    random_state: int = 0


def _accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def benchmark_random_forest(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    worker_list: Iterable[int] = (1, 2, 4, 8),
    n_estimators: int = 64,
    max_depth: int = 10,
    min_samples_split: int = 2,
    repetitions: int = 5,
    random_state: int = 0,
) -> pd.DataFrame:
    """Mesure le temps d'entraînement pour plusieurs nombres de workers.

    Le speedup est défini par :
    S(p) = T(1) / T(p)
    où T(p) est le temps moyen avec p workers.
    """

    worker_list = tuple(worker_list)
    rows = []
    baseline_time = None

    for n_jobs in worker_list:
        warmup_model = RandomForestClassifierSimple(
            n_estimators=4,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features="sqrt",
            bootstrap=True,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        warmup_limit = min(256, X_train.shape[0])
        warmup_model.fit(X_train[:warmup_limit], y_train[:warmup_limit])

        fit_times = []
        accuracies = []

        for repetition in range(repetitions):
            model = RandomForestClassifierSimple(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                max_features="sqrt",
                bootstrap=True,
                n_jobs=n_jobs,
                random_state=random_state + repetition,
            )

            start = perf_counter()
            model.fit(X_train, y_train)
            fit_times.append(perf_counter() - start)

            y_pred = model.predict(X_test)
            accuracies.append(_accuracy_score(y_test, y_pred))

        mean_time = float(np.mean(fit_times))
        mean_accuracy = float(np.mean(accuracies))

        if baseline_time is None:
            baseline_time = mean_time

        rows.append(
            {
                "workers": n_jobs,
                "mean_fit_time_s": mean_time,
                "std_fit_time_s": float(np.std(fit_times, ddof=0)),
                "mean_accuracy": mean_accuracy,
                "speedup": float(baseline_time / mean_time),
            }
        )

    results = pd.DataFrame(rows)
    results["efficiency"] = results["speedup"] / results["workers"]
    return results


def save_benchmark_table_rows(results: pd.DataFrame, output_path: str | os.PathLike[str]) -> None:
    r"""Écrit les lignes du tableau LaTeX à partir des résultats mesurés.

    Le rapport peut ensuite inclure ce fichier avec \input{} pour éviter toute
    incohérence entre le tableau, le CSV et les figures.
    """

    rows = []
    for _, row in results.iterrows():
        rows.append(
            f"{int(row['workers'])} & {row['mean_fit_time_s']:.3f} & {row['std_fit_time_s']:.3f} & {row['mean_accuracy']:.3f} & {row['speedup']:.2f} \\\\"
        )

    with open(output_path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    from src.data import make_train_test_split

    X_train, X_test, y_train, y_test = make_train_test_split()
    df = benchmark_random_forest(X_train, X_test, y_train, y_test)
    print(df.to_string(index=False))
