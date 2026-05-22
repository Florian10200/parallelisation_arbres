"""Création des figures du projet."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_benchmark_figures(results: pd.DataFrame, output_dir: str | Path) -> None:
    """Sauvegarde les graphiques temps/speedup."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(
        results["workers"],
        results["mean_fit_time_s"],
        marker="o",
        linewidth=2,
        color="#1f77b4",
        label="Temps mesuré",
    )
    ax.set_xlabel("Nombre de workers")
    ax.set_ylabel("Temps d'entraînement (s)")
    ax.set_title("Temps d'entraînement en fonction du parallélisme")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "time_vs_workers.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(
        results["workers"],
        results["speedup"],
        marker="o",
        linewidth=2,
        color="#d62728",
        label="Speedup mesuré",
    )
    ax.plot(
        results["workers"],
        results["workers"],
        linestyle="--",
        color="black",
        label="Speedup idéal",
    )
    ax.set_xlabel("Nombre de workers")
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup observé et speedup idéal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "speedup_vs_workers.png", dpi=160)
    plt.close(fig)
