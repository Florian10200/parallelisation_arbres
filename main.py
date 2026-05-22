"""Point d'entrée du projet.

Ce script lance l'expérience complète :
- génération des données,
- benchmark séquentiel et parallèle,
- sauvegarde des résultats,
- création des figures.
"""

from __future__ import annotations

from pathlib import Path

from benchmark import benchmark_random_forest, save_benchmark_table_rows
from plots import save_benchmark_figures
from src.data import make_train_test_split


def run_project() -> None:
    """Lance le pipeline complet du projet."""

    project_root = Path(__file__).resolve().parent
    results_dir = project_root / "results"
    figures_dir = project_root / "report" / "figures"

    X_train, X_test, y_train, y_test = make_train_test_split(
        n_samples=5000,
        n_features=12,
        n_informative=6,
        test_size=0.25,
        random_state=42,
    )

    results = benchmark_random_forest(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        worker_list=(1, 2, 4, 8),
        n_estimators=64,
        max_depth=10,
        min_samples_split=2,
        repetitions=5,
        random_state=123,
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "benchmark_results.csv"
    table_rows_path = project_root / "report" / "benchmark_table_rows.tex"
    results.to_csv(csv_path, index=False)
    save_benchmark_table_rows(results, table_rows_path)

    save_benchmark_figures(results, figures_dir)

    print("\nRésultats du benchmark :")
    print(results.to_string(index=False))
    print(f"\nTableau enregistré dans : {csv_path}")
    print(f"Lignes LaTeX du tableau enregistrées dans : {table_rows_path}")
    print(f"Figures enregistrées dans : {figures_dir}")


if __name__ == "__main__":
    run_project()
