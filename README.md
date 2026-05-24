# Programmation parallèle pour le Machine Learning

Projet ENSAE Paris sur la parallélisation des arbres de décision et des random forests.

## Contenu

- `main.py` : point d'entrée qui lance l'expérience complète.
- `benchmark.py` : mesures de temps et calcul du speedup.
- `plots.py` : génération des figures.
- `src/` : implémentation de l'arbre de décision et de la random forest.
- `report/main.tex` : rapport LaTeX prêt pour Overleaf.


## Exécution

```bash
python main.py
```

Cette commande :

- génère un jeu de données artificiel,
- entraîne une random forest séquentielle puis parallèle,
- mesure les temps pour 1, 2, 4 et 8 workers,
- calcule le speedup,
- enregistre un tableau de résultats dans `results/benchmark_results.csv`,
- écrit les lignes du tableau LaTeX dans `results/benchmark_table_rows.tex`,
- produit les figures dans `report/figures/`.

## Rapport

Compilable avec :

```bash
cd report
latexmk -pdf main.tex
```

ou :

```bash
pdflatex main.tex
pdflatex main.tex
```

Le rapport inclut directement les lignes du tableau générées par le benchmark, ce qui garantit la cohérence entre le CSV, le tableau et les figures.
