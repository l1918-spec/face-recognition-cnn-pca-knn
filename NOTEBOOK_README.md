# Notebook Guide — `notebook.ipynb`

Self-contained Jupyter notebook covering the entire Exposé: dataset, three algorithms, comparison, conclusion. Already executed end-to-end with **all outputs and plots embedded** — readable without re-running.

## Open it

```powershell
.venv\Scripts\activate
jupyter notebook notebook.ipynb
```

Or open in VS Code (Jupyter extension renders it inline).

## Re-execute (refresh outputs)

```powershell
jupyter nbconvert --to notebook --execute notebook.ipynb --output notebook.ipynb --ExecutePreprocessor.timeout=600
```

CNN training takes ~15 s; everything else is instant.

## Layout — 31 cells, 9 sections

| § | Title | Cells | What it shows |
|---|-------|-------|---------------|
| 1 | Imports + setup | 1 code | TensorFlow + sklearn + matplotlib loaded, seeds set |
| 2 | Chargement du dataset | 2 code | Olivetti shape `(400, 64, 64)`, 40 classes, 4 × 10 sample grid |
| 3 | Split train/test | 1 code | 320 train / 80 test, stratified |
| 4 | **KNN** | 2 code | Train + accuracy + confusion matrix |
| 5 | **PCA + KNN (Eigenfaces)** | 4 code | Train, top 12 Eigenfaces, variance curve, confusion matrix |
| 6 | **CNN** | 4 code | Architecture summary, train, loss/acc curves, confusion matrix |
| 7 | Comparaison | 2 code | Pandas table + bar chart (accuracy + compute cost) |
| 8 | Visualisations | 2 code | 20 first test predictions + errors-only grid |
| 9 | Conclusion | 1 markdown | Final table + bilan + limites |

## Key results in the notebook

| Model | Accuracy | F1 | Train | Inference |
|-------|----------|-----|-------|-----------|
| KNN | 0.888 | 0.888 | < 1 ms | ~20 ms/img |
| PCA+KNN | 0.900 | 0.900 | ~50 ms | ~0.1 ms/img |
| **CNN** | **0.988 – 1.000** | **0.987 – 1.000** | ~15 s | ~2 ms/img |

CNN reaches **100 % on a good seed**, ≥ 98 % on most random seeds.

## Reading order for the Exposé

1. **§1–3** — context, dataset, split. Skip fast.
2. **§4 KNN** — show baseline (raw pixels are slow).
3. **§5 PCA** — *the eigenfaces plot* is the visual highlight. Discuss the variance curve (50 components ≈ 90 % variance).
4. **§6 CNN** — show the architecture summary, then the loss/accuracy curves to prove convergence.
5. **§7 Comparaison** — bar chart is the slide-worthy graphic.
6. **§8 Predictions** — point at the failure cases (lunettes, angle).
7. **§9 Conclusion** — bilan + limites.

## Replacing dataset

The first code cell of §2 hard-codes Olivetti. To run on your own data, replace it with:

```python
from src.dataset import load_custom
X, y, label_names = load_custom("data/raw")
```

Re-run the whole notebook (`Kernel → Restart & Run All`).

## Exporting

- **PDF for handout**: `jupyter nbconvert --to pdf notebook.ipynb` (needs LaTeX)
- **HTML for sharing**: `jupyter nbconvert --to html notebook.ipynb`
- **Slides**: `jupyter nbconvert --to slides notebook.ipynb --post serve` (Reveal.js)

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Outputs missing | re-run with `nbconvert --execute` |
| Kernel doesn't start | activate venv first, then launch jupyter |
| Plots look squashed | drag the cell separator wider, or `plt.figure(figsize=...)` |
| `MissingIDFieldWarning` | harmless — old nbformat warning |
| Dataset download fails | check internet on first run; cached afterward in `~/scikit_learn_data/` |
