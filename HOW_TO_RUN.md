# How to Run — Step by Step

## 1. Setup environment (once)

Open **PowerShell** in `C:\Users\debai\Desktop\bio`:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Wait ~5–10 min. TensorFlow is the heaviest dependency.

Verify install:
```powershell
python -c "import cv2, sklearn, tensorflow; print('OK')"
```

---

## 2. Train + evaluate all 3 models

### Path A — Olivetti dataset (zero setup)

The default uses `sklearn.datasets.fetch_olivetti_faces` — 40 people × 10 images, 64×64 grayscale. Auto-downloaded on first run.

```powershell
python main.py
```

Takes ~30 s. Output:

- `models/knn.joblib`, `models/pca_knn.joblib`, `models/cnn.keras`, `models/labels.pkl`
- `results/cm_knn.png`, `results/cm_pca_knn.png`, `results/cm_cnn.png`
- `results/eigenfaces.png`
- `results/comparison.png`
- `results/metrics.json`

Console prints a summary table.

### Path B — Custom dataset (your face + others)

**B1. Capture faces with webcam** (one command per person):

```powershell
python -m src.capture_faces --name yourname --count 30
python -m src.capture_faces --name friend1  --count 30
```

Press **SPACE** save face. Press **q** quit. Vary angle and lighting.

Need ≥ 2 people, ≥ 10 images each. More = better (especially for the CNN).

**B2. Train all 3 models on your data:**

```powershell
python main.py --data data/raw
```

---

## 3. Live webcam demo

```powershell
python -m src.webcam_demo --model cnn
python -m src.webcam_demo --model pca_knn
python -m src.webcam_demo --model knn
```

Webcam opens, faces detected with green boxes + predicted name. Press **q** to quit.

---

## 4. Demo without webcam (predict single image)

Use `--image` to predict on a saved face file (no camera needed):

```powershell
# regular photo (full pipeline: Haar detect + crop + equalize)
python -m src.webcam_demo --model cnn --image path/to/photo.jpg

# already-cropped face (skip Haar + equalize) — use for Olivetti samples
python -m src.webcam_demo --model cnn --image demo_samples/sample_00_true01.jpg --raw
```

Generate sample images with their true labels:
```powershell
python demo_offline.py
```
Drops 5 cropped Olivetti faces in `demo_samples/` and prints predictions from each model.

---

## 5. Visualize test set predictions

After `python main.py`, run:

```powershell
python show_test.py            # 20 first test images, color-coded right/wrong
python show_test.py --errors   # only images where ≥1 model fails
python show_test.py --n 80     # full test set
```

Output saved to `results/test_preds.png` and `results/test_errors.png`.

---

## 6. Jupyter notebook (full report)

`notebook.ipynb` — entire pipeline in one runnable file: data exploration, preprocessing, the 3 algorithms, plots, comparison, conclusion.

```powershell
jupyter notebook notebook.ipynb
```

Or open it in VS Code directly. The notebook ships **with outputs already filled in**, so you can read it without re-running anything. To regenerate:

```powershell
jupyter nbconvert --to notebook --execute notebook.ipynb --output notebook.ipynb
```

Use the notebook for the Exposé presentation — every cell has its output and plots inline.

---

## 7. View results

```powershell
explorer results
explorer models
explorer demo_samples
```

Open the PNGs and `metrics.json`.

---

## 8. CLI reference

### `main.py`

| Flag | Default | Meaning |
|------|---------|---------|
| `--data PATH` | none (Olivetti) | custom dataset folder |
| `--n-components N` | 50 | PCA components |
| `--k N` | 3 | KNN neighbors |
| `--epochs N` | 120 | CNN max epochs |
| `--seed N` | 42 | random seed |

### `src/capture_faces.py`

| Flag | Default | Meaning |
|------|---------|---------|
| `--name` | required | person folder name |
| `--count` | 30 | images to save |
| `--out` | data/raw | output dir |
| `--cam` | 0 | webcam index |

### `src/webcam_demo.py`

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | pca_knn | knn / pca_knn / cnn |
| `--cam` | 0 | webcam index |
| `--threshold` | 0.0 | min confidence to color box green |
| `--image PATH` | none | predict on single image instead of webcam |
| `--raw` | false | with `--image`, skip Haar + equalize (pre-cropped face) |

### `show_test.py`

| Flag | Default | Meaning |
|------|---------|---------|
| `--errors` | false | only show misclassified images |
| `--n N` | 20 | number of test images to display |

---

## 9. Typical workflow for the Exposé

1. `python main.py` (Olivetti) → screenshot `comparison.png`, `eigenfaces.png`, confusion matrices
2. Capture own faces → `python main.py --data data/raw` → repeat plots
3. `python show_test.py --errors` → discuss failure cases
4. `python -m src.webcam_demo --model cnn` → record live demo screen
5. Open `notebook.ipynb` in the slides — every cell has its output ready
6. Put the metrics table from `results/metrics.json` on a slide

---

## 10. Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | venv not activated. Run `.venv\Scripts\activate` |
| `Cannot open webcam` | wrong cam index. Try `--cam 1` |
| TF very slow | normal on CPU. Reduce `--epochs 60` |
| Olivetti download fail | check internet. Cached at `~/scikit_learn_data/` |
| CNN low acc on custom data | capture more images (50+/person), vary pose/lighting |
| `Failed to load cascade` | reinstall `opencv-python` |
| `--image` predicts wrong on Olivetti sample | add `--raw` flag (face already cropped) |
| Notebook output missing | re-execute with `jupyter nbconvert --execute` |

---

## 11. File map

| File | Purpose |
|------|---------|
| `main.py` | trainer + evaluator entrypoint |
| `notebook.ipynb` | full Exposé as Jupyter notebook |
| `show_test.py` | per-image test prediction visualizer |
| `demo_offline.py` | save sample faces + predict (no webcam) |
| `src/capture_faces.py` | build dataset from webcam |
| `src/webcam_demo.py` | live recognition + offline single-image mode |
| `src/preprocess.py` | Haar detect + crop + normalize |
| `src/dataset.py` | load Olivetti or custom folder |
| `src/model_knn.py` | raw-pixel KNN |
| `src/model_pca_knn.py` | Eigenfaces (PCA) + KNN |
| `src/model_cnn.py` | Keras CNN |
| `src/evaluate.py` | metrics + confusion matrix + comparison plot |
| `models/` | trained models (created at runtime) |
| `results/` | metrics + plots (created at runtime) |
| `demo_samples/` | offline demo images (created by `demo_offline.py`) |
| `data/raw/<person>/*.jpg` | custom dataset (you create) |

---

## TL;DR

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
python -m src.webcam_demo --model cnn
```
