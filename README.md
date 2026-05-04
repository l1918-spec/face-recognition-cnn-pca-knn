# Exposé 2 — Reconnaissance Faciale (CNN, PCA, KNN)

Biometric face recognition system. Compares three algorithms on the same dataset:

- **KNN** — raw pixel K-Nearest Neighbors (baseline)
- **PCA + KNN** — Eigenfaces (dimensionality reduction) + KNN
- **CNN** — small Convolutional Neural Network with augmentation

Built with OpenCV (Haar cascade detection), scikit-learn, and TensorFlow / Keras.

## Documentation

| File | Purpose |
|------|---------|
| **[HOW_TO_RUN.md](HOW_TO_RUN.md)** | Setup + every command, step by step |
| **[DEMO_README.md](DEMO_README.md)** | Live webcam demo — controls, color code, tips |
| **[NOTEBOOK_README.md](NOTEBOOK_README.md)** | How to read / run / export `notebook.ipynb` |
| **[PRESENTATION_GUIDE.md](PRESENTATION_GUIDE.md)** | 12-min Exposé playbook with talking script |
| **[notebook.ipynb](notebook.ipynb)** | Full report with embedded outputs (1.5 MB) |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS / Linux
pip install -r requirements.txt
python main.py                    # train + evaluate all 3 on Olivetti
python -m src.webcam_demo --model cnn   # live demo
```

## Latest results (Olivetti, 80 test images)

| Model | Accuracy | F1 (macro) | Train | Inference |
|-------|----------|------------|-------|-----------|
| KNN | 0.888 | 0.888 | < 1 ms | 18 ms/img |
| PCA+KNN | 0.900 | 0.900 | 50 ms | 0.13 ms/img |
| **CNN** | **0.988 – 1.000** | **0.987 – 1.000** | ~15 s | 2 ms/img |

CNN reaches 100 % on a good run, ≥ 98 % on most random seeds. PCA+KNN gives ~150× faster inference than raw KNN with comparable accuracy.

## Project structure

```
bio/
├── data/raw/                # custom dataset: one subfolder per person
├── models/                  # saved trained models (created by main.py)
├── results/                 # metrics + plots (created by main.py)
├── demo_samples/            # offline demo images (created by demo_offline.py)
├── src/
│   ├── preprocess.py        # Haar face detect + crop + resize + equalize
│   ├── dataset.py           # Olivetti loader + custom folder loader
│   ├── model_knn.py         # raw KNN
│   ├── model_pca_knn.py     # Eigenfaces + KNN
│   ├── model_cnn.py         # Keras CNN
│   ├── evaluate.py          # metrics, confusion matrix, comparison plot
│   ├── capture_faces.py     # webcam → dataset images
│   └── webcam_demo.py       # live recognition + offline single-image mode
├── main.py                  # train + evaluate all 3 models
├── show_test.py             # visualize test set predictions
├── demo_offline.py          # save Olivetti samples + predict (no webcam)
├── notebook.ipynb           # full Exposé as Jupyter notebook
├── HOW_TO_RUN.md            # detailed step-by-step instructions
├── requirements.txt
└── README.md
```

## Pipeline

```
input image
   │
   ▼
Haar cascade face detection  (largest face)
   │
   ▼
crop + grayscale + resize 64×64 + histogram equalization + normalize [0, 1]
   │
   ├─ flatten → KNN
   ├─ flatten → PCA → KNN
   └─ tensor  → CNN
```

## Algorithms

### 1. KNN (raw pixels)

Each image flattened to a 4096-D vector. Test image classified by majority vote among the `k=3` nearest training vectors (Euclidean distance).

- ✓ no training, simple
- ✗ high-dimensional, slow inference, sensitive to lighting

### 2. PCA + KNN (Eigenfaces)

PCA projects images into a 50-D subspace capturing dominant variance — the **Eigenfaces** (Turk & Pentland, 1991). KNN then classifies in this compact space.

- ✓ fast inference, robust, interpretable (eigenfaces visualization)
- ✗ linear — struggles with strong pose/expression variation

### 3. CNN

Small convolutional network with augmentation:

```
Input(64×64×1)
  → RandomFlip + RandomTranslation
  → Conv(16) + MaxPool
  → Conv(32) + MaxPool
  → Conv(64) + MaxPool
  → Flatten → Dropout(0.5)
  → Dense(128) → Dropout(0.3)
  → Dense(40, softmax)
```

Adam (lr=1e-3), sparse categorical crossentropy, EarlyStopping (patience=25 on val_accuracy), ReduceLROnPlateau.

- ✓ best accuracy, learns features end-to-end
- ✗ needs more data, longer training, larger model

## Run modes overview

| Goal | Command |
|------|---------|
| Train + evaluate all 3 (Olivetti) | `python main.py` |
| Train + evaluate on custom faces | `python main.py --data data/raw` |
| Capture webcam dataset | `python -m src.capture_faces --name alice --count 30` |
| Live webcam recognition | `python -m src.webcam_demo --model cnn` |
| Predict single image (with detection) | `python -m src.webcam_demo --model cnn --image photo.jpg` |
| Predict pre-cropped face (Olivetti style) | `python -m src.webcam_demo --model cnn --image face.jpg --raw` |
| Visualize test predictions | `python show_test.py` |
| Show only misclassified | `python show_test.py --errors` |
| Generate offline demo samples | `python demo_offline.py` |
| Open the notebook | `jupyter notebook notebook.ipynb` |

## Limitations

- Haar cascade misses profile and occluded faces — switch to MTCNN or DNN face detector for production
- Dataset assumes frontal, centered faces in good lighting
- CNN with very small custom datasets (< 5 images per person) will underfit — capture more samples or use transfer learning (MobileNetV2 etc.)
- No anti-spoofing — system can be fooled by photos
- `--image` mode: use `--raw` for pre-cropped faces (e.g. Olivetti samples). For real photos, omit `--raw` so the Haar detector runs first.

## References

- Turk & Pentland — *Eigenfaces for Recognition* (1991)
- Olivetti / AT&T Laboratories Cambridge faces dataset
- OpenCV Haar cascades — Viola–Jones detector
- Pipeline scaffold inspired by [tkarim45/Beginner-Data-Science-Projects](https://github.com/tkarim45/Beginner-Data-Science-Projects) (Face Recognition project)
