# Live Demo — Webcam Face Recognition

Real-time face recognition using a trained model (KNN, PCA+KNN, or CNN). Open webcam, draw a box around any detected face, and label it with the predicted identity + confidence.

## Launch

```powershell
.venv\Scripts\activate
python -m src.webcam_demo --model cnn
```

Press **`q`** to quit.

## What you see on screen

```
┌─────────────────────────────────────────┐
│  ┌────────────────┐                     │
│  │ samy (0.97)    │  ← name + confidence│
│  │ ███████████░░░ │  ← confidence bar   │
│  │  ╔══════════╗  │                     │
│  │  ║          ║  │  ← box around face  │
│  │  ║   FACE   ║  │     color = conf    │
│  │  ║          ║  │                     │
│  │  ╚══════════╝  │                     │
│  └────────────────┘                     │
│                                         │
│  model: cnn                             │
│  high>0.80  med 0.50-0.80  low<0.50     │
│  press q to quit                        │
└─────────────────────────────────────────┘
```

## Color code (box + name + bar)

| Color | Confidence | Meaning |
|-------|-----------|---------|
| 🟢 **Green** | ≥ 0.80 | High — model very sure |
| 🟠 **Orange** | 0.50 – 0.80 | Medium — probably right |
| 🔴 **Red** | < 0.50 | Low — model unsure / unknown person |
| 🔵 **Blue** | n/a | No probability available (KNN baseline) |

The thin bar above the box also fills proportionally to confidence.

## Model choice

```powershell
python -m src.webcam_demo --model cnn       # best accuracy (98–100%)
python -m src.webcam_demo --model pca_knn   # fastest inference
python -m src.webcam_demo --model knn       # baseline
```

CNN gives meaningful confidence values (softmax probability). KNN/PCA+KNN give voting-based confidence (1/3, 2/3, 3/3 typically).

## What the model recognizes

- Trained on **Olivetti** (default `python main.py`) → recognizes 40 anonymous subjects labeled `s1` to `s40`. **Won't recognize you.**
- Trained on **custom data** (`python main.py --data data/raw`) → recognizes the people whose folders are in `data/raw/`.

## Build a custom dataset (so it recognizes YOU)

```powershell
python -m src.capture_faces --name yourname --count 30
python -m src.capture_faces --name friend1  --count 30
python main.py --data data/raw
python -m src.webcam_demo --model cnn
```

In capture mode: press **SPACE** save, **q** quit. Vary angle, expression, lighting.

Need ≥ 2 people, ≥ 10 images each (50+ recommended for CNN).

## CLI flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | `pca_knn` | `knn` / `pca_knn` / `cnn` |
| `--cam` | `0` | webcam index (try `1` if `0` doesn't open) |
| `--threshold` | `0.0` | (deprecated — color now gradient) |
| `--image PATH` | none | predict a single image instead of webcam |
| `--raw` | false | with `--image`, skip Haar + equalize (use for pre-cropped Olivetti samples) |

## Demo without webcam (fallback)

```powershell
python demo_offline.py                                  # save 5 sample images + show predictions
python -m src.webcam_demo --model cnn --image demo_samples/sample_00_true01.jpg --raw
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Cannot open webcam` | wrong cam index → try `--cam 1` |
| Black window | other app (Zoom, Teams) using the camera — close it |
| Wrong predictions all the time | model trained on Olivetti, not on your face. Run `--data data/raw` workflow |
| Detection lost when you move | move slower, face the camera frontally, ensure good lighting |
| Box flicker | normal — Haar runs frame-by-frame independently |
| Glasses cause errors | add training images with + without glasses |

## Tips for the Exposé

- **Open lighting** before launch — dim rooms hurt accuracy.
- **Frontal pose** — the Haar cascade is frontal-only.
- Show **all 3 models in turn** so audience see ranking: KNN flickers a lot, PCA+KNN steady, CNN solid green.
- If your lab Wi-Fi is slow, run `python main.py` once at home so Olivetti is cached locally.
- Have a **backup**: `demo_offline.py` works with no camera.

## Hand controls cheat-sheet

| Key | Action |
|-----|--------|
| `q` | quit demo |

(Single-key only for now. Live model switching, snapshot, recording etc. listed under "Upgrade ideas" in `HOW_TO_RUN.md`.)
