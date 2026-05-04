"""Dataset loaders: Olivetti (default) or custom folder-per-person."""
from pathlib import Path
import numpy as np
import cv2
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

from .preprocess import preprocess_image, get_cascade

IMG_SIZE = (64, 64)


def load_olivetti():
    """40 ppl × 10 imgs, 64x64 grayscale, already normalized [0,1]."""
    data = fetch_olivetti_faces(shuffle=False)
    X = data.images.astype(np.float32)  # (400, 64, 64)
    y = data.target.astype(np.int64)
    label_names = [f"s{i+1}" for i in range(40)]
    return X, y, label_names


def load_custom(data_dir, size=IMG_SIZE):
    """Load images organized as data_dir/<person>/<img>.jpg."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)
    cascade = get_cascade()
    X, y, label_names = [], [], []
    persons = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    for label, person in enumerate(persons):
        label_names.append(person.name)
        for img_path in sorted(person.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            face = preprocess_image(img_path, size=size, cascade=cascade)
            if face is None:
                continue
            X.append(face)
            y.append(label)
    if not X:
        raise RuntimeError(f"No images found under {data_dir}")
    return np.stack(X), np.array(y, dtype=np.int64), label_names


def split(X, y, test_size=0.2, seed=42):
    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )


def flatten(X):
    """(N, H, W) → (N, H*W)."""
    return X.reshape(X.shape[0], -1)
