"""Offline demo: pick a few Olivetti test faces, save as JPG, predict with all 3 models."""
from pathlib import Path
import numpy as np
import cv2
import pickle
import joblib
import tensorflow as tf

from src import dataset

MODELS_DIR = Path("models")
DEMO_DIR = Path("demo_samples")
DEMO_DIR.mkdir(exist_ok=True)


def main():
    X, y, _ = dataset.load_olivetti()
    _, X_test, _, y_test = dataset.split(X, y, test_size=0.2, seed=42)

    knn = joblib.load(MODELS_DIR / "knn.joblib")
    pca_knn = joblib.load(MODELS_DIR / "pca_knn.joblib")
    cnn = tf.keras.models.load_model(MODELS_DIR / "cnn.keras")
    with open(MODELS_DIR / "labels.pkl", "rb") as f:
        labels = pickle.load(f)

    pick = [0, 4, 8, 12, 27]
    print(f"{'file':<28}{'true':>6}{'KNN':>10}{'PCA+KNN':>12}{'CNN':>10}")
    for i in pick:
        face = (X_test[i] * 255).astype(np.uint8)
        path = DEMO_DIR / f"sample_{i:02d}_true{y_test[i]:02d}.jpg"
        cv2.imwrite(str(path), face)

        flat = face.reshape(1, -1).astype(np.float32) / 255.0
        p_k = int(knn.predict(flat)[0])
        p_p = int(pca_knn.predict(flat)[0])
        x = (face.astype(np.float32) / 255.0)[np.newaxis, ..., np.newaxis]
        p_c = int(cnn.predict(x, verbose=0).argmax())
        true = int(y_test[i])
        mk = lambda p: f"{labels[p]}{'OK' if p==true else 'X'}"
        print(f"{path.name:<28}{labels[true]:>6}{mk(p_k):>10}{mk(p_p):>12}{mk(p_c):>10}")

    print(f"\nSamples saved to {DEMO_DIR}/")
    print("\nTry one with the demo script (no webcam needed):")
    print(f"  python -m src.webcam_demo --model cnn --image {DEMO_DIR}/sample_00_true01.jpg")


if __name__ == "__main__":
    main()
