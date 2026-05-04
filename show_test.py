"""Visualize test predictions: true vs predicted faces for all 3 models."""
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf

from src import dataset

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def main(n_show=20, only_errors=False):
    # same load + split as main.py (seed=42)
    X, y, label_names = dataset.load_olivetti()
    X_train, X_test, y_train, y_test = dataset.split(X, y, test_size=0.2, seed=42)

    # load models
    knn = joblib.load(MODELS_DIR / "knn.joblib")
    pca_knn = joblib.load(MODELS_DIR / "pca_knn.joblib")
    cnn = tf.keras.models.load_model(MODELS_DIR / "cnn.keras")

    # predict
    Xf = X_test.reshape(X_test.shape[0], -1)
    p_knn = knn.predict(Xf)
    p_pca = pca_knn.predict(Xf)
    p_cnn = cnn.predict(X_test[..., np.newaxis], verbose=0).argmax(axis=1)

    # pick which test indices to show
    if only_errors:
        mask = (p_knn != y_test) | (p_pca != y_test) | (p_cnn != y_test)
        idx = np.where(mask)[0]
        title = "Test images where at least one model failed"
    else:
        idx = np.arange(len(X_test))
        title = "Test predictions"
    idx = idx[:n_show]
    if len(idx) == 0:
        print("No errors to show.")
        return

    cols = 5
    rows = (len(idx) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 3))
    axes = np.array(axes).reshape(rows, cols)
    for ax, i in zip(axes.flat, idx):
        ax.imshow(X_test[i], cmap="gray")
        true = y_test[i]
        ok_k = p_knn[i] == true
        ok_p = p_pca[i] == true
        ok_c = p_cnn[i] == true

        def mark(ok, name, pred):
            tag = "OK" if ok else "X"
            return f"{name}:{tag}({pred})"

        cap = (
            f"true={true}\n"
            f"{mark(ok_k, 'KNN', p_knn[i])}\n"
            f"{mark(ok_p, 'PCA', p_pca[i])}\n"
            f"{mark(ok_c, 'CNN', p_cnn[i])}"
        )
        color = "green" if (ok_k and ok_p and ok_c) else "red"
        ax.set_title(cap, fontsize=7, color=color)
        ax.axis("off")
    for ax in axes.flat[len(idx):]:
        ax.axis("off")
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    out = RESULTS_DIR / ("test_errors.png" if only_errors else "test_preds.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    # text summary
    print("\n=== test set per-image (first 30) ===")
    print(f"{'idx':>4}{'true':>6}{'KNN':>6}{'PCA':>6}{'CNN':>6}")
    for i in range(min(30, len(X_test))):
        marks = lambda p: "OK" if p == y_test[i] else "X"
        print(f"{i:>4}{y_test[i]:>6}"
              f"  {p_knn[i]:>2}{marks(p_knn[i]):>2}"
              f"  {p_pca[i]:>2}{marks(p_pca[i]):>2}"
              f"  {p_cnn[i]:>2}{marks(p_cnn[i]):>2}")

    print(f"\nKNN     {(p_knn == y_test).sum()}/{len(y_test)} correct")
    print(f"PCA+KNN {(p_pca == y_test).sum()}/{len(y_test)} correct")
    print(f"CNN     {(p_cnn == y_test).sum()}/{len(y_test)} correct")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--errors", action="store_true", help="only show misclassified images")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    main(n_show=args.n, only_errors=args.errors)
