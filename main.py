"""Train + evaluate + compare KNN, PCA+KNN, CNN on face recognition.

Usage:
    python main.py                       # use Olivetti
    python main.py --data data/raw       # use custom folder dataset
"""
import argparse
import pickle
from pathlib import Path
import numpy as np
import joblib

from src import dataset, evaluate
from src import model_knn, model_pca_knn

try:
    from src import model_cnn
    _HAS_TF = True
except Exception as _e:
    _HAS_TF = False
    print(f"[warn] TensorFlow not available ({type(_e).__name__}). CNN will be skipped.")

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")


def main(data_path=None, n_components=50, k=3, epochs=40, seed=42):
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    np.random.seed(seed)

    # 1. load
    if data_path:
        print(f"[data] custom: {data_path}")
        X, y, label_names = dataset.load_custom(data_path)
    else:
        print("[data] Olivetti faces (40 ppl x 10 imgs)")
        X, y, label_names = dataset.load_olivetti()
    print(f"[data] X={X.shape} classes={len(label_names)}")

    X_train, X_test, y_train, y_test = dataset.split(X, y, test_size=0.2, seed=seed)
    print(f"[split] train={len(X_train)} test={len(X_test)}")

    with open(MODELS_DIR / "labels.pkl", "wb") as f:
        pickle.dump(label_names, f)

    results = {}

    # 2. KNN
    print("\n[1/3] KNN (raw pixels)")
    knn, t_train = model_knn.train(X_train, y_train, k=k)
    y_pred, t_inf = model_knn.predict(knn, X_test)
    m = evaluate.compute_metrics(y_test, y_pred)
    m.update(train_time=t_train, infer_time_ms=t_inf * 1000)
    results["KNN"] = m
    joblib.dump(knn, MODELS_DIR / "knn.joblib")
    evaluate.plot_confusion(y_test, y_pred, "KNN", RESULTS_DIR / "cm_knn.png")
    print(f"  acc={m['accuracy']:.3f} f1={m['f1']:.3f} train={t_train:.2f}s")

    # 3. PCA + KNN
    print("\n[2/3] PCA + KNN (Eigenfaces)")
    pca_knn, t_train = model_pca_knn.train(X_train, y_train, n_components=n_components, k=k)
    y_pred, t_inf = model_pca_knn.predict(pca_knn, X_test)
    m = evaluate.compute_metrics(y_test, y_pred)
    m.update(train_time=t_train, infer_time_ms=t_inf * 1000)
    results["PCA+KNN"] = m
    joblib.dump(pca_knn, MODELS_DIR / "pca_knn.joblib")
    evaluate.plot_confusion(y_test, y_pred, "PCA+KNN", RESULTS_DIR / "cm_pca_knn.png")
    model_pca_knn.plot_eigenfaces(
        pca_knn, img_shape=X.shape[1:], save_path=RESULTS_DIR / "eigenfaces.png"
    )
    print(f"  acc={m['accuracy']:.3f} f1={m['f1']:.3f} train={t_train:.2f}s")

    # 4. CNN
    if _HAS_TF:
        print("\n[3/3] CNN")
        cnn, t_train, hist = model_cnn.train(
            X_train, y_train, X_test, y_test, epochs=epochs, verbose=0
        )
        y_pred, t_inf = model_cnn.predict(cnn, X_test)
        m = evaluate.compute_metrics(y_test, y_pred)
        m.update(train_time=t_train, infer_time_ms=t_inf * 1000)
        results["CNN"] = m
        cnn.save(MODELS_DIR / "cnn.keras")
        evaluate.plot_confusion(y_test, y_pred, "CNN", RESULTS_DIR / "cm_cnn.png")
        print(f"  acc={m['accuracy']:.3f} f1={m['f1']:.3f} train={t_train:.2f}s")
    else:
        print("\n[3/3] CNN — SKIPPED (TensorFlow not installed)")

    # 5. compare
    evaluate.plot_comparison(results, RESULTS_DIR / "comparison.png")
    evaluate.save_report(results, RESULTS_DIR / "metrics.json")

    print("\n=== summary ===")
    print(f"{'model':<12}{'acc':>8}{'f1':>8}{'train(s)':>12}{'infer(ms)':>12}")
    for name, m in results.items():
        print(f"{name:<12}{m['accuracy']:>8.3f}{m['f1']:>8.3f}"
              f"{m['train_time']:>12.2f}{m['infer_time_ms']:>12.2f}")
    print(f"\nartifacts: {MODELS_DIR}/  {RESULTS_DIR}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="custom dataset dir (folder per person)")
    ap.add_argument("--n-components", type=int, default=50)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.data, args.n_components, args.k, args.epochs, args.seed)
