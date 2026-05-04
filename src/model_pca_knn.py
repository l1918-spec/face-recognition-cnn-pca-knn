import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from .dataset import flatten


def train(X_train, y_train, n_components=50, k=3):
    Xtr = flatten(X_train)

    n_components = min(n_components, Xtr.shape[0] - 1, Xtr.shape[1])
    n_components = max(1, n_components)

    k = min(k, len(Xtr))
    k = max(1, k)

    pipe = Pipeline([
        ("pca", PCA(n_components=n_components, whiten=True, random_state=42)),
        ("knn", KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=-1)),
    ])

    t0 = time.time()
    pipe.fit(Xtr, y_train)
    train_time = time.time() - t0

    return pipe, train_time


def predict(model, X):
    Xf = flatten(X)

    t0 = time.time()
    y_pred = model.predict(Xf)
    infer_time = (time.time() - t0) / max(len(X), 1)

    return y_pred, infer_time


def plot_eigenfaces(model, img_shape=(64, 64), n=12, save_path=None):
    pca = model.named_steps["pca"]

    available = pca.components_.shape[0]
    n = min(n, available)

    if n <= 0:
        return

    eigenfaces = pca.components_[:n].reshape((n, *img_shape))

    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for i in range(n):
        axes[i].imshow(eigenfaces[i], cmap="gray")
        axes[i].set_title(f"PC {i+1}", fontsize=8)

    plt.suptitle("Top Eigenfaces", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")

    plt.close(fig)
