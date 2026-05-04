import time
from sklearn.neighbors import KNeighborsClassifier

from .dataset import flatten


def train(X_train, y_train, k=3, metric="euclidean"):
    Xtr = flatten(X_train)

    k = min(k, len(Xtr))
    k = max(1, k)

    model = KNeighborsClassifier(
        n_neighbors=k,
        metric=metric,
        n_jobs=-1
    )

    t0 = time.time()
    model.fit(Xtr, y_train)
    train_time = time.time() - t0

    return model, train_time


def predict(model, X):
    Xf = flatten(X)

    t0 = time.time()
    y_pred = model.predict(Xf)
    infer_time = (time.time() - t0) / max(len(X), 1)

    return y_pred, infer_time
