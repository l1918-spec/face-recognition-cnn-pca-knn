import time
import os
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


def build_cnn(input_shape, n_classes):
    inputs = layers.Input(shape=input_shape)

    # mild augmentation only
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomTranslation(0.05, 0.05)(x)

    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="face_cnn")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _add_channel(X):
    if X.ndim == 3:
        return X[..., np.newaxis]
    return X


def train(X_train, y_train, X_val=None, y_val=None,
          epochs=120, batch_size=32, verbose=0):

    X_train = _add_channel(X_train)

    if X_val is not None:
        X_val = _add_channel(X_val)

    n_classes = int(np.max(y_train)) + 1

    if n_classes < 2:
        print("[CNN] Warning: only 1 class detected → CNN training skipped.")
        return None, 0.0, None

    model = build_cnn(X_train.shape[1:], n_classes)

    cbs = [
        callbacks.EarlyStopping(
            monitor="val_accuracy" if X_val is not None else "accuracy",
            patience=25,
            mode="max",
            restore_best_weights=True,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss" if X_val is not None else "loss",
            factor=0.5,
            patience=10,
            min_lr=1e-5,
            mode="min",
        ),
    ]

    val = (X_val, y_val) if X_val is not None else None

    t0 = time.time()

    history = model.fit(
        X_train,
        y_train,
        validation_data=val,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cbs,
        verbose=verbose,
        shuffle=True,
    )

    train_time = time.time() - t0

    return model, train_time, history


def predict(model, X):
    X = _add_channel(X)

    if model is None:
        return np.zeros(len(X), dtype=int), 0.0

    t0 = time.time()
    probs = model.predict(X, verbose=0)
    infer_time = (time.time() - t0) / max(len(X), 1)

    return probs.argmax(axis=1), infer_time
