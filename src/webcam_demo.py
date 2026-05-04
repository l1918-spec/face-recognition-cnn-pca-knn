"""Live webcam face recognition with chosen trained model.

Examples:
    python -m src.webcam_demo --model pca_knn
    python -m src.webcam_demo --model cnn
    python -m src.webcam_demo --model cnn --image data/raw/alice/alice_001.jpg
"""
import argparse
from pathlib import Path
import pickle
import numpy as np
import cv2

from .preprocess import get_cascade, preprocess_array

MODELS_DIR = Path("models")
LABELS_PATH = MODELS_DIR / "labels.pkl"


def load_model(name):
    if name == "cnn":
        import tensorflow as tf
        return ("cnn", tf.keras.models.load_model(str(MODELS_DIR / "cnn.keras")))
    import joblib
    path = MODELS_DIR / f"{name}.joblib"
    return (name, joblib.load(path))


def load_labels():
    with open(LABELS_PATH, "rb") as f:
        return pickle.load(f)


def predict_face(kind, model, face):
    if kind == "cnn":
        x = face[np.newaxis, ..., np.newaxis]
        probs = model.predict(x, verbose=0)[0]
        idx = int(probs.argmax())
        return idx, float(probs[idx])
    flat = face.reshape(1, -1)
    idx = int(model.predict(flat)[0])
    conf = float("nan")
    if hasattr(model, "predict_proba"):
        try:
            conf = float(model.predict_proba(flat).max())
        except Exception:
            pass
    return idx, conf


def predict_image(model_name, image_path, raw=False):
    """Predict on a single image file (no webcam needed).

    raw=True : assume image is already a cropped face (e.g. Olivetti). Just
               resize + normalize, skip Haar detection and histogram equalization.
    raw=False: full pipeline (Haar detect → crop → equalize → resize → normalize).
    """
    kind, model = load_model(model_name)
    label_names = load_labels()

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if raw:
        face = cv2.resize(gray, (64, 64)).astype(np.float32) / 255.0
    else:
        face = preprocess_array(gray, size=(64, 64), cascade=get_cascade())
    idx, conf = predict_face(kind, model, face)
    name = label_names[idx] if idx < len(label_names) else "?"
    print(f"[{model_name}] {image_path}  raw={raw}")
    print(f"  predicted: {name} (label {idx})  confidence: {conf:.3f}")
    return idx, name, conf


def confidence_color(conf):
    """Map confidence [0,1] → BGR color. Red (low) → orange (mid) → green (high).
    NaN → blue (no probability available)."""
    if conf != conf:  # NaN
        return (255, 128, 0)  # blue
    conf = max(0.0, min(1.0, conf))
    if conf >= 0.80:
        return (0, 255, 0)        # green — high confidence
    if conf >= 0.50:
        return (0, 165, 255)      # orange — medium
    return (0, 0, 255)            # red — low confidence


def run_webcam(model_name, cam_index=0, threshold=0.0):
    kind, model = load_model(model_name)
    label_names = load_labels()
    cascade = get_cascade()
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    print("[demo] press q to quit")
    print("       box color: green > 0.80, orange 0.50-0.80, red < 0.50, blue = no conf")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
        for (x, y, w, h) in faces:
            crop = gray[y:y + h, x:x + w]
            face = preprocess_array(crop, size=(64, 64), cascade=cascade)
            idx, conf = predict_face(kind, model, face)
            name = label_names[idx] if idx < len(label_names) else "?"
            label = f"{name} ({conf:.2f})" if conf == conf else name
            color = confidence_color(conf)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            # confidence bar above box
            if conf == conf:
                bar_w = int(w * conf)
                cv2.rectangle(frame, (x, y - 12), (x + w, y - 6), (60, 60, 60), -1)
                cv2.rectangle(frame, (x, y - 12), (x + bar_w, y - 6), color, -1)
            cv2.putText(frame, label, (x, y - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # legend bottom-left
        h_frame = frame.shape[0]
        cv2.putText(frame, f"model: {model_name}", (10, h_frame - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "high>0.80", (10, h_frame - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(frame, "med 0.50-0.80", (110, h_frame - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
        cv2.putText(frame, "low<0.50", (260, h_frame - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.putText(frame, "press q to quit", (10, h_frame - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow(f"face recognition [{model_name}]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pca_knn",
                    choices=["knn", "pca_knn", "cnn"])
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--image", default=None,
                    help="predict on a single image file (no webcam)")
    ap.add_argument("--raw", action="store_true",
                    help="image already cropped face — skip Haar + equalize "
                         "(use this for Olivetti samples)")
    args = ap.parse_args()
    if args.image:
        predict_image(args.model, args.image, raw=args.raw)
    else:
        run_webcam(args.model, args.cam, args.threshold)
