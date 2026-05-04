"""Face detection + preprocessing using Haar cascade."""
from pathlib import Path
import cv2
import numpy as np

CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"


def get_cascade():
    cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
    if cascade.empty():
        raise RuntimeError(f"Failed to load cascade at {CASCADE_PATH}")
    return cascade


def detect_face(gray, cascade=None, min_size=(30, 30)):
    if cascade is None:
        cascade = get_cascade()
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=min_size
    )
    if len(faces) == 0:
        return None
    # pick largest
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    return gray[y:y + h, x:x + w], (x, y, w, h)


def preprocess_image(path, size=(64, 64), cascade=None, fallback_full=True):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = detect_face(gray, cascade)
    if out is None:
        if not fallback_full:
            return None
        face = gray
    else:
        face = out[0]
    face = cv2.resize(face, size)
    face = cv2.equalizeHist(face)
    return face.astype(np.float32) / 255.0


def preprocess_array(gray, size=(64, 64), cascade=None):
    """Already grayscale numpy array → cropped + resized + normalized."""
    out = detect_face(gray, cascade)
    face = out[0] if out is not None else gray
    face = cv2.resize(face, size)
    face = cv2.equalizeHist(face)
    return face.astype(np.float32) / 255.0
