import argparse
from pathlib import Path
import cv2

from .preprocess import get_cascade


def capture(name, count=30, out_dir="data/raw", cam_index=0):
    out = Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    cascade = get_cascade()
    cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        print("[ERROR] Webcam not accessible.")
        print("👉 Fix: macOS Settings → Privacy → Camera → allow Terminal/Python")
        return

    saved = 0
    print(f"[capture] SPACE = save | q = quit | target={count}")

    while saved < count:
        ok, frame = cap.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(frame, f"{saved}/{count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("capture", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord(" ") and len(faces) > 0:
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            face = gray[y:y+h, x:x+w]

            path = out / f"{name}_{saved:03d}.jpg"
            cv2.imwrite(str(path), face)

            saved += 1
            print("saved", path)

    cap.release()
    cv2.destroyAllWindows()
    print(f"[done] {saved} images → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--cam", type=int, default=0)
    args = ap.parse_args()

    capture(args.name, args.count, args.out, args.cam)
