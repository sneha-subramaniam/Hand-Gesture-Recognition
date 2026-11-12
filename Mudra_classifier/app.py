"""Live mudra recognition from a webcam (desktop version).

    python app.py              # auto-pick a working camera
    python app.py --camera 1   # force a camera index

If the camera will not open on macOS, the browser version (web/index.html) is
usually the easier route. See README.md.
"""
import argparse
import collections
import pickle
import platform
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from features import landmarks_to_raw, normalize

WINDOW = "Mudra Recognizer (press q to quit)"
SMOOTH = 8
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def open_camera(preferred=None):
    """Try camera indices until one actually delivers frames.

    On macOS index 0 is sometimes an iPhone (Continuity Camera) or a virtual
    camera, and AVFoundation often returns a few empty frames while warming up.
    """
    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
    indices = [preferred] if preferred is not None else [0, 1, 2]
    for idx in indices:
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        for _ in range(30):  # ~1-2 s warm-up
            ok, frame = cap.read()
            if ok and frame is not None and frame.mean() > 1:
                print(f"Using camera {idx}")
                return cap
            time.sleep(0.05)
        cap.release()
    msg = "Could not get frames from any camera."
    if platform.system() == "Darwin":
        msg += ("\nOn macOS: System Settings > Privacy & Security > Camera, enable the app you run "
                "this from (Terminal, iTerm, VS Code, PyCharm), then fully quit and reopen it. "
                "Also close Zoom/FaceTime/Photo Booth. Or use the browser version: web/index.html")
    sys.exit(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=0.6)
    args = ap.parse_args()

    with open("mudra_model.p", "rb") as f:
        bundle = pickle.load(f)
    if "scaler" not in bundle:
        sys.exit("mudra_model.p is from the old training script. Run: python train_model.py")
    clf, scaler = bundle["model"], bundle["scaler"]
    classes = clf.classes_

    def predict(raw):
        return clf.predict_proba(scaler.transform(normalize(raw[None])))[0]

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    detector = vision.HandLandmarker.create_from_options(options)
    cap = open_camera(args.camera)
    history = collections.deque(maxlen=SMOOTH)
    last_ts, misses = 0, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            misses += 1
            if misses > 60:
                print("Camera stopped sending frames.")
                break
            continue
        misses = 0

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = max(int(time.monotonic() * 1000), last_ts + 1)  # must strictly increase
        last_ts = ts
        result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        h, w = frame.shape[:2]
        hands = result.hand_landmarks
        for hand in hands:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (95, 138, 236), 2)
            for p in pts:
                cv2.circle(frame, p, 3, (255, 255, 255), -1)

        frame = cv2.flip(frame, 1)  # mirror for display only (model is mirror-invariant)

        if hands:
            probs = predict(landmarks_to_raw(hands))
            if len(hands) > 1:  # a one-hand mudra with the other hand resting in view
                for single in (predict(landmarks_to_raw([hd])) for hd in hands):
                    if single.max() > probs.max() + 0.25:
                        probs = single
            history.append(probs)
            avg = np.mean(history, axis=0)
            top = np.argsort(avg)[::-1][:3]
            if avg[top[0]] >= args.threshold:
                text, color = f"{classes[top[0]]} ({avg[top[0]]:.0%})", (60, 200, 60)
            else:
                text, color = "Not sure yet", (0, 165, 255)
            cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5)
            cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            for i, k in enumerate(top):
                cv2.putText(frame, f"{classes[k]}: {avg[k]:.0%}", (20, 90 + 28 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            history.clear()
            cv2.putText(frame, "Show your hand(s)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

        cv2.imshow(WINDOW, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
