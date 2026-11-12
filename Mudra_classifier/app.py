"""Live mudra recognition from a webcam (desktop version).

    python app.py              # auto-pick a working camera
    python app.py --camera 1   # force a camera index

If the camera will not open on macOS, the browser version (web/index.html) is
usually the easier route. See README.md.
"""
import argparse
import collections
import platform
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from features import to_pixels
from recognizer import Recognizer

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


def draw_results(frame, results, threshold, classes):
    y = 45
    for who, probs in results:
        top = np.argsort(probs)[::-1][:3]
        if len(results) > 1 or who == "Both hands":
            cv2.putText(frame, who.upper(), (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
            y += 34
        if probs[top[0]] >= threshold:
            text, color = f"{classes[top[0]]} ({probs[top[0]]:.0%})", (60, 200, 60)
        else:
            text, color = "Not sure yet", (0, 165, 255)
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5)
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
        y += 32
        for k in top:
            cv2.putText(frame, f"{classes[k]}: {probs[k]:.0%}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 24
        y += 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--mode", choices=["auto", "single", "double"], default="auto",
                    help="auto = all mudras, single = single-hand only, double = two-hand only")
    args = ap.parse_args()

    recognizer = Recognizer("mudra_model.p")
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
    history = {}  # who -> recent probability vectors
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

        frame = cv2.flip(frame, 1)  # mirror for display only

        results = recognizer.predict([to_pixels(hd, w, h) for hd in hands], args.mode)
        for k in list(history):
            if k not in {who for who, _ in results}:
                del history[k]
        smoothed = []
        for who, probs in results:
            dq = history.setdefault(who, collections.deque(maxlen=SMOOTH))
            dq.append(probs)
            smoothed.append((who, np.mean(dq, axis=0)))

        if not hands:
            cv2.putText(frame, "Show your hand(s)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        elif not results:
            cv2.putText(frame, "Bring your hands together", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        else:
            draw_results(frame, smoothed, args.threshold, recognizer.classes)

        cv2.imshow(WINDOW, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
