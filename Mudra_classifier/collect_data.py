"""Extract hand landmarks from the image dataset into mudra_landmarks.npz.

Dataset: https://github.com/jisharajr/Bharatanatyam-Mudra-Dataset

    git clone --depth 1 https://github.com/jisharajr/Bharatanatyam-Mudra-Dataset
    python collect_data.py --data ./Bharatanatyam-Mudra-Dataset

Landmarks are stored in pixel units for every detected hand (up to 2), so the
position of the hands relative to each other is kept.
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np

from features import clean_label

MODEL_PATH = "hand_landmarker.task"
_detector = None


def _init():
    global _detector
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=2, min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3)
    _detector = vision.HandLandmarker.create_from_options(options)


def _extract(item):
    import mediapipe as mp
    label, path = item
    try:
        image = mp.Image.create_from_file(path)
        result = _detector.detect(image)
    except Exception:
        return None
    hands = np.full((2, 21, 3), np.nan, np.float32)
    for i, hand in enumerate(result.hand_landmarks[:2]):
        hands[i] = [[lm.x * image.width, lm.y * image.height, lm.z * image.width] for lm in hand]
    return label, os.path.basename(path), hands, len(result.hand_landmarks[:2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./Bharatanatyam-Mudra-Dataset")
    ap.add_argument("--out", default="mudra_landmarks.npz")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    items = []
    for folder in sorted(os.listdir(args.data)):
        d = os.path.join(args.data, folder)
        if folder.startswith(".") or not os.path.isdir(d):
            continue
        items += [(clean_label(folder), os.path.join(d, f)) for f in sorted(os.listdir(d))
                  if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
    print(f"{len(items)} images in {args.data}")

    rows = []
    with Pool(args.workers, initializer=_init) as pool:
        for i, r in enumerate(pool.imap(_extract, items, chunksize=32)):
            if r is not None and r[3] > 0:
                rows.append(r)
            if i % 2000 == 0:
                print(f"  {i}/{len(items)}", flush=True)

    labels, files, hands, n_hands = zip(*rows)
    np.savez_compressed(args.out, labels=np.array(labels), files=np.array(files),
                        hands=np.stack(hands), n_hands=np.array(n_hands, np.int8))
    print(f"Saved {len(rows)} samples with hands to {args.out}")


if __name__ == "__main__":
    main()
