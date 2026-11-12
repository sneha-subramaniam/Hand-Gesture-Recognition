"""Train the one-hand and two-hand mudra models.

    python collect_data.py   # once, creates mudra_landmarks.npz
    python train_model.py    # creates mudra_model.p and web/model.json
    python train_model.py --extra my_mudras.json   # also learn from samples
                                                   # exported from the webpage

Testing uses a held-out block of each class (the highest-numbered images), which
is stricter than a random split because neighbouring images in this dataset are
often near-identical shots.
"""
import argparse
import json
import pickle
import re

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from features import DOUBLE, TOUCH_GAP, augment, hand_features, pair_features

ap = argparse.ArgumentParser()
ap.add_argument("--extra", nargs="*", default=[], help="sample files exported from the webpage")
args = ap.parse_args()

COPIES = 4
rng = np.random.default_rng(42)

data = np.load("mudra_landmarks.npz")
labels, files, hands, n_hands = data["labels"], data["files"], data["hands"], data["n_hands"]

# Personal samples recorded on the webpage ("Teach" panel -> Export). They are
# only ever used for training, never for the held-out test numbers.
extra_l, extra_h, extra_n = [], [], []
for path in args.extra:
    with open(path) as f:
        for s in json.load(f)["samples"]:
            h = np.full((2, 21, 3), np.nan, np.float32)
            for i, hand in enumerate(s["hands"][:2]):
                h[i] = hand
            extra_l.append(s["label"]); extra_h.append(h); extra_n.append(min(len(s["hands"]), 2))
if extra_l:
    print(f"Adding {len(extra_l)} personal samples from {', '.join(args.extra)}")
    labels = np.concatenate([labels, extra_l]); hands = np.concatenate([hands, extra_h])
    n_hands = np.concatenate([n_hands, np.array(extra_n, np.int8)])
    files = np.concatenate([files, [f"personal_{i}" for i in range(len(extra_l))]])
is_double = np.isin(labels, list(DOUBLE))

# Held-out test block: top 20% of image numbers in each class
num = np.array([int(re.findall(r"\d+", f)[-1]) for f in files])
personal = np.char.startswith(files.astype(str), "personal_")
test = np.zeros(len(labels), bool)
for c in np.unique(labels):
    idx = np.where((labels == c) & ~personal)[0]
    test[idx[num[idx] >= np.percentile(num[idx], 80)]] = True

# One-hand samples: every single-hand detection, plus the first hand of the rare
# single-hand-mudra images where MediaPipe reported the same hand twice.
one = (n_hands == 1) | ((n_hands == 2) & ~is_double)
two = (n_hands == 2) & is_double


def fit(X, y, name):
    scaler = StandardScaler().fit(X)
    clf = MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-4, batch_size=256, max_iter=80,
                        early_stopping=True, n_iter_no_change=6, random_state=0)
    clf.fit(scaler.transform(X), y)
    print(f"{name}: {len(clf.classes_)} classes, {len(y)} training rows (with augmentation)")
    return clf, scaler


def one_hand_set(mask, aug):
    H = hands[mask][:, :1]
    if aug:
        H = np.concatenate([H] + [augment(H, rng) for _ in range(COPIES)])
    return hand_features(H[:, 0]), np.tile(labels[mask], len(H) // mask.sum())


def two_hand_set(mask, aug):
    H = hands[mask]
    if aug:
        H = np.concatenate([H] + [augment(H, rng) for _ in range(COPIES)])
    return pair_features(H[:, 0], H[:, 1]), np.tile(labels[mask], len(H) // mask.sum())


one_clf, one_sc = fit(*one_hand_set(one & ~test, True), "One-hand model")
two_clf, two_sc = fit(*two_hand_set(two & ~test, True), "Two-hand model")

# ---- Evaluation on held-out images ----
print("\nHeld-out accuracy")
X, y = one_hand_set(one & test, False)
p = one_clf.classes_[one_clf.predict_proba(one_sc.transform(X)).argmax(1)]
single_true = ~np.isin(y, list(DOUBLE))
print(f"  one hand visible, all mudras      {np.mean(p == y):6.1%}  (n={len(y)})")
print(f"  one hand visible, single mudras   {np.mean(p[single_true] == y[single_true]):6.1%}")
print(f"  single mudra called a two-hand mudra   {np.mean(np.isin(p[single_true], list(DOUBLE))):6.1%}")


def tilt(H, deg):
    t = np.deg2rad(deg)
    c = H[..., :2].mean(axis=(-2, -3), keepdims=True)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]], np.float32)
    H = H.copy()
    H[..., :2] = (H[..., :2] - c) @ R.T + c
    return H


for deg in (-30, 30):
    Ht = tilt(hands[one & test][:, 0], deg)
    p = one_clf.classes_[one_clf.predict_proba(one_sc.transform(hand_features(Ht))).argmax(1)]
    print(f"  one hand, tilted {deg:+d} degrees    {np.mean(p == labels[one & test]):6.1%}")
X, y = two_hand_set(two & test, False)
p = two_clf.classes_[two_clf.predict_proba(two_sc.transform(X)).argmax(1)]
print(f"  both hands visible, two-hand mudras {np.mean(p == y):6.1%}  (n={len(y)})")

with open("mudra_model.p", "wb") as f:
    pickle.dump({"one": (one_clf, one_sc), "two": (two_clf, two_sc)}, f)


def export(clf, sc):
    return {
        "classes": [str(c) for c in clf.classes_],
        "mean": sc.mean_.round(6).tolist(),
        "scale": sc.scale_.round(6).tolist(),
        "layers": [{"W": W.round(5).tolist(), "b": b.round(5).tolist()}
                   for W, b in zip(clf.coefs_, clf.intercepts_)],
    }


with open("web/model.json", "w") as f:
    # personal_tau: how close (in standardised feature distance) a live frame must
    # be to a taught sample before the taught label takes over
    json.dump({"version": 3, "touch_gap": TOUCH_GAP, "double": sorted(DOUBLE),
               "personal_tau": {"one": 2.0, "two": 2.8},
               "one": export(one_clf, one_sc), "two": export(two_clf, two_sc)},
              f, separators=(",", ":"))
print("\nSaved mudra_model.p and web/model.json")
