"""Train the one-hand and two-hand mudra models.

    python collect_data.py   # once, creates mudra_landmarks.npz
    python train_model.py    # creates mudra_model.p and web/model.json

Testing uses a held-out block of each class (the highest-numbered images), which
is stricter than a random split because neighbouring images in this dataset are
often near-identical shots.
"""
import json
import pickle
import re

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from features import DOUBLE, TOUCH_GAP, augment, hand_features, pair_features

COPIES = 4
rng = np.random.default_rng(42)

data = np.load("mudra_landmarks.npz")
labels, files, hands, n_hands = data["labels"], data["files"], data["hands"], data["n_hands"]
is_double = np.isin(labels, list(DOUBLE))

# Held-out test block: top 20% of image numbers in each class
num = np.array([int(re.findall(r"\d+", f)[-1]) for f in files])
test = np.zeros(len(labels), bool)
for c in np.unique(labels):
    idx = np.where(labels == c)[0]
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
    json.dump({"version": 2, "touch_gap": TOUCH_GAP, "double": sorted(DOUBLE),
               "one": export(one_clf, one_sc), "two": export(two_clf, two_sc)},
              f, separators=(",", ":"))
print("\nSaved mudra_model.p and web/model.json")
