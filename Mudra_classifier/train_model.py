"""Train the mudra classifier and export it for Python (mudra_model.p) and the
browser (web/model.json).

    python train_model.py
"""
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from features import augment, clean_label, normalize

rng = np.random.default_rng(42)
df = pd.read_csv("mudra_data.csv")
X = df.iloc[:, :-1].to_numpy(np.float32)
y = df.iloc[:, -1].map(clean_label).to_numpy()

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

copies = 4
X_aug = augment(X_tr, rng, copies=copies)
y_aug = np.tile(y_tr, copies + 1)
F_aug = normalize(X_aug)

scaler = StandardScaler().fit(F_aug)
mlp = MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-4, batch_size=256,
                    max_iter=60, early_stopping=True, n_iter_no_change=6, random_state=0)
mlp.fit(scaler.transform(F_aug), y_aug)


def evaluate(name, predict_proba, classes, T):
    p = predict_proba(T)
    acc = (classes[p.argmax(1)] == y_te).mean()
    print(f"  {name:22s} accuracy {acc:6.1%}   avg confidence {p.max(1).mean():5.1%}")


# Realistic test conditions the webcam produces
def mirror(A):
    B = A.reshape(-1, 42, 3).copy(); B[..., 0] *= -1; return B.reshape(-1, 126)

conds = {
    "clean photos": X_te,
    "mirrored (selfie)": mirror(X_te),
    "hand 30% smaller": X_te * 0.7,
    "16:9 webcam aspect": (X_te.reshape(-1, 42, 3) * [0.75, 1, 1]).reshape(-1, 126),
}
print("New model (MLP + normalisation + augmentation):")
for k, T in conds.items():
    evaluate(k, lambda A: mlp.predict_proba(scaler.transform(normalize(A))), mlp.classes_, T)

with open("mudra_model.p", "wb") as f:
    pickle.dump({"model": mlp, "scaler": scaler, "classes": list(mlp.classes_)}, f)

export = {
    "classes": [str(c) for c in mlp.classes_],
    "mean": scaler.mean_.round(6).tolist(),
    "scale": scaler.scale_.round(6).tolist(),
    "layers": [{"W": W.round(5).tolist(), "b": b.round(5).tolist()}
               for W, b in zip(mlp.coefs_, mlp.intercepts_)],
}
with open("web/model.json", "w") as f:
    json.dump(export, f, separators=(",", ":"))
print("Saved mudra_model.p and web/model.json")
