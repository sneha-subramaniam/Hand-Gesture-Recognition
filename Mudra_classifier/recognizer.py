"""Decides which model to use for a frame and returns per-hand predictions.

Same logic as web/classifier.js.

  * Two hands close together (gap < TOUCH_GAP palm lengths) -> two-hand model
  * Two hands far apart -> each hand alone, single-hand mudras only
    (the other hand is just resting in view)
  * One hand -> one-hand model over all mudras (overlapping two-hand mudras
    such as Matsya often look like one hand to the detector)

mode: "auto", "single" (only single-hand mudras) or "double" (only two-hand mudras)
"""
import pickle

import numpy as np

from features import DOUBLE, TOUCH_GAP, hand_features, hand_gap, order_pair, pair_features


class Recognizer:
    def __init__(self, path="mudra_model.p"):
        with open(path, "rb") as f:
            m = pickle.load(f)
        if "one" not in m:
            raise SystemExit("mudra_model.p is from an older version. Run: python train_model.py")
        self.one_clf, self.one_sc = m["one"]
        self.two_clf, self.two_sc = m["two"]
        self.classes = list(self.one_clf.classes_)
        self.double_mask = np.isin(self.classes, list(DOUBLE))
        self.two_index = [self.classes.index(c) for c in self.two_clf.classes_]

    def _one(self, h, allowed):
        p = self.one_clf.predict_proba(self.one_sc.transform(hand_features(h)[None]))[0]
        p = np.where(allowed, p, 0)
        return p / max(p.sum(), 1e-9)

    def _two(self, a, b):
        q = self.two_clf.predict_proba(self.two_sc.transform(pair_features(a, b)[None]))[0]
        p = np.zeros(len(self.classes))
        p[self.two_index] = q
        return p

    def predict(self, hands, mode="auto"):
        """hands: list of (21, 3) pixel-space arrays. Returns [(who, probs), ...]
        where who is "Both hands", "Hand", "Right hand" or "Left hand" (as the
        dancer sees it in a mirrored preview)."""
        single_only = ~self.double_mask
        if len(hands) >= 2:
            a, b = order_pair(hands[0], hands[1])
            if mode != "single" and hand_gap(a, b) < TOUCH_GAP:
                return [("Both hands", self._two(a, b))]
            if mode == "double":
                return []  # hands too far apart for a two-hand mudra
            # a is further left in the camera image = the dancer's right hand
            return [("Right hand", self._one(a, single_only)), ("Left hand", self._one(b, single_only))]
        if len(hands) == 1:
            allowed = {"auto": np.ones(len(self.classes), bool), "single": single_only,
                       "double": self.double_mask}[mode]
            return [("Hand", self._one(hands[0], allowed))]
        return []
