"""Shared feature pipeline for training, the Python app and the browser app.

Landmarks are always converted to pixel units first (x*W, y*H, z*W) so the
camera's aspect ratio no longer distorts hand shape.

Two models are used:
  * one-hand model:  63 features = one hand's shape
  * two-hand model: 130 features = both hands' shapes + how the hands sit
                     relative to each other (offset, size ratio, gap)

route() decides which model applies to a frame. The browser app
(web/classifier.js) implements exactly the same maths.
"""
import numpy as np

PALM = 9           # middle finger MCP; wrist->MCP9 is a rigid "palm length"
TOUCH_GAP = 1.5    # two hands closer than this many palm lengths = two-hand mudra

DOUBLE = {
    "Anjali", "Berunda", "Chakra", "Garuda", "Kapotham", "Karkatta", "Kartariswastika",
    "Katakavardhana", "Khatva", "Kilaka", "Kurma", "Matsya", "Nagabandha", "Pasha",
    "Pushpaputa", "Sakata", "Samputa", "Shanka", "Shivalinga", "Swastikam", "Varaha",
}


def clean_label(label):
    return str(label).replace("(1)", "").replace("_", " ").strip()


def to_pixels(hand_landmarks, width, height):
    """MediaPipe landmarks of one hand -> (21, 3) array in pixel units."""
    return np.array([[lm.x * width, lm.y * height, lm.z * width] for lm in hand_landmarks],
                    dtype=np.float32)


def palm_length(h):
    return np.linalg.norm(h[..., PALM, :2] - h[..., 0, :2], axis=-1)


def hand_features(h):
    """(..., 21, 3) pixel landmarks -> (..., 63) wrist-relative, unit palm length."""
    rel = h - h[..., :1, :]
    return (rel / palm_length(h)[..., None, None]).reshape(*h.shape[:-2], 63)


def hand_gap(a, b):
    """Smallest distance between the two hands, in mean palm lengths."""
    d = np.linalg.norm(a[..., :, None, :2] - b[..., None, :, :2], axis=-1).min(axis=(-1, -2))
    return d / ((palm_length(a) + palm_length(b)) / 2)


def order_pair(a, b):
    """Canonical order: hand whose wrist is further left in the image first."""
    swap = a[..., 0, 0] > b[..., 0, 0]
    s = swap[..., None, None]
    return np.where(s, b, a), np.where(s, a, b)


def pair_features(a, b):
    """(..., 21, 3) x2 pixel landmarks -> (..., 130)."""
    a, b = order_pair(a, b)
    pa, pb = palm_length(a), palm_length(b)
    mean = (pa + pb) / 2
    offset = (b[..., 0, :2] - a[..., 0, :2]) / mean[..., None]
    extra = np.concatenate([offset, np.log(pb / pa)[..., None], hand_gap(a, b)[..., None]], axis=-1)
    return np.concatenate([hand_features(a), hand_features(b), extra], axis=-1)


def augment(hands, rng):
    """Random mirror, rotation, scale and landmark noise. hands: (N, k, 21, 3)."""
    H = hands.copy()
    n = len(H)
    centre = H[..., :2].mean(axis=(1, 2), keepdims=True)
    H[..., :2] -= centre
    flip = rng.random(n) < 0.5
    H[flip, ..., 0] *= -1                       # mirror: selfie view, left vs right hand
    t = np.deg2rad(rng.uniform(-15, 15, n))[:, None, None]
    x, y = H[..., 0].copy(), H[..., 1].copy()
    H[..., 0] = x * np.cos(t) - y * np.sin(t)
    H[..., 1] = x * np.sin(t) + y * np.cos(t)
    H[..., 0] *= rng.uniform(0.85, 1.15, n)[:, None, None]  # mild stretch
    size = palm_length(H).mean(axis=1)[:, None, None, None]
    H += rng.normal(0, 0.03, H.shape).astype(np.float32) * size
    return H
