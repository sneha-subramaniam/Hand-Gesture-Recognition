"""Shared feature pipeline for training, the Python app and the browser app.

Raw input: up to 2 hands x 21 landmarks x (x, y, z), wrist-relative, zero-padded
to 126 values (this is the format stored in mudra_data.csv).

normalize() makes the features independent of:
  * distance from camera  -> every hand is scaled to unit palm length
  * camera aspect ratio    -> handled by training-time augmentation
  * mirroring / left-right -> handled by training-time augmentation
  * hand order             -> handled by training-time augmentation
The browser app (web/app.js) implements exactly the same normalize().
"""
import numpy as np

N_HANDS, N_LM = 2, 21
PALM = 9  # middle finger MCP; wrist->MCP9 is a rigid "palm length"


def landmarks_to_raw(hand_landmarks_list):
    """MediaPipe hand_landmarks result -> 126 wrist-relative values."""
    coords = []
    for hand in hand_landmarks_list[:N_HANDS]:
        w = hand[0]
        for lm in hand:
            coords.extend([lm.x - w.x, lm.y - w.y, lm.z - w.z])
    coords.extend([0.0] * (N_HANDS * N_LM * 3 - len(coords)))
    return np.array(coords, dtype=np.float32)


def normalize(raw):
    """raw: (..., 126) -> (..., 126) scale-normalised per hand."""
    A = np.asarray(raw, dtype=np.float32).reshape(-1, N_HANDS, N_LM, 3).copy()
    palm = np.linalg.norm(A[:, :, PALM, :2], axis=-1, keepdims=True)[..., None]
    present = palm > 1e-6
    A = np.where(present, A / np.where(present, palm, 1.0), 0.0)
    return A.reshape(np.shape(raw))


def augment(raw, rng, copies=4):
    """Return augmented copies of raw (wrist-relative) samples."""
    A = np.asarray(raw, dtype=np.float32).reshape(-1, N_HANDS, N_LM, 3)
    out = [A]
    for _ in range(copies):
        B = A.copy()
        n = len(B)
        # mirror half the samples (selfie view / left vs right hand)
        flip = rng.random(n) < 0.5
        B[flip, :, :, 0] *= -1
        # aspect ratio / anisotropic distortion (webcam 16:9 vs photos)
        B[:, :, :, 0] *= rng.uniform(0.7, 1.4, n)[:, None, None]
        # small in-plane rotation
        t = np.deg2rad(rng.uniform(-15, 15, n))[:, None, None]
        x, y = B[..., 0].copy(), B[..., 1].copy()
        B[..., 0] = x * np.cos(t) - y * np.sin(t)
        B[..., 1] = x * np.sin(t) + y * np.cos(t)
        # landmark jitter (relative to hand size)
        size = np.linalg.norm(B[:, :, PALM, :2], axis=-1)[..., None, None]
        B += rng.normal(0, 0.03, B.shape).astype(np.float32) * size
        # swap hand order for two-hand samples (MediaPipe order is arbitrary)
        two = np.abs(B[:, 1]).sum(axis=(1, 2)) > 0
        swap = two & (rng.random(n) < 0.5)
        B[swap] = B[swap][:, ::-1]
        out.append(B)
    return np.concatenate(out).reshape(-1, N_HANDS * N_LM * 3)


def clean_label(label):
    return str(label).replace("(1)", "").replace("_", " ").strip()
