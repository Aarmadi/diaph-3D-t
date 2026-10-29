"""2D+t frame access and resampling to the fixed tracking grid."""

import numpy as np

# Every 2D+t frame (image and segmentation) is resampled to this grid before tracking.
TARGET_H = 512
TARGET_W = 448


def n_frames(arr):
    """Number of frames of a 2D+t array of shape (H, W, T) or (H, W, 1, T)."""
    return arr.shape[2] if arr.ndim == 3 else arr.shape[3]


def get_frame(arr, t):
    """Frame t of a 2D+t array (clamped to the last frame)."""
    t = int(min(t, n_frames(arr) - 1))
    return arr[:, :, t] if arr.ndim == 3 else arr[:, :, 0, t]


def preprocess_frame(frame_ni_nj):
    """Image frame -> (TARGET_H, TARGET_W) float image in [0, 1]."""
    from PIL import Image
    arr = frame_ni_nj.T  # (nj, ni)
    arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-6)
    img = Image.fromarray((arr * 255).astype(np.uint8)).resize(
        (TARGET_W, TARGET_H), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def preprocess_seg(seg_ni_nj):
    """Segmentation frame -> (TARGET_H, TARGET_W) binary mask."""
    from PIL import Image
    arr = (seg_ni_nj > 0.5).astype(np.uint8)
    img = Image.fromarray((arr.T * 255).astype(np.uint8)).resize(
        (TARGET_W, TARGET_H), Image.NEAREST)
    return (np.array(img, dtype=np.float32) > 127).astype(np.float32)
