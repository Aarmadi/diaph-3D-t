"""Synthetic dataset: two diaphragm domes moving vertically, seen by three sagittal planes.

Coordinates in mm (RAS). Each hemidiaphragm is a dome z = h - ((x - xc)^2 + y^2) / 60 of
4 mm thickness, centred at xc = -20 (left) or +20 (right), with a 4 mm gap between them.
The dome top h is 0 at the inspiratory breath-hold, 20 at the expiratory one and 10 at
the spontaneous one; during the 2D+t sequence it oscillates between 0 and 20."""

import os

import nibabel as nib
import numpy as np

SUBJECT = 'SUBJ01'
VIEW = 'D'
PLANES_X = (14.0, 20.0, 26.0)     # sagittal planes through the right dome
N_FRAMES = 30
HALF = 40                          # field of view [-40, 40) mm, 1 mm voxels


def dome_top(t):
    """Height of the dome top at frame t (period of 10 frames)."""
    return 10.0 - 10.0 * np.cos(2 * np.pi * t / 10.0)


def _inside(x, y, z, h):
    xc = np.where(x >= 0, 20.0, -20.0)
    surface = h - ((x - xc) ** 2 + y ** 2) / 60.0
    return (np.abs(z - surface) <= 2.0) & (np.abs(x) >= 2.0)


def _volume(h):
    g = np.arange(-HALF, HALF, dtype=float)
    x, y, z = np.meshgrid(g, g, g, indexing='ij')
    return _inside(x, y, z, h).astype(np.float32)


def _plane(x0, h):
    g = np.arange(-HALF, HALF, dtype=float)
    y, z = np.meshgrid(g, g, indexing='ij')        # array axes: i -> A (y), j -> S (z)
    return _inside(np.full_like(y, x0), y, z, h).astype(np.float32)


def make_dataset(root):
    """Write the synthetic segmentations and images; returns (seg_dir, img_dir)."""
    seg_dir, img_dir = os.path.join(root, 'seg'), os.path.join(root, 'img')
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    aff3 = np.eye(4)
    aff3[:3, 3] = -HALF
    for state, h in (('inspi', 0.0), ('expi', 20.0), ('normal', 10.0)):
        nib.save(nib.Nifti1Image(_volume(h), aff3),
                 os.path.join(seg_dir, f'{SUBJECT}_{state}_mask_diaphragme.nii.gz'))
    for k, x0 in enumerate(PLANES_X):
        aff = np.zeros((4, 4))
        aff[:3, 0] = (0, 1, 0)
        aff[:3, 1] = (0, 0, 1)
        aff[:3, 2] = (1, 0, 0)
        aff[:3, 3] = (x0, -HALF, -HALF)
        aff[3, 3] = 1
        # Plane k is acquired k/3 frame after plane 0.
        seg = np.stack([_plane(x0, dome_top(t + k / 3)) for t in range(N_FRAMES)], axis=-1)
        img = 0.1 + 0.8 * seg + 0.02 * np.random.default_rng(k).standard_normal(seg.shape)
        nib.save(nib.Nifti1Image(seg, aff), os.path.join(seg_dir, f'{k}_{SUBJECT}_{VIEW}_seg.nii.gz'))
        nib.save(nib.Nifti1Image(img.astype(np.float32), aff),
                 os.path.join(img_dir, f'{k}_{SUBJECT}_{VIEW}.nii.gz'))
    return seg_dir, img_dir


def segmentation_tracker(imgs, anchor, query_rc, model=None, device=None):
    """Stand-in for CoTracker: each point keeps its column and follows the centre of the
    bright band of that column (exact for this vertical motion)."""
    nt = len(imgs)
    tracks = np.zeros((nt, len(query_rc), 2))
    for t, img in enumerate(imgs):
        band = img > 0.5
        for n, (r, c) in enumerate(query_rc):
            rows = np.where(band[:, int(round(c))])[0]
            tracks[t, n] = (rows.mean() if len(rows) else r, c)
    return tracks, np.ones((nt, len(query_rc)), bool), np.ones((nt, len(query_rc)))
