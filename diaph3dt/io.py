"""Input files: discovery and loading.

Expected layout (see README):
  <seg_dir>/<plane>_<subject>_<view>_seg.nii.gz      2D+t segmentation of one plane
  <img_dir>/<plane>_<subject>_<view>.nii.gz          2D+t images of the same plane
  <seg_dir>/<subject>_<state>_mask_diaphragme.nii.gz 3D breath-hold segmentation
with <view> in {D (right sagittal), G (left sagittal), cor (coronal)}, <plane> an integer
(acquisition order of the planes of a view) and <state> in {inspi, expi, normal,
normal_inspi, normal_expi}."""

import glob
import os
import re

import nibabel as nib
import numpy as np
from scipy.ndimage import label as _label_cc

from .geometry import compute_phi
from .imaging import get_frame, n_frames, preprocess_frame, preprocess_seg

VIEWS = ('D', 'G', 'cor')
_SEG_PAT = re.compile(r'^(\d+)_(.+?)_(cor|D|G)(_2d|_prediction_2d|_seg)\.nii\.gz$')
_RAW_PAT = re.compile(r'^(\d+)_(.+?)_(cor|D|G)\.nii\.gz$')


def mask_path(seg_dir, subject, state):
    """3D breath-hold segmentation of a subject (state: inspi, expi, normal, ...)."""
    return os.path.join(seg_dir, f'{subject}_{state}_mask_diaphragme.nii.gz')


def find_subjects(seg_dir):
    """Subjects with both an inspiratory and an expiratory 3D segmentation."""
    out = []
    for p in sorted(glob.glob(os.path.join(seg_dir, '*_inspi_mask_diaphragme.nii.gz'))):
        subject = os.path.basename(p).split('_inspi_mask_diaphragme')[0]
        if os.path.exists(mask_path(seg_dir, subject, 'expi')):
            out.append(subject)
    return out


def find_planes(seg_dir, img_dir, subject, view):
    """Planes of a view having both a segmentation and images, in acquisition order."""
    def planes(folder, pat):
        return {m.group(1) for m in map(pat.match, os.listdir(folder))
                if m and m.group(2) == subject and m.group(3) == view}
    return sorted(planes(seg_dir, _SEG_PAT) & planes(img_dir, _RAW_PAT), key=int)


def _find(folder, pat, subject, view, plane=None):
    for fn in sorted(os.listdir(folder)):
        m = pat.match(fn)
        if m and m.group(2) == subject and m.group(3) == view:
            if plane is None or m.group(1) == str(plane):
                return os.path.join(folder, fn), m.group(1)
    return None, None


def find_seg(seg_dir, subject, view, plane=None):
    """(path, plane) of the 2D+t segmentation of a plane (first plane if None)."""
    return _find(seg_dir, _SEG_PAT, subject, view, plane)


def find_raw(img_dir, subject, view, plane=None):
    """(path, plane) of the 2D+t images of a plane (first plane if None)."""
    return _find(img_dir, _RAW_PAT, subject, view, plane)


def largest_component_2d(seg_hw):
    """Keep the largest connected component of a 2D mask (used for sagittal views only:
    the coronal view legitimately shows two disjoint hemidiaphragms).
    Returns (mask, changed)."""
    mask = seg_hw > 0.5
    if not mask.any():
        return seg_hw, False
    labeled, n = _label_cc(mask, structure=np.ones((3, 3)))
    if n <= 1:
        return seg_hw, False
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    out = seg_hw.copy()
    out[labeled != int(sizes.argmax())] = 0
    return out, True


def load_plane_data(seg_dir, img_dir, subject, view, plane):
    """Load one 2D+t plane: preprocessed images and segmentations, and phi(t).

    Returns a dict with imgs, segs (lists of (TARGET_H, TARGET_W) arrays), phis,
    valid_frames (frames with a defined phi), nt, and the plane geometry (aff_seg, ni, nj)."""
    seg_path, plane = find_seg(seg_dir, subject, view, plane)
    raw_path, _ = find_raw(img_dir, subject, view, plane)
    if seg_path is None or raw_path is None:
        raise FileNotFoundError(f'No segmentation/images for {subject} {view} plane {plane} '
                                f'in {seg_dir} / {img_dir}')
    img_s = nib.load(seg_path)
    arr_s, aff_s = np.asarray(img_s.get_fdata(), dtype=np.float32), img_s.affine
    arr_r = np.asarray(nib.load(raw_path).get_fdata(), dtype=np.float32)
    nt = n_frames(arr_s)
    imgs = [preprocess_frame(get_frame(arr_r, t)) for t in range(nt)]
    segs = [preprocess_seg(get_frame(arr_s, t)) for t in range(nt)]
    if view in ('D', 'G'):
        segs = [largest_component_2d(s)[0] for s in segs]

    def load(state):
        img = nib.load(mask_path(seg_dir, subject, state))
        return np.asarray(img.get_fdata(), dtype=np.float32), img.affine

    phis = compute_phi(arr_s, aff_s, *load('inspi'), *load('expi'))
    valid_frames = [t for t, p in enumerate(phis) if p is not None]
    if not valid_frames:
        raise ValueError(f'{subject} {view} plane {plane}: no frame with a defined phi.')
    return dict(plane=plane, imgs=imgs, segs=segs, phis=phis, valid_frames=valid_frames,
                nt=nt, aff_seg=aff_s, ni=arr_s.shape[0], nj=arr_s.shape[1])


def planes_geometry(seg_dir, subject, view, planes):
    """(affine, ni, nj) of each 2D+t plane, used to select the hemidiaphragm the view
    observes (see largest_connected_component)."""
    geom = []
    for plane in planes:
        seg_path, _ = find_seg(seg_dir, subject, view, plane)
        if seg_path is None:
            continue
        img = nib.load(seg_path)
        geom.append((img.affine, img.shape[0], img.shape[1]))
    return geom


# Output layout: one folder per reconstructed case, <out_dir>/<subject>_<view>/
CASE_FILES = {
    'points': 'control_points.json',      # control points and their trajectories
    'tracking': 'tracking.html',          # 2D+t tracking viewer
    'anchoring': 'anchoring.html',        # trajectory / template anchoring viewer
    'viewer': 'reconstruction.html',      # dynamic reconstruction viewer
}


def case_dir(out_dir, subject, view):
    """Output folder of one reconstructed case."""
    return os.path.join(out_dir, f'{subject}_{view}')


def case_file(case_dir_, kind):
    """Path of an output file of a case; kind is a key of CASE_FILES or a breath-hold
    state (inspi, expi, normal, ...) for the 3D mask filtered for the view."""
    name = CASE_FILES.get(kind, f'mask_{kind}.nii.gz')
    return os.path.join(case_dir_, name)
