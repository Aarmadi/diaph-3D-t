"""Scanner-frame geometry shared by the 2D+t planes and the 3D volumes.

All acquisitions of a subject share the scanner (RAS) coordinate system, so a point of a
2D+t plane is mapped to 3D through the plane affine, without image registration."""

import numpy as np
from scipy.ndimage import label as _cc_label

from .imaging import TARGET_H, TARGET_W, get_frame, n_frames

SLICE_MM = 6.0         # 2D+t slice thickness
PHI_PERCENTILE = 90    # dome height = 90th percentile of the segmented heights


def ijk_to_ras(ijk, aff):
    """Voxel indices (N, 3) -> RAS coordinates (N, 3), in mm."""
    h = np.hstack([ijk, np.ones((len(ijk), 1))])
    return (aff @ h.T).T[:, :3]


def plane_axis(aff_2d):
    """Index (0=R, 1=A, 2=S) of the axis normal to a 2D+t plane."""
    return int(np.argmax(np.abs(aff_2d[:3, 2])))


def plane_center_ras(ni, nj, aff_2d):
    """Position of the plane along its normal axis, in mm."""
    ax = plane_axis(aff_2d)
    return float((aff_2d @ np.array([ni / 2, nj / 2, 0, 1]))[ax])


def voxels_near_plane(arr_3d, aff_3d, plane_pos, p_ax):
    """RAS coordinates of the voxels of a 3D mask lying within the slice thickness."""
    pts = np.argwhere(arr_3d > 0).astype(float)
    if not len(pts):
        return np.empty((0, 3))
    ras = ijk_to_ras(pts, aff_3d)
    return ras[np.abs(ras[:, p_ax] - plane_pos) <= SLICE_MM / 2]


def model_pts_to_ras(pts_hw, aff_2d, ni, nj):
    """Points (row, col) of the tracking grid (TARGET_H x TARGET_W) -> RAS (mm).

    Inverse of the preprocessing (transpose + resize): model row <-> j, model col <-> i."""
    j_orig = pts_hw[:, 0] * (nj / TARGET_H)
    i_orig = pts_hw[:, 1] * (ni / TARGET_W)
    ijk = np.column_stack([i_orig, j_orig, np.zeros(len(pts_hw))])
    return ijk_to_ras(ijk, aff_2d)


def split_by_ras_x(mask, aff_3d):
    """Split a fused diaphragm mask into its two halves along RAS x.

    Returns (left, right): small x = left, large x = right."""
    pts = np.argwhere(mask > 0.5).astype(float)
    small = np.zeros_like(mask, dtype=np.uint8)
    large = np.zeros_like(mask, dtype=np.uint8)
    if len(pts) == 0:
        return small, large
    ras_x = ijk_to_ras(pts, aff_3d)[:, 0]
    # Geometric middle of the x extent (not the voxel median, which a partially
    # segmented hemidiaphragm would bias), refined by the mean of each side.
    geo_mid = (ras_x.min() + ras_x.max()) / 2.0
    lo, hi = ras_x < geo_mid, ras_x >= geo_mid
    cut = geo_mid if not lo.any() or not hi.any() else (ras_x[lo].mean() + ras_x[hi].mean()) / 2.0
    idx_small = pts[ras_x < cut].astype(int)
    idx_large = pts[ras_x >= cut].astype(int)
    if len(idx_small):
        small[tuple(idx_small.T)] = 1
    if len(idx_large):
        large[tuple(idx_large.T)] = 1
    return small, large


def _plane_overlap(ras, planes_geom):
    """Number of points lying within the slice thickness of the given planes."""
    count = 0
    for aff_2d, ni, nj in planes_geom:
        p_ax = plane_axis(aff_2d)
        count += int(np.sum(np.abs(ras[:, p_ax] - plane_center_ras(ni, nj, aff_2d)) <= SLICE_MM / 2))
    return count


def largest_connected_component(mask, aff_3d=None, planes_geom=None, view=None,
                                assume_prefiltered=False):
    """Keep the part of a 3D diaphragm mask observed by the planes of a view.

    Coronal view: the whole mask (both hemidiaphragms). Sagittal views: the connected
    component crossed by the planes (`planes_geom`, list of (affine, ni, nj)); a single
    fused component is first split into its two halves. Without planes, the largest
    component is kept."""
    labeled, n = _cc_label(mask > 0.5)
    if view == 'cor':
        return mask
    if n <= 1:
        if assume_prefiltered or view not in ('D', 'G') or aff_3d is None or not mask.any():
            return mask
        left, right = split_by_ras_x(mask, aff_3d)
        if planes_geom:
            def score(piece):
                idx = np.argwhere(piece > 0).astype(float)
                return _plane_overlap(ijk_to_ras(idx, aff_3d), planes_geom) if len(idx) else -1
            return left if score(left) >= score(right) else right
        return left if view == 'G' else right
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    if planes_geom:
        best_label, best_count = 1, -1
        for label_id in range(1, n + 1):
            ras = ijk_to_ras(np.argwhere(labeled == label_id).astype(float), aff_3d)
            count = _plane_overlap(ras, planes_geom)
            if count > best_count:
                best_count, best_label = count, label_id
        return (labeled == best_label).astype(mask.dtype)
    return (labeled == int(sizes.argmax())).astype(mask.dtype)


def compute_phi(arr_seg, aff_seg, arr_inspi, aff_inspi, arr_expi, aff_expi):
    """Respiratory level phi(t) of each frame of a 2D+t plane.

    phi = (h(t) - h_inspi) / (h_expi - h_inspi), where h is the dome height (90th
    percentile of the segmented heights, RAS z): 0 = inspiratory breath-hold,
    1 = expiratory breath-hold. h_inspi and h_expi are measured on the 3D masks within the
    slice thickness of the plane. None for frames without segmentation."""
    ni, nj = arr_seg.shape[0], arr_seg.shape[1]
    p_ax = plane_axis(aff_seg)
    p_center = plane_center_ras(ni, nj, aff_seg)

    def height(arr3d, aff3d):
        vox = voxels_near_plane(arr3d, aff3d, p_center, p_ax)
        return float(np.percentile(vox[:, 2], PHI_PERCENTILE)) if len(vox) else None

    h_inspi, h_expi = height(arr_inspi, aff_inspi), height(arr_expi, aff_expi)
    if h_inspi is None or h_expi is None:
        raise ValueError('Breath-hold masks do not intersect this plane.')
    denom = h_expi - h_inspi
    if abs(denom) < 1e-6:
        raise ValueError(f'Inspiratory and expiratory heights coincide (denom={denom:.4f}).')
    phis = []
    for t in range(n_frames(arr_seg)):
        pts = np.argwhere(get_frame(arr_seg, t) > 0).astype(float)
        if not len(pts):
            phis.append(None)
            continue
        z = ijk_to_ras(np.column_stack([pts, np.zeros(len(pts))]), aff_seg)[:, 2]
        phis.append((float(np.percentile(z, PHI_PERCENTILE)) - h_inspi) / denom)
    return phis
