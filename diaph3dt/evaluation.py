"""Evaluation of the dynamic reconstruction against 3D breath-hold segmentations.

The reconstruction of a case is compared with the 3D segmentation of a breath-hold that
was not used to build it (spontaneous breath-hold 'normal', and optionally the
expiratory breath-hold 'expi'). The compared instant t* is chosen without looking at the
reconstruction: it is the instant where the tracked points (observations, before any
deformation) lie closest to the reference surface (median distance per plane, averaged
over the planes). All deformation methods are evaluated at this same instant.

Metrics (distances in mm, from the reference surface to the reconstruction and back):
NSD@tau (fraction of points within tau mm), ASD, p90, HD95; prefix r_ for the reverse
direction (reconstruction -> reference, restricted to the footprint of the reference)
and s_ for the symmetric version (s_asd = ASSD). Each metric is computed on the whole
reference ('all') and on the part between the extreme acquisition planes ('slab')."""

import itertools
import time
from dataclasses import dataclass, field

import nibabel as nib
import numpy as np

from .deformation import BiharmonicDeformer, PointDeformer, closest_barycentric
from .geometry import PHI_PERCENTILE, plane_axis, plane_center_ras, voxels_near_plane
from .io import case_file, find_planes, find_seg, mask_path, planes_geometry
from .metrics import SurfaceDistance, footprint_mask, metrics_from_distances, two_way_metrics
from .template import load_mesh
from .viewer import RBF_REG_N_REF, load_points

REFERENCE_METHODS = ['none', 'rigid', 'affine']
METHODS = ['rbf', 'rbf_affine', 'tps', 'biharm']
TARGETS = ['normal', 'expi']
REGIONS = ['all', 'slab']


@dataclass
class EvalSettings:
    """Parameter grids and evaluation options (defaults: those of the paper)."""
    methods: list = field(default_factory=lambda: list(METHODS))
    targets: list = field(default_factory=lambda: list(TARGETS))
    rbf_sigmas_mm: list = field(default_factory=lambda: [10, 20, 30, 40, 60])
    regs: list = field(default_factory=lambda: [1e-3, 1e-2, 1e-1])
    biharm_weights: list = field(default_factory=lambda: [3e-2, 0.1, 0.3, 1.0, 3.0, 10.0,
                                                          30.0, 100.0])
    interp_steps: int = 3          # interpolated instants between consecutive frames
    expi_reach: float = 1.0        # expi reached if the sequence reaches phi >= expi_reach
    footprint_mm: float = 3.0      # footprint of the reference for the reverse direction
    fixed: dict = None             # {method: params}: evaluate only these (test cohort)


def effective_reg(reg, sigma, sigma_ref, n):
    """Regularisation of the Gaussian RBF as applied in the viewer: grows with sigma and
    with the number of points (the kernel matrix gets ill-conditioned)."""
    return reg * max(1.0, (sigma / sigma_ref) ** 3) * max(1.0, n / RBF_REG_N_REF)


def reference_phase_per_plane(seg_dir, img_dir, subject, view, state):
    """Respiratory level of a breath-hold in each plane of the view, measured as for the
    frames (dome height between the inspiratory (0) and expiratory (1) breath-holds).
    {plane: phi}, empty if unavailable."""
    masks = {}
    for st in ('inspi', 'expi', state):
        img = nib.load(mask_path(seg_dir, subject, st))
        masks[st] = (np.asarray(img.dataobj), img.affine)
    out = {}
    for plane in find_planes(seg_dir, img_dir, subject, view):
        seg_path, _ = find_seg(seg_dir, subject, view, plane)
        seg = nib.load(seg_path)
        ni, nj = seg.shape[0], seg.shape[1]
        p_ax, p_center = plane_axis(seg.affine), plane_center_ras(ni, nj, seg.affine)
        z = {}
        for st, (arr, aff) in masks.items():
            vox = voxels_near_plane(arr, aff, p_center, p_ax)
            z[st] = float(np.percentile(vox[:, 2], PHI_PERCENTILE)) if len(vox) else None
        if None in z.values() or abs(z['expi'] - z['inspi']) < 1e-6:
            continue
        out[str(plane)] = (z[state] - z['inspi']) / (z['expi'] - z['inspi'])
    return out


def interpolated_frames(frames, n_interp):
    """n_interp linearly interpolated instants between consecutive frames (one plane step
    apart, never across a gap); a point missing at either end stays missing."""
    real = sorted(frames, key=lambda fr: fr['t'])
    if n_interp <= 0 or len(real) < 2:
        return []
    dts = np.diff([fr['t'] for fr in real])
    step = float(np.min(dts[dts > 1e-9])) if np.any(dts > 1e-9) else 1.0
    out = []
    for a, b in zip(real[:-1], real[1:]):
        if abs((b['t'] - a['t']) - step) > 1e-6:
            continue
        for k in range(1, n_interp + 1):
            f = k / (n_interp + 1)
            out.append({'t': a['t'] + f * step,
                        'points': a['points'] + f * (b['points'] - a['points']),
                        'interpolated': True})
    return out


def frame_phases(frames):
    """Respiratory level of each frame, taken on the least advanced plane: each real
    frame carries the level of its own plane, interpolated in time for the other planes."""
    per_plane = {}
    for fr in frames:
        op = fr.get('origin_plane')
        if op is not None and fr.get('phi_mean') is not None and not fr.get('interpolated'):
            per_plane.setdefault(op, []).append((fr['t'], fr['phi_mean']))
    per_plane = [np.array(sorted(v)) for v in per_plane.values() if len(v) >= 2]
    if not per_plane:
        return [None] * len(frames)
    return [min(float(np.interp(fr['t'], v[:, 0], v[:, 1])) for v in per_plane) for fr in frames]


def point_planes(rest, geom):
    """Plane of each control point. Returns (unit normal, plane index of each point,
    position of each plane along the normal in mm)."""
    n = np.cross(geom[0][0][:3, 0], geom[0][0][:3, 1])
    n /= np.linalg.norm(n)
    pos = sorted(float(n @ g[0][:3, 3]) for g in geom)
    labels = np.argmin(np.abs((np.asarray(rest) @ n)[:, None] - np.array(pos)[None, :]), axis=1)
    return n, labels, pos


def choose_instant(frames, usable, plane_of, n_planes, target_verts, target_dist):
    """Index of the frame where the tracked points are closest to the reference surface:
    median point-to-surface distance per plane, averaged over the planes (each plane
    weighs the same whatever its number of points)."""
    score = {}
    for i in usable:
        p = np.asarray(frames[i]['points'], float)
        ok = ~np.isnan(p[:, 0])
        per = [float(np.median(target_dist.distances(p[ok & (plane_of == k)], target_verts)))
               for k in range(n_planes) if (ok & (plane_of == k)).sum() >= 3]
        if per:
            score[i] = float(np.mean(per))
    return min(score, key=score.get)


def evaluate_case(seg_dir, img_dir, case_dir_, subject, view, settings, write_row):
    """Evaluate all deformation methods on one reconstructed case; write_row(dict) is
    called for each (method, parameters, reference, region)."""
    s = settings
    verts, faces = load_mesh(case_file(case_dir_, 'inspi'), view, prefiltered=True, top_only=True)
    rest, frames, _ = load_points(case_file(case_dir_, 'points'))
    all_frames = list(frames) + interpolated_frames(frames, s.interp_steps)
    phases = frame_phases(all_frames)
    usable = [i for i, ph in enumerate(phases) if ph is not None]
    if not usable:
        print(f'  {subject} {view}: no frame with a known respiratory level, skipped')
        return
    phase_min = min(phases[i] for i in usable)
    phase_max = max(phases[i] for i in usable)

    planes = find_planes(seg_dir, img_dir, subject, view)
    n_plane, plane_of, plane_pos = point_planes(rest, planes_geometry(seg_dir, subject, view, planes))
    slab_lo, slab_hi = min(plane_pos), max(plane_pos)

    # References: surface, evaluated points per region, chosen instant, reached or not.
    targets = []
    for name in s.targets:
        try:
            tv, tf = load_mesh(case_file(case_dir_, name), view, prefiltered=True, top_only=True)
        except FileNotFoundError:
            continue
        tdist = SurfaceDistance(tf, len(tv))
        proj = tv @ n_plane
        regions = {'all': tv, 'slab': tv[(proj >= slab_lo) & (proj <= slab_hi)]}
        if name == 'expi':
            ref_phase, reached = 1.0, phase_max >= s.expi_reach
        else:
            phi = reference_phase_per_plane(seg_dir, img_dir, subject, view, name)
            ref_phase = float(np.mean(list(phi.values()))) if phi else None
            reached = ref_phase is None or phase_min <= ref_phase <= phase_max
        i_star = choose_instant(all_frames, usable, plane_of, len(plane_pos), tv, tdist)
        targets.append(dict(name=name, verts=tv, dist=tdist, regions=regions, i=i_star,
                            phase=ref_phase, reached=reached))
        print(f'  {subject} {view} {name}: t*={all_frames[i_star]["t"]:.2f} '
              f'(phase {phases[i_star]:.2f}), reference phase '
              + ('unknown' if ref_phase is None else f'{ref_phase:.2f}')
              + f', sequence [{phase_min:.2f}, {phase_max:.2f}]')
    if not targets:
        print(f'  {subject} {view}: no reference breath-hold, skipped')
        return

    frame_data = {}
    for t in targets:
        p = np.asarray(all_frames[t['i']]['points'], float)
        ok = ~np.isnan(p[:, 0])
        frame_data[t['i']] = (ok, rest[ok], p[ok])
    sdist = SurfaceDistance(faces, len(verts))
    skipped = set()

    def evaluate(method, params, deform):
        """deform(i) -> deformed template vertices at frame i."""
        if s.fixed is not None and method not in REFERENCE_METHODS:
            if method not in s.fixed:
                if method not in skipped:
                    print(f'  [warning] {method} missing from the fixed parameters, skipped')
                    skipped.add(method)
                return
            if params != s.fixed[method]:
                return
        t0 = time.perf_counter()
        cache = {}
        for t in targets:
            i = t['i']
            if i not in cache:
                cache[i] = deform(i)
            recon = cache[i]
            for region, pts in t['regions'].items():
                if len(pts) < 10:
                    continue
                d_fwd = sdist.distances(pts, recon)
                m = metrics_from_distances(d_fwd)
                fp = footprint_mask(recon, pts, s.footprint_mm)
                d_rev = t['dist'].distances(recon[fp], t['verts']) if fp.any() else np.empty(0)
                m.update(two_way_metrics(d_fwd, d_rev))
                write_row(dict(
                    subject=subject, view=view, method=method, params=params,
                    target=t['name'], region=region,
                    frame_t=round(float(all_frames[i]['t']), 4),
                    phase=round(float(phases[i]), 3), reach=round(float(phase_max), 3),
                    reached=int(t['reached']),
                    target_phase='' if t['phase'] is None else round(float(t['phase']), 3),
                    **{k: round(v, 4) for k, v in m.items()},
                    seconds=round(time.perf_counter() - t0, 2)))

    # Reference methods (no parameter): no deformation, rigid and affine least squares.
    evaluate('none', '-', lambda i: verts)

    def rigid(i):
        _, rp, tp = frame_data[i]
        cr, ct = rp.mean(0), tp.mean(0)
        U, _, Vt = np.linalg.svd((rp - cr).T @ (tp - ct))
        R = (U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt).T
        return (verts - cr) @ R.T + ct
    evaluate('rigid', '-', rigid)

    def affine(i):
        _, rp, tp = frame_data[i]
        A, *_ = np.linalg.lstsq(np.hstack([np.ones((len(rp), 1)), rp]), tp - rp, rcond=None)
        return verts + np.hstack([np.ones((len(verts), 1)), verts]) @ A
    evaluate('affine', '-', affine)

    def point_deformer(i, **kw):
        _, rp, tp = frame_data[i]
        return PointDeformer(rp, tp, **kw).deform(verts)

    for sigma, reg in itertools.product(s.rbf_sigmas_mm, s.regs):
        label = f'sigma={sigma:g}mm;reg={reg:g}'
        for method, use_affine in (('rbf', False), ('rbf_affine', True)):
            if method in s.methods:
                evaluate(method, label, lambda i, a=use_affine: point_deformer(
                    i, sigmas=sigma, affine=a,
                    noise=effective_reg(reg, sigma, sigma, len(frame_data[i][1]))))
    if 'tps' in s.methods:
        for reg in s.regs:
            evaluate('tps', f'reg={reg:g}', lambda i: point_deformer(
                i, kind='tps', noise=reg * float(np.linalg.norm(
                    frame_data[i][1].max(0) - frame_data[i][1].min(0)))))
    if 'biharm' in s.methods:
        bih = BiharmonicDeformer(verts, faces)
        bf, bw, _ = closest_barycentric(rest, verts, faces)
        for weight in s.biharm_weights:
            disp = bih.displacement_fn(bf, bw, weight)
            evaluate('biharm', f'weight={weight:g}',
                     lambda i: verts + disp(frame_data[i][0], frame_data[i][2] - frame_data[i][1]))
