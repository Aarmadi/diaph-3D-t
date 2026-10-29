"""Temporal assembly of the control points of all planes of a view.

The planes of a view are acquired in turn: plane 0, plane 1, ..., plane P-1, then plane 0
again. Frame t of plane p is therefore acquired at time t + p / P (in frames of a plane),
assuming a uniform timing. Each control point is observed only at the instants of its
own plane; it is linearly interpolated in time at the instants of the other planes."""

import numpy as np


def plane_time_offset(plane, planes_per_round=3):
    """Acquisition delay of a plane, in frames (plane = file prefix 0, 1, 2, ...)."""
    return int(plane) / planes_per_round


def frames_from_control_points(control_points):
    """[{'t', 'points'}] sorted by time; points[n] is the RAS position of control point
    n at time t, or None if it is not observed then."""
    all_t = sorted({t for cp in control_points for t in cp['by_t']})
    frames = []
    for t in all_t:
        row = [None] * len(control_points)
        for n, cp in enumerate(control_points):
            pos = cp['by_t'].get(t)
            if pos is not None:
                row[n] = [round(float(c), 3) for c in pos]
        frames.append({'t': round(float(t), 6), 'points': row})
    return frames


def attach_phi(frames, plane_phis, planes_per_round=3):
    """Add the respiratory level of each frame (that of the plane it comes from), in
    place. plane_phis: {plane (str): list of phi per frame}."""
    for fr in frames:
        t = fr['t']
        plane = int(round((t % 1) * planes_per_round))
        local_t = int(round(t - plane / planes_per_round))
        phis = plane_phis.get(str(plane))
        phi = phis[local_t] if phis is not None and 0 <= local_t < len(phis) else None
        fr['phi_mean'] = float(phi) if phi is not None else None
        fr['phi_by_plane'] = {plane: float(phi)} if phi is not None else {}
        fr['plane_missing'] = []
        fr['phi_divergent'] = False
        fr['origin_plane'] = plane
    return frames


def fill_missing(frames):
    """Fill the missing positions of each control point by linear interpolation in time
    between its two nearest observations (held constant before the first and after the
    last). Returns a copy."""
    n = len(frames)
    out = [dict(fr, points=list(fr['points'])) for fr in frames]
    if n == 0:
        return out
    ts = [fr['t'] for fr in frames]
    for j in range(len(frames[0]['points'])):
        known = [i for i in range(n) if frames[i]['points'][j] is not None]
        if not known:
            continue
        for i in range(n):
            if out[i]['points'][j] is not None:
                continue
            k = int(np.searchsorted(known, i))
            prev_i = known[k - 1] if k > 0 else None
            next_i = known[k] if k < len(known) else None
            if prev_i is not None and next_i is not None:
                frac = (ts[i] - ts[prev_i]) / (ts[next_i] - ts[prev_i])
                p0, p1 = frames[prev_i]['points'][j], frames[next_i]['points'][j]
                out[i]['points'][j] = [round(p0[c] + frac * (p1[c] - p0[c]), 3) for c in range(3)]
            else:
                src = prev_i if prev_i is not None else next_i
                out[i]['points'][j] = list(frames[src]['points'][j])
    return out


def interpolate_frames(frames, n_interp, step):
    """Insert n_interp linearly interpolated frames between consecutive frames that are
    exactly `step` apart (never across a gap). Interpolated frames are flagged with
    'interpolated': True. Returns a new list."""
    out = []
    for i, fr in enumerate(frames):
        out.append(dict(fr, interpolated=False))
        if n_interp <= 0 or i + 1 >= len(frames):
            continue
        nxt = frames[i + 1]
        if abs((nxt['t'] - fr['t']) - step) > 1e-6:
            continue
        for k in range(1, n_interp + 1):
            frac = k / (n_interp + 1)
            points = [[round(p0[c] + frac * (p1[c] - p0[c]), 3) for c in range(3)]
                      if p0 is not None and p1 is not None else None
                      for p0, p1 in zip(fr['points'], nxt['points'])]
            phi0, phi1 = fr.get('phi_mean'), nxt.get('phi_mean')
            phi = (round(phi0 + frac * (phi1 - phi0), 4)
                   if phi0 is not None and phi1 is not None else None)
            out.append(dict(t=round(fr['t'] + frac * step, 6), points=points, phi_mean=phi,
                            phi_by_plane={}, plane_missing=[], phi_divergent=False,
                            interpolated=True))
    return out
