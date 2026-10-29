"""Surface distances between meshes and the derived metrics (NSD, ASD, HD95)."""

import numpy as np
from scipy.spatial import cKDTree


def _vertex_faces(n_verts, faces):
    """Faces incident to each vertex, as a (V, max degree) array padded with -1."""
    counts = np.bincount(faces.ravel(), minlength=n_verts)
    deg = int(counts.max()) if len(counts) else 0
    vf = np.full((n_verts, max(deg, 1)), -1, dtype=np.int64)
    fill = np.zeros(n_verts, dtype=np.int64)
    for fi, f in enumerate(faces):
        for v in f:
            vf[v, fill[v]] = fi
            fill[v] += 1
    return vf


class SurfaceDistance:
    """Point-to-surface distances for a fixed mesh topology (the faces do not change
    from one frame to the next, only the vertex positions)."""

    def __init__(self, faces, n_verts):
        self.faces = np.asarray(faces, dtype=np.int64)
        self.vf = _vertex_faces(n_verts, self.faces)

    def distances(self, points, verts, k=6):
        """Distance of each point to the surface (verts, self.faces)."""
        points = np.asarray(points, dtype=float)
        verts = np.asarray(verts, dtype=float)
        _, nn = cKDTree(verts).query(points, k=k)
        cand = self.vf[nn].reshape(len(points), -1)                  # (P, k*deg)
        valid = cand >= 0
        cf = np.where(valid, cand, 0)
        tri = verts[self.faces[cf]]                                   # (P, C, 3, 3)
        a, b, c = tri[:, :, 0], tri[:, :, 1], tri[:, :, 2]
        p = points[:, None, :]
        e0, e1, e2 = b - a, c - a, p - a
        d00 = (e0 * e0).sum(-1)
        d01 = (e0 * e1).sum(-1)
        d11 = (e1 * e1).sum(-1)
        d20 = (e2 * e0).sum(-1)
        d21 = (e2 * e1).sum(-1)
        den = np.where(np.abs(d00 * d11 - d01 * d01) < 1e-18, 1e-18, d00 * d11 - d01 * d01)
        v = (d11 * d20 - d01 * d21) / den
        w = (d00 * d21 - d01 * d20) / den
        bw = np.stack([1 - v - w, v, w], axis=-1)
        bw = np.clip(bw, 0, None)
        bw /= np.maximum(bw.sum(-1, keepdims=True), 1e-12)
        q = (tri * bw[..., None]).sum(-2)                             # closest point of the triangle
        d = np.linalg.norm(q - p, axis=-1)
        d = np.where(valid, d, np.inf)
        # Point outside all candidate triangles: the clamped projection above gives a
        # point of the triangle (an upper bound); the nearest vertex is another one.
        return np.minimum(d.min(axis=1), np.linalg.norm(verts[nn[:, 0]] - points, axis=1))


def metrics_from_distances(d, taus=(2.0, 3.0, 5.0)):
    """NSD@tau (fraction of points within tau mm), ASD (mean distance), p90, HD95."""
    out = {f'nsd{t:g}': float(np.mean(d <= t)) for t in taus}
    out.update(asd=float(d.mean()), p90=float(np.percentile(d, 90)),
               hd95=float(np.percentile(d, 95)))
    return out


def footprint_mask(recon_verts, target_pts, radius=3.0, axes=(0, 1)):
    """Vertices of the reconstruction inside the footprint of the reference seen from
    above (R-A plane by default): within `radius` mm, in this plane, of a reference point.
    The borders of the template extending beyond the (smaller) reference are excluded."""
    ax = list(axes)
    d, _ = cKDTree(np.asarray(target_pts)[:, ax]).query(np.asarray(recon_verts)[:, ax])
    return d <= radius


TWO_WAY_KEYS = ([f'r_{k}' for k in ('nsd2', 'nsd3', 'nsd5', 'asd', 'p90', 'hd95')]
                + [f's_{k}' for k in ('nsd2', 'nsd3', 'nsd5', 'asd', 'p90', 'hd95')])


def two_way_metrics(d_fwd, d_rev, taus=(2.0, 3.0, 5.0)):
    """Metrics of the reverse direction (reconstruction -> reference, prefix r_) and
    symmetric ones (prefix s_): symmetric NSD and ASD over the points of both surfaces
    (s_asd = ASSD), symmetric p90 and HD95 = max of both directions."""
    if len(d_rev) == 0:
        return {k: float('nan') for k in TWO_WAY_KEYS}
    out = {f'r_{k}': v for k, v in metrics_from_distances(d_rev, taus).items()}
    both = np.concatenate([d_fwd, d_rev])
    for t in taus:
        out[f's_nsd{t:g}'] = float(np.mean(both <= t))
    out['s_asd'] = float(both.mean())
    out['s_p90'] = float(max(np.percentile(d_fwd, 90), np.percentile(d_rev, 90)))
    out['s_hd95'] = float(max(np.percentile(d_fwd, 95), np.percentile(d_rev, 95)))
    return out
