"""Deformation of the template from the control point displacements.

Two families:
- PointDeformer: displacement field interpolated in 3D from the control points, with a
  Gaussian RBF kernel (with or without a global affine part) or the 3D thin-plate spline
  kernel. The regularisation (noise) relaxes the exact interpolation of the points.
- BiharmonicDeformer: displacement propagated along the template surface by minimising
  the biharmonic energy (cotangent Laplacian), after removing a global affine fit."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree


def _kernel(a, b, sigmas, amps, chunk=4000, kind='gauss'):
    """Kernel matrix (len(a), len(b)).

    kind='gauss': sum of Gaussians sum_k amps_k exp(-|a-b|^2 / 2 sigma_k^2).
    kind='tps': 3D thin-plate spline phi(r) = -r (biharmonic kernel in 3D, conditionally
    definite, hence the mandatory affine part; Bookstein's sign convention). No scale."""
    out = np.empty((len(a), len(b)))
    for s in range(0, len(a), chunk):
        d2 = ((a[s:s + chunk, None, :] - b[None, :, :]) ** 2).sum(-1)
        if kind == 'tps':
            out[s:s + chunk] = -np.sqrt(d2)
        else:
            out[s:s + chunk] = sum(am * np.exp(-d2 / (2 * sg * sg))
                                   for sg, am in zip(sigmas, amps))
    return out


def _affine_basis(pts, center, scale):
    """[1, x', y', z'] with x' = (x - center) / scale (for conditioning)."""
    return np.hstack([np.ones((len(pts), 1)), (pts - center) / scale])


class PointDeformer:
    """Displacement field fitted on point pairs (rest_pts -> target_pts).

    sigmas/amps: kernel scales and weights (one scale = standard Gaussian RBF).
    noise: added to the kernel diagonal (regularisation; 0 = exact interpolation).
    affine: adds a global affine part (otherwise the displacement decays to 0 far from
    the points). kind: 'gauss' or 'tps' (the latter always has the affine part).
    """

    def __init__(self, rest_pts, target_pts, sigmas=1.0, amps=None, noise=1e-2, affine=True,
                 kind='gauss'):
        self.rest = np.asarray(rest_pts, dtype=float)
        self.kind = kind
        if kind == 'tps':
            affine = True   # required by a conditionally definite kernel
        self.sigmas = list(np.atleast_1d(sigmas))
        self.amps = list(amps) if amps is not None else [1.0] * len(self.sigmas)
        n = len(self.rest)
        self.affine = affine and n >= 4
        self.center = self.rest.mean(axis=0)
        self.scale = max(float(np.linalg.norm(self.rest.max(0) - self.rest.min(0))), 1e-9)
        K = _kernel(self.rest, self.rest, self.sigmas, self.amps, kind=self.kind)
        K[np.diag_indices(n)] += noise
        if self.affine:
            P = _affine_basis(self.rest, self.center, self.scale)
            # -eps on the zero block: with coplanar points (a single plane) the affine part
            # is undetermined along the normal, without making the system singular.
            M = np.block([[K, P], [P.T, -1e-8 * np.eye(4)]])
        else:
            M = K
        self.M = M
        rhs = np.zeros((len(M), 3))
        rhs[:n] = np.asarray(target_pts, dtype=float) - self.rest
        self.coef = np.linalg.solve(M, rhs)

    def _features(self, x):
        kx = _kernel(x, self.rest, self.sigmas, self.amps, kind=self.kind)
        if self.affine:
            return np.hstack([kx, _affine_basis(x, self.center, self.scale)])
        return kx

    def displacement(self, x, chunk=20000):
        x = np.asarray(x, dtype=float)
        out = np.empty((len(x), 3))
        for s in range(0, len(x), chunk):
            out[s:s + chunk] = self._features(x[s:s + chunk]) @ self.coef
        return out

    def deform(self, x):
        return np.asarray(x, dtype=float) + self.displacement(x)


def _cotangent_edges(verts, faces, w_max=5.0):
    """Undirected edges (I, J, w) with cotangent weights w_ij = (cot a + cot b) / 2,
    clipped to [0, w_max] (degenerate marching-cubes triangles would give negative or
    infinite cotangents)."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]

    def cot(a, b):
        cr = np.linalg.norm(np.cross(a, b), axis=1)
        return (a * b).sum(axis=1) / np.maximum(cr, 1e-12)

    c0 = cot(v1 - v0, v2 - v0)    # angle at v0, opposite to edge (1, 2)
    c1 = cot(v2 - v1, v0 - v1)    # opposite to (2, 0)
    c2 = cot(v0 - v2, v1 - v2)    # opposite to (0, 1)
    I = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    J = np.concatenate([faces[:, 2], faces[:, 0], faces[:, 1]])
    W = 0.5 * np.concatenate([c0, c1, c2])
    lo, hi = np.minimum(I, J), np.maximum(I, J)
    n = len(verts)
    Wm = sp.coo_matrix((W, (lo, hi)), shape=(n, n)).tocsr()   # sum over the 2 adjacent triangles
    Wm.sum_duplicates()
    Wm = Wm.tocoo()
    return Wm.row, Wm.col, np.clip(Wm.data, 0.0, w_max)


class BiharmonicDeformer:
    """Biharmonic deformation of a surface mesh; Q = L M^-1 L is precomputed once."""

    def __init__(self, verts, faces):
        self.verts = np.asarray(verts, dtype=float)
        self.faces = np.asarray(faces, dtype=int)
        n = len(self.verts)
        I, J, W = _cotangent_edges(self.verts, self.faces)
        Wm = sp.coo_matrix((W, (I, J)), shape=(n, n))
        Wm = (Wm + Wm.T).tocsr()
        L = (sp.diags(np.asarray(Wm.sum(axis=1)).ravel()) - Wm).tocsr()
        v0, v1, v2 = (self.verts[self.faces[:, k]] for k in range(3))
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        mass = np.zeros(n)
        for k in range(3):
            np.add.at(mass, self.faces[:, k], area / 3)
        mass = np.maximum(mass, 1e-6 * max(mass.mean(), 1e-12))
        self.Q = (L @ sp.diags(1.0 / mass) @ L).tocsc()
        self.q_scale = max(float(self.Q.diagonal().mean()), 1e-12)

    def operator(self, bary_faces, bary_weights, weight=1e2, affine=True):
        """Dense G (V, K) such that vertex displacements = G @ point displacements.

        The points are given by barycentric coordinates on the mesh. weight is relative to
        the scale of Q (large = strong attachment). A small epsilon keeps a component
        without any point at zero displacement (otherwise the system is singular).

        affine=True: a global affine transform is first fitted (least squares) on the
        points and only the residual is propagated by the biharmonic operator. Without it,
        the natural boundary conditions on an open surface pull the displacement towards a
        constant near the border and the periphery of the dome is poorly extrapolated."""
        n, k = len(self.verts), len(bary_faces)
        bary_faces = np.asarray(bary_faces)
        bary_weights = np.asarray(bary_weights, dtype=float)
        C = sp.csr_matrix((bary_weights.ravel(),
                           (np.repeat(np.arange(k), 3), bary_faces.ravel())), shape=(k, n))
        w = weight * self.q_scale
        A = (self.Q + w * (C.T @ C) + 1e-8 * self.q_scale * sp.identity(n)).tocsc()
        lu = spla.splu(A)
        G = w * lu.solve(np.asarray(C.T.todense()))
        if not affine or k < 4:
            return G
        pts = (self.verts[bary_faces] * bary_weights[:, :, None]).sum(axis=1)
        center = pts.mean(axis=0)
        scale = max(float(np.linalg.norm(pts.max(0) - pts.min(0))), 1e-9)
        Pa = np.hstack([np.ones((k, 1)), (pts - center) / scale])            # (K, 4)
        Xa = np.hstack([np.ones((n, 1)), (self.verts - center) / scale])     # (V, 4)
        Pinv = np.linalg.pinv(Pa, rcond=1e-6)                                # (4, K)
        # u = Xa A + G (d - Pa A), A = Pinv d
        return Xa @ Pinv + G @ (np.eye(k) - Pa @ Pinv)

    def displacement_fn(self, bary_faces, bary_weights, weight=1e2, affine=True):
        """Like operator(), for any subset of the points with a single factorisation.

        Returns f(ok, d) -> vertex displacements when only the points `ok` are observed
        (d: (ok.sum(), 3)); identical to operator(bary_faces[ok], bary_weights[ok]) @ d.
        Removing the missing points is a low-rank update handled by the Woodbury formula
        from X = A^-1 C^T (cost ~ V K m instead of a new V x V factorisation)."""
        n, k = len(self.verts), len(bary_faces)
        bary_faces = np.asarray(bary_faces)
        bary_weights = np.asarray(bary_weights, dtype=float)
        C = sp.csr_matrix((bary_weights.ravel(),
                           (np.repeat(np.arange(k), 3), bary_faces.ravel())), shape=(k, n))
        w = weight * self.q_scale
        A = (self.Q + w * (C.T @ C) + 1e-8 * self.q_scale * sp.identity(n)).tocsc()
        X = spla.splu(A).solve(np.asarray(C.T.todense()))       # (V, K)
        CX = np.asarray(C @ X)                                  # (K, K)
        pts_all = (self.verts[bary_faces] * bary_weights[:, :, None]).sum(axis=1)

        def f(ok, d):
            ok = np.asarray(ok, dtype=bool)
            d = np.asarray(d, dtype=float)
            if not ok.any():
                return np.zeros((n, 3))
            m = ~ok

            def solve_part(r):
                # w * A_ok^-1 C_ok^T r  (A_ok: matrix without the missing points)
                y = X[:, ok] @ r
                if m.any():
                    S = np.eye(int(m.sum())) / w - CX[np.ix_(m, m)]
                    y = y + X[:, m] @ np.linalg.solve(S, CX[np.ix_(m, ok)] @ r)
                return w * y
            if not affine or ok.sum() < 4:
                return solve_part(d)
            pts = pts_all[ok]
            center = pts.mean(axis=0)
            scale = max(float(np.linalg.norm(pts.max(0) - pts.min(0))), 1e-9)
            Pa = np.hstack([np.ones((len(pts), 1)), (pts - center) / scale])
            Xa = np.hstack([np.ones((n, 1)), (self.verts - center) / scale])
            Aff = np.linalg.pinv(Pa, rcond=1e-6) @ d
            return Xa @ Aff + solve_part(d - Pa @ Aff)
        return f


def closest_barycentric(points, verts, faces, k=12):
    """Attach each point to the mesh: closest face among those incident to its k nearest
    vertices, with the barycentric weights of the projection (clamped to the triangle).
    Returns (face vertex indices (P, 3), weights (P, 3), distance (P,))."""
    points = np.asarray(points, dtype=float)
    vert_faces = [[] for _ in range(len(verts))]
    for fi, f in enumerate(faces):
        for v in f:
            vert_faces[v].append(fi)
    _, nn = cKDTree(verts).query(points, k=min(k, len(verts)))
    out_f = np.zeros((len(points), 3), dtype=int)
    out_w = np.zeros((len(points), 3))
    out_d = np.full(len(points), np.inf)
    for i, p in enumerate(points):
        cand = {fi for v in np.atleast_1d(nn[i]) for fi in vert_faces[v]}
        for fi in cand:
            a, b, c = verts[faces[fi]]
            e0, e1, e2 = b - a, c - a, p - a
            d00, d01, d11 = e0 @ e0, e0 @ e1, e1 @ e1
            d20, d21 = e2 @ e0, e2 @ e1
            den = d00 * d11 - d01 * d01
            if den <= 1e-18:
                continue
            v_ = (d11 * d20 - d01 * d21) / den
            w_ = (d00 * d21 - d01 * d20) / den
            bw = np.clip(np.array([1 - v_ - w_, v_, w_]), 0, None)
            bw /= max(bw.sum(), 1e-12)
            q = bw @ verts[faces[fi]]
            dist = float(np.linalg.norm(q - p))
            if dist < out_d[i]:
                out_d[i], out_f[i], out_w[i] = dist, faces[fi], bw
    return out_f, out_w, out_d
