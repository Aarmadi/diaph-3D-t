"""Centreline (medial line) of a 2D diaphragm segmentation, where query points are placed."""

from collections import deque

import numpy as np
from scipy.ndimage import distance_transform_edt, label as _label_cc
from skimage.morphology import skeletonize


def _longest_path_order(coords):
    """Order the pixels (N, 2) of one 8-connected skeleton component along its longest path.

    Double breadth-first search: the farthest pixel from an arbitrary start is one end of
    the graph diameter, the farthest pixel from it is the other end."""
    n = len(coords)
    if n <= 1:
        return coords
    index = {(int(r), int(c)): i for i, (r, c) in enumerate(coords)}
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    adj = [[j for dr, dc in offsets if (j := index.get((int(r) + dr, int(c) + dc))) is not None]
           for r, c in coords]

    def farthest(start):
        dist, parent, far = {start: 0}, {start: -1}, start
        q = deque([start])
        while q:
            u = q.popleft()
            if dist[u] > dist[far]:
                far = u
            for v in adj[u]:
                if v not in dist:
                    dist[v], parent[v] = dist[u] + 1, u
                    q.append(v)
        return far, parent

    a, _ = farthest(0)
    b, parent = farthest(a)
    path, node = [], b
    while node != -1:
        path.append(node)
        node = parent[node]
    return coords[path[::-1]]


def medial_line_points(seg_hw):
    """Ordered pixels (row, col) of the medial line of the largest skeleton component,
    sorted from left to right. Raises ValueError if the line is empty or too short."""
    mask = seg_hw > 0.5
    skel = skeletonize(mask) if mask.any() else None
    if skel is None or not skel.any():
        raise ValueError('Empty medial line on the anchor frame: check the segmentation.')
    labeled, n = _label_cc(skel, structure=np.ones((3, 3)))
    components = [_longest_path_order(np.argwhere(labeled == k)).astype(np.float64)
                  for k in range(1, n + 1) if (labeled == k).sum() >= 2]
    if not components:
        raise ValueError('Empty medial line on the anchor frame: check the segmentation.')
    pts = max(components, key=len)
    if len(pts) < 4:
        raise ValueError('Medial line too short on the anchor frame: check the segmentation.')
    return pts[np.argsort(pts[:, 1])].astype(float)


def contour_points(seg_hw, n=50):
    """n points regularly sampled on the border of a 2D mask (for display), or None."""
    m = seg_hw > 0.5
    if not m.any():
        return None
    border = np.argwhere(m & ~(distance_transform_edt(m) > 1)).astype(float)
    if len(border) == 0:
        return None
    return border[np.round(np.linspace(0, len(border) - 1, n)).astype(int)]
