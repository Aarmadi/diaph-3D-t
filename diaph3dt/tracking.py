"""2D+t point tracking of the diaphragm with CoTracker3.

For each plane, query points are placed on the medial line of the segmentation at several
anchor frames spread over the sequence. Each anchor is tracked independently over the
whole sequence and the consensus (medoid) tracking is kept. Points whose motion is
inconsistent with their neighbours (islands) are then excluded."""

import json
import os

import numpy as np
from scipy.ndimage import label as _label_cc
from scipy.spatial import cKDTree

from .centerline import contour_points, medial_line_points
from .imaging import TARGET_H, TARGET_W
from .io import load_plane_data

MIN_COMPONENT_AREA_FRAC = 0.15   # a 2nd component is kept if >= 15 % of the largest one


def _subsample_indices(n_total, n_want):
    """n_want indices evenly spread over [0, n_total) (fewer if rounding creates duplicates)."""
    n_total = int(n_total)
    n_want = max(1, min(int(n_want), n_total))
    if n_want == 1:
        return [n_total // 2]
    idx = np.round(np.linspace(0, n_total - 1, n_want)).astype(int)
    return sorted(set(int(i) for i in idx))


def _point_color(idx, n):
    """Fixed HSV colour of a point, shared by all planes of a view."""
    import colorsys
    h = (idx / max(n, 1)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
    return '#%02x%02x%02x' % (round(r * 255), round(g * 255), round(b * 255))


def _component_submasks(seg_hw, min_area_frac=MIN_COMPONENT_AREA_FRAC, max_components=2):
    """Split a 2D mask into 1 or 2 connected components (two hemidiaphragms in coronal
    view). The second one is kept only if its area is >= min_area_frac of the largest.
    Returns [(submask, area)] by decreasing area."""
    mask = seg_hw > 0.5
    if not mask.any():
        return []
    labeled, n_cc = _label_cc(mask, structure=np.ones((3, 3)))
    sizes = sorted(((k, int((labeled == k).sum())) for k in range(1, n_cc + 1)),
                   key=lambda x: x[1], reverse=True)[:max_components]
    if not sizes:
        return []
    kept = [(k, s) for k, s in sizes if s >= min_area_frac * sizes[0][1]]
    return [(np.where(labeled == k, seg_hw, 0.0).astype(np.float32), s) for k, s in kept]


def _medial_line_lengths(segs, frames):
    """[(t, length)] of the medial line (summed over components) of each usable frame."""
    out = []
    for t in frames:
        lengths = []
        for submask, _ in _component_submasks(segs[t]):
            try:
                pts = medial_line_points(submask)
            except ValueError:
                continue
            lengths.append(float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
        if lengths:
            out.append((t, sum(lengths)))
    return out


def select_anchor_frames(segs, frames, n_candidates=15, min_length_ratio=0.5):
    """Anchor frames evenly spread over the sequence.

    Frames whose medial line is shorter than min_length_ratio x the median length
    (truncated or missing segmentation) are discarded first. Returns [(t, length)]."""
    lengths = _medial_line_lengths(segs, frames)
    if not lengths:
        raise ValueError('No frame has a usable medial line.')
    med_len = float(np.median([l for _, l in lengths]))
    usable = sorted((t, l) for t, l in lengths if l >= min_length_ratio * med_len)
    if len(usable) <= n_candidates:
        return usable
    pick = np.linspace(0, len(usable) - 1, n_candidates).round().astype(int)
    return [usable[k] for k in sorted(set(pick))]


def _detect_edge_islands(tracks_rc, ratio_threshold=20.0, min_base_dist=1.0):
    """Flag points whose motion is inconsistent with their neighbours along the chain.

    The distances between adjacent points are normalised at each frame by their median
    (which removes the respiratory stretching); the chain is cut between two points when
    the max/min ratio of their normalised distance over time exceeds ratio_threshold, and
    the longest segment is kept. Returns island_mask (N,) bool, True = excluded."""
    n = tracks_rc.shape[1]
    if n < 2:
        return np.zeros(n, dtype=bool)
    d = np.maximum(np.linalg.norm(np.diff(tracks_rc, axis=1), axis=2), min_base_dist)  # (nt, N-1)
    normalized = d / np.median(d, axis=1, keepdims=True)
    ratio = normalized.max(axis=0) / normalized.min(axis=0)
    cuts = np.where(ratio > ratio_threshold)[0]
    bounds = [0] + [int(c) + 1 for c in cuts] + [n]
    start, end = max(zip(bounds[:-1], bounds[1:]), key=lambda s: s[1] - s[0])
    island_mask = np.ones(n, dtype=bool)
    island_mask[start:end] = False
    return island_mask


def _detect_islands_grouped(tracks_rc, group_sizes, ratio_threshold=20.0):
    """_detect_edge_islands applied to each group separately (the gap between the two
    hemidiaphragms is anatomical, not a tracking error)."""
    masks, offset = [], 0
    for size in group_sizes:
        masks.append(_detect_edge_islands(tracks_rc[:, offset:offset + size], ratio_threshold))
        offset += size
    return np.concatenate(masks) if masks else np.zeros(tracks_rc.shape[1], dtype=bool)


def _flag_static_groups(tracks_rc, group_sizes, min_motion_ratio=0.15):
    """Flag a whole group as spurious when its median motion amplitude is below
    min_motion_ratio x that of the other group (a static artefact picked as a component
    at the anchor frame). Only applies with at least two groups."""
    n = tracks_rc.shape[1]
    mask = np.zeros(n, dtype=bool)
    if len(group_sizes) < 2:
        return mask
    motion = np.linalg.norm(tracks_rc - tracks_rc[0:1], axis=2).max(axis=0)   # (N,)
    bounds = np.cumsum([0] + list(group_sizes))
    group_motion = [float(np.median(motion[a:b])) for a, b in zip(bounds[:-1], bounds[1:])]
    max_motion = max(group_motion)
    if max_motion <= 0:
        return mask
    for a, b, gm in zip(bounds[:-1], bounds[1:], group_motion):
        if gm < min_motion_ratio * max_motion:
            mask[a:b] = True
    return mask


def _candidate_shape_distance(cand_a, cand_b, nt):
    """Shape distance between two trackings (their points do not correspond one to one):
    mean symmetric chamfer distance at each frame, median over the sequence."""
    per_frame = []
    for t in range(nt):
        pa = cand_a['tracks_rc'][t][cand_a['visible'][t]]
        pb = cand_b['tracks_rc'][t][cand_b['visible'][t]]
        if len(pa) == 0 or len(pb) == 0:
            continue
        d_ab, _ = cKDTree(pb).query(pa)
        d_ba, _ = cKDTree(pa).query(pb)
        per_frame.append(float(np.concatenate([d_ab, d_ba]).mean()))
    return float(np.median(per_frame)) if per_frame else None


def medoid_consensus(dist, outlier_factor=3.0):
    """Consensus tracking among n candidates from their (n, n) distance matrix.

    1) the most central tracking m0 (smallest median distance to the others);
    2) trackings farther than outlier_factor x the median distance to m0 are discarded;
    3) among the remaining ones, the one with the smallest mean distance to the others.
    Returns (index, kept indices)."""
    n = len(dist)
    if n == 1:
        return 0, [0]
    off = ~np.eye(n, dtype=bool)
    m0 = int(np.argmin([np.median(dist[i][off[i]]) for i in range(n)]))
    ref = np.median(dist[m0][off[m0]])
    kept = [i for i in range(n) if i == m0 or dist[i, m0] <= outlier_factor * max(ref, 1e-6)]
    if len(kept) == 1:
        return m0, kept
    best = min(kept, key=lambda i: np.mean([dist[i, j] for j in kept if j != i]))
    return best, kept


def _extract_query_points(seg_hw, n_points, min_area_frac=MIN_COMPONENT_AREA_FRAC):
    """Query points on the medial line, one group per connected component (1 in sagittal
    view, 1-2 in coronal view), shared in proportion to the areas.
    Returns (query_rc (N, 2), group_sizes)."""
    submasks = _component_submasks(seg_hw, min_area_frac=min_area_frac)
    if not submasks:
        raise ValueError('Empty segmentation at the anchor frame.')
    total_area = sum(s for _, s in submasks)
    pieces, group_sizes = [], []
    for submask, area in submasks:
        try:
            medial = medial_line_points(submask)
        except ValueError:
            continue
        n_i = n_points if len(submasks) == 1 else max(2, round(n_points * area / total_area))
        pts = medial[_subsample_indices(len(medial), n_i)]
        pieces.append(pts)
        group_sizes.append(len(pts))
    if not pieces:
        raise ValueError('No usable medial line at the anchor frame.')
    return np.concatenate(pieces, axis=0), group_sizes


def load_cotracker_model(model_name='cotracker3_offline', device='cuda'):
    """Load CoTracker3 through torch.hub (cached locally after the first call)."""
    import torch
    device = device if torch.cuda.is_available() else 'cpu'
    # Load from the local cache when present: parallel runs re-downloading the hub repo
    # at the same time would otherwise delete each other's copy.
    local = os.path.join(torch.hub.get_dir(), 'facebookresearch_co-tracker_main')
    if os.path.isfile(os.path.join(local, 'hubconf.py')):
        return torch.hub.load(local, model_name, source='local').to(device), device
    return torch.hub.load('facebookresearch/co-tracker', model_name).to(device), device


def _merge_forward_backward(tracks_fwd, vis_fwd, conf_fwd, tracks_bwd, vis_bwd, conf_bwd,
                            query_frame):
    """Frames before the query frame come from the backward pass (same rule as
    CoTrackerPredictor._compute_backward_tracks), the others from the forward pass."""
    mask = np.arange(tracks_fwd.shape[0])[:, None] < np.asarray(query_frame)[None, :]
    tracks, vis, conf = tracks_fwd.copy(), vis_fwd.copy(), conf_fwd.copy()
    tracks[mask], vis[mask], conf[mask] = tracks_bwd[mask], vis_bwd[mask], conf_bwd[mask]
    return tracks, vis, conf


def _rescale(pts_rc, src_hw, dst_hw):
    """(row, col) positions from a src_hw grid to a dst_hw grid (CoTracker convention)."""
    out = np.asarray(pts_rc, dtype=float).copy()
    out[..., 0] *= (dst_hw[0] - 1) / (src_hw[0] - 1)
    out[..., 1] *= (dst_hw[1] - 1) / (src_hw[1] - 1)
    return out


def run_cotracker(imgs, anchor_idx, query_rc, model, device='cuda'):
    """Track query points from frame anchor_idx over the whole sequence (forward and
    backward passes).

    Unlike the public CoTracker API, also returns the per-frame positional confidence of
    each point (probability that the position is within 12 px of the truth), which is
    distinct from the visibility (occlusion). The forward pass of the underlying model is
    called directly for this.

    imgs: nt frames (H, W) in [0, 1]; query_rc: (N, 2) in (row, col).
    Returns (tracks_rc (nt, N, 2), visible (nt, N) bool, confidence (nt, N))."""
    import torch
    import torch.nn.functional as F

    device = device if torch.cuda.is_available() else 'cpu'
    video = np.stack(imgs, axis=0)
    H, W = video.shape[1], video.shape[2]
    video = np.repeat(video[:, None, :, :], 3, axis=1)                  # (nt, 3, H, W)
    video_t = torch.from_numpy(video * 255.0).float()[None].to(device)

    interp_shape = tuple(model.interp_shape)
    B, T, C = video_t.shape[:3]
    video_r = F.interpolate(video_t.reshape(B * T, C, H, W), interp_shape,
                            mode='bilinear', align_corners=True)
    video_r = video_r.reshape(B, T, C, *interp_shape)

    query = _rescale(np.asarray(query_rc, dtype=np.float32), (H, W), interp_shape)
    n = len(query)
    queries = np.zeros((n, 3), dtype=np.float32)
    queries[:, 0] = anchor_idx
    queries[:, 1] = query[:, 1]   # x = column
    queries[:, 2] = query[:, 0]   # y = row
    queries_t = torch.from_numpy(queries)[None].to(device)

    with torch.no_grad():
        tr_f, vis_f, conf_f, _ = model.model.forward(video=video_r, queries=queries_t, iters=6)
        video_inv = torch.flip(video_r, dims=[1])
        queries_inv = queries_t.clone()
        queries_inv[:, :, 0] = video_inv.shape[1] - queries_inv[:, :, 0] - 1
        tr_b, vis_b, conf_b, _ = model.model.forward(video=video_inv, queries=queries_inv, iters=6)

    def np_(x, flip=False):
        a = x[0].cpu().numpy()
        return a[::-1].copy() if flip else a

    tracks_xy, vis, conf = _merge_forward_backward(
        np_(tr_f), np_(vis_f), np_(conf_f), np_(tr_b, True), np_(vis_b, True), np_(conf_b, True),
        np.full(n, anchor_idx))
    tracks_rc = _rescale(np.stack([tracks_xy[..., 1], tracks_xy[..., 0]], axis=-1),
                         interp_shape, (H, W))
    return tracks_rc, vis > 0.9, conf


def track_plane(seg_dir, img_dir, subject, view, plane, model, device='cuda', n_points=50,
                n_candidates=15, min_length_ratio=0.5, outlier_factor=3.0,
                island_ratio_thr=20.0, static_group_min_motion_ratio=0.15, verbose=True,
                tracker=run_cotracker):
    """Track one plane: multi-anchor tracking, medoid consensus, island removal.

    `tracker(imgs, anchor, query_rc, model, device)` -> (tracks_rc, visible, confidence)
    can be replaced for tests. Returns a dict with the plane data (imgs, segs, phis,
    geometry) and the retained tracking (tracks_rc, visible, confidence, island_mask)."""
    pd_ = load_plane_data(seg_dir, img_dir, subject, view, plane)
    imgs, segs, nt = pd_['imgs'], pd_['segs'], pd_['nt']

    anchors = select_anchor_frames(segs, pd_['valid_frames'], n_candidates, min_length_ratio)
    candidates = []
    for anchor, length in anchors:
        query_rc, group_sizes = _extract_query_points(segs[anchor], n_points)
        tracks_rc, visible, confidence = tracker(imgs, anchor, query_rc, model, device)
        candidates.append(dict(anchor=anchor, anchor_length=length, query_rc=query_rc,
                               group_sizes=group_sizes, tracks_rc=tracks_rc,
                               visible=visible, confidence=confidence))
    n = len(candidates)

    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _candidate_shape_distance(candidates[i], candidates[j], nt)
            dist[i, j] = dist[j, i] = np.inf if d is None else d
    best, kept = medoid_consensus(dist, outlier_factor)
    winner = candidates[best]

    island_mask = (_detect_islands_grouped(winner['tracks_rc'], winner['group_sizes'],
                                           island_ratio_thr)
                   | _flag_static_groups(winner['tracks_rc'], winner['group_sizes'],
                                         static_group_min_motion_ratio))
    visible = winner['visible'].copy()
    visible[:, island_mask] = False
    if verbose:
        print(f'  [plane {plane}] {n} anchor(s), {len(kept)}/{n} in agreement, '
              f'retained anchor frame {winner["anchor"]}, '
              f'{int(island_mask.sum())} point(s) excluded (islands)')

    return dict(plane=pd_['plane'], imgs=imgs, segs=segs, nt=nt, phis=pd_['phis'],
                aff_seg=pd_['aff_seg'], ni=pd_['ni'], nj=pd_['nj'],
                anchor=winner['anchor'], anchor_length=winner['anchor_length'],
                tracks_rc=winner['tracks_rc'], visible=visible,
                confidence=winner['confidence'], n_points=len(winner['query_rc']),
                n_groups=len(winner['group_sizes']), island_mask=island_mask,
                n_island_removed=int(island_mask.sum()), n_candidates=n,
                n_agree=len(kept),
                contours=[_contour_or_empty(segs[t]) for t in range(nt)])


def _img_to_b64(arr01):
    """Preprocessed frame in [0, 1] -> base64 grayscale PNG."""
    import base64
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray((np.clip(arr01, 0, 1) * 255).astype(np.uint8), mode='L').save(
        buf, format='PNG', optimize=True)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def _contour_or_empty(seg_hw, n=120):
    """Contour of a mask as a list of [row, col] (empty if there is none)."""
    pts = contour_points(seg_hw, n=n)
    if pts is None:
        return []
    return [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in pts]


CSS_CT = """
body{margin:0;font-family:-apple-system,Segoe UI,sans-serif;background:#0d1117;color:#cdd9e5}
h1{text-align:center;padding:14px 0 3px;font-size:1.25rem;margin:0}
.sub{text-align:center;color:#666;font-size:.8rem;margin:0 16px 10px;max-width:780px;
  margin-left:auto;margin-right:auto}
#panels{display:flex;gap:18px;justify-content:center;flex-wrap:wrap;margin:10px 16px}
.panel{display:flex;flex-direction:column;align-items:center}
.panel h2{font-size:.85rem;color:#58a6ff;margin:0 0 6px}
.anchor-note{font-size:.72rem;color:#8b949e;text-align:center;margin:0 0 6px}
.anchor-note a{color:#58a6ff;text-decoration:none}
.anchor-note a:hover{text-decoration:underline}
.island-note{color:#f0883e;font-weight:600}
.metrics-note{display:flex;gap:14px;justify-content:center;flex-wrap:wrap;
  font-size:.72rem;color:#cdd9e5;margin:0 0 8px;font-family:monospace}
.metrics-note span{background:#161b22;border:1px solid #30363d;border-radius:5px;
  padding:2px 8px}
canvas.ct-cv{background:#111827;border:1px solid #30363d;border-radius:6px;max-width:340px;
  width:100%;height:auto}
.tabs{display:flex;gap:6px;justify-content:center;margin:14px 16px 0;flex-wrap:wrap}
.tab-btn{padding:7px 16px;background:#161b22;border:1px solid #30363d;border-radius:6px 6px 0 0;
  color:#aaa;cursor:pointer;font-size:.85rem}
.tab-btn.active{background:#21262d;color:#58a6ff;border-color:#58a6ff}
.tab-btn:hover{background:#1c2128}
.tab-content{margin:0 16px}
.big-layout{display:flex;gap:18px;justify-content:center;flex-wrap:wrap;margin:10px 0}
canvas.big-cv{background:#111827;border:1px solid #30363d;border-radius:6px;cursor:grab;
  touch-action:none;max-width:640px;width:100%;height:auto}
canvas.big-cv:active{cursor:grabbing}
.side-controls{display:flex;flex-direction:column;gap:8px;min-width:200px;max-width:260px;
  font-size:.8rem}
.side-controls label{display:flex;align-items:center;gap:6px;cursor:pointer}
.ctrl-row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.ctrl-row button{padding:4px 10px;background:#21262d;border:1px solid #444;border-radius:4px;
  color:#ccc;cursor:pointer;font-size:.75rem}
.ctrl-row button:hover{background:#2d333b}
.point-box{display:flex;flex-direction:column;gap:2px;max-height:260px;overflow-y:auto;
  border:1px solid #30363d;border-radius:6px;padding:6px;background:#0d1117}
.pt-item{font-size:.75rem;display:flex;align-items:center;gap:5px}
.zoom-hint{text-align:center;color:#555;font-size:.72rem;margin:2px 0 8px}
.nav{display:flex;gap:14px;justify-content:center;align-items:center;margin:14px 0 4px;
  font-size:.85rem;flex-wrap:wrap}
.nav input[type=range]{width:360px;accent-color:#58a6ff}
.nav button{padding:5px 14px;background:#21262d;border:1px solid #444;border-radius:4px;
  color:#ccc;cursor:pointer}
.nav button:hover{background:#2d333b}
.legend{font-family:monospace;font-size:.75rem;color:#888;text-align:center;margin:6px 16px 26px}
"""


JS_CT = r"""
var DATA = {};
var IMG_CACHE = {};
var curIdx = 0;

var hiddenPoints = [];
var showSeg = [];
var showTraj = [];
var showPrev = [];
var viewZoom = [];
var colorMode = [];

function confColor(c){
  var r = Math.round(255 * (1 - c)), g = Math.round(255 * c);
  return 'rgb(' + r + ',' + g + ',80)';
}

function setColorMode(pIdx, mode){
  colorMode[pIdx] = mode;
  buildPointCheckboxes(pIdx);
  render();
}

function switchTab(tab){
  document.querySelectorAll('.tab-content').forEach(function(el){ el.style.display = 'none'; });
  document.querySelectorAll('.tab-btn').forEach(function(el){ el.classList.remove('active'); });
  if (tab === 'overview') {
    document.getElementById('tab-overview').style.display = '';
    document.getElementById('tabbtn-overview').classList.add('active');
  } else {
    document.getElementById('tab-plane-' + tab).style.display = '';
    document.getElementById('tabbtn-' + tab).classList.add('active');
  }
}

function drawPointMarker(ctx, x, y, col, visible, r, isIsland){
  ctx.beginPath();
  if (isIsland) {
    ctx.setLineDash([2, 2]);
    ctx.arc(x, y, r, 0, 6.283);
    ctx.strokeStyle = '#8b949e'; ctx.lineWidth = 1.4; ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(x - r * 0.6, y - r * 0.6); ctx.lineTo(x + r * 0.6, y + r * 0.6);
    ctx.moveTo(x - r * 0.6, y + r * 0.6); ctx.lineTo(x + r * 0.6, y - r * 0.6);
    ctx.strokeStyle = '#8b949e'; ctx.lineWidth = 1.2; ctx.stroke();
    return;
  }
  ctx.arc(x, y, r, 0, 6.283);
  if (visible) {
    ctx.fillStyle = col; ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,0.55)'; ctx.lineWidth = 1; ctx.stroke();
  } else {
    ctx.strokeStyle = col; ctx.lineWidth = 1.8; ctx.stroke();
  }
}

function renderOverview(){
  DATA.planes.forEach(function(pl, pIdx){
    var cv = document.getElementById('ct-cv-' + pIdx);
    if (!cv) return;
    var ctx = cv.getContext('2d');
    var cw = cv.width, ch = cv.height;
    var sx = cw / DATA.tw, sy = ch / DATA.th;
    ctx.clearRect(0, 0, cw, ch);
    ctx.fillStyle = '#111827'; ctx.fillRect(0, 0, cw, ch);

    var t = Math.min(curIdx, pl.nt - 1);
    var img = IMG_CACHE[pIdx + '_' + t];
    if (img && img.complete) ctx.drawImage(img, 0, 0, cw, ch);

    var pts = pl.tracks[t], vis = pl.visible[t];
    for (var i = 0; i < pts.length; i++) {
      drawPointMarker(ctx, pts[i][1] * sx, pts[i][0] * sy, DATA.colors[i], vis[i], 4.2,
                       pl.islandMask[i]);
    }
  });
}

function renderBigTab(pIdx){
  var pl = DATA.planes[pIdx];
  var cv = document.getElementById('big-cv-' + pIdx);
  if (!cv || !pl) return;
  var ctx = cv.getContext('2d');
  var cw = cv.width, ch = cv.height;
  var sx = cw / DATA.tw, sy = ch / DATA.th;
  var view = viewZoom[pIdx];

  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, cw, ch);
  ctx.fillStyle = '#111827'; ctx.fillRect(0, 0, cw, ch);
  ctx.save();
  ctx.translate(view.panX, view.panY);
  ctx.scale(view.zoom, view.zoom);

  var t = Math.min(curIdx, pl.nt - 1);
  var img = IMG_CACHE[pIdx + '_' + t];
  if (img && img.complete) ctx.drawImage(img, 0, 0, cw, ch);

  if (showSeg[pIdx]) {
    ctx.fillStyle = 'rgba(255,211,61,0.55)';
    (pl.contours[t] || []).forEach(function(p){
      ctx.beginPath(); ctx.arc(p[1] * sx, p[0] * sy, 1.6 / Math.sqrt(view.zoom), 0, 6.283);
      ctx.fill();
    });
  }

  var hidden = hiddenPoints[pIdx];
  var nPts = pl.tracks[0].length;

  if (showTraj[pIdx]) {
    for (var i = 0; i < nPts; i++) {
      if (hidden.has(i)) continue;
      ctx.strokeStyle = DATA.colors[i]; ctx.globalAlpha = 0.5;
      ctx.lineWidth = 1.4 / view.zoom;
      ctx.beginPath();
      for (var tt = 0; tt < pl.nt; tt++) {
        var pt = pl.tracks[tt][i];
        var x = pt[1] * sx, y = pt[0] * sy;
        if (tt === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.globalAlpha = 1.0;
    }
  }

  if (showPrev[pIdx]) {
    for (var i2 = 0; i2 < nPts; i2++) {
      if (hidden.has(i2)) continue;
      for (var tp = 0; tp < t; tp++) {
        var pp = pl.tracks[tp][i2];
        ctx.beginPath();
        ctx.arc(pp[1] * sx, pp[0] * sy, 2.0 / Math.sqrt(view.zoom), 0, 6.283);
        ctx.fillStyle = DATA.colors[i2]; ctx.globalAlpha = 0.35; ctx.fill();
        ctx.globalAlpha = 1.0;
      }
    }
  }

  var pts = pl.tracks[t], vis = pl.visible[t];
  var byConf = (colorMode[pIdx] === 'confidence' && pl.confidence);
  for (var i3 = 0; i3 < pts.length; i3++) {
    if (hidden.has(i3)) continue;
    var col3 = byConf ? confColor(pl.confidence[t][i3]) : DATA.colors[i3];
    drawPointMarker(ctx, pts[i3][1] * sx, pts[i3][0] * sy, col3, vis[i3],
                     6.0 / Math.sqrt(view.zoom), pl.islandMask[i3]);
  }

  ctx.restore();
}

function render(){
  renderOverview();
  DATA.planes.forEach(function(pl, pIdx){
    renderBigTab(pIdx);
    updatePointConfidenceLabels(pIdx);
  });
  var info = document.getElementById('frame-info');
  if (info) info.textContent = 'frame ' + curIdx + ' / ' + (DATA.nt - 1);
}

function seek(v){
  curIdx = parseInt(v, 10);
  render();
}

function stepFrame(delta){
  var s = document.getElementById('frame-slider');
  curIdx = Math.max(0, Math.min(DATA.nt - 1, curIdx + delta));
  s.value = curIdx;
  render();
}

function goToAnchor(pIdx){
  curIdx = Math.min(DATA.planes[pIdx].anchor, DATA.nt - 1);
  var s = document.getElementById('frame-slider');
  if (s) s.value = curIdx;
  render();
}

var playing = false, playTimer = null;
function togglePlay(){
  playing = !playing;
  document.querySelectorAll('.play-btn').forEach(function(b){
    b.textContent = playing ? '⏸ Pause' : '▶ Play';
  });
  if (playing) {
    playTimer = setInterval(function(){
      curIdx = (curIdx + 1) % DATA.nt;
      var s = document.getElementById('frame-slider');
      if (s) s.value = curIdx;
      render();
    }, 150);
  } else if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
  }
}


function pointVisibleRate(pl, i){
  var nVis = 0;
  for (var t = 0; t < pl.nt; t++) { if (pl.visible[t][i]) nVis++; }
  return pl.nt ? nVis / pl.nt : 0;
}

function buildPointCheckboxes(pIdx){
  var pl = DATA.planes[pIdx];
  var n = pl.tracks[0].length;
  var byConf = (colorMode[pIdx] === 'confidence' && pl.confidence);
  var html = '';
  for (var i = 0; i < n; i++) {
    var checked = hiddenPoints[pIdx].has(i) ? '' : 'checked';
    var val, valColor, title;
    if (byConf) {
      var c = pl.confidence[curIdx][i];
      val = Math.round(100 * c); valColor = confColor(c);
      title = 'CoTracker positional confidence at the current frame';
    } else {
      val = Math.round(100 * pointVisibleRate(pl, i));
      valColor = val < 50 ? '#f0883e' : (val < 80 ? '#d2a020' : '#8b949e');
      title = 'Fraction of frames where this point is visible';
    }
    html += '<label class="pt-item" style="color:' + DATA.colors[i] + '">' +
      '<input type="checkbox" ' + checked + ' onchange="togglePoint(' + pIdx + ',' + i +
      ',this.checked)"> #' + i +
      ' <span id="pt-val-' + pIdx + '-' + i + '" style="color:' + valColor + '" title="' +
      title + '">(' + val + '%)</span></label>';
  }
  var box = document.getElementById('point-box-' + pIdx);
  if (box) box.innerHTML = html;
  updatePointCount(pIdx);
}

function updatePointConfidenceLabels(pIdx){
  var pl = DATA.planes[pIdx];
  if (!pl || colorMode[pIdx] !== 'confidence' || !pl.confidence) return;
  var n = pl.tracks[0].length;
  for (var i = 0; i < n; i++) {
    var span = document.getElementById('pt-val-' + pIdx + '-' + i);
    if (!span) continue;
    var c = pl.confidence[curIdx][i];
    span.textContent = '(' + Math.round(100 * c) + '%)';
    span.style.color = confColor(c);
  }
}

function updatePointCount(pIdx){
  var pl = DATA.planes[pIdx];
  var n = pl.tracks[0].length;
  var cnt = document.getElementById('point-count-' + pIdx);
  if (cnt) cnt.textContent = (n - hiddenPoints[pIdx].size) + '/' + n;
}

function togglePoint(pIdx, i, checked){
  if (checked) hiddenPoints[pIdx].delete(i); else hiddenPoints[pIdx].add(i);
  updatePointCount(pIdx);
  render();
}

function selectAllPoints(pIdx){
  hiddenPoints[pIdx].clear();
  buildPointCheckboxes(pIdx);
  render();
}

function selectNonePoints(pIdx){
  var n = DATA.planes[pIdx].tracks[0].length;
  for (var i = 0; i < n; i++) hiddenPoints[pIdx].add(i);
  buildPointCheckboxes(pIdx);
  render();
}

function toggleShowSeg(pIdx, checked){ showSeg[pIdx] = checked; render(); }
function toggleShowTraj(pIdx, checked){ showTraj[pIdx] = checked; render(); }
function toggleShowPrev(pIdx, checked){ showPrev[pIdx] = checked; render(); }


function resetZoom(pIdx){
  viewZoom[pIdx] = {zoom: 1, panX: 0, panY: 0};
  render();
}

function initBigCanvasHandlers(pIdx){
  var cv = document.getElementById('big-cv-' + pIdx);
  if (!cv) return;
  var dragging = false, lastX = 0, lastY = 0;
  cv.addEventListener('wheel', function(evt){
    evt.preventDefault();
    var view = viewZoom[pIdx];
    var r = cv.getBoundingClientRect();
    var mx = (evt.clientX - r.left) * (cv.width / r.width);
    var my = (evt.clientY - r.top) * (cv.height / r.height);
    var factor = evt.deltaY < 0 ? 1.15 : 1 / 1.15;
    var newZoom = Math.max(0.5, Math.min(12, view.zoom * factor));
    view.panX = mx - (mx - view.panX) * (newZoom / view.zoom);
    view.panY = my - (my - view.panY) * (newZoom / view.zoom);
    view.zoom = newZoom;
    render();
  }, {passive: false});
  cv.addEventListener('mousedown', function(evt){
    dragging = true; lastX = evt.clientX; lastY = evt.clientY;
  });
  window.addEventListener('mousemove', function(evt){
    if (!dragging) return;
    var view = viewZoom[pIdx];
    var r = cv.getBoundingClientRect();
    var scaleX = cv.width / r.width, scaleY = cv.height / r.height;
    view.panX += (evt.clientX - lastX) * scaleX;
    view.panY += (evt.clientY - lastY) * scaleY;
    lastX = evt.clientX; lastY = evt.clientY;
    render();
  });
  window.addEventListener('mouseup', function(){ dragging = false; });
}

var mediaRecorder = [], recordedChunks = [];
function startRecording(pIdx){
  if (mediaRecorder[pIdx]) return;
  var cv = document.getElementById('big-cv-' + pIdx);
  if (!cv || !cv.captureStream) {
    alert("Video recording not supported by this browser (no captureStream).");
    return;
  }
  var stream = cv.captureStream(30);
  var mimeCandidates = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
  var mimeType = '';
  for (var i = 0; i < mimeCandidates.length; i++) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(mimeCandidates[i])) { mimeType = mimeCandidates[i]; break; }
  }
  try {
    mediaRecorder[pIdx] = mimeType ? new MediaRecorder(stream, {mimeType: mimeType}) : new MediaRecorder(stream);
  } catch (e) {
    alert("Video recording not supported by this browser.");
    return;
  }
  recordedChunks[pIdx] = [];
  mediaRecorder[pIdx].ondataavailable = function(e){ if (e.data && e.data.size > 0) recordedChunks[pIdx].push(e.data); };
  mediaRecorder[pIdx].onstop = function(){
    var outType = mediaRecorder[pIdx].mimeType || 'video/webm';
    var blob = new Blob(recordedChunks[pIdx], {type: outType});
    var ext = outType.indexOf('mp4') !== -1 ? 'mp4' : 'webm';
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    var pi = (DATA.planes[pIdx] || {}).pi;
    a.href = url; a.download = 'cotracker_plan' + (pi !== undefined ? pi : pIdx) + '_' + Date.now() + '.' + ext;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    setTimeout(function(){ URL.revokeObjectURL(url); }, 2000);
    mediaRecorder[pIdx] = null;
  };
  mediaRecorder[pIdx].start();
  var btnStart = document.getElementById('rec-start-' + pIdx), btnStop = document.getElementById('rec-stop-' + pIdx);
  if (btnStart) btnStart.disabled = true;
  if (btnStop) btnStop.disabled = false;
  var ind = document.getElementById('rec-indicator-' + pIdx); if (ind) ind.style.display = 'inline';
}
function stopRecording(pIdx){
  if (!mediaRecorder[pIdx]) return;
  mediaRecorder[pIdx].stop();
  var btnStart = document.getElementById('rec-start-' + pIdx), btnStop = document.getElementById('rec-stop-' + pIdx);
  if (btnStart) btnStart.disabled = false;
  if (btnStop) btnStop.disabled = true;
  var ind = document.getElementById('rec-indicator-' + pIdx); if (ind) ind.style.display = 'none';
}

function initCoTracker(data){
  DATA = data;
  curIdx = 0;
  data.planes.forEach(function(pl, pIdx){
    hiddenPoints[pIdx] = new Set();
    showSeg[pIdx] = false;
    showTraj[pIdx] = false;
    showPrev[pIdx] = false;
    colorMode[pIdx] = 'identity';
    viewZoom[pIdx] = {zoom: 1, panX: 0, panY: 0};
    for (var t = 0; t < pl.nt; t++) {
      var im = new Image();
      im.src = 'data:image/png;base64,' + pl.images[t];
      im.onload = render;
      IMG_CACHE[pIdx + '_' + t] = im;
    }
    buildPointCheckboxes(pIdx);
    initBigCanvasHandlers(pIdx);
  });
  render();
  var m = /^#plane-(.+)$/.exec(location.hash);
  if (m) {
    var idx = data.planes.findIndex(function(pl){ return String(pl.pi) === m[1]; });
    if (idx >= 0) switchTab(idx);
  }
}
"""


def build_tracking_html(subject, view, planes, n_points):
    """Interactive viewer of the 2D+t tracking of all planes of a view.

    planes: list of track_plane results. Filled point = visible, hollow point =
    occluded at this frame; excluded (island) points are listed separately."""
    colors = [_point_color(i, n_points) for i in range(n_points)]
    nt_shared = min(pl['nt'] for pl in planes)

    payload = []
    for pl in planes:
        conf = pl.get('confidence')
        payload.append(dict(
            pi=pl['plane'], nt=pl['nt'], anchor=pl['anchor'],
            images={str(t): _img_to_b64(pl['imgs'][t]) for t in range(pl['nt'])},
            tracks=[[[round(float(r), 2), round(float(c), 2)] for r, c in frame]
                    for frame in pl['tracks_rc']],
            visible=[[bool(v) for v in frame] for frame in pl['visible']],
            islandMask=[bool(v) for v in pl['island_mask']],
            confidence=([[round(float(c), 4) for c in frame] for frame in conf]
                        if conf is not None else None),
            contours=pl['contours']))
    dj = json.dumps(dict(planes=payload, colors=colors, nt=nt_shared, tw=TARGET_W, th=TARGET_H),
                    separators=(',', ':'))

    def island_note(pl):
        idx = [i for i, bad in enumerate(pl['island_mask']) if bad]
        if not idx:
            return ''
        return (f'<p class="anchor-note island-note">&#9888; {len(idx)} point(s) excluded '
                f'(island, motion inconsistent with the diaphragm): '
                f'{", ".join(f"#{i}" for i in idx)}</p>')

    panels_html = ''.join(
        f'<div class="panel"><h2>plane {pl["plane"]}</h2>'
        f'<p class="anchor-note">anchor: frame {pl["anchor"]} '
        f'<a href="#" onclick="goToAnchor({i});return false;">(go)</a></p>'
        f'{island_note(pl)}'
        f'<canvas class="ct-cv" id="ct-cv-{i}" width="{TARGET_W}" height="{TARGET_H}">'
        '</canvas></div>'
        for i, pl in enumerate(planes))

    tab_buttons = ''.join(
        f'<button class="tab-btn" id="tabbtn-{i}" onclick="switchTab({i})">'
        f'Plane {pl["plane"]}</button>'
        for i, pl in enumerate(planes))

    tab_contents = ''.join(
        f'<div id="tab-plane-{i}" class="tab-content" style="display:none">'
        f'<p class="anchor-note">Anchor frame: <b>{pl["anchor"]}</b> '
        f'<a href="#" onclick="goToAnchor({i});return false;">(go)</a></p>'
        f'{island_note(pl)}'
        '<div class="big-layout">'
        f'<canvas class="big-cv" id="big-cv-{i}" width="{TARGET_W}" height="{TARGET_H}">'
        '</canvas>'
        '<div class="side-controls">'
        '<div class="ctrl-row">'
        '<button class="play-btn" onclick="togglePlay()">&#9654; Play</button>'
        '</div>'
        '<div class="ctrl-row">'
        f'<button id="rec-start-{i}" onclick="startRecording({i})">&#9210; Record</button>'
        f'<button id="rec-stop-{i}" onclick="stopRecording({i})" disabled>&#9209; Stop</button>'
        f'<span id="rec-indicator-{i}" style="display:none;color:#ff3b3b;font-size:.8rem">'
        '&#9679; REC</span>'
        '</div>'
        '<div class="ctrl-row">'
        f'<button onclick="selectAllPoints({i})">All</button>'
        f'<button onclick="selectNonePoints({i})">None</button>'
        f'<span id="point-count-{i}" style="font-family:monospace"></span>'
        '</div>'
        f'<div class="point-box" id="point-box-{i}"></div>'
        + ('<div class="ctrl-row">'
           f'<label><input type="radio" name="colormode-{i}" value="identity" checked '
           f'onchange="setColorMode({i},this.value)"> Identity colour</label>'
           f'<label><input type="radio" name="colormode-{i}" value="confidence" '
           f'onchange="setColorMode({i},this.value)"> Confidence colour</label>'
           '</div>' if pl.get('confidence') is not None else '')
        + f'<label><input type="checkbox" onchange="toggleShowSeg({i},this.checked)"> '
        'Segmentation</label>'
        f'<label><input type="checkbox" onchange="toggleShowTraj({i},this.checked)"> '
        'Full trajectory</label>'
        f'<label><input type="checkbox" onchange="toggleShowPrev({i},this.checked)"> '
        'Previous positions</label>'
        f'<button onclick="resetZoom({i})">Reset zoom</button>'
        '</div></div>'
        '<p class="zoom-hint">Wheel = zoom (centred on the cursor), drag = pan.</p>'
        '</div>'
        for i, pl in enumerate(planes))

    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>Tracking - {subject} {view}</title>'
        f'<style>{CSS_CT}</style><script>{JS_CT}</script></head><body>'
        f'<h1>CoTracker point tracking - {subject} {view}</h1>'
        f'<p class="sub">{n_points} query point(s) on the medial line of the retained anchor '
        'frame, tracked independently on each plane of the view. Filled point = visible, '
        'hollow point = occluded at this frame (CoTracker visibility).</p>'
        '<div class="tabs">'
        '<button class="tab-btn active" id="tabbtn-overview" '
        'onclick="switchTab(\'overview\')">Overview</button>'
        f'{tab_buttons}'
        '</div>'
        f'<div id="tab-overview" class="tab-content"><div id="panels">{panels_html}</div></div>'
        f'{tab_contents}'
        '<div class="nav">'
        '<button class="play-btn" onclick="togglePlay()">&#9654; Play</button>'
        '<button onclick="stepFrame(-1)">&larr; Previous frame</button>'
        f'<input type="range" id="frame-slider" min="0" max="{nt_shared - 1}" value="0" '
        'oninput="seek(this.value)">'
        '<span id="frame-info" style="font-family:monospace"></span>'
        '<button onclick="stepFrame(1)">Next frame &rarr;</button>'
        '</div>'
        '<p class="legend">One colour per point (identity mode), shared by all planes. '
        '"Confidence colour": red (low positional confidence) to green (high), per frame, '
        'independent of the visibility.</p>'
        f'<script>initCoTracker({dj});</script>'
        '</body></html>'
    )
