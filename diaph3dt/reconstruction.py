"""Reconstruction of one case (subject, view): tracking, anchoring, temporal assembly.

Writes to <out_dir>/<subject>_<view>/:
  control_points.json   rest position of each control point on the template and its
                        positions over time (input of the deformation)
  mask_<state>.nii.gz   3D breath-hold masks restricted to the part seen by the view
                        (mask_inspi is the template)
  tracking.html, anchoring.html, reconstruction.html   interactive viewers"""

import json
import os
from dataclasses import dataclass

import numpy as np

from .anchoring import anchor_plane, build_anchor_html
from .io import case_dir, case_file, find_planes, mask_path, planes_geometry
from .template import load_mesh, save_filtered_mask, top_surface_faces
from .timeline import (attach_phi, fill_missing, frames_from_control_points,
                       interpolate_frames, plane_time_offset)
from .tracking import build_tracking_html, track_plane
from .viewer import build_case_html

STATES = ['inspi', 'expi', 'normal', 'normal_inspi', 'normal_expi']


@dataclass
class ReconSettings:
    """Reconstruction options (defaults: those of the paper)."""
    n_points: int = 50                 # query points per plane
    n_candidates: int = 15             # anchor frames of the multi-anchor tracking
    min_length_ratio: float = 0.5      # discard anchor frames with a truncated medial line
    outlier_factor: float = 3.0        # medoid consensus
    island_ratio_thr: float = 20.0     # island removal
    min_confidence: float = 0.8        # per-frame CoTracker confidence threshold
    planes_per_round: int = 3          # planes acquired in turn (acquisition delay)
    interp_steps: int = 3              # interpolated instants between frames (viewer)
    mesh_step_size: int = 2            # marching cubes step (template)
    html: bool = True                  # write the viewers


def reconstruct_case(seg_dir, img_dir, out_dir, subject, view, model, device='cuda',
                     settings=None, planes=None, viewer_kwargs=None, tracker=None):
    """Reconstruct one case and write its outputs. Returns the case folder.

    tracker: replaces CoTracker (see track_plane), for tests."""
    s = settings or ReconSettings()
    out = case_dir(out_dir, subject, view)
    os.makedirs(out, exist_ok=True)
    planes = planes or find_planes(seg_dir, img_dir, subject, view)
    if not planes:
        raise FileNotFoundError(f'No plane found for {subject} {view}')
    print(f'{subject} {view}: planes {planes}')

    # Template: upper surface of the inspiratory mask, restricted to the side seen by the
    # planes of the view. The other breath-hold masks are filtered the same way.
    geom = planes_geometry(seg_dir, subject, view, planes)
    for state in STATES:
        if os.path.exists(mask_path(seg_dir, subject, state)):
            save_filtered_mask(mask_path(seg_dir, subject, state), view, geom,
                               case_file(out, state))
    mesh_verts, mesh_faces = load_mesh(mask_path(seg_dir, subject, 'inspi'), view,
                                       s.mesh_step_size, planes_geom=geom)
    anchor_faces = top_surface_faces(mesh_verts, mesh_faces)

    control_points, tracks, anchors, plane_phis = [], [], [], {}
    for plane in planes:
        track = track_plane(seg_dir, img_dir, subject, view, plane, model, device,
                            n_points=s.n_points, n_candidates=s.n_candidates,
                            min_length_ratio=s.min_length_ratio,
                            outlier_factor=s.outlier_factor,
                            island_ratio_thr=s.island_ratio_thr,
                            **({'tracker': tracker} if tracker else {}))
        plane_phis[str(plane)] = track['phis']
        cps, diag = anchor_plane(track, mesh_verts, anchor_faces, s.min_confidence,
                                 t_offset=plane_time_offset(plane, s.planes_per_round))
        control_points += cps
        if s.html:
            tracks.append(track)
            anchors.append(diag)
    if not control_points:
        raise RuntimeError(f'{subject} {view}: no control point anchored on the template')

    # Time axis: frame t of plane p at t + p / planes_per_round; each point is
    # interpolated in time at the instants of the other planes.
    frames = fill_missing(attach_phi(frames_from_control_points(control_points), plane_phis,
                                     s.planes_per_round))
    data = dict(rest_points=[[round(float(c), 3) for c in cp['rest_pos']] for cp in control_points],
                frames=frames)
    if s.interp_steps > 0:
        data['frames_interp'] = interpolate_frames(frames, s.interp_steps,
                                                   step=1.0 / s.planes_per_round)
    with open(case_file(out, 'points'), 'w', encoding='utf-8') as f:
        json.dump(data, f)
    print(f'{subject} {view}: {len(control_points)} control points, {len(frames)} instants '
          f'-> {case_file(out, "points")}')

    if s.html:
        n_pts = tracks[0]['n_points']
        with open(case_file(out, 'tracking'), 'w', encoding='utf-8') as f:
            f.write(build_tracking_html(subject, view, tracks, n_pts))
        with open(case_file(out, 'anchoring'), 'w', encoding='utf-8') as f:
            f.write(build_anchor_html(subject, view, mesh_verts, mesh_faces, anchors))
        with open(case_file(out, 'viewer'), 'w', encoding='utf-8') as f:
            f.write(build_case_html(out, view, f'{subject} {view}', **(viewer_kwargs or {})))
    return out
