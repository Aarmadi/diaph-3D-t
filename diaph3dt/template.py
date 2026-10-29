"""Inspiratory template: 3D breath-hold mask -> surface mesh of the diaphragm.

The 3D masks are first restricted to the part observed by the view (one hemidiaphragm for
a sagittal view, both for the coronal view). The template is the upper (lung side) surface
of the marching-cubes mesh of the inspiratory mask."""

import nibabel as nib
import numpy as np
from skimage.measure import marching_cubes

from .geometry import largest_connected_component


def _classify_top_surface(mesh_verts, mesh_faces):
    """Boolean mask of the vertices of the upper (lung side) surface.

    The segmented diaphragm has a thickness, so its mesh has an upper and a lower
    surface. Vertices are split by the sign of the z component of their normal and the
    group with the highest mean z is kept (marching_cubes does not fix the normal
    orientation)."""
    v, f = mesh_verts, mesh_faces
    face_normals = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vertex_normals = np.zeros_like(v)
    for c in range(3):
        np.add.at(vertex_normals, f[:, c], face_normals)
    z_sign = vertex_normals[:, 2] >= 0
    if not z_sign.any() or z_sign.all():
        return np.ones(len(v), dtype=bool)
    return z_sign if v[z_sign, 2].mean() >= v[~z_sign, 2].mean() else ~z_sign


def mesh_signed_volume(verts, faces):
    """Signed volume of a mesh; its sign gives the face orientation (for back-face
    culling in the viewers)."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return float(np.einsum('ij,ij->i', v0, np.cross(v1 - v0, v2 - v0)).sum() / 6.0)


def load_filtered_mask(volume_path, view, level=0.5, planes_geom=None,
                       assume_prefiltered=False):
    """Load a 3D breath-hold mask and keep the part observed by the view (see
    largest_connected_component). Returns (mask, affine)."""
    img = nib.load(volume_path)
    mask = (np.asarray(img.get_fdata(), dtype=np.float32) > level).astype(np.float32)
    if not mask.any():
        raise ValueError(f'Empty 3D mask (no voxel > {level}): {volume_path}')
    mask = largest_connected_component(mask, aff_3d=img.affine, planes_geom=planes_geom,
                                       view=view, assume_prefiltered=assume_prefiltered)
    return mask, img.affine


def save_filtered_mask(volume_path, view, planes_geom, out_path):
    """Filter a 3D mask for a view (see load_filtered_mask) and save it."""
    mask, affine = load_filtered_mask(volume_path, view, planes_geom=planes_geom)
    nib.save(nib.Nifti1Image(mask, affine), out_path)
    return out_path


def load_mesh(volume_path, view, step_size=2, planes_geom=None, prefiltered=False,
              top_only=False):
    """Mesh (vertices in RAS mm, faces) extracted by marching cubes from a 3D mask.

    The mask is filtered for the view, unless prefiltered (a mask already saved by
    save_filtered_mask). top_only keeps the upper surface only."""
    mask, affine = load_filtered_mask(volume_path, view, planes_geom=planes_geom,
                                      assume_prefiltered=prefiltered)
    verts, faces, _, _ = marching_cubes(mask, level=0.5, step_size=step_size)
    verts = nib.affines.apply_affine(affine, verts).astype(np.float64)
    faces = faces.astype(np.int64)
    if top_only:
        verts, faces = top_surface_submesh(verts, faces)
    return verts, faces


def top_surface_faces(mesh_verts, mesh_faces):
    """Faces of the upper surface (the only ones a trajectory is anchored to)."""
    top = _classify_top_surface(mesh_verts, mesh_faces)
    return mesh_faces[top[mesh_faces].all(axis=1)]


def top_surface_submesh(mesh_verts, mesh_faces):
    """Upper surface as a separate mesh (unused vertices removed)."""
    faces = top_surface_faces(mesh_verts, mesh_faces)
    used = np.unique(faces)
    remap = np.full(len(mesh_verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return mesh_verts[used], remap[faces]
