#!/usr/bin/env python3
"""Split 2D+t segmentations stored with the planes of a view concatenated in time.

Input: <subject>_<view>_seg.nii.gz of shape (ni, nj, P * nt), the P planes one after the
other along the last axis. Output: one file <plane>_<subject>_<view>_seg.nii.gz per plane.
The concatenated file has a single affine while the planes are at different positions,
so each output takes the affine of the image file of its plane
(<plane>_<subject>_<view>.nii.gz in --img-dir).

Example:
  python scripts/split_planes.py --in-dir seg_concat --img-dir data/img --out-dir data/seg
"""

import argparse
import os
import re

import nibabel as nib
import numpy as np

CONCAT_RE = re.compile(r'^(?!\d+_)(.+?)_(cor|D|G)_seg\.nii\.gz$')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in-dir', required=True)
    ap.add_argument('--img-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--n-planes', type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for name in sorted(os.listdir(args.in_dir)):
        m = CONCAT_RE.match(name)
        if not m:
            continue
        subject, view = m.groups()
        img = nib.load(os.path.join(args.in_dir, name))
        arr = np.asarray(img.get_fdata(), dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] % args.n_planes:
            print(f'{name}: unexpected shape {arr.shape}, skipped')
            continue
        nt = arr.shape[2] // args.n_planes
        for plane in range(args.n_planes):
            raw = os.path.join(args.img_dir, f'{plane}_{subject}_{view}.nii.gz')
            if not os.path.exists(raw):
                print(f'{name}: missing {raw}, plane {plane} skipped')
                continue
            out = nib.Nifti1Image(arr[:, :, plane * nt:(plane + 1) * nt], nib.load(raw).affine,
                                  img.header)
            out.set_data_dtype(np.float32)
            nib.save(out, os.path.join(args.out_dir, f'{plane}_{subject}_{view}_seg.nii.gz'))
        print(f'{name}: {args.n_planes} planes of {nt} frames')


if __name__ == '__main__':
    main()
