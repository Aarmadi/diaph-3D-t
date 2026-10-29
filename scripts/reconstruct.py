#!/usr/bin/env python3
"""Dynamic 3D reconstruction of the diaphragm from 2D+t planes (step 1).

For each subject and view: CoTracker tracking of each plane, anchoring of the
trajectories on the inspiratory template, temporal assembly of the planes. Outputs in
<out-dir>/<subject>_<view>/ (see diaph3dt/reconstruction.py).

Example:
  python scripts/reconstruct.py --seg-dir data/seg --img-dir data/img --out-dir out
  python scripts/reconstruct.py ... --subjects SUBJ01 --views D G
"""

import argparse
import traceback

from diaph3dt.io import VIEWS, find_subjects
from diaph3dt.reconstruction import ReconSettings, reconstruct_case
from diaph3dt.tracking import load_cotracker_model


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seg-dir', required=True, help='2D+t and 3D segmentations')
    ap.add_argument('--img-dir', required=True, help='2D+t images')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--subjects', nargs='*', default=None,
                    help='default: all subjects with inspiratory and expiratory 3D masks')
    ap.add_argument('--views', nargs='*', default=list(VIEWS), choices=VIEWS)
    ap.add_argument('--device', default='cuda')
    d = ReconSettings()
    ap.add_argument('--n-points', type=int, default=d.n_points)
    ap.add_argument('--n-candidates', type=int, default=d.n_candidates)
    ap.add_argument('--island-ratio-thr', type=float, default=d.island_ratio_thr)
    ap.add_argument('--min-confidence', type=float, default=d.min_confidence)
    ap.add_argument('--planes-per-round', type=int, default=d.planes_per_round)
    ap.add_argument('--no-html', action='store_true', help='skip the viewers')
    args = ap.parse_args()

    settings = ReconSettings(n_points=args.n_points, n_candidates=args.n_candidates,
                             island_ratio_thr=args.island_ratio_thr,
                             min_confidence=args.min_confidence,
                             planes_per_round=args.planes_per_round, html=not args.no_html)
    subjects = args.subjects or find_subjects(args.seg_dir)
    print(f'{len(subjects)} subject(s), views {args.views}')
    model, device = load_cotracker_model(device=args.device)
    failed = []
    for subject in subjects:
        for view in args.views:
            try:
                reconstruct_case(args.seg_dir, args.img_dir, args.out_dir, subject, view,
                                 model, device, settings)
            except Exception:   # one failing case does not stop the others
                traceback.print_exc()
                failed.append(f'{subject} {view}')
    print(f'Done, {len(failed)} failure(s)' + (f': {", ".join(failed)}' if failed else ''))


if __name__ == '__main__':
    main()
