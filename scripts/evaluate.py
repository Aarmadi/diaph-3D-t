#!/usr/bin/env python3
"""Evaluation of the reconstructions against the 3D breath-hold segmentations (step 2).

Evaluates every deformation method over its parameter grid on every reconstructed case,
writes one CSV row per (case, method, parameters, reference, region), then selects for
each method the parameters with the best mean NSD@3 mm on the spontaneous breath-hold
('normal', region 'all') and prints the summary table.

On a test cohort, --fixed-params evaluates only the parameters selected on the tuning
cohort (its best_params.json).

Example:
  python scripts/evaluate.py --seg-dir data/seg --img-dir data/img --out-dir out --jobs 8
  python scripts/evaluate.py ... --out-dir out_test --fixed-params out/best_params.json
"""

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor

from diaph3dt.evaluation import METHODS, TARGETS, EvalSettings, evaluate_case
from diaph3dt.io import VIEWS, case_dir, case_file, find_subjects
from diaph3dt.summary import best_params, read_rows, results_table

COLUMNS = (['subject', 'view', 'method', 'params', 'target', 'region', 'frame_t', 'phase',
            'reach', 'reached', 'target_phase']
           + [f'{p}{k}' for p in ('', 'r_', 's_')
              for k in ('nsd2', 'nsd3', 'nsd5', 'asd', 'p90', 'hd95')]
           + ['seconds'])


def _evaluate(task):
    seg_dir, img_dir, out_dir, subject, view, settings = task
    rows = []
    try:
        evaluate_case(seg_dir, img_dir, case_dir(out_dir, subject, view), subject, view,
                      settings, rows.append)
    except Exception as e:   # one failing case does not stop the others
        print(f'  {subject} {view}: error {type(e).__name__}: {e}')
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seg-dir', required=True)
    ap.add_argument('--img-dir', required=True)
    ap.add_argument('--out-dir', required=True, help='folder of the reconstructions')
    ap.add_argument('--subjects', nargs='*', default=None)
    ap.add_argument('--views', nargs='*', default=list(VIEWS), choices=VIEWS)
    ap.add_argument('--methods', nargs='+', default=METHODS, choices=METHODS)
    ap.add_argument('--targets', nargs='+', default=TARGETS,
                    help='reference breath-holds (default: %(default)s)')
    ap.add_argument('--fixed-params', default=None,
                    help='best_params.json of the tuning cohort (test cohort)')
    ap.add_argument('--tune-target', default='normal')
    ap.add_argument('--reach-thr', type=float, default=1.0,
                    help='expiratory reference: keep the cases reaching phi >= this value '
                         'in the table (default %(default)s)')
    ap.add_argument('--jobs', type=int, default=1)
    args = ap.parse_args()

    fixed = None
    if args.fixed_params:
        with open(args.fixed_params, encoding='utf-8') as f:
            fixed = {m: b['params'] for m, b in json.load(f).items()}
    settings = EvalSettings(methods=args.methods, targets=args.targets, fixed=fixed)
    subjects = args.subjects or find_subjects(args.seg_dir)
    tasks = [(args.seg_dir, args.img_dir, args.out_dir, s, v, settings)
             for s in subjects for v in args.views
             if os.path.exists(case_file(case_dir(args.out_dir, s, v), 'points'))]
    print(f'{len(tasks)} reconstructed case(s) to evaluate')

    csv_path = os.path.join(args.out_dir, 'evaluation.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        if args.jobs > 1:
            with ProcessPoolExecutor(args.jobs) as pool:
                for rows in pool.map(_evaluate, tasks):
                    writer.writerows(rows)
                    f.flush()
        else:
            for task in tasks:
                writer.writerows(_evaluate(task))
                f.flush()
    print(f'Results: {csv_path}')

    rows = read_rows(csv_path)
    # With --fixed-params only the fixed parameters were evaluated, so they are selected.
    best = best_params(rows, target=args.tune_target)
    if fixed is None:
        with open(os.path.join(args.out_dir, 'best_params.json'), 'w', encoding='utf-8') as f:
            json.dump(best, f, indent=2)
    for region in ('all', 'slab'):
        print(results_table(rows, best, region=region, reach_thr=args.reach_thr))


if __name__ == '__main__':
    main()
