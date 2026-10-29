#!/usr/bin/env python3
"""Regenerate the reconstruction viewers with the selected parameters (step 3).

The page opens on the chosen method (default: the best mean NSD@3 mm of
best_params.json) with its kernel scale and regularisation. No tracking is redone.

Example:
  python scripts/render.py --out-dir out
  python scripts/render.py --out-dir out_test --best-params out/best_params.json --method tps
"""

import argparse
import json
import os

from diaph3dt.io import VIEWS, case_file
from diaph3dt.viewer import build_case_html

VIEWER_METHOD = {'rbf': 'exact', 'rbf_affine': 'affine', 'tps': 'tps', 'biharm': 'biharm'}


def parse_params(params):
    """'sigma=20mm;reg=0.01' -> {'sigma': '20mm', 'reg': '0.01'}"""
    return dict(item.split('=', 1) for item in params.split(';') if '=' in item)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', required=True, help='folder of the reconstructions')
    ap.add_argument('--best-params', default=None, help='default: <out-dir>/best_params.json')
    ap.add_argument('--method', default=None, choices=sorted(VIEWER_METHOD))
    args = ap.parse_args()

    with open(args.best_params or os.path.join(args.out_dir, 'best_params.json'),
              encoding='utf-8') as f:
        best = {m: b for m, b in json.load(f).items() if m in VIEWER_METHOD}
    method = args.method or max(best, key=lambda m: best[m]['nsd3'])
    p = parse_params(best[method]['params'])
    kwargs = dict(default_method=VIEWER_METHOD[method], reg=float(p.get('reg', 1e-2)))
    if 'sigma' in p:
        kwargs['sigma'] = float(p['sigma'].replace('mm', ''))
    if 'biharm' in best:
        kwargs['biharm_weight'] = float(parse_params(best['biharm']['params'])['weight'])
    print(f'Method {method} ({best[method]["params"]})')

    for name in sorted(os.listdir(args.out_dir)):
        folder = os.path.join(args.out_dir, name)
        if not os.path.exists(case_file(folder, 'points')):
            continue
        subject, view = name.rsplit('_', 1)
        if view not in VIEWS:
            continue
        with open(case_file(folder, 'viewer'), 'w', encoding='utf-8') as f:
            f.write(build_case_html(folder, view, f'{subject} {view}', **kwargs))
        print(f'  {case_file(folder, "viewer")}')


if __name__ == '__main__':
    main()
