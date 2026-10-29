"""Selection of the best parameters and summary tables of the evaluation CSV."""

import csv
from collections import defaultdict

import numpy as np

FWD_METRICS = ['nsd2', 'nsd3', 'nsd5', 'asd', 'p90', 'hd95']
METRICS = FWD_METRICS + [f'r_{k}' for k in FWD_METRICS] + [f's_{k}' for k in FWD_METRICS]
TABLE_METRICS = ['nsd2', 'nsd3', 'nsd5', 'asd', 's_asd', 'p90', 'hd95']   # s_asd = ASSD
VIEWS = ['D', 'G', 'cor']


def higher_is_better(metric):
    return 'nsd' in metric


def read_rows(csv_path):
    """Rows of the evaluation CSV, metrics converted to float."""
    with open(csv_path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in METRICS + ['reached', 'reach']:
            if r.get(k) not in (None, ''):
                r[k] = float(r[k])
    return rows


def best_params(rows, target='normal', metric='nsd3', region='all'):
    """{method: {'params', <mean metrics>, 'n_cases'}}: for each method, the parameter set
    with the best mean `metric` on `target`, among those evaluated on all its cases."""
    rows = [r for r in rows if r['target'] == target and r['region'] == region]
    sign = 1 if higher_is_better(metric) else -1
    out = {}
    for method in sorted({r['method'] for r in rows}):
        mrows = [r for r in rows if r['method'] == method and isinstance(r.get(metric), float)]
        n_cases = len({(r['subject'], r['view']) for r in mrows})
        by_params = defaultdict(list)
        for r in mrows:
            by_params[r['params']].append(r)
        complete = {p: v for p, v in by_params.items() if len(v) == n_cases}
        if not complete:
            continue
        p = max(complete, key=lambda k: sign * np.mean([x[metric] for x in complete[k]]))
        rec = dict(params=p, n_cases=n_cases)
        for k in METRICS:
            vals = [x[k] for x in complete[p] if isinstance(x.get(k), float)]
            if vals:
                rec[k] = float(np.mean(vals))
        out[method] = rec
    return out


def results_table(rows, best, region='all', reach_thr=None):
    """Mean metrics per reference, method and view (and all views), each method with its
    best parameters. reach_thr: for the expiratory reference, keep only the cases whose
    sequence reaches phi >= reach_thr. Returns a printable string."""
    acc = defaultdict(lambda: defaultdict(list))
    cases = defaultdict(set)
    for r in rows:
        m = r['method']
        if r['region'] != region or m not in best or r['params'] != best[m]['params']:
            continue
        if reach_thr is not None and r['target'] == 'expi' and r.get('reach', 0.0) < reach_thr:
            continue
        for view in (r['view'], 'all'):
            key = (r['target'], m, view)
            cases[key].add((r['subject'], r['view']))
            for k in TABLE_METRICS:
                if isinstance(r.get(k), float):
                    acc[key][k].append(r[k])
    lines = []
    for target in sorted({k[0] for k in acc}, key=lambda t: (t != 'normal', t)):
        note = f', phase max >= {reach_thr:g}' if reach_thr is not None and target == 'expi' else ''
        lines.append(f'\n=== reference: {target} (region {region}{note})')
        lines.append(f"{'method':12s} {'view':4s} {'n':>3s}" + ''.join(f'{k:>8s}' for k in TABLE_METRICS))
        for m in [m for m in best if any(k[:2] == (target, m) for k in acc)]:
            for v in VIEWS + ['all']:
                key = (target, m, v)
                if key not in acc:
                    continue
                vals = [np.mean(acc[key][k]) if acc[key][k] else np.nan for k in TABLE_METRICS]
                lines.append(f'{m:12s} {v:4s} {len(cases[key]):3d}'
                             + ''.join(f'{x:8.3f}' for x in vals))
            lines.append('')
    lines.append('NSD as a fraction (1 = whole reference within tau mm); asd, s_asd (ASSD), '
                 'p90 and hd95 in mm.')
    return '\n'.join(lines)
