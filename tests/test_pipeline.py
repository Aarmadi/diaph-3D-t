"""End-to-end test on synthetic data (CoTracker replaced by a segmentation-based tracker)."""

import json
import os

import numpy as np

from diaph3dt.evaluation import EvalSettings, evaluate_case
from diaph3dt.io import case_file
from diaph3dt.reconstruction import ReconSettings, reconstruct_case
from diaph3dt.summary import best_params

from synthetic import SUBJECT, VIEW, make_dataset, segmentation_tracker


def test_reconstruct_and_evaluate(tmp_path):
    seg_dir, img_dir = make_dataset(str(tmp_path / 'data'))
    out_dir = str(tmp_path / 'out')
    case = reconstruct_case(seg_dir, img_dir, out_dir, SUBJECT, VIEW, model=None,
                            settings=ReconSettings(n_candidates=3),
                            tracker=segmentation_tracker)

    with open(case_file(case, 'points'), encoding='utf-8') as f:
        data = json.load(f)
    rest = np.array(data['rest_points'])
    assert len(rest) > 50                                   # most points anchored
    assert rest[:, 0].min() > 2                             # all on the right dome
    for kind in ('tracking', 'anchoring', 'viewer', 'inspi', 'expi', 'normal'):
        assert os.path.exists(case_file(case, kind)), kind
    assert len(data['frames']) == 3 * 30                    # planes interleaved in time

    rows = []
    evaluate_case(seg_dir, img_dir, case, SUBJECT, VIEW,
                  EvalSettings(rbf_sigmas_mm=[20], regs=[1e-2], biharm_weights=[1.0]),
                  rows.append)
    best = best_params(rows_as_floats(rows), target='normal')
    none, tps = best['none']['nsd3'], best['tps']['nsd3']
    print({m: round(b['nsd3'], 3) for m, b in best.items()})
    assert none < 0.5          # the template is 10 mm away from the spontaneous breath-hold
    assert tps > 0.8           # the deformed template matches it
    assert {r['region'] for r in rows} == {'all', 'slab'}


def rows_as_floats(rows):
    return [{k: (float(v) if isinstance(v, (int, float)) else v) for k, v in r.items()}
            for r in rows]
