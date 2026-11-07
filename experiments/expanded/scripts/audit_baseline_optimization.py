"""Check exact legacy-vs-precomputed V21 scores before using optimization."""
from __future__ import annotations
import json
from pathlib import Path
import time

import numpy as np

from baseline_adapter import build_index, _fast_diverse_top_indices


def main():
    experiment = Path(__file__).resolve().parents[1]
    project = Path(r'${INTENTRAG_PROJECT_ROOT}')
    index = build_index(project)
    cases = [
        ('Boeing 737', 'Boeing 737 在出现气象雷达地面杂波时，应执行哪些处置或排故动作？'),
        ('Boeing 737', 'For Boeing 737 inspect hydraulic engine warning threshold Evidence cue: HNA Technic 海航航空技术有限公司 硬着陆检查条件.'),
        ('Cessna 172', 'For Cessna 172, what inspection actions and warnings apply to the engine oil pressure?'),
        ('Cessna 172', 'For Cessna 172, retrieve threshold and condition support. Include evidence cues: frequently the tasks define more specifically the scope and extent of each required inspection; warning do not exceed maximum rpm during ground operation.'),
    ]
    ev, features = index._v21('Boeing 737')
    rows = []
    for plane, question in cases:
        before = time.perf_counter()
        old = index.v21_module.query_scores(question, plane, ev, *features)
        old_time = time.perf_counter()-before
        before = time.perf_counter()
        optimized = index._optimized_v21_scores(question, plane, ev, features)
        new_time = time.perf_counter()-before
        equal = {method: bool(np.array_equal(old[method], optimized[method])) for method in old}
        max_abs = {method: float(np.max(np.abs(old[method]-optimized[method]))) for method in old}
        rows.append({'aircraft': plane, 'question': question, 'all_arrays_bitwise_equal': all(equal.values()),
                     'by_method_bitwise_equal': equal, 'max_abs_difference': max_abs,
                     'old_wall_time_s': old_time, 'optimized_wall_time_s': new_time,
                     'speedup': old_time/new_time, 'n_joint_index_rows': len(ev)})
    diversity_rows = []
    rng = np.random.default_rng(20261014)
    for corpus in (ev, ev.loc[ev.aircraft_model.eq('Boeing 737')].reset_index(drop=True)):
        frame = corpus[['page_id', 'role_group', 'role']]
        for kind, scores in (('random', rng.random(len(frame))),
                             ('quantized_ties', np.round(rng.random(len(frame)), 1)),
                             ('all_equal', np.ones(len(frame)))):
            for page_weight, role_weight in ((0.06, 0.04), (0.05, 0.0)):
                before = time.perf_counter()
                legacy = index.original_diverse_top_indices(scores, frame, k=10,
                    page_weight=page_weight, role_weight=role_weight)
                old_time = time.perf_counter()-before
                before = time.perf_counter()
                fast = _fast_diverse_top_indices(scores, frame, k=10,
                    page_weight=page_weight, role_weight=role_weight)
                new_time = time.perf_counter()-before
                diversity_rows.append({'n_rows': len(frame), 'score_kind': kind,
                    'page_weight': page_weight, 'role_weight': role_weight,
                    'identical_ordered_indices': bool(np.array_equal(legacy, fast)),
                    'old_wall_time_s': old_time, 'optimized_wall_time_s': new_time,
                    'old_indices': legacy.tolist(), 'new_indices': fast.tolist()})
    passed = all(row['all_arrays_bitwise_equal'] for row in rows) and all(row['identical_ordered_indices'] for row in diversity_rows)
    report = {'status': 'PASS' if passed else 'FAIL',
              'cases': rows, 'original_source_hashes': index.source_hashes,
              'diversity_selection_cases': diversity_rows,
              'scope': 'All six V21 score arrays on full joint corpus; no weight, IDF or rank changes.'}
    destination = experiment/'reports'/'baseline_optimization_equivalence.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
