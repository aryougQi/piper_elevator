#!/usr/bin/env python3
"""Joint-offset + hand-eye estimation with strict held-out validation."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_handeye_model as model  # noqa: E402


JOINT_INDICES = [1, 2, 3, 4]  # joint2..joint5, zero-based


def load_dir(directory):
    samples = [json.loads(path.read_text())
               for path in sorted(Path(directory).glob('sample_*.json'))]
    for sample in samples:
        model.cache[id(sample)] = (
            np.asarray(sample['camera']['k']).reshape(3, 3),
            np.asarray(sample['camera']['d']),
            np.asarray(sample['corners']),
        )
    return samples


def fit(train, initial, bound_deg):
    lower = np.r_[np.full(12, -np.inf),
                  np.full(4, -np.deg2rad(bound_deg))]
    upper = np.r_[np.full(12, np.inf),
                  np.full(4, np.deg2rad(bound_deg))]
    return least_squares(
        lambda values: model.residual(values, train, JOINT_INDICES),
        initial, bounds=(lower, upper), loss='huber', f_scale=2.0,
        max_nfev=2000,
    )


def rmse(values, samples):
    return float(np.sqrt(np.mean(
        model.residual(values, samples, JOINT_INDICES) ** 2
    )))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fit_dir', type=Path)
    parser.add_argument('--validation-dir', action='append', type=Path, default=[])
    parser.add_argument('--extra-train-dir', action='append', type=Path, default=[])
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--offset-bound-deg', type=float, default=15.0)
    args = parser.parse_args()
    fit_samples = load_dir(args.fit_dir)
    validation = [sample for directory in args.validation_dir
                  for sample in load_dir(directory)]
    extra_train = [sample for directory in args.extra_train_dir
                   for sample in load_dir(directory)]
    if len(fit_samples) < 10:
        raise RuntimeError('at least 10 fit samples are required')
    initial = np.r_[model.initial, np.zeros(4)]
    solution = fit(fit_samples + extra_train, initial, args.offset_bound_deg)
    rows = []
    for fold in range(5):
        train = [sample for index, sample in enumerate(fit_samples)
                 if index % 5 != fold]
        held_out = [sample for index, sample in enumerate(fit_samples)
                    if index % 5 == fold]
        fold_solution = fit(train + extra_train, solution.x, args.offset_bound_deg)
        rows.append({
            'held_out': [sample['id'] for sample in held_out],
            'train_rmse_px': rmse(fold_solution.x, train),
            'held_out_rmse_px': rmse(fold_solution.x, held_out),
            'validation_rmse_px': None if not validation else
                rmse(fold_solution.x, validation),
            'offset_degrees': (fold_solution.x[12:] * 180 / np.pi).tolist(),
        })
    report = {
        'fit_count': len(fit_samples),
        'extra_train_count': len(extra_train),
        'validation_count': len(validation),
        'offset_joint_names': ['joint2', 'joint3', 'joint4', 'joint5'],
        'full_fit_rmse_px': rmse(solution.x, fit_samples),
        'extra_train_rmse_px': None if not extra_train else rmse(solution.x, extra_train),
        'validation_rmse_px': None if not validation else rmse(solution.x, validation),
        'offset_degrees': (solution.x[12:] * 180 / np.pi).tolist(),
        'five_fold': rows,
        'passed': False,
    }
    # A pass requires every held-out fold and every independent validation
    # sample set to be below 2 px; keep the rule explicit in the artifact.
    report['passed'] = bool(
        all(row['held_out_rmse_px'] < 2.0 for row in rows)
        and validation and report['validation_rmse_px'] < 2.0
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'joint_offset_handeye_cv.json').write_text(
        json.dumps(report, indent=2) + '\n'
    )
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
