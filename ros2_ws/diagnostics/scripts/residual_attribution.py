#!/usr/bin/env python3
"""Attribute paired directional pixel motion to FK-predicted motion.

This is an offline diagnostic only. It never writes URDF, ROS parameters, or
online calibration values.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import diagnose_handeye_model as model  # noqa: E402


def load_sample(path):
    sample = json.loads(path.read_text())
    sample['_path'] = str(path)
    return sample


def pair_paths(directory):
    paths = sorted(Path(directory).glob('sample_*.json'))
    groups = {}
    for path in paths:
        stem = path.stem.removeprefix('sample_')
        if stem.endswith('_increasing'):
            key = stem.removesuffix('_increasing')
            groups.setdefault(key, {})['increasing'] = path
        elif stem.endswith('_decreasing'):
            key = stem.removesuffix('_decreasing')
            groups.setdefault(key, {})['decreasing'] = path
    return {key: value for key, value in groups.items()
            if {'increasing', 'decreasing'} <= set(value)}


def project_for(sample, params, joint_indices):
    k = np.asarray(sample['camera']['k'], dtype=float).reshape(3, 3)
    d = np.asarray(sample['camera']['d'], dtype=float)
    q = np.asarray(sample['joints'], dtype=float).copy()
    offsets = np.zeros(6)
    for joint, offset in zip(joint_indices, params[12:]):
        offsets[joint] = offset
    q += offsets
    fk = model.kin.forward(
        'base_link', 'tcp_link',
        dict(zip(sample['joint_names'], q)),
    )
    predicted, depth = model.project_points(
        model.obj,
        fk,
        model.params_to_transform(params[:6]),
        model.params_to_transform(params[6:12]),
        k,
        d,
    )
    if np.any(depth <= 0.03):
        raise ValueError(f'{sample["id"]}: board depth is invalid')
    return predicted


def stats(vector):
    vector = np.asarray(vector, dtype=float)
    norms = np.linalg.norm(vector, axis=-1)
    return {
        'component_mean_px': np.mean(vector, axis=0).tolist(),
        'component_std_px': np.std(vector, axis=0).tolist(),
        'component_rms_px': np.sqrt(np.mean(vector ** 2, axis=0)).tolist(),
        'norm_median_px': float(np.median(norms)),
        'norm_mean_px': float(np.mean(norms)),
        'norm_rms_px': float(np.sqrt(np.mean(norms ** 2))),
        'norm_max_px': float(np.max(norms)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model-json', type=Path,
                        default=model.OUT / 'model_hypotheses.json')
    args = parser.parse_args()
    hypotheses = json.loads(args.model_json.read_text())
    models = {
        'baseline': (np.asarray(hypotheses['baseline']['parameters']), []),
        'joint2_5_offsets': (
            np.asarray(hypotheses['joint_2_3_4_5_offsets']['parameters']),
            [1, 2, 3, 4],
        ),
    }
    rows = []
    for directory in args.directories:
        for key, pair in pair_paths(directory).items():
            first = load_sample(pair['increasing'])
            second = load_sample(pair['decreasing'])
            observed = (np.asarray(second['corners'], dtype=float)
                        - np.asarray(first['corners'], dtype=float))
            row = {
                'directory': str(directory),
                'pair': key,
                'increasing_id': first['id'],
                'decreasing_id': second['id'],
                'joint_delta_rad': (
                    np.asarray(second['joints'])
                    - np.asarray(first['joints'])
                ).tolist(),
                'observed_shift': stats(observed),
                'models': {},
            }
            for name, (params, indices) in models.items():
                predicted = (project_for(second, params, indices)
                             - project_for(first, params, indices))
                unexplained = observed - predicted
                row['models'][name] = {
                    'predicted_shift': stats(predicted),
                    'unexplained_residual': stats(unexplained),
                }
            rows.append(row)
    if not rows:
        raise RuntimeError('No increasing/decreasing sample pairs found')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        'method': 'observed pixel delta minus model projected pixel delta for each paired q',
        'input_directories': [str(path) for path in args.directories],
        'rows': rows,
    }
    output = args.output_dir / 'residual_attribution.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    lines = [
        '# Pixel residual attribution', '',
        'Each row compares the increasing/decreasing pair at the same target.',
        'Residual = observed pixel shift - FK/model predicted pixel shift.', '',
        '| Pair | Model | observed RMS | predicted RMS | unexplained RMS |',
        '| --- | --- | ---: | ---: | ---: |',
    ]
    for row in rows:
        observed = row['observed_shift']['norm_rms_px']
        for name, result in row['models'].items():
            lines.append(
                f"| {row['pair']} | {name} | {observed:.3f} | "
                f"{result['predicted_shift']['norm_rms_px']:.3f} | "
                f"{result['unexplained_residual']['norm_rms_px']:.3f} |"
            )
    (args.output_dir / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'pairs': len(rows), 'output': str(output)}, indent=2))


if __name__ == '__main__':
    main()
