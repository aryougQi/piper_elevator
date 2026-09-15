#!/usr/bin/env python3
"""Offline sample-level sensitivity of the persisted checkerboard hand-eye fit."""

import argparse
import importlib.util
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


DIAGNOSTICS = Path(__file__).resolve().parents[1]
CALIBRATION = DIAGNOSTICS.parent.parent.parent / 'handeye_calibration/calibration'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration-root', type=Path, default=CALIBRATION)
    parser.add_argument('--runtime-snapshot', type=Path, default=(
        DIAGNOSTICS / 'data/handeye_frame_audit/20260908_140832/runtime_snapshot.json'
    ))
    parser.add_argument('--projection-report', type=Path, default=(
        DIAGNOSTICS / 'data/coarse_detection_loss/projection_geometry.json'
    ))
    parser.add_argument('--capture-report', type=Path, default=(
        DIAGNOSTICS / 'data/coarse_detection_loss/capture.json'
    ))
    parser.add_argument('--bootstrap-count', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20260908)
    parser.add_argument('--output', type=Path,
                        default=DIAGNOSTICS / 'data/handeye_resampling_audit.json')
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f'Preserve previous evidence: {args.output} already exists')
    module_path = args.calibration_root.parent / 'scripts/solve_handeye.py'
    spec = importlib.util.spec_from_file_location('offline_handeye_solver', module_path)
    solver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solver)
    kinematics = solver.UrdfKinematics(args.calibration_root / 'piper_handeye_input.urdf')
    original = json.loads((
        args.calibration_root / 'result_20260902_linear/handeye_result.json'
    ).read_text())
    samples = [json.loads(path.read_text()) for path in sorted(
        (args.calibration_root / 'samples_20260902').glob('sample_*.json')
    )]
    samples = [sample for sample in samples
               if sample['id'] not in original['excluded_sample_ids']]
    validation = [json.loads(path.read_text()) for path in sorted(
        (args.calibration_root / 'validation_20260902/samples').glob('sample_*.json')
    )]
    matrix = np.asarray(samples[0]['camera']['k']).reshape(3, 3)
    distortion = np.asarray(samples[0]['camera']['d'])
    if any(not np.array_equal(np.asarray(sample['camera']['k']).reshape(3, 3), matrix)
           or not np.array_equal(sample['camera']['d'], distortion)
           for sample in samples + validation):
        raise RuntimeError('This vectorized replay requires identical camera intrinsics')
    board_points = np.asarray([[c * 0.019, r * 0.019, 0.0]
                               for r in range(11) for c in range(8)])
    board_homogeneous = np.column_stack((board_points, np.ones(len(board_points)))).T

    def robot_poses(values):
        return np.asarray([kinematics.forward('base_link', 'tcp_link', {
            f'joint{index + 1}': angle for index, angle in enumerate(sample['joints'])
        }) for sample in values])

    fit_poses, validation_poses = robot_poses(samples), robot_poses(validation)
    observed = np.asarray([sample['corners'] for sample in samples])
    validation_observed = np.asarray([sample['corners'] for sample in validation])
    camera = original['tcp_to_camera_color_optical_frame']
    board = original['base_to_checkerboard_first_corner']
    camera_transform = solver.transform(
        camera['xyz'], Rotation.from_quat(camera['quaternion_xyzw']).as_matrix(),
    )
    board_transform = solver.transform(
        board['xyz'], Rotation.from_euler('xyz', board['rpy']).as_matrix(),
    )
    initial = np.r_[solver.transform_to_params(camera_transform),
                    solver.transform_to_params(board_transform)]

    def residual(values, poses, pixels):
        camera = solver.params_to_transform(values[:6])
        board = solver.params_to_transform(values[6:])
        camera_points = (np.linalg.inv(poses @ camera) @ board @ board_homogeneous)
        camera_points = camera_points.transpose(0, 2, 1)[..., :3]
        projected, _ = cv2.projectPoints(
            camera_points.reshape(-1, 3), np.zeros(3), np.zeros(3), matrix, distortion,
        )
        difference = projected.reshape(pixels.shape) - pixels
        if np.any(camera_points[..., 2] <= 0.03):
            difference = difference + 1000.0
        return difference.ravel()

    snapshot = json.loads(args.runtime_snapshot.read_text())
    feedback = snapshot['joint_states']['/feedback/joint_states']
    current_joints = dict(zip(feedback['name'], feedback['position']))
    capture = json.loads(args.capture_report.read_text())
    planning = json.loads(capture['latest']['/button_approach_planner/observation_status'])
    home_joints = planning['last_ik_search']['current_joints']
    home = kinematics.forward('base_link', 'tcp_link', home_joints)
    current = kinematics.forward('base_link', 'tcp_link', current_joints)
    geometry = json.loads(args.projection_report.read_text())
    fixed_world_point = np.r_[geometry['planned_button'], 1.0]
    fixed_home_camera = np.linalg.inv(home @ camera_transform) @ fixed_world_point
    fixed_current_camera = np.linalg.inv(current @ camera_transform) @ fixed_world_point
    measured_current_camera = []
    for sample in geometry['samples']:
        measured_current_camera.append(np.asarray(sample['camera_rotation']).T @ (
            np.asarray(sample['measured_base']) - sample['camera_translation']
        ))
    measured_current_camera = np.r_[np.median(measured_current_camera, axis=0), 1.0]

    def metrics(solution, ids, kind, omitted=None):
        camera = solver.params_to_transform(solution.x[:6])
        delta = np.linalg.inv(camera_transform) @ camera
        home_point = home @ camera @ fixed_home_camera
        current_point = current @ camera @ fixed_current_camera
        measured_point = current @ camera @ measured_current_camera
        drift = 1000 * (current_point[:3] - home_point[:3])
        remaining = 1000 * (measured_point[:3] - home_point[:3])
        fit_error = residual(solution.x, fit_poses, observed)
        validation_error = residual(solution.x, validation_poses, validation_observed)
        output = {
            'kind': kind, 'sample_ids': [samples[index]['id'] for index in ids],
            'unique_sample_count': len(set(ids)), 'success': bool(solution.success),
            'message': str(solution.message), 'nfev': int(solution.nfev),
            'camera_translation_delta_mm': float(np.linalg.norm(delta[:3, 3]) * 1000),
            'camera_rotation_delta_degrees': float(np.degrees(
                Rotation.from_matrix(delta[:3, :3]).magnitude())),
            'camera_delta_xyz_mm_in_original_camera_frame': (delta[:3, 3] * 1000).tolist(),
            'tcp_to_optical_xyz_m': camera[:3, 3].tolist(),
            'tcp_to_optical_quaternion_xyzw': Rotation.from_matrix(camera[:3, :3]).as_quat().tolist(),
            'original_14_frame_rmse_px': float(np.sqrt(np.mean(fit_error ** 2))),
            'validation_3_frame_rmse_px': float(np.sqrt(np.mean(validation_error ** 2))),
            'fixed_point_current_minus_home_change_mm': drift.tolist(),
            'fixed_point_current_minus_home_change_norm_mm': float(np.linalg.norm(drift)),
            'observed_current_minus_inferred_home_mm': remaining.tolist(),
            'observed_current_minus_inferred_home_norm_mm': float(np.linalg.norm(remaining)),
        }
        if omitted is not None:
            error = residual(solution.x, fit_poses[[omitted]], observed[[omitted]])
            output['omitted_id'] = samples[omitted]['id']
            output['omitted_frame_rmse_px'] = float(np.sqrt(np.mean(error ** 2)))
        return output

    def fit(ids, loss='linear'):
        return least_squares(
            residual, initial, args=(fit_poses[ids], observed[ids]),
            method='trf', loss=loss, f_scale=2.0,
            x_scale=np.r_[np.full(3, 0.1), np.ones(3), np.full(3, 0.5), np.ones(3)],
            max_nfev=200, ftol=1e-9, xtol=1e-9, gtol=1e-9,
        )

    started = time.monotonic()
    indices = np.arange(len(samples))
    results = [metrics(fit(indices), indices.tolist(), 'all_samples_refit')]
    print(json.dumps(results[0]), flush=True)
    for omitted in indices:
        kept = indices[indices != omitted]
        row = metrics(fit(kept), kept.tolist(), 'leave_one_out', omitted=int(omitted))
        results.append(row)
        print(json.dumps({key: row[key] for key in [
            'kind', 'omitted_id', 'success', 'camera_translation_delta_mm',
            'camera_rotation_delta_degrees', 'fixed_point_current_minus_home_change_norm_mm',
            'observed_current_minus_inferred_home_norm_mm',
        ]}), flush=True)
    rng = np.random.default_rng(args.seed)
    for index in range(args.bootstrap_count):
        drawn = rng.integers(0, len(samples), size=len(samples))
        row = metrics(fit(drawn), drawn.tolist(), 'sample_bootstrap')
        row['bootstrap_index'] = index
        results.append(row)
        if (index + 1) % 8 == 0:
            print(f'Bootstrap {index + 1}/{args.bootstrap_count}', flush=True)
    fields = [
        'camera_translation_delta_mm', 'camera_rotation_delta_degrees',
        'original_14_frame_rmse_px', 'validation_3_frame_rmse_px',
        'fixed_point_current_minus_home_change_norm_mm',
        'observed_current_minus_inferred_home_norm_mm',
    ]
    summaries = {}
    for kind in ['leave_one_out', 'sample_bootstrap']:
        selected = [row for row in results if row['kind'] == kind and row['success']]
        summaries[kind] = {
            'successful_fits': len(selected),
            'attempted_fits': sum(row['kind'] == kind for row in results),
            'min_median_p95_max': {
                field: [float(value) for value in np.percentile(
                    [row[field] for row in selected], [0, 50, 95, 100],
                )] for field in fields
            } if selected else {},
        }
    output = {
        'offline_only': True, 'loss': 'linear', 'bootstrap_unit': 'whole_robot_pose',
        'bootstrap_seed': args.seed, 'original_sample_ids': [sample['id'] for sample in samples],
        'original_excluded_sample_ids': original['excluded_sample_ids'],
        'home_joints_from_plan_snapshot': home_joints,
        'current_joints_from_runtime_snapshot': current_joints,
        'fixed_world_point_m': fixed_world_point[:3].tolist(),
        'baseline_reproduced_rmse_px': float(np.sqrt(np.mean(
            residual(initial, fit_poses, observed) ** 2))),
        'results': results, 'summaries': summaries,
        'elapsed_seconds': time.monotonic() - started,
        'limitations': [
            'No robot, ROS node, calibration deployment, or motion command is used.',
            'Resampling sensitivity is conditional on the recorded joint vectors and 19 mm board model being correct.',
            'Whole poses are resampled so 88 correlated corners are not treated as independent samples.',
            'All refits use the deployed result as initialization; this is local sensitivity, not a search over all possible minima.',
            'The inferred home camera point comes from the frozen planned world point and logged home joints, not a separately saved raw home RGB-D sample.',
            'The measured-current residual includes the small ideal-versus-real optical-frame difference documented by the frame audit.',
            'A distribution from 14 poses cannot bound unobserved kinematic biases outside the calibration workspace.',
            'No candidate result is selected or exported as a replacement calibration.',
        ],
    }
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'summaries': summaries, 'elapsed_seconds': output['elapsed_seconds'],
                      'output': str(args.output)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
