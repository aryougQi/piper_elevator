"""Compare isolated geometry candidates on recorded data; no ROS nodes or motion."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import cv2
import numpy as np

from piper_elevator_app.detector_core import Detection, estimate_surface_normal
from piper_elevator_app.experimental_button_geometry import (
    estimate_sampled_surface, measure_button_point, panel_sampling_mask,
    refine_rectangular_center,
)


def xyxy(box):
    x, y, width, height = box
    return np.array([x-width/2, y-height/2, x+width/2, y+height/2])


def camera_arguments(metadata):
    camera = metadata['camera_info']
    return dict(camera_matrix=np.asarray(camera['k']).reshape(3, 3),
                distortion=np.asarray(camera['d']),
                distortion_model=camera['distortion_model'])


def compare(path, output_dir):
    result = {'capture': str(path), 'methods': {}}
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata_json']))
        camera = camera_arguments(metadata)
        parameters = metadata['detector_parameters']
        all_boxes = json.loads(str(data['all_boxes_json'])) if 'all_boxes_json' in data else None
        result['all_detected_buttons_excluded'] = all_boxes is not None
        # A saved color image is usable only for its exactly matching depth/box.
        # Never reuse it for other frames in a depth-only recording.
        color_path = path.with_name(path.stem+'_color.png')
        color = cv2.imread(str(color_path)) if color_path.exists() else None
        color_stamp = metadata.get('color_stamp_ns')
        result['center'] = {'reason': 'no_synchronized_color_frame'}
        for method in ('legacy', 'button_ransac', 'panel_ransac'):
            normals, elapsed, reasons, valid_indices = [], [], Counter(), []
            for index, (depth, box, rotation) in enumerate(zip(data['depths'], data['boxes'], data['rotations'])):
                bounds = xyxy(box)
                detection = Detection(*bounds, 1.0, 0, metadata['selected_button'])
                started = time.perf_counter()
                if method == 'legacy':
                    normal = estimate_surface_normal(
                        depth, detection, camera['camera_matrix'],
                        unit_scale=parameters['depth_unit_scale'],
                        inner_ratio=parameters['surface_inner_ratio'],
                        min_depth_m=parameters['min_depth_m'], max_depth_m=parameters['max_depth_m'],
                        min_samples=parameters['surface_minimum_samples'],
                        max_samples=parameters['surface_maximum_samples'],
                        max_residual_m=parameters['surface_max_residual_m'],
                        max_tilt_degrees=parameters['surface_max_tilt_degrees'],
                        distortion_coefficients=camera['distortion'], distortion_model=camera['distortion_model'],
                    )
                    reason = 'accepted' if normal is not None else 'rejected'
                else:
                    if method == 'panel_ransac':
                        mask = panel_sampling_mask(depth.shape, bounds, all_boxes[index] if all_boxes is not None else [])
                    else:
                        yy, xx = np.ogrid[:depth.shape[0], :depth.shape[1]]
                        half = (bounds[2:]-bounds[:2])*parameters['surface_inner_ratio']/2
                        mask = (abs(xx-box[0]) <= half[0]) & (abs(yy-box[1]) <= half[1])
                    estimate = estimate_sampled_surface(
                        depth, mask, box[:2], **camera,
                        threshold_m=.0015 if method == 'panel_ransac' else parameters['surface_max_residual_m'],
                        minimum_samples=60 if method == 'panel_ransac' else parameters['surface_minimum_samples'],
                        max_samples=800 if method == 'panel_ransac' else parameters['surface_maximum_samples'],
                        require_surrounding=method == 'panel_ransac',
                        unit_scale=parameters['depth_unit_scale'],
                        min_depth_m=parameters['min_depth_m'], max_depth_m=parameters['max_depth_m'],
                        max_tilt_degrees=parameters['surface_max_tilt_degrees'],
                    )
                    normal = None if estimate.plane is None else estimate.plane.normal
                    reason = estimate.reason
                elapsed.append((time.perf_counter()-started)*1000)
                reasons[reason] += 1
                if normal is not None:
                    normals.append(rotation @ normal)
                    valid_indices.append(index)
                if method == 'panel_ransac' and color is not None and int(data['stamps'][index]) == color_stamp:
                    center = refine_rectangular_center(color, bounds)
                    point = measure_button_point(depth, center, **camera,
                                                  unit_scale=parameters['depth_unit_scale'])
                    result['center'] = dict(
                        reason=center.reason, frame_index=index, stamp_ns=color_stamp,
                        old_center=list(map(float, box[:2])),
                        refined_center=None if center.center is None else center.center.tolist(),
                        button_point=None if point is None else point.tolist(),
                        candidates=center.candidates,
                    )
                    overlay = color.copy()
                    overlay[mask] = (overlay[mask]*.55 + np.array([0, 180, 0])*.45).astype(np.uint8)
                    cv2.rectangle(overlay, tuple(bounds[:2].astype(int)), tuple(bounds[2:].astype(int)), (255, 180, 0), 1)
                    cv2.drawMarker(overlay, tuple(np.rint(box[:2]).astype(int)), (255, 0, 0), cv2.MARKER_CROSS, 10, 1)
                    if center.center is not None:
                        cv2.polylines(overlay, [center.corners.astype(np.int32)], True, (0, 255, 255), 1)
                        cv2.drawMarker(overlay, tuple(np.rint(center.center).astype(int)), (0, 0, 255), cv2.MARKER_CROSS, 10, 1)
                    label = f'center: {center.reason}; panel: {estimate.reason}'
                    cv2.putText(overlay, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3)
                    cv2.putText(overlay, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
                    cv2.imwrite(str(output_dir/(path.stem+'_overlay.png')), overlay)
                    cv2.imwrite(str(output_dir/(path.stem+'_mask.png')), mask.astype(np.uint8)*255)
            spread = None
            if normals:
                normals = np.asarray(normals)
                center = normals.mean(axis=0)
                center /= np.linalg.norm(center)
                spread = float(np.percentile(np.degrees(np.arccos(np.clip(normals @ center, -1, 1))), 95))
            result['methods'][method] = dict(
                frames=len(data['depths']), valid=len(normals),
                normal_spread_p95_degrees=spread, median_ms=float(np.median(elapsed)),
                reasons=dict(reasons), valid_frame_indices=valid_indices,
            )
    return result


def main():
    data_dir = Path(__file__).resolve().parents[1] / 'data'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures', type=Path, nargs='*')
    parser.add_argument('--output', type=Path,
                        default=data_dir/'button_geometry_v2_20260913/results.json')
    args = parser.parse_args()
    captures = args.captures or [data_dir/name for name in (
        'vision_stability_current.npz', 'surface_support_diagnostic.npz',
        'vision_stability_full_context.npz')]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = [compare(path, args.output.parent) for path in captures]
    args.output.write_text(json.dumps(results, indent=2)+'\n')
    for result in results:
        print(Path(result['capture']).name, result['center'])
        for method, metrics in result['methods'].items():
            print(method, {k: v for k, v in metrics.items() if k != 'valid_frame_indices'})


if __name__ == '__main__':
    main()
