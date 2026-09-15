#!/usr/bin/env python3
"""Compare saved button images at fixed crop scales using CPU-only ONNX."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime

from piper_elevator_app.detector_core import (
    Detection, YoloOnnxDetector, extract_padded_square_crop,
    intersection_over_union, remap_crop_detections,
)


DIAGNOSTICS = Path(__file__).resolve().parents[1]
NEAR_OBJECTS = {
    'true_up': {'box': [450, 307, 489, 349], 'expected_label': 'up'},
    'screw': {'box': [451, 252, 466, 267], 'expected_label': None},
    'ceiling_light': {'box': [190, 55, 236, 101], 'expected_label': None},
}
PANEL_OBJECTS = {
    'top_screw': {'box': [410, 17, 422, 30], 'expected_label': None},
    'bottom_screw': {'box': [405, 158, 419, 172], 'expected_label': None},
    'up': {'box': [398, 58, 431, 92], 'expected_label': 'up'},
    'down': {'box': [397, 103, 430, 136], 'expected_label': 'down'},
    'alarm': {'box': [361, 207, 405, 246], 'expected_label': 'alarm'},
    'call': {'box': [428, 205, 474, 244], 'expected_label': 'call'},
    'floor_3': {'box': [395, 248, 440, 290], 'expected_label': '3'},
    'floor_2': {'box': [394, 295, 441, 340], 'expected_label': '2'},
    'floor_1': {'box': [394, 345, 441, 387], 'expected_label': '1'},
}


def as_record(detection):
    return {
        'label': detection.class_name, 'score': float(detection.confidence),
        'box_xyxy': [detection.x1, detection.y1, detection.x2, detection.y2],
        'center_px': list(detection.center),
        'size_px': [detection.width, detection.height],
    }


def intersection_area(first, second):
    return max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0, min(first[3], second[3]) - max(first[1], second[1]),
    )


def associate(detections, reference, observed_bounds, production_threshold):
    box = reference['box']
    area = (box[2] - box[0]) * (box[3] - box[1])
    visible = intersection_area(box, observed_bounds) / area
    target = Detection(*box, 1.0, 0, '')
    matches = sorted(
        [item for item in detections if intersection_over_union(item, target) >= 0.15],
        key=lambda item: intersection_over_union(item, target), reverse=True,
    )
    kept = [item for item in matches if item.confidence >= production_threshold]
    primary = kept[0] if kept else None
    if visible < 0.5:
        status = 'outside_or_mostly_clipped'
    elif primary is None:
        status = 'not_reported'
    elif reference['expected_label'] is None:
        status = 'nonbutton_reported_as_button'
    elif primary.class_name == reference['expected_label']:
        status = 'expected_label_reported'
    else:
        status = 'different_label_reported'
    return {
        'reference_box_xyxy': box, 'expected_label': reference['expected_label'],
        'visible_fraction': visible, 'status_at_production_threshold': status,
        'primary_at_production_threshold': as_record(primary) if primary else None,
        'matches_above_diagnostic_floor': [as_record(item) for item in matches],
    }


def summarize(records, scene, object_name, samples_only=False):
    rows = [row for row in records if row['scene'] == scene
            and (not samples_only or row['image_name'].startswith('sample_'))]
    result = {}
    for mode in ['full', 'crop_160', 'crop_240', 'crop_320', 'crop_480']:
        group = [row for row in rows if row['mode'] == mode]
        if not group:
            continue
        observations = [row['objects'][object_name] for row in group]
        kept = [item['primary_at_production_threshold'] for item in observations
                if item['primary_at_production_threshold'] is not None]
        stats = {
            'frames': len(group), 'reported_frames': len(kept),
            'status_counts': dict(Counter(
                item['status_at_production_threshold'] for item in observations
            )),
            'label_counts': dict(Counter(item['label'] for item in kept)),
        }
        if kept:
            scores = np.asarray([item['score'] for item in kept])
            centers = np.asarray([item['center_px'] for item in kept])
            sizes = np.asarray([item['size_px'] for item in kept])
            stats.update({
                'score_min_median_max': [float(np.min(scores)), float(np.median(scores)),
                                         float(np.max(scores))],
                'center_mean_px': np.mean(centers, axis=0).tolist(),
                'center_range_px': np.ptp(centers, axis=0).tolist(),
                'size_median_px': np.median(sizes, axis=0).tolist(),
                'size_range_px': np.ptp(sizes, axis=0).tolist(),
            })
        result[mode] = stats
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DIAGNOSTICS / 'data')
    parser.add_argument('--output', type=Path,
                        default=DIAGNOSTICS / 'data/button_stability_projection/roi_comparison.json')
    parser.add_argument('--model', type=Path, default=(
        DIAGNOSTICS.parent / 'src/piper_elevator_app/models/elevator_buttons_yolov10s.onnx'
    ))
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--diagnostic-floor', type=float, default=0.10)
    parser.add_argument('--production-threshold', type=float, default=0.40)
    parser.add_argument('--reuse-existing', action='store_true',
                        help='Reuse matching saved model outputs while extending this comparison')
    args = parser.parse_args()
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = onnxruntime.InferenceSession(
        str(args.model), sess_options=options, providers=['CPUExecutionProvider'],
    )
    model = YoloOnnxDetector(
        str(args.model), ['__model_metadata__'], ['*'],
        confidence_threshold=args.diagnostic_floor, nms_iou_threshold=0.45,
        input_size=1280, inference_device='cpu', session=session,
    )
    if session.get_providers() != ['CPUExecutionProvider']:
        raise RuntimeError('This diagnostic must not use the live GPU')
    model_hash = hashlib.sha256(args.model.read_bytes()).hexdigest()
    previous, previous_hashes = {}, {}
    if args.reuse_existing and args.output.exists():
        saved = json.loads(args.output.read_text())
        if (saved['model_sha256'] != model_hash
                or saved['diagnostic_confidence_floor'] != args.diagnostic_floor):
            raise RuntimeError('Cached comparison used a different model or detection floor')
        previous_hashes = saved['image_pixel_sha256']
        previous = {
            (row['image_relative_to_data'], row['mode'], tuple(row['crop_center_px'])): row
            for row in saved['records']
        }
    near = args.data_dir / 'button_stability_projection'
    scenes = [
        ('near_up', near / name, (469, 328), NEAR_OBJECTS, 'true_up', [None, 160, 240, 320, 480])
        for name in ['color.png'] + [f'sample_{index}.png' for index in range(4)]
    ]
    panel_image = args.data_dir / 'vision_stability_full_context_color.png'
    scenes += [
        ('panel_floor_2', panel_image, (417, 317), PANEL_OBJECTS, 'floor_2', [None, 160, 240, 320, 480]),
        ('panel_alarm', panel_image, (383, 227), PANEL_OBJECTS, 'alarm', [None, 160, 240, 320, 480]),
        ('panel_down', panel_image, (413, 119), PANEL_OBJECTS, 'down', [None, 160, 240, 320, 480]),
        ('panel_alarm', args.data_dir / 'vision_stability_current_color.png',
         (383, 227), PANEL_OBJECTS, 'alarm', [None, 240, 320]),
        ('panel_alarm', args.data_dir / 'surface_support_diagnostic_color.png',
         (383, 227), {key: value for key, value in PANEL_OBJECTS.items()
                     if key not in {'up', 'down', 'top_screw', 'bottom_screw'}},
         'alarm', [None, 240, 320]),
    ]
    records, hashes, full_cache = [], {}, {}
    started = time.monotonic()
    for scene, image_path, center, objects, selected, scales in scenes:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f'Cannot read diagnostic image: {image_path}')
        height, width = image.shape[:2]
        pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
        relative_path = str(image_path.relative_to(args.data_dir))
        hashes[relative_path] = pixel_hash
        first_record = len(records)
        for size in scales:
            mode = 'full' if size is None else f'crop_{size}'
            inference_started = time.monotonic()
            cached = previous.get((relative_path, mode, tuple(center)))
            if cached and previous_hashes.get(relative_path) == pixel_hash:
                detections = [Detection(*row['box_xyxy'], row['score'], 0, row['label'])
                              for row in cached['detections']]
                bounds = cached['observed_bounds_xyxy']
                origin, padding = cached['crop_origin_px'], cached['padded_fraction']
            elif size is None:
                if pixel_hash not in full_cache:
                    full_cache[pixel_hash] = model.infer(image)
                detections = full_cache[pixel_hash]
                bounds, origin, padding = [0, 0, width, height], [0, 0], 0.0
            else:
                crop, x0, y0 = extract_padded_square_crop(image, center, size)
                detections = remap_crop_detections(
                    model.infer(crop), x0, y0, width, height,
                )
                bounds = [max(0, x0), max(0, y0), min(width, x0 + size), min(height, y0 + size)]
                origin = [x0, y0]
                padding = 1.0 - (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) / size ** 2
            associations = {
                name: associate(detections, probe, bounds, args.production_threshold)
                for name, probe in objects.items()
            }
            records.append({
                'scene': scene, 'image_name': image_path.name,
                'image_relative_to_data': str(image_path.relative_to(args.data_dir)),
                'mode': mode, 'selected_object': selected, 'crop_center_px': list(center),
                'crop_origin_px': origin, 'observed_bounds_xyxy': bounds,
                'padded_fraction': padding, 'objects': associations,
                'detections': [as_record(item) for item in detections],
                'reused_saved_inference': bool(cached and previous_hashes.get(relative_path) == pixel_hash),
                'processing_ms': (cached['processing_ms']
                                  if cached and previous_hashes.get(relative_path) == pixel_hash
                                  else 1000 * (time.monotonic() - inference_started)),
            })
        print(json.dumps({
            'completed': scene, 'image': image_path.name,
            'selected': {row['mode']: row['objects'][selected]['primary_at_production_threshold']
                         for row in records[first_record:]},
            'screw': {row['mode']: row['objects'].get('screw', {}).get(
                'primary_at_production_threshold') for row in records[first_record:]},
        }), flush=True)
    summaries = {
        'near_up_four_frame_target': summarize(records, 'near_up', 'true_up', samples_only=True),
        'near_up_four_frame_screw': summarize(records, 'near_up', 'screw', samples_only=True),
        'panel_floor_2': summarize(records, 'panel_floor_2', 'floor_2'),
        'panel_alarm': summarize(records, 'panel_alarm', 'alarm'),
        'panel_down': summarize(records, 'panel_down', 'down'),
    }
    neighbor_findings = [
        {
            'scene': row['scene'], 'image_relative_to_data': row['image_relative_to_data'],
            'mode': row['mode'], 'object': name,
            'status': probe['status_at_production_threshold'],
            'visible_fraction': probe['visible_fraction'],
            'detection': probe['primary_at_production_threshold'],
        }
        for row in records for name, probe in row['objects'].items()
        if probe['status_at_production_threshold'] in {
            'nonbutton_reported_as_button', 'different_label_reported',
        }
    ]
    output = {
        'offline': True, 'providers': session.get_providers(),
        'intra_op_threads': args.threads, 'model_input_size': 1280,
        'model_sha256': model_hash,
        'image_pixel_sha256': hashes,
        'diagnostic_confidence_floor': args.diagnostic_floor,
        'unchanged_production_threshold_for_reporting': args.production_threshold,
        'summaries': summaries, 'neighbor_findings': neighbor_findings, 'records': records,
        'elapsed_seconds': time.monotonic() - started,
        'limitations': [
            'Saved images only; no ROS node, live camera, GPU inference or hardware command is used.',
            'The four near-up samples assess fixed-view repeatability, not detection accuracy.',
            'Reference object boxes were assigned by visual inspection, not precision annotation.',
            'Association uses geometric overlap independent of predicted class and score.',
            'Extra panel classes have one archived view, with two additional archived alarm views; these compare scales, not temporal accuracy.',
            'Model probabilities are not calibrated accuracy; a high-scoring screw remains a false positive.',
            'Only native detector output is evaluated; temporal state and semantic recovery are excluded.',
            'Crop center is fixed to isolate scale effects; moving-camera tracking still needs validation.',
            'Full-frame detections are cached for repeated analysis of the same image.',
        ],
    }
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
