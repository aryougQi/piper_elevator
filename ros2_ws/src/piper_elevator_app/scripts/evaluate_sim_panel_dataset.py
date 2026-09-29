#!/usr/bin/env python3
"""Evaluate the current ONNX detector against collected simulation labels."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import cv2

sys.path.insert(0, '/workspace/ros2_ws/src/piper_elevator_app')
from piper_elevator_app.detector_core import YoloOnnxDetector


CLASSES = [str(i) for i in range(1, 11)] + ['up', 'down', 'open', 'close']
# Same labeled subset as collect_sim_panel_dataset.py.
TARGETS = ('1', '2', '3', 'open', 'close', 'up', 'down')


def iou(first, second):
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    overlap = max(0, x2-x1) * max(0, y2-y1)
    area_a = max(0, first[2]-first[0]) * max(0, first[3]-first[1])
    area_b = max(0, second[2]-second[0]) * max(0, second[3]-second[1])
    return overlap / (area_a + area_b - overlap) if area_a + area_b > overlap else 0


def ground_truth(path, width, height):
    result = []
    for line in path.read_text().splitlines():
        values = line.split()
        cls = CLASSES[int(values[0])]
        x, y, w, h = map(float, values[1:])
        result.append((cls, ((x-w/2)*width, (y-h/2)*height,
                             (x+w/2)*width, (y+h/2)*height)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('/workspace/sim-dataset/Raw'))
    parser.add_argument('--model', type=Path, default=Path(
        '/workspace/ros2_ws/src/piper_elevator_app/models/elevator_buttons_yolo11s.onnx'))
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--limit-per-scene', type=int, default=0)
    parser.add_argument('--output', type=Path, default=Path('/workspace/sim-dataset/evaluation.json'))
    args = parser.parse_args()
    detector = YoloOnnxDetector(str(args.model), ['__model_metadata__'], ['*'],
                                confidence_threshold=args.threshold,
                                input_size=640, inference_device='cpu')
    totals = defaultdict(Counter)
    by_light = defaultdict(lambda: defaultdict(Counter))
    confusion = Counter()
    frames = 0
    for run in sorted(args.raw.glob('20*')):
        manifest_file = run / 'manifest.json'
        if not manifest_file.is_file():
            continue
        manifest = json.loads(manifest_file.read_text())
        light = manifest.get('scene_id', '')
        if light not in {f'L{i}' for i in range(6)}:
            continue
        selected_per_scene = Counter()
        for sample in manifest['samples']:
            if not sample['saved']:
                continue
            scene = sample['name'].rsplit('_j1_', 1)[0]
            if args.limit_per_scene and selected_per_scene[scene] >= args.limit_per_scene:
                continue
            selected_per_scene[scene] += 1
            stem = sample['name']
            image = cv2.imread(str(run / 'images' / f'{stem}.png'))
            if image is None:
                raise FileNotFoundError(stem)
            height, width = image.shape[:2]
            truth = ground_truth(run / 'labels' / f'{stem}.txt', width, height)
            predictions = detector.infer(image)
            matched = set()
            for cls, box in truth:
                totals[cls]['ground_truth'] += 1
                by_light[light][cls]['ground_truth'] += 1
                candidates = [(iou(box, (d.x1,d.y1,d.x2,d.y2)), index, d)
                              for index, d in enumerate(predictions)
                              if index not in matched and d.class_name == cls]
                best = max(candidates, default=(0, -1, None))
                if best[0] >= 0.5:
                    matched.add(best[1])
                    confusion[(cls, cls)] += 1
                    totals[cls]['correct'] += 1
                    by_light[light][cls]['correct'] += 1
                    continue
                # A box at the right location with another label is a
                # classification error, not a correct class match.
                others = [(iou(box, (d.x1,d.y1,d.x2,d.y2)), index, d)
                          for index, d in enumerate(predictions)
                          if index not in matched]
                best = max(others, default=(0, -1, None))
                if best[0] >= 0.5:
                    matched.add(best[1])
                    confusion[(cls, best[2].class_name)] += 1
            for index, pred in enumerate(predictions):
                if index not in matched:
                    totals[pred.class_name]['unmatched_predictions'] += 1
                    by_light[light][pred.class_name]['unmatched_predictions'] += 1
            frames += 1
            if frames % 100 == 0:
                print(f'Evaluated {frames} images', flush=True)
    report = {'frames': frames, 'threshold': args.threshold, 'iou_threshold': 0.5,
              'per_class': {key: dict(totals[key]) for key in TARGETS},
              'by_light': {light: {key: dict(counts[key]) for key in TARGETS}
                           for light, counts in by_light.items()},
              'confusion': [{'actual': actual, 'predicted': predicted, 'count': count}
                            for (actual, predicted), count in confusion.most_common()]}
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['per_class'], indent=2), flush=True)
    print(f'Report: {args.output}', flush=True)


if __name__ == '__main__':
    main()
