#!/usr/bin/env python3
"""Summarize measured ROS captures, keeping wall time distinct from simulation time."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(path):
    data = json.loads(path.read_text())
    rows = data['stream_records']['/sam2_button_tracker/center']
    start, end = rows[0][0], rows[-1][0]
    steady = data['performance'][1:] or data['performance']
    def weighted(key, rate='mask_fps'):
        valid = [p for p in steady if p.get(key) is not None]
        weights = [p['window_seconds']*p[rate] for p in valid]
        return sum(p[key]*w for p,w in zip(valid,weights))/sum(weights) if sum(weights) else None
    gpu = [r[1] for r in data['gpu_samples'] if start <= r[0] <= end]
    # Exclude initialization/first five masks and one already running YOLO call.
    yolo = [r for r in data['stream_records']['/button_detections'] if start+2 < r[0] < end-.2]
    states = []
    for t,topic,text in data['events']:
        if topic == '/button_tracking_state':
            try:
                event = json.loads(text)
                if event.get('source') == 'sam2_button_tracker':
                    states.append(event)
            except ValueError:
                pass
    return dict(
        file=str(path), all_tracking_streams=data['streams'],
        steady_mask_fps=sum(p['mask_fps']*p['window_seconds'] for p in steady)/sum(p['window_seconds'] for p in steady),
        steady_inference_ms=weighted('inference_ms'),
        steady_geometry_fps=sum(p['geometry_fps']*p['window_seconds'] for p in steady)/sum(p['window_seconds'] for p in steady),
        steady_target_latency_ms=weighted('target_latency_ms','geometry_fps'),
        steady_target_age_ros_ms=weighted('target_age_ros_ms','geometry_fps'),
        mask_stamp_duplicates=len(rows)-len(set(r[1] for r in rows)),
        mask_stamp_regressions=sum(b[1]<=a[1] for a,b in zip(rows,rows[1:])),
        tracking_gpu_utilization_mean=statistics.mean(gpu) if gpu else None,
        yolo_outputs_during_steady_tracking=len(yolo),
        sam2_lost_events=sum(s['state']=='LOST' for s in states),
        camera_receipt_to_target_ms=data['camera_receipt_to_target_ms'],
        rtf=data['rtf'],
        task_results=[text for _,topic,text in data['events'] if topic=='/elevator_task/result'],
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = [summarize(path) for path in args.captures]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
