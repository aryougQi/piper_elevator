#!/usr/bin/env python3
"""Summarize recorded simulation evidence without assuming success from node startup."""
import argparse
import json
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('evidence',type=Path)
parser.add_argument('--output',type=Path)
a=parser.parse_args()
runs=[];run=None
for line in a.evidence.read_text().splitlines():
    row=json.loads(line);kind=row['kind'];data=row.get('data','')
    if kind=='/elevator_task/status' and data.startswith('WAITING_FOR_NODES'):
        run=dict(start_wall=row['wall'],button=data.split('button=')[-1],projections=[],pose_stamps=[],ready_wall=None,servo_wall=None,blind=False,result=None)
        runs.append(run)
    if run is None:continue
    if kind=='projection':run['projections'].append(row)
    if kind=='sam2_pose' and run['servo_wall'] is None:run['pose_stamps'].append(row['stamp'])
    if kind=='/button_tracking_state':
        payload=json.loads(data)
        if payload.get('source')=='sam2_button_tracker' and payload.get('reason')=='tracker_ready' and run['ready_wall'] is None:
            run['ready_wall']=row['wall']
    if kind=='/elevator_task/status' and data.startswith('VISUAL_SERVO') and run['servo_wall'] is None:run['servo_wall']=row['wall']
    if kind=='/button_visual_servo/status' and ('VISION_LOSS_CONTINUING' in data or 'WITH_LOCKED_TARGET' in data):run['blind']=True
    if kind=='/elevator_task/result' and (data.startswith('COMPLETE:') or data.startswith('FAILED:')):run['result']=data
report=[]
for r in runs:
    projections=r.pop('projections');stamps=r.pop('pose_stamps')
    r.update(actual_coarse_samples=len(projections),minimum_fov_margin_px=min((p['margin'] for p in projections),default=None),minimum_depth_m=min((p['z'] for p in projections),default=None),unique_sam2_frames_before_servo=len(set(stamps)))
    r['strict_order_verified']=r['ready_wall'] is not None and r['servo_wall'] is not None and r['ready_wall']<=r['servo_wall'] and r['unique_sam2_frames_before_servo']>=5
    r['fov_verified']=bool(projections) and r['minimum_fov_margin_px']>=60 and r['minimum_depth_m']>0
    report.append(r)
text=json.dumps(report,ensure_ascii=False,indent=2)
if a.output:a.output.write_text(text+'\n')
print(text)
