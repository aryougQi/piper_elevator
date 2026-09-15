#!/usr/bin/env python3
"""Offline hypotheses only; never export robot parameters or calibrated URDFs."""
import json, sys
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'handeye_calibration/scripts'))
from solve_handeye import UrdfKinematics,transform,params_to_transform,transform_to_params,project_points
DATA=Path(__file__).resolve().parents[1]/'data'
SRC=DATA/'handeye_recheck_20mm'
OUT=DATA/'handeye_recheck_20mm_analysis'
OUT.mkdir(exist_ok=True)
kin=UrdfKinematics(ROOT/'handeye_calibration/calibration/piper_handeye_input.urdf')
sets={name:[json.loads(f.read_text()) for f in sorted((SRC/name).glob('sample_*.json'))] for name in ['fit','validation','repeat']}
result=json.loads((SRC/'result/handeye_result.json').read_text())
x=result['tcp_to_camera_color_optical_frame']; b=result['base_to_checkerboard_first_corner']
initial=np.r_[transform_to_params(transform(x['xyz'],R.from_quat(x['quaternion_xyzw']).as_matrix())),transform_to_params(transform(b['xyz'],R.from_euler('xyz',b['rpy']).as_matrix()))]
obj=np.array([[c*.020,r*.020,0] for r in range(11) for c in range(8)])
cache={}
for group,samples in sets.items():
 for s in samples:
  k=np.array(s['camera']['k']).reshape(3,3);d=np.array(s['camera']['d']);uv=np.array(s['corners'])
  cache[id(s)]=(k,d,uv)

def residual(v,samples,joint_indices=(),board_scale=False):
 X=params_to_transform(v[:6]); B=params_to_transform(v[6:12]); offsets=np.zeros(6)
 for j,o in zip(joint_indices,v[12:]): offsets[j]=o
 points=obj.copy()
 if board_scale: points[:,:2]*=np.exp(v[12:14])
 parts=[]
 for s in samples:
  joints=np.array(s['joints'])+offsets
  A=kin.forward('base_link','tcp_link',dict(zip(s['joint_names'],joints)))
  k,d,uv=cache[id(s)]
  pred,depth=project_points(points,A,X,B,k,d)
  parts.append((pred-uv).ravel()+(1000 if np.any(depth<=.03) else 0))
 return np.concatenate(parts)

reports={}
def run(name,indices=(),scale=False):
 count=2 if scale else len(indices)
 start=np.r_[initial,np.zeros(count)]
 limit=.10 if scale else np.deg2rad(10)
 lower=np.r_[np.full(12,-np.inf),np.full(count,-limit)]
 upper=np.r_[np.full(12,np.inf),np.full(count,limit)]
 sol=least_squares(lambda v:residual(v,sets['fit'],indices,scale),start,bounds=(lower,upper),max_nfev=300,ftol=1e-9,xtol=1e-9,gtol=1e-7)
 report={'success':bool(sol.success),'nfev':sol.nfev,'parameters':sol.x.tolist(),'joint_indices_one_based':[j+1 for j in indices]}
 for group,samples in sets.items():
  errors=residual(sol.x,samples,indices,scale).reshape(len(samples),88,2)
  report[group+'_rmse_px']=float(np.sqrt(np.mean(errors**2)))
  report[group+'_per_sample_rmse_px']={s['id']:float(np.sqrt(np.mean(e**2))) for s,e in zip(samples,errors)}
 if indices: report['offset_degrees']={str(j+1):float(o*180/np.pi) for j,o in zip(indices,sol.x[12:])}
 if scale: report['square_xy_mm']=(20*np.exp(sol.x[12:14])).tolist()
 reports[name]=report
 (OUT/'model_hypotheses.json').write_text(json.dumps(reports,indent=2))
 print(name,json.dumps({k:v for k,v in report.items() if k.endswith('_rmse_px') and 'per_sample' not in k or k in ['offset_degrees','square_xy_mm']}),flush=True)
 return sol

if __name__=='__main__':
 run('baseline')
 for j in [1,2,3,4]: run('joint_'+str(j+1)+'_offset',[j])
 run('joint_2_3_4_5_offsets',[1,2,3,4])
 run('board_xy_scale',scale=True)
