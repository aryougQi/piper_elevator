#!/usr/bin/env python3
"""Offline five-fold cross-check; requires diagnose_handeye_model.py outputs."""
import sys,json,numpy as np
from scipy.optimize import least_squares
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import diagnose_handeye_model as m
full=json.loads((m.OUT/'model_hypotheses.json').read_text())['joint_2_3_4_5_offsets']['parameters']
indices=[1,2,3,4]; rows=[]
for fold in range(5):
 train=[s for i,s in enumerate(m.sets['fit']) if i%5!=fold]; test=[s for i,s in enumerate(m.sets['fit']) if i%5==fold]
 sol=least_squares(lambda v:m.residual(v,train,indices),full,bounds=(np.r_[np.full(12,-np.inf),np.full(4,-np.deg2rad(10))],np.r_[np.full(12,np.inf),np.full(4,np.deg2rad(10))]),max_nfev=200)
 row={'held_out':[s['id'] for s in test],'offset_degrees':(sol.x[12:]*180/np.pi).tolist()}
 for label,samples in [('train',train),('held_out',test),('validation',m.sets['validation'])]: row[label+'_rmse_px']=float(np.sqrt(np.mean(m.residual(sol.x,samples,indices)**2)))
 rows.append(row); print(json.dumps(row),flush=True)
(m.OUT/'cross_validation.json').write_text(json.dumps(rows,indent=2))
# Inspect single-frame PnP board constancy under baseline and offset model.
from scipy.spatial.transform import Rotation as R
import cv2
summary={}
for name in ['baseline','joint_2_3_4_5_offsets']:
 v=np.array(json.loads((m.OUT/'model_hypotheses.json').read_text())[name]['parameters']); X=m.params_to_transform(v[:6]); offsets=np.zeros(6)
 if len(v)>12: offsets[1:5]=v[12:]
 stats={}
 for group,samples in m.sets.items():
  poses=[]
  for s in samples:
   k,d,uv=m.cache[id(s)];_,r,t=cv2.solvePnP(m.obj,uv,k,d)
   A=m.kin.forward('base_link','tcp_link',dict(zip(s['joint_names'],np.array(s['joints'])+offsets)))
   poses.append(A@X@m.transform(t.ravel(),R.from_rotvec(r.ravel()).as_matrix()))
  ref=m.params_to_transform(v[6:12]); stats[group]=[{'id':s['id'],'board_translation_error_mm':float(np.linalg.norm(p[:3,3]-ref[:3,3])*1000),'board_rotation_error_deg':float(R.from_matrix(ref[:3,:3].T@p[:3,:3]).magnitude()*180/np.pi)} for s,p in zip(samples,poses)]
 summary[name]=stats
(m.OUT/'board_consistency.json').write_text(json.dumps(summary,indent=2))
print('DONE')
