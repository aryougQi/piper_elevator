#!/usr/bin/env python3
"""Expand one held-out fold with pose, pixel, depth, and residual details."""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_handeye_model as m

INDICES=[1,2,3,4]

def load_samples(directory):
    rows=[json.loads(p.read_text()) for p in sorted(Path(directory).glob('sample_*.json'))]
    for s in rows:m.cache[id(s)]=(np.array(s['camera']['k']).reshape(3,3),np.array(s['camera']['d']),np.array(s['corners']))
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('fit_dir',type=Path);ap.add_argument('--fold',type=int,default=4);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args()
    samples=load_samples(a.fit_dir);train=[s for i,s in enumerate(samples) if i%5!=a.fold];held=[s for i,s in enumerate(samples) if i%5==a.fold]
    full=np.array(json.load((m.OUT/'model_hypotheses.json').open())['joint_2_3_4_5_offsets']['parameters'])
    lo=np.r_[np.full(12,-np.inf),np.full(4,-np.deg2rad(15))];hi=np.r_[np.full(12,np.inf),np.full(4,np.deg2rad(15))]
    sol=least_squares(lambda v:m.residual(v,train,INDICES),full,bounds=(lo,hi),max_nfev=2000,loss='huber',f_scale=2)
    X=m.params_to_transform(sol.x[:6]);B=m.params_to_transform(sol.x[6:12]);off=np.zeros(6);off[1:5]=sol.x[12:]
    details=[]
    for s in held:
        q=np.array(s['joints']);A=m.kin.forward('base_link','tcp_link',dict(zip(s['joint_names'],q+off)));k,d,uv=m.cache[id(s)];pred,depth=m.project_points(m.obj,A,X,B,k,d);e=pred-uv;cent_obs=uv.mean(0);cent_pred=pred.mean(0);cent_res=e.mean(0)
        details.append({'id':s['id'],'q2_q5_rad':q[1:5].tolist(),'q2_q5_deg':np.degrees(q[1:5]).tolist(),'pixel_centroid_observed':cent_obs.tolist(),'pixel_centroid_predicted':cent_pred.tolist(),'pixel_centroid_residual':cent_res.tolist(),'residual_component_rms_px':np.sqrt(np.mean(e**2,axis=0)).tolist(),'residual_norm_rms_px':float(np.sqrt(np.mean(e**2))),'residual_norm_median_px':float(np.median(np.linalg.norm(e,axis=1))),'depth_min_m':float(np.min(depth)),'depth_median_m':float(np.median(depth)),'depth_max_m':float(np.max(depth)),'approach_direction':'unknown: capture JSON stores no motion trajectory/direction'})
    report={'fold':a.fold,'held_out_ids':[s['id'] for s in held],'offset_degrees':(sol.x[12:]*180/np.pi).tolist(),'details':details}
    a.output_dir.mkdir(parents=True,exist_ok=True);(a.output_dir/'failure_fold_analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
