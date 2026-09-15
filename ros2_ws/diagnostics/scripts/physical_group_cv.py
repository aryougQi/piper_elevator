#!/usr/bin/env python3
"""Leave-one-physical-group-out validation for joint-offset hand-eye fits."""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_handeye_model as m

INDICES=[1,2,3,4]

def load(directory):
    rows=[json.loads(p.read_text()) for p in sorted(Path(directory).glob('sample_*.json'))]
    for s in rows:m.cache[id(s)]=(np.array(s['camera']['k']).reshape(3,3),np.array(s['camera']['d']),np.array(s['corners']))
    return rows

def fit(train, initial):
    lo=np.r_[np.full(12,-np.inf),np.full(4,-np.deg2rad(15))];hi=np.r_[np.full(12,np.inf),np.full(4,np.deg2rad(15))]
    return least_squares(lambda v:m.residual(v,train,INDICES),initial,bounds=(lo,hi),max_nfev=1500,loss='huber',f_scale=2)

def score(v, rows): return float(np.sqrt(np.mean(m.residual(v,rows,INDICES)**2)))

def q_bins(rows, index):
    values=np.array([s['joints'][index] for s in rows]); edges=np.quantile(values,[0,.5,1]);
    return [([i for i,x in enumerate(values) if x <= edges[1]], {'axis':f'joint{index+1}','range_rad':[float(edges[0]),float(edges[1])]}) ,([i for i,x in enumerate(values) if x > edges[1]], {'axis':f'joint{index+1}','range_rad':[float(edges[1]),float(edges[2])]})]

def image_regions(rows):
    centers=np.array([np.mean(s['corners'],axis=0) for s in rows]);mx=np.median(centers[:,0]);my=np.median(centers[:,1]);
    groups=[]
    for label,mask in [('left_top',(centers[:,0]<=mx)&(centers[:,1]<=my)),('left_bottom',(centers[:,0]<=mx)&(centers[:,1]>my)),('right_top',(centers[:,0]>mx)&(centers[:,1]<=my)),('right_bottom',(centers[:,0]>mx)&(centers[:,1]>my))]:
        groups.append(([i for i,x in enumerate(mask) if x],{'region':label,'pixel_median_xy':[float(mx),float(my)]}))
    return groups

def main():
    ap=argparse.ArgumentParser();ap.add_argument('fit_dir',type=Path);ap.add_argument('--validation-dir',action='append',type=Path,default=[]);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args()
    rows=load(a.fit_dir); validation=[x for d in a.validation_dir for x in load(d)]; initial=np.r_[m.initial,np.zeros(4)]
    groups=[]
    for j in [1,2,3,4]: groups.append((f'leave_one_joint{j+1}_half',q_bins(rows,j)))
    groups.append(('leave_one_image_region',image_regions(rows)))
    report={'approach_direction_status':'unavailable: persisted JSON has no commanded trajectory or direction label','groups':{}}
    for name,definitions in groups:
        out=[]
        for indices,meta in definitions:
            if len(indices)<2 or len(indices)==len(rows):continue
            test=[s for i,s in enumerate(rows) if i in indices];train=[s for i,s in enumerate(rows) if i not in indices];sol=fit(train,initial)
            out.append({**meta,'held_out':[s['id'] for s in test],'train_rmse_px':score(sol.x,train),'held_out_rmse_px':score(sol.x,test),'validation_rmse_px':None if not validation else score(sol.x,validation),'offset_degrees':(sol.x[12:]*180/np.pi).tolist()})
        report['groups'][name]=out
    a.output_dir.mkdir(parents=True,exist_ok=True);(a.output_dir/'physical_group_cv.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
