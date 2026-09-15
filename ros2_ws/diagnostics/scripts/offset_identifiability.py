#!/usr/bin/env python3
"""Jacobian conditioning and bootstrap stability for joint offset estimates."""
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
def fit(rows, initial):
 lo=np.r_[np.full(12,-np.inf),np.full(4,-np.deg2rad(15))];hi=np.r_[np.full(12,np.inf),np.full(4,np.deg2rad(15))]
 return least_squares(lambda v:m.residual(v,rows,INDICES),initial,bounds=(lo,hi),max_nfev=1500,loss='huber',f_scale=2)
def cond(sol):
 s=np.linalg.svd(sol.jac,compute_uv=False);return {'singular_values':s.tolist(),'condition_number':float(s[0]/s[-1]),'offset_degrees':(sol.x[12:]*180/np.pi).tolist()}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('fit_dir',type=Path);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--bootstrap',type=int,default=100);a=ap.parse_args();rows=load(a.fit_dir);initial=np.r_[m.initial,np.zeros(4)];sol=fit(rows,initial);out={'full_fit':cond(sol),'five_fold':[],'bootstrap_count':a.bootstrap}
 for fold in range(5):
  tr=[s for i,s in enumerate(rows) if i%5!=fold];out['five_fold'].append({'held_out':[s['id'] for i,s in enumerate(rows) if i%5==fold],**cond(fit(tr,sol.x))})
 rng=np.random.default_rng(20260911);boots=[]
 for _ in range(a.bootstrap):boots.append(fit([rows[i] for i in rng.integers(0,len(rows),len(rows))],sol.x).x[12:]*180/np.pi)
 b=np.array(boots);out['bootstrap_offset_degrees']={'median':np.median(b,0).tolist(),'p05':np.percentile(b,5,0).tolist(),'p95':np.percentile(b,95,0).tolist(),'std':np.std(b,0).tolist()}
 a.output_dir.mkdir(parents=True,exist_ok=True);(a.output_dir/'offset_identifiability.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
