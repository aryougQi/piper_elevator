"""Inspect spatial depth noise before changing the independent estimator."""
from pathlib import Path
import json
import numpy as np
from piper_elevator_app.experimental_button_geometry import panel_sampling_mask, camera_rays
from piper_elevator_app.plane_core import fit_plane_consensus
root=Path(__file__).resolve().parents[1]/'data'
for name in ('vision_stability_current','surface_support_diagnostic','vision_stability_full_context'):
 with np.load(root/(name+'.npz')) as d:
  m=json.loads(str(d['metadata_json'])); k=np.array(m['camera_info']['k']).reshape(3,3); dist=m['camera_info']['d']; other=json.loads(str(d['all_boxes_json'])) if 'all_boxes_json' in d else None
  for tile in (6,10):
   normals=[]; counts=[]; raw=[]
   for i in range(0,len(d['depths']),5):
    x,y,w,h=d['boxes'][i]; z=d['depths'][i]*.001; mask=panel_sampling_mask(z.shape,[x-w/2,y-h/2,x+w/2,y+h/2],other[i] if other else [])
    yy,xx=np.nonzero(mask & (z>.1)&(z<2)); keys=(yy//tile)*10000+xx//tile; pts=[]
    for key in np.unique(keys):
     use=keys==key; v=z[yy[use],xx[use]]
     if len(v)<tile*tile*.5:continue
     if np.percentile(v,90)-np.percentile(v,10)>.012:continue
     uv=np.array([[np.median(xx[use]),np.median(yy[use])]])
     pts.append(camera_rays(uv,k,dist,'plumb_bob')[0]*np.median(v))
    if len(pts)<12:continue
    fit=fit_plane_consensus(pts,threshold_m=.0015,minimum_samples=12)
    counts.append(len(pts))
    if fit is not None:
     normals.append(d['rotations'][i]@fit.normal)
   if normals:
    n=np.array(normals);mean=n.mean(0);mean/=np.linalg.norm(mean);spread=np.percentile(np.degrees(np.arccos(np.clip(n@mean,-1,1))),95)
   else:spread=None
   print(name,tile,'accepted',len(normals),'/',len(counts),'spread',spread,'tiles',np.median(counts) if counts else 0)
