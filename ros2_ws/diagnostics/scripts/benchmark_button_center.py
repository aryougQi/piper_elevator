"""Known-truth synthetic center benchmark; unrelated to real-image accuracy."""
import json
from pathlib import Path
import cv2
import numpy as np
from piper_elevator_app.experimental_button_geometry import refine_rectangular_center


def benchmark():
 rng=np.random.default_rng(1309); rows=[]
 base=np.array([[58,60],[135,72],[127,132],[64,138]],float)
 for i in range(100):
  corners=base+rng.uniform(-4,4,(4,2))
  hi=np.full((800,800),180,np.uint8)
  cv2.fillConvexPoly(hi,np.rint(corners*4).astype(np.int32),25)
  image=cv2.resize(hi,(200,200),interpolation=cv2.INTER_AREA)
  image=cv2.GaussianBlur(image,(3,3),.7)
  image=np.clip(image.astype(float)+rng.normal(0,2,image.shape),0,255).astype(np.uint8)
  cv2.putText(image,'1',(103,113),cv2.FONT_HERSHEY_SIMPLEX,.7,240,2)
  box=np.r_[corners.min(0)-2,corners.max(0)+2]
  h=np.column_stack((corners,np.ones(4))); p=np.cross(np.cross(h[0],h[2]),np.cross(h[1],h[3]));truth=p[:2]/p[2]
  # Pixel-center mapping for 4x area downsampling: (x+0.5)/4-0.5.
  truth-=.375
  row={}
  for mode in (False,True):
   result=refine_rectangular_center(image,box,refine_edges=mode)
   row[str(mode)]=None if result.center is None else float(np.linalg.norm(result.center-truth))
  rows.append(row)
 summary={}
 for mode in ('False','True'):
  errors=[r[mode] for r in rows if r[mode] is not None]
  summary[mode]={'accepted':len(errors),'median_error_px':float(np.median(errors)),'p95_error_px':float(np.percentile(errors,95))}
 common=[r for r in rows if all(v is not None for v in r.values())]
 summary['common_frames']=len(common)
 summary['common_mean_error_px']={mode:float(np.mean([r[mode] for r in common])) for mode in ('False','True')}
 return summary,rows

if __name__=='__main__':
 summary,rows=benchmark()
 output=Path(__file__).resolve().parents[1]/'data/button_geometry_v3_20260913/center_benchmark.json'
 output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps({'summary':summary,'frames':rows},indent=2)+'\n')
 print(json.dumps(summary,indent=2))
