#!/usr/bin/env python3
"""Compare complete official legacy six-joint geometry offline; no ROS or deployment."""
import json,hashlib
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import least_squares
import diagnose_handeye_model as m
old_path=m.OUT/'official_piper_description_old.urdf'
old=ET.parse(old_path).getroot()
differences=[]
for i in range(1,7):
 name=f'joint{i}'
 current=m.kin.root.find(f"joint[@name='{name}']")
 legacy=old.find(f"joint[@name='{name}']")
 for tag in ['parent','child','axis','origin']:
  a,b=current.find(tag),legacy.find(tag)
  if a.attrib!=b.attrib: differences.append({'joint':name,'field':tag,'current':dict(a.attrib),'legacy':dict(b.attrib)})
  if tag in ['parent','child'] and a.attrib!=b.attrib: raise RuntimeError('Legacy chain differs')
  a.attrib.clear();a.attrib.update(b.attrib)
sol=least_squares(lambda v:m.residual(v,m.sets['fit']),m.initial,max_nfev=300)
r={'source':'https://raw.githubusercontent.com/agilexrobotics/piper_ros/noetic/src/piper_description/urdf/piper_description_old.urdf','source_sha256':hashlib.sha256(old_path.read_bytes()).hexdigest(),'description':'All six joint origins and axes from legacy model; existing fixed TCP chain retained. Offline only; no limits or collision meshes deployed.','differences':differences}
for group,samples in m.sets.items():
 errors=m.residual(sol.x,samples).reshape(len(samples),88,2)
 r[group+'_rmse_px']=float(np.sqrt(np.mean(errors**2)))
 r[group+'_per_sample_rmse_px']={s['id']:float(np.sqrt(np.mean(e**2))) for s,e in zip(samples,errors)}
(m.OUT/'legacy_geometry_check.json').write_text(json.dumps(r,indent=2))
print(json.dumps({k:v for k,v in r.items() if k.endswith('_rmse_px') and 'per_sample' not in k},indent=2))
print('Joint geometry field differences:',len(differences))
