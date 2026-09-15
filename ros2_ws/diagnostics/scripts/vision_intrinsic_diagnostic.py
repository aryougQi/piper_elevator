#!/usr/bin/env python3
"""Offline-only camera intrinsic and hand-eye sensitivity diagnostic.

Uses fit images to estimate camera intrinsics, then fits hand-eye on fit only and
reports independent validation/repeat projection errors. Never writes production
configuration or moves a robot.
"""
from pathlib import Path
import json, sys
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

ROOT=Path(__file__).resolve().parents[4]
DIAG=Path(__file__).resolve().parents[1]
SRC=DIAG/'data/handeye_recheck_20mm'
OUT=DIAG/'data/handeye_resolution_20260911/vision'; OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'handeye_calibration/scripts'))
from solve_handeye import UrdfKinematics, transform, params_to_transform, transform_to_params, project_points

sets={g:[json.loads(f.read_text()) for f in sorted((SRC/g).glob('sample_*.json'))] for g in ('fit','validation','repeat')}
obj=np.array([[c*.02,r*.02,0] for r in range(11) for c in range(8)],float)
nomK=np.array(sets['fit'][0]['camera']['k'],float).reshape(3,3); nomD=np.array(sets['fit'][0]['camera']['d'],float)
kin=UrdfKinematics(ROOT/'handeye_calibration/calibration/piper_handeye_input.urdf')
rj=json.loads((SRC/'result/handeye_result.json').read_text())
x=rj['tcp_to_camera_color_optical_frame']; b=rj['base_to_checkerboard_first_corner']
initial=np.r_[transform_to_params(transform(x['xyz'],R.from_quat(x['quaternion_xyzw']).as_matrix())), transform_to_params(transform(b['xyz'],R.from_euler('xyz',b['rpy']).as_matrix()))]

def calibrate(samples, flags=0):
    objl=[obj.astype(np.float32)]*len(samples); imgl=[np.asarray(s['corners'],np.float32) for s in samples]
    K=nomK.copy(); d=np.zeros(5)
    if flags:
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    rms,K,d,rvecs,tvecs=cv2.calibrateCamera(objl,imgl,(848,480),K,d,flags=flags)
    return {'rms_px':float(rms),'K':K.tolist(),'D':d.ravel().tolist(),'rvecs':rvecs,'tvecs':tvecs}

# fit-only alternatives; fixed principal point/aspect are useful identifiability checks
alts={'nominal':(nomK,nomD),'fit_free':(np.array(calibrate(sets['fit'])['K']),np.array(calibrate(sets['fit'])['D']))}
for nm,flags in [('fit_fix_aspect',cv2.CALIB_FIX_ASPECT_RATIO),('fit_fix_principal',cv2.CALIB_FIX_PRINCIPAL_POINT),('fit_zero_tangent',cv2.CALIB_ZERO_TANGENT_DIST)]:
    c=calibrate(sets['fit'],flags); alts[nm]=(np.array(c['K']),np.array(c['D']))

# pose coverage and repeated corner consistency
coverage=[]
for s in sets['fit']:
    uv=np.asarray(s['corners'],float); ok,rvec,tvec=cv2.solvePnP(obj.astype(np.float32),uv.astype(np.float32),nomK,nomD,flags=cv2.SOLVEPNP_ITERATIVE)
    e=np.linalg.norm((cv2.projectPoints(obj,rvec,tvec,nomK,nomD)[0].reshape(-1,2)-uv),axis=1)
    rv=R.from_rotvec(rvec.ravel()); coverage.append({'id':s['id'],'board_rpy_deg':rv.as_euler('xyz',degrees=True).tolist(),'board_t_m':tvec.ravel().tolist(),'image_center_xy':np.mean(uv,axis=0).tolist(),'pnp_rmse_px':float(np.sqrt(np.mean(e*e)))})

def residual(v,samples,K,D):
    X=params_to_transform(v[:6]); B=params_to_transform(v[6:12]); out=[]
    for s in samples:
        A=kin.forward('base_link','tcp_link',dict(zip(s['joint_names'],s['joints'])))
        pred,depth=project_points(obj,A,X,B,K,D)
        out.append((pred-np.asarray(s['corners'])).ravel())
    return np.concatenate(out)

reports={}
for nm,(K,D) in alts.items():
    sol=least_squares(lambda v:residual(v,sets['fit'],K,D),initial,max_nfev=800,ftol=1e-10,xtol=1e-10,gtol=1e-9)
    rep={'K':K.tolist(),'D':D.tolist(),'fit_intrinsic_or_nominal':nm,'fit_solver_cost':float(sol.cost),'fit_solver_optimality':float(sol.optimality)}
    for g,ss in sets.items():
        er=residual(sol.x,ss,K,D).reshape(len(ss),88,2)
        rep[g+'_rmse_px']=float(np.sqrt(np.mean(er*er)))
        rep[g+'_per_sample_rmse_px']={s['id']:float(np.sqrt(np.mean(e*e))) for s,e in zip(ss,er)}
    reports[nm]=rep

# leave-one-pose intrinsic variation: quantify weakly constrained values
loo=[]
for drop in range(len(sets['fit'])):
    ss=sets['fit'][:drop]+sets['fit'][drop+1:]
    c=calibrate(ss)
    K=np.array(c['K']); D=np.array(c['D'])
    loo.append({'dropped_id':sets['fit'][drop]['id'],'rms_px':c['rms_px'],'fx':float(K[0,0]),'fy':float(K[1,1]),'cx':float(K[0,2]),'cy':float(K[1,2]),'k1':float(D[0]),'k2':float(D[1])})

# repeated-image differences, with nearest same-label return samples
repeats=[]
for f in sorted((SRC/'repeat').glob('*.json')):
    s=json.loads(f.read_text()); uv=np.asarray(s['corners'])
    # compare to fit pose with same id where available
    cand=[q for q in sets['fit'] if q['id']==s['id']]
    row={'file':f.name,'id':s['id']}
    if cand:
        du=np.linalg.norm(uv-np.asarray(cand[0]['corners']),axis=1); row.update({'vs_fit_median_px':float(np.median(du)),'vs_fit_rms_px':float(np.sqrt(np.mean(du*du)))})
    repeats.append(row)

out={'source':str(SRC),'checkerboard':{'rows':11,'columns':8,'square_size_m':0.02},'nominal_K':nomK.tolist(),'nominal_D':nomD.tolist(),'intrinsic_fits':{k:{'K':v[0].tolist(),'D':v[1].tolist()} for k,v in alts.items()},'coverage':coverage,'leave_one_out_intrinsics':loo,'repeat_corner_comparisons':repeats,'handeye':reports,'interpretation':{'fit_only_intrinsic_estimate_is_not_deployment_calibration':True,'validation_not_used_for_any_fit':True}}
(OUT/'vision_intrinsic_diagnostic.json').write_text(json.dumps(out,indent=2))
# concise markdown evidence
free=reports['fit_free']; nominal=reports['nominal']; c=alts['fit_free']
md=['# Vision intrinsic diagnostic (offline)','',f'- Source: `{SRC}`; fit={len(sets["fit"])} validation={len(sets["validation"])} repeat={len(sets["repeat"])}.','- Validation samples were never used to estimate intrinsics or hand-eye.','', '## Intrinsic estimates','',f'- Nominal K = `{nomK.tolist()}`; D = `{nomD.tolist()}`.',f'- Fit-only free calibration K = `{c[0].tolist()}`; D = `{c[1].tolist()}`.',f'- Fit-only calibration RMS = `{calibrate(sets["fit"])["rms_px"]:.4f} px`; this is image reprojection error, not hand-eye accuracy.','', '## Hand-eye sensitivity','', '| camera model | fit RMSE px | validation RMSE px | repeat RMSE px |','|---|---:|---:|---:|']
for nm,r in reports.items(): md.append(f'| {nm} | {r["fit_rmse_px"]:.3f} | {r["validation_rmse_px"]:.3f} | {r["repeat_rmse_px"]:.3f} |')
md += ['', '## Limits', '', '- Intrinsics and planar board poses are partially coupled; the fit-only estimate is a sensitivity experiment, not evidence to replace CameraInfo.', '- Leave-one-pose values and pose coverage are recorded in `vision_intrinsic_diagnostic.json`; large spread indicates weak observability.', '- This diagnostic does not resolve the dominant cross-pose kinematic inconsistency if free intrinsics leave validation error high.']
(OUT/'REPORT.md').write_text('\n'.join(md)+'\n')
print('wrote',OUT)
for nm,r in reports.items(): print(nm, r['fit_rmse_px'],r['validation_rmse_px'],r['repeat_rmse_px'])
