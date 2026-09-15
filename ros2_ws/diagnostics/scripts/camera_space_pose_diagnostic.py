#!/usr/bin/env python3
"""Camera-space residual and pose-correlation diagnostics; offline only."""
import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np
from scipy.stats import pearsonr
sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_handeye_model as m

def load(directory):
    rows=[]
    for path in sorted(Path(directory).glob('sample_*.json')):
        sample=json.loads(path.read_text());sample['_path']=str(path);rows.append(sample)
    for s in rows:m.cache[id(s)]=(np.array(s['camera']['k']).reshape(3,3),np.array(s['camera']['d']),np.array(s['corners']))
    return rows

def one(s,params):
    k,d,uv=m.cache[id(s)];q=np.array(s['joints']);A=m.kin.forward('base_link','tcp_link',dict(zip(s['joint_names'],q)));pred,depth=m.project_points(m.obj,A,m.params_to_transform(params[:6]),m.params_to_transform(params[6:12]),k,d);return uv,pred,depth

def stats(e):
    return {'du_mean_px':float(np.mean(e[:,0])),'dv_mean_px':float(np.mean(e[:,1])),'du_rms_px':float(np.sqrt(np.mean(e[:,0]**2))),'dv_rms_px':float(np.sqrt(np.mean(e[:,1]**2))),'rmse_px':float(np.sqrt(np.mean(e**2))),'norm_median_px':float(np.median(np.linalg.norm(e,axis=1)))}

def corr(x,y):
    x=np.asarray(x);y=np.asarray(y);ok=np.isfinite(x)&np.isfinite(y)
    return None if ok.sum()<3 else {'r':float(pearsonr(x[ok],y[ok]).statistic),'p':float(pearsonr(x[ok],y[ok]).pvalue)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fit-dir',type=Path,required=True);ap.add_argument('--validation-dir',action='append',type=Path,default=[]);ap.add_argument('--result',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args()
    rows=load(a.fit_dir)+[s for d in a.validation_dir for s in load(d)];result=json.loads(a.result.read_text());p=result['tcp_to_camera_color_optical_frame'];params=np.r_[m.transform_to_params(m.transform(p['xyz'],m.R.from_quat(p['quaternion_xyzw']).as_matrix())),m.transform_to_params(m.transform(result['base_to_checkerboard_first_corner']['xyz'],m.R.from_euler('xyz',result['base_to_checkerboard_first_corner']['rpy']).as_matrix()))]
    all_rows=[];region_vectors={'left':[],'middle':[],'right':[]};out=a.output_dir;a.output_dir.mkdir(parents=True,exist_ok=True)
    for s in rows:
        uv,pred,depth=one(s,params);e=pred-uv;u=uv[:,0]; thirds=[u<848/3,(u>=848/3)&(u<2*848/3),u>=2*848/3]
        reg={name:stats(e[mask]) for name,mask in zip(['left','middle','right'],thirds) if mask.any()};
        for name,mask in zip(['left','middle','right'],thirds):
            if mask.any():region_vectors[name].append(e[mask])
        row={'id':s['id'],'q2_q5_deg':np.degrees(np.array(s['joints'])[1:5]).tolist(),'image_u_mean_px':float(np.mean(u)),'image_v_mean_px':float(np.mean(uv[:,1])),'depth_median':float(np.median(depth)),'overall':stats(e),'regions':reg};all_rows.append(row)
        if s['id'] in {'05','10','15'}:
            image=cv2.imread(str(Path(s['_path']).with_suffix('.png')))
            if image is not None:
                for actual,estimate in zip(uv,pred):
                    p0=tuple(np.rint(actual).astype(int));p1=tuple(np.rint(estimate).astype(int));cv2.arrowedLine(image,p0,p1,(0,0,255),1,tipLength=.2);cv2.circle(image,p0,2,(0,255,0),-1)
                cv2.imwrite(str(out/f'residual_overlay_{s["id"]}.png'),image)
    region_summary={name:stats(np.concatenate(v)) for name,v in region_vectors.items() if v}
    cameras=[s['camera'] for s in rows];frames=[s.get('capture_integrity',{}).get('image_frame') for s in rows]
    pose={}
    for label,index in [('q4',3),('q5',4)]:
        for metric in ['du_mean_px','dv_mean_px','rmse_px','image_u_mean_px','depth_median']:
            pose[f'{metric}_vs_{label}']=corr([r[metric] if metric in r else r['overall'][metric] for r in all_rows],[np.degrees(s['joints'][index]) for s in rows])
    for metric in ['du_mean_px','dv_mean_px','rmse_px']:
        pose[f'{metric}_vs_image_u']=corr([r['overall'][metric] for r in all_rows],[r['image_u_mean_px'] for r in all_rows]);pose[f'{metric}_vs_depth']=corr([r['overall'][metric] for r in all_rows],[r['depth_median'] for r in all_rows])
    report={'sample_count':len(rows),'camera_consistency':{'dimensions':sorted({(s['camera']['width'],s['camera']['height']) for s in rows}),'distortion_models':sorted({s['camera']['distortion_model'] for s in rows}),'frames':sorted(set(frames)),'intrinsics_identical':all(s['camera']==cameras[0] for s in cameras),'raw_vs_rectified_status':'not encoded in persisted samples; image topic was camera/color/image_raw'},'region_summary':region_summary,'per_pose':all_rows,'pose_correlations':pose}
    (out/'camera_space_pose_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n');(out/'REPORT.md').write_text('# Camera-space and pose residual diagnostic\n\n'+json.dumps({'camera_consistency':report['camera_consistency'],'region_summary':region_summary,'pose_correlations':pose},indent=2)+'\n');print(json.dumps({'samples':len(rows),'output':str(out)},indent=2))
if __name__=='__main__':main()
