import os, argparse, numpy as np, pandas as pd
from rapidfuzz.distance import Levenshtein as Lev
ROOT='/download_job_code/payload/sandbox'
FB=open(f'{ROOT}/workspaces/regulus_iota/baseline_constant.txt').read().strip()

def load_pred(path, name):
    d=pd.read_csv(path)
    col='inchi' if 'inchi' in d.columns else ('inchi_from_smiles' if 'inchi_from_smiles' in d.columns else 'InChI_pred')
    s=d[col].fillna('').astype(str).replace('nan','')
    if 'valid' in d.columns: valid=d['valid'].fillna(False).astype(bool).values
    else: valid=(s!='').values & (s!=FB).values
    s=np.where(valid & (s!='') & (s!=FB), s, '')
    out=pd.DataFrame({'image_id':d.image_id.astype(str)})
    out[name]=s; out[name+'_valid']=np.asarray(s!='',dtype=bool)
    c=d['conf'] if 'conf' in d.columns else (d['confidence'] if 'confidence' in d.columns else pd.Series(np.zeros(len(d))))
    out[name+'_conf']=c.astype(float).values
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--iota', required=True)
    ap.add_argument('--ms', default='')
    ap.add_argument('--rot', default='')
    ap.add_argument('--omega', default='')
    ap.add_argument('--out', required=True)
    ap.add_argument('--q', type=float, default=0.05)
    ap.add_argument('--ms_thr', type=float, default=0.88)
    ap.add_argument('--gt', default='')
    ap.add_argument('--report', default='')
    a=ap.parse_args()
    t=load_pred(a.iota,'iota')
    sources=[('iota',None)]
    if a.ms:
        f=load_pred(a.ms,'ms'); t=t.merge(f,on='image_id',how='left'); print('ms covered',f.image_id.nunique())
    if a.rot:
        f=load_pred(a.rot,'rot'); t=t.merge(f,on='image_id',how='left'); print('rot covered',f.image_id.nunique())
    if a.omega:
        f=load_pred(a.omega,'omega'); t=t.merge(f,on='image_id',how='left'); print('omega covered',f.image_id.nunique())
    conf=t['iota_conf'].values; thr=np.quantile(conf,a.q)
    print('n',len(t),'q',a.q,'conf_thr',round(float(thr),5),'invalid',round(float((~t.iota_valid).mean()),5))
    def gv(r,name,col,default):
        v=getattr(r,col,default)
        if isinstance(v,float) and pd.isna(v): return default
        return v
    pred=[]; n_ms=0; n_rot=0; n_om=0; n_fb=0
    for r in t.itertuples():
        iv=r.iota_valid; ival=r.iota
        if not iv:
            if hasattr(r,'ms_valid') and r.ms_valid and getattr(r,'ms'):
                pred.append(r.ms); n_ms+=1
            elif hasattr(r,'rot_valid') and r.rot_valid and getattr(r,'rot'):
                pred.append(r.rot); n_rot+=1
            elif hasattr(r,'omega_valid') and r.omega_valid and getattr(r,'omega'):
                pred.append(r.omega); n_om+=1
            else:
                pred.append(FB); n_fb+=1
        elif r.iota_conf<=thr and hasattr(r,'ms_valid') and r.ms_valid and getattr(r,'ms') and r.ms_conf>a.ms_thr:
            pred.append(r.ms); n_ms+=1
        else:
            pred.append(ival)
    out=pd.DataFrame({'image_id':t.image_id.values,'InChI':pred})
    out.to_csv(a.out,index=False)
    print('picked ms',n_ms,'rot',n_rot,'omega',n_om,'fallback',n_fb)
    print('wrote',a.out,len(out),'dupes',out.image_id.duplicated().sum())
    if a.gt:
        gt=pd.read_csv(a.gt)[['image_id','InChI']]
        mm=gt.merge(out,on='image_id',suffixes=('_gt','_pred'))
        lev=np.mean([Lev.distance(p if isinstance(p,str) and p else FB,g) for p,g in zip(mm.InChI_pred,mm.InChI_gt)])
        base=gt.merge(t[['image_id','iota','iota_valid']],on='image_id')
        bl=np.mean([Lev.distance(p if v and isinstance(p,str) and p else FB,g) for p,v,g in zip(base.iota,base.iota_valid,base.InChI)])
        print('VAL mean Lev',round(float(lev),4),'base',round(float(bl),4))
        if a.report:
            pd.DataFrame([{'lev':lev,'base':bl,'q':a.q,'ms_thr':a.ms_thr,'n':len(mm)}]).to_csv(a.report,index=False)
if __name__=='__main__': main()
