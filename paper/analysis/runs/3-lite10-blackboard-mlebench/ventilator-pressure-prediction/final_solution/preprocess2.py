import numpy as np, pandas as pd, os, json, time

T = 80
OUT = os.path.dirname(os.path.abspath(__file__))

def load(path, is_train):
    cols = ['breath_id','R','C','time_step','u_in','u_out'] + (['pressure'] if is_train else [])
    df = pd.read_csv(path, usecols=cols)
    n = len(df)//T
    assert len(df) == n*T
    return {c: df[c].values.reshape(n, T).astype(np.float32) for c in cols}

def shift(a, k, fill=0.0):
    out = np.full_like(a, fill, dtype=np.float32)
    if k > 0: out[:, k:] = a[:, :-k]
    elif k < 0: out[:, :k] = a[:, -k:]
    return out

def rep(a): return np.repeat(a[:, None], T, axis=1)

def feat(d):
    u=d['u_in']; t=d['time_step']; R=d['R']; C=d['C']; uo=d['u_out']
    ar=np.arange(1,T+1,dtype=np.float32)[None,:]
    step=np.arange(T,dtype=np.float32)[None,:]
    dt=np.empty_like(t); dt[:,0]=0; dt[:,1:]=t[:,1:]-t[:,:-1]
    tau=(R*C)+1e-3
    cum=np.cumsum(u,axis=1); cummean=cum/ar
    area=np.cumsum(u*dt,axis=1)
    f={
      'u_in':u,'u_out':uo,'time_step':t,'R':R,'C':C,
      'cumsum':cum,'cummean':cummean,'cum_frac':cum/(cum[:,-1:]+1e-3),
      'lag1':shift(u,1),'lag2':shift(u,2),'lag3':shift(u,3),'lag4':shift(u,4),
      'lead1':shift(u,-1),'lead2':shift(u,-2),'lead3':shift(u,-3),
      'diff1':u-shift(u,1),'diff2':u-shift(u,2),
      'u_first':rep(u[:,0]),'area':area,'area2':t*cum,'dt':dt,'u_duration':step*dt,
      'R_C':R*C,'R_over_C':R/(C+1e-6),'C_over_R':C/(R+1e-6),
      't_over_tau':t/tau,'u_over_tau':u/tau,'u_times_C':u*C,'u_over_R':u/R,
      'u_sq':u*u,'u_max_run':np.maximum.accumulate(u,axis=1),
      'exp_tau':np.exp(-t/tau),'one_minus_exp_tau':1.0-np.exp(-t/tau),
      'u_1mexp':u*(1.0-np.exp(-t/tau)),
      'u_out_cumsum':np.cumsum(uo,axis=1),
      'u_out_lead1':shift(uo,-1),'u_out_lead2':shift(uo,-2),
      'cumsum_over_C':cum/C,'cummean_over_C':cummean/C,'u_over_C':u/C,
      'cumsum_over_RC':cum/(R*C+1e-3),'cumsum_plus_C':cum+C,'cumsum_plus_R':cum+R,
      't_norm':t/(t[:,-1:]+1e-6),
      'R_5':(R==5).astype(np.float32),'R_20':(R==20).astype(np.float32),'R_50':(R==50).astype(np.float32),
      'C_10':(C==10).astype(np.float32),'C_20':(C==20).astype(np.float32),'C_50':(C==50).astype(np.float32),
    }
    names=sorted(f.keys())
    X=np.stack([f[k] for k in names],axis=-1).astype(np.float32)
    X=np.nan_to_num(X,nan=0.0,posinf=0.0,neginf=0.0)
    return X,names

if __name__=='__main__':
    t0=time.time()
    tr=load(os.path.join(OUT,'..','..','data','train.csv'),True)
    te=load(os.path.join(OUT,'..','..','data','test.csv'),False)
    Xtr,names=feat(tr); Xte,_=feat(te)
    ntr=len(Xtr); nte=len(Xte)
    A=Xtr.reshape(ntr*T,-1); B=Xte.reshape(nte*T,-1)
    mean=A.mean(0); std=A.std(0); std[std<1e-6]=1.0
    A=(A-mean)/std; B=(B-mean)/std
    np.save(os.path.join(OUT,'X2_train.npy'),A.astype(np.float32))
    np.save(os.path.join(OUT,'X2_test.npy'),B.astype(np.float32))
    json.dump({'feature_names':names,'mean':mean.tolist(),'std':std.tolist(),'T':T},
              open(os.path.join(OUT,'feature_meta2.json'),'w'))
    print('X2',A.shape,B.shape,'nfeat',len(names),'time',round(time.time()-t0,1))
    print(names)
