"""Final postprocessing for NYC taxi fare predictions.

Handles the 4 invalid-coordinate categories differently (see blackboard
[[dubhe_tau_invalid_categories_refined.md]]):
  - allzero  : group-mean fallback (year,hour,pax) fitted on training allzero rows
  - partialzero: dedicated XGB model on valid-endpoint features
  - oob      : global mean of 'other' training rows
  - swap     : keep the model prediction (do NOT fall back)
  - valid    : keep the model prediction
Then lower-clip at 2.5 (no upper clip).
"""
import os, json, argparse
import numpy as np, pandas as pd, xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
PP = f"{HERE}/postproc"
ROOT = "/download_job_code/payload/sandbox"
CEN=(40.7580,-73.9855); JFK=(40.6413,-73.7781); LGA=(40.7769,-73.8740); EWR=(40.6895,-74.1745)
R=6371.0088
CLIP_LO=2.5


def hav(lat1,lon1,lat2,lon2):
    lat1=np.radians(lat1);lat2=np.radians(lat2);dlat=lat2-lat1;dlon=np.radians(lon2-lon1)
    a=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R*2*np.arcsin(np.sqrt(np.clip(a,0,1)))


def categorize(df):
    plon=df.pickup_longitude.values;plat=df.pickup_latitude.values
    dlo=df.dropoff_longitude.values;dlat=df.dropoff_latitude.values
    zero=(plon==0)|(plat==0)|(dlo==0)|(dlat==0)
    allz=(plon==0)&(plat==0)&(dlo==0)&(dlat==0)
    sw=(((plon>=39)&(plon<=42)&(plat>=-75)&(plat<=-72))|((dlo>=39)&(dlo<=42)&(dlat>=-75)&(dlat<=-72)))
    inbox=(plon>=-75)&(plon<=-72)&(plat>=39)&(plat<=42)&(dlo>=-75)&(dlo<=-72)&(dlat>=39)&(dlat<=42)
    cat=np.where(allz,'allzero',np.where(sw,'swap',np.where(zero|~inbox,'other','valid')))
    return cat


def partial_feats(df):
    plon=df.pickup_longitude.values;plat=df.pickup_latitude.values
    dlo=df.dropoff_longitude.values;dlat=df.dropoff_latitude.values
    pu_ok=(plon!=0)&(plat!=0); do_ok=(dlo!=0)&(dlat!=0)
    use_pu=pu_ok
    vlat=np.where(use_pu,plat,dlat); vlon=np.where(use_pu,plon,dlo)
    side=use_pu.astype(np.float32)
    okend=pu_ok|do_ok
    dcen=hav(vlat,vlon,*CEN); dj=hav(vlat,vlon,*JFK); dl=hav(vlat,vlon,*LGA); de=hav(vlat,vlon,*EWR)
    amin=np.minimum(np.minimum(dj,dl),de)
    return np.column_stack([side,okend.astype(np.float32),vlat,vlon,dcen,dj,dl,de,amin,
                            (amin<2).astype(np.float32),
                            df.year.values,df.month.values,df.day.values,df.hour.values,
                            df.minute.values,df.weekday.values,df.passenger_count.values]).astype(np.float32)


def _ensure_feats(df):
    need = {'year','month','day','hour','minute','weekday','passenger_count'}
    if not need.issubset(df.columns):
        dt = pd.to_datetime(df['pickup_datetime'], format='%Y-%m-%d %H:%M:%S UTC', errors='coerce')
        df = df.copy()
        df['year']=dt.dt.year; df['month']=dt.dt.month; df['day']=dt.dt.day
        df['hour']=dt.dt.hour; df['minute']=dt.dt.minute; df['weekday']=dt.dt.weekday
    return df


def apply_postprocess(pred, test_df=None, verbose=True):
    if test_df is None:
        test_df = pd.read_csv(f"{ROOT}/data/test.csv")
    test_df = _ensure_feats(test_df)
    pred = np.asarray(pred, dtype=np.float64).copy()
    cat = categorize(test_df)
    meta = json.load(open(f"{PP}/fallback_meta.json"))
    az_glob = meta['az_glob']; oth_glob = meta['oth_glob']

    # allzero
    m = cat=='allzero'
    if m.any():
        g = pd.read_csv(f"{PP}/fallback_allzero.csv")
        idx = pd.MultiIndex.from_arrays([test_df.year.to_numpy()[m], test_df.hour.to_numpy()[m],
                                         test_df.passenger_count.to_numpy()[m]],
                                        names=['year','hour','passenger_count'])
        vals = g.set_index(['year','hour','passenger_count'])['mean'].reindex(idx).to_numpy()
        pred[m] = np.where(np.isnan(vals), az_glob, vals)
    # other (partialzero or oob)
    m = cat=='other'
    if m.any():
        sub = test_df.loc[m]
        plon=sub.pickup_longitude.values;plat=sub.pickup_latitude.values
        dlo=sub.dropoff_longitude.values;dlat=sub.dropoff_latitude.values
        pu_ok=(plon!=0)&(plat!=0); do_ok=(dlo!=0)&(dlat!=0)
        zero=(plon==0)|(plat==0)|(dlo==0)|(dlat==0)
        use_partial = (zero & (pu_ok|do_ok))
        vals = np.full(m.sum(), oth_glob, dtype=np.float64)
        if use_partial.any():
            bst = xgb.Booster(); bst.load_model(f"{PP}/partial_model.json")
            best = meta['partial_best_iter']
            Xp = partial_feats(sub.loc[use_partial])
            vals[use_partial] = bst.predict(xgb.DMatrix(Xp), iteration_range=(0, best+1))
        pred[m] = vals
    # swap: keep model pred
    pred = np.clip(pred, CLIP_LO, None)
    if verbose:
        print("postprocess categories:", pd.Series(cat).value_counts().to_dict())
    return pred


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--out', required=True)
    a=ap.parse_args()
    p=pd.read_csv(a.pred)
    test=pd.read_csv(f"{ROOT}/data/test.csv")
    assert list(p['key'])==list(test['key']), "pred order must match test.csv"
    outp=apply_postprocess(p['fare_amount'].values, test)
    pd.DataFrame({'key':test['key'],'fare_amount':outp}).to_csv(a.out,index=False)
    print("wrote",a.out,len(outp))
