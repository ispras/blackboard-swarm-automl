import pandas as pd, numpy as np, gc, time
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

ROOT="/download_job_code/payload/sandbox"; W=f"{ROOT}/workspaces"
A=f"{W}/achernar_sigma"; R=f"{W}/rigel_delta"
t0=time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]",*a,flush=True)

PAIR={"pv1":(f"{A}/oof_pair_lgbv1.parquet","pred"),"pv2":(f"{A}/oof_pair_lgbv2.parquet","pred"),
      "prig":(f"{R}/oof_pair.parquet","prob")}
PAIR_T={"pv1":(f"{A}/testpred_pair_lgbv1.parquet","pred"),"pv2":(f"{A}/testpred_pair_lgbv2.parquet","pred"),
        "prig":(f"{R}/testpred_pair.parquet","prob")}
RAW_T=["dist","closing","rel_speed","dist_rmin3","dist_rmin7","dist_rmean5","closing_rmean5",
       "past_close_5","is_nn1","is_nn2","is_nn_opp1","is_nn_opp2","speed_sum","speed_absdiff",
       "sa_absdiff","toward1","toward2","nn_opp_1","nn_opp_2","n_opp_within2_1","n_opp_within2_2"]
RAW_H=["x_ndist_min","x_ndist_max","x_ndist_mean","x_nviews","Endzone_ndist","Sideline_ndist",
       "Endzone_iou","Sideline_iou","x_ndist_rmin5","x_ndist_rmean5","x_ndist_min_d1",
       "Endzone_nn_hel_ratio_p1","Sideline_nn_hel_ratio_p1","Endzone_nn_hel_ratio_p2",
       "Sideline_nn_hel_ratio_p2","Endzone_n_hel_near_p1","Sideline_n_hel_near_p2"]

def parse(df):
    p=df.contact_id.str.split("_",expand=True)
    df["game_play"]=p[0]+"_"+p[1]; df["step"]=p[2].astype(int); df["p1"]=p[3].astype(np.int64); df["p2"]=p[4].astype(np.int64)
    return df

def rank_by_fold(x, folds):
    out=np.empty(len(x),np.float64)
    for f in np.unique(folds):
        m=folds==f; out[m]=rankdata(x[m])/max(m.sum(),1)
    return out

def main():
    log("load base")
    base=pd.read_parquet(PAIR["pv1"][0])[["contact_id","y","fold"]].copy(); base=parse(base)
    base["fold"]=base.fold.astype(int)
    for k,(f,c) in PAIR.items():
        d=pd.read_parquet(f); mp=d.set_index("contact_id")[c]
        base[k]=base.contact_id.map(mp).values
    for k in PAIR: base[k+"_r"]=rank_by_fold(base[k].values, base.fold.values)
    base=base.sort_values(["game_play","p1","p2","step"]).reset_index(drop=True)
    log("base",base.shape)
    tr=pd.read_parquet(f"{A}/train_tracking_features.parquet")
    tr=tr[tr.contact_id.str[-1]!="G"][["contact_id"]+[c for c in RAW_T if c in tr.columns]]
    he=pd.read_parquet(f"{A}/train_helmet_features.parquet")
    he=he[["contact_id"]+[c for c in RAW_H if c in he.columns]]
    if "step" in tr.columns: tr=tr.drop(columns=["step"])
    raw=tr.merge(he,on="contact_id",how="left"); del tr,he; gc.collect()
    base=base.merge(raw,on="contact_id",how="left")
    tcols=[k+"_r" for k in PAIR]
    gb=base.groupby(["game_play","p1","p2"],sort=False)
    for c in tcols:
        base[c+"_lag1"]=gb[c].shift(1).astype(np.float32)
        base[c+"_lead1"]=gb[c].shift(-1).astype(np.float32)
        base[c+"_lag2"]=gb[c].shift(2).astype(np.float32)
        base[c+"_lead2"]=gb[c].shift(-2).astype(np.float32)
        base[c+"_m3"]=gb[c].transform(lambda x: x.rolling(3,center=True,min_periods=1).mean()).astype(np.float32)
        base[c+"_m5"]=gb[c].transform(lambda x: x.rolling(5,center=True,min_periods=1).mean()).astype(np.float32)
        base[c+"_m11"]=gb[c].transform(lambda x: x.rolling(11,center=True,min_periods=1).mean()).astype(np.float32)
        base[c+"_std5"]=gb[c].transform(lambda x: x.rolling(5,center=True,min_periods=1).std()).astype(np.float32)
    feats=[c for c in base.columns if c not in ("contact_id","y","fold","game_play","step","p1","p2") and base[c].dtype!=object]
    log("feats",len(feats))
    y=base.y.values.astype(np.int8); foldv=base.fold.values
    X=base[feats].astype(np.float32).values
    oof=np.zeros(len(base))
    params=dict(objective="binary",metric="auc",learning_rate=0.03,num_leaves=63,min_data_in_leaf=200,
                feature_fraction=0.7,bagging_fraction=0.85,bagging_freq=1,lambda_l2=2.0,max_bin=127,
                num_threads=4,verbose=-1,seed=11)
    for f in sorted(np.unique(foldv)):
        va=foldv==f; trn=~va
        dt=lgb.Dataset(X[trn],label=y[trn],feature_name=feats)
        dv=lgb.Dataset(X[va],label=y[va],reference=dt)
        bst=lgb.train(params,dt,num_boost_round=3000,valid_sets=[dv],
                      callbacks=[lgb.early_stopping(120,verbose=False),lgb.log_evaluation(300)])
        oof[va]=bst.predict(X[va],num_iteration=bst.best_iteration)
        log("fold",f,"iter",bst.best_iteration,"auc",round(roc_auc_score(y[va],oof[va]),5))
        del dt,dv,bst; gc.collect()
    log("STACK PAIR OOF AUC",roc_auc_score(y,oof))
    pd.DataFrame({"contact_id":base.contact_id,"y":y,"pred":oof.astype(np.float32),"fold":foldv,
                  "game_play":base.game_play}).to_parquet(f"{A}/oof_pair_gstack.parquet",index=False)
    # test
    tb=pd.read_parquet(PAIR_T["pv1"][0])[["contact_id"]].copy(); tb=parse(tb)
    for k,(f,c) in PAIR_T.items():
        d=pd.read_parquet(f); mp=d.set_index("contact_id")[c]
        tb[k]=tb.contact_id.map(mp).values
    for k in PAIR_T: tb[k+"_r"]=rankdata(tb[k].values)/len(tb)
    tb=tb.sort_values(["game_play","p1","p2","step"]).reset_index(drop=True)
    tr=pd.read_parquet(f"{A}/test_tracking_features.parquet")
    tr=tr[tr.contact_id.str[-1]!="G"][["contact_id"]+[c for c in RAW_T if c in tr.columns]]
    he=pd.read_parquet(f"{A}/test_helmet_features.parquet")
    he=he[["contact_id"]+[c for c in RAW_H if c in he.columns]]
    if "step" in tr.columns: tr=tr.drop(columns=["step"])
    raw=tr.merge(he,on="contact_id",how="left"); del tr,he; gc.collect()
    tb=tb.merge(raw,on="contact_id",how="left")
    gbt=tb.groupby(["game_play","p1","p2"],sort=False)
    for c in tcols:
        tb[c+"_lag1"]=gbt[c].shift(1).astype(np.float32)
        tb[c+"_lead1"]=gbt[c].shift(-1).astype(np.float32)
        tb[c+"_lag2"]=gbt[c].shift(2).astype(np.float32)
        tb[c+"_lead2"]=gbt[c].shift(-2).astype(np.float32)
        tb[c+"_m3"]=gbt[c].transform(lambda x: x.rolling(3,center=True,min_periods=1).mean()).astype(np.float32)
        tb[c+"_m5"]=gbt[c].transform(lambda x: x.rolling(5,center=True,min_periods=1).mean()).astype(np.float32)
        tb[c+"_m11"]=gbt[c].transform(lambda x: x.rolling(11,center=True,min_periods=1).mean()).astype(np.float32)
        tb[c+"_std5"]=gbt[c].transform(lambda x: x.rolling(5,center=True,min_periods=1).std()).astype(np.float32)
    Xte=tb[feats].astype(np.float32).values
    pt=np.zeros(len(tb))
    for f in sorted(np.unique(foldv)):
        va=foldv==f; trn=~va
        dt=lgb.Dataset(X[trn],label=y[trn],feature_name=feats)
        dv=lgb.Dataset(X[va],label=y[va],reference=dt)
        bst=lgb.train(params,dt,num_boost_round=3000,valid_sets=[dv],
                      callbacks=[lgb.early_stopping(120,verbose=False)])
        pt+=bst.predict(Xte,num_iteration=bst.best_iteration)/len(np.unique(foldv))
        del dt,dv,bst; gc.collect()
    pd.DataFrame({"contact_id":tb.contact_id,"pred":pt.astype(np.float32)}).to_parquet(f"{A}/testpred_pair_gstack.parquet",index=False)
    log("DONE")

if __name__=="__main__":
    main()
