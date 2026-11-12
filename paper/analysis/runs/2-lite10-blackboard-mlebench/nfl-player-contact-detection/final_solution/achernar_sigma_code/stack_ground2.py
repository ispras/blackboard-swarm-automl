import pandas as pd, numpy as np, gc, time
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

ROOT="/download_job_code/payload/sandbox"; W=f"{ROOT}/workspaces"
A=f"{W}/achernar_sigma"; C=f"{W}/castor_alpha"; R=f"{W}/rigel_delta"
t0=time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]",*a,flush=True)

GROUND={"v1":(f"{A}/oof_ground_lgbv1.parquet","pred"),"v2":(f"{A}/oof_ground_lgbv2.parquet","pred"),
        "gfus":(f"{A}/oof_ground_gfus.parquet","pred"),"cvid":(f"{C}/ground_oof_video.parquet","pred"),
        "ceff":(f"{C}/ground_oof_video_eff.parquet","pred"),
        "ctab":(f"{C}/ground_oof_tab.parquet","oof"),"rig":(f"{R}/oof_ground.parquet","prob"),
        "rstack":(f"{R}/ground_oof_stack_r.parquet","prob")}
GROUND_T={"v1":(f"{A}/testpred_ground_lgbv1.parquet","pred"),"v2":(f"{A}/testpred_ground_lgbv2.parquet","pred"),
          "gfus":(f"{A}/testpred_ground_gfus.parquet","pred"),"cvid":(f"{C}/ground_test_video.parquet","pred"),
          "ceff":(f"{C}/ground_test_video_eff.parquet","pred"),
          "ctab":(f"{C}/ground_test_tab.parquet","pred"),"rig":(f"{R}/testpred_ground.parquet","prob"),
          "rstack":(f"{R}/ground_test_stack_r.parquet","prob")}
PAIR={"pstk":(f"{A}/oof_pair_gstack.parquet","pred"),"pv2":(f"{A}/oof_pair_lgbv2.parquet","pred"),
      "prig":(f"{R}/oof_pair.parquet","prob"),"ph":(f"{R}/oof_pair_h.parquet","prob")}
PAIR_T={"pstk":(f"{A}/testpred_pair_gstack.parquet","pred"),"pv2":(f"{A}/testpred_pair_lgbv2.parquet","pred"),
        "prig":(f"{R}/testpred_pair.parquet","prob"),"ph":(f"{R}/testpred_pair_h.parquet","prob")}

RAW_TRACK=["nn_opp_1","nn_any_1","n_opp_within2_1","n_within4_1","speed_1","acceleration_1","sa_1",
           "speed_rmean5_1","sa_rmean5_1","orient_dir_diff_1","nn_opp_rmin5_1","nn_opp_rmax5_1",
           "n_opp_within2_rmax5_1","speed_lag3_1","sa_lag3_1","x_position_1","y_position_1","vx_1","vy_1","cum_dist_1"]
RAW_HELM=["Endzone_cy_rank","Sideline_cy_rank","Endzone_bottom_to_max","Sideline_bottom_to_max",
          "Endzone_size_rel","Sideline_size_rel","Endzone_cy_d1","Sideline_cy_d1","Endzone_height_d1",
          "Sideline_height_d1","Endzone_cy_to_max","Sideline_cy_to_max","Endzone_nn_hel_ratio_p1",
          "Sideline_nn_hel_ratio_p1","Endzone_n_hel_near_p1","Sideline_n_hel_near_p1",
          "Endzone_max_iou_p1","Sideline_max_iou_p1"]

def parse(df):
    p=df.contact_id.str.split("_",expand=True)
    df["game_play"]=p[0]+"_"+p[1]; df["step"]=p[2].astype(int); df["p1"]=p[3].astype(np.int64)
    return df

def rank_by_fold(x, folds):
    out=np.empty(len(x),np.float64)
    for f in np.unique(folds):
        m=folds==f; out[m]=rankdata(x[m])/max(m.sum(),1)
    return out

def pair_context(pair_file, col, is_test):
    d=pd.read_parquet(pair_file)
    p=d.contact_id.str.split("_",expand=True)
    d["game_play"]=p[0]+"_"+p[1]; d["step"]=p[2].astype(int)
    d["pid1"]=p[3].astype(np.int64); d["pid2"]=pd.to_numeric(p[4],errors="coerce").astype("Int64")
    if not is_test:
        vals=rank_by_fold(d[col].values, d["fold"].values)
    else:
        vals=rankdata(d[col].values)/len(d)
    d["p"]=vals
    a=d[["game_play","step","pid1","p"]].rename(columns={"pid1":"pid"})
    b=d[["game_play","step","pid2","p"]].dropna().rename(columns={"pid2":"pid"})
    b["pid"]=b.pid.astype(np.int64)
    lo=pd.concat([a,b],ignore_index=True); del a,b,d; gc.collect()
    ctx=lo.groupby(["game_play","step","pid"]).p.agg(["max","sum","mean"]).reset_index()
    ctx.columns=["game_play","step","pid","max","sum","mean"]
    return ctx

def main():
    log("load ground base")
    base=pd.read_parquet(GROUND["v1"][0])[["contact_id","y","fold"]].copy()
    base=parse(base)
    base["fold"]=base.fold.astype(int)
    for k,(f,c) in GROUND.items():
        d=pd.read_parquet(f)
        mp=d.set_index("contact_id")[c]
        base[k]=base.contact_id.map(mp).values
    # rank-normalise each base model within fold
    for k in GROUND:
        base[k+"_r"]=rank_by_fold(base[k].values, base.fold.values)
    base=base.sort_values(["game_play","p1","step"]).reset_index(drop=True)
    log("ground base",base.shape)

    # raw features
    trn=pd.read_parquet(f"{A}/train_tracking_features.parquet")
    trn=trn[trn.contact_id.str.endswith("_G")][["contact_id"]+[c for c in RAW_TRACK if c in trn.columns]]
    hem=pd.read_parquet(f"{A}/train_helmet_features.parquet")
    hem=hem[["contact_id"]+[c for c in RAW_HELM if c in hem.columns]]
    raw=trn.merge(hem,on="contact_id",how="left"); del trn,hem; gc.collect()
    base=base.merge(raw,on="contact_id",how="left")
    log("raw merged",base.shape)

    rawc=[c for c in base.columns if c not in ("contact_id","y","fold","game_play","step","p1") and not c.endswith("_r")]
    for k in GROUND:
        rawc.append(k+"_r")
    # pair context
    for pk,(pf,pc) in PAIR.items():
        ctx=pair_context(pf,pc,False).rename(columns={"pid":"p1","max":pk+"_max","sum":pk+"_sum","mean":pk+"_mean"})
        base=base.merge(ctx,on=["game_play","step","p1"],how="left")
        log("ctx",pk,base.shape)
    ctxcols=[c for c in base.columns if c.startswith(("pstk_","pv2_","prig_","ph_"))]
    # temporal features on base probs + context
    grp=["game_play","p1"]
    gb=base.groupby(grp,sort=False)
    tcols=[k+"_r" for k in GROUND]+ctxcols
    for c in tcols:
        base[c+"_lag1"]=gb[c].shift(1).astype(np.float32)
        base[c+"_lead1"]=gb[c].shift(-1).astype(np.float32)
        base[c+"_lag2"]=gb[c].shift(2).astype(np.float32)
        base[c+"_lead2"]=gb[c].shift(-2).astype(np.float32)
        base[c+"_m5"]=gb[c].transform(lambda x: x.rolling(5,center=True,min_periods=1).mean()).astype(np.float32)
        base[c+"_m11"]=gb[c].transform(lambda x: x.rolling(11,center=True,min_periods=1).mean()).astype(np.float32)
        base[c+"_std5"]=gb[c].transform(lambda x: x.rolling(5,center=True,min_periods=1).std()).astype(np.float32)
    feats=rawc+tcols+[c for c in base.columns if any(c.startswith(t+"_") for t in tcols)]
    feats=[c for c in dict.fromkeys(feats) if c in base.columns and base[c].dtype!=object]
    log("n feats",len(feats))
    y=base.y.values.astype(np.int8); foldv=base.fold.values
    X=base[feats].astype(np.float32).values
    oof=np.zeros(len(base)); 
    params=dict(objective="binary",metric="auc",learning_rate=0.02,num_leaves=63,min_data_in_leaf=80,
                feature_fraction=0.5,bagging_fraction=0.85,bagging_freq=1,lambda_l2=2.0,max_bin=127,
                num_threads=4,verbose=-1,seed=7)
    for f in sorted(np.unique(foldv)):
        va=foldv==f; tr=~va
        dt=lgb.Dataset(X[tr],label=y[tr],feature_name=feats)
        dv=lgb.Dataset(X[va],label=y[va],reference=dt)
        bst=lgb.train(params,dt,num_boost_round=3000,valid_sets=[dv],
                      callbacks=[lgb.early_stopping(120,verbose=False),lgb.log_evaluation(300)])
        oof[va]=bst.predict(X[va],num_iteration=bst.best_iteration)
        log("fold",f,"iter",bst.best_iteration,"auc",round(roc_auc_score(y[va],oof[va]),5))
        del dt,dv,bst; gc.collect()
    log("STACK OOF AUC",roc_auc_score(y,oof))
    pd.DataFrame({"contact_id":base.contact_id,"y":y,"pred":oof.astype(np.float32),"fold":foldv,
                  "game_play":base.game_play}).to_parquet(f"{A}/oof_ground_gstack2.parquet",index=False)

    # ---------- TEST ----------
    log("test base")
    tb=pd.read_parquet(GROUND_T["v1"][0])[["contact_id"]].copy(); tb=parse(tb)
    for k,(f,c) in GROUND_T.items():
        d=pd.read_parquet(f); mp=d.set_index("contact_id")[c]
        tb[k]=tb.contact_id.map(mp).values
    for k in GROUND_T:
        tb[k+"_r"]=rankdata(tb[k].values)/len(tb)
    tb=tb.sort_values(["game_play","p1","step"]).reset_index(drop=True)
    # raw test
    trn=pd.read_parquet(f"{A}/test_tracking_features.parquet")
    trn=trn[trn.contact_id.str.endswith("_G")][["contact_id"]+[c for c in RAW_TRACK if c in trn.columns]]
    hem=pd.read_parquet(f"{A}/test_helmet_features.parquet")
    hem=hem[["contact_id"]+[c for c in RAW_HELM if c in hem.columns]]
    raw=trn.merge(hem,on="contact_id",how="left"); del trn,hem; gc.collect()
    tb=tb.merge(raw,on="contact_id",how="left")
    for pk,(pf,pc) in PAIR_T.items():
        ctx=pair_context(pf,pc,True).rename(columns={"pid":"p1","max":pk+"_max","sum":pk+"_sum","mean":pk+"_mean"})
        tb=tb.merge(ctx,on=["game_play","step","p1"],how="left")
    gbt=tb.groupby(grp,sort=False)
    for c in tcols:
        tb[c+"_lag1"]=gbt[c].shift(1).astype(np.float32)
        tb[c+"_lead1"]=gbt[c].shift(-1).astype(np.float32)
        tb[c+"_lag2"]=gbt[c].shift(2).astype(np.float32)
        tb[c+"_lead2"]=gbt[c].shift(-2).astype(np.float32)
        tb[c+"_m5"]=gbt[c].transform(lambda x: x.rolling(5,center=True,min_periods=1).mean()).astype(np.float32)
        tb[c+"_m11"]=gbt[c].transform(lambda x: x.rolling(11,center=True,min_periods=1).mean()).astype(np.float32)
        tb[c+"_std5"]=gbt[c].transform(lambda x: x.rolling(5,center=True,min_periods=1).std()).astype(np.float32)
    Xte=tb[feats].astype(np.float32).values
    # retrain on all train folds to predict test (use fold models avg for consistency)
    pt=np.zeros(len(tb))
    for f in sorted(np.unique(foldv)):
        va=foldv==f; tr=~va
        dt=lgb.Dataset(X[tr],label=y[tr],feature_name=feats)
        dv=lgb.Dataset(X[va],label=y[va],reference=dt)
        bst=lgb.train(params,dt,num_boost_round=3000,valid_sets=[dv],
                      callbacks=[lgb.early_stopping(120,verbose=False)])
        pt+=bst.predict(Xte,num_iteration=bst.best_iteration)/len(np.unique(foldv))
        del dt,dv,bst; gc.collect()
    pd.DataFrame({"contact_id":tb.contact_id,"pred":pt.astype(np.float32)}).to_parquet(f"{A}/testpred_ground_gstack2.parquet",index=False)
    log("DONE")

if __name__=="__main__":
    main()
