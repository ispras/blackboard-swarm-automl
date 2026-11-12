import pandas as pd, numpy as np, gc, time, os
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

ROOT = "/download_job_code/payload/sandbox"
W = f"{ROOT}/workspaces/achernar_sigma"
t0=time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]",*a,flush=True)

def load_ours(split):
    tr = pd.read_parquet(f"{W}/{split}_tracking_features.parquet")
    g = tr.contact_id.str.endswith("_G").values
    tr = tr.loc[g].reset_index(drop=True)
    if split=="train":
        tr["target"]=tr["contact"].astype(np.int8); tr=tr.drop(columns=["contact"])
    else:
        tr["target"]=np.nan
    tr=tr.drop(columns=[c for c in ["nfl_player_id_2_x","nfl_player_id_2_y"] if c in tr.columns])
    he=pd.read_parquet(f"{W}/{split}_helmet_features.parquet").set_index("contact_id")
    he=he.reindex(tr.contact_id.values).reset_index()
    d=pd.concat([tr.reset_index(drop=True),he.drop(columns=["contact_id"])],axis=1)
    return d

def main():
    tr=load_ours("train"); te=load_ours("test")
    log("ours",tr.shape,te.shape)
    # castor extras
    ctr=pd.read_parquet(f"{ROOT}/workspaces/castor_alpha/ground_train_feats.parquet")
    cte=pd.read_parquet(f"{ROOT}/workspaces/castor_alpha/ground_test_feats.parquet")
    ccols=[c for c in ctr.columns if c!="contact_id" and ctr[c].dtype!=object and c in cte.columns]
    ctr=ctr.set_index("contact_id")[ccols].add_suffix("_c").reset_index()
    cte=cte.set_index("contact_id")[ccols].add_suffix("_c").reset_index()
    tr=tr.merge(ctr,on="contact_id",how="left"); te=te.merge(cte,on="contact_id",how="left")
    log("castor merged",tr.shape,te.shape)
    # rigel extras
    rtr=pd.read_parquet(f"{ROOT}/workspaces/rigel_delta/train_ground.parquet")
    rte=pd.read_parquet(f"{ROOT}/workspaces/rigel_delta/test_ground.parquet")
    drops={"contact_id","game_play","step","nfl_player_id_1","nfl_player_id_2","contact","is_ground","game_key",
           "nfl_player_id","target","datetime"}
    rcols=[c for c in rtr.columns if c not in drops and rtr[c].dtype!=object and c in rte.columns]
    rtr=rtr.set_index("contact_id")[rcols].add_suffix("_r").reset_index()
    rte=rte.set_index("contact_id")[rcols].add_suffix("_r").reset_index()
    tr=tr.merge(rtr,on="contact_id",how="left"); te=te.merge(rte,on="contact_id",how="left")
    log("rigel merged",tr.shape,te.shape)
    del ctr,cte,rtr,rte; gc.collect()

    folds=pd.read_csv(f"{ROOT}/workspaces/castor_alpha/folds.csv")
    tr=tr.merge(folds,on="game_play",how="left"); assert tr.fold.notna().all(); tr["fold"]=tr.fold.astype(int)
    drops={"contact_id","game_play","contact","target","fold","nfl_player_id_2_x","nfl_player_id_2_y"}
    feats=[c for c in tr.columns if c not in drops and tr[c].dtype!=object]
    cat=[c for c in ["position_code_1"] if c in feats]
    for c in cat:
        tr[c]=tr[c].fillna(-1).astype("int32"); te[c]=te[c].fillna(-1).astype("int32")
    log("features",len(feats))
    y=tr.target.values.astype(np.int8); foldv=tr.fold.values
    X=tr[feats].values.astype(np.float32); Xte=te[feats].values.astype(np.float32)
    ids=tr.contact_id.values; gps=tr.game_play.values; teids=te.contact_id.values
    del tr,te; gc.collect()
    oof=np.full(len(X),np.nan,np.float32); testp=np.zeros(len(Xte))
    params=dict(objective="binary",metric="auc",learning_rate=0.03,num_leaves=63,min_data_in_leaf=60,
                feature_fraction=0.6,bagging_fraction=0.85,bagging_freq=1,lambda_l2=1.0,max_bin=127,
                num_threads=4,verbose=-1,seed=123)
    for f in sorted(np.unique(foldv)):
        va=foldv==f; trn=~va
        dt=lgb.Dataset(X[trn],label=y[trn],feature_name=feats,categorical_feature=cat)
        dv=lgb.Dataset(X[va],label=y[va],reference=dt)
        bst=lgb.train(params,dt,num_boost_round=6000,valid_sets=[dv],
                      callbacks=[lgb.early_stopping(150,verbose=False),lgb.log_evaluation(400)])
        oof[va]=bst.predict(X[va],num_iteration=bst.best_iteration).astype(np.float32)
        testp+=bst.predict(Xte,num_iteration=bst.best_iteration)/len(np.unique(foldv))
        log("fold",f,"iter",bst.best_iteration,"auc",round(roc_auc_score(y[va],oof[va]),5))
        del dt,dv,bst; gc.collect()
    log("OOF AUC",roc_auc_score(y,oof))
    pd.DataFrame({"contact_id":ids,"y":y,"pred":oof,"fold":foldv,"game_play":gps}).to_parquet(f"{W}/oof_ground_gfus.parquet",index=False)
    pd.DataFrame({"contact_id":teids,"pred":testp.astype(np.float32)}).to_parquet(f"{W}/testpred_ground_gfus.parquet",index=False)
    log("DONE")

if __name__=="__main__":
    main()
