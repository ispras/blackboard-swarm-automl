import pandas as pd, numpy as np, gc
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import lightgbm as lgb
R="/download_job_code/payload/sandbox"; W=f"{R}/workspaces/castor_alpha"
OFF=290.0; FPS=5.994

def neighbor_feats(tr):
    tr=tr[["game_play","step","nfl_player_id","x_position","y_position"]].drop_duplicates(["game_play","step","nfl_player_id"])
    outs=[]
    for (gp,st),d in tr.groupby(["game_play","step"],sort=False):
        xy=d[["x_position","y_position"]].to_numpy(np.float32); pid=d.nfl_player_id.to_numpy()
        n=len(d)
        if n<2:
            outs.append(pd.DataFrame({"game_play":gp,"step":st,"nfl_player_id":pid,"nn_dist":np.nan,"nn2_dist":np.nan,"cnt15":0,"cnt25":0,"cnt35":0})); continue
        diff=xy[:,None,:]-xy[None,:,:]
        dist=np.sqrt((diff*diff).sum(-1)); np.fill_diagonal(dist,np.inf)
        nn=dist.min(1)
        srt=np.sort(dist,axis=1)
        outs.append(pd.DataFrame({"game_play":gp,"step":st,"nfl_player_id":pid,"nn_dist":nn,"nn2_dist":srt[:,1],
            "cnt15":(dist<1.5).sum(1),"cnt25":(dist<2.5).sum(1),"cnt35":(dist<3.5).sum(1)}))
    return pd.concat(outs,ignore_index=True)

def build(is_train):
    if is_train:
        lab=pd.read_csv(f"{R}/data/train_labels.csv")
        g=lab[lab.nfl_player_id_2.astype(str)=="G"].copy()
        g=g.rename(columns={"nfl_player_id_1":"pid"}); g["y"]=g.contact.astype(np.int8)
    else:
        ss=pd.read_csv(f"{R}/data/sample_submission.csv")
        parts=ss.contact_id.str.split("_",n=4,expand=True)
        ss["game_play"]=parts[0]+"_"+parts[1]; ss["step"]=parts[2].astype(int)
        ss["pid"]=parts[3].astype(int)
        g=ss[parts[4]=="G"].copy(); g["y"]=0
    g=g.reset_index(drop=True)
    g["frame"]=np.round(OFF+g.step.astype(float)*FPS).astype(int)
    # tracking
    tr=pd.read_csv(f"{R}/data/{'train' if is_train else 'test'}_player_tracking.csv")
    tr=tr[["game_play","step","nfl_player_id","speed","distance","acceleration","sa","x_position","y_position","orientation","direction","team","position"]].copy()
    nb=neighbor_feats(tr)
    g=g.merge(nb,left_on=["game_play","step","pid"],right_on=["game_play","step","nfl_player_id"],how="left")
    g=g.merge(tr,left_on=["game_play","step","pid"],right_on=["game_play","step","nfl_player_id"],how="left",suffixes=("","_tr"))
    g["team_home"]=(g.team=="home").astype(np.int8) if "team" in g else 0
    # temporal tracking feats
    g=g.sort_values(["game_play","pid","step"]).reset_index(drop=True)
    grp=g.groupby(["game_play","pid"],sort=False)
    for c in ["speed","acceleration","sa","y_position","x_position"]:
        g[c+"_lag1"]=grp[c].shift(1); g[c+"_lead1"]=grp[c].shift(-1)
        g[c+"_d1"]=g[c]-g[c+"_lag1"]
        g[c+"_m5"]=grp[c].transform(lambda s: s.rolling(5,center=True,min_periods=1).mean())
        g[c+"_std5"]=grp[c].transform(lambda s: s.rolling(5,center=True,min_periods=1).std())
        g[c+"_min5"]=grp[c].transform(lambda s: s.rolling(5,center=True,min_periods=1).min())
        g[c+"_max5"]=grp[c].transform(lambda s: s.rolling(5,center=True,min_periods=1).max())
    # helmets
    h=pd.read_csv(f"{R}/data/{'train' if is_train else 'test'}_baseline_helmets.csv",
                  usecols=["game_play","view","frame","nfl_player_id","left","width","top","height"])
    h=h[h.view.isin(["Sideline","Endzone"])].copy()
    h["bottom"]=h.top+h.height; h["cy"]=h.top+h.height/2; h["cx"]=h.left+h.width/2
    h["area"]=h.width*h.height
    key=["game_play","view","frame"]
    h["bot_rank"]=h.groupby(key).bottom.rank(pct=True)
    h["cy_rank"]=h.groupby(key).cy.rank(pct=True)
    h["n_helm"]=h.groupby(key).bottom.transform("size")
    h["bot_gap"]=h.groupby(key).bottom.transform("max")-h.bottom
    hcols=["game_play","view","frame","nfl_player_id","left","width","top","height","bottom","cy","cx","area","bot_rank","cy_rank","n_helm","bot_gap"]
    hh=h[hcols]
    for view,suf in [("Sideline","s"),("Endzone","e")]:
        hv=hh[hh.view==view].drop(columns="view")
        acc=None
        for off in [0,-1,1,-2,2,-3,3]:
            cur=g[["game_play","frame","pid"]].copy(); cur["frame"]=cur.frame+off
            cur=cur.merge(hv,left_on=["game_play","frame","pid"],right_on=["game_play","frame","nfl_player_id"],how="left")
            cur=cur.rename(columns={c:f"{c}_{suf}" for c in hcols if c not in ["game_play","view","frame","nfl_player_id"]})
            cur=cur[[c for c in cur.columns if c.startswith(("left","width","top","height","bottom","cy","cx","area","bot_rank","cy_rank","n_helm","bot_gap"))]]
            if acc is None: acc=cur
            else:
                for c in cur.columns:
                    acc[c]=acc[c].fillna(cur[c])
        acc[f"vis_{suf}"]=acc[f"width_{suf}"].notna().astype(np.int8)
        g=pd.concat([g.reset_index(drop=True),acc.reset_index(drop=True)],axis=1)
    # cross-view
    for c in ["width","height","bottom","cy","bot_rank","area","n_helm"]:
        g[c+"_avg"]=g[[f"{c}_s",f"{c}_e"]].mean(axis=1)
    g["bot_rank_min"]=g[["bot_rank_s","bot_rank_e"]].min(axis=1)
    g["vis_n"]=g.vis_s+g.vis_e
    for view in ["s","e"]:
        g[f"w_{view}_n"]=g[f"width_{view}"]/1280.0
        g[f"h_{view}_n"]=g[f"height_{view}"]/720.0
        g[f"bot_{view}_n"]=g[f"bottom_{view}"]/720.0
        g[f"cy_{view}_n"]=g[f"cy_{view}"]/720.0
    keep=[c for c in g.columns if c not in ["datetime","contact","nfl_player_id","nfl_player_id_2","nfl_player_id_tr"]]
    return g[keep].copy(), [c for c in keep if c not in ["game_play","pid","y","contact_id"]]

if __name__=="__main__":
    tr,gf=build(True); print("train",tr.shape,flush=True)
    te,gf2=build(False); print("test",te.shape,flush=True)
    feats=[c for c in gf if c in te.columns and c in tr.columns]
    # ensure numeric
    for c in feats:
        tr[c]=pd.to_numeric(tr[c],errors="coerce"); te[c]=pd.to_numeric(te[c],errors="coerce")
    folds=pd.read_csv(f"{W}/folds.csv"); tr=tr.merge(folds,on="game_play",how="left")
    print("feats",len(feats),flush=True)
    oof=np.zeros(len(tr)); models=[]
    for f in sorted(tr.fold.unique()):
        va=tr.fold==f; trn=~va
        m=lgb.LGBMClassifier(n_estimators=3000,learning_rate=0.05,num_leaves=127,min_child_samples=50,
            subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,n_jobs=8,random_state=42,objective="binary")
        m.fit(tr.loc[trn,feats],tr.loc[trn,"y"],eval_set=[(tr.loc[va,feats],tr.loc[va,"y"])],
              eval_metric="auc",callbacks=[lgb.early_stopping(150)])
        oof[va]=m.predict_proba(tr.loc[va,feats])[:,1]
        models.append(m); print("fold",f,"best",m.best_iteration_,"auc",round(roc_auc_score(tr.loc[va,"y"],oof[va]),4),flush=True)
    auc=roc_auc_score(tr.y,oof); print("OOF AUC",round(auc,4),flush=True)
    best=(0,0.5)
    for th in np.unique(np.quantile(oof,np.linspace(0.5,0.999,450))):
        mcc=matthews_corrcoef(tr.y,(oof>=th).astype(int))
        if mcc>best[0]: best=(mcc,th)
    print("OOF MCC",round(best[0],4),"th",best[1],flush=True)
    tr[["contact_id","game_play","pid","step","y","fold"]].assign(oof=oof).to_parquet(f"{W}/ground_oof_tab.parquet",index=False)
    te_pred=np.mean([m.predict_proba(te[feats])[:,1] for m in models],axis=0)
    te[["contact_id","game_play","pid","step"]].assign(pred=te_pred).to_parquet(f"{W}/ground_test_tab.parquet",index=False)
    tr[["contact_id"]+feats].to_parquet(f"{W}/ground_train_feats.parquet",index=False)
    te[["contact_id"]+feats].to_parquet(f"{W}/ground_test_feats.parquet",index=False)
    imp=pd.Series(models[0].feature_importances_,index=feats).sort_values(ascending=False)
    print(imp.head(25).to_string(),flush=True)
    print("DONE",flush=True)
