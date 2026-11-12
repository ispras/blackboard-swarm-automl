import pandas as pd, numpy as np, os, json, time, gc
from scipy.stats import rankdata

ROOT="/download_job_code/payload/sandbox"; W=f"{ROOT}/workspaces"
A=f"{W}/achernar_sigma"; C=f"{W}/castor_alpha"; R=f"{W}/rigel_delta"
t0=time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]",*a,flush=True)

def parse_ids(df):
    p=df.contact_id.str.split("_",expand=True)
    df["game_play"]=p[0]+"_"+p[1]; df["step"]=p[2].astype(int)
    df["p1"]=p[3].astype(np.int64)
    df["p2"]=p[4]
    return df

def mcc_conf(tp,fp,fn,tn):
    tp,fp,fn,tn=map(float,(tp,fp,fn,tn)); den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return (tp*tn-fp*fn)/den if den>0 else 0.0

def mcc_at_rate(p,y,rate):
    th=np.quantile(p,1-rate); pred=p>=th
    tp=np.sum(pred&(y==1)); fp=np.sum(pred&(y==0)); fn=np.sum((~pred)&(y==1)); tn=np.sum((~pred)&(y==0))
    return mcc_conf(tp,fp,fn,tn),th

def load_model(oof_path,col,test_path=None,test_col=None):
    d=pd.read_parquet(oof_path)
    y=d["contact"].values if "contact" in d.columns else d["y"].values
    o=pd.DataFrame({"contact_id":d.contact_id.values,"p":d[col].values})
    t=None
    if test_path is not None:
        dd=pd.read_parquet(test_path); tc=test_col or col
        if tc not in dd.columns:
            for c in ["pred","prob","oof","p"]:
                if c in dd.columns: tc=c; break
        t=pd.DataFrame({"contact_id":dd.contact_id.values,"p":dd[tc].values})
    return o,y,t

def build_pair():
    specs={
      "track":(f"{R}/oof_pair.parquet","prob",f"{R}/testpred_pair.parquet","prob"),
      "pairh":(f"{R}/oof_pair_h.parquet","prob",f"{R}/testpred_pair_h.parquet","prob"),
      "pstack":(f"{A}/oof_pair_gstack.parquet","pred",f"{A}/testpred_pair_gstack.parquet","pred"),
      "v1":(f"{A}/oof_pair_lgbv1.parquet","pred",f"{A}/testpred_pair_lgbv1.parquet","pred"),
      "v2":(f"{A}/oof_pair_lgbv2.parquet","pred",f"{A}/testpred_pair_lgbv2.parquet","pred"),
      "rstack":(f"{R}/oof_pair_stack_r.parquet","prob",f"{R}/testpred_pair_stack_r.parquet","prob"),
    }
    if os.path.exists(f"{A}/oof_pair_lgbv3.parquet"):
        specs["v3"]=(f"{A}/oof_pair_lgbv3.parquet","pred",f"{A}/testpred_pair_lgbv3.parquet","pred")
    skip=set(x for x in os.environ.get("SKIP_PAIR","").split(",") if x)
    for k in list(specs):
        if k in skip: specs.pop(k)
    seq_oof=[f"{R}/oof_pair_seq_s{i}.parquet" for i in range(5)]
    seq_test=[f"{R}/testpred_pair_seq_s{i}.parquet" for i in range(5)]
    base=None; y=None; data={}; testids=None; tdata={}
    for k,(of,oc,tf,tc) in specs.items():
        o,yy,t=load_model(of,oc,tf,tc)
        if base is None: base=o[["contact_id"]].copy(); y=yy
        else: base=base.merge(o[["contact_id"]],on="contact_id",how="inner")
        data[k]=o.set_index("contact_id")["p"]
        if t is not None:
            if testids is None: testids=t[["contact_id"]].copy()
            tdata[k]=t.set_index("contact_id")["p"]
    # seq average ranks
    r_o=[]; r_t=[]
    use_seq="seq" not in skip
    for of in seq_oof:
        d=pd.read_parquet(of); r_o.append(pd.Series(rankdata(d["prob"].values)/len(d),index=d.contact_id.values))
    for tf in seq_test:
        d=pd.read_parquet(tf); r_t.append(pd.Series(rankdata(d["prob"].values)/len(d),index=d.contact_id.values))
    seq_o=pd.concat(r_o,axis=1).mean(axis=1); seq_t=pd.concat(r_t,axis=1).mean(axis=1)
    if use_seq:
        data["seq"]=seq_o; tdata["seq"]=seq_t
    base=parse_ids(base)
    base["y"]=base.contact_id.map(pd.Series(y,index=pd.read_parquet(specs["track"][0]).contact_id.values)).values
    base=base.sort_values(["game_play","p1","p2","step"]).reset_index(drop=True)
    names=list(data.keys())
    ranks={k:rankdata(data[k].reindex(base.contact_id).values)/len(base) for k in names}
    # test ranks
    testids=parse_ids(testids); testids=testids.sort_values(["game_play","p1","p2","step"]).reset_index(drop=True)
    tranks={k:rankdata(tdata[k].reindex(testids.contact_id).values)/len(testids) for k in names}
    return base, names, ranks, testids, tranks

def build_ground():
    specs={
      "track":(f"{R}/oof_ground.parquet","prob",f"{R}/testpred_ground.parquet","prob"),
      "gfus":(f"{A}/oof_ground_gfus.parquet","pred",f"{A}/testpred_ground_gfus.parquet","pred"),
      "v2":(f"{A}/oof_ground_lgbv2.parquet","pred",f"{A}/testpred_ground_lgbv2.parquet","pred"),
      "gs1":(f"{A}/oof_ground_gstack.parquet","pred",f"{A}/testpred_ground_gstack.parquet","pred"),
      "gs2":(f"{A}/oof_ground_gstack2.parquet","pred",f"{A}/testpred_ground_gstack2.parquet","pred"),
      "cvid":(f"{C}/ground_oof_video.parquet","pred",f"{C}/ground_test_video.parquet","pred"),
      "ceff":(f"{C}/ground_oof_video_eff.parquet","pred",f"{C}/ground_test_video_eff.parquet","pred"),
      "ctab":(f"{C}/ground_oof_tab.parquet","oof",f"{C}/ground_test_tab.parquet","pred"),
      "cr34":(f"{C}/ground_oof_video_r34.parquet","pred",f"{C}/ground_test_video_r34.parquet","pred"),
      "cnx":(f"{C}/ground_oof_video_cnx.parquet","pred",f"{C}/ground_test_video_cnx.parquet","pred"),
      "rstack":(f"{R}/ground_oof_stack_r.parquet","prob",f"{R}/ground_test_stack_r.parquet","prob"),
    }
    base=None; y=None; data={}; tdata={}; testids=None
    skip=set(x for x in os.environ.get("SKIP_GROUND","").split(",") if x)
    for k in list(specs):
        if k in skip: specs.pop(k)
    for k,(of,oc,tf,tc) in specs.items():
        o,yy,t=load_model(of,oc,tf,tc)
        if base is None: base=o[["contact_id"]].copy(); y=yy
        else: base=base.merge(o[["contact_id"]],on="contact_id",how="inner")
        data[k]=o.set_index("contact_id")["p"]
        if testids is None: testids=t[["contact_id"]].copy()
        tdata[k]=t.set_index("contact_id")["p"]
    base=parse_ids(base)
    yser=pd.read_parquet(specs["track"][0])[["contact_id","contact"]].set_index("contact_id")["contact"]
    base["y"]=base.contact_id.map(yser).values
    base=base.sort_values(["game_play","p1","step"]).reset_index(drop=True)
    names=list(data.keys())
    ranks={k:rankdata(data[k].reindex(base.contact_id).values)/len(base) for k in names}
    testids=parse_ids(testids); testids=testids.sort_values(["game_play","p1","step"]).reset_index(drop=True)
    tranks={k:rankdata(tdata[k].reindex(testids.contact_id).values)/len(testids) for k in names}
    return base,names,ranks,testids,tranks

def smooth_arr(vals,gid,w):
    return pd.Series(vals).groupby(gid,sort=False).transform(lambda x:x.rolling(w,center=True,min_periods=1).mean()).values

def search(base,names,ranks,ws,rate,seed=0):
    y=base.y.values.astype(np.float64)
    if "p2" in base.columns:
        gid=base.groupby(["game_play","p1","p2"],sort=False).ngroup().values
    else:
        gid=base.groupby(["game_play","p1"],sort=False).ngroup().values
    # pre-smooth per model per w
    S={w:{k:smooth_arr(ranks[k],gid,w) for k in names} for w in ws}
    rng=np.random.RandomState(seed); best=(-1,None,None,None)
    starts=[np.ones(len(names))/len(names)]
    for _ in range(5): starts.append(rng.dirichlet(np.ones(len(names))))
    for w in ws:
        Sl=[S[w][k] for k in names]
        for st in starts:
            wv=st.copy(); sc=mcc_at_rate(sum(wv[i]*Sl[i] for i in range(len(names))),y,rate)[0]
            for it in range(4):
                imp=False
                for i in range(len(names)):
                    for d in [0.1,-0.1,0.05,-0.05,0.02,-0.02]:
                        w2=wv.copy(); w2[i]=max(0,w2[i]+d); w2=w2/w2.sum()
                        s2=mcc_at_rate(sum(w2[j]*Sl[j] for j in range(len(names))),y,rate)[0]
                        if s2>sc+1e-7: sc=s2; wv=w2; imp=True
                if not imp: break
            if sc>best[0]: best=(sc,w,wv.copy(),True)
    return best,S

def apply_test(testids,tranks,names,best,S,rate,tag):
    sc,w,wv,_=best
    if "p2" in testids.columns:
        gid=testids.groupby(["game_play","p1","p2"],sort=False).ngroup().values
    else:
        gid=testids.groupby(["game_play","p1"],sort=False).ngroup().values
    St=[smooth_arr(tranks[k],gid,w) for k in names]
    p=sum(wv[i]*St[i] for i in range(len(names)))
    th=np.quantile(p,1-rate)
    out=pd.DataFrame({"contact_id":testids.contact_id.values,tag:(p>=th).astype(int)})
    return out, th

def main():
    log("build pair")
    pb,pn,pr,pt,ptr=build_pair()
    log("pair loaded",pb.shape,pn)
    pb_best,pb_S=search(pb,pn,pr,[3,5],0.0120,seed=1)
    sp,w_p,wv_p,_=pb_best
    log("PAIR best",sp,"w",w_p,"weights",dict(zip(pn,np.round(wv_p,3))))
    tp_out,thp=apply_test(pt,ptr,pn,pb_best,pb_S,0.0120,"pair")
    yp=pb.y.values.astype(np.float64)
    poof=sum(wv_p[i]*pb_S[w_p][k] for i,k in enumerate(pn))
    m_p,thp2=mcc_at_rate(poof,yp,0.0120)
    log("pair OOF mcc",m_p,"oof_th",thp2,"test_th",thp)
    gc.collect()

    log("build ground")
    gb,gn,gr,gt,gtr=build_ground()
    log("ground loaded",gb.shape,gn)
    gb_best,gb_S=search(gb,gn,gr,[7,9,11,15],0.0450,seed=2)
    sg_,w_g,wv_g,_=gb_best
    log("GROUND best",sg_,"w",w_g,"weights",dict(zip(gn,np.round(wv_g,3))))
    tg_out,thg=apply_test(gt,gtr,gn,gb_best,gb_S,0.0450,"ground")
    yg=gb.y.values.astype(np.float64)
    goof=sum(wv_g[i]*gb_S[w_g][k] for i,k in enumerate(gn))
    m_g,thg2=mcc_at_rate(goof,yg,0.0450)
    log("ground OOF mcc",m_g,"oof_th",thg2,"test_th",thg)

    pp=poof>=thp2; pg=goof>=thg2; ypp=yp==1; ygg=yg==1
    tp=np.sum(pp&ypp)+np.sum(pg&ygg); fp=np.sum(pp&~ypp)+np.sum(pg&~ygg)
    fn=np.sum(~pp&ypp)+np.sum(~pg&ygg); tn=np.sum(~pp&~ypp)+np.sum(~pg&~ygg)
    jm=mcc_conf(tp,fp,fn,tn)
    log("PAIR mcc",m_p,"GROUND mcc",m_g,"JOINT mcc",jm)

    sub=pd.read_csv(f"{ROOT}/data/sample_submission.csv")
    v=pd.Series(index=sub.contact_id.values,dtype=float)
    v.loc[tp_out.contact_id.values]=tp_out["pair"].values.astype(float)
    v.loc[tg_out.contact_id.values]=tg_out["ground"].values.astype(float)
    assert v.notna().all()
    sub["contact"]=v.loc[sub.contact_id.values].astype(int).values
    out=f"{A}/submission_best.csv"; sub.to_csv(out,index=False)
    gsel=sub.contact_id.str.endswith("_G")
    log("wrote",out,"rows",len(sub),"overall",round(sub.contact.mean(),5),
        "pair",round(sub.loc[~gsel,"contact"].mean(),5),"ground",round(sub.loc[gsel,"contact"].mean(),5))
    json.dump({"pair":{"w":w_p,"weights":dict(zip(pn,map(float,wv_p))),"rate":0.012,"oof_mcc":float(m_p)},
               "ground":{"w":w_g,"weights":dict(zip(gn,map(float,wv_g))),"rate":0.045,"oof_mcc":float(m_g)},
               "joint_oof_mcc":float(jm)},open(f"{A}/best_blend.json","w"),indent=2)

if __name__=="__main__":
    main()
