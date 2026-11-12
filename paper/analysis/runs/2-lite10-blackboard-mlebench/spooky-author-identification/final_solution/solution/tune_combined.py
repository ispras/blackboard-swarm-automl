import numpy as np, pandas as pd, time, json
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from scipy.sparse import hstack
SEED=42
DATA='/download_job_code/payload/sandbox/data'; OUT='/download_job_code/payload/sandbox/workspaces/canopus_kappa'
tr=pd.read_csv(f'{DATA}/train.csv'); te=pd.read_csv(f'{DATA}/test.csv')
classes=['EAP','HPL','MWS']; yt=tr['author'].map({c:i for i,c in enumerate(classes)}).values
folds=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(tr,yt))
txt=tr['text'].values; txtf=te['text'].values
res={}
def cv(name,X,T,C):
    oof=np.zeros((len(yt),3)); tp=np.zeros((T.shape[0],3)); t0=time.time()
    for i,j in folds:
        m=LogisticRegression(C=C,max_iter=3000,n_jobs=-1); m.fit(X[i],yt[i])
        oof[j]=m.predict_proba(X[j]); tp+=m.predict_proba(T)/5
    s=log_loss(yt,oof); print(f'{name} C={C}: {s:.5f} ({time.time()-t0:.0f}s)',flush=True)
    np.save(f'{OUT}/oof_{name}_C{C}.npy',oof); np.save(f'{OUT}/test_{name}_C{C}.npy',tp)
    res[f'{name}_C{C}']=s; return s
configs={
 'wc_md2': dict(w=dict(ngram_range=(1,2),min_df=2),c=dict(analyzer='char_wb',ngram_range=(2,5),min_df=2)),
 'wc_big': dict(w=dict(ngram_range=(1,3),min_df=1),c=dict(analyzer='char_wb',ngram_range=(2,6),min_df=1)),
 'wc_w1': dict(w=dict(ngram_range=(1,1),min_df=1),c=dict(analyzer='char_wb',ngram_range=(3,5),min_df=2)),
}
for name,cfg in configs.items():
    vw=TfidfVectorizer(sublinear_tf=True,strip_accents='unicode',max_features=300000,**cfg['w'])
    vc=TfidfVectorizer(sublinear_tf=True,strip_accents='unicode',max_features=500000,**cfg['c'])
    X=hstack([vw.fit_transform(txt),vc.fit_transform(txt)]).tocsr(); T=hstack([vw.transform(txtf),vc.transform(txtf)]).tocsr()
    print(name,X.shape,flush=True)
    for C in [10,30,100]:
        cv(name,X,T,C)
json.dump(res,open(f'{OUT}/tune2.json','w'),indent=2)
