import numpy as np, pandas as pd, time, json
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from scipy.sparse import csr_matrix
SEED=42
DATA='/download_job_code/payload/sandbox/data'; OUT='/download_job_code/payload/sandbox/workspaces/canopus_kappa'
tr=pd.read_csv(f'{DATA}/train.csv'); te=pd.read_csv(f'{DATA}/test.csv')
classes=['EAP','HPL','MWS']; y=tr['author'].map({c:i for i,c in enumerate(classes)}).values
folds=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(tr,y))
txt=tr['text'].values; txtf=te['text'].values

def nb_scale(Xtr,ytr,c,alpha=1.0):
    pos=np.asarray(Xtr[ytr==c].sum(0)).ravel()+alpha
    neg=np.asarray(Xtr[ytr!=c].sum(0)).ravel()+alpha
    r=np.log((pos/pos.sum())/(neg/neg.sum()))
    return csr_matrix(r)

def run(name, vec, alpha=1.0, C=1.0):
    X=vec.fit_transform(txt).tocsr(); T=vec.transform(txtf).tocsr()
    Xb=(X>0).astype(np.float64).tocsr() if hasattr(X,'astype') else X
    # keep non-binary tfidf? use binary counts
    X=(X>0).astype(np.float64).tocsr(); T=(T>0).astype(np.float64).tocsr()
    oof=np.zeros((len(y),3)); tp=np.zeros((T.shape[0],3)); t0=time.time()
    for i,j in folds:
        for c in range(3):
            r=nb_scale(X[i],y[i],c,alpha)
            m=LogisticRegression(C=C,max_iter=3000,solver='liblinear')
            m.fit(X[i].multiply(r), (y[i]==c).astype(int))
            oof[j,c]=m.predict_proba(X[j].multiply(r))[:,1]
            tp[:,c]+=m.predict_proba(T.multiply(r))[:,1]/5
    oof/=oof.sum(1,keepdims=True); tp/=tp.sum(1,keepdims=True)
    s=log_loss(y,oof); print(f'{name} a={alpha} C={C}: {s:.5f} ({time.time()-t0:.0f}s)',flush=True)
    np.save(f'{OUT}/oof_{name}_a{alpha}_C{C}.npy',oof); np.save(f'{OUT}/test_{name}_a{alpha}_C{C}.npy',tp)
    return s

v1=CountVectorizer(ngram_range=(1,2),min_df=2,binary=True)
v2=CountVectorizer(analyzer='char_wb',ngram_range=(2,5),min_df=3,binary=True)
res={}
for a in [1.0]:
    for C in [0.5,1.0,3.0]:
        res[f'word_a{a}_C{C}']=run('nbsvm_word',v1,a,C)
for a in [1.0]:
    for C in [0.5,1.0]:
        res[f'char_a{a}_C{C}']=run('nbsvm_char',v2,a,C)
json.dump(res,open(f'{OUT}/nbsvm.json','w'),indent=2)
