import numpy as np, pandas as pd, time, os, json
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from scipy.sparse import hstack, csr_matrix

SEED=42
DATA='/download_job_code/payload/sandbox/data'
OUT='/download_job_code/payload/sandbox/workspaces/canopus_kappa'
tr=pd.read_csv(f'{DATA}/train.csv'); te=pd.read_csv(f'{DATA}/test.csv')
classes=['EAP','HPL','MWS']
yt=tr['author'].map({c:i for i,c in enumerate(classes)}).values
skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
folds=list(skf.split(tr,yt))

# ---------- vectorizers ----------
word=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True,max_features=200000,strip_accents='unicode')
char=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),min_df=3,sublinear_tf=True,max_features=300000,strip_accents='unicode')
Xw=word.fit_transform(tr['text']); Tw=word.transform(te['text'])
Xc=char.fit_transform(tr['text']); Tc=char.transform(te['text'])
X=hstack([Xw,Xc]).tocsr(); T=hstack([Tw,Tc]).tocsr()
print('X',X.shape,'T',T.shape,flush=True)

def softmax(z):
    z=z-z.max(axis=1,keepdims=True); e=np.exp(z); return e/e.sum(axis=1,keepdims=True)

def run_cv(name, make_model, X, T, proba_fn=None):
    oof=np.zeros((X.shape[0],3)); tp=np.zeros((T.shape[0],3)); t0=time.time()
    for k,(itr,iva) in enumerate(folds):
        m=make_model()
        m.fit(X[itr],yt[itr])
        if proba_fn is None:
            oof[iva]=m.predict_proba(X[iva]); tp+=m.predict_proba(T)/len(folds)
        else:
            oof[iva]=proba_fn(m,X[iva]); tp+=proba_fn(m,T)/len(folds)
    sc=log_loss(yt,oof); print(f'{name}: {sc:.5f}  ({time.time()-t0:.1f}s)',flush=True)
    np.save(f'{OUT}/oof_{name}.npy',oof); np.save(f'{OUT}/test_{name}.npy',tp)
    return sc

def nb_ratio(Xtr,ytr):
    p=np.asarray(Xtr[ytr==0].sum(0)).ravel()+1; q=np.asarray(Xtr[ytr==1].sum(0)).ravel()+1
    r=np.asarray(Xtr[ytr==2].sum(0)).ravel()+1
    return np.log((p/p.sum())/((p.sum()/p.sum()) if False else 1.0))

def run_nbsvm(C=1.0):
    # one-vs-rest NB-SVM: multiply features by log count ratio per class, calibrate via sigmoid on decision
    oof=np.zeros((X.shape[0],3)); tp=np.zeros((T.shape[0],3)); t0=time.time()
    for k,(itr,iva) in enumerate(folds):
        Xtr=X[itr]; ytr=yt[itr]; Xva=X[iva]
        for c in range(3):
            pos=np.asarray(Xtr[ytr==c].sum(0)).ravel()+1
            neg=np.asarray(Xtr[ytr!=c].sum(0)).ravel()+1
            r=np.log((pos/pos.sum())/(neg/neg.sum()))
            rm=csr_matrix(r)
            m=LinearSVC(C=C,class_weight='balanced',max_iter=5000)
            m.fit(Xtr.multiply(rm), (ytr==c).astype(int))
            for Xs,dst in ((Xva,'va'),(T,'te')):
                d=m.decision_function(Xs.multiply(rm))
                pr=1/(1+np.exp(-d))
                if dst=='va': oof[iva,c]=pr
                else: tp[:,c]+=pr/len(folds)
    oof=oof/oof.sum(1,keepdims=True); tp=tp/tp.sum(1,keepdims=True)
    sc=log_loss(yt,oof); print(f'nbsvm_lsvc: {sc:.5f} ({time.time()-t0:.1f}s)',flush=True)
    np.save(f'{OUT}/oof_nbsvm_lsvc.npy',oof); np.save(f'{OUT}/test_nbsvm_lsvc.npy',tp)

results={}
results['logreg']=run_cv('logreg',lambda: LogisticRegression(C=10.0,solver='lbfgs',max_iter=1000,n_jobs=-1),X,T)
results['sgd_mh']=run_cv('sgd_mh',lambda: SGDClassifier(loss='modified_huber',alpha=1e-5,max_iter=30,random_state=SEED,n_jobs=-1),X,T)
results['cnb']=run_cv('cnb',lambda: ComplementNB(alpha=1.0),X,T)
run_nbsvm(C=0.1)
json.dump(results,open(f'{OUT}/linear_results.json','w'),indent=2)
