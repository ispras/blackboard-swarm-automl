import numpy as np, pandas as pd, time, os, sys
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.metrics import log_loss
from scipy import sparse
from scipy.special import softmax

SEED=42
DATA='/download_job_code/payload/sandbox/data'
OUT='/download_job_code/payload/sandbox/workspaces/aldebaran_xi'
tr=pd.read_csv(f'{DATA}/train.csv'); te=pd.read_csv(f'{DATA}/test.csv')
classes=['EAP','HPL','MWS']
y=tr['author'].map({c:i for i,c in enumerate(classes)}).values
skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
folds=list(skf.split(tr,y))

def build_union(train_texts, test_texts):
    w=TfidfVectorizer(sublinear_tf=True,ngram_range=(1,2),min_df=2,max_df=0.95,
                      analyzer='word',token_pattern=r'\w{1,}',strip_accents='unicode')
    c=TfidfVectorizer(sublinear_tf=True,ngram_range=(3,5),min_df=3,max_df=0.95,
                      analyzer='char_wb',strip_accents='unicode')
    Xw=w.fit_transform(train_texts); Xw_t=w.transform(test_texts)
    Xc=c.fit_transform(train_texts); Xc_t=c.transform(test_texts)
    return sparse.hstack([Xw,Xc]).tocsr(), sparse.hstack([Xw_t,Xc_t]).tocsr()

def cv_model(make, name, X, Xt):
    oof=np.zeros((len(tr),3)); test=np.zeros((len(te),3))
    for f,(ti,vi) in enumerate(folds):
        m=make(); m.fit(X[ti],y[ti])
        oof[vi]=m.predict_proba(X[vi]); test+=m.predict_proba(Xt)/5
    ll=log_loss(y,oof)
    np.save(f'{OUT}/oof_{name}.npy',oof); np.save(f'{OUT}/test_{name}.npy',test)
    print(f'{name}: OOF logloss={ll:.4f}',flush=True)
    return ll

t0=time.time()
X,Xt=build_union(tr['text'].values,te['text'].values)
print('features',X.shape,Xt.shape,f'{time.time()-t0:.1f}s',flush=True)

cv_model(lambda: LogisticRegression(C=10,solver='liblinear',max_iter=2000),'lr',X,Xt)
cv_model(lambda: MultinomialNB(alpha=0.02),'nb',X,Xt)
cv_model(lambda: CalibratedClassifierCV(LinearSVC(C=0.1),cv=3),'svc',X,Xt)
print('done total',f'{time.time()-t0:.1f}s')
