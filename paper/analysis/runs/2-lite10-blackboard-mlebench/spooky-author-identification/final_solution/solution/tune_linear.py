import numpy as np, pandas as pd, time, json
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import log_loss
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import normalize

SEED=42
DATA='/download_job_code/payload/sandbox/data'; OUT='/download_job_code/payload/sandbox/workspaces/canopus_kappa'
tr=pd.read_csv(f'{DATA}/train.csv'); te=pd.read_csv(f'{DATA}/test.csv')
classes=['EAP','HPL','MWS']; yt=tr['author'].map({c:i for i,c in enumerate(classes)}).values
folds=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(tr,yt))
txt=tr['text'].values; txtf=te['text'].values

vw=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True,max_features=200000,strip_accents='unicode')
Xw=vw.fit_transform(txt); Tw=vw.transform(txtf)
vc=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),min_df=2,sublinear_tf=True,max_features=400000,strip_accents='unicode')
Xc=vc.fit_transform(txt); Tc=vc.transform(txtf)
print('word',Xw.shape,'char',Xc.shape,flush=True)

def cv_pred(name, make, X, T):
    oof=np.zeros((len(yt),3)); tp=np.zeros((T.shape[0],3)); t0=time.time()
    for k,(i,j) in enumerate(folds):
        m=make(); m.fit(X[i],yt[i])
        oof[j]=m.predict_proba(X[j]); tp+=m.predict_proba(T)/5
    s=log_loss(yt,oof); print(f'{name}: {s:.5f} ({time.time()-t0:.0f}s)',flush=True)
    np.save(f'{OUT}/oof_{name}.npy',oof); np.save(f'{OUT}/test_{name}.npy',tp); return s

res={}
for C in [1,3,10,30]:
    res[f'word_lr_C{C}']=cv_pred(f'word_lr_C{C}',lambda C=C: LogisticRegression(C=C,max_iter=2000,n_jobs=-1),Xw,Tw)
for C in [1,3,10,30]:
    res[f'char_lr_C{C}']=cv_pred(f'char_lr_C{C}',lambda C=C: LogisticRegression(C=C,max_iter=2000,n_jobs=-1),Xc,Tc)
for a in [1e-6,1e-5,1e-4]:
    res[f'word_sgd_a{a}']=cv_pred(f'word_sgd_a{a}',lambda a=a: SGDClassifier(loss='log_loss',alpha=a,max_iter=100,tol=1e-4,random_state=SEED,n_jobs=-1),Xw,Tw)
json.dump(res,open(f'{OUT}/tune1.json','w'),indent=2)

# blend word+char best OOF (geometric and arithmetic) to see headroom
def blend(a,b,w):
    return w*a+(1-w)*b
best_w=min([c for c in res if c.startswith('word_lr')],key=lambda k:res[k])
best_c=min([c for c in res if c.startswith('char_lr')],key=lambda k:res[k])
A=np.load(f'{OUT}/oof_{best_w}.npy'); B=np.load(f'{OUT}/oof_{best_c}.npy')
for w in [0.3,0.4,0.5,0.6,0.7]:
    print('blend',w,log_loss(yt,blend(A,B,w)))
