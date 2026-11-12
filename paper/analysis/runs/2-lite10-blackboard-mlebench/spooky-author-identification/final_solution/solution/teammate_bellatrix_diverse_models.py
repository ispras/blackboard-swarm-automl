import numpy as np, pandas as pd, time, json, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neural_network import MLPClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss
from scipy.sparse import hstack

OUT=os.path.dirname(os.path.abspath(__file__)); DATA='/download_job_code/payload/sandbox/data'
tr=pd.read_csv(f'{DATA}/train.csv'); te=pd.read_csv(f'{DATA}/test.csv')
y=tr['author'].map({'EAP':0,'HPL':1,'MWS':2}).values
Xtxt=tr['text'].fillna('').values; Ttxt=te['text'].fillna('').values
fold_id=np.load(f'{OUT}/fold_id.npy')
folds=[(np.where(fold_id!=k)[0],np.where(fold_id==k)[0]) for k in range(5)]
N,M=len(tr),len(te)
names=['mlp_svd','nystroem_rbf','knn_svd']
oof={n:np.zeros((N,3)) for n in names}; tst={n:np.zeros((M,3)) for n in names}
t0=time.time()
for k,(itr,iva) in enumerate(folds):
    tw=TfidfVectorizer(ngram_range=(1,2),min_df=3,sublinear_tf=True,strip_accents='unicode',max_features=60000)
    Xw=tw.fit_transform(Xtxt[itr]); Vw=tw.transform(Xtxt[iva]); Tw=tw.transform(Ttxt)
    tc=TfidfVectorizer(ngram_range=(3,5),min_df=3,sublinear_tf=True,strip_accents='unicode',analyzer='char_wb',max_features=100000)
    Xc=tc.fit_transform(Xtxt[itr]); Vc=tc.transform(Xtxt[iva]); Tc=tc.transform(Ttxt)
    Xb=hstack([Xw,Xc]).tocsr(); Vb=hstack([Vw,Vc]).tocsr(); Tb=hstack([Tw,Tc]).tocsr()
    svd=TruncatedSVD(n_components=200,random_state=42); svd.fit(Xb)
    A=svd.transform(Xb); B=svd.transform(Vb); C=svd.transform(Tb)
    sc=StandardScaler(); A=sc.fit_transform(A); B=sc.transform(B); C=sc.transform(C)

    clf=MLPClassifier(hidden_layer_sizes=(256,64),alpha=1e-3,batch_size=128,learning_rate_init=1e-3,
                      max_iter=120,early_stopping=True,n_iter_no_change=8,validation_fraction=0.1,random_state=42)
    clf.fit(A,y[itr]); oof['mlp_svd'][iva]=clf.predict_proba(B); tst['mlp_svd']+=clf.predict_proba(C)/5

    ny=Nystroem(kernel='rbf',gamma=1.0/(2*200),n_components=400,random_state=42)
    An=ny.fit_transform(A); Bn=ny.transform(B); Cn=ny.transform(C)
    clf=LogisticRegression(C=1.0,max_iter=3000); clf.fit(An,y[itr])
    oof['nystroem_rbf'][iva]=clf.predict_proba(Bn); tst['nystroem_rbf']+=clf.predict_proba(Cn)/5

    knn=KNeighborsClassifier(n_neighbors=60,weights='distance',metric='cosine',n_jobs=4)
    knn.fit(normalize(A),y[itr])
    oof['knn_svd'][iva]=knn.predict_proba(normalize(B)); tst['knn_svd']+=knn.predict_proba(normalize(C))/5
    print(f'fold {k} done {time.time()-t0:.0f}s',flush=True)

res={}
for n in names:
    ll=log_loss(y,oof[n],labels=[0,1,2]); res[n]=round(float(ll),5)
    np.save(f'{OUT}/oof_{n}.npy',oof[n]); np.save(f'{OUT}/test_{n}.npy',tst[n])
    print(n,ll,flush=True)
json.dump(res,open(f'{OUT}/diverse_results.json','w'),indent=2)
print(json.dumps(res,indent=2))
