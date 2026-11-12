import os, json, numpy as np, math, datetime, csv
from functools import partial
print=partial(__import__('builtins').print, flush=True)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

DATA='/download_job_code/payload/sandbox/data'
OUT='/download_job_code/payload/sandbox/workspaces/altair_theta'
os.makedirs(OUT,exist_ok=True)

labels={}
with open(f'{DATA}/essential_data/rec_labels_test_hidden.txt') as f:
    next(f)
    for line in f:
        p=line.strip().split(','); rid=int(p[0]); rest=[x for x in p[1:] if x!='']
        labels[rid]=('?' if (rest and rest[0]=='?') else set(int(x) for x in rest))
fold={}
with open(f'{DATA}/essential_data/CVfolds_2.txt') as f:
    next(f)
    for line in f:
        r,fo=line.strip().split(','); fold[int(r)]=int(fo)
fn={}
with open(f'{DATA}/essential_data/rec_id2filename.txt') as f:
    next(f)
    for line in f:
        r,f2=line.strip().split(','); fn[int(r)]=f2
train_ids=sorted(r for r in fold if fold[r]==0)
test_ids=sorted(r for r in fold if fold[r]==1)

def load_mat(path):
    rows={}
    with open(path) as f:
        next(f)
        for line in f:
            p=line.strip().rstrip(',').split(',')
            rows[int(p[0])]=np.array([float(x) for x in p[1:]])
    return rows
hist=load_mat(f'{DATA}/supplemental_data/histogram_of_segments.txt')

stations=sorted(set(fn[r].split('_')[0] for r in fn)); st_idx={s:i for i,s in enumerate(stations)}
def meta(rid):
    p=fn[rid].split('_'); st=p[0]; date=p[1]; t=p[2]
    y=int(date[:4]); mo=int(date[4:6]); d=int(date[6:8])
    doy=datetime.date(y,mo,d).timetuple().tm_yday
    hh=int(t[:2]); mm=int(t[2:4]); ss=int(t[4:6]); mins=hh*60+mm+ss/60
    v=[1.0 if i==st_idx[st] else 0.0 for i in range(len(stations))]
    v+=[math.sin(2*math.pi*doy/365),math.cos(2*math.pi*doy/365),1.0 if y==2010 else 0.0]
    v+=[math.sin(2*math.pi*mins/1440),math.cos(2*math.pi*mins/1440),mins/1440.0]
    return np.array(v)
def build(ids):
    return np.array([np.concatenate([hist.get(r,np.zeros(100)),meta(r)]) for r in ids])

Xtr=build(train_ids); Xte=build(test_ids)
Y=np.zeros((len(train_ids),19))
for i,r in enumerate(train_ids):
    if isinstance(labels.get(r),set):
        for c in labels[r]: Y[i,c]=1

probs=np.zeros((len(test_ids),19))
for c in range(19):
    y=Y[:,c]
    if y.sum()==0:
        probs[:,c]=0.0; continue
    m=lgb.LGBMClassifier(n_estimators=400,learning_rate=0.03,num_leaves=15,min_child_samples=5,
        subsample=0.8,colsample_bytree=0.7,reg_lambda=1.0,verbose=-1,n_jobs=8)
    m.fit(Xtr,y)
    probs[:,c]=m.predict_proba(Xte)[:,1]
print("test probs stats", probs.min(), probs.max(), probs.mean())

# write submission preserving sample order
rows=[]
with open(f'{DATA}/sample_submission.csv') as f:
    r=csv.DictReader(f)
    for row in r:
        Id=int(row['Id']); rid=Id//100; sp=Id%100
        if rid in test_ids and sp<19:
            rows.append((Id, probs[test_ids.index(rid),sp]))
        else:
            rows.append((Id, 0.0))
os.makedirs('/download_job_code/payload/sandbox/final_solution',exist_ok=True)
with open('/download_job_code/payload/sandbox/final_solution/submission.csv','w',newline='') as f:
    f.write('Id,Probability\n')
    for Id,p in rows:
        f.write(f'{Id},{p:.6f}\n')
print("wrote submission", len(rows), "rows")
