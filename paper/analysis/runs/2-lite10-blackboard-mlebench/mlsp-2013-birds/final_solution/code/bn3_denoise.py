import os, sys, time
os.environ['OMP_NUM_THREADS']='4'
sys.path.insert(0,'vendor')
import numpy as np, pandas as pd, librosa, onnxruntime as ort
base='/download_job_code/payload/sandbox'
rec2f={}
for line in open(f'{base}/data/essential_data/rec_id2filename.txt').read().splitlines()[1:]:
    a,b=line.split(','); rec2f[int(a)]=b
labels=[l.strip() for l in open('models/birdnet3/full/birdnet-v3.0-preview3.1-labels-b1.txt')]
idx={}
for i,l in enumerate(labels): idx.setdefault(l.split('_')[0],i)
SCI=["Certhia americana","Troglodytes pacificus","Empidonax difficilis","Sitta canadensis","Junco hyemalis","Contopus cooperi","Catharus guttatus","Poecile rufescens","Ixoreus naevius","Setophaga occidentalis","Catharus ustulatus","Empidonax hammondii","Piranga ludoviciana","Pheucticus melanocephalus","Regulus satrapa","Vireo gilvus","Geothlypis tolmiei","Cyanocitta stelleri","Chordeiles minor"]
tgt=[idx[s] for s in SCI]
so=ort.SessionOptions(); so.intra_op_num_threads=4; so.inter_op_num_threads=1
sess=ort.InferenceSession('models/birdnet3/full/birdnet-v3.0-preview3.1-fp32-b1.onnx',sess_options=so,providers=['CPUExecutionProvider'])
iname=sess.get_inputs()[0].name
def denoise(y16):
    D=librosa.stft(y16,n_fft=1024,hop_length=256); mag,phase=np.abs(D),np.angle(D)
    noise=np.percentile(mag,15,axis=1,keepdims=True)
    mask=np.clip((mag-1.5*noise)/(mag+1e-8),0,1)
    mask=np.apply_along_axis(lambda r:np.convolve(r,np.ones(5)/5,mode='same'),1,mask)
    return librosa.istft(mag*mask*np.exp(1j*phase),hop_length=256,length=len(y16)).astype(np.float32)
BATCH=8; SR=32000; W=5*SR; H=int(2.5*SR)
ids=sorted(rec2f); clips=[]; owner=[]
for rid in ids:
    y16,_=librosa.load(f'{base}/data/essential_data/src_wavs/{rec2f[rid]}.wav',sr=16000,mono=True)
    y=librosa.resample(denoise(y16),orig_sr=16000,target_sr=SR)
    if len(y)<W: y=np.pad(y,(0,W-len(y)))
    starts=list(range(0,len(y)-W+1,H))
    if starts[-1]+W<len(y): starts.append(len(y)-W)
    for st in starts: clips.append(y[st:st+W].astype('float32')); owner.append(rid)
print('clips',len(clips),flush=True)
P={r:[] for r in ids}; Em={r:[] for r in ids}; t0=time.time()
for s in range(0,len(clips),BATCH):
    b=clips[s:s+BATCH]; n=len(b)
    while len(b)<BATCH: b.append(np.zeros(W,dtype='float32'))
    out=sess.run(None,{iname:np.stack(b)})
    for j in range(n): P[owner[s+j]].append(out[0][j,tgt].copy()); Em[owner[s+j]].append(out[1][j].copy())
    if s%240==0: print(s,round(time.time()-t0,1),flush=True)
rows=[]
for rid in ids:
    A=np.stack(P[rid]); rows.append({'rec_id':rid,**{f'pred_{k}':float(A[:,k].max()) for k in range(19)}})
pd.DataFrame(rows).sort_values('rec_id').to_csv('artifacts/bn3_dn_probs.csv',index=False)
np.savez_compressed('artifacts/bn3_dn_emb.npz',rec_id=np.array(ids),emb=np.stack([np.mean(Em[r],0) for r in ids]))
print('DONE',round(time.time()-t0,1),flush=True)
