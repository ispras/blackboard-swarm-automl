"""BirdSet ConvNeXT-Base (spectrogram image model) embeddings.
5 s windows (hop 2.5 s) at 32 kHz, 128-mel log-spectrogram resized to 224x224,
penultimate 1024-d pooler pooled mean/max over windows.
Saves workspaces/altair_theta/emb_convnext_birdset.npz (recs, mean, max).
"""
import os, time
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
import numpy as np, torch, torchaudio
DATA="/download_job_code/payload/sandbox/data"
OUT="/download_job_code/payload/sandbox/workspaces/altair_theta/emb_convnext_birdset.npz"
MODEL="DBD-research-group/ConvNeXT-Base-BirdSet-XCL"

def rec2file():
    d={}
    for line in open(f"{DATA}/essential_data/rec_id2filename.txt").read().splitlines()[1:]:
        r,fn=line.split(","); d[int(r)]=fn
    return d

def main():
    from transformers import ConvNextForImageClassification
    dev="cuda"
    model=ConvNextForImageClassification.from_pretrained(MODEL).to(dev).eval()
    mel=torchaudio.transforms.MelSpectrogram(sample_rate=32000, n_fft=1024, hop_length=320, n_mels=128, power=2.0).to(dev)
    r2f=rec2file(); recs=sorted(r2f)
    win=5*32000; hop=32000  # 5s window, 1s hop -> 6 windows
    allimgs=[]; owner=[]
    t0=time.time()
    for i,r in enumerate(recs):
        y,sr=torchaudio.load(f"{DATA}/essential_data/src_wavs/{r2f[r]}.wav")
        y=y.mean(0,keepdim=True)
        if sr!=32000: y=torchaudio.functional.resample(y,sr,32000)
        y=y.to(dev)
        segs=[]
        if y.shape[1]<=win:
            segs=[y]
        else:
            for s in range(0, y.shape[1]-win+1, hop): segs.append(y[:,s:s+win])
            if segs[-1].shape[1] < win: pass
        for seg in segs:
            if seg.shape[1] < win:
                seg=torch.nn.functional.pad(seg,(0,win-seg.shape[1]))
            S=mel(seg)               # 1,128,T
            S=torch.log(S+1e-6)
            S=(S-S.mean())/(S.std()+1e-6)
            S=S.unsqueeze(0)         # 1,1,128,T
            S=torch.nn.functional.interpolate(S,size=(224,224),mode="bilinear",align_corners=False)
            allimgs.append(S[0]); owner.append(i)
        if (i+1)%40==0: print(f"prep {i+1}/{len(recs)} {time.time()-t0:.1f}s",flush=True)
    X=torch.stack(allimgs)           # N,1,224,224
    print("images",X.shape,flush=True)
    embs=[]
    with torch.inference_mode(), torch.autocast("cuda",dtype=torch.float16):
        for i in range(0,X.shape[0],64):
            out=model.convnext(X[i:i+64]).pooler_output.float().cpu().numpy()
            embs.append(out)
    E=np.vstack(embs); owner=np.array(owner)
    n=len(recs); mean=np.zeros((n,1024),np.float32); mx=np.zeros((n,1024),np.float32)
    for j in range(n):
        rows=E[owner==j]
        mean[j]=rows.mean(0); mx[j]=rows.max(0)
    np.savez(OUT, recs=np.array(recs), mean=mean, max=mx, windows=E, owner=owner)
    print("saved",OUT,mean.shape,mx.shape,"in",time.time()-t0,flush=True)

if __name__=="__main__": main()
