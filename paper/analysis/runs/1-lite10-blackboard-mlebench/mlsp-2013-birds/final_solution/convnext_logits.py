"""BirdSet ConvNeXT-Base direct 19-species logits (5s windows, hop 2.5s), sigmoid, mean/max pooled."""
import os, time
os.environ["HF_HOME"]="/workspace/.cache/huggingface"
import numpy as np, torch, torchaudio
DATA="/download_job_code/payload/sandbox/data"
OUT="/download_job_code/payload/sandbox/workspaces/altair_theta/convnext_birdset_probs.npz"
MODEL="DBD-research-group/ConvNeXT-Base-BirdSet-XCL"
IDX=[7206,6306,4810,7191,8871,4786,7111,7770,8457,6435,7109,4806,6934,6952,8098,5813,6394,6055,485]
def rec2file():
    d={}
    for line in open(f"{DATA}/essential_data/rec_id2filename.txt").read().splitlines()[1:]:
        r,fn=line.split(","); d[int(r)]=fn
    return d
def main():
    from transformers import ConvNextForImageClassification
    dev="cuda"; model=ConvNextForImageClassification.from_pretrained(MODEL).to(dev).eval()
    mel=torchaudio.transforms.MelSpectrogram(sample_rate=32000,n_fft=1024,hop_length=320,n_mels=128,power=2.0).to(dev)
    r2f=rec2file(); recs=sorted(r2f); win=5*32000; hop=32000; t0=time.time()
    allimgs=[]; owner=[]
    for i,r in enumerate(recs):
        y,sr=torchaudio.load(f"{DATA}/essential_data/src_wavs/{r2f[r]}.wav"); y=y.mean(0,keepdim=True)
        if sr!=32000: y=torchaudio.functional.resample(y,sr,32000)
        y=y.to(dev)
        if y.shape[1]<=win: segs=[y]
        else:
            segs=[y[:,s:s+win] for s in range(0,y.shape[1]-win+1,hop)]
        for seg in segs:
            if seg.shape[1]<win: seg=torch.nn.functional.pad(seg,(0,win-seg.shape[1]))
            S=mel(seg); S=torch.log(S+1e-6); S=(S-S.mean())/(S.std()+1e-6)
            S=torch.nn.functional.interpolate(S.unsqueeze(0),size=(224,224),mode="bilinear",align_corners=False)
            allimgs.append(S[0]); owner.append(i)
    X=torch.stack(allimgs); owner=np.array(owner)
    print("images",X.shape,flush=True)
    probs=[]
    with torch.inference_mode(), torch.autocast("cuda",dtype=torch.float16):
        for i in range(0,X.shape[0],64):
            lg=model(X[i:i+64]).logits.float()[:, IDX].cpu().numpy()
            probs.append(1/(1+np.exp(-lg)))
    P=np.vstack(probs)
    n=len(recs); mean=np.zeros((n,19),np.float32); mx=np.zeros((n,19),np.float32)
    for j in range(n):
        rows=P[owner==j]; mean[j]=rows.mean(0); mx[j]=rows.max(0)
    np.savez(OUT, recs=np.array(recs), mean=mean, max=mx, idx=np.array(IDX))
    print("saved",OUT,mean.shape,"in",time.time()-t0,flush=True)
if __name__=="__main__": main()
