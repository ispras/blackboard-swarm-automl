import os, time
os.environ["HF_HOME"]="/workspace/.cache/huggingface"
import numpy as np, torch, soundfile as sf, torchaudio
from transformers import ClapModel, ClapFeatureExtractor
MODEL="laion/clap-htsat-unfused"
OUT="/download_job_code/payload/sandbox/workspaces/altair_theta/emb_clap.npz"
DATA="/download_job_code/payload/sandbox/data"
def main():
    from functools import partial
    print=partial(__import__('builtins').print, flush=True)
    fe=ClapFeatureExtractor.from_pretrained(MODEL)
    m=ClapModel.from_pretrained(MODEL).cuda().eval()
    r2f={}
    for line in open(f"{DATA}/essential_data/rec_id2filename.txt").read().splitlines()[1:]:
        r,fn=line.split(","); r2f[int(r)]=fn
    recs=sorted(r2f); X=[]; t0=time.time()
    with torch.inference_mode():
        for i,r in enumerate(recs):
            y,sr=sf.read(f"{DATA}/essential_data/src_wavs/{r2f[r]}.wav")
            if y.ndim>1: y=y.mean(1)
            yt=torch.tensor(y,dtype=torch.float32)
            yt=torchaudio.functional.resample(yt,sr,48000)
            inp=fe([yt.numpy()],sampling_rate=48000,return_tensors="pt")
            f=m.get_audio_features(input_features=inp["input_features"].cuda()).float().cpu().numpy()
            X.append(f[0])
            if (i+1)%80==0: print(i+1, round(time.time()-t0,1), flush=True)
    np.savez(OUT, recs=np.array(recs), X=np.vstack(X))
    print("saved",OUT,np.vstack(X).shape,flush=True)
if __name__=="__main__": main()
