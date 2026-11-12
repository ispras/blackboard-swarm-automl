"""Extract Perch v2 embeddings for augmented versions of each rec (for aug-trained probe)."""
import os, sys, time
os.environ["HF_HOME"]="/workspace/.cache/huggingface"
sys.path.insert(0,"/download_job_code/payload/sandbox/workspaces/rigel_omega/perchv2_pytorch_src")
import numpy as np, torch
from perchv2_pytorch import PerchV2, load_audio, frame_audio
DATA="/download_job_code/payload/sandbox/data"
OUT="/download_job_code/payload/sandbox/workspaces/altair_theta/emb_perch_aug.npz"
KAUG=3
def main():
    from functools import partial
    print=partial(__import__('builtins').print, flush=True)
    torch.manual_seed(0); gen=torch.Generator().manual_seed(0)
    torch.set_num_threads(8)
    m=PerchV2.from_pretrained(embeddings_only=True).cuda().eval()
    r2f={}
    for line in open(f"{DATA}/essential_data/rec_id2filename.txt").read().splitlines()[1:]:
        r,fn=line.split(","); r2f[int(r)]=fn
    recs=sorted(r2f); t0=time.time()
    allE=np.zeros((len(recs),KAUG,3072),np.float32)
    with torch.inference_mode(), torch.autocast("cuda",dtype=torch.float16):
        for i,r in enumerate(recs):
            wav=load_audio(f"{DATA}/essential_data/src_wavs/{r2f[r]}.wav")
            std=wav.std().clamp_min(1e-6)
            augs=[]
            # 1) shift by +1.5s (zero-pad front)
            s=int(1.5*32000)
            sh=torch.cat([torch.zeros(s), wav[:-s]]) if wav.numel()>s else wav
            augs.append(sh)
            # 2) additive noise ~ SNR
            augs.append(wav + 0.05*std*torch.randn(wav.shape, generator=gen))
            # 3) gain 0.6
            augs.append(0.6*wav)
            for k in range(KAUG):
                w=frame_audio(augs[k]).cuda()
                e=m(w)["embedding"].float().cpu().numpy()
                allE[i,k]=np.concatenate([e.mean(0), e.max(0)])
            if (i+1)%60==0: print(i+1, round(time.time()-t0,1), flush=True)
    np.savez(OUT, recs=np.array(recs), aug=allE)
    print("saved",OUT,allE.shape,"in",time.time()-t0,flush=True)
if __name__=="__main__": main()
