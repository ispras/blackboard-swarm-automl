"""Save per-window Perch v2 embeddings (1536-d) for MIML."""
import os, sys, time
os.environ["HF_HOME"]="/workspace/.cache/huggingface"
sys.path.insert(0,"/download_job_code/payload/sandbox/workspaces/rigel_omega")
sys.path.insert(0,"/download_job_code/payload/sandbox/workspaces/rigel_omega/perchv2_pytorch_src")
import numpy as np, torch
from perchv2_pytorch import PerchV2, load_audio, frame_audio
DATA="/download_job_code/payload/sandbox/data"
OUT="/download_job_code/payload/sandbox/workspaces/altair_theta/emb_perch_windows.npz"
def main():
    from functools import partial
    print=partial(__import__('builtins').print, flush=True)
    torch.set_num_threads(8)
    m=PerchV2.from_pretrained(embeddings_only=True).cuda().eval()
    r2f={}
    for line in open(f"{DATA}/essential_data/rec_id2filename.txt").read().splitlines()[1:]:
        r,fn=line.split(","); r2f[int(r)]=fn
    recs=sorted(r2f); E=[]; owner=[]; t0=time.time()
    with torch.inference_mode(), torch.autocast("cuda",dtype=torch.float16):
        for i,r in enumerate(recs):
            w=frame_audio(load_audio(f"{DATA}/essential_data/src_wavs/{r2f[r]}.wav")).cuda()
            e=m(w)["embedding"].float().cpu().numpy()
            E.append(e)
            owner += [i]*e.shape[0]
            if (i+1)%60==0: print(i+1, round(time.time()-t0,1), "windows/rec", e.shape[0], flush=True)
    E=np.vstack(E).astype(np.float32); owner=np.array(owner)
    np.savez(OUT, recs=np.array(recs), windows=E, owner=owner)
    print("saved",OUT,E.shape,"nwin",E.shape[0],flush=True)
if __name__=="__main__": main()
