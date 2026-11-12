"""Save full AST patch-token tensor for all recs (fp16) to enable pooling variants.
Output: ast_tokens.npz  recs, tokens [N,T,768] fp16, cls [N,768], logits [N,527]
Also saves TTA -1s shifted tokens (np.roll) pooled later.
"""
import os, sys, time
os.environ["HF_HOME"]="/workspace/.cache/huggingface"
import numpy as np
DATA="/download_job_code/payload/sandbox/data"
OUT="/download_job_code/payload/sandbox/workspaces/altair_theta/ast_tokens.npz"
MODEL="MIT/ast-finetuned-audioset-10-10-0.4593"
def rec2file():
    d={}
    with open(f"{DATA}/essential_data/rec_id2filename.txt") as f:
        f.readline()
        for line in f:
            line=line.strip()
            if line:
                r,fnn=line.split(","); d[int(r)]=fnn
    return d
def main():
    import torch, soundfile as sf
    from transformers import AutoFeatureExtractor, ASTForAudioClassification
    dev="cuda" if torch.cuda.is_available() else "cpu"
    print("device",dev,flush=True)
    fe=AutoFeatureExtractor.from_pretrained(MODEL)
    model=ASTForAudioClassification.from_pretrained(MODEL).to(dev).eval()
    r2f=rec2file(); recs=sorted(r2f); N=len(recs)
    tokens=[]; cls=[]; logits=[]
    t0=time.time(); bs=16
    with torch.no_grad():
        for i in range(0,N,bs):
            batch=recs[i:i+bs]; wavs=[]
            for r in batch:
                y,sr=sf.read(f"{DATA}/essential_data/src_wavs/{r2f[r]}.wav")
                if y.ndim>1: y=y.mean(1)
                wavs.append(y.astype(np.float32))
            inp=fe(wavs,sampling_rate=16000,return_tensors="pt",padding=True)
            out=model(inp["input_values"].to(dev),output_hidden_states=True)
            h=out.hidden_states[-1]
            logits.append(torch.sigmoid(out.logits).float().cpu().numpy())
            cls.append(h[:,0,:].float().cpu().numpy())
            pat=h[:,2:,:] if h.shape[1]>500 else h
            tokens.append(pat.half().cpu().numpy())
            print(f"{i+len(batch)}/{N} T={h.shape[1]} {time.time()-t0:.1f}s",flush=True)
    tokens=np.concatenate(tokens,0); cls=np.vstack(cls); logits=np.vstack(logits)
    np.savez(OUT, recs=np.array(recs), tokens=tokens, cls=cls, logits=logits)
    print("saved",OUT,tokens.shape,cls.shape,logits.shape,flush=True)
if __name__=="__main__":
    main()
