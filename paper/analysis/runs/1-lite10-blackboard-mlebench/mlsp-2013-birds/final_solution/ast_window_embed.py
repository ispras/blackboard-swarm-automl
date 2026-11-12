"""Windowed / patch-pooled AST features (altair_theta).

For each of the 322 recs compute:
  - full-clip AudioSet logits (527)
  - full-clip AST patch-token max/mean pooled embeddings (768 each)
  - windowed (2s, hop 1s) AudioSet logits max/mean (527 each)
  - windowed pooler_output max/mean (768 each)
Saves workspaces/altair_theta/ast_window_feats.npz
"""
import os, sys, time
sys.path.insert(0, "/download_job_code/payload/sandbox/workspaces/altair_theta")
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
import numpy as np

DATA = "/download_job_code/payload/sandbox/data"
OUT = "/download_job_code/payload/sandbox/workspaces/altair_theta/ast_window_feats.npz"
MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"

def rec2file():
    d = {}
    with open(f"{DATA}/essential_data/rec_id2filename.txt") as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line: continue
            r, fn = line.split(",")
            d[int(r)] = fn
    return d

def main():
    import torch, soundfile as sf
    from transformers import AutoFeatureExtractor, ASTForAudioClassification
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", dev, flush=True)
    fe = AutoFeatureExtractor.from_pretrained(MODEL)
    model = ASTForAudioClassification.from_pretrained(MODEL).to(dev).eval()
    r2f = rec2file(); recs = sorted(r2f.keys())
    N = len(recs)
    FLO = np.zeros((N, 527), np.float32)          # full logits
    PMAX = np.zeros((N, 768), np.float32); PMEAN = np.zeros((N, 768), np.float32)  # patch pooled
    WLOmax = np.zeros((N, 527), np.float32); WLOmean = np.zeros((N, 527), np.float32)
    WPmax = np.zeros((N, 768), np.float32); WPmean = np.zeros((N, 768), np.float32)

    t0 = time.time(); bs = 16
    with torch.no_grad():
        for i in range(0, N, bs):
            batch = recs[i:i+bs]; wavs = []
            for r in batch:
                y, sr = sf.read(f"{DATA}/essential_data/src_wavs/{r2f[r]}.wav")
                if y.ndim > 1: y = y.mean(1)
                wavs.append(y.astype(np.float32))
            inp = fe(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
            iv = inp["input_values"].to(dev)
            out = model(iv, output_hidden_states=True)
            FLO[i:i+len(batch)] = torch.sigmoid(out.logits).float().cpu().numpy()
            h = out.hidden_states[-1]  # [B, T, 768]
            # drop 2 special tokens if present
            pat = h[:, 2:, :] if h.shape[1] > 500 else h
            PMAX[i:i+len(batch)] = pat.max(1).values.float().cpu().numpy()
            PMEAN[i:i+len(batch)] = pat.mean(1).float().cpu().numpy()
            print(f"full {i+len(batch)}/{N} {time.time()-t0:.1f}s", flush=True)

    # windowed 2s hop 1s
    win = 2*16000; hop = 1*16000
    t1 = time.time()
    with torch.no_grad():
        for i in range(N):
            r = recs[i]
            y, sr = sf.read(f"{DATA}/essential_data/src_wavs/{r2f[r]}.wav")
            if y.ndim > 1: y = y.mean(1)
            y = y.astype(np.float32)
            segs = []
            if len(y) <= win:
                segs = [y]
            else:
                for s in range(0, len(y)-win+1, hop):
                    segs.append(y[s:s+win])
                if (len(y)-win) % hop != 0:
                    segs.append(y[-win:])
            inp = fe(segs, sampling_rate=16000, return_tensors="pt", padding=True)
            iv = inp["input_values"].to(dev)
            out = model(iv, output_hidden_states=True)
            lo = torch.sigmoid(out.logits).float().cpu().numpy()
            hh = out.hidden_states[-1]
            po = hh[:, 0, :].float().cpu().numpy()  # CLS token
            WLOmax[i] = lo.max(0); WLOmean[i] = lo.mean(0)
            WPmax[i] = po.max(0); WPmean[i] = po.mean(0)
            if (i+1) % 50 == 0:
                print(f"win {i+1}/{N} {time.time()-t1:.1f}s", flush=True)
    np.savez(OUT, recs=np.array(recs), full_logits=FLO, patch_max=PMAX, patch_mean=PMEAN,
             win_logits_max=WLOmax, win_logits_mean=WLOmean,
             win_pool_max=WPmax, win_pool_mean=WPmean)
    print("saved", OUT, "in", time.time()-t0, flush=True)

if __name__ == "__main__":
    main()
