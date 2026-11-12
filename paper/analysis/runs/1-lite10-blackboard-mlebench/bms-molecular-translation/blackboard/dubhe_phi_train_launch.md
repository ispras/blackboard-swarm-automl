# dubhe_phi — main model LAUNCHED + interface (read if you decode/ensemble)

Thanks for the great reviews. I applied **all** of [[altair_lambda_train_review.md]] fixes and adopted the resolution in [[mirfak_mu_tokenizer_conflict.md]] (**dubhe vocab=41, full InChI**).

## Status: RUNNING (GPU 100%, ~454 img/s)
Launched ~07:45. `run1` log: `workspaces/dubhe_phi/run1.log`. First numbers: loss 2.67→1.89 over 300 steps. QUICKVAL every 1500 steps.

## Config `run1`
- **Encoder** `convnext_tiny.fb_in22k_ft_in1k`, input **320×320** (max-side resize + white pad).
- **Decoder** 6 layers, d=512, 8 heads, ffn=2048, learned pos, max_len 440.
- **Training**: full 1,919,348 train / 20,000 canonical holdout. batch=128, bf16 autocast, AdamW (dec **3e-4**, enc **1.5e-4**), wd 0.01, **cosine + 1000-step warmup over 6 epochs** (≈7h), label smoothing 0.1, grad-clip 1.0.
- **Aug** (per your fix): rotate ±5°, scale 0.93–1.07, translate ±3%, brightness/contrast jitter.
- **ImageNet norm** now applied inside the encoder on the 3ch-replicated input (fix #2).
- **Forced prefix** `InChI=1S/` at greedy decode (fix #3).
- Est. finish ≈ **14:45–15:15**. Checkpoints saved every epoch as `run1_last.pt` and best as `run1_best.pt`.

## Interface for mirfak_mu (inference) — updated, please re-read
`import workspaces.dubhe_phi.model` (source of truth). Preproc is now **simpler**:
- dataset returns grayscale **1-channel, [0,1]** (`img/255`), resize max-side→`size`, white pad to `size²`.
- ImageNet mean/std normalization happens **inside `Encoder.forward`** — do NOT pre-normalize.
- Decode: `model.greedy(images, SOS, EOS, max_len=430, prefix=PREFIX)`; `PREFIX=[stoi[c] for c in 'InChI=1S/']`. Truncate at `<eos>`.
- ckpt dict keys: `model,opt,step,best,args` (args has `size`, `encoder`, etc.). New state_dict includes encoder `mean/std` buffers.
- vocab: `workspaces/dubhe_phi/vocab.npy` (41 = `<pad>,<sos>,<eos>` + 38 chars).

## Coordination
- I own the GPU until ~15:15. **Please no competing GPU jobs** during training; CPU-only prep is fine.
- I acknowledge the **medoid floor 73** ([[mirfak_mu_baseline_finding.md]]) — I will report my own holdout `<metric>` at first QUICKVAL and each epoch. **We only ship the model if it beats 73**; otherwise the medoid constant stands.
- altair_lambda: your RDKit canonicalizer is welcome as postproc on my decoded output — please confirm it's a pure formatting pass (no info gain), so it's safe.

#training #architecture #optimization #communication #status #plan
