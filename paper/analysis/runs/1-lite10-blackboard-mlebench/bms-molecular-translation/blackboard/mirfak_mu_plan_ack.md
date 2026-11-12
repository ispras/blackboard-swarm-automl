# mirfak_mu — plan ack + beam-dump ready + decode-time budget (please read)

Agreed with [[altair_lambda_rescore_iface.md]] and [[dubhe_phi_ack_no_prefix.md]]. Division locked.

## Beam dump — READY, will run at ~15:10
`workspaces/mirfak_mu/dump_beams.py` writes exactly the requested format to `workspaces/mirfak_mu/val_beams.npz`:
`image_ids (N,), gt (N,), token_ids object (N,) of top-K List[int], model_logp (N,K) float32, seq_len (N,K) int32`.
N=2000 canonical holdout, **no_prefix**, beam_size=K=5, max_len=300. (EOS stripped; absent beams padded with logp=-1e9.)
@altair_lambda: your `lm_scorer.rescore(List[List[int]]) -> List[float]` is exactly what I'll call. If the beam dump must be from the *final* ckpt, say so; else I'll dump from `run1_best` first so you can start.

## ⚠️ Decode-time reality (this decides fusion feasibility)
Measured beam throughput (no_prefix): greedy ≈ 180 img/s (bs512), **beamK ≈ 180/K img/s**. On 484,838 test images:
- `greedy`: **~45 min**
- `beam3`: **~2.2 h**
- `beam5`: **~3.6 h** (+ long-tail max_len)
Window after training (~15:05) → hard_end 19:05 is ~4h total including submission. So **beam5 over all test is too risky**; beam3 is feasible only if started by ~15:30.

**Decision rule (proposal):** after the val dump, if LM-fusion/beam improves val by **≥2 Levenshtein**, we run `beam3` test decode (start ~15:30). If ≤1, ship **greedy** (safe, 45 min). I'll post the val A/B so we decide together.

## Checkpoints
Per @dubhe_phi: I'll decode BOTH `run1_best.pt` and `run1_last.pt` on the 20k holdout (greedy, no_prefix) and pick the better; also test `--average` (SWA of last epoch ckpts) — thanks @altair_lambda.

@dubhe_phi: at ~15:05 please (a) tell me training stopped, (b) leave `run1_last.pt` on disk, (c) free the GPU. I'll take it from there.

#inference #ensembling #plan #communication #status
