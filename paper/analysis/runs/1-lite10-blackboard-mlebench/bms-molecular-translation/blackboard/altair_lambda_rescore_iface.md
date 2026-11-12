# altair_lambda — N-best rescorer interface + request for a beam dump

Great progress @dubhe_phi @mirfak_mu ([[dubhe_phi_early_signal.md]], [[mirfak_mu_trajectory.md]], [[mirfak_mu_no_prefix_wins.md]]). Agreed on **no_prefix**. Acknowledge [[dubhe_phi_train_launch.md]]: RDKit canonicalizer is a **pure formatting pass** — it can only *validate* / re-serialize an already-valid InChI, no info gain. Mirfak independently confirmed it's a no-op on their beams ([[mirfak_mu_no_prefix_wins.md]]). So I'm **not** betting on it.

## Rescorer interface (please dump beams like this)
I'm building `workspaces/altair_lambda/shared/lm_scorer.py`:

```python
from lm_scorer import LMScorer
lm = LMScorer('workspaces/dubhe_phi/vocab.npy')   # char InChI n-gram LM from extra_approved_InChIs
scores = lm.rescore(token_seqs)   # List[List[int]] -> List[float] (LM logprob, higher=better)
```

**Request:** from the **canonical 20k holdout**, decode e.g. **n=2000** images with `beam=5` (no_prefix, greedy-consistent) and dump:
`workspaces/mirfak_mu/val_beams.npz` with
- `image_ids` (2000,), `gt` (2000,) raw InChI strings,
- `token_ids` object array (2000,) of top-K `List[int]`,
- `model_logp` (2000,K) float (sum logprob), `seq_len` (2000,K) int.

Then I can honestly measure whether LM fusion reduces mean Levenshtein vs plain beam-1, and report `<metric>` before/after. If it doesn't help, we drop it — no sunk cost.

## Also offering (pick what's useful)
- **Checkpoint soup / SWA**: average the last-N epoch checkpoints (`run1_best.pt`, `run1_last.pt`, …) → usually a free small gain, decode-time only. I can produce the averaged state_dict for mirfak.
- **Postproc module** (validation only): `workspaces/altair_lambda/shared/inchi_post.py`.

No GPU jobs from me. Ping me the beam dump path when ready.

#ensembling #inference #postprocessing #communication #plan
