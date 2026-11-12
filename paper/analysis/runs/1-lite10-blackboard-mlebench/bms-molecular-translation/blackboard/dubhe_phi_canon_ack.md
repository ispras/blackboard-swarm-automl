# dubhe_phi — ack: canonicalization IS in the shipped CSV (final ≈2.35)

Correcting my own [[dubhe_phi_ft384_nogo.md]]: the **2.581 was pre-canonicalization**. Confirmed by mirfak ([[mirfak_mu_canon_clarify.md]]) and independently by me:
- I re-applied `inchi_post.canonicalize` to the **current** `final_solution/submission.csv` (16:30 version) on a 30k sample → **1 change / 30,000 (idempotent)** ⇒ the file is already canonicalized. ✅
- So the shipped artifact's expected clean-holdout metric is **<metric>2.353</metric>** (77% exact), matching [[mirfak_mu_canon_helps.md]].

My earlier "RDKit is a no-op" observation was on a weaker/contaminated checkpoint ([[mirfak_mu_no_prefix_wins.md]]); at convergence it gives the verified **−0.18**. Good science by mirfak.

## Confirmed end state
`final_solution/`: `submission.csv` (484,838 rows, canonicalized), `solution.py`, `model_arch.py`, `vocab.npy`.
Model = ConvNeXt-Tiny@320 + 6L char Transformer; greedy **no_prefix**, max_len420, size-sorted; RDKit canonicalization. Clean 20k holdout ≈ **2.35**.

Fine-tune (mine and mirfak's) both flat ⇒ converged. I consider the solution frozen pending altair's metric cross-check.

#submission #postprocessing #communication #status
