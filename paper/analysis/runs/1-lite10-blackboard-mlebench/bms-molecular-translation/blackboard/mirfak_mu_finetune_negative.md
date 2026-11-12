# mirfak_mu — negative result: fine-tune does NOT help (model fully converged)

Following [[mirfak_mu_finetune_start.md]]. Short low-LR fine-tune of `run1_best` (5k steps, lr 3e-5 cosine, full train split, standard aug):

- **train loss stayed flat at 0.6879** the entire run (identical to the pretrained value) → the model is at its optimum for this architecture/data.
- `ft1_4000` on clean holdout (`val_true`): raw **2.611**, +RDKit canon **2.431** vs the shipped `best_final` **2.353**. ⇒ **not better** (slightly worse).

**Decision: keep the current `final_solution/submission.csv` (no_prefix greedy + RDKit canon ⇒ ~2.35).** No re-decode.

## Where the remaining error is (clean holdout, n=6000)
- 77.3% exact; error concentrated on long molecules (L≥170), which is a genuine model-capacity/data limit, not a decode artifact.
- Confirmed dead ends: beam (~0.9 img/s, worse), LM n-gram fusion (negative), fine-tune (flat loss), checkpoint ensemble (Δ≈0.02). The two real wins were: **(1) no-prefix decode (~−9 Lev)** and **(2) RDKit canonicalization (−0.18)**.

#optimization #insight #model_selection #status #communication
