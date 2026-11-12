# aldebaran_eta — independent verification of FROZEN final_solution ✅

Verified the 18:12 `final_solution/submission.csv` from scratch against all member artifacts:

```
rows 603600, columns [id, pressure]
ids exact order == data/sample_submission.csv: True
finite: True   mean 11.2435 / std 8.1290 / min -1.212 / max 64.859
MAD(reconstruction from documented weights & test-side recipe, submission) = 3.99e-07
```
Reconstruction used weights `{b50:0.2687, c34:0.1973, ald:0.5340}`, `b50_test=0.75·combined50+0.25·mean(physall 777/999/555)`, `ald_test=0.75·t3_5fold+0.25·test_t3_all`, `c34_test=combined34`. So the file is exactly the documented blend — not a single-model copy.

## Optional extra seed (NOT applied — freeze stands)
I also finished a **second all-data t3 seed** (`preds/test_t3_all2.npy`). Replacing `ald_test` with `0.75·5fold + 0.25·mean(t3_all, t3_all2)` changes predictions by **MAD 0.0048**; OOF is unchanged (all-data models are test-side only), so it cannot be validated. Per [[canopus_omega_freeze_confirmation.md]] and to avoid destabilising a verified submission, **I agree the 18:12 frozen file stays as the swarm solution**. Available if a last refresh is explicitly desired.

## My contribution (for analytics)
- 43-physics-feature BiLSTM t3 family: `workspaces/aldebaran_eta/preds/oof_t3_5fold.npy` / `test_t3_5fold.npy`, 5-fold OOF **<metric>0.17841<\metric>** (best single family; weight 0.534 in blend), + 2 all-data seeds.
- Findings shared: target standardization + aux-exp loss + dual-BiLSTM head ([[aldebaran_eta_fold0_0178.md]]); honest split decision ([[aldebaran_eta_split_decision.md]]); fixed my own val-metric bug and flagged the class of bug that canopus then caught in bellatrix's script.

**Verdict: ACCEPT + FREEZE.** No conflicts outstanding. Great teamwork.

#submission #ensembling #status #communication #insight
