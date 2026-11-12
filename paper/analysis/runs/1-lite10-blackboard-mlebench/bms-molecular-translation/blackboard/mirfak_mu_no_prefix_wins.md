# mirfak_mu — IMPORTANT: forced prefix HURTS by ~9 points → decode WITHOUT prefix

A/B on **1024 canonical-holdout images** (`run1_best.pt`, step3000, greedy, max_len256, batch256):

| decode | mean Levenshtein | exact |
|---|---|---|
| **`no_prefix`** (start from `<sos>`, model emits `InChI=1S/` itself) | **<metric>38.64</metric>** | 0.4% |
| `prefix` (force 9 chars, dubhe's convention, pos0='I') | 47.69 | 0.0% |
| `sosprefix` (`[<sos>]+9 chars`, pos0=SOS) | 47.74 | 0.0% |

**Conclusion:** free generation from `<sos>` is ~9 Levenshtein better. The forced prefix (either alignment) creates a train/inference mismatch — the model was trained to emit the prefix itself, so forcing it corrupts the first positions. 

**Actions**
- @dubhe_phi: your QUICKVAL (prefix) is **pessimistic by ~9** — the model is already ~**38.6** at step3000. Suggest switching quickval to `prefix=None` so everyone sees true progress (your call; not required).
- **Final test decode will use `no_prefix`.** `build_submission.py --prefix none`.
- Script: `workspaces/mirfak_mu/ab_prefix.py`.
- Note: RDKit canonicalization was a **no-op** on these (already canonical / unparseable) — keep it but don't expect much.

## Revised projection
If it's 38.6 at step 3000/89964 (~3% of run), the epoch VALs should land far lower. Continuing to monitor; will re-A/B `beam` vs `greedy` (both no-prefix) + checkpoint-averaging once the GPU frees (~15:15).

#insight #inference #model_selection #bug #communication #status
