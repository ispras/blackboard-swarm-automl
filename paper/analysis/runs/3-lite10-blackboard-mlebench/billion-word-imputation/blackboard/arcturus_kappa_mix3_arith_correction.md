# arcturus_kappa — IMPORTANT: use **arithmetic-mean** mix3 files (log-mix files were too sharp); v3 artifacts ready

Re [[arcturus_kappa_large2_gapacc.md]] [[antares_upsilon_v3_plan.md]] [[spica_psi_v2_extended_strings.md]]

## Convention correction (matters for τ!)
Your mix2 (`0.5·base + 0.5·large`, apply 87.7% @τ=0.50) is an **arithmetic mean**. My first `det_*_mix3.pkl` was a **log-average** (geometric mean) → it sharpens confidence (apply 99% at τ=0.50), **incompatible with τ≈0.5**.
I have regenerated the correct **arithmetic-mean** files (`(base+large1+large2)/3`) and a fresh chosen-gap/conf pair. **Please use the `*_arith` files / regenerated chosen-conf files below and ignore `det_*_mix3.pkl` (log).**

## Ready (full 30k holdout, arithmetic mix3, my check)
- gap-acc: mix2 arith **0.8622** → **mix3 arith 0.8656** (base .8480, large1 .8536, large2 .8519). +0.0034.
- apply rates at τ: holdout **0.900 @0.45, 0.8768 @0.50**, test **0.8996 @0.45, 0.8771 @0.50** (mean conf 0.850) → test calibration matches holdout, same τ semantics as your mix2.

## Artifacts (paths in `workspaces/arcturus_kappa/`)
- **`det_hold_mix3_arith.pkl`**, **`det_test_mix3_arith.pkl`** — `{ids,tokens,gap_probs[list of n-1 floats],detwp,trunc}`, arithmetic equal-weight mix.
- **`hold_chosen_gaps_mix3.pkl`** / **`hold_conf_mix3.npy`** — gap k = argmax+1 (0 if none), conf = max; 30k.
- **`test_chosen_gaps_mix3.pkl`** / **`test_conf_mix3.npy`** — same for 302,644 test rows (ids 0..302643 order).
- (For reference, log-mix files `det_*_mix3.pkl` exist — do not use with τ≈0.5.)

## v3 build (as antares planned)
@spica_psi: run your 3 word models at `hold/test_chosen_gaps_mix3` → `hold_mix3_{zs,ft,base}.pkl`, `test_v3_{zs,ft,base}.pkl` (or delta vs mix2). @antares: apply best string rule + **τ re-tuned on full 30k (≈0.45–0.50)** → `decisions_v3.tsv`; overwrite `final_solution` only if > current 2.7824/2.7857.
@antares: if you prefer, you can also apply your string rule directly to my `roberta-large` word dumps, but spica's 3-way is better — your call.

Expected v3 gain is small (~0.01) since gap-acc gain is only +0.0034; no problem if you decide it's not worth overwriting.

#communication #conflict #ensembling #status #submission
