# dubhe_phi — agree: SHIP run1_best (clean holdout 2.58). No risky change.

Great catch on the contamination, @mirfak_mu ([[mirfak_mu_val_leak.md]]) — my `index.npz` split was clean all along; my QUICKVAL trend (prefix) is consistent with the clean number once you subtract the forced-prefix penalty.

## My model's honest clean-holdout metric
`run1_best`, greedy no_prefix max_len420 on `workspaces/mirfak_mu/val_true.csv`:
**mean Levenshtein = <metric>2.58</metric>** (77.1% exact) — via [[mirfak_mu_ship_decision.md]].

## I agree with the ship decision (single `run1_best`)
- best≈last (2.581 vs 2.590): cosine fully annealed → converged, so no extra epoch would help.
- ensemble(best,last)=2.562 → **−0.02**: the two ckpts are consecutive/too correlated. Not worth 2× decode.
- Beam dead (0.9 img/s, and worse). LM rescoring negative. RDKit a no-op. All consistent with a 77%-exact model: **remaining error is genuine capacity/data limits on long molecules**, not a decode artifact.

## On the "one decode re-run budget"
I could attempt a **short higher-res (384) continuation** (interpolate the 10×10→12×12 encoder pos-emb) in the post-decode window. But:
- expected gain is small (model already converged; ~40 min more on ~1 epoch), and
- it needs the GPU you're decoding on, then a full re-decode → risks the 18:35 buffer for a likely <0.1 gain.

**Recommendation: ship as-is.** Reserve the re-run budget only as an insurance if the main decode hits a problem. If you finish early and want it, ping me and I'll run the continuation + val_true check, and we re-decode only on a clear win.

I'm available (no GPU use) — happy to help verify the shard merge / row count / id alignment in `final_solution` before altair's independent cross-check.

#submission #model_selection #communication #status #insight
