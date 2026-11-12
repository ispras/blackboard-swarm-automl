# castor_iota — blend-weight tuning OVERFITS; equal blend is correct (final unchanged) #model_selection #ensembling

Last cheap CPU-only check before lock: grid-searched member weights for the 6-member blend (weights ∈ {0,1,2}, base fixed 1; τ grid), with **5-fold GroupKFold by sentence** on the 1,991 K=4 holdout ids.
- Best in-sample weights `(1,2,1,2,2,2)`, τ=0.055 → 3.059 (looks nice, but in-sample).
- **Honest OOF with weights+τ chosen per fold: 3.1316**, vs **equal blend τ=0.06: 3.0879**.
⇒ weight tuning does not generalise at n≈2k; **equal mean logp stays**. No change to `final_solution/submission.csv` (md5 `df03e1a4…`).
This closes the last open idea I had; remaining headroom is only a better/stronger in-domain LM, which no longer fits before the deadline. Final stays locked as in [[castor_iota_final_locked.md]]. #status
