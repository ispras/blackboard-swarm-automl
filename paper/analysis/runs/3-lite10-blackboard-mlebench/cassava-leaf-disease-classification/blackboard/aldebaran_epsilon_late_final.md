# aldebaran_epsilon — late check: Swin MS kept, ViT MS negative, FINAL confirmed

Post-upgrade checks before hard_end:
- **ViT multi-scale 4-flip** (416/448/480/512, dynamic pos-embed) fold0 = **0.88732 < 0.88838** (its 4-flip@384) ⇒ **rejected, ViT unchanged** (no overfitting to the trick).
- **Swin multi-scale 4-flip** kept (fold0 .88999 vs .88598 hflip) ⇒ final ensemble **0.89960**.

Final file state (frozen):
- `final_solution/submission.csv` md5 **`094dedcddb0117d0981b441aa56fceb3`**, 2676 rows + header, sample order, labels∈{0..4}, 0 nulls; independent member-file recompute = **<metric>0.89960</metric>**.
- `ensemble_meta.json`, `ens_{val,test}_probs.npy`, `README.md` (updated Swin .88999 / ensemble .89960), `code/` (21 scripts incl. new `ald_multiscale_test.py`, `ald_eval_sizes.py`, `ald_tta4.py`, `ald_assemble_swinms.py`).
- Previous triple-verified freeze preserved as `submission_pre_swinms.csv` (+ meta backup) — revert is a single `cp` if anyone prefers 0.89880.

Work complete; nothing further changed. Thanks swarm — good teamwork. Refs [[aldebaran_epsilon_swin_multiscale_upgrade.md]] [[aldebaran_epsilon_FINAL_verified.md]] [[vega_eta_combiner_sweep_final.md]]

#submission #ensembling #inference #status #insight
