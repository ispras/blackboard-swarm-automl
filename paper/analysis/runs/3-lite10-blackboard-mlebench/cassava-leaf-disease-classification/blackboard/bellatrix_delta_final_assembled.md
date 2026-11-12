# bellatrix_delta — FINAL assembled (TTA-upgraded) + one last multi-scale attempt

## Assembled now (freeze candidate)
`final_solution/submission.csv` = **uniform 8-model hedged ensemble**, per-model best TTA chosen on fold0.
**Fold0 (3,745 held-out) accuracy = <metric>0.89853</metric>**. `final_solution/` also has `README.md`, `ensemble_meta.json`, `ens_{val,test}_probs.npy`, `code/` (all agents' scripts).
Member fold0 accs (best TTA, honest; then 0.5*honest+0.5*FT hedge, agreement ≥0.948): B5@512 .89266, B5@640 .88999, ConvNeXt-B .89025, EVA02 .89079, CaFormer .88892, Swin .88598, ViT .88838, ConvNeXtV2 .89506.
Refs: [[vega_eta_tta_done.md]] [[aldebaran_epsilon_tta4_done.md]] [[bellatrix_delta_final_decision.md]]

## Last attempt (safe, fold0-selected)
GPU idle → running **multi-scale 4-flip TTA** for my 3 models (B5@512 {480,512,576}, B5@640 {576,640,704}, ConvNeXt {352,384,416}), writing to `run_*/tta4ms/`. I will swap a member only if its fold0 acc improves over the current TTA; then re-assemble. ETA ~06:25 → final freeze ~06:35 (well before 07:23).
**@vega_eta**: after I re-run `assemble_final.py` (~06:30) please run `verify_final.py` on `final_solution/submission.csv` (format + independent recompute). Ping me.

#submission #ensembling #inference #status #communication
