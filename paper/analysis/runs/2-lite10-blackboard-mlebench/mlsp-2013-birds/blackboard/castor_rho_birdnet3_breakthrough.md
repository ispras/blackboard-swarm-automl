# BirdNET v3.0 is ALSO a game changer: zero-shot micro 0.9716 / macro 0.9781 (better than Perch), decorrelated

Following [[castor_rho_perch_breakthrough.md]] and [[arcturus_kappa_final_updated.md]]. Found the new **BirdNET v3.0 preview3.1** (Cornell/ Chemnitz, EfficientNetV2-S, 11,560 species) on HF `tphakala/BirdNET-v3.0-Models`; all 19 target species present (note `Empidonax difficilis` is now labeled “Western Flycatcher”, mapping by scientific name).
- Input **5s @ 32kHz** (160000 samples), same frontend as Perch; outputs sigmoid probabilities + 1280-d embeddings.
- Ran 3 windows / 2.5s hop, max over windows on all 322 WAVs (`onnxruntime` unpacked into my vendor dir; fp32 ONNX 557MB).
- **bn3_max: zero-shot macro <metric>0.9781<\metric> / micro 0.9716** (Perch: 0.9753/.9614; BirdNET v2.4: .9082/.8927). bn3_mean: .9760/.9681.
- **Correlation with Perch only 0.269** (with BN v2.4: 0.78) -> simple Perch+bn3(0.5) already macro .9836 / micro .9748 before any prior/stacking.

## Components published (`workspaces/castor_rho/oof/`)
`bn3_b3_max_{oof,test}.csv`, `bn3_b3_mean_{oof,test}.csv`; embeddings `artifacts/birdnet3_emb.npz` (1280-d, rec_id aligned); model `workspaces/castor_rho/models/birdnet3/full/...onnx`.

## My blend with it (greedy on shared folds)
`0.5*perch + 1.5*bn3_max + 1.5*bn3_mean + 0.2*mlp_perch + 0.1*prior_g + 0.05*meta_LR + 0.05*pw + 0.8*logit(prev)` (per-class z)
- OOF micro **<metric>0.9889<\metric>** / macro 0.9876.
- Honest per-fold (z inside fold): micro 0.9811 / macro 0.9876 (arc final: ~0.9823/0.9833) — micro is tied within noise, **macro +0.004**, so bn3 mainly strengthens rare/weak species.
`arcturus_kappa`: please test adding `bn3_b3_max` (+ mean) to the canonical blend; I think it raises the ceiling even if the honest 52-rec micro check is noisy. `alderamin_phi`: Perch/BN3 embeddings both worth MLP probing; bn3 emb is 1280-d.

#insight #feature_engineering #inference #ensembling #model_selection #status
