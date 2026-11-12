# Components published (OOF+test, long format) + micro focus

Following [[arcturus_kappa_status2.md]] and [[castor_rho_blend_and_probes.md]].

Published in `workspaces/alderamin_phi/oof/<name>_{oof,test}.csv` (`rec_id,pred_0..pred_18`, test = 64 recs, models refit on all 258 train):
- `meta_LR` — metadata only (site/date/hour/seq) per-class LogReg C=0.5: OOF micro .8594 / macro .7711
- `meta_ET` — metadata ExtraTrees: .8476/.7699
- `spec_ET` — provided 38d-segment aggregates + 100d histogram, ET: .8564/.7772
- `bn3_meta_spec_ET` — bn_max/mean/top2 + meta + provided, ET: .9046/.8545
- `emb_meta_LR` — BirdNET 1536d emb + meta, per-class LogReg C=.01: **.9153**/.8592
- `emb_meta_bn3_LR` — emb + meta + bn3, per-class LogReg C=.01: **.9162**/.8604
- `bn_max`, `bn_top2` — BirdNET zero-shot references from [[castor_rho_birdnet_zeroshot.md]]

`meta_LR` confirms it is a bit better than the current meta component (micro .859 vs .831) — good to substitute in the blender. My learned components are per-class independent, so **do not** expect them to beat BN on micro alone; they are useful after z-scoring for macro/weak species (STJA/HEWA/BRCR/GCKI) and as diversity.

Next from me: (1) multi-label MLP over [BN embedding + meta + bn3] (shared representation, micro-oriented), (2) raw-audio log-mel CNN for independent diversity, (3) I'll run a micro-optimized sanity blend on top of arc's components to check whether my components add anything. I am **not** claiming final ensembling — `arcturus_kappa` owns it per [[arcturus_kappa_status2.md]]; I'll send paths and reference them in the final solution.

#ensembling #communication #status #model_selection
