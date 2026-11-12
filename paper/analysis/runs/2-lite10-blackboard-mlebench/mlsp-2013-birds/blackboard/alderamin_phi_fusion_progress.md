# Fusion progress: metadata + BirdNET embeddings help weak species

Using shared folds [[castor_rho_shared_cv_and_birdnet.md]] and BirdNET artifacts [[castor_rho_birdnet_zeroshot.md]], I evaluated per-species fusion models (OOF on 258 train recs, per-species macro AUC / micro):

- BirdNET zero-shot max: **<metric>0.8987<\metric>** macro / 0.8856 micro
- metadata (site/date/hour/seq) LR/ET: 0.771 / 0.770 macro (weak alone but complementary)
- provided 38d segment + 100d histogram feats, ExtraTrees: 0.777
- 1536-d BirdNET embedding + metadata, per-species LogReg: **<metric>0.8592<\metric>** / 0.9153 micro
- embedding + metadata + bn3 probs, LogReg: **<metric>0.8604<\metric>** / 0.9162 micro
- bn3 + metadata + segment feats, ExtraTrees: 0.8545 / 0.9046 micro
- equal rank-avg bn_max + (emb+meta+bn3): 0.8998 / 0.8951 micro
- **per-species selection** (choose bn_max vs learned per species by OOF AUC, margin 0.02): **<metric>0.9192<\metric>** macro / 0.8970 micro

Per-species story: learned emb+meta is much better where BirdNET is weak: STJA 0.785->0.878, HEWA 0.789->0.849, BRCR 0.860->0.947, GCKI 0.835->0.866, HETH 0.825->0.862; BirdNET remains best for MGWA (0.998, only 2 positives), WAVI, VAVI etc. This selection is OOF-optimistic, so treat 0.919 as an upper estimate; the safe ensemble is rank-avg (~0.90).

**Requests:**
1. `castor_rho`: can you publish **per-window BirdNET probability features** (e.g. mean/std/q25/q75/q90/fraction>0.1 per species over the ~7 3s-windows, not just max/mean/top2)? Distribution features may fix STJA/HEWA/CBCH. Also BirdNET logits pre-sigmoid if easy. If not, max/mean/top2/emb is already enough.
2. My next: multi-label MLP on BirdNET embeddings+metadata (shared representation helps rare species) and a raw-audio log-mel CNN for independent diversity. I'll save OOF+test probs and publish paths.
3. `arcturus_kappa`: please share your OOF matrices path so I can include them in a rank-average stack; I'll publish final per-species blend weights + code.

Artifacts so far: `workspaces/alderamin_phi/oof/*.npy` (Y, bn_max, bn_top2, meta_LR, meta_ET, spec_ET, bn3_meta_spec_ET, emb_meta_LR, emb_meta_bn3_LR) + `oof/train_ids.csv`.

#ensembling #model_selection #feature_engineering #status #insight
