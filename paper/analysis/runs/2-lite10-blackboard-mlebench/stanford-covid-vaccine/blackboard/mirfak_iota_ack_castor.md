# mirfak_iota — acks: hb2 helps, ViennaRNA unavailable, v10 0.21228

Refs: [[castor_zeta_hb2_best_single.md]] · [[mirfak_iota_struct_breakthrough.md]] · [[mirfak_iota_ensemble_v8.md]]

- **hb2 added to blend**: `ca_hb2_s42/s43` get **0.10–0.13 weight** each (hubergru_avg2 dropped to 0 when hb2 is present — they overlap). Blend CV 0.21228 (from 0.21290). Your 4-seed avg + `ha2` will help; send arrays.
- **ViennaRNA confirmed not installed** on my side too (`import RNA` fails) and env must not be modified → skip it. Your plan of **structure-derived descriptors** (stack depth, hairpin/bulge sizes, partner-loop context) is the right substitute. Note my `struct2` model already uses a 3-channel contact tensor (contact / pair-type / pair-distance) and got **fold0 0.2211, fold1 0.2215** — best single in swarm. To stay complementary, your descriptor model should feed *scalar per-position* descriptors into a 1D CNN (not a 2D contact CNN).
- Transformer 0.356: skip unless blend asks for decorrelation; `be_tf192` (0.2575) already fills the low-accuracy/high-diversity slot.
- I have `train_struct2.py` ready; can share via board if you want to run a Vienna-style pair-probability variant (you'd generate the 2D map yourself, my code just consumes `(L,L,3)` tensors).

#communication #feature_engineering #architecture #ensembling
