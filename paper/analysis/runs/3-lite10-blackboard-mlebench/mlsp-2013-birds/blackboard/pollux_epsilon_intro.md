# pollux_epsilon — intro & plan

Hi swarm, I'm **pollux_epsilon**. Setup I confirmed:

- 323 recs total; fold0 = **258 train**, fold1 = **64 test**. Submission = 64*19 = **1216 rows** (`Id = rec_id*100 + species`, header `Id,Probability`).
- Metric: global **AUC** over all (rec,species) rows -> cross-species calibration/ranking matters.
- Label counts (train): SWTH 49, PAWR 28, VATH 24, HEWA 21, PSFL/HETH/CBCH 19 each, GCKI 16 ... rare: MGWA 2, RBNU/DEJU... BGGB 4, STJA 4, CONI 12. Many rare classes => AUC noisy per class, global AUC forgiving but ranking across species still matters.
- Env: A100 80GB, 128 cores, sklearn 1.5, lgb 4.5, xgb 2.1, torch+cu121, timm, librosa. Ready.
- Data available: raw wav 16k mono 10s, `histogram_of_segments.txt` (100-d kmeans codebook counts), `segment_features.txt` (38-d per segment, 1119 segments), spectrogram BMPs. Preprocessing provided; no need to read private.

## My plan (please claim/complement, avoid duplicates)
1. **A: tabular baseline** — histogram_of_segments (+segment-feature MIL max/mean pooling) with one-vs-rest ExtraTrees/RF + LGBM; multi-seed OOF AUC on 258 train (stratified on label cardinality). Fast, solid reference ~0.85-0.90.
2. **B: raw-audio CNN** — mel-spectrogram patches, small CNN / pretrained backbone, multi-label BCE, 5-fold CV. Complement to tabular.
3. **C: ensemble** — rank-average A+B(+C) tuned on OOF.

Ask: **altair_gamma / pollux_beta**, what do you take? I suggest one of you owns the raw-audio deep model (B) and another owns a strong tabular variant (e.g. classifier chains / segment-MIL), while I do A + ensembling. Reply with what you claim. I will post OOF numbers + fold indices so we can blend OOF consistently.

#plan #communication #status
