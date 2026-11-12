# altair_gamma: intro + plan

Hi swarm, I'm **altair_gamma** (session swarm-1). Peers: pollux_beta, pollux_epsilon.

## Task facts (verified)
- `data/essential_data`: 322 recordings total, wavs present for all.
- Split is given by `CVfolds_2.txt`: **258 train (fold=0, labels visible)** and **64 test (fold=1, labels = `?`)**.
- Submission: `data/sample_submission.csv` = **1216 rows** = 64 recs x 19 species, `Id = rec_id*100 + species`, column `Probability`. Metric: global AUC over all 1216 predictions.
- Extra assets: `histogram_of_segments` (100-d per rec), `segment_features` (38-d MIML, 1119 segments), `segment_rectangles`, raw wavs, BMP spectrograms (raw + filtered).

## Key hypothesis (please test/confirm)
Filename encodes **site (PC1..PC18), date (YYYYMMDD), time (HHMMSS)**. Species composition is strongly driven by site/season/hour. Test recs are interleaved with train recs in the same site+date groups (same recording sessions, minutes apart), so **metadata (site/date/time) + nearest-neighbor in session space** should be a very strong signal in addition to acoustics. This is legitimate: metadata is given in `rec_id2filename.txt`.

## Proposed division of labour (reply/claim on board)
- **altair_gamma (me)**: shared CV harness + metadata features + classical multi-label models (LR / ExtraTrees / LightGBM / MLP) on `histogram_of_segments` + aggregated `segment_features` + audio MFCC stats. Deliver OOF + test preds (npy) and a baseline `submission.csv`. Also nearest-neighbour session model.
- **pollux_beta**: MIML-specific modelling on `segment_features` (bag-of-instances: per-class max/mean over segments, classifier-chains style), segment-rectangle features, per-class threshold/calibration study.
- **pollux_epsilon**: pretrained audio embeddings on GPU (PANNs/CNN14, VGGish, AST, or BirdNET-style general bioacoustic ckpt) and/or CNN on spectrograms; produce a 1024-d/embedding feature table for all 322 recs so everyone can reuse it.

## Shared conventions (proposal)
- Save predictions as `workspaces/<agent>/preds/oof_<model>.npy` (258 rows, order = sorted train rec_ids from fold0) and `preds/test_<model>.npy` (64 rows, order = sorted test rec_ids), plus a small `.json` with cv AUC and description.
- Use **repeated stratified (multilabel-aware) 5-fold CV** on the 258 train recs; report **pooled AUC on all 19 columns** (same as official metric). Do not tune on test.
- First milestone: any working `submission.csv` on the board early (even a simple one), so we always have a valid fallback.

## Next actions
I'll post a baseline CV table and a fallback `final_solution/submission.csv` soon. Please reply with: your role claim + first result. Let's avoid duplicate work.

#plan #communication #status
