# MLSP-2013 Birds — swarm solution (owner: altair_theta, finalizer)

Metric: **ROC-AUC** (MLE-bench `mlsp-2013-birds`). Submission: `submission.csv` (`Id,Probability`, `Id = rec_id*100 + species`), 1216 rows (64 test recs × 19 species).

## TL;DR
The final prediction is a **probability-mean ensemble (softmax/ top-k weighted) over 150+ per-class OOF models** built by the swarm from frozen public audio/bioacoustic embeddings + context features. Honest **nested CV global AUC ≈ 0.977** (5 seeds × 5 folds). See `blend_report.json` for the exact combiner/params/members and `blend_oof_test.npz` for the blend OOF + test predictions.

## Data
- Train = `CVfolds_2.txt` fold 0 (258 recs, 145 with species, 113 empty-label); test = fold 1 (64 recs, hidden). Test and train share the same 13 stations and 8 dates.
- Provided tabular: `histogram_of_segments` (100-d), `segment_features` (38-d/segment), spectrograms (BMP), raw WAVs.

## Key signals (chronological discoveries)
1. **Metadata/context** from `rec_id2filename` (`STATION_YYYYMMDD_HHMMSS_seq`): station + date + time-of-day, fully shared train/test. Alone → macro 0.79 / global 0.85. (altair_theta)
2. **Frozen AST** (`MIT/ast-finetuned-audioset-10-10-0.4593`) CLS embedding → 0.911 global. (rigel_omega)
3. **AST patch-token max-pooling** (`patch_max`, per-frequency-band `band_max`) — altair_theta; a different, complementary view of AST.
4. **BirdNET v2.4** embeddings + 19-species logits (rigel_omega).
5. **BioLingual** bioacoustic CLAP embedding (rigel_omega).
6. **BirdSet-Wav2Vec2** embedding (rigel_omega).
7. **Perch v2** (Google DeepMind bioacoustics) — strongest single signal (~0.97 global). (rigel_omega)

## Final combiner
`finalize.py` globs every `workspaces/*/**/oof*.npz` with `{rec_ids, probs, test_rec_ids, test_probs}`, ranks members by OOF global AUC, and chooses the combiner (top-k or softmax-weighted prob-mean, prob/logit space) by **honest nested CV**. Per-class selection and rank-mean were shown to hurt (both metrics). Equal/soft weights on calibrated probabilities are key.

## Final result (final run, 10 seeds × 5 folds)
- Combiner chosen: **top-k = 60 probability-mean** over 188 members.
- Honest nested CV: **global AUC = 0.9775 ± 0.0092**.
- Full-OOF: global 0.9782 / macro 0.9755.
- Verified independently with CV seeds not used for member training (0.9758–0.9764), i.e. the selection gain is real.
- See `blend_report.json` for the exact member list and `blend_oof_test.npz` for OOF/test predictions.
- Submission validated: 1216 rows, `Id` set+order identical to `data/sample_submission.csv`, all probabilities ∈ [0,1], no NaN.

## Reproduce
```
pyenv shell sis-black
python finalize.py            # writes submission.csv + blend_report.json + blend_oof_test.npz
```
`finalize.py` depends on the member `.npz` files produced across `workspaces/{altair_theta,rigel_omega,capella_iota}/`. Feature-generating code for altair_theta's members: `ast_window_embed.py`, `ast_tokens.py`, `export_window_member.py`, `export_band2.py`, `export_best_single.py`, `export_combine_member.py`, `export_perch_member.py`. Other members' code lives in the respective agents' workspaces.
