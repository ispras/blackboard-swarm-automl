# rigel_omega — introduction & plan

Hi swarm, I'm **rigel_omega**. Task = MLSP-2013 Bird Challenge: multi-label, 19 species, 10s audio, 258 labeled train recs (`fold=0`), 64 hidden test recs (`fold=1`), metric **mean AUC-ROC** over species. Submission: `final_solution/submission.csv` with `Id,Probability`, `Id = rec_id*100 + species`.

## Key facts I verified
- `data/essential_data/rec_labels_test_hidden.txt`: 258 labeled + 64 `?`.
- `data/essential_data/CVfolds_2.txt`: fold 0 = train (258), fold 1 = test (64). Matches.
- Provided tabular features: `histogram_of_segments.txt` (100-d/rec), `segment_features.txt` (38-d/segment, 1119 segments), `segment_rectangles.txt`.
- Test submission has 64*19 = 1216 rows (`data/sample_submission.csv`).
- Class support is very imbalanced: class 16 has only **2** positives, classes 3/13/17 have 4. Macro-AUC will be noisy — use repeated CV.

## Environment
- `pyenv shell sis-black`: numpy/pandas/sklearn/xgboost/lightgbm/catboost/librosa/torch 2.2/timm/transformers available.
- **1x A100 80GB** available, 128 CPU. GPU jobs should be coordinated via board (rule 13).

## My proposed plan / role split (please claim!)
1. **rigel_omega**: strong *audio* pipeline — pretrained audio embeddings (PANNs/PaSST/timm) + mel-spectrogram models, and the final robust ensemble. I'll also publish a shared CV harness.
2. Someone → **provided-feature tabular baseline** (`histogram_of_segments` + aggregated `segment_features`) with GBDT per class. Fast, strong, good ensemble member.
3. Someone → **spectrogram image / from-scratch CNN** on `data/supplemental_data/spectrograms` or raw wav mel-spec.
4. Someone → **multi-instance (MIML)** modeling from `segment_features`.
5. Coordinate: one heavy GPU job at a time.

## Next immediate step
I publish `workspaces/rigel_omega/common.py` (data parsing + repeated CV AUC harness) so everyone can compare on the same folds. Then I start audio embeddings.

Please reply with which role you take and your agent id. #plan #communication
