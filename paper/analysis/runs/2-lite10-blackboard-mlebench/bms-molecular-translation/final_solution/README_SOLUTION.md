# BMS Molecular Translation — swarm final solution

`submission.csv`: 484,838 rows (`image_id,InChI`), order identical to `data/sample_submission.csv`.
Accepted holdout (shared 20k split, first-3000 ids): mean Levenshtein **1.2507**
(first-1000 1.203; halves 1.117/1.384). Baselines: best constant ~72, iota-only 1.77-1.91.

## Final pipeline
1. **Primary (regulus_iota)**: from-scratch image->SMILES seq2seq — ResNet34 encoder (256px,
   pad-to-square grayscale) + 6-layer Transformer decoder (d=512, char-level SMILES, KV-cache
   greedy). Trained on all 1.94M `train_labels.csv` labels converted to canonical isomeric
   SMILES; final phase-3 checkpoint after two cosine LR-decay polish rounds. SMILES->InChI
   with RDKit. Holdout first-3000 1.773.
2. **Full test inference**: `solution_code/infer.py` (sorted ids, resumable raw shards).
3. **Tail medoid ensemble**: routed ids = primary invalid ∪ lowest 8% primary confidence
   (39,607). Candidates: {primary, previous iota checkpoint, primary rot180, omega rot180-TTA,
   MolScribe zero-shot with aspect-preserving letterbox}. Choose the candidate with minimum
   sum of Levenshtein distance to the other valid candidates; constant-InChI fallback.
   Non-routed ids stay primary. Assembly: `solution_code/make_medoid_f8.py`.
4. All training/inference/cascade/assembly code is in `solution_code/` (see its README).

Validation used only the 20k holdout cut from `train_labels.csv`
(`workspaces/regulus_iota/val_20k.csv`); no test labels or external feedback were used.
