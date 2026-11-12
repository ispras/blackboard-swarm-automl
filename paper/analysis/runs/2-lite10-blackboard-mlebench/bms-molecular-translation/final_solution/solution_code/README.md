# Solution code — BMS molecular translation (final: phase-3 seq2seq + f8 medoid consensus)

## Final policy that produced `../submission.csv` (holdout mean Lev 1.2507 on first-3000;

## first-1000 = 1.203; halves 1.117/1.384)
1. **Primary model**: character-level image->SMILES seq2seq (`common.py`; ResNet34 encoder at
   256px pad-to-square grayscale, 6-layer Transformer decoder d=512, KV-cache greedy decode).
   Final checkpoint = phase-3 best (`ckpt_phase3/ckpt_best.pt`, step 38k), produced by a
   cosine LR-decay polish of the main run. Held-out first-3000 Lev 1.773 (first-1000 1.203).
2. **Full-test inference**: `infer.py` (bs256, sorted ids), SMILES -> InChI via RDKit
   (invalid -> empty + valid=False). ~20 min for 484,838 images on one A100.
3. **Routing**: `lowconf_p3_f8.csv` = primary invalid ∪ lowest 8% primary confidence
   (39,607 ids; conf threshold -0.102049).
4. **Medoid consensus on routed ids** (`make_medoid_f8.py`): candidates =
   {primary (p3), old pre-phase-3 iota model, primary rot180, omega rot180-TTA,
   MolScribe zero-shot letterbox}; pick the candidate minimizing the sum of Levenshtein
   distances to the other valid candidates; constant fallback if none. Non-routed ids keep
   the primary prediction.
   - `make_medoid_phase3.py` builds the phase-3 medoid variant.
   - `medoid_eval.py` evaluates variants on the shared first-3000 holdout.
   - `run_ms_raw.py` / `run_ms_subset.py` produce MolScribe zero-shot predictions on routed ids
     (raw chartok SMILES, conversion afterwards).
5. `baseline_constant.txt` — constant InChI fallback (holdout Lev ~72).

## Training
- `precompute_smiles.py`: canonical isomeric SMILES from train InChI labels (round-trip exact
  99.99%, SMILES ~3x shorter than InChI).
- `train.py` + `common.py`: AMP bf16, AdamW, cosine LR, batch 256, 256px, no h-flip.
- Validation/early stopping only on the shared 20k holdout cut from `train_labels.csv`.
  Test labels are never used.

## Other/earlier scripts
`assemble_final.py` (rule-based iota+MolScribe fallback), `score_preds.py` (Lev scorer),
`run_ms_subset.py` (earlier MS runner with inline graph conversion; superseded by `run_ms_raw.py`).
