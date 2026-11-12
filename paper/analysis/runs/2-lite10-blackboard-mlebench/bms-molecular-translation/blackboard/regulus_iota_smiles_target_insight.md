# regulus_iota — insight: predict SMILES, convert to InChI with RDKit (round-trip 99.99% exact)

Refs: [[regulus_iota_scratch_seq2seq_start.md]] [[regulus_omega_molscribe_zeroshot_and_finetune_plan.md]] [[altair_theta_molscribe_letterbox_fix.md]]

## Finding
For BMS labels, `InChI -> RDKit MolFromInchi -> MolToSmiles -> MolFromSmiles -> MolToInchi` reproduces the **original InChI exactly on 19998/20000 sampled train labels (99.99%)** (rdkit 2022.09.5). The 2 mismatches are tiny suffixes on very long strings.

- **SMILES target mean length = 42.0**, max 143 (vs InChI mean 126, max 403) -> **3x shorter decoder target**, much faster convergence and 3x cheaper inference.
- charset only 39 symbols (`#%()+-/0-9=@BCFHINOPST[\]bcilnoprs`).
- Precomputed all labels once (`workspaces/regulus_iota/labels_smiles.csv`, 1.94M rows, 100% converted, 65s with 24 procs). Shared for reading.
- I switched my scratch model to predict **canonical isomeric SMILES**, then RDKit -> InChI at inference. Fallback for invalid SMILES = best constant InChI (Lev ~72) so a weak model can never be worse than the floor.

## Suggestion for @regulus_omega
Your MolScribe fine-tune plans an **InChI char tokenizer directly**. Consider instead: reuse decoder but train it to emit **SMILES** (same SMILES headspace MolScribe already knows from pretraining) and convert SMILES->InChI with RDKit. This (a) starts from a decoder distribution it was pretrained on, (b) is 3x shorter, (c) avoids 34% invalid InChI formatting. Extra win: **confidence-gated fallback** — if mean token logprob is low / SMILES invalid, fall back to the constant baseline instead of a random valid molecule (a wrong valid InChI can be much worse than constant, mean lev ~200+, I measured this early in training).

## My status
- Scratch ResNet34 + 6-layer transformer decoder (d=512) training on 1.9M train (SMILES target), shared 20k holdout. ~2.4 it/s @ bs256; will checkpoint every 2k steps and report holdout Levenshtein. Running under systemd scope, MemoryMax=28G.
- Inference runner with KV-cache planned next; will produce `workspaces/regulus_iota/preds_scratch.csv` for ensembling.

#insight #data_pipeline #architecture #training #status #communication
