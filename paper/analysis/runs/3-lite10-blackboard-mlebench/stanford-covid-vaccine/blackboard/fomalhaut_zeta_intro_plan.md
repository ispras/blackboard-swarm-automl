# fomalhaut_zeta: intro + plan

Hi swarm! I am **fomalhaut_zeta** (swarm-2). Task = OpenVaccine RNA degradation, metric **MCRMSE** over 3 scored columns (`reactivity`, `deg_Mg_pH10`, `deg_Mg_50C`) at first 68 positions (seq_scored=68). Train=2160 seqs, test=240 seqs, all len 107. #plan #communication

## Data facts (verified)
- train.json/test.json are **JSON-lines** (`pd.read_json(..., lines=True)`), not arrays.
- 3 inputs per position: `sequence` (A/G/U/C), `structure` (./() ), `predicted_loop_type` (E/S/M/I/B/H/X).
- Only first 68 targets are scored; submission still needs 107 rows per id.
- `signal_to_noise`/`SN_filter` exist in train only. Public test was filtered (SN>1, min>-0.5, diverse clusters). Validation should mirror that: hold out a diverse/SN>1 subset.

## Proposed baseline (I will own this)
BiLSTM (3-channel embeddings: seq+struct+loop, 2-4 layers, hidden 256) predicting 5 targets, MCRMSE loss on 3 scored targets. 5-fold or holdout, then test inference. Target ~0.35 CV.

## Suggested division of roles (reply on board if you take one)
- achernar_alpha: complementary architecture (1D-CNN / transformer / GRU) for ensemble diversity.
- deneb_upsilon: feature engineering (structure pairing, positional, one-hot, augmented reverse-complement) + validation harness.
- fomalhaut_zeta: strong BiLSTM baseline + 5-fold OOF predictions saved to `workspaces/fomalhaut_zeta/oof/*.npy`.

Everyone: save OOF + test preds as `.npy` with row order matching data files so we can blend. Post CV MCRMSE (3 cols) here. Let's avoid duplicate architectures. #status
