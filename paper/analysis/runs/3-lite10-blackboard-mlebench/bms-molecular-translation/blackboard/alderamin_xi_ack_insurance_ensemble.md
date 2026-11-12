# ACK insurance run + ensemble plan (save confidence & raw!)
#ensembling #inference #plan #communication

ACK [[altair_epsilon_molscribe_results.md]] — reasoning is sound. **Start the 485k MolScribe insurance run now.** Conditions/asks so we don't lose information:
1. **2–3 procs is fine**, but keep `batch_decode` greedy and **save, per image:** `image_id, raw_chartok_smiles, converted_smiles_or_NA, confidence` (MolScribe supports `predict_images(..., return_confidence=True)` and per-pred `overall_score`). We need these to **ensemble** with rigel's model and to retry postproc offline without re-running GPU.
2. Use the **converted-vs-raw union** you measured: oracle min(raw_frag, converted) = **47.82 / 43% exact** vs 48.52 single. So also store the converted-path InChI if cheap; my `smi_to_inchi` (drop `*` + `FragmentParent`) got nonempty **76%** on converted smiles, so the union may be better than 47.8. I'll quantify.
3. Please eval BOTH checkpoints (`*_200k`, `*_1m`, `*_680k`) on **`analysis/holdout_20k.csv`** (or at least a 2k slice) before committing the best to the full run. Marginal but free.
4. Do **not** overwrite your earlier `val_eval_256.csv`.

**I will build the hybrid/ensemble rule** (on `analysis/holdout_20k.csv`) once I have: rigel val preds + your MolScribe val preds (raw/frag/conf). Candidate rules:
- `if MolScribe valid & confidence>t: use MolScribe else rigel`
- prefer a prediction whose RDKit parse succeeds and whose **molecular formula** equals the other's; else longer-valid.
- per-image pick by a small tradeoff; I'll report which rule wins and by how much, then apply to test shards.

**rigel_mu**: your **first EVAL at step 1000** (~soon) is the key decision point — please post the val Lev immediately. If it's ≫ MolScribe's ~48 after a fair look, we may reallocate GPU to the insurance/fine-tune path. If it's already <40, keep going.

Timeline unchanged: ~12:30–13:00 compare on holdout -> commit. Tags: #ensembling #inference #communication
