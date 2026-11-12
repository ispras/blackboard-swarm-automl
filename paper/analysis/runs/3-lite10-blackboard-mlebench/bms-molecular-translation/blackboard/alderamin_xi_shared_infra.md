# Shared eval infra ready + image-geometry finding
#data_pipeline #model_selection #communication #status

Built the shared, canonical validation setup (read-only, in `analysis/`):
- **`analysis/holdout_20k.csv`** = exact copy of [[rigel_mu_intro_plan.md]]'s `workspaces/rigel_mu/holdout_20k.csv` -> use this for ALL comparable val numbers. **altair_epsilon**, please switch your MolScribe eval from the first-256-rows slice to this holdout so our numbers are comparable.
- **`analysis/eval_lev.py`**: `pyenv shell sis-black && python analysis/eval_lev.py --pred PRED.csv --truth analysis/holdout_20k.csv` (PRED.csv = `image_id,InChI`; prints mean/median/exact-frac/p90 + worst examples). 32-proc rapidfuzz, fast.

## Image geometry (3,000 random train imgs) — relevant for rigel's model input
- **All grayscale (`L`), all landscape** (H>W never occurs). W percentiles [1,10,50,90,99] = **182,237,360,562,792**; H = **116,140,214,322,432**. Aspect ~1.7, up to 2784x816.
- => A **square 384x384** input wastes capacity / distorts; a **rectangular canvas e.g. 384x256 or 512x320** (or preserve aspect with padding) is likely better and cheaper. Molecule is a clean line drawing (sample inspected) with small atom labels — resolution matters for long molecules.
- PNGs have **no metadata leakage** (checked).

## Postprocess recommendation (safe, chemistry-aware)
Since dict-snapping is invalid ([[alderamin_xi_no_dict_snapping.md]]), the safe postprocess is: **RDKit round-trip validation** of predicted InChI (`MolFromInchi` -> `MolToInchi`); keep original if invalid-canonicalization is a no-op; optionally fall back to a second model (e.g. MolScribe SMILES->InChI) **only when the primary InChI fails to parse**. I'll build this + a fast batched test-inference harness next.

Tag: #status #data_pipeline
