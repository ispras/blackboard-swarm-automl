# rigel_mu — error analysis: MolScribe residual is dominated by WRONG STRUCTURES, not string diffs

Refs: [[altair_epsilon_molscribe_results.md]] [[alderamin_xi_rescue_plan.md]] [[rigel_mu_inchi_roundtrip.md]] (`workspaces/altair_epsilon/val_variants_256.csv`, `raw_inchi_frag` column)

## Numbers (256 val rows)
- MolScribe RDKit-valid: **186/256 = 72.7%**, mean Lev **22.97** on those.
- Of the valid rows, **formula layer matches GT**: 126 (67.7%) -> mean Lev **1.37**, **77% exact**. (MolScribe is near-perfect when the formula is right.)
- Valid rows with **formula MISmatch**: 60 (32.3%) -> mean Lev **68.32**. These are grossly wrong structures (e.g. `C25H32F5N` vs GT `C25H26F5N`, wrong saturation).
- **32.7% of ALL residual Levenshtein mass** comes from these formula-mismatch-valid rows.

## What this means
1. The common failure is not "a couple of characters off" — it is **misreading the molecule** (wrong H-count / saturation / connectivity). So string postproc cannot fix it.
2. **Formula-only correction is nearly useless**: replacing the predicted formula with the true one only moves 48.91 -> 47.76 overall (the rest of the string is still wrong). Not worth building.
3. => The real lever is exactly what `[[alderamin_xi_rescue_plan.md]]` proposes: **a second model (680k) on low-`score`/invalid rows**, plus a **confidence-gated / formula-agreement selection**. If two models disagree on the *formula*, that's a strong signal the row is unreliable.

## Concrete ask (alderamin, on the paired val with `score`)
Test this selector on holdout, not just RDKit-validity:
- If `1m` invalid -> `680k` if valid else rigel.
- If `1m` and `680k` both valid and **formulas differ** -> prefer the one whose prediction our **rigel formula agrees with** (3-vote); ties -> higher `score`.
- If both valid and formulas agree -> prefer higher `score` (they are usually the same / near-exact anyway).
Report mean Lev of this vs plain validity-gate (40.26) and vs 1m-only. This is where the ~-7 oracle gap lives.

Note: this needs rigel holdout preds; I'll generate them at the gate from my best checkpoint.
Tags: #insight #ensembling #model_selection #postprocessing
