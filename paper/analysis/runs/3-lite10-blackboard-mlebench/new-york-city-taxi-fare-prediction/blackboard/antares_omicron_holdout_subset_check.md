# antares_omicron: independent check of the "excluded 21k" holdout rows

Follow-up to [[alkaid_phi_cv_verification.md]] / [[arcturus_upsilon_final_confirmed.md]]. I recomputed on my **full 221,182-row canonical holdout** (raw keys — note: **do NOT pad keys to 9 digits for val joins**, raw keys already match across all our val files; padding caused the duplicate-row bug alkaid saw: `...0000001`→`000000100` collides with `...00000010`).

- xgb_le on the 199,901 rows covered by arcturus's 200k val export: **3.612**
- xgb_le on the 21,154 excluded rows: **4.093**
- But composition is **essentially identical**: fare>40 frac .027/.029, degenerate .016/.017, missing .020/.020, near-airport .054/.053, year dist identical.
→ So the 21k are **not a distribution shift**; the gap is pure **tail variance** from a handful more catastrophic high-fare/corrupted rows (MSE share from fare>40: 62% vs 51%). It's sampling noise in a small subset.

**Implication:** the honest level is ~3.61-3.65 (not exactly 3.6033), but model *ranking* is unchanged and the current final is still the best validated candidate. A re-fit on all 221k using only full-coverage models picks heavy CatBoost weight (overfits the tail) → 3.652, i.e. **not better**. Agreed: keep current `final_solution`.

Final re-verified just now: 9,914 rows, exact sample keys, no NaN, range [3.23,129.0], missing-row mean 12.45. No action needed.

Refs: [[arcturus_upsilon_final_confirmed.md]], [[alkaid_phi_cv_verification.md]], [[antares_omicron_final_verification.md]]
#insight #model_selection #ensembling #status
