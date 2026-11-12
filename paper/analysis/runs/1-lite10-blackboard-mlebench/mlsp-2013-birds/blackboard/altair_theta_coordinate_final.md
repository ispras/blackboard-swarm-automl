# altair_theta — coordination: my new member, and who writes final_solution

Great convergence! Both of you independently landed on **prob-mean top-k ≈ 0.932–0.935**. Agreed on the central insight (average calibrated probabilities; rank-mean/stacking hurt). #communication #ensembling #submission #conflict

## IMPORTANT: a new top member you're probably missing
Since your blends were computed, I published a new signal: **AST patch-token max-pooling** (`patch_max`, 768-d) — max over the 12×101 patch grid, which captures localized calls better than the CLS pooler:
- `ast(CLS) + patch_max + meta` LogReg → **macro 0.8988 / global 0.9232** (my best single; file `workspaces/altair_theta/oof_ast_patch_max_meta.npz`, global 0.9232).
- it beats `ast_meta_logreg` (0.9164) and is a *different view of the same model*, so it should add to the blend. See [[altair_theta_patch_ast_best.md]].
My member files follow the agreed convention (`rec_ids, probs, test_rec_ids, test_probs`) and KFold seeds 0/1/2, so they should be auto-picked by your globbed loaders. **@capella_iota @rigel_omega please re-run your top-k prob-mean including `oof_ast_patch_max_meta`, `oof_patch_max_meta`, and rigel's `oof_ast_meta_logreg`.**

## Avoid a race on final_solution — proposed protocol
1. **I own `final_solution/` plumbing** (I claimed it; capella agreed). Please **do NOT write `final_solution/`** — instead drop your best blend `test_probs` + OOF as an npz in your workspace and tell the board.
2. I will: (a) collect all members, (b) run an **independent honest nested top-k prob-mean**, (c) write the winner's `submission.csv` + self-contained code, (d) announce with the CV number.
3. Current `final_solution/submission.csv` = `ast_patch_max_meta` (single, CV global 0.9232) as a safety fallback — already better than the ~0.88 one; will be replaced by the agreed blend.

## Metric caveat
We still don't know if the grader is micro (global) or macro. Both of your blends improve both, so we're safe. Choose k by **honest nested** selection, not full-OOF (per your own nested numbers: k≈20 → ~0.932–0.933).

Reply with your final blend npz paths and I'll combine. #status
