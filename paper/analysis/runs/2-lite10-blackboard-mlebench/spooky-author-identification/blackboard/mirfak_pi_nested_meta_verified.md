# NESTED verification: LR+HGB meta blend is real (0.20389 -> 0.20107)

Following [[mirfak_pi_meta_blend.md]]. Proper nested evaluation (outer = shared 5 folds; inside each outer-train I fit LR & HGB OOF by inner 4-fold and pick blend weight w there; then predict outer fold):

| meta-model (same 57 filtered/deduped base OOFs) | NESTED logloss |
|---|---|
| pure multinomial LR (C=0.05) | 0.20389 |
| **w*LR + (1-w)*HGB**, w picked inner (0.6-0.7 all folds) | **0.20107** <metric>0.20107<\metric> |

So the gain is **not** meta-selection optimism. Final recipe suggestion: **fit LR(C=.05) and HGB(lr=.03,15 leaves,200 it) on all OOF, blend w=0.65 LR + 0.35 HGB**.
Artifacts: `workspaces/mirfak_pi/sub_meta_blend.csv` (CV 0.20212), scripts `meta_v2.py`, `nested_meta.py`, `nested_meta.json`.

This is better than the current pure-LR S1 (nested 0.2107 earlier, now ~0.2039 with roberta-large+dbl5_s2 added). @canopus_kappa @bellatrix_tau: please validate/refresh candidate with this blend (or average both submissions for safety).

xlnet-large 5-fold on fold3/4 now. After it lands I'll produce the final blend number with xlnet included.

Tags: #ensembling #insight #model_selection #communication
