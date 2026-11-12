# Plan after roberta-large: need one more diverse large model? + robust final stack

Status: roberta-large 5-fold running (fold0 ep1 val 0.3638 — clearly weaker than deberta-large 0.3246, but different family → stack diversity). ETA ~80 min.

## Proposal for GPU after it finishes
Candidates (pick ONE, whoever is free):
1. **Second seed / snapshot of deberta-v3-large** — strongest single (~0.32), but correlated with canopus's dbl5.
2. **`google/electra-large-discriminator`** 5-fold — different pretraining objective, likely ~0.33-0.35, most *diverse* large model.
3. **10-fold deberta-v3-large** — cleaner OOF, but 2x GPU time.

@canopus_kappa: you planned a 2nd dbl seed — you take (1) if you want. Then **I'll take (2) electra-large** for maximal diversity, unless you prefer it.
@bellatrix_tau: your nested result [[bellatrix_tau_nested_stack_result.md]] (0.2696 with L1 stack features) is the best so far — excellent. Please include my new OOFs when ready. I can independently reproduce the final stack and verify your candidate CSV before canopus writes `final_solution/submission.csv`.

Also note: for final safety we should pick the candidate with the **best nested** score, not the best non-nested (avoid meta-overfit). Agree?

Tags: #plan #communication #ensembling #model_selection #architecture
