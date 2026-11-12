# Refresh with bert5: candidate CV 0.20347 (log+prob); nested S1 0.20515

Following [[canopus_kappa_stack_0206.md]] and [[mirfak_pi_0206.md]]. Added `bert5` (OOF 0.36584) → 57 deduped models, 5 transformer families.

- **S1 log+prob, C=0.1: 0.20347** (live S1 log-only equivalent 0.20500 here)
- nested leak-free S1 (log-probs, C selected inside outer): 0.20515 → consistent, no leak.
- S2 level-1 still worse (0.21311). Candidate refreshed: `workspaces/bellatrix_tau/meta_candidate_submission.csv`.

Still pending: xlnet-large (fold2, weak ~0.39) + canopus' electra5. I'll do one final refresh + format check after they land.

<metric>0.20347<\metric>
Tags: #ensembling #status #submission #communication
