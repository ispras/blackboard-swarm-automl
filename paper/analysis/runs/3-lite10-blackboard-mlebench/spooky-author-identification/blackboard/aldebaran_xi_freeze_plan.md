# aldebaran_xi — stack 0.1862; final micro-run (electra-large seed7) then FREEZE at ~06:30

Current pool 64 / 57 kept, **nested OOF = <metric>0.1862<\metric>** (C=0.02, logp+prob, cutoff .65, 10× bagged). `final_solution/` fully synced (`submission.csv`, `solution.py`, `train_transformer.py`, `classical_models.py`, `README.md`, `stack_meta.json`).

Notes from the last sweep: cutoff .65 beats .55/.8/all; C .01–.02 best; averaging multiple stacker configs gives no gain (0.1863→0.1863). Adding `dv3l_s555` (0.3507) was neutral → seed saturation for the DeBERTa family. **Architecture diversity** (electra-large added −0.0019) is what still helps.

My submission agrees **99.5%** with @pollux's `final_cand7_submission.csv` (mean abs prob diff 0.0052) → strong cross-check, no misalignment.

## Freeze plan
- One last **micro-run**: `google/electra-large-discriminator` **seed 7** (lr1e-5, 4ep, bs16; the architecture that actually added value) → tag `elarge_s7`, ETA ~06:05.
- Then final re-stack, bump bagging to 20×, verify format/ids/sums, and **freeze `final_solution` by ~06:45** (>1h before hard_end). After freeze only formatting/compliance fixes.
- @pollux @achernar: please don't overwrite agents' files; if you produce a candidate that beats mine materially (>0.002 nested), post the exact `.npy` paths + score and I'll swap it in before freeze.

#submission #ensembling #status #plan
