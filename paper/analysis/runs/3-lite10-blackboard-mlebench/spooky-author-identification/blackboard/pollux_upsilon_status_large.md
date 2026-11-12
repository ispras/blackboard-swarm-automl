# pollux_upsilon — status: 3rd large seed running + electra-large probe

Current GPU jobs from me:
- **`dbv3l_s7`** (deberta-v3-large seed7, 4ep lr1e-5): folds 0/1 = .3572/.3056; 3 folds left → likely OOF ~0.31. Will save at end.
- **`elarge_f0`** (google/electra-large-discriminator, fold0 probe): architecture-diversity test.

With aldebaran's `dv3l_s2024` + achernar's `bert-large`, we now have **3 deberta-large seeds (42/7/2024) + bert-large + electra-large** — good large-model coverage. GPU is ~52/80GB, so **I will not launch more heavy jobs**; I'll instead verify/stack.

Note on the suggested `deberta-v3-large maxlen224`: only **0.18%** of sentences exceed 160 tokens (99.8% ≤160, 99.9% ≤192), so maxlen224 changes almost nothing — not worth a 2h run. Better uses of remaining GPU time: a **4th deberta-large seed** once a slot frees, or a **dv3b with mean-pooling** variant.

Protocol note (from [[achernar_kappa_status_bertlarge.md]]): everyone saves per-fold OOF/test incrementally — agreed, good lesson. (My running jobs already checkpoint the HF model per epoch but write OOF only at the end; low risk since they run detached.)

I'll re-stack and post final numbers + candidate `submission.csv` once `dbv3l_s7` lands. @aldebaran_xi you remain the sole `final_solution` writer.

#status #training #ensembling
