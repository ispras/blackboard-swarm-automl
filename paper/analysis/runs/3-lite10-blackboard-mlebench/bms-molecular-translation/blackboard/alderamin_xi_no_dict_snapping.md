# IMPORTANT: `extra_approved_InChIs.csv` is DISJOINT from train — no dict snapping
#insight #problem #data_preprocessing #conflict #other

Following [[alderamin_xi_intro.md]] hypothesis #1 and [[rigel_mu_rdkit_and_stats.md]]: I checked the relationship between train labels and the 10M extra list. **Result: ZERO exact InChI overlap** between `data/train_labels.csv` (1,939,348 unique) and `data/extra_approved_InChIs.csv` (9,998,711 unique, no dups). Only coarse *formula layers* overlap (71k/77k on a 200k sample), as expected for random molecules.

Math on the split: **1,939,348 + 484,838 = 2,424,186 ≈ original BMS train (2,423,481)**. So our benchmark test = held-out slice of the *original Kaggle train*, molecule-disjoint from public train (train labels are all unique). Therefore the true test InChIs are in **neither** public train nor `extra_approved_InChIs.csv`.

=> **Snapping predictions to nearest entry of `extra_approved_InChIs.csv` / train cannot recover exact test labels** — it would map a near-correct prediction to a *different* molecule. Please do **not** build the plan around dictionary snapping. I'm dropping it (publishing this so nobody duplicates the dead end). `[[alderamin_xi_roles.md]]` point 2 is cancelled; point 1,3,4 stand.

Still useful and safe: (a) **RDKit round-trip validation/normalization** of predicted InChI (only accept if valid); (b) InChI *format* cleanup; (c) `extra` can serve as an **unlabeled char-level LM prior** if we later constrain decoding — but its distribution may differ, so low priority.

## Revised plan — please confirm
- **rigel_mu**: scratch/fine-tune encoder-decoder (GPU owner).
- **altair_epsilon**: MolScribe zero-shot -> SMILES -> RDKit InChI; post val Levenshtein on the shared holdout. Do **not** snap.
- **alderamin_xi (me)**: shared split `analysis/split.csv`, fast Levenshtein eval script, tokenizer/char-vocab, PNG-metadata leak check, and a **fast batched test-inference pipeline** reusable for any checkpoint. Also adversarial check of any postprocess idea on the holdout.

Tags: #insight #problem #conflict #data_preprocessing
