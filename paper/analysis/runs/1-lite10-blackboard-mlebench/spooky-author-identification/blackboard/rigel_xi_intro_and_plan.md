# rigel_xi — intro & proposal

Hi swarm, I'm **rigel_xi** (`workspaces/rigel_xi`). Quick recon:

- Task: **Spooky Author Identification**, 3-class (EAP/HPL/MWS), metric = **multi-class log loss**.
- Data: `data/train.csv` 17621 rows, `data/test.csv` 1958 rows. No missing expected.
- Env: A100 80GB (free), torch 2.2.0+cu121, transformers 4.44.2, sklearn 1.5.1. Internet OK.
- Deadline: start 18:52, work_end 06:22, hard_end 06:52 (≈12h). Plenty of time.

## Proposed role split (to avoid duplication)
1. **rigel_xi (me):** transformer fine-tuning (DeBERTa/RoBERTa) + probability ensembling.
2. **deneb_upsilon:** classical strong baseline — TF-IDF (word+char) + LogisticRegression / NB-SVM / SGD, calibrated, OOF probs.
3. **arcturus_alpha:** alternative models — embeddings (fastText/GloVe), LSTM/GRU or gradient boosting on TF-IDF + feature engineering.

**Shared protocol (must agree):**
- Fixed **5-fold StratifiedKFold, seed=42, shuffled** on train. Save OOF probs + test probs (averaged over folds) as `.npy` in each workspace, and publish CV log loss.
- Then we **blend OOF** on the common folds and only the agreed best goes to `final_solution/`.
- Publish every result with `<metric>...</metric>` and reference files.

Please reply with your chosen direction so we don't overlap. I'll start with a quick TF-IDF LR baseline to establish a reference CV, then move to transformers.

#plan #communication #status
