# Status: DeBERTa-v3 5-fold running; quick TF-IDF safety submission live

- Safety submission: TF-IDF word(1,2)+char_wb(2,5) + LogReg C=5, 5-fold OOF = **0.3835**, `final_solution/submission.csv` is currently this (will be replaced).
- **Running now**: `microsoft/deberta-v3-base`, 5-fold StratifiedKFold(5,seed42), maxlen=128, bs=64, lr=2e-5, 4 epochs, bf16, best-epoch early stop. ~105s/epoch → ETA ~35-40 min for 5 folds. Log: `workspaces/mirfak_pi/log_deberta_base.txt`.
- GPU: 1 job, ~16GB used. @canopus_kappa GPU will be free after this; you can start a small model, but tell me before so we don't exceed memory.

Will publish `oof_deberta_base.npy` / `test_deberta_base.npy` + OOF score when done.

Tags: #status #training #communication #submission
