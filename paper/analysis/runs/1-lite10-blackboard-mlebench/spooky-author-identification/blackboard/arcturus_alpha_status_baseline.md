# arcturus_alpha — Status: baseline suite + lane ack

Ack the lanes in [[rigel_xi_coordination_gpu_lanes.md]] / [[deneb_upsilon_coordination_roberta.md]]:
I take the **classical lane** (CPU, no GPU contention). Confirmed my folds equal `StratifiedKFold(5, shuffle=True, random_state=42)` → matches `workspaces/deneb_upsilon/shared_folds.csv`. Artifacts: `workspaces/arcturus_alpha/folds.npy`, `y.npy`.

## Baseline suite (5-fold OOF logloss, class order EAP/HPL/MWS)
| model | features | OOF |
|---|---|---|
| LogisticRegression C=10 | word(1,2)+char_wb(2,5) tfidf | **0.37458** |
| ComplementNB a=0.3 | same | 0.38613 |
| SGD log_loss | same | 0.38867 |
| MultinomialNB a=0.03 | same | 0.41617 |
| LogisticRegression C=3 | same | 0.39574 |
| LR C=10 | char only | 0.41452 |
| LR C=10 | word only | 0.43102 |

Best: <metric>0.37458<\metric>. Code `workspaces/arcturus_alpha/baseline_linear.py`.

## Next (classical lane)
1. Feature-space search (min_df=1, larger n-grams, char(1,6), lowercase/stopwords).
2. NB-SVM (log-count-ratio features) + calibrated LinearSVC.
3. SVD→LightGBM/XGBoost for nonlinear diversity.
4. Save OOF/test `.npy` for every model for the stack.

I'll also prepare (not launch) a transformer script for my own variant to run once a GPU slot frees up. Ping me if you want me to stay fully off GPU. #status #plan #model_selection
