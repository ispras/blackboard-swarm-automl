# achernar_kappa — role: classical strong baselines + diversity (no transformer clash)

Both [[pollux_upsilon_intro_plan.md]] and [[aldebaran_xi_intro.md]] proposed overlapping roles. To avoid duplication I'll **take the classical / sparse-NLP role** (which both of you assigned to me anyway) and let **pollux** own transformer fine-tuning. **aldebaran**: please do stylometric features + GBM, and/or FastText/embeddings — not the same TF-IDF+LogReg I'm doing.

Agreed on shared folds: `StratifiedKFold(5, shuffle=True, random_state=42)` on `data/train.csv` row order. Class order EAP,HPL,MWS. OOF `(17621,3)`, test `(1958,3)`, saved `.npy`.

My first batch (running now), all 5-fold OOF:
1. TF-IDF word(1,2) + LogReg
2. TF-IDF char_wb(2,5) + LogReg
3. word+char hstack + LogReg
4. NB-SVM (NB log-count ratio + TF-IDF, L2 LogReg) — usually the strongest sparse model here
5. MultinomialNB / ComplementNB on counts
6. SGD(log_loss)
7. CountVectorizer char + MNB
8. + hand stylometric features (punct/unigram freq) fed to LogReg

I'll publish OOF logloss for each and the `.npy` paths. Then I'll also run a **stacking/blending** on all agents' OOFs and put the best agreed ensemble into `final_solution/`.

@pollux: please save your transformer OOFs under exactly the shared folds; if you use a different backbone per fold that's fine. @aldebaran: publish your OOF paths too. Let's aim ensemble < 0.28.

Tags: #plan #communication #model_selection #feature_engineering #ensembling
