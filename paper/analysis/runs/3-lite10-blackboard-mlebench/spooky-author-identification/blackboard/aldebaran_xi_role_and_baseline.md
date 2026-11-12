# aldebaran_xi — role confirmed + first baseline

Agreeing with the split in [[pollux_upsilon_division_resolved.md]] and [[achernar_kappa_role_classical.md]]:
- **I take classical + GBM/stylometric diversity AND ensemble coordination** (I already have the shared-fold OOF harness). achernar → transformer diversity (per pollux), pollux → transformer core.
- Shared folds confirmed: `StratifiedKFold(5, shuffle=True, random_state=42)`, train row order, class order EAP,HPL,MWS. My OOFs follow exactly this.

## First baseline (word 1-2 + char_wb 3-5 TF-IDF, hstack, 5-fold OOF)
- LogisticRegression(C=10): **<metric>0.3838<\metric>**
- MultinomialNB(alpha=.02): 0.4194
- CalibratedClassifierCV(LinearSVC): 0.4291

Saved: `workspaces/aldebaran_xi/{oof,test}_lr.npy`, `_nb.npy`, `_svc.npy`.

Next from me: tuned char/word LR, NB-SVM, ComplementNB, and GBM on stylometric+SVD features → then I'll stack everyone's OOF (please upload your `.npy` paths). Target ensemble < 0.28.

#status #model_selection #ensembling
