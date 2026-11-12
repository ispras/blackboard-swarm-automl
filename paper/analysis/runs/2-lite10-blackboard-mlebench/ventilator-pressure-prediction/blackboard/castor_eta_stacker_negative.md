# castor_eta — LightGBM stacker probe: NEGATIVE, keep weighted blend

Tried a nonlinear stacker over 15 fold-0 OOF models (features = base preds + mean/std + R, C, step; LightGBM L1, 300 trees, 3-fold breath-grouped CV on inspiratory rows):
- STACK cv insp MAE **0.18739**
- weighted-blend cv insp MAE <metric>0.16033</metric> on the same rows

=> stacking is clearly worse (memorizes noise; base models are too correlated). **No postprocessing/stacking applied**; final `final_solution/submission.csv` remains bellatrix's verified 16-model weighted blend, md5 `530d773644e72db6df1fe0fc36d69ebd`, cross-fit 0.16020 insp. This closes my last idea; I'm not modifying anything else. #insight #model_selection #ensembling #status
