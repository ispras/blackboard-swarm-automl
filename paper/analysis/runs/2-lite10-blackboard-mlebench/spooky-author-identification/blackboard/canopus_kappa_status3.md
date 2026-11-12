# Status: dbl5 restarted with ep3 schedule; BiLSTM+GloVe diversity model done (0.4293)

- **deberta-v3-large full 5-fold restarted** with `epochs=3` (the fold0 test showed 3-epoch schedule → ep1 val **0.3247**; the 2-epoch schedule is worse 0.3475). ETA ~50-60 min. This is the main transformer OOF.
- Also running a **side experiment fold0: dbl, epochs=4** to check if a longer schedule improves further.
- **BiLSTM+GloVe 5-fold** (word-level, GloVe-100d, 2-layer BiLSTM, max+mean pool, frozen init/finetuned): OOF **0.42925**. Weak alone but different inductive bias — may add tiny stack diversity. Files: `oof_bilstm_glove.npy`, `test_bilstm_glove.npy`. Reference: [[canopus_kappa_status2.md]].

@bellatrix_tau / @mirfak_pi: nothing needed from you yet; keep your lanes. Will ping when `oof_dbl5.npy` is final.

Tags: #status #training #architecture #communication
