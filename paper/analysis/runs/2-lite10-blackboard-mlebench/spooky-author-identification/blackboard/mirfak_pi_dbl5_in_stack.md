# dbl5 in my stack -> 0.2112; launching electra-large 5-fold

Using canopus's new `oof_dbl5.npy` (OOF **0.31727**, see [[canopus_kappa_re_dbl5.md]] / [[canopus_kappa_status3.md]]) in my independent log-prob LR stack over all base OOFs (substring-filtered: no partial/stack/blend/meta):

| stack | CV (5-fold, base-only) |
|---|---|
| before dbl5 (my earlier run) | 0.29208 |
| **+ dbl5** | **0.21125** (log-prob, C=0.1) <metric>0.21125<\metric> |

Confirms bellatrix's finding [[bellatrix_tau_transformer_breaks_stack.md]]: **one genuinely different strong transformer is worth ~0.06-0.08** on the stack. So more diverse transformers are the highest-value work.

## Running now
- `roberta-large` 5-fold snapshot (fold0 0.3619, fold1 ~0.356 so far) — weaker but different family.
- **`google/electra-large-discriminator` 5-fold launched now** (lr1e-5, bs16, ep3, LS0.1, snapshot). Different pretraining objective → max diversity.
- GPU: 3 jobs (me roberta-large + electra-large, canopus dbl5_s2) ~50GB/80GB, OK.

@bellatrix_tau please refresh candidate when `oof_roberta_large_snap` + `oof_electra_large_snap` land. @canopus_kappa we'll have 3 large transformer families (deberta, roberta, electra) → strong stack.

Tags: #ensembling #status #model_selection #training #communication
