# arcturus_kappa — 5th word model (longformer) does NOT help; v4 stays final

Re [[antares_upsilon_v4_FINAL.md]] [[spica_psi_v4_ready.md]]

Tested adding `longformer-large-4096` as a 5th word distribution at the **mix3 chosen gaps** (full 30k, same kind2/λ=0.08/gate cw≤0.40):
| ensemble | metric |
|---|---|
| 4-way (shipped v4) | **2.7426** |
| + longformer 0.05 | 2.7433 |
| + longformer 0.10 | 2.7468 |
| + longformer 0.15 | 2.7462 |

No gain (roberta-family ⇒ correlated with roberta-large). Combined with earlier negatives (XLNet, DeBERTa, ELECTRA, xlm-r, window-TTA, weight sweeps), **the word side is exhausted**.

## Final state
`final_solution/` = **v4, holdout 2.7423**, frozen and fully audited by me (rows/ids/quoting/single-token insertion/decisions↔submission 0 mismatches; md5 `77c029000188766c9a1762ba380b1055` at time of audit). I am making **no further changes** and will do the final read-only md5 re-check ~18:50.

#status #model_selection #ensembling #communication
