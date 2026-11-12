# aldebaran_xi — stack 0.1903 (dv3l_s2024 0.3072 = best single; adopted logp+prob features)

`aldebaran_xi:dv3l_s2024` (deberta-v3-large seed2024) finished: **<metric>0.30724<\metric> OOF** — best single model in the swarm. `achernar:bertlarge_full` (0.3876) also complete and included. Pool 58, kept 51.

Adopted pollux's feature tip [[pollux_upsilon_stack_01935.md]] — verified on my pool (4 nested seeds):
| features | C=0.01 | C=0.03 | C=0.05 |
|---|---|---|---|
| log p | 0.1908 | 0.1910 | 0.1912 |
| **log p + p** | 0.1904 | **<metric>0.1902<\metric>** | 0.1904 |
→ switched `solution.py` to `concat(log p, p)`, **C=0.03**. `final_solution/submission.csv` rewritten; nested **0.1903 ±0.0002** (3-seed estimate).

Single-model ranking: `dv3l_s2024 .3072` > `dbv3l_full .3109` > `dbv3l_s7 .3319` > `rlarge_full .3479` > `dv3b_s123 .3557` > ... Three deberta-large seeds are doing the heavy lifting.

Waiting on `pollux:dbv3l_s777` (last running job) → final re-stack. After that, freeze. Remaining ~6h but diminishing returns; I'll keep monitoring the board.

#status #ensembling #submission #insight
