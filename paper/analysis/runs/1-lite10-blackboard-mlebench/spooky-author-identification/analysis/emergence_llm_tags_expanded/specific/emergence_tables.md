### Table 1. LLM specific tag specialization per agent

| agent | messages | tag tokens | unique tags | H (normalized) | SI | JSD to pooled | SI 95% CI | first window SI | last window SI | slope/h | p (trend) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rigel_xi | 30 | 112 | 24 | 0.8695 | 0.1305 | 0.0479 | [0.109, 0.220] | 0.3691 | 0.4415 | 0.0050 | 0.1639 |
| deneb_upsilon | 32 | 117 | 24 | 0.8970 | 0.1030 | 0.0310 | [0.091, 0.179] | 0.3001 | 0.3428 | 0.0045 | 0.1513 |
| arcturus_alpha | 11 | 42 | 16 | 0.7641 | 0.2359 | 0.2426 | [0.222, 0.368] | 0.3264 | 0.4053 | 0.0152 | 0.0530 |

### Table 2. LLM specific tag behavioral divergence

| pair | JSD (bits) | permutation p |
|---|---|---|
| rigel_xi–deneb_upsilon | 0.1180 | 0.0763 |
| rigel_xi–arcturus_alpha | 0.3906 | 0.0007 |
| deneb_upsilon–arcturus_alpha | 0.2723 | 0.0147 |
| mean | 0.2603 | 0.0007 |

### Table 3. Workstream specialization per agent

| agent | messages | SI | JSD to pooled | SI 95% CI | early SI | late SI | slope/window | p (trend) |
|---|---|---|---|---|---|---|---|---|
| rigel_xi | 30 | 0.3256 | 0.0543 | [0.217, 0.555] | 0.2321 | 0.5374 | 0.0088 | 0.2229 |
| deneb_upsilon | 32 | 0.1899 | 0.0134 | [0.128, 0.379] | 0.3212 | 0.4657 | 0.0011 | 0.4582 |
| arcturus_alpha | 11 | 0.1563 | 0.2831 | [0.116, 0.545] | 0.3154 | 0.5117 | 0.0590 | 0.0983 |

### Table 3b. Coding validation

| keyword classifier accuracy | Cohen's kappa |
|---|---|
| 0.8493 | 0.7995 |

### Table 4. Workstream behavioral divergence

| pair | workstream JSD (bits) |
|---|---|
| rigel_xi–deneb_upsilon | 0.0774 |
| rigel_xi–arcturus_alpha | 0.4983 |
| deneb_upsilon–arcturus_alpha | 0.2957 |

### Table 5. Reactivity quotient

| agent | messages | messages citing peers | rho message | rho 95% CI | external link share | external link 95% CI |
|---|---|---|---|---|---|---|
| rigel_xi | 30 | 24 | 0.8000 | [0.627, 0.905] | 0.6912 | [0.574, 0.788] |
| deneb_upsilon | 32 | 23 | 0.7188 | [0.546, 0.844] | 0.7273 | [0.582, 0.837] |
| arcturus_alpha | 11 | 10 | 0.9091 | [0.623, 0.984] | 0.8800 | [0.700, 0.958] |

### Table 6. Influence matrix W (citation links)

| source \ target | rigel_xi | deneb_upsilon | arcturus_alpha | row sum | net flow |
|---|---|---|---|---|---|
| rigel_xi | 21 | 31 | 16 | 68 | 10 |
| deneb_upsilon | 24 | 12 | 8 | 44 | -8 |
| arcturus_alpha | 13 | 9 | 3 | 25 | -2 |

### Table 7. Citation null tests

| statistic | value | null reference | p |
|---|---|---|---|
| observed median citation lag | 24.3500 | 68.3248 | 0.0003 |
| observed mean citation lag | 53.9331 | 156.3405 | 0.0003 |
| observed external citation share | 0.7372 | 0.6260 | 0.0037 |

### Table 8. Per-agent external citation lags

| agent | links | median lag (min) | null median lag (min) | p (shorter) |
|---|---|---|---|---|
| rigel_xi | 68 | 23.6583 | 51.0965 | 0.0003 |
| deneb_upsilon | 44 | 14.6583 | 67.8005 | 0.0003 |
| arcturus_alpha | 25 | 93.6833 | 177.0720 | 0.0110 |

### Table 9. Citation graph statistics

| statistic | value |
|---|---|
| nodes | 73.0000 |
| citation edges | 131.0000 |
| density | 0.0249 |
| reciprocity | 0.0153 |
| author assortativity | -0.1687 |
| assortativity null mean | -0.0135 |
| assortativity permutation p | 0.0053 |
| modularity | 0.6060 |
| communities | 6.0000 |

### Table 10. Top cited messages

| rank | agent | time | title | in_degree | external_in_degree |
|---|---|---|---|---|---|
| 1 | arcturus | 2026-10-09 19:01:46 | arcturusalpha — Status: baseline suite + lane ack | 6 | 6 |
| 2 | deneb | 2026-10-10 00:17:45 | denebupsilon — ⚡ FGM adversarial training is a BIG lever (fold0 0.375 → 0.309) | 6 | 4 |
| 3 | rigel | 2026-10-09 19:22:21 | rigelxi — bert-base-uncased full 5-fold OOF | 6 | 1 |
| 4 | arcturus | 2026-10-09 19:56:31 | arcturusalpha — 10 classical models with OOF and test preds (blocker fixed) | 5 | 5 |
| 5 | deneb | 2026-10-09 19:12:12 | denebupsilon — Status: classical + roberta results, plan | 5 | 4 |
| 6 | deneb | 2026-10-09 20:55:14 | denebupsilon — seed-averaging works big: bert-cased 2-seed OOF <metric>0.33442<\metric> | 5 | 4 |
| 7 | rigel | 2026-10-09 19:55:44 | rigelxi — finalsolution ownership + GPU contention (please read) | 4 | 4 |
| 8 | rigel | 2026-10-09 20:25:29 | rigelxi — GPU lanes + stacking plan (reply to arcturus & deneb) | 4 | 4 |
| 9 | deneb | 2026-10-09 19:29:38 | denebupsilon — roberta-base full 5-fold OOF (+ blend refresh) | 4 | 3 |
| 10 | arcturus | 2026-10-09 20:22:08 | arcturusalpha — LGBM negative result + proposal to add a distinct large transformer | 4 | 3 |
| 11 | rigel | 2026-10-09 21:38:01 | rigelxi — Blend v10: deberta-v3-large lands → OOF <metric>0.20601<\metric> | 4 | 3 |
| 12 | rigel | 2026-10-09 19:22:18 | rigelxi — Blend v1: stacking OOF <metric>0.27436<\metric> | 4 | 2 |
| 13 | rigel | 2026-10-09 20:48:08 | rigelxi — Blend v9b <metric>0.21716<\metric>; deberta-v3-large running (strong!) | 4 | 2 |
| 14 | rigel | 2026-10-09 22:52:05 | rigelxi — Blend v11: OOF <metric>0.20240<\metric> (nested-honest 0.20268) | 3 | 3 |
| 15 | deneb | 2026-10-10 01:42:16 | denebupsilon — FGM lifts my stack2 to 0.19977; combine w/ yours → nested <metric>0.19364<\metric> | 3 | 3 |
