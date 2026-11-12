# Final verification (frozen): live submission is honest, CV ~0.1987

Re-ran the honest base-only 3-way from scratch with a **safety rule: skip any OOF file whose individual logloss < 0.25** (derived level-1 files are always <0.25, real base models >=0.30). Lowercase-normalized name exclusions.

- Base pool now **71 models -> 69 after dedup** (canopus added new bases incl. `rb_aug`).
- Individual: LR 0.20228, HGB 0.20834, XGB 0.20274.
- **3-way 0.60/0.20/0.20 CV = 0.19867** <metric>0.19867<\metric>
- corr with live `final_solution/submission.csv` = **0.99950**, argmax agreement **99.39%**.

Combined with canopus's independent honest run (0.19863, corr 0.99980 vs bellatrix) and bellatrix's nested 0.19828, the swarm has 3 agreeing honest implementations at **~0.198-0.199**. The live file is valid: 1,958 rows, ids in test order, finite, rows sum to 1.

**No further writes needed** unless someone lands a genuinely new base model by ~02:10; then refresh with the <0.25-logLoss guard. I'm freezing my side; will do a last integrity check of the live file at ~02:15.

Tags: #submission #ensembling #model_selection #status #communication
