# alkaid_psi — FINAL CONFIRMATION (hard_end 19:59)

Read `[[2026-10-31_19-29-28_SYSTEM_final_phase.md]]`, `[[capella_pi_final_confirmation.md]]`. **Experiments stopped; no further changes.**

## Shipped (verified at 19:52 MSK, md5 `9eaf3f19d6812b50dd222fdf63ba2140`)
- `final_solution/submission.csv` — 463,243 rows, exact `sample_submission.csv` id order, int {0,1}, no NaN/dupes, **7,947 positives (1.716%)**.
- `final_solution/final_config.json` + `final_solution/sources/*` (7 frozen prediction parquets) reproduce it **bit-exactly** via `final_solution/code/make_submission.py` (independent reproducers: canopus diff 0, capella diff 0).
- `README.md` (pipeline, GroupKFold CV, OOF/nested metrics, external-model log: `timm/resnet18.a1_in1k`, Apache-2.0) and `code/` (alkaid video/ground/ensemble code + `code/canopus_pi`, `code/capella_pi`).

## Final validated metrics
- **OOF MCC <metric>0.7860<\metric>** (pp 0.7804, ground 0.7974); **nested-threshold 0.7850**; 24-play hold-out global ~0.7815-0.7871.
- Fixed OOF thresholds (rate-matching rejected: 0.7699 vs 0.7815 in 24-play simulation).
- Late additions (ground GRU v6, capella mixv4, CNN-embedding GBDT) tested and rejected as noise — see `[[alkaid_psi_gru_v6_check.md]]`.

No test labels or private/grading data were used at any point. Ready for official grading. #submission #status #communication
