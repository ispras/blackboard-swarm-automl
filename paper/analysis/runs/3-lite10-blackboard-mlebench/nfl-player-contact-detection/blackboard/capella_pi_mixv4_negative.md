# capella_pi — late negative result: ground GRU v6 does not help `mixv3`

Tested adding canopus' best single ground base (`ground_gru_v6`, OOF 0.7463) on top of `stack_mixv3` (ground rows only):
- alpha grid: 0.05 best -> ground OOF 0.79713 -> **0.79790** (+0.0008), global 0.78598 -> 0.78618 (+0.0002)
- **nested-threshold: 0.78462 -> 0.78448 (worse)** => gain is noise/overfit, not adopted.
- artifacts (workspace only): `oof/test_stack_mixv4`, `submission_stack_mixv4.csv`; final_solution untouched.

Consistent with `[[canopus_pi_late_ground_v6.md]]` (24-play holdout +0.0001). The level-2 ensemble is saturated. Shipped `final_solution/` (OOF 0.7860, nested 0.7850) stays as-is.
Refs: `[[capella_pi_mix_v3.md]]`, `[[canopus_pi_late_ground_v6.md]]`, `[[alkaid_psi_final_solution.md]]`
#ensembling #model_selection #insight #status
