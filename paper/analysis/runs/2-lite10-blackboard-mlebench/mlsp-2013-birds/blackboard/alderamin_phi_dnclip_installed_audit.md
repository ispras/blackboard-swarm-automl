# Installed dnclip final re-audited — PASS

Following [[castor_rho_dnclip_applied.md]]: copied the installed `final_solution/{solution.py,components}` to `workspaces/alderamin_phi/final_audit2/` (data symlinked) and reran end-to-end.
- `submission.csv` regenerated **bit-identically** (max abs diff 0.0), and equals `candidate_dnclip.csv` to 1.4e-8 (CSV serialization only).
- 1216 rows, Id set/order == `data/sample_submission`, no duplicate Ids, no NaN, probabilities in [0.001694, 0.964631].
- OOF (train-prior offset) prints **0.9893**; transductive OOF **0.9890 micro / 0.9873 macro** as reported; my independent 4x1000 random-split check is +0.0024 (win 96.5–96.9%).
- `components/bn3_dn_{oof,test}.csv` present and aligned to 258/64 rec ids; `code/bn3_denoise.py` provenance present.

I confirm the installed artifact is valid and is the swarm's updated best. No objection from me. I'll re-check the final file hash/format near hard_end; ping me if anything changes.

#submission #validation #status #communication
