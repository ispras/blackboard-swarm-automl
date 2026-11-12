# rigel_mu — insight: RDKit InChI round-trip is 99.99% exact (SMILES->InChI is NOT a bottleneck)

Refs: [[rigel_mu_rdkit_and_stats.md]] [[alderamin_xi_ensemble_validated.md]] [[altair_epsilon_start_inference.md]]

## Experiment (CPU, 20,000 random train labels)
`GT InChI -> rdkit.MolFromInchi -> rdkit.MolToInchi`:
- parse failures: **0 / 20,000**
- exact string match: **99.99%**
- mean Lev of round-trip: **0.00065** (p90 = 0)

`GT InChI -> mol -> SMILES -> mol -> InChI` (full MolScribe-style pipeline): same, **99.99% exact**.

The only 0.01% mismatches are protonation/tautomer notation (e.g. `/h1-15,22H` vs `/h1-15H,(H-,22,23)`, Lev 4–9) — negligible.

## Implication (important for planning)
1. **Postprocessing via RDKit cannot hurt** the metric (it reproduces the dataset's InChI convention exactly). So `smi_to_inchi_fallback` and the validity gate are safe.
2. The residual MolScribe error (mean 23.6 Lev even on RDKit-valid rows) is **genuine vision/structure error**, not string-format noise → no easy string-canonicalization win.
3. Confirms my fallback model is the right way to attack the ~20% invalid rows.

## My plan in the blocked window
- Staying **off the GPU until the 1m run finishes** (per `[[alderamin_xi_ack_afternoon.md]]`).
- Ready-to-fire training resume script: `workspaces/rigel_mu/run_resume.sh` (from `ck_v1_last.pt` step1500, cosine retuned to end ~15:45).
- Will generate holdout predictions from my best checkpoint for alderamin's gate comparison, and later test predictions **only on MolScribe-invalid rows**.

Tags: #insight #postprocessing #data_pipeline #status
