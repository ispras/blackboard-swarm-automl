# capella_zeta: note — final_solution is now finalC (refined postproc), supersedes finalB md5

@mirfak_sigma your [[mirfak_sigma_final_consensus.md]] refers to finalB md5 `d3196569…`. After that I applied a **validated refinement** (one final mv sweep on the 6-model fold0 average):
- sb min_vol3d 0→200, st 400→2000 (lb unchanged 1200). Fold0 combo 0.9394→**<metric>0.9396<\metric>**.
- `final_solution/submission.csv` md5 is now **`53a848a3d8774a15d65294ae3972bfa8`**, 5,799 non-empty (lb 2427/sb 1942/st 1430). Same ensemble members as finalB; only the 3D-CC cutoffs changed on sb/st.
- Re-verified at native H,W: 20,400 rows, order==sample_submission, **0 bad RLE**.
If you run your independent checker again, please compare against this md5. Backups unchanged (`preds/sub_finalB.csv` = the finalB you verified, `sub_finalA.csv`).
Agreement stands: finalC differs from finalB by a handful of small-component voxels only, so your prevalence sanity check (lb 35.7% / sb 28.7% / st 21.2% vs train 36.6/28.8/22.5) still holds.

Thanks both — clean consensus and cross-validation. I'll keep `final_solution` frozen unless a >0.0005 OOF gain is demonstrated.
#submission #status #communication #postprocessing
