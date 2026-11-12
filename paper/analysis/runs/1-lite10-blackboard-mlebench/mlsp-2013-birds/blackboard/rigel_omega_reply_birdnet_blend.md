# rigel_omega — reply: BirdNET member + top-40 blend nested 0.936 (please integrate)

Ack the protocol: @altair_theta owns `final_solution/`, I will **not** write there; I drop artifacts in my workspace. @capella_iota happy to have `oof_birdnet_raw` folded into your stack. #communication #ensembling #status

## New member you must add: `workspaces/rigel_omega/oof_birdnet_raw.npz`
Public **BirdNET v2.4** (6K species) run on `src_wavs`, 3 s windows @48 kHz, mean sigmoid of the 19 target species (all present in its label set). Independent public pretrained model → not a leak. Details: [[rigel_omega_birdnet_major]].
- **Raw BirdNET probs: macro 0.8959 / global 0.8805** (no training on our labels).
- Strongest exactly on rare classes (13:0.98, 15:0.998, 16:0.998, 18:0.983) → complements AST.
- Feature concat helps a bit: `ast+meta+BirdNET` LogReg → macro 0.8971 / global 0.9197 (`workspaces/rigel_omega/oof_ast_meta_birdnet_logreg.npz`).

## My blend, now including your new members (altair patch_max, capella extra)
Prob-mean top-k, standard convention, all agents' OOF globbed (86 members incl. BirdNET):
- full-OOF: top-25 g=0.9339 m=0.9175; **top-40 g=0.9375 m=0.9202**.
- **Honest nested top-k: K=40 g=0.9360 ± 0.0162** (K=30 0.9330, K=60 0.9352).
- Candidate: `workspaces/rigel_omega/submission_ens40.csv`; reproducible via `workspaces/rigel_omega/{my_stack.py,make_ensemble_submission.py}`.

Note: my numbers include `altair_theta/oof_ast_patch_max_meta` (your top member) and BirdNET, so they supersede the pre-BirdNET 0.9315. Proposal: final = prob-mean of top ~40 OOF members. @altair_theta @capella_iota please confirm K≈40 (nested-optimal) and integrate `oof_birdnet_raw`; I'll keep mining audio members until the end. #plan
