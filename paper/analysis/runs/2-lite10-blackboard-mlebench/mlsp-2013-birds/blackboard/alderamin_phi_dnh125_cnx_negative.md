# Negative: bn3_dn_h125 replacement and ConvNeXt add nothing on top of installed dnclip

Tested the two newest artifacts against the installed candidate (my sim, 600 fresh random-64 draws, seed 555; same draws for all variants; base = installed dnclip):
- `bn3_dn_h125` (denser 1.25s hop on denoised audio; standalone micro .9799 / macro .9853) as replacement for `bn3_dn` 1.0: OOF .9889/.9871, random-split **-0.0001** (win 33%).
- ConvNeXt-BirdSet `convnext max` (standalone .9184/.9464) added at 0.1 / 0.2: OOF .9892/.9876, random-split **+0.0001 / +0.0000** (win 52–60%).
- h125 + cnx combined: also ≈0.

Base OOF .9890/.9873, random-split mean .9873 (q10 .9812). Nothing clears the +0.002 bar; installed `final_solution` should stay as is. Current installed artifact already re-audited by me in [[alderamin_phi_dnclip_installed_audit.md]].

#model_selection #ensembling #status #insight
