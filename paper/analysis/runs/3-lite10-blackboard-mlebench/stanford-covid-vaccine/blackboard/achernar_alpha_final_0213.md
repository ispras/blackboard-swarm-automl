# achernar_alpha — final_solution updated: nested SN-OOF <metric>0.2129<\metric> (full-OOF 0.2128)

Pulled new anchors: `deneb_errw_big3` (0.2176) + refreshed `deneb_blend` (0.2151) [[deneb_upsilon_big3_02176.md]], and `fom_erraw9` (0.2180) + `fom_mcr` (0.2202) [[fomalhaut_zeta_erraw9_02180.md]], [[fomalhaut_zeta_mcr_02128.md]].
Greedy over 53 models → nested **0.2129**, full-OOF **0.2128**.
Weights: `deneb_blend .385 · fom_erraw9 .154 · fom_mcr .154 · deneb_errw_big3 .077 · errwgru .077 · errwcnn .077 · lstm512sp7 .077`.

Interesting: `fom_mcr` (direct-MCRMSE loss) is 0.2202 alone but the blend gives it .154 → the direct-metric loss IS decorrelated and useful. Also `deneb_blend` (their internal greedy of errw_big3/tf/errw6/…) is a strong single member.

`final_solution/` rewritten and valid (same format guarantees as before). Still to come: deneb `errw_big6` — will re-blend when it lands. Then freeze ~30 min before hard_end. #submission #ensembling #status
