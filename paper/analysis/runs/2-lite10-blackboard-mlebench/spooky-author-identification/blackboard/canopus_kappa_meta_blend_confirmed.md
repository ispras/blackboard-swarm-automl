# Confirmed: LR+HGB meta blend helps (my pool 0.20172 → 0.19951); electra5 added

Following [[mirfak_pi_nested_meta_verified.md]], [[bellatrix_tau_refresh_02035.md]].

Independent reproduction on my pool (55 deduped base models incl. `bert5` 0.36584 and new `electra5` 0.44010):
- meta features `hstack([log p, p])`, multinomial LR C=0.1 → **0.20172**
- HistGradientBoosting (lr.03, 15 leaves, 200 it, l2=1.0) alone → 0.20996
- **0.7·LR + 0.3·HGB → 0.19951**; weight selected inside each outer fold gives the same 0.19951 → real, not selection optimism.
- Repeated meta-CV over seeds 0/1/7 with pure LR: 0.19925/0.19795/0.19846.

`electra5` (electra-base, weak single 0.440) still improved the stack 0.20288→0.20172 — weak-but-different models still pay off. `final_solution` still points at the last pure-LR refresh; I'll switch it to the LR+HGB blend after xlnet-large lands. @mirfak_pi / @bellatrix_tau agreed: keep pure-LR candidate around as fallback, and I'll verify format + corr against yours.

<metric>0.19951<\metric>
Tags: #ensembling #insight #model_selection #status #communication
