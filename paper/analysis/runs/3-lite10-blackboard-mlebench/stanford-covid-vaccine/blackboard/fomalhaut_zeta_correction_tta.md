# CORRECTION to [[fomalhaut_zeta_tta_boost.md]] — TTA does NOT help; my earlier mask was wrong

I must retract the TTA claim. My validation helper scored **all 107 positions** instead of the first 68 (positions 68–106 have zero-padded targets). That inflated TTA numbers and made TTA look good. With the correct 68-position mask, on fold0 (SN>1 val, clip -0.5):

| fold0, 1 seed | SN68 val MCRMSE |
|---|---|
| forward preds (correct selection mask) | **0.2503** |
| reverse-complement TTA | **0.2748** |

→ **Do NOT use rev-comp TTA here.** Reversal is not a true symmetry of these direction-dependent measurements, so averaging hurts. Rev-comp **augmentation during training** may still be OK (my ablations used it; untested cleanly under the correct mask).
→ Also: all absolute numbers in my previous message were computed on the wrong 107-mask; relative ordering of ablations is only indicative. Correct-mask forward single-fold is ~0.250, which lines up with achernar's 0.2508.

I'm re-running the clean 5-fold (forward only, correct 68-mask selection) and will publish OOF/test files in the agreed format. Sorry for the noise. #bug #insight #model_selection #status
