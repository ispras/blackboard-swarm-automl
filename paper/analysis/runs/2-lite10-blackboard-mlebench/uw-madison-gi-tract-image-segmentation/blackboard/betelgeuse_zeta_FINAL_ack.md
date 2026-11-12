# betelgeuse_zeta — ACK FINAL (3-model ensemble 0.9529), independent verify PASS

Read [[dubhe_zeta_FINAL.md]], [[mimosa_sigma_final_solution_updated.md]], [[betelgeuse_zeta_final_verify.md]].

Agreed and confirmed: `final_solution/submission.csv` = 3-model equal ensemble (mimosa Unet/b4/320/5ch + dubhe Unet/b3/384/3ch + my Unet++/b3/384/3ch), honest fold-0 holdout **<metric>0.9529</metric>**, md5 **`d4f90051987dd6101aff1a397e175bfa`**.
- My independent validator: 20400 rows, id/class order == `data/test.csv`, 5798 non-empty, 0 structural RLE errors, all bounds within `H*W`, strict start ordering, no pair overlaps. PASS.
- `final_solution/` has submission + `recipe_ens_eq.json` + all three members' code + README with ImageNet-backbone attribution. Good.

On the 4th/5th members: @mimosa_sigma's `mix4` result matches my expectation — adding my b4/384 or dubhe's FPN at any tested weight did **not** improve the honest holdout, so correctly rejected. My b4 was a diversity attempt; no objection to keeping it out.

I'm idle/standby. If a strictly better validated holdout is found before freeze, @dubhe_zeta ping me and I'll independently re-verify md5/format before the re-copy. Otherwise nothing more from me — the frozen file is the swarm's best.
#submission #status #communication #ensembling
