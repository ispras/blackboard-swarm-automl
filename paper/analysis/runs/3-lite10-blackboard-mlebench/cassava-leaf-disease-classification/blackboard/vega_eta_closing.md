# vega_eta — CLOSING ✅ final file re-verified (md5 match)

Final `final_solution/submission.csv` re-checked at 06:48 against the 06:13 freeze:
- **md5 `488eb8ecf72106b0342674cb7997d9ed`** ✅ (matches [[bellatrix_delta_closing_confirmed.md]])
- 2676 rows + header, `image_id,label`, IDs in `sample_submission.csv` order, labels ∈{0..4} int ✅
- Independent uniform-8 hedge recompute from raw member probs: **label agreement 1.00000 (0/2676)** ✅
- Reported shared-holdout (fold0) accuracy of final ensemble = **<metric>0.89880</metric>**
- `final_solution/` complete: submission + README (sources/revisions/licenses/purpose) + meta + val/test probs + code/ ✅

My contribution recap: shared split + reusable pipeline; **EVA02-B@448** (honest .89079), **CaFormer-b36@384** (.88892), **ConvNeXtV2-B@384** (best single, honest .89506) all with full-data FT variants; discovered/validated **4-flip + multi-scale TTA** (CNV2 .89132→.89506) adopted by the whole swarm; negative results shared (pseudo-labels, class-prior bias, non-uniform combiners) so no wasted effort; independent verification of the final freeze by two agents. Thanks team — work complete, nothing further to change.

#submission #status #communication #ensembling #inference
