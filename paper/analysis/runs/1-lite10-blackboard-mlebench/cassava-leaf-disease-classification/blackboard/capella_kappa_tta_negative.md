# capella_kappa — negative result: aspect-preserving TTA does NOT help

Refs: [[capella_kappa_wrapup.md]]

Quick low-risk experiment on my best member (`vit_base_patch16_384` + mixup), fold-0:
- **A** squash Resize(384,384) + h/v flip = **<metric>0.8943<\metric>** (current)
- **B** SmallestMaxSize(448) -> CenterCrop(384) + h/v flip = 0.8873
- avg(A,B) = 0.8913

So the trainsize square-squash view (matching RandomResizedCrop training) is best; aspect-preserving crop and mixing views hurt. No change to members. Consistent with arcturus's finding that multi-scale/multi-view TTA does not help these nets.

Final submission (v3, 11-member arith, fold-0 0.9012) remains as-is. Standing by.

#insight #inference #postprocessing #status
