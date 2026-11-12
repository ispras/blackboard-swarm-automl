import sys, csv, numpy as np
npz=sys.argv[1]; out=sys.argv[2]
z=np.load(npz)
trecs=[int(x) for x in z['test_rec_ids']]; P=z['test_probs']
rec2i={r:i for i,r in enumerate(trecs)}
with open(out,'w',newline='') as f:
    f.write('Id,Probability\n')
    for r in sorted(trecs):
        for c in range(19):
            f.write(f'{r*100+c},{P[rec2i[r],c]:.6f}\n')
print('wrote',out)
