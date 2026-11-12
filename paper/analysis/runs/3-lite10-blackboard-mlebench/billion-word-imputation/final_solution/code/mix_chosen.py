import pickle,argparse
import numpy as np
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mix',required=True); ap.add_argument('--out_gaps',required=True); ap.add_argument('--out_conf',required=True)
    args=ap.parse_args()
    D=pickle.load(open(args.mix,'rb'))
    gaps=[]; conf=[]
    for i in range(len(D['tokens'])):
        gp=D['gap_probs'][i]
        if gp is None or len(gp)==0:
            gaps.append(0); conf.append(0.0); continue
        g=int(np.argmax(gp))+1; gaps.append(g); conf.append(float(gp[g-1]))
    np.save(args.out_conf,np.array(conf,dtype=np.float32))
    pickle.dump(gaps,open(args.out_gaps,'wb'))
    conf=np.array(conf)
    print('n',len(gaps),'apply@0.45',float((conf>=0.45).mean()),'@0.50',float((conf>=0.50).mean()),'@0.55',float((conf>=0.55).mean()),'meanconf',float(conf.mean()))
if __name__=='__main__': main()
