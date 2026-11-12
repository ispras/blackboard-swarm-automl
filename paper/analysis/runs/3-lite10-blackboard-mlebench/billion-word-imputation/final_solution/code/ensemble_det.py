import pickle,argparse
import numpy as np
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--a',required=True); ap.add_argument('--b',required=True)
    ap.add_argument('--wa',type=float,default=1.0); ap.add_argument('--wb',type=float,default=1.0)
    ap.add_argument('--mode',default='log')  # log|lin
    ap.add_argument('--c',default='')
    ap.add_argument('--wc',type=float,default=1.0)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    A=pickle.load(open(args.a,'rb')); B=pickle.load(open(args.b,'rb')); C=pickle.load(open(args.c,'rb')) if args.c else None
    assert len(A['tokens'])==len(B['tokens'])
    gps=[]; dws=[]; trunc=[]
    for i in range(len(A['tokens'])):
        ga=A['gap_probs'][i]; gb=B['gap_probs'][i]
        if ga is None or gb is None or len(ga)==0 or len(ga)!=len(gb):
            gps.append(ga if ga is not None else gb); dws.append(A['detwp'][i] if A['detwp'][i] is not None else B['detwp'][i]); trunc.append(True); continue
        if args.mode=='log':
            s=args.wa*np.log(np.clip(ga,1e-12,1))+args.wb*np.log(np.clip(gb,1e-12,1))
            if C is not None:
                gc=C['gap_probs'][i]; s=s+args.wc*np.log(np.clip(gc,1e-12,1))
            s=s-s.max(); p=np.exp(s); p=p/p.sum()
        else:
            p=args.wa*ga+args.wb*gb
            if C is not None: p=p+args.wc*C['gap_probs'][i]
            p=p/p.sum()
        gps.append(p.astype(np.float32)); dws.append(A['detwp'][i]); trunc.append(A['trunc'][i])
    out={'ids':A.get('ids',[]),'tokens':A['tokens'],'gap_probs':gps,'detwp':dws,'trunc':trunc}
    pickle.dump(out,open(args.out,'wb'),protocol=4); print('saved',args.out)
if __name__=='__main__': main()
