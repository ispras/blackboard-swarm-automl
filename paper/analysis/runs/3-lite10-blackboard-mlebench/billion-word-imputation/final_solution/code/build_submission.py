import csv,sys,argparse
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--test',default='/download_job_code/payload/sandbox/workspaces/arcturus_kappa/test_v2.txt')
    ap.add_argument('--decisions',required=True); ap.add_argument('--out',required=True)
    args=ap.parse_args()
    dec={}
    for l in open(args.decisions):
        a=l.rstrip('\n').split('\t')
        if len(a)<3: continue
        dec[int(a[0])]=(int(a[1]),a[2])
    with open(args.test,newline='') as f:
        r=csv.reader(f); next(r); rows=[(int(x[0]),x[1]) for x in r]
    n=0
    with open(args.out,'w',newline='') as f:
        f.write('id,"sentence"\n')
        w=csv.writer(f,quoting=csv.QUOTE_NONNUMERIC,lineterminator='\n')
        for i,s in rows:
            if i in dec:
                g,u=dec[i]; t=s.split()
                if 1<=g<=len(t)-1: s=' '.join(t[:g])+' '+u+' '+' '.join(t[g:]); n+=1
            w.writerow([i,s])
    print('rows',len(rows),'inserted',n,'->',args.out)
if __name__=='__main__': main()
