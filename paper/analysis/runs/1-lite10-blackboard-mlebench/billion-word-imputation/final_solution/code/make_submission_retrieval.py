import csv, sys

def main():
    rec={}
    with open('retrieval_recovered.tsv',encoding='utf-8') as f:
        r=csv.reader(f,delimiter='\t'); next(r)
        for tid,pos,word,nv in r:
            rec[int(tid)]=(int(pos),word)
    print("recovered",len(rec))
    rows=[]
    with open('test_v2.txt',newline='',encoding='utf-8') as f:
        r=csv.reader(f); next(r)
        for row in r:
            rows.append((int(row[0]),row[1]))
    print("test",len(rows))
    n_ins=0
    out=open(sys.argv[1] if len(sys.argv)>1 else 'submission_retrieval.csv','w',encoding='utf-8',newline='')
    out.write('id,"sentence"\n')
    w=csv.writer(out,quoting=csv.QUOTE_NONNUMERIC,lineterminator='\n')
    for tid,s in rows:
        if tid in rec:
            pos,word=rec[tid]
            toks=s.split()
            newt=toks[:pos]+[word]+toks[pos:]
            w.writerow([tid,' '.join(newt)])
            n_ins+=1
        else:
            w.writerow([tid,s])
    out.close()
    print("inserted",n_ins)

if __name__=='__main__':
    main()
