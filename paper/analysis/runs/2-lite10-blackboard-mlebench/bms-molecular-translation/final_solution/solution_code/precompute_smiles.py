import os, sys, time
import pandas as pd, numpy as np
from multiprocessing import Pool

def init():
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')

def conv(x):
    from rdkit import Chem
    try:
        m=Chem.MolFromInchi(x)
        return Chem.MolToSmiles(m) if m is not None else None
    except Exception:
        return None

if __name__=='__main__':
    root='/download_job_code/payload/sandbox'
    tr=pd.read_csv(f'{root}/data/train_labels.csv')
    t=time.time()
    with Pool(24, initializer=init) as p:
        smi=p.map(conv, tr.InChI.tolist(), chunksize=1000)
    print('converted', len(smi), 'sec', round(time.time()-t,1), flush=True)
    tr['smiles']=smi
    bad=tr.smiles.isna().sum()
    print('failed', bad, flush=True)
    tr=tr[tr.smiles.notna()].reset_index(drop=True)
    tr[['image_id','smiles','InChI']].to_csv(f'{root}/workspaces/regulus_iota/labels_smiles.csv', index=False)
    print('saved', tr.shape, flush=True)
    print(tr.smiles.str.len().describe(), flush=True)
    print('chars', sorted(set(''.join(tr.smiles.sample(300000,random_state=0).tolist()))), flush=True)
