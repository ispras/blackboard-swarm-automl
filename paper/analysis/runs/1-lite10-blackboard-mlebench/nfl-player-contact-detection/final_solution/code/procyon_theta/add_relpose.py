import os
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); CACHE=os.path.join(HERE,'cache')

def add(split):
    d=pd.read_parquet(os.path.join(CACHE,f'featg_all_{split}.parquet'))
    rp=pd.read_parquet(os.path.join(CACHE,f'relpose_{split}.parquet'))
    relc=[c for c in rp.columns if c not in ('game_play','step','player')]
    d['_p1n']=d.p1.astype(np.int64)
    d['_p2n']=pd.to_numeric(d.p2,errors='coerce')
    r1=rp.rename(columns={'player':'_p1n',**{c:f'{c}_p1' for c in relc}})
    d=d.merge(r1,on=['game_play','step','_p1n'],how='left')
    r2=rp.rename(columns={'player':'_p2n',**{c:f'{c}_p2' for c in relc}})
    d=d.merge(r2,on=['game_play','step','_p2n'],how='left')
    d=d.drop(columns=['_p1n','_p2n'])
    d.to_parquet(os.path.join(CACHE,f'featg_rel_all_{split}.parquet'))
    print(split,d.shape)

if __name__=='__main__':
    add('train'); add('test')
