import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from fe_lib import engineer, FEATURES

df = pd.read_csv('data/test.csv', dtype={'key': str})
print('test rows', len(df))
feat = engineer(df)
out = pd.DataFrame({'key': df['key'].values})
for c in FEATURES:
    out[c] = feat[c].values
tbl = pa.Table.from_pandas(out, preserve_index=False)
pq.write_table(tbl, 'workspaces/deneb_phi/features_test.parquet')
print('saved', out.shape)
print(out.head())
