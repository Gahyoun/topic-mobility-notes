"""Write aligned researcher's annual and window center tensors with observation masks."""
import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from native_dag import DIMENSIONS


def assignment(index, frame, key, paper_count):
    order = pd.Series(np.arange(len(index)), index=pd.MultiIndex.from_frame(index[['uid',key]]))
    join = frame.merge(order.rename('row'),left_on=['uid',key],right_index=True,how='inner',
                       validate='many_to_one')
    count = np.bincount(join.row.to_numpy(dtype=int), minlength=len(index))
    matrix = csr_matrix((1/count[join.row],(join.row,join.ix)),shape=(len(index),paper_count),
                        dtype=np.float32)
    return count,matrix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('database',type=Path)
    p.add_argument('tensor_directory',type=Path)
    a=p.parse_args()
    root=a.tensor_directory
    axis=np.load(root/'d0032/paper_axis.npz',allow_pickle=False)['work']
    with sqlite3.connect(a.database.resolve().as_uri()+'?mode=ro&immutable=1',uri=True) as db:
        faculty=pd.read_sql_query('SELECT * FROM faculty_observation ORDER BY uid',db)
        annual=pd.read_sql_query('SELECT uid,field,year,n_papers FROM annual_activity ORDER BY uid,year',db)
        windows=pd.read_sql_query('SELECT uid,field,year_start,year_end,n_papers FROM portfolio_windows '
                                  'ORDER BY uid,year_start',db)
        frame=pd.read_sql_query('SELECT uid,work,year FROM publication_metadata',db)
    frame['ix']=frame.work.map(pd.Series(np.arange(len(axis)),index=axis))
    frame=frame.dropna(subset=['ix']).astype({'ix':'int32'})
    first=frame.uid.map(faculty.set_index('uid').first_year).astype(int)
    frame['year_start']=first+5*((frame.year-first)//5)
    annual_n, annual_matrix=assignment(annual,frame,'year',len(axis))
    window_n, window_matrix=assignment(windows,frame,'year_start',len(axis))
    for table,n in ((annual,annual_n),(windows,window_n)):
        table['n_positioned']=n
        table['state']=np.select([table.n_papers.eq(0),n==0,n==1],
                                 ['zero','unpositioned','singleton'],default='multiple')
        assert (table.n_papers>=n).all()
    faculty.to_csv(root/'faculty_roster.csv',index=False)
    annual.to_csv(root/'annual_index.csv',index=False)
    windows.to_csv(root/'window_index.csv',index=False)
    for d in DIMENSIONS:
        folder=root/f'd{d:04}'
        assert np.array_equal(np.load(folder/'paper_axis.npz',allow_pickle=False)['work'],axis)
        x=np.load(folder/'paper_vectors.npy',mmap_mode='r',allow_pickle=False)
        for kind,count,matrix in (('annual',annual_n,annual_matrix),('window',window_n,window_matrix)):
            center=matrix@x
            center[count==0]=np.nan
            np.save(folder/f'faculty_{kind}_centers.npy',center,allow_pickle=False)
    print('Aligned faculty tensors:',len(faculty),len(annual),len(windows),flush=True)


if __name__=='__main__':
    main()
