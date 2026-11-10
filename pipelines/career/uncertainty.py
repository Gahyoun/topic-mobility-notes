"""Faculty-cluster bootstrap for aggregate transport components."""
import argparse
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

COMPONENTS=('location','scale','shape','transport')


def summarize(complete,rng):
    sums=complete.groupby('uid')[list(COMPONENTS)].sum().to_numpy()
    n=len(sums)
    draws=np.concatenate([sums[rng.integers(n,size=(100,n))].sum(1) for _ in range(100)])
    shares=draws[:,:3]/draws[:,3:]
    return dict(zip(COMPONENTS,np.quantile(shares,[.025,.975],axis=0).T.tolist()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('db',type=Path)
    p.add_argument('summary',type=Path)
    a=p.parse_args()
    result=json.loads(a.summary.read_text())
    with sqlite3.connect(a.db.resolve().as_uri()+'?mode=ro&immutable=1',uri=True) as db:
        data=pd.read_sql_query('''SELECT uid,field,location,scale,shape,transport
                                  FROM portfolio_transitions WHERE geometry_complete=1''',db)
    rng=np.random.default_rng(221110)
    for field,row in result.items():
        group=data if field=='all' else data[data.field.eq(field)]
        assert len(group)==row['geometry_pairs_complete']
        row['transport_ci95']=summarize(group,rng)
    a.summary.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
