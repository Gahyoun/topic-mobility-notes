"""Audit directed cycles in the archived core graph and the embedding graph."""
import argparse
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def cycle_summary(n, source, target):
    graph=csr_matrix((np.ones(len(source),dtype=np.int8),(source,target)),shape=(n,n))
    _,labels=connected_components(graph,directed=True,connection='strong')
    sizes=np.bincount(labels)
    loops=source==target
    members=sizes[labels]>1
    return {'core_edges':len(source),'self_loops':int(loops.sum()),
            'multi_node_cyclic_components':int((sizes>1).sum()),
            'nodes_in_multi_node_cycles':int(members.sum()),
            'core_edges_in_multi_node_cycles':int((members[source] & (labels[source]==labels[target])).sum()),
            'max_component_size':int(sizes.max())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('cohort','portfolio','output'):
        p.add_argument(key,type=Path)
    a=p.parse_args()
    with sqlite3.connect(a.portfolio.resolve().as_uri()+'?mode=ro&immutable=1',uri=True) as db:
        works=pd.read_sql_query('SELECT work,MIN(year) AS year FROM publication_metadata GROUP BY work ORDER BY work',db)
    with sqlite3.connect(a.cohort.resolve().as_uri()+'?mode=ro&immutable=1',uri=True) as db:
        db.execute('CREATE TEMP TABLE nodes(work TEXT PRIMARY KEY,ix INTEGER) WITHOUT ROWID')
        db.executemany('INSERT INTO nodes VALUES (?,?)',zip(works.work,range(len(works))))
        edges=pd.read_sql_query('''SELECT n.ix AS source,r.ix AS target FROM nodes n
                                   JOIN outgoing_reference e ON e.work_id=n.work
                                   JOIN nodes r ON r.work=e.reference_id''',db,dtype=np.int64)
    source,target=edges.source.to_numpy(),edges.target.to_numpy()
    year=works.year.to_numpy()
    diff=year[source]-year[target]
    raw=cycle_summary(len(works),source,target)
    strict=cycle_summary(len(works),source[diff>0],target[diff>0])
    assert strict['max_component_size']==1 and strict['self_loops']==0
    out={'core_nodes':len(works),'raw':raw,'strict_earlier_year':strict,
         'same_year_edges':int((diff==0).sum()),'reverse_year_edges':int((diff<0).sum())}
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    main()
