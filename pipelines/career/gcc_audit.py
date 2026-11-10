"""Measure the largest weakly connected component under three citation-graph definitions."""
import argparse
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


@njit
def root(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


@njit
def component_labels(n, sources, targets):
    parent = np.arange(n, dtype=np.int32)
    size = np.ones(n, dtype=np.int32)
    for s,t in zip(sources,targets):
        a,b = root(parent,s),root(parent,t)
        if a==b:
            continue
        if size[a]<size[b]:
            a,b=b,a
        parent[b]=a
        size[a]+=size[b]
    return np.array([root(parent,i) for i in range(n)],dtype=np.int32)


def coverage(labels, n_core, links, positioned, roster, base):
    sizes=np.bincount(labels)
    core_sizes=np.bincount(labels[:n_core],minlength=len(sizes))
    biggest=sizes.argmax()
    selected=labels[:n_core]==biggest
    present=links[selected[links.ix.to_numpy()]&positioned[links.ix.to_numpy()]]
    metrics={}
    for field in ['all',*sorted(base.field.unique())]:
        b=base if field=='all' else base[base.field==field]
        p=present if field=='all' else present[present.field==field]
        counts=p.groupby('uid').size()
        assert counts.index.isin(roster.uid).all()
        metrics[field]={'base_faculty':len(b),'at_least_one_positioned':int(b.uid.isin(counts.index).sum()),
                        'at_least_two_positioned':int(b.uid.isin(counts[counts>=2].index).sum())}
    return {'all_nodes':len(labels),'all_core_nodes':n_core,'components':int(np.count_nonzero(sizes)),
            'components_with_core_nodes':int(np.count_nonzero(core_sizes)),
            'isolated_core_nodes':int((sizes[labels[:n_core]]==1).sum()),
            'gcc_all_nodes':int(sizes[biggest]),'gcc_core_nodes':int(selected.sum()),
            'gcc_positioned_core':int((selected&positioned).sum()),
            'largest_core_memberships':sorted(core_sizes[core_sizes>0].tolist(),reverse=True)[:10],
            'faculty':metrics},selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('cohort','portfolio','output'):
        p.add_argument(arg,type=Path)
    p.add_argument('--membership',type=Path,help='Private path for paper component membership')
    a=p.parse_args()
    ro=lambda file: file.resolve().as_uri()+'?mode=ro&immutable=1'
    with sqlite3.connect(ro(a.portfolio),uri=True) as db:
        works=pd.read_sql_query('SELECT work, MIN(year) AS year FROM publication_metadata GROUP BY work ORDER BY work',db)
        links=pd.read_sql_query('SELECT uid,work,field FROM publication_metadata',db)
        roster=pd.read_sql_query('SELECT uid,field,n_supported_refs FROM faculty_observation',db)
        coordinates=pd.read_sql_query('SELECT work FROM dag_native_coordinates',db)
    n=len(works)
    index=pd.Series(np.arange(n),index=works.work)
    links['ix']=links.work.map(index)
    pos=works.work.isin(coordinates.work).to_numpy()
    base=roster[roster.n_supported_refs>0]
    with sqlite3.connect(ro(a.cohort),uri=True) as db:
        db.execute('CREATE TEMP TABLE nodes(work TEXT PRIMARY KEY, ix INTEGER) WITHOUT ROWID')
        db.executemany('INSERT INTO nodes VALUES (?,?)',zip(works.work,range(n)))
        edges=pd.read_sql_query('''SELECT p.ix AS source, CAST(SUBSTR(e.reference_id,2) AS INTEGER) AS ref_number,
                                          COALESCE(q.ix,-1) AS target
                   FROM nodes p JOIN outgoing_reference e ON e.work_id=p.work
                   LEFT JOIN nodes q ON e.reference_id=q.work''',db,dtype=np.int64)
    src,tgt,ref=(edges[k].to_numpy() for k in ('source','target','ref_number'))
    core=tgt>=0
    years=works.year.to_numpy()
    strict=core & (ref>0) & (years[src]>years[tgt])
    external=(~core)&(ref>0)
    external_ids,inverse=np.unique(ref[external],return_inverse=True)
    raw=component_labels(n,src[core],tgt[core])
    dag=component_labels(n,src[strict],tgt[strict])
    aug=component_labels(n+len(external_ids),np.r_[src[strict],src[external]],np.r_[tgt[strict],n+inverse])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    out={'core_links_raw':int(core.sum()),'core_links_strict':int(strict.sum()),
         'external_links':int(external.sum()),'external_sinks':len(external_ids),
         'positioned_core_nodes':int(pos.sum()),'graphs':{}}
    result={}
    for name,labels in (('raw_core',raw),('earlier_year_core',dag),('earlier_year_with_external_sinks',aug)):
        out['graphs'][name],result[name]=coverage(labels,n,links,pos,roster,base)
    _,scc=connected_components(csr_matrix((np.ones(core.sum(),dtype=np.int8),(src[core],tgt[core])),shape=(n,n)),
                               directed=True,connection='strong')
    out['raw_core_largest_strong_component']=int(np.bincount(scc).max())
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    if a.membership:
        pd.DataFrame({'work':works.work,**result}).to_csv(a.membership,index=False)
    print(json.dumps(out,indent=2),flush=True)


if __name__=='__main__':
    main()
