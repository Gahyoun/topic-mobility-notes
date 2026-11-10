"""Describe the citing side of the frozen faculty citation archive."""
import argparse
import json
from pathlib import Path
import sqlite3

import pandas as pd

KNOWN = 'roster_verified'
NONE = 'no_resolved_roster_author'
UNKNOWN = 'unresolved'


def ro(path):
    return sqlite3.connect(path.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--paths',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--author-audit',type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(a.paths.read_text())
    with ro(Path(cfg['web_db'])) as web:
        roster=pd.read_sql_query('SELECT professor_uid,openalex_id FROM professors',web)
        roster['author_id']=roster.openalex_id.str.extract(r'(A[0-9]+)$',expand=False)
    with ro(a.archive) as db:
        papers=pd.read_sql_query('SELECT work_id, professor_uid, subject FROM faculty_publication',db)
        e=pd.read_sql_query('SELECT citing_id,cited_id,citing_year,new_external FROM openalex_incoming',db)
        targets=pd.read_sql_query('SELECT work_id AS cited_id, publication_year AS cited_year FROM graph_work',db)
        cited_subject=pd.read_sql_query('SELECT DISTINCT work_id AS cited_id, subject FROM matched_panel',db)
    e=e.merge(targets,on='cited_id',validate='many_to_one')
    assert e.citing_year.le(2025).all() and (e.citing_year-e.cited_year).between(1,3).all()
    assert not e.duplicated(['citing_id','cited_id']).any()
    source=e[['citing_id']].drop_duplicates().copy()
    source['status']=UNKNOWN
    source.loc[source.citing_id.isin(papers.work_id), 'status']=KNOWN
    if a.author_audit:
        meta=json.loads(a.author_audit.with_suffix('.json').read_text())
        assert meta['roster_query_complete']
        audit=pd.read_csv(a.author_audit)
        assert audit.citing_id.is_unique
        assert audit.citing_id.isin(source.citing_id).all()
        assert audit.status.isin([KNOWN,UNKNOWN]).all()
        source=source.merge(audit,on='citing_id',how='left',validate='one_to_one',suffixes=('','_api'))
        source.loc[source.status.ne(KNOWN),'status']=source.loc[source.status.ne(KNOWN),'status_api'].fillna(NONE)
    e=e.merge(source[['citing_id','status']],on='citing_id',validate='many_to_one')
    summary=e.groupby('status').agg(edges=('citing_id','size'),citing_works=('citing_id','nunique'),
        cited_works=('cited_id','nunique')).reset_index()
    summary['edge_share']=summary.edges/len(e)
    summary['citing_work_share']=summary.citing_works/len(source)
    summary.to_csv(a.out/'by_author_group.csv',index=False)
    assert summary.edges.sum()==len(e) and summary.citing_works.sum()==len(source)
    fields=e.merge(cited_subject,on='cited_id',how='inner',validate='many_to_many')
    by_field=fields.groupby(['subject','status']).agg(edges=('citing_id','size'),
        citing_works=('citing_id','nunique'),cited_works=('cited_id','nunique')).reset_index()
    by_field['edge_share']=by_field.edges/by_field.groupby('subject').edges.transform('sum')
    by_field.to_csv(a.out/'by_field.csv',index=False)
    by_external=e.groupby(['new_external','status']).agg(edges=('citing_id','size'),
        citing_works=('citing_id','nunique')).reset_index()
    by_external.to_csv(a.out/'by_cache_coverage.csv',index=False)
    e['lag']=e.citing_year-e.cited_year
    e.groupby(['lag','status']).size().rename('edges').reset_index().to_csv(a.out/'by_lag.csv',index=False)
    per_source=e.groupby(['status','citing_id']).size().rename('links')
    per_source.groupby('status').agg(['mean','median','max']).to_csv(a.out/'citing_work_degree.csv')
    # For verified positives only, distinguish a citation sharing a roster
    # faculty author from another roster faculty member's citation.
    inside=e.loc[e.status.eq(KNOWN),['citing_id','cited_id']]
    ep=inside.merge(papers[['work_id','professor_uid']],left_on='citing_id',right_on='work_id').drop(columns='work_id')
    self_edges=ep.merge(papers[['work_id','professor_uid']],left_on=['cited_id','professor_uid'],
                        right_on=['work_id','professor_uid']).drop_duplicates(['citing_id','cited_id'])
    report={'roster_professors':len(roster),'roster_professors_with_id':int(roster.author_id.notna().sum()),
            'distinct_roster_author_ids':int(roster.author_id.nunique()),'edges':len(e),
            'citing_works':len(source),'cited_works':int(e.cited_id.nunique()),
            'roster_shared_author_edges':len(self_edges),'verified_roster_edges':len(inside),
            'roster_query_complete':bool(a.author_audit)}
    (a.out/'diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(summary.to_string(index=False))
    print(by_field.to_string(index=False))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
