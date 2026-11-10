"""Find chronologically ordered citation returns from the archived data."""
import argparse
import base64
import hashlib
import hmac
import json
from pathlib import Path
import sqlite3

import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('archive','salt','release','out'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(mode=0o700,parents=True,exist_ok=True)
    with sqlite3.connect(args.archive.resolve().as_uri()+'?mode=ro&immutable=1',uri=True) as c:
        pub=pd.read_sql_query('SELECT professor_uid,work_id,publication_year FROM faculty_publication',c)
        first=pd.read_sql_query('SELECT cited_id AS a1,citing_id AS b,citing_year AS yb FROM openalex_incoming',c)
        last=pd.read_sql_query('SELECT citing_id AS a2,cited_id AS b FROM faculty_outgoing',c)
    last=last.loc[last.b.isin(first.b)]
    last=last.merge(pub.rename(columns={'work_id':'a2','publication_year':'y2'}),on='a2')
    first=first.loc[first.b.isin(last.b)]
    first=first.merge(pub.rename(columns={'work_id':'a1','publication_year':'y1'}),on='a1')
    first=first.loc[(first.yb-first.y1).between(1,3)]
    paths=first.merge(last,on=['professor_uid','b'])
    paths=paths.loc[(paths.y1<paths.yb) & (paths.yb<paths.y2) & (paths.y2<=2025)]
    paths=paths.drop_duplicates(['professor_uid','a1','b','a2'])
    # A recognized coauthorship on B disqualifies a path even if the OpenAlex
    # authors list is incomplete or a linked author ID has changed since collection.
    blocked=pd.MultiIndex.from_frame(pub[['professor_uid','work_id']])
    paths=paths.loc[~pd.MultiIndex.from_frame(paths[['professor_uid','b']]).isin(blocked)]
    salt=args.salt.read_bytes()
    def anon(uid):
        digest=hmac.new(salt,str(uid).encode(),hashlib.sha256).digest()
        return 'P-'+base64.b32encode(digest).decode()[:10]
    paths['anon']=paths.professor_uid.map(anon)
    paths=paths.loc[paths.anon.isin({x.name for x in args.release.iterdir() if x.is_dir()})].copy()
    # Publish only paths whose own papers both occur in the existing person's chart.
    present={}
    for key in paths.anon.unique():
        doc=json.loads((args.release/key/'alluvial.json').read_text())
        lookup={}
        for x in doc['papers']:
            wid=x.get('id') or x.get('work_id') or x.get('openalex_id')
            if wid and str(wid).split('/')[-1].startswith('W'):
                lookup[int(str(wid).split('/')[-1][1:])]=x
        present[key]=lookup
    keep=[(int(x.a1) in present[x.anon] and int(x.a2) in present[x.anon]) for x in paths.itertuples(index=False)]
    if not any(keep) and len(paths):
        raise RuntimeError('No matched A1/A2 paper IDs in topic release; inspect the release paper schema')
    paths=paths.loc[keep].copy()
    paths['a1_title']=[present[x.anon][x.a1].get('title') for x in paths.itertuples(index=False)]
    paths['a2_title']=[present[x.anon][x.a2].get('title') for x in paths.itertuples(index=False)]
    paths['a1_topic']=[present[x.anon][x.a1].get('topic') for x in paths.itertuples(index=False)]
    paths['a2_topic']=[present[x.anon][x.a2].get('topic') for x in paths.itertuples(index=False)]
    paths.to_csv(args.out/'paths_to_audit.csv.gz',index=False)
    print(json.dumps({'candidate_paths':len(paths),'middle_papers':paths.b.nunique(),
                      'researchers':paths.anon.nunique()},indent=2),flush=True)


if __name__=='__main__':
    main()
