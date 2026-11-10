"""Check citing works against the OpenAlex index for all known roster author IDs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'external'))
from collect_openalex import OpenAlex


def ro(path):
    return sqlite3.connect(Path(path).as_uri()+'?mode=ro&immutable=1',uri=True)


def id_num(value, letter):
    if not isinstance(value,str):
        return None
    last=value.rsplit('/',1)[-1]
    if not last.startswith(letter) or not last[1:].isdigit():
        return None
    return int(last[1:])


def cursor_works(api, filt, fields):
    params={'filter':filt,'select':fields,'per_page':100,'cursor':'*'}
    ids=set()
    expected=None
    while True:
        reply=api.get(params)
        if expected is None:
            expected=int(reply['meta']['count'])
        for work in reply['results']:
            w=id_num(work['id'],'W')
            if w is not None and w not in ids:
                ids.add(w)
                yield w,work
        if len(ids)==expected:
            return
        cursor=reply['meta'].get('next_cursor')
        if len(ids)>expected or not cursor or not reply['results'] or cursor==params['cursor']:
            raise RuntimeError('OpenAlex result count could not be reconciled')
        params['cursor']=cursor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('paths','archive','refs','cache','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prepaid-cap',type=float,default=.60)
    args=p.parse_args()
    cfg=json.loads(args.paths.read_text())
    args.cache.mkdir(mode=0o700,parents=True,exist_ok=True)
    with ro(cfg['web_db']) as db:
        roster=pd.read_sql_query('SELECT professor_uid,openalex_id FROM professors',db)
    roster['aid']=roster.openalex_id.map(lambda s:id_num(s,'A'))
    known=roster.dropna(subset=['aid']).copy()
    known.aid=known.aid.astype('int64')
    distinct=sorted(set(known.aid))
    unambiguous=set(known.loc[~known.aid.duplicated(keep=False),'aid'])
    with ro(args.archive) as db:
        all_citing={w for w, in db.execute('SELECT DISTINCT citing_id FROM openalex_incoming')}
        verified={w for w, in db.execute('SELECT DISTINCT citing_id FROM openalex_incoming JOIN faculty_work ON work_id=citing_id')}
        lower,upper=db.execute('SELECT min(citing_year),max(citing_year) FROM openalex_incoming').fetchone()
    with ro(args.refs) as db:
        aliases={id_num(k,'W'):id_num(v,'W') for k,v in db.execute('SELECT requested_id,canonical_id FROM work_alias')}
    with ro(cfg['koad_db']) as db:
        refused=pd.read_sql_query("SELECT professor_uid,work_id FROM work_decision WHERE decision='namesake_paper' OR recommended_action='drop'",db)
    refused=refused.merge(known[['professor_uid','aid']],on='professor_uid')
    refused['work_id']=refused.work_id.map(lambda x:id_num(x,'W'))
    refused['work_id']=refused.work_id.map(aliases).fillna(refused.work_id)
    excluded=set(refused[['work_id','aid']].itertuples(index=False,name=None))
    key=os.environ.get('OPENALEX_API_KEY')
    if not key:
        p.error('An authorized OpenAlex key is required')
    api=OpenAlex(key,6000,100,args.prepaid_cap,.042)

    def collect(part):
        fingerprint=hashlib.sha256(json.dumps([lower,upper,part]).encode()).hexdigest()[:20]
        path=args.cache/(fingerprint+'.json')
        if path.exists():
            saved=json.loads(path.read_text())
            assert saved['authors']==part
            return saved
        def enumerate_works(authors):
            filt='author.id:'+'|'.join(f'A{x}' for x in authors)+f',publication_year:{lower}-{upper}'
            try:
                works={w for w,_ in cursor_works(api,filt,'id')}
            except RuntimeError as error:
                if str(error) != 'OpenAlex result count could not be reconciled' or len(authors) == 1:
                    raise
                cut=len(authors)//2
                return enumerate_works(authors[:cut]) | enumerate_works(authors[cut:])
            return works

        source_works=enumerate_works(part)
        works=len(source_works)
        hits={aliases.get(w,w) for w in source_works}
        hits &= all_citing
        saved={'authors':part,'works':works,'hits':sorted(hits)}
        temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(saved,separators=(',',':')))
        temp.replace(path)
        return saved

    parts=[distinct[i:i+100] for i in range(0,len(distinct),100)]
    hit=set()
    results=[]
    with ThreadPoolExecutor(max_workers=26) as pool:
        for f in as_completed([pool.submit(collect,part) for part in parts]):
            saved=f.result()
            results.append(saved)
            hit.update(saved['hits'])
            print(f'{len(results)}/{len(parts)} roster batches verified; {api.calls} calls',flush=True)
    assert {a for g in results for a in g['authors']}==set(distinct)
    candidates=sorted(hit-verified)
    verdict={w:'unresolved' for w in candidates}

    def check(part):
        filt='openalex_id:'+'|'.join(f'W{x}' for x in part)
        returned={}
        for w,work in cursor_works(api,filt,'id,authorships'):
            w=aliases.get(w,w)
            author_ids={id_num(item.get('author',{}).get('id'),'A') for item in work.get('authorships') or []}
            allowed={aid for aid in author_ids & unambiguous if (w,aid) not in excluded}
            if w in part and allowed:
                returned[w]='roster_verified'
        return returned

    with ThreadPoolExecutor(max_workers=26) as pool:
        for d in pool.map(check,[candidates[i:i+100] for i in range(0,len(candidates),100)]):
            verdict.update(d)
    pd.DataFrame(sorted(verdict.items()),columns=['citing_id','status']).to_csv(args.output,index=False)
    summary={'roster_query_complete':True,'roster_professors':len(roster),
        'professors_with_openalex_id':len(known),'queried_author_ids':len(distinct),
        'ids_shared_by_multiple_professors':len(distinct)-len(unambiguous),
        'citing_ids_index_match':len(hit),'local_verified_citing_ids':len(verified),
        'local_verified_missing_from_api_index':len(verified-hit),
        'additional_verified_by_index_and_authorships':sum(x=='roster_verified' for x in verdict.values()),
        'index_matches_unresolved':sum(x=='unresolved' for x in verdict.values()),
        'profile_work_results_across_batches':sum(x['works'] for x in results),
        'api_calls':api.calls,'actual_cost_upper_bound_usd':api.calls*.0001}
    args.output.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
