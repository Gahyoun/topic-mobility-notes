"""Download and cache bibliographic evidence for the middle papers of ABA paths."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'external'))
from collect_openalex import OpenAlex


def num(s):
    token=str(s).rsplit('/',1)[-1]
    return int(token[1:]) if len(token)>1 and token[1:].isdigit() else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--paths',type=Path,required=True)
    p.add_argument('--max-requests',type=int,default=1200)
    p.add_argument('--prepaid-cap',type=float,default=0.)
    p.add_argument('--featured-professor-uid', help='Optional researcher UID from the private workspace to prioritize')
    a=p.parse_args()
    a.cache.mkdir(mode=0o700,parents=True,exist_ok=True)
    paths=pd.read_csv(a.paths,usecols=['professor_uid','b','yb','a1','a2'])
    # Start with one middle paper per person, then round-robin further papers.
    ordered=paths.sort_values(['yb','b','a1','a2'],ascending=[False,True,True,True])
    pairs=ordered[['professor_uid','b']].drop_duplicates().copy()
    pairs['rank']=pairs.groupby('professor_uid').cumcount()
    ordered=list(dict.fromkeys(pairs.sort_values(['rank','b']).b.tolist()))
    if a.featured_professor_uid:
        featured_ids = paths.loc[paths.professor_uid.eq(a.featured_professor_uid), 'b'].drop_duplicates()
        ordered = list(dict.fromkeys([*featured_ids, *ordered]))
    done=set()
    for f in a.cache.glob('*.json'):
        done.update(json.loads(f.read_text())['requested'])
    pending=[b for b in ordered if b not in done]
    batches=[pending[i:i+100] for i in range(0,len(pending),100)]
    key=os.environ.get('OPENALEX_API_KEY')
    api=OpenAlex(key,a.max_requests,100,a.prepaid_cap,.043)
    if not key:
        api.interval=.2
    def fetch(ids):
        q={'filter':'openalex_id:'+'|'.join(f'W{i}' for i in ids),'per_page':100,'select':'id,title,doi,publication_year,authorships'}
        r=api.get(q)
        if r['meta']['count']!=len(r['results']):
            raise RuntimeError('Incomplete middle-paper ID batch')
        result=[]
        for w in r['results']:
            wid=num(w.get('id'))
            if wid not in ids:
                continue
            authors=[num((x.get('author') or {}).get('id')) for x in (w.get('authorships') or [])]
            result.append([wid,w.get('title'),w.get('publication_year'),w.get('doi'),authors])
        name=hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:20]
        dest=a.cache/(name+'.json')
        tmp=dest.with_suffix('.tmp')
        tmp.write_text(json.dumps({'requested':ids,'records':result},ensure_ascii=False,separators=(',',':')))
        tmp.replace(dest)
        return len(result)
    selected=batches[:a.max_requests]
    n=0
    failed=0
    with ThreadPoolExecutor(max_workers=20 if key else 7) as pool:
        for f in as_completed([pool.submit(fetch,part) for part in selected]):
            try:
                n+=f.result()
            except RuntimeError:
                failed+=1
    print(json.dumps({'requested_before':len(done),'new_records':n,'new_requests':api.calls,
                      'failed_batches':failed,'total_candidates':len(ordered)},indent=2),flush=True)


if __name__=='__main__':
    main()
