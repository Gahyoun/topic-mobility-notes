"""Validate B authorship and seal ABA paths using the dashboard title key."""
import argparse
import base64
from collections import defaultdict
from functools import lru_cache
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import unicodedata

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import pandas as pd

ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567'


def author_id(value):
    if isinstance(value,str):
        result=re.search(r'A(\d+)$',value)
        if result:
            return int(result[1])
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('paths','cache','key-meta','out','paths-config'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--featured-professor-uid', help='Optional researcher UID from the private workspace to prioritize')
    a=p.parse_args()
    cfg=json.loads(a.paths_config.read_text())
    db=Path(cfg['web_db'])
    with sqlite3.connect(db.as_uri()+'?mode=ro&immutable=1',uri=True) as con:
        faculty=dict(con.execute('SELECT professor_uid,openalex_id FROM professors'))
        alias_rows=con.execute("SELECT DISTINCT professor_uid,openalex_id FROM work_authorship_evidence WHERE identity_decision='keep'").fetchall()
    principal={uid:author_id(value) for uid,value in faculty.items()}
    ownership=defaultdict(set)
    for uid,aid in principal.items():
        if aid is not None:
            ownership[aid].add(uid)
    possible=defaultdict(set)
    for uid,value in alias_rows:
        aid=author_id(value)
        if aid is not None:
            possible[uid].add(aid)
    meta=json.loads(a.key_meta.read_text())
    password=os.environ['DASHBOARD_TITLE_PASSWORD']
    key=hashlib.pbkdf2_hmac('sha256',password.encode(),base64.b64decode(meta['salt']),meta['iterations'],dklen=32)
    aes=AESGCM(key)
    check=meta['check']
    decoded=aes.decrypt(base64.b64decode(check['nonce']),base64.b64decode(check['ciphertext']),
                        (meta['aad']+'check').encode())
    assert json.loads(gzip.decompress(decoded))=='ok'
    def title_key(year,title):
        return int(year), ' '.join(unicodedata.normalize('NFC',title).split()).casefold()

    @lru_cache(maxsize=None)
    def live_titles(anon):
        file=a.key_meta.parent/(anon+'.json')
        if not file.exists():
            return set()
        sealed=json.loads(file.read_text())
        raw=aes.decrypt(base64.b64decode(sealed['nonce']),base64.b64decode(sealed['ciphertext']),
                        (meta['aad']+anon).encode())
        return {title_key(r['year'],r['title']) for r in json.loads(gzip.decompress(raw))}

    records={}
    attempted=set()
    for f in a.cache.glob('*.json'):
        j=json.loads(f.read_text())
        attempted.update(j['requested'])
        for record in j['records']:
            records[record[0]]=record
    # Copy each paper title once per encrypted shard.
    shards={key:{'version':1,'works':{},'researchers':{}} for key in ALPHABET}
    rejects=defaultdict(int)
    matched_middle=set()
    researchers=set()
    preview=None
    paths=pd.read_csv(a.paths).sort_values(['y2','yb','y1','a2','b','a1'],ascending=[False,False,False,True,True,True])
    for r in paths.itertuples(index=False):
        b=records.get(r.b)
        if b is None:
            rejects['middle_unavailable']+=1
            continue
        wid,title,year,doi,authors=b
        if year!=r.yb or not isinstance(title,str) or not title.strip():
            rejects['middle_metadata_conflict']+=1
            continue
        aid=principal.get(r.professor_uid)
        if aid is None or len(ownership[aid])!=1 or not authors or None in authors or len(authors)>=100:
            rejects['author_not_verified']+=1
            continue
        if aid in authors or possible[r.professor_uid].intersection(authors):
            rejects['middle_has_focal_author']+=1
            continue
        if not (isinstance(r.a1_title,str) and r.a1_title.strip() and isinstance(r.a2_title,str) and r.a2_title.strip()):
            rejects['endpoint_title_unavailable']+=1
            continue
        if not (r.y1<year<r.y2<=2025 and year-r.y1<=3 and len({r.a1,r.b,r.a2})==3):
            raise ValueError('Invalid chronological path')
        assert re.fullmatch(r'P-[A-Z2-7]{10}',r.anon)
        live=live_titles(r.anon)
        if title_key(r.y1,r.a1_title) not in live or title_key(r.y2,r.a2_title) not in live:
            rejects['endpoint_absent_in_current_release']+=1
            continue
        blob=shards[r.anon[2]]
        for k,y,t in ((r.a1,r.y1,r.a1_title),(wid,year,title),(r.a2,r.y2,r.a2_title)):
            paper=blob['works'].setdefault(str(k),[int(y),t.strip()])
            if paper[0]!=int(y):
                raise ValueError('Conflicting publication years for one paper')
        blob['researchers'].setdefault(r.anon,[]).append([int(r.a1),int(wid),int(r.a2)])
        matched_middle.add(wid)
        researchers.add(r.anon)
        if r.professor_uid == a.featured_professor_uid:
            preview=r.anon
    total=sum(len(rows) for b in shards.values() for rows in b['researchers'].values())
    if not total:
        raise ValueError('No paths passed the author checks')
    a.out.mkdir(parents=True,exist_ok=True)
    for shard,obj in shards.items():
        raw=json.dumps(obj,ensure_ascii=False,separators=(',',':')).encode()
        nonce=os.urandom(12)
        aad=f'topic-aba-v1:{shard}'.encode()
        sealed=aes.encrypt(nonce,gzip.compress(raw,compresslevel=9,mtime=0),aad)
        assert gzip.decompress(aes.decrypt(nonce,sealed,aad))==raw
        (a.out/(shard+'.json')).write_text(json.dumps({'nonce':base64.b64encode(nonce).decode(),
            'ciphertext':base64.b64encode(sealed).decode()},separators=(',',':'))+'\n')
    diag={'candidate_paths':len(paths),'candidate_middle_papers':paths.b.nunique(),
          'middle_requested':len(attempted),'middle_metadata_available':len(records),
          'verified_paths':total,'verified_middle':len(matched_middle),
          'researchers_with_verified_paths':len(researchers),'excluded_by_reason':dict(rejects)}
    (a.paths.parent/'published_summary.json').write_text(json.dumps(diag,indent=2)+'\n')
    (a.paths.parent/'preview-id').write_text((preview or min(researchers))+'\n')
    print(json.dumps(diag,indent=2))


if __name__=='__main__':
    main()
