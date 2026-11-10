"""Collect complete OpenAlex cited-by batches. No names or private IDs are transmitted."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import random
import threading
import time

import pandas as pd
import requests


class OpenAlex:
    def __init__(self, key, max_requests, batch_size, prepaid_cap, key_interval):
        self.key = key
        self.max_requests = max_requests
        self.calls = 0
        self.remaining = None
        self.allow_prepaid = prepaid_cap > 0
        self.prepaid_remaining = None
        self.prepaid_floor = None
        self.stop_reason = None
        if key:
            response = requests.get('https://api.openalex.org/rate-limit',
                                    headers={'Authorization': 'Bearer ' + key}, timeout=15)
            if not response.ok:
                raise RuntimeError(f'OpenAlex quota check returned HTTP {response.status_code}')
            quota = response.json()['rate_limit']
            daily = int(quota['credits_remaining'])
            self.prepaid_remaining = int(quota['onetime_credits_remaining'])
            prepaid = min(int(prepaid_cap * 10000 + .00001), self.prepaid_remaining)
            self.prepaid_floor = self.prepaid_remaining - prepaid
            # Reserve more than the maximum number of requests in flight.
            self.max_requests = min(max_requests, max(0, daily + prepaid - 64))
            print(f'Request budget: {self.max_requests}; free quota: {daily}; authorized prepaid cap: {prepaid}',flush=True)
        elif self.allow_prepaid:
            raise ValueError('Prepaid use requires an API key')
        self.interval = key_interval if key else (.18 if batch_size <= 5 else 1.05)
        self.next_request = 0.
        self.lock = threading.Lock()
        self.clients = threading.local()

    def get(self, params):
        if not hasattr(self.clients, 'session'):
            self.clients.session = requests.Session()
        client = self.clients.session
        headers = {'Authorization': 'Bearer ' + self.key} if self.key else {}
        for retry in range(5):
            with self.lock:
                if self.stop_reason:
                    raise RuntimeError(self.stop_reason)
                free_low = self.remaining is not None and self.remaining <= (64 if self.key else 6)
                prepaid_low = self.prepaid_remaining is not None and self.prepaid_remaining <= self.prepaid_floor + 64
                if self.calls >= self.max_requests or (free_low and (not self.allow_prepaid or prepaid_low)):
                    raise RuntimeError('API request budget reached')
                now = time.monotonic()
                pause = max(0, self.next_request - now)
                self.next_request = now + pause + self.interval
                self.calls += 1
            time.sleep(pause)
            try:
                r = client.get('https://api.openalex.org/works', params=params, headers=headers, timeout=(8,45))
            except requests.RequestException:
                if retry == 4:
                    raise RuntimeError('OpenAlex connection retry limit reached') from None
                time.sleep(min(12,2**retry))
                continue
            raw_remaining = r.headers.get('x-ratelimit-remaining')
            if raw_remaining and raw_remaining.isdigit():
                with self.lock:
                    self.remaining = min(int(raw_remaining), self.remaining if self.remaining is not None else int(raw_remaining))
            prepaid_header = r.headers.get('x-ratelimit-onetime-remaining')
            if self.prepaid_remaining is not None and prepaid_header and prepaid_header.isdigit():
                with self.lock:
                    self.prepaid_remaining = min(self.prepaid_remaining, int(prepaid_header))
            if r.ok:
                return r.json()
            if r.status_code == 429:
                delay = r.headers.get('retry-after','')
                if delay.isdigit() and int(delay) > 180:
                    with self.lock:
                        self.stop_reason = 'OpenAlex rate limit reached'
                    raise RuntimeError(self.stop_reason)
                wait = max(int(delay) if delay.isdigit() else 1, 2**retry)
                with self.lock:
                    self.interval = max(self.interval, .22) if self.key else self.interval
                    self.next_request = max(self.next_request, time.monotonic() + wait)
                time.sleep(wait)
                continue
            if r.status_code in (500,502,503,504):
                time.sleep(2**retry)
                continue
            raise RuntimeError(f'OpenAlex returned HTTP {r.status_code}')
        raise RuntimeError('OpenAlex temporary failures exceeded retry limit')


def short(identifier):
    return int(identifier.rsplit('/',1)[-1].removeprefix('W'))


def fetch(api, ids, year):
    expect=set(ids)
    fields='id,publication_year,referenced_works'

    def pull(lower,upper,targets):
        filters = 'cites:'+'|'.join(f'W{x}' for x in targets)+f',publication_year:{lower}-{upper}'
        params={'filter':filters,'per_page':100,'select':fields}
        if api.key:
            params['cursor']='*'
        else:
            params['page']=1
        first=api.get(params)
        count=first['meta']['count']
        if not api.key and count > 10000:
            if len(targets)>1:
                mid=len(targets)//2
                return pull(lower,upper,targets[:mid]) | pull(lower,upper,targets[mid:])
            if lower<upper:
                return {k:v for y in range(lower,upper+1) for k,v in pull(y,y,targets).items()}
            raise RuntimeError('A target-year exceeds the unkeyed paging cap')
        seen={}
        result=first
        while True:
            for work in result['results']:
                citing=short(work['id'])
                y=work['publication_year']
                if y is None or not lower<=y<=upper:
                    raise ValueError('OpenAlex returned a publication year outside the query')
                refs=sorted({short(x) for x in work.get('referenced_works') or []} & expect)
                seen[citing]=(y,refs)
            if len(seen)==count:
                return seen
            if api.key:
                cursor=result['meta'].get('next_cursor')
                if not cursor:
                    break
                params['cursor']=cursor
            else:
                if params['page']>=math.ceil(count/100):
                    break
                params['page']+=1
            result=api.get(params)
            if not result['results']:
                break
        # A changing result order can duplicate or omit hits in basic pagination.
        # Partition the date range, then split target IDs if a single year still fails.
        if lower<upper:
            return {k:v for y in range(lower,upper+1) for k,v in pull(y,y,targets).items()}
        if len(targets)>1:
            mid=len(targets)//2
            return pull(lower,upper,targets[:mid]) | pull(lower,upper,targets[mid:])
        raise RuntimeError('API result count could not be reconciled with distinct citing IDs')

    got=pull(year+1,year+3,ids)
    return {'year':year,'targets':ids,'citing_records':len(got),
            'unresolved_records':sum(not refs for _,refs in got.values()),
            'edges':[[citing,cited,y] for citing,(y,refs) in got.items() for cited in refs]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--targets',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-requests',type=int,default=930)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=75)
    p.add_argument('--prepaid-cap-usd',type=float,default=0,
                   help='Use existing prepaid credits only up to this explicit cap, after free quota')
    p.add_argument('--key-interval',type=float,default=.075)
    args=p.parse_args()
    args.cache.mkdir(parents=True,exist_ok=True,mode=0o700)
    targets=pd.read_csv(args.targets,usecols=['id','year']).drop_duplicates()
    if not targets.id.is_unique:
        raise ValueError('A target has conflicting publication years')
    existing = []
    covered = set()
    planned = set(targets.itertuples(index=False, name=None))
    for path in sorted(args.cache.glob('*.json')):
        saved = json.loads(path.read_text())
        pairs = {(work, saved['year']) for work in saved['targets']}
        if not pairs <= planned or pairs & covered:
            raise ValueError('Checkpoint targets do not partition the requested sample')
        covered.update(pairs)
        existing.append(saved)
    todo = targets.loc[[(work,year) not in covered for work,year in targets.itertuples(index=False, name=None)]]
    groups=[]
    for year,sub in todo.groupby('year'):
        ids=sorted(sub.id.tolist())
        for start in range(0,len(ids),args.batch_size):
            batch=ids[start:start+args.batch_size]
            digest=hashlib.sha256(json.dumps([int(year),batch]).encode()).hexdigest()[:20]
            groups.append((int(year),batch,args.cache/(digest+'.json')))
    # Spread query progress across years. Re-running uses completed batches.
    random.Random(62724).shuffle(groups)
    if args.prepaid_cap_usd < 0 or args.key_interval < 1/28:
        p.error('Prepaid cap must be nonnegative; key interval must respect the API limit')
    api=OpenAlex(os.environ.get('OPENALEX_API_KEY'),args.max_requests,args.batch_size,
                 args.prepaid_cap_usd,args.key_interval)

    def collect(group):
        year,ids,path=group
        if path.exists():
            result=json.loads(path.read_text())
            if result['year']!=year or result['targets']!=ids:
                raise ValueError('Invalid checkpoint')
            return result
        result=fetch(api,ids,year)
        temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(result,separators=(',',':')))
        temp.replace(path)
        return result

    completed,failed=list(existing),[]
    total_groups=len(existing)+len(groups)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(collect,g):g for g in groups}
        for job in as_completed(jobs):
            try:
                completed.append(job.result())
            except (ValueError,RuntimeError,KeyError) as e:
                failed.append(str(e))
            if (len(completed)+len(failed))%10==0:
                print(f'{len(completed)}/{total_groups} groups complete; {api.calls} calls; {len(failed)} not complete',flush=True)
    triples=pd.DataFrame([edge for batch in completed for edge in batch['edges']],
                         columns=['citing','cited','citing_year']).drop_duplicates()
    triples.to_csv(args.output,index=False)
    output={'groups':total_groups,'completed_groups':len(completed),
            'planned_targets':len(targets),'completed_targets':sum(len(x['targets']) for x in completed),
            'unresolved_result_records':sum(x['unresolved_records'] for x in completed),
            'new_api_requests':api.calls,'distinct_citing_records':triples.citing.nunique(),
            'edges':len(triples),'errors':sorted(set(failed))}
    args.output.with_suffix('.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2),flush=True)
    if failed:
        raise SystemExit(2)


if __name__=='__main__':
    main()
