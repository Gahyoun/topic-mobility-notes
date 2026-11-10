import csv,json,re
from pathlib import Path
p=Path(__file__).parent
pending=json.loads((p/'pending.json').read_text())
with (p/'previous-decisions.tsv').open() as f:
 rows=list(csv.reader(f,delimiter='\t'))
ids={row[0] for row in rows[1:]}
for line in (p/'round2.psv').read_text().splitlines():
 subject,school,name,status,year,inst,url,basis=line.split('|')
 matches=[r for r in pending if (r['subject'],r['school'],r['name'])==(subject,school,name)]
 assert len(matches)==1,(subject,school,name)
 uid=matches[0]['uid']
 assert uid not in ids and re.fullmatch(r'https://[^/]+\.ac\.kr/\S*',url),(uid,url)
 assert status in ('confirmed','ambiguous','current_only')
 assert bool(year)==(status=='confirmed')
 ids.add(uid)
 rows.append([uid,status,year,inst,url,basis])
with (p/'decisions.tsv').open('w') as f:
 csv.writer(f,delimiter='\t',lineterminator='\n').writerows(rows)
print(f'{len(rows)-1} cumulative judgments, {sum(r[1]=="confirmed" for r in rows[1:])} verified')
