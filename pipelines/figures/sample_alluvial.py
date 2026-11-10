"""Render aggregate appointment-analysis coverage and metric overlap."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sqlite3

FIELDS={'all':'전체','mathematics':'수학','physics':'물리','chemistry':'화학','biology':'생물'}


def summarize(db_file, rows_file):
    source=db_file.resolve().as_uri()+'?mode=ro&immutable=1'
    with sqlite3.connect(source,uri=True) as db:
        roster={uid:(field,papers,refs) for uid,field,papers,refs in db.execute(
            'SELECT uid,field,n_papers,n_supported_refs FROM faculty_observation')}
    with rows_file.open(newline='') as f:
        rows={r['uid']:r for r in csv.DictReader(f)}
    assert set(rows)=={uid for uid,(_,_,refs) in roster.items() if refs>0}

    data={}
    for key,label in FIELDS.items():
        counts=Counter()
        overlap=Counter()
        for uid,(field,papers,refs) in roster.items():
            if key!='all' and field!=key:
                continue
            counts['roster']+=1
            counts['linked']+=int(papers>0)
            if not refs:
                counts['no_link' if not papers else 'no_reference']+=1
                continue
            counts['references']+=1
            r=rows[uid]
            assert r['field']==field
            if r['dated']=='False':
                counts['no_date']+=1
                continue
            assert r['dated']=='True'
            counts['dated']+=1
            before,after=(float(r[key]) for key in ('pre_papers','post_papers'))
            flags=(r['excess']!='',r['pre_multi']!='' and r['post_multi']!='',r['branch']!='')
            if float(r['followup'])==0:
                counts['no_followup']+=1
            elif after==0:
                counts['post_zero']+=1
            elif before==0:
                counts['pre_zero']+=1
            else:
                counts['pre_post']+=1
                gcc=float(r['pre_positioned'])>0 and float(r['post_positioned'])>0
                counts['gcc_both']+=int(gcc)
                counts['no_gcc_both']+=int(not gcc)
                counts['gcc_unmeasured']+=int(gcc and not any(flags))
                mask=sum(2**i for i,flag in enumerate(flags) if flag)
                overlap[mask]+=1
                for metric,flag in zip(('transport','parallel','branch'),flags,strict=True):
                    counts[metric]+=int(flag)
                if mask:
                    counts['any']+=1
                    counts['all_metrics']+=int(mask==7)
                    counts['two_only']+=int(sum(flags)==2)
                    counts['one_only']+=int(sum(flags)==1)
                assert not any(flags) or gcc
                continue
            assert not any(flags)
        assert counts['roster']==counts['references']+counts['no_link']+counts['no_reference']
        assert counts['linked']==counts['references']+counts['no_reference']
        assert counts['references']==counts['dated']+counts['no_date']
        assert counts['dated']==sum(counts[k] for k in ('pre_post','pre_zero','post_zero','no_followup'))
        assert counts['pre_post']==sum(counts[k] for k in ('any','no_gcc_both','gcc_unmeasured'))
        assert counts['any']==sum(counts[k] for k in ('all_metrics','two_only','one_only'))
        data[key]={'label':label,'counts':dict(counts),'overlap':dict(overlap)}
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('portfolio','faculty_csv','output'):
        p.add_argument(key,type=Path)
    p.add_argument('--counts',type=Path)
    a=p.parse_args()
    data=summarize(a.portfolio,a.faculty_csv)
    html=Path(__file__).with_suffix('.html').read_text()
    a.output.write_text(html.replace('@@COUNTS@@',json.dumps(data,ensure_ascii=False,separators=(',',':'))))
    if a.counts:
        a.counts.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
