"""Freeze the prior local comparison and define a balanced API pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compare_network import load_inputs, citation_edges, graph, KEY, LEVELS, SUBJECTS


def fingerprint(rows):
    frame = rows[KEY + ['id','year'] + LEVELS].sort_values(KEY+['id'])
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-root', type=Path, required=True)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--sample-dir', type=Path, required=True)
    parser.add_argument('--per-field', type=int, default=32)
    args = parser.parse_args()
    here = args.sample_dir
    here.mkdir(parents=True, exist_ok=True, mode=0o700)
    cfg = json.loads(args.paths.read_text())
    rows, space_ids, _, good, edges, years, _ = load_inputs(
        cfg,args.private_root,args.selection)
    eligible = rows.id.isin(space_ids[good]) & rows.id.map(years).eq(rows.year)
    core = pd.Index(np.sort(rows.loc[eligible,'id'].unique()))
    ref, incoming, added = citation_edges(edges,core,years)
    _, out, _ = graph(core,ref)
    _, combined, _ = graph(core,pd.concat([ref,added],ignore_index=True))
    common = rows.loc[eligible & rows.id.isin(core[out & combined])].copy()
    common = common.loc[common.groupby(KEY).id.transform('size') >= 2].copy()
    assert common.id.nunique()==117602 and common.groupby(KEY).ngroups==19849 and common.professor_uid.nunique()==2985
    common[KEY+['id','year']+LEVELS].to_csv(here/'frozen_rows.csv.gz',index=False)
    pd.DataFrame({'id':core,'year':years.loc[core].astype(int).to_numpy()}).to_csv(here/'graph_targets.csv.gz',index=False)

    # Seeded stratified random draw, determined before any API results.
    # Retain each chosen teacher's existing windows from 6 years before to
    # 19 years after appointment. Require three post-appointment windows.
    rng=np.random.default_rng(61108)
    pilot=[]
    for field in SUBJECTS:
        frame=common.loc[common.subject.eq(field) & common.win.between(-3,9)]
        counts=frame.groupby('professor_uid').size()
        post=frame.loc[frame.win.ge(0)].groupby('professor_uid').win.nunique()
        # Limit the number of queries without truncating individual windows.
        candidates=counts.index[(counts<=65) & (post.reindex(counts.index,fill_value=0)>=3)]
        chosen=rng.choice(np.sort(candidates),size=min(args.per_field,len(candidates)),replace=False)
        pilot.append(frame.loc[frame.professor_uid.isin(chosen)])
    pilot=pd.concat(pilot,ignore_index=True)
    pilot[KEY+['id','year']].to_csv(here/'pilot_rows.csv.gz',index=False)
    sizes=pilot.groupby('subject').agg(professors=('professor_uid','nunique'),works=('id','nunique')).to_dict('index')
    print(json.dumps({'frozen_sha256':fingerprint(common),'graph_targets':len(core),
        'pilot':sizes,'pilot_works':pilot.id.nunique(),'pilot_windows':pilot.groupby(KEY).ngroups},indent=2))
    (here/'sample_summary.json').write_text(json.dumps({'frozen_sha256':fingerprint(common),'graph_targets':len(core),
        'pilot':sizes,'pilot_works':pilot.id.nunique(),'pilot_windows':pilot.groupby(KEY).ngroups},indent=2)+'\n')

if __name__=='__main__':
    main()
