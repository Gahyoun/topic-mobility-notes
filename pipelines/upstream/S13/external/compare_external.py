"""Add OpenAlex links to the original graph and compare on exactly fixed paper lists."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compare_network import (
    load_inputs, ro, citation_edges, graph, pivot_mds, validate_distances,
    calculate_panel, comparison_stats, KEY, LEVELS,
)
from freeze_sample import fingerprint


METHODS = ['classification','shared_references','citation_network_external','citation_network_local']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--private-root',type=Path,required=True)
    p.add_argument('--paths',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--sample-dir',type=Path,required=True)
    p.add_argument('--edges',type=Path)
    p.add_argument('--local-only',action='store_true')
    p.add_argument('--all',action='store_true',help='Require complete API coverage of the full graph and compare the full frozen sample')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--save-coordinates',type=Path,help='Private path for fitted coordinates')
    args=p.parse_args()
    if not args.local_only and not args.edges:
        p.error('--edges is required unless --local-only is set')
    args.out.mkdir(parents=True,exist_ok=True)
    here=args.sample_dir
    cfg=json.loads(args.paths.read_text())
    source=args.private_root
    _, space_ids, vectors, _, cache_edges, years, _=load_inputs(
        cfg,source,args.selection)
    frozen=pd.read_csv(here/'frozen_rows.csv.gz',dtype={k:str for k in LEVELS})
    recorded=json.loads((here/'sample_summary.json').read_text())
    assert fingerprint(frozen)==recorded['frozen_sha256']
    selected=frozen
    if not args.all:
        pilot=pd.read_csv(here/'pilot_rows.csv.gz')
        selected=frozen.merge(pilot,on=KEY+['id','year'],validate='one_to_one')
        assert len(selected)==len(pilot)
    core=pd.Index(pd.read_csv(here/'graph_targets.csv.gz').id)
    ref, _, added=citation_edges(cache_edges,core,years)
    _,out_core,_=graph(core,ref)
    adj,local_core,local_info=graph(core,pd.concat([ref,added],ignore_index=True))
    common=np.flatnonzero(out_core & local_core)
    common_ids=core[common]
    assert selected.id.isin(common_ids).all()
    local_cache=here/'local_coordinates_and_pivots.npz'
    if local_cache.exists():
        with np.load(local_cache) as saved:
            np.testing.assert_array_equal(saved['ids'],common_ids.to_numpy())
            local=saved['coords']
            anchors=saved['pivots']
    else:
        anchors=[]
        with threadpool_limits(limits=4):
            local,_=pivot_mds(adj,common,150,64,pivots_out=anchors)
        np.savez(local_cache,ids=common_ids.to_numpy(),coords=local,pivots=anchors)
    print(f'Frozen original fit ({len(common_ids):,} graph works) ready',flush=True)
    if args.local_only:
        return
    acquisition=json.loads(args.edges.with_suffix('.json').read_text())
    n_targets=len(core) if args.all else selected.id.nunique()
    assert acquisition['planned_targets']==acquisition['completed_targets']==n_targets
    assert acquisition['completed_groups']==acquisition['groups'] and not acquisition['errors']
    # Use only new citing works outside the entire original cache. Retain old
    # outgoing edges exactly, even if the live API has updated them since then.
    with ro(source/'refs/work_references.sqlite') as db:
        known_cache=pd.read_sql_query(
            "SELECT CAST(substr(work_id,2) AS INTEGER) AS id FROM work_ref_status WHERE status != 'missing_in_openalex'",db)
        aliases=dict((int(a[1:]),int(b[1:])) for a,b in db.execute('SELECT requested_id,canonical_id FROM work_alias'))
    fresh=pd.read_csv(args.edges)
    fresh['citing']=fresh.citing.map(aliases).fillna(fresh.citing).astype('int64')
    fresh=fresh.drop_duplicates(['citing','cited'])
    queried=core if args.all else pd.Index(selected.id.unique())
    assert fresh.cited.isin(queried).all()
    year_diff=fresh.citing_year - fresh.cited.map(years)
    assert year_diff.between(1,3).all()
    external=fresh.loc[~fresh.citing.isin(known_cache.id) & ~fresh.citing.isin(core)]
    incident=external.groupby('citing').cited.nunique()
    assert len(external)>0
    joint,joint_core,joint_info=graph(core,pd.concat([ref,added,external[['citing','cited']]],ignore_index=True))
    assert joint_core[common].all()
    with threadpool_limits(limits=4):
        enriched,_=pivot_mds(joint,common,150,64)
        fixed,_=pivot_mds(joint,common,150,64,fixed_pivots=anchors)
    if args.save_coordinates:
        np.savez_compressed(args.save_coordinates,ids=common_ids.to_numpy(),
                            with_external=enriched,local=local,fixed_pivots=fixed)
    panel, matched=calculate_panel(selected, common_ids, {
        'shared_references':(space_ids,vectors),'citation_network_external':(common_ids,enriched),
        'citation_network_local':(common_ids,local)})
    assert fingerprint(matched)==fingerprint(selected)
    for name,data in comparison_stats(panel,methods=METHODS,last_post_window=14 if args.all else 9).items():
        data.to_csv(args.out/(name+'.csv'),index=False)
    alternative, same=calculate_panel(selected, common_ids, {
        'shared_references':(space_ids,vectors),'citation_network_external':(common_ids,fixed),
        'citation_network_local':(common_ids,local)})
    assert fingerprint(same)==fingerprint(selected)
    for name,data in comparison_stats(alternative,methods=METHODS,last_post_window=14 if args.all else 9).items():
        data.to_csv(args.out/('fixed_pivots_'+name+'.csv'),index=False)
    field_coverage = {}
    for field,g in matched.groupby('subject'):
        targets=g.id.unique()
        sub=external.loc[external.cited.isin(targets)]
        field_coverage[field] = {'target_works':len(targets),'target_works_cited':sub.cited.nunique(),
                                 'new_external_links':len(sub)}
    report=dict(
        mode='full' if args.all else 'stratified pilot', original_frozen_sha256=fingerprint(frozen),
        matched_sample_sha256=fingerprint(matched), windows=len(panel), faculty=panel.professor_uid.nunique(),
        publications=matched.id.nunique(), links=len(matched),
        graph_focal_works=len(core), field_coverage=field_coverage, external_links=len(external),
        external_citing_works=external.citing.nunique(), focal_works_cited_externally=external.cited.nunique(),
        external_citing_multiple_targets=int((incident>=2).sum()),
        edges_increase_after_pruning=joint_info['pruned_edges']-local_info['pruned_edges'],
        local_graph=local_info | validate_distances(adj,common,local,150),
        new_graph=joint_info | validate_distances(joint,common,enriched,150), acquisition=acquisition)
    (args.out/'diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
