"""Decompose the KOAD annual-parallel contrast and audit bibliographic coverage."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from koad_sensitivity import FIELDS, SEED, common_results
from motion import bootstrap
from open_followup import annual_multi, DIMENSIONS


def connect(path):
    return closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro&immutable=1', uri=True))


def work_links(path):
    with connect(path) as db:
        frame = pd.read_sql_query(
            'SELECT uid,work,year,field FROM publication_metadata ORDER BY uid,year,work', db)
    return frame.set_index(['uid','work'], verify_integrity=True)


def coverage_and_audit(cohort):
    with connect(cohort) as db:
        roster = pd.read_sql_query(
            'SELECT professor_uid AS uid,subject AS field,n_supported,n_supported_refs,n_linked,'
            'openalex_id_assigned FROM roster_coverage ORDER BY professor_uid', db).set_index('uid')
        removed = pd.read_sql_query(
            'SELECT professor_uid AS uid,work_id AS work FROM namesake_filter_audit', db)
        refs = db.execute("""
            SELECT c.state,count(*),count(DISTINCT f.professor_uid)
            FROM faculty_publication f JOIN roster_coverage r USING(professor_uid)
            JOIN reference_coverage c USING(work_id)
            WHERE r.n_supported>0 AND r.n_supported_refs=0 AND f.review_only=0
            GROUP BY c.state
        """).fetchall()
    table = {}
    for field in FIELDS:
        r = roster if field=='all' else roster[roster.field==field]
        unlinked = r[r.n_supported.eq(0)]
        table[field] = {
            'roster': len(r),
            'with_refs': int(r.n_supported_refs.gt(0).sum()),
            'no_supported_work': len(unlinked),
            'no_author_id': int(unlinked.openalex_id_assigned.eq(0).sum()),
            'id_no_candidate': int((unlinked.openalex_id_assigned.eq(1)&unlinked.n_linked.eq(0)).sum()),
            'id_review_only': int((unlinked.openalex_id_assigned.eq(1)&unlinked.n_linked.gt(0)).sum()),
            'supported_no_refs': int((r.n_supported.gt(0)&r.n_supported_refs.eq(0)).sum()),
        }
        row = table[field]
        if row['with_refs']+row['no_supported_work']+row['supported_no_refs'] != len(r):
            raise ValueError('Coverage groups do not partition the faculty roster')
    index = pd.MultiIndex.from_frame(removed)
    if not index.is_unique or not roster.index.is_unique:
        raise ValueError('Duplicate namesake audit pair or faculty row')
    return table, dict((s, {'links': int(n), 'faculty': int(m)}) for s,n,m in refs), index


def read_axis(root, dimension, full_root=None):
    folder = root / f'd{dimension:04}'
    with np.load(folder/'paper_axis.npz', allow_pickle=False) as f:
        works, labels = pd.Index(f['work']), f['cluster']
    if full_root is None:
        coords = np.load(folder/'paper_vectors.npy', mmap_mode='r', allow_pickle=False)
    else:
        full_works, full_labels, full_vectors = read_axis(full_root, dimension)
        loc = full_works.get_indexer(works)
        if (loc < 0).any() or not np.array_equal(full_labels[loc], labels):
            raise ValueError('The full tensor archive cannot reconstruct the GCC axes')
        coords = full_vectors[loc]
    if not works.is_unique or coords.shape != (len(works), dimension) or len(labels) != len(works):
        raise ValueError('Paper axis and tensor disagree')
    return works, labels, coords


def rates(papers, events, labels, vectors, column):
    """Calculate the same two annual parallel shares as the appointment pipeline."""
    counts = ['pre_multi','post_multi']
    out = pd.DataFrame(np.nan, index=events.index, columns=counts)
    for uid, group in papers.groupby(level='uid', sort=False):
        year = group.year.to_numpy()
        pos = group[column].to_numpy(dtype=int)
        topics, x = labels[pos], vectors[pos]
        event = events.at[uid]
        for side, mask in zip(counts, (year<event,year>event), strict=True):
            _,value = annual_multi(year[mask],topics[mask],x[mask])
            out.at[uid,side] = value
    return out


def parallel(frame):
    return frame.post_multi - frame.pre_multi


def evaluate(versions, pairs, audit, old_root, new_root, old_full):
    dim_results = {}
    membership = {}
    for d, (original, refitted) in versions.items():
        a, c = original.loc[pairs], refitted.loc[pairs]
        old_axis, old_labels, old_vectors = read_axis(old_root,d,old_full)
        new_axis, new_labels, new_vectors = read_axis(new_root,d)
        work = audit.index.get_level_values('work')
        frame = audit.assign(old=old_axis.get_indexer(work), new=new_axis.get_indexer(work))
        old = frame[frame.old.ge(0)]
        retained = frame[~frame['removed']]
        fixed = retained[retained.old.ge(0)]
        refit = retained[retained.new.ge(0)]
        shared = fixed[fixed.new.ge(0)]
        old_only = fixed.index.difference(shared.index)
        new_only = refit.index.difference(shared.index)
        removed_here = old[old.removed]
        events = a['event']
        b = a[['pre_multi','post_multi']].copy()
        touched = pd.Index(removed_here.index.get_level_values('uid').unique())
        if len(touched):
            b.loc[touched] = rates(fixed.loc[fixed.index.get_level_values('uid').isin(touched)],
                                   events.loc[touched],old_labels,old_vectors,'old')
        if b.isna().any().any():
            raise ValueError('Filtering removed an eligible side in the fixed common faculty sample')
        c_rates = c[['pre_multi','post_multi']].copy()
        old_shared, new_shared = b.copy(), c_rates.copy()
        for changed, target, labels, coords, col in (
            (old_only,old_shared,old_labels,old_vectors,'old'),
            (new_only,new_shared,new_labels,new_vectors,'new'),
        ):
            touched = pd.Index(changed.get_level_values('uid').unique())
            if len(touched):
                target.loc[touched] = rates(shared.loc[shared.index.get_level_values('uid').isin(touched)],
                                            events.loc[touched],labels,coords,col)
        # Work IDs align the cluster labels independently of their numeric names.
        new_labels_on_old = np.full(len(old_axis),-1,dtype=new_labels.dtype)
        same = new_axis.get_indexer(old_axis)
        new_labels_on_old[same>=0] = new_labels[same[same>=0]]
        new_groups = rates(shared, events, new_labels_on_old, old_vectors, 'old')
        if any(r.isna().any().any() for r in (old_shared,new_shared,new_groups)):
            raise ValueError('The same-paper check lacks annual observations on one side')
        decomp = pd.DataFrame({
            'original':parallel(a), 'filter_only':parallel(b), 'refit':parallel(c_rates),
            'refit_pre':c_rates.pre_multi-b.pre_multi,
            'refit_post':c_rates.post_multi-b.post_multi,
            'filter_difference':parallel(b)-parallel(a),
            'refit_difference':parallel(c_rates)-parallel(b),
            'total_difference':parallel(c_rates)-parallel(a),
            'eligibility_difference':parallel(old_shared)-parallel(b)+parallel(c_rates)-parallel(new_shared),
            'cluster_difference':parallel(new_groups)-parallel(old_shared),
            'coordinate_difference':parallel(new_shared)-parallel(new_groups),
        })
        np.testing.assert_allclose(decomp.refit_difference,
            decomp.eligibility_difference+decomp.cluster_difference+decomp.coordinate_difference, atol=1e-14)
        np.testing.assert_allclose(decomp.total_difference,
            decomp.filter_difference+decomp.refit_difference, atol=1e-14)
        changed = decomp[['eligibility_difference','cluster_difference','coordinate_difference']].abs().gt(1e-12)
        membership[d] = {'old_only_links': len(old_only), 'new_only_links': len(new_only),
                         'matched_links': len(shared), 'removed_positioned_links': len(removed_here),
                         'changed_faculty': {k: int(v) for k,v in changed.sum().items()}}
        dim_results[d] = decomp
        print(d, membership[d], flush=True)
    return dim_results, membership


def summarize(before_db, after_db, cohort, old_root, new_root, before_rates, after_rates, old_full=None):
    versions, fields, _ = common_results(before_rates, after_rates)
    pairs = fields.index
    coverage, reference_states, removed = coverage_and_audit(cohort)
    before_links, after_links = map(work_links, (before_db,after_db))
    kept = before_links.index.difference(removed)
    if not before_links.loc[kept].sort_index().equals(after_links.sort_index()):
        raise ValueError('Publication rows must differ only by the audited faculty-work links')
    if len(before_links)-len(kept) != len(removed):
        raise ValueError('An audited drop did not match the original publication rows')
    audit = before_links[before_links.index.get_level_values('uid').isin(pairs)].copy()
    audit['removed'] = audit.index.isin(removed)
    estimates, membership = evaluate(versions,pairs,audit,old_root,new_root,old_full)

    rng = np.random.default_rng(SEED)
    summary = {}
    for field in FIELDS:
        ids = pairs if field=='all' else pairs[fields==field]
        raw = pd.concat({d:r.loc[ids] for d,r in estimates.items()}, axis=1)
        ci = bootstrap(raw.to_numpy(), rng)
        result = {}
        for (d,m),stat in zip(raw.columns,ci,strict=True):
            result.setdefault(d,{})[m] = stat.tolist()
        summary[field] = {'n':len(ids), 'dimensions':result}
    return {'fields':summary, 'membership':membership, 'coverage':coverage,
            'reference_states':reference_states, 'removed_links':len(removed),
            'affected_rows':int(removed.get_level_values('uid').nunique()),
            'removed_links_in_common':int(audit.removed.sum())}


def render(summary, target, font):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, fontManager
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator

    fontManager.addfont(font)
    plt.rcParams.update({'font.family':[FontProperties(fname=font).get_name(),'DejaVu Sans'],
                         'text.color':'#172d47'})
    rows = summary['fields']
    fig, axes = plt.subplots(1,5,figsize=(13,4.3),sharey=True,sharex=True)
    fig.subplots_adjust(left=.07,right=.97,top=.67,bottom=.22,wspace=.18)
    fig.text(.045,.94,'KOAD 정정 후 연간 병행의 변화는 어디서 생겼나', fontsize=22, weight='bold')
    fig.text(.045,.86,f"공통 교원 {rows['all']['n']:,}행 · 임용 후−전 병행률 변화의 단계별 차이",fontsize=12,color='#425a76')
    handles = [Line2D([0],[0],marker='o',markerfacecolor='white',markeredgecolor='#157c81',
                      color='none',label='오연결만 제외'),
               Line2D([0],[0],marker='o',color='#275b97',label='이후 임베딩·군집 재적합')]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.039,.82),ncol=2,frameon=False,fontsize=11)
    for ax,(field,label) in zip(axes,FIELDS.items(),strict=True):
        row = rows[field]
        ax.set_title(f"{label}  ·  {row['n']:,}행",fontsize=12,weight='bold',pad=9)
        ax.axvline(0,color='#8396ab',lw=.9,ls=(0,(2,3)),zorder=1)
        ax.grid(axis='x',color='#e8eff6')
        ax.set_ylim(len(DIMENSIONS)-.45,-.55)
        ax.set_yticks(range(len(DIMENSIONS)),[str(d) for d in DIMENSIONS])
        ax.tick_params(axis='y',length=0,labelsize=9,colors='#435c79')
        ax.tick_params(axis='x',length=2,labelsize=9,colors='#435c79')
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.set_xlabel('차이 (%p)',fontsize=9)
        for spine in ('top','left','right'):
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color('#dce5ef')
        for k,d in enumerate(DIMENSIONS):
            for metric,dy,color,fill in (
                ('filter_difference',-.13,'#157c81','white'),('refit_difference',.13,'#275b97','#275b97')
            ):
                estimate,lo,hi = (100*v for v in row['dimensions'][d][metric])
                ax.hlines(k+dy,lo,hi,color=color,lw=1.3,zorder=2)
                ax.plot(estimate,k+dy,'o',ms=4.1,color=color,mfc=fill,zorder=3)
    extent = 110*max(abs(v) for row in rows.values() for est in row['dimensions'].values()
                     for m in ('filter_difference','refit_difference') for v in est[m])
    axes[0].set_xlim(-extent,extent)
    axes[0].set_ylabel('임베딩 차원',fontsize=9)
    fig.text(.045,.065,'점은 같은 교원에서 계산한 차이의 평균, 선은 교원 단위 95% 짝재표집 구간',
             fontsize=10,color='#425a76')
    fig.savefig(target,dpi=170,facecolor='white')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('before_portfolio','after_portfolio','corrected_cohort','before_tensors','after_tensors',
                 'before_rates','after_rates','output'):
        p.add_argument(name,type=Path)
    p.add_argument('--before-full-tensors',type=Path,help='Reconstruct the original GCC vectors from the full tensor archive')
    p.add_argument('--figure',type=Path)
    p.add_argument('--font',type=Path,default=Path('/Library/Fonts/NanumHumanRegular.ttf'))
    a = p.parse_args()
    result = summarize(a.before_portfolio,a.after_portfolio,a.corrected_cohort,a.before_tensors,
                       a.after_tensors,a.before_rates,a.after_rates,a.before_full_tensors)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if a.figure:
        render(result,a.figure,a.font)


if __name__=='__main__':
    main()
