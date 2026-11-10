"""Estimate reference variation across the supported publication careers in the full roster."""
import argparse
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from analyze import FIELDS, METRICS, BOOTSTRAPS


def measure(rows, key, moments, centers, center_squared):
    out = []
    for label, group in rows.groupby(key, sort=True, observed=True):
        if len(group) < 2:
            continue
        ix = group.ix.to_numpy()
        x = centers[ix]
        mu = x.mean(axis=0)
        w = float((moments[ix] - center_squared[ix]).mean())
        b = float(np.square(x - mu).sum(axis=1).mean())
        t = float(moments[ix].mean() - mu @ mu)
        assert abs(w + b - t) < 1e-9
        if t > 1e-10:
            if len(key) == 1:
                label = (label,)
            out.append((*label, len(group), w, b, b / t))
    return pd.DataFrame(out, columns=key + ['papers', *METRICS])


def bootstrap(values, rng):
    n = len(values)
    if n < 2:
        return None
    means = np.concatenate([values[rng.integers(n, size=(100, n))].mean(axis=1)
                            for _ in range(BOOTSTRAPS // 100)])
    return np.quantile(means, [.025, .975], axis=0)


def point_interval(values, rng):
    limits = bootstrap(values, rng)
    return {k: [float(values[:, i].mean()), *map(float, limits[:, i])]
            for i, k in enumerate(METRICS)} if limits is not None else {}


def corrected(metrics, n):
    result = metrics.copy()
    result[:, 1] *= n / (n - 1)
    result[:, 2] = result[:, 1] / (result[:, 0] + result[:, 1])
    return result


def summarize(complete, refs, stages, years, modes, event, activity, full, rng):
    stage_summary = {}
    field_summary = {}
    for field in FIELDS:
        select = (lambda f: f) if field == 'all' else (lambda f: f[f.field == field])
        ref_group = select(refs)
        whole = select(full)
        field_summary[field] = {
            'roster': len(select(complete)), 'refs': len(ref_group), 'pooled': len(whole),
            'first_without_anchor': int((~whole.uid.isin(years['database'].uid)).sum()),
        }
        for tag in ('0–4', '5–9', '10–19', '20+'):
            subset = select(stages[stages.stage.eq(tag)])
            stage_summary.setdefault(field, {})[tag] = {
                'n': len(subset), 'metrics': point_interval(subset[METRICS].to_numpy(), rng),
            }

    event_summary = {}
    for mode in modes:
        dates = years[mode]
        data = event[mode]
        counts = activity[mode]
        paired = data['pre'].merge(data['inclusive'], on=['uid', 'field'], suffixes=('_pre', '_include'),
                                   validate='one_to_one')
        paired = paired.merge(data['later'], on=['uid', 'field'], validate='one_to_one')
        observed = set(data['later'].uid)
        for field in FIELDS:
            select = (lambda f: f) if field == 'all' else (lambda f: f[f.field == field])
            row = select(paired)
            active = select(counts)
            later = row[METRICS].to_numpy()
            included = row[[k+'_include' for k in METRICS]].to_numpy()
            pre = row[[k+'_pre' for k in METRICS]].to_numpy()
            event_summary.setdefault(mode, {})[field] = {
                'anchored': len(select(dates)), 'n': len(row), 'usable_later': int(active.uid.isin(observed).sum()),
                'post_long_no_link': int((active.post_late.eq(0) & active.anchor.le(2020)).sum()),
                'post_followup_short': int(active.anchor.gt(2020).sum()),
                'early_ref_usable': int(row.uid.isin(data['early'].uid).sum()),
                'early_included': int((row.papers_include > row.papers).sum()),
                'paper_counts_median': [float(row[c].median()) for c in ('papers_pre', 'papers_include', 'papers')],
                'count_corrected': point_interval(
                    corrected(later, row.papers.to_numpy()) -
                    corrected(included, row.papers_include.to_numpy()), rng),
                'pre': dict(zip(METRICS, pre.mean(axis=0))),
                'include': dict(zip(METRICS, included.mean(axis=0))),
                'later': dict(zip(METRICS, later.mean(axis=0))),
                'delta': point_interval(later - included, rng),
                'from_pre': point_interval(later - pre, rng),
            }
    return field_summary, stage_summary, event_summary


def main(argv=None, collect=None, collect_roster=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cohort', 'appointment', 'out'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args(argv)
    with sqlite3.connect(args.cohort.resolve().as_uri() + '?mode=ro&immutable=1', uri=True) as db:
        roster = pd.read_sql_query('''SELECT professor_uid uid, subject field, n_supported_refs,
                                      roster_year, first_ranked_roster_year, reconstructed_year,
                                      conflicting_years, shared_author_id
                                      FROM roster_coverage ORDER BY uid''', db)
        linked = pd.read_sql_query('''SELECT f.professor_uid uid, f.work_id work, f.year, r.subject field, f.topic_id, f.work_type
                FROM faculty_publication f JOIN roster_coverage r USING (professor_uid)
                WHERE f.review_only=0 AND f.year<=2025 ORDER BY uid, f.year, work''', db)
        raw_refs = pd.read_sql_query('''SELECT CAST(substr(o.work_id,2) AS INTEGER) AS citing,
                 CAST(substr(o.reference_id,2) AS INTEGER) AS cited
                 FROM outgoing_reference o WHERE o.work_id<>o.reference_id AND EXISTS (
                     SELECT 1 FROM faculty_publication f WHERE f.work_id=o.work_id
                     AND f.review_only=0 AND f.year<=2025)
                 ORDER BY o.work_id, o.reference_id''', db)
    refs = roster[roster.n_supported_refs > 0]
    assert len(roster) == 3937 and len(refs) == 3446 and set(linked.field).issubset(FIELDS)
    assert raw_refs.gt(0).all().all()
    ids = np.sort(raw_refs.citing.unique())
    n_refs = raw_refs.groupby('citing').size().reindex(ids).to_numpy()
    df = raw_refs.groupby('cited').size()
    kept_df = df[df >= 2]
    raw_refs = raw_refs[raw_refs.cited.isin(kept_df.index)]
    row = pd.Index(ids).get_indexer(raw_refs.citing)
    col = kept_df.index.get_indexer(raw_refs.cited)
    del raw_refs
    incidence = sparse.csr_matrix((np.ones(len(row), dtype=np.float32), (row, col)),
                                  shape=(len(ids), len(kept_df)))
    assert incidence.nnz == len(row)
    counts = np.diff(incidence.indptr)
    good = counts >= (1 if collect_roster is not None else 2)
    idf = (1 + np.log((1 + len(ids)) / (1 + kept_df.to_numpy()))).astype(np.float32)
    with threadpool_limits(limits=4):
        model = TruncatedSVD(n_components=32, n_iter=4, random_state=1109)
        model.fit(normalize(incidence.multiply(idf).tocsr()))
    coords = normalize(model.components_.T).astype(np.float64)
    divisor = np.maximum(counts, 1)
    centers = (incidence @ coords) / divisor[:, None]
    moments = (incidence @ np.einsum('ij,ij->i', coords, coords)) / divisor
    center_squared = np.einsum('ij,ij->i', centers, centers)
    ix = pd.Series(np.arange(len(ids)), index=ids)
    linked['ix'] = linked.work.str[1:].astype(int).map(ix)
    safe_ix = linked.ix.fillna(0).astype(int)
    papers = linked[linked.ix.notna() & good[safe_ix]].copy()
    papers['ix'] = papers.ix.astype(int)
    papers['n_positioned_refs'] = counts[papers.ix.to_numpy()]
    assert set(papers.uid).issubset(refs.uid)
    assert n_refs.shape == counts.shape and np.all(counts <= n_refs)
    start = linked.groupby('uid').year.min()
    assert (papers.year >= papers.uid.map(start)).all()
    papers['stage'] = pd.cut(papers.year - papers.uid.map(start), [-1, 4, 9, 19, np.inf],
                             labels=['0–4', '5–9', '10–19', '20+'])
    full = measure(papers, ['uid', 'field'], moments, centers, center_squared)
    stages = measure(papers, ['uid', 'field', 'stage'], moments, centers, center_squared)
    print('Represented full careers:', len(full), flush=True)

    stored = pd.DataFrame(json.loads(args.appointment.read_text()))
    modes = ['official', 'baseline', 'database', 'proxy']
    cols = ['uid', 'field', 'anchor', 'basis']
    audited = stored.loc[~stored.shared_author_id.astype(bool)].rename(columns={'year': 'anchor'})
    dates = {mode: audited.loc[audited['mode'].eq(mode), cols].copy()
             for mode in ('official', 'baseline')}
    unassigned = refs.loc[~refs.uid.isin(dates['baseline'].uid)
                          & ~refs.shared_author_id.astype(bool)
                          & ~refs.conflicting_years.astype(bool)].copy()
    def date_ok(year):
        return year.notna() & year.le(2025) & (
            unassigned.first_ranked_roster_year.isna() | year.le(unassigned.first_ranked_roster_year))
    first_role = date_ok(unassigned.reconstructed_year)
    unassigned['anchor'] = unassigned.reconstructed_year.where(
        first_role, unassigned.roster_year.where(date_ok(unassigned.roster_year)))
    unassigned['basis'] = np.where(first_role, 'first_role_in_database', 'roster_date_in_database')
    dates['database'] = pd.concat([dates['baseline'], unassigned.dropna(subset=['anchor'])[cols]],
                                  ignore_index=True).sort_values('uid')
    proxy_only = audited.loc[audited['mode'].eq('proxy') & audited.basis.eq('first_affiliated_paper')
                             & ~audited.uid.isin(dates['database'].uid), cols]
    dates['proxy'] = pd.concat([dates['database'], proxy_only], ignore_index=True).sort_values('uid')
    if collect_roster is not None:
        collect_roster(args.cohort, roster, linked, papers, centers, dates['database'])
        return

    event, activity = {}, {}
    for mode in modes:
        a = dates[mode].copy()
        assert a.uid.is_unique and a.anchor.notna().all()
        a['anchor'] = a.anchor.astype(int)
        pool = papers.merge(a[['uid', 'anchor']], on='uid', validate='many_to_one')
        pool['event_year'] = pool.year - pool.anchor
        grouped = {'pre': pool.event_year < 0, 'inclusive': pool.event_year >= 1,
                   'later': pool.event_year >= 3, 'early': pool.event_year.between(1, 2)}
        event[mode] = {name: measure(pool[flag], ['uid', 'field'], moments, centers, center_squared)
                       for name, flag in grouped.items()}
        if mode == 'database' and collect is not None:
            collect(pool, centers, moments, event[mode])
        linked_rel = linked.merge(a[['uid', 'anchor']], on='uid', validate='many_to_one')
        n_late = linked_rel.loc[linked_rel.year - linked_rel.anchor >= 3].groupby('uid').size()
        activity[mode] = a.assign(post_late=a.uid.map(n_late).fillna(0).astype(int))
    rng = np.random.default_rng(20261111)
    f, s, e = summarize(roster, refs, stages, dates, modes, event, activity, full, rng)
    args.out.write_text(json.dumps({'fields': f, 'stages': s, 'event': e,
                                    'retained_works': int(good.sum()), 'references': len(kept_df),
                                    'bootstrap_reps': BOOTSTRAPS,
                                    'basis_counts': dates['database'].basis.value_counts().to_dict(),
                                    'proxy_extra': len(proxy_only)}, indent=2, allow_nan=False)+'\n')
    for mode in modes:
        row = e[mode]['all']
        print(mode, row['anchored'], row['n'], row['early_ref_usable'], row['post_long_no_link'], row['delta'])
    for field in FIELDS:
        print(field, f[field], {k: v['n'] for k, v in s[field].items()}, e['database'][field]['n'])


if __name__ == '__main__':
    main()
