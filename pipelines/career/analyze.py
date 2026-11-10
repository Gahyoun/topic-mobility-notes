"""Re-bin the frozen 908-faculty reference pilot without changing the embedding."""
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

KEY = ['uid', 'field', 'win']
METRICS = ['W', 'B', 'B_share']
FIELDS = ['all', 'mathematics', 'physics', 'chemistry', 'biology']
BOOTSTRAPS = 10000
ARCHIVE_YEAR = 2022


def cells(rows, paper_means, moments, means_squared):
    records = []
    for (uid, field, win), papers in rows.groupby(KEY):
        if len(papers) < 2:
            continue
        index = papers.ix.to_numpy()
        means = paper_means[index]
        center = means.mean(0)
        within = float((moments[index] - means_squared[index]).mean())
        between = float(np.square(means - center).sum(axis=1).mean())
        total = float(moments[index].mean() - center @ center)
        assert abs(within + between - total) < 1e-10 and within >= -1e-9 and between >= 0
        if total > 1e-10:
            records.append((uid, field, win, within, between, total, between / total))
    return pd.DataFrame(records, columns=KEY + ['W', 'B', 'T', 'B_share'])


def paired_summary(before, after, early, late_counts, rng):
    delta = (after - before).to_numpy()
    mean = delta.mean(axis=0)
    indices = rng.integers(len(delta), size=(BOOTSTRAPS, len(delta)))
    interval = np.quantile(delta[indices].mean(axis=1), [.025, .975], axis=0)
    np.testing.assert_allclose(after.to_numpy() - before.to_numpy(),
                               ((after - early) / (late_counts + 1).values[:, None]).fillna(0),
                               atol=1e-12)
    return {
        'n': len(delta), 'early': int(early.W.notna().sum()),
        'include': before.mean().to_dict(), 'later': after.mean().to_dict(),
        'delta': {k: [float(mean[j]), float(interval[0, j]), float(interval[1, j])]
                  for j, k in enumerate(METRICS)},
    }


def main(argv=None, collect=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('db', 'appointments', 'previous', 'out'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args(argv)
    with sqlite3.connect(args.db.resolve().as_uri() + '?mode=ro&immutable=1', uri=True) as db:
        rows = pd.read_sql_query(
            '''SELECT professor_uid AS uid, subject AS field, window_index AS win,
                      work_id, publication_year AS year
               FROM matched_panel WHERE window_index BETWEEN -5 AND 4
               ORDER BY professor_uid, window_index, work_id''', db)
        edges = pd.read_sql_query('SELECT citing_id, cited_id FROM outgoing_reference', db)
    baseline = rows[['uid', 'win']].drop_duplicates()
    bins = baseline.groupby(['uid', baseline.win.ge(0)]).size().unstack(fill_value=0)
    eligible = bins.index[(bins[False] >= 2) & (bins[True] >= 2)]
    assert len(eligible) == 1219
    rows = rows[rows.uid.isin(eligible)].copy()
    anchors = pd.read_csv(args.appointments, usecols=['professor_uid', 'appointment_year'])
    years = rows.uid.map(anchors.set_index('professor_uid').appointment_year)
    assert years.notna().all() and (((rows.year - years) // 2) == rows.win).all()
    assert (years + 2 * rows.win + 1 <= ARCHIVE_YEAR).all()
    rows['relative_year'] = (rows.year - years).astype(int)
    rows['anchor'] = years.astype(int)
    ids = np.sort(rows.work_id.unique())
    edges = edges[edges.citing_id.isin(ids)]
    counts = edges.groupby('citing_id').size().reindex(ids, fill_value=0).to_numpy()
    df = edges.groupby('cited_id').size()
    supported = df[df >= 2]
    edges = edges[edges.cited_id.isin(supported.index)]
    ix = pd.Index(ids).get_indexer(edges.citing_id)
    jx = supported.index.get_indexer(edges.cited_id)
    X = sparse.csr_matrix((np.ones(len(ix), np.float32), (ix, jx)), shape=(len(ids), len(supported)))
    kept = np.diff(X.indptr)
    idf = (1 + np.log((1 + len(ids)) / (1 + supported.to_numpy()))).astype(np.float32)
    with threadpool_limits(limits=4):
        svd = TruncatedSVD(n_components=32, n_iter=4, random_state=1109).fit(
            normalize(X.multiply(idf).tocsr()))
    coords = normalize(svd.components_.T).astype(np.float64)
    retained = (kept >= 5) & (2 * kept >= counts)
    rows['ix'] = pd.Index(ids).get_indexer(rows.work_id)
    valid = rows[retained[rows.ix]].copy()
    divisor = np.maximum(kept, 1)
    paper_means = (X @ coords) / divisor[:, None]
    moments = (X @ np.einsum('ij,ij->i', coords, coords)) / divisor
    means_squared = np.einsum('ij,ij->i', paper_means, paper_means)
    old = cells(valid, paper_means, moments, means_squared)
    by = old.groupby(['uid', old.win.ge(0)]).size().unstack(fill_value=0)
    cohort = by.index[(by[False] >= 2) & (by[True] >= 2)]
    old = old[old.uid.isin(cohort)]
    assert len(cohort) == 908 and len(old) == 6357
    baseline_stats = json.loads(args.previous.read_text())
    assert baseline_stats['represented_works'] == retained.sum()
    old_means = old.groupby(['uid', 'field', old.win.ge(0)])[['W', 'B', 'T', 'B_share']].mean()
    for field in baseline_stats['results']:
        chosen = old_means if field['field'] == 'all' else old_means.xs(field['field'], level='field')
        for post, name in [(False, 'pre'), (True, 'post')]:
            estimate = chosen.xs(post, level='win')
            assert len(estimate) == field['n']
            np.testing.assert_allclose(estimate.mean(), [field[name][c] for c in estimate.columns], atol=1e-8, rtol=0)

    shifted = valid.loc[valid.uid.isin(cohort) & valid.relative_year.between(1, 8)].copy()
    shifted['win'] = 1 + 2 * ((shifted.relative_year - 1) // 2)
    # Both overlapping source bins had to be mature in the frozen panel.
    shifted = shifted[shifted.anchor + shifted.win + 2 <= ARCHIVE_YEAR]
    post = cells(shifted, paper_means, moments, means_squared)
    later = post[post.win >= 3].groupby(['uid', 'field'])[METRICS].mean()
    include = post.groupby(['uid', 'field'])[METRICS].mean().reindex(later.index)
    early = post[post.win == 1].set_index(['uid', 'field'])[METRICS].reindex(later.index)
    late_counts = post[post.win >= 3].groupby(['uid', 'field']).size().reindex(later.index)
    members = old[['uid', 'field']].drop_duplicates()
    rng = np.random.default_rng(20261110)
    output = {}
    strict = {}
    for field in FIELDS:
        take = np.ones(len(later), dtype=bool) if field == 'all' else later.index.get_level_values('field') == field
        both = take & early.W.notna().to_numpy()
        output[field] = paired_summary(include[take], later[take], early[take], late_counts[take], rng)
        output[field]['original'] = len(members) if field == 'all' else int(members.field.eq(field).sum())
        strict[field] = paired_summary(include[both], later[both], early[both], late_counts[both], rng)
    if collect is not None:
        collect(shifted, paper_means, moments, output)
    coverage = {
        'early_windows': int(post.win.eq(1).sum()),
        'later_windows': int(post.win.ge(3).sum()),
        'early_only': int(len(set(post.loc[post.win.eq(1), 'uid']) - set(later.index.get_level_values('uid')))),
    }
    args.out.write_text(json.dumps({'results': output, 'strict': strict, 'coverage': coverage,
                                    'bootstrap_reps': BOOTSTRAPS}, indent=2) + '\n')
    print('Reproduced all baseline field estimates.')
    for field in FIELDS:
        r = output[field]
        print(field, r['original'], r['n'], r['early'], 'deltas',
              [[round(x, 5) for x in r['delta'][k]] for k in METRICS])
    print(coverage)


if __name__ == '__main__':
    main()
