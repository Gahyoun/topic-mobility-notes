"""Decompose portfolio transport residuals and verify citation-linked stream branches."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from appointment_dag import CUTOFF, K, read_data, registry_from_stream
from flow import FIELDS, field_local, mean_q
from motion import bootstrap
from native_dag import DIMENSIONS
from native_flow import subtract

METRICS = ('excess', 'within', 'between', 'shape')


def scatter_parts(x, labels):
    """Return a covariance factor and the traces of its within/between components."""
    x = np.asarray(x, dtype=np.float64)
    _, idx, n = np.unique(labels, return_inverse=True, return_counts=True)
    centers = np.zeros((len(n), x.shape[1]))
    np.add.at(centers, idx, x)
    centers /= n[:, None]
    grand = x.mean(axis=0)
    within = np.square(x - centers[idx]).sum() / (len(x) - 1)
    between = n @ np.square(centers - grand).sum(axis=1) / (len(x) - 1)
    return (x - grand) / np.sqrt(len(x) - 1), float(within), float(between)


def shape_gap(a, b, ta, tb):
    if min(ta, tb) <= 1e-12:
        return np.nan
    a, b = a / np.sqrt(ta), b / np.sqrt(tb)
    # Pick the smaller matrix axis for the Frobenius inner product.
    if max(len(a), len(b)) > a.shape[1]:
        return float(np.linalg.norm(a.T @ a - b.T @ b))
    squared = (np.square(a @ a.T).sum() + np.square(b @ b.T).sum()
               - 2 * np.square(a @ b.T).sum())
    return float(np.sqrt(max(squared, 0.0)))


def residual(advected, prior_groups, observed, later_groups):
    a, wa, ba = scatter_parts(advected, prior_groups)
    b, wb, bb = scatter_parts(observed, later_groups)
    denominator = wa + ba + wb + bb
    if denominator <= 1e-12:
        return None
    within, between = (wb - wa) / denominator, (bb - ba) / denominator
    excess = within + between
    qa, qb = mean_q(advected.astype(float)), mean_q(observed.astype(float))
    np.testing.assert_allclose(excess, (qb - qa) / (qb + qa), atol=1e-9)
    return excess, within, between, shape_gap(a, b, wa + ba, wb + bb)


def supported_groups(groups):
    n = np.bincount(groups, minlength=K)
    return set(np.flatnonzero((n >= 2) & (n >= .15 * len(groups))))


def separated(cloud, groups, a, b):
    x, y = cloud[groups == a], cloud[groups == b]
    m, n = x.mean(0), y.mean(0)
    scale = np.square(x - m).sum() / len(x) + np.square(y - n).sum() / len(y)
    return np.square(m - n).sum() >= max(scale, 1e-12)


def branch_counts(year, groups, cloud, work, links, first_year):
    """Count eligible triples, persistent separated splits and direct ancestry evidence."""
    start_years = range(first_year, CUTOFF - 3, 5)
    blocks = []
    for start in start_years:
        rows = np.flatnonzero((year >= start) & (year < start + 5))
        blocks.append((start, rows, supported_groups(groups[rows])))
    total = np.zeros(3, dtype=int)
    for (_, a, ga), (middle, b, gb), (_, c, gc) in zip(blocks, blocks[1:], blocks[2:]):
        if min(len(a), len(b), len(c)) < 2:
            continue
        total[0] += 1
        persistent = gb & gc
        parents = ga & persistent
        new = persistent - set(groups[year < middle])
        if not parents or not new:
            continue
        split = False
        linked = False
        for parent in parents:
            anchors = set(work[a][groups[a] == parent])
            for child in new:
                if not (separated(cloud[b], groups[b], parent, child)
                        and separated(cloud[c], groups[c], parent, child)):
                    continue
                split = True
                linked |= any(anchors & links.get(w, set()) for w in work[b][groups[b] == child])
        total[1] += split
        total[2] += linked
    return tuple(total)


def measure(roster, publications, links, folder, dimension):
    with np.load(folder / 'paper_axis.npz', allow_pickle=False) as paper:
        axis, labels = paper['work'], paper['cluster']
    x = np.load(folder / 'paper_vectors.npy', mmap_mode='r', allow_pickle=False)
    assert x.shape == (len(axis), dimension)
    origin, peers = registry_from_stream(folder / 'annual_stream.npz', dimension)
    index = pd.Series(np.arange(len(axis)), index=axis)
    positioned = publications.assign(ix=publications.work.map(index)).dropna(subset=['ix'])
    positioned['ix'] = positioned.ix.astype(int)
    by_field = {f: g.drop_duplicates('work').set_index('work')
                for f, g in positioned.groupby('field')}
    by_uid = {uid: g for uid, g in positioned.groupby('uid', sort=False)}
    first_years = publications.groupby('uid').year.min()
    records = []
    for row in roster.itertuples(index=False):
        record = {'uid': row.uid, 'field': row.field}
        g = by_uid.get(row.uid, positioned.iloc[:0])
        ix = g.ix.to_numpy(dtype=int)
        y, work, lab, cloud = g.year.to_numpy(), g.work.to_numpy(), labels[ix], x[ix]
        first = int(first_years[row.uid])
        direct = {w: links.get((row.uid, w), set()) for w in work}
        triple, split, branch = branch_counts(y, lab, cloud, work, direct, first)
        record.update(triples=triple, persistent_splits=split, linked_branches=branch)
        if pd.isna(row.appointment_year):
            records.append(record)
            continue
        event = int(row.appointment_year)
        before, after = y < event, y > event
        if min(before.sum(), after.sum()) >= 2:
            own = by_field[row.field].loc[g.work.unique()]
            own_ix = own.ix.to_numpy(dtype=int)
            args = (own.year.to_numpy(), labels[own_ix], x[own_ix])
            peer = peers[row.field]
            a = subtract(peer, *args, max(origin, first), event - 1, origin, K)
            b = subtract(peer, *args, event + 1, CUTOFF, origin, K)
            mass = np.bincount(lab[before], minlength=K) / before.sum()
            _, _, support, shifts = field_local(a, b, mass, K)
            record['supported_mass'] = support
            if shifts is not None:
                values = residual(cloud[before] + shifts[lab[before]], lab[before],
                                  cloud[after], lab[after])
                if values is not None:
                    record.update(zip(METRICS, values, strict=True))
        records.append(record)
    return pd.DataFrame.from_records(records)


def summarize(data):
    rng = np.random.default_rng(621112)
    result = {}
    for f in FIELDS:
        sub = data if f == 'all' else data[data.field == f]
        estimates = {}
        for columns in (METRICS[:3], METRICS[3:]):
            observations = sub.loc[:, list(columns)].dropna()
            if len(observations):
                bounds = bootstrap(observations.to_numpy(), rng)
                estimates.update({c: {'n': len(observations), 'estimate': v.tolist()}
                                  for c, v in zip(columns, bounds, strict=True)})
        counts = {c: int(sub[c].sum()) for c in ('triples', 'persistent_splits', 'linked_branches')}
        counts['faculty_with_triples'] = int(sub.triples.gt(0).sum())
        counts['faculty_with_linked_branches'] = int(sub.linked_branches.gt(0).sum())
        assert counts['linked_branches'] <= counts['persistent_splits'] <= counts['triples']
        result[f] = {'estimates': estimates, 'branch': counts}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('portfolio', 'tensors', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    args.output.mkdir(mode=0o700)
    roster, papers, links = read_data(args.portfolio)
    result = {}
    for d in DIMENSIONS:
        data = measure(roster, papers, links, args.tensors / f'd{d:04}', d)
        data.to_csv(args.output / f'faculty-d{d:04}.csv', index=False)
        result[d] = summarize(data)
        print(d, result[d]['all'], flush=True)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
