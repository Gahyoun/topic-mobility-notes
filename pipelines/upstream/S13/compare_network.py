"""Matched-paper comparison of OpenAlex topics, cached references, and citation graphs."""
import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

LEVELS = ['topic_id', 'subfield_id', 'field_id', 'domain_id']
KEY = ['professor_uid', 'subject', 'win']
METHODS = ['classification', 'shared_references', 'citation_network', 'graph_out_only']
SUBJECTS = ['physics', 'chemistry', 'biology', 'mathematics']
SOURCE_YEAR = 2025
LAG = 3
LAST_YEAR = SOURCE_YEAR - LAG
WINDOW = 2


def ro(path):
    return sqlite3.connect(Path(path).as_uri() + '?mode=ro&immutable=1', uri=True)


def work_number(values):
    return values.str.rsplit('/', n=1).str[-1].str.removeprefix('W').astype('int64')


def load_inputs(cfg, source, selection, extra_citations=None):
    rows = pd.read_csv(selection, dtype={key: str for key in LEVELS})
    with ro(cfg['web_db']) as db:
        professors = pd.read_sql_query(
            'SELECT professor_uid, appointment_year FROM professors WHERE appointment_year IS NOT NULL', db)
        source_years = pd.read_sql_query("""
            SELECT work_id, min(publication_year) AS earliest, max(publication_year) AS latest
            FROM work_authorship_evidence WHERE publication_year IS NOT NULL GROUP BY work_id
            """, db)
    with ro(source / 'refs/work_references.sqlite') as db:
        aliases = dict((int(a[1:]), int(b[1:])) for a, b in db.execute(
            'SELECT requested_id, canonical_id FROM work_alias'))
        if any(b in aliases for b in aliases.values()):
            raise ValueError('Resolve alias chains before running')
        counts = db.execute("""
            SELECT count(*) FROM work_references
            WHERE work_id NOT GLOB 'W[0-9]*' OR referenced_work_id NOT GLOB 'W[0-9]*'
            """).fetchone()[0]
        if counts:
            raise ValueError('Expected short OpenAlex work IDs')
        raw_edges = pd.read_sql_query("""
            SELECT CAST(substr(work_id,2) AS INTEGER) AS citing,
                   CAST(substr(referenced_work_id,2) AS INTEGER) AS cited
            FROM work_references
            """, db)
    for col in ['citing', 'cited']:
        raw_edges[col] = raw_edges[col].map(aliases).fillna(raw_edges[col]).astype('int64')
    raw_edges = raw_edges.loc[raw_edges.citing != raw_edges.cited].drop_duplicates(ignore_index=True)
    source_years['raw'] = work_number(source_years.work_id)
    source_years['id'] = source_years.raw.map(aliases).fillna(source_years.raw).astype('int64')
    # Prefer the canonical record's date when available. Use an alias date only
    # when all recorded dates for the alias group agree.
    alias_years = source_years.groupby('id').agg(earliest=('earliest', 'min'), latest=('latest', 'max'))
    known_years = alias_years.earliest.where(alias_years.earliest == alias_years.latest).dropna()
    native = source_years[source_years.raw == source_years.id].set_index('id')
    known_years = known_years.reindex(alias_years.index)
    known_years.update(native.earliest.where(native.earliest == native.latest).dropna())
    rows['raw'] = work_number(rows.work_id)
    rows['id'] = rows.raw.map(aliases).fillna(rows.raw).astype('int64')
    rows = rows.merge(professors, on='professor_uid', validate='many_to_one')
    rows['win'] = (rows.year - rows.appointment_year) // WINDOW
    # Both publication years of a 2-year window must have full citation exposure.
    mature = rows.appointment_year + WINDOW * rows.win + WINDOW - 1 <= LAST_YEAR
    rows = rows.loc[mature].dropna(subset=LEVELS)
    rows = rows.assign(canonical=rows.raw == rows.id).sort_values(
        ['canonical', 'work_id'], ascending=[False, True], kind='stable')
    rows = rows.drop_duplicates(['professor_uid', 'id'])
    with np.load(source / 'outputs/S08_space/space_b2025.npz') as saved:
        space_ids = pd.Index([int(s[1:]) for s in saved['ids']])
        space = saved['vectors']
    good_norm = np.linalg.norm(space, axis=1) > .99
    raw_edges = raw_edges.assign(citing_year=raw_edges.citing.map(known_years))
    n_cache_edges = len(raw_edges)
    if extra_citations is not None:
        extra = pd.read_csv(extra_citations, dtype={'citing_work_id': str, 'cited_work_id': str})
        for col in ('citing', 'cited'):
            ids = work_number(extra[col + '_work_id'])
            extra[col] = ids.map(aliases).fillna(ids).astype('int64')
        raw_edges = pd.concat([raw_edges, extra[['citing','cited','citing_year']]], ignore_index=True)
        raw_edges = raw_edges.sort_values('citing_year', na_position='last').drop_duplicates(['citing','cited'])
        raw_edges = raw_edges.loc[raw_edges.citing != raw_edges.cited]
    return rows, space_ids, space, good_norm, raw_edges, known_years, n_cache_edges


def citation_edges(edges, ids, known_years):
    # Original reference lists for focal works. Do not treat absence from the
    # citation cache as zero total incoming citations.
    out = edges.loc[edges.citing.isin(ids), ['citing', 'cited']]
    received = edges.loc[edges.cited.isin(ids), ['citing', 'cited', 'citing_year']].copy()
    lag = received.citing_year - received.cited.map(known_years)
    received = received.loc[lag.between(1, LAG), ['citing', 'cited']]
    added = received.loc[~received.citing.isin(ids)]
    return out, received, added


def graph(ids, edges):
    # Canonical undirected edges, with all focal records occupying the first n
    # positions. Remove single-attachment neighbors, which cannot shorten paths
    # between focal works on this one-hop graph.
    source = edges.citing.to_numpy()
    target = edges.cited.to_numpy()
    ordered = pd.DataFrame({'u': np.minimum(source, target), 'v': np.maximum(source, target)})
    ordered = ordered.drop_duplicates(ignore_index=True)
    u = ids.get_indexer(ordered.u)
    v = ids.get_indexer(ordered.v)
    assert not ((u < 0) & (v < 0)).any()
    ext = np.r_[ordered.u.to_numpy()[u < 0], ordered.v.to_numpy()[v < 0]]
    uniq, degree = np.unique(ext, return_counts=True)
    vertices = ids.append(pd.Index(uniq[degree >= 2]))
    u = vertices.get_indexer(ordered.u)
    v = vertices.get_indexer(ordered.v)
    keep = (u >= 0) & (v >= 0)
    adj = sparse.csr_matrix((np.ones(keep.sum(), dtype=np.uint8), (u[keep], v[keep])),
                            shape=(len(vertices), len(vertices)))
    adj = adj.maximum(adj.T).tocsr()
    n_comp, membership = connected_components(adj, directed=False)
    biggest = np.bincount(membership[:len(ids)]).argmax()
    in_core = membership[:len(ids)] == biggest
    return adj, in_core, {
        'focal_nodes': len(ids), 'raw_edges': len(ordered), 'pruned_edges': int(adj.nnz // 2),
        'nodes': len(vertices), 'components': int(n_comp), 'main_focal_nodes': int(in_core.sum()),
        'neighbors_removed_degree_one': int((degree == 1).sum()),
    }


@njit
def breadth_first(indptr, indices, start):
    distance = np.full(len(indptr) - 1, -1, dtype=np.int32)
    queue = np.empty(len(distance), dtype=np.int32)
    queue[0] = start
    distance[start] = 0
    read, end = 0, 1
    while read < end:
        vertex = queue[read]
        read += 1
        new_distance = distance[vertex] + 1
        for j in range(indptr[vertex], indptr[vertex+1]):
            neighbor = indices[j]
            if distance[neighbor] < 0:
                distance[neighbor] = new_distance
                queue[end] = neighbor
                end += 1
    return distance


def pivot_mds(adj, core, landmarks, dimensions, *, fixed_pivots=None, pivots_out=None):
    indptr, indices = adj.indptr, adj.indices
    # Same deterministic farthest-point rule for both graph specifications.
    start = core[np.argmax(np.diff(indptr)[core])]
    pivots = np.empty(landmarks, dtype=int)
    distances = np.empty((len(core), landmarks), dtype=np.float32)
    nearest = np.full(len(core), np.iinfo(np.int32).max)
    if fixed_pivots is not None and len(fixed_pivots) != landmarks:
        raise ValueError('The fixed pivot count differs from the requested count')
    for j in range(landmarks):
        if fixed_pivots is not None:
            start = core[fixed_pivots[j]]
        dist = breadth_first(indptr, indices, start)[core]
        assert (dist >= 0).all()
        pivots[j] = np.searchsorted(core, start)
        distances[:, j] = dist
        if fixed_pivots is None:
            nearest = np.minimum(nearest, dist)
            start = core[np.argmax(nearest)]
    if pivots_out is not None:
        pivots_out.extend(pivots.tolist())
    np.square(distances, out=distances)
    pivot_squared = distances[pivots].astype(np.float64)
    colmean = pivot_squared.mean(axis=0)
    gram = -.5 * (pivot_squared - colmean - colmean[:, None] + colmean.mean())
    eig, basis = np.linalg.eigh(gram)
    positive = np.flatnonzero(eig > max(1e-10, eig[-1] * 1e-10))[::-1]
    keep = positive[:dimensions]
    distances -= distances.mean(axis=1, keepdims=True)
    distances -= colmean.astype(np.float32)
    distances += colmean.mean()
    coordinates = distances @ ((-.5 * basis[:, keep]) / np.sqrt(eig[keep])).astype(np.float32)
    return coordinates, int(len(positive))


def validate_distances(adj, core, vectors, pivot_count):
    rng = np.random.default_rng(13)
    a = rng.choice(len(core), size=min(80, len(core)), replace=False)
    b = rng.integers(len(core), size=(len(a), 64))
    truth = np.vstack([breadth_first(adj.indptr, adj.indices, core[i])[core[other]]
                       for i, other in zip(a, b, strict=True)]).ravel()
    embedded = np.linalg.norm(vectors[a, None] - vectors[b], axis=2).ravel()
    mask = truth > 0
    truth, embedded = truth[mask], embedded[mask]
    correction = float((truth @ embedded) / (embedded @ embedded))
    return {
        'pivots': pivot_count, 'dimensions': vectors.shape[1], 'random_pairs': int(mask.sum()),
        'stress1': float(np.linalg.norm(truth - embedded) / np.linalg.norm(truth)),
        'stress1_with_global_rescale': float(np.linalg.norm(truth - correction * embedded) / np.linalg.norm(truth)),
        'distance_spearman': float(spearmanr(truth, embedded).statistic),
    }


@njit
def group_spread(vectors, where, starts, ends):
    result = np.empty(len(starts), dtype=np.float64)
    dim = vectors.shape[1]
    for g in range(len(starts)):
        n = ends[g] - starts[g]
        mean = np.zeros(dim, dtype=np.float64)
        for j in range(starts[g], ends[g]):
            for k in range(dim):
                mean[k] += vectors[where[j], k]
        mean /= n
        total = 0.
        for j in range(starts[g], ends[g]):
            for k in range(dim):
                total += (vectors[where[j], k] - mean[k]) ** 2
        result[g] = 2 * total / (n-1)
    return result


def calculate_panel(rows, ids, spaces):
    sizes = rows.groupby(KEY).id.transform('size')
    data = rows.loc[sizes >= 2].sort_values(KEY, kind='stable')
    groups = data.groupby(KEY, sort=False)
    panel = groups.size().rename('n_papers').to_frame()
    n = panel.n_papers.to_numpy()
    within = []
    for level in LEVELS:
        counts = data.groupby(KEY + [level]).size()
        same = (counts * (counts-1)).groupby(level=KEY).sum().reindex(panel.index)
        within.append(1 - same.to_numpy() / (n * (n-1)))
    panel['classification'] = np.mean(within, axis=0)
    ends = np.cumsum(n)
    starts = ends - n
    for method, (lookup, vectors) in spaces.items():
        pos = lookup.get_indexer(data.id)
        assert (pos >= 0).all()
        panel[method] = group_spread(vectors, pos, starts, ends)
    # Verify the fast estimator against explicit distances on sampled real cells.
    rng = np.random.default_rng(424)
    for j in rng.choice(len(panel), min(30, len(panel)), replace=False):
        cell = data.iloc[starts[j]:ends[j]]
        i, k = np.triu_indices(len(cell), 1)
        for method, (lookup, vectors) in spaces.items():
            vec = vectors[lookup.get_indexer(cell.id)].astype('float64')
            np.testing.assert_allclose(panel[method].iloc[j], np.square(vec[i]-vec[k]).sum(axis=1).mean(),
                                       atol=1e-9, rtol=1e-7)
        labels = cell[LEVELS].to_numpy()
        np.testing.assert_allclose(panel.classification.iloc[j], np.mean(labels[i] != labels[k]), atol=1e-12)
    return panel.reset_index(), data


def boot_ratio(num, den, rng, repeats=2000):
    idx = rng.integers(len(den), size=(repeats, len(den)))
    denominator = den[idx].sum(axis=1)
    return num.sum(axis=0) / den.sum(), num[idx].sum(axis=1) / denominator[:, None]


def comparison_stats(panel, methods=METHODS, last_post_window=14):
    methods = list(methods)
    rng = np.random.default_rng(37164)
    profiles, slopes, correlations, contrasts, appointment = [], [], [], [], []
    for subj in SUBJECTS:
        data = panel.loc[panel.subject == subj].copy()
        field = data.loc[data.win.between(-3, 17)]
        sd = field[methods].std()
        center = field[methods].mean()
        for win, g in field.groupby('win'):
            for m in methods:
                profiles.append(dict(subject=subj, win_start=int(win*WINDOW), n=len(g),
                    method=m, q_mean=g[m].mean(), mean_z=(g[m].mean()-center[m])/sd[m]))
        g = field.loc[field.win.between(0, last_post_window)].copy()
        g = g.loc[g.groupby('professor_uid').win.transform('size') >= 2]
        y = g[methods] / sd
        x = (g.win * WINDOW + .5) / 10
        uid = g.professor_uid
        xd = x - x.groupby(uid).transform('mean')
        yd = y - y.groupby(uid).transform('mean')
        numerator = yd.mul(xd, axis=0).groupby(uid).sum().to_numpy()
        denominator = (xd**2).groupby(uid).sum().to_numpy()
        point, boot = boot_ratio(numerator, denominator, rng)
        for j, m in enumerate(methods):
            lo, hi = np.percentile(boot[:,j], [2.5, 97.5])
            slopes.append(dict(subject=subj, method=m, sd_per_decade=point[j], lo=lo, hi=hi,
                               q_per_decade=point[j]*sd[m], professors=len(numerator), windows=len(g)))
            if j != 2:
                delta = boot[:,2] - boot[:,j]
                lo, hi = np.percentile(delta, [2.5,97.5])
                contrasts.append(dict(subject=subj, versus=m, sd_per_decade_delta=point[2]-point[j], lo=lo, hi=hi))
        for i,m in enumerate(methods):
            for other in methods[i+1:]:
                correlations.append(dict(subject=subj, left=m, right=other, n=len(field),
                    spearman=spearmanr(field[m],field[other]).statistic))
        before = field.loc[field.win == -1].set_index('professor_uid')[methods]
        after = field.loc[field.win == 0].set_index('professor_uid')[methods]
        delta = ((after - before) / sd).dropna().to_numpy()
        if len(delta):
            point, boot = boot_ratio(delta, np.ones(len(delta)), rng)
            for j, m in enumerate(methods):
                lo, hi = np.percentile(boot[:,j], [2.5, 97.5])
                appointment.append(dict(subject=subj, method=m, delta_sd=point[j], lo=lo, hi=hi,
                                        professors=len(delta)))
    return {key:pd.DataFrame(value) for key,value in (
        ('profiles', profiles), ('slopes', slopes), ('correlations', correlations),
        ('slope_differences', contrasts), ('appointment_pair', appointment))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paths', type=Path, required=True)
    p.add_argument('--existing-private-root', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--extra-citations', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--pivots', type=int, default=100)
    p.add_argument('--dimensions', type=int, default=32)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.paths.read_text())
    rows, space_ids, space, good, edges, years, n_cache = load_inputs(
        cfg, args.existing_private_root, args.selection, args.extra_citations)
    print(f'{len(rows):,} mature, classified researcher-paper records; {len(edges):,} directed edges', flush=True)
    eligible = rows.id.isin(space_ids[good]) & rows.id.map(years).eq(rows.year)
    core = pd.Index(np.sort(rows.loc[eligible, 'id'].unique()))
    reference, received, added = citation_edges(edges, core, years)
    a, acore, ai = graph(core, reference)
    joint, jcore, ji = graph(core, pd.concat([reference, added], ignore_index=True))
    matched = np.flatnonzero(acore & jcore)
    ids = core[matched]
    print(f'Graph records: {len(core):,}; common largest-component records: {len(ids):,}; '
          f'{len(received):,} observed incoming, {len(added):,} additional directed links', flush=True)

    def fit(adj):
        with threadpool_limits(limits=4):
            return pivot_mds(adj, matched, args.pivots, args.dimensions)

    combined, combined_rank = fit(joint)
    print('Combined graph embedding ready', flush=True)
    outgoing, outgoing_rank = fit(a)
    print('Outgoing graph embedding ready', flush=True)
    spaces = {'shared_references': (space_ids, space), 'citation_network': (ids, combined),
              'graph_out_only': (ids, outgoing)}
    panel, matched_rows = calculate_panel(rows.loc[eligible & rows.id.isin(ids)], ids, spaces)
    coverage = []
    for subj in SUBJECTS:
        den = rows.loc[rows.subject == subj]
        num = matched_rows.loc[matched_rows.subject == subj]
        eligible_subj = rows.loc[eligible & (rows.subject == subj)]
        for phase, bounds in {'all':(-100,100), 'pre_-6_-1':(-3,-1), 'post_0_9':(0,4),
                              'post_10_19':(5,9), 'post_20_29':(10,14)}.items():
            d, e, n = (x.loc[x.win.between(*bounds)] for x in (den,eligible_subj,num))
            counts = d.groupby(KEY).size()
            coverage.append(dict(subject=subj, phase=phase, candidate_rows=len(d),
                shared_and_date_rows=len(e), matched_rows=len(n),
                candidate_windows=int((counts>=2).sum()), matched_windows=n.groupby(KEY).ngroups,
                matched_professors=n.professor_uid.nunique(),
                works_with_observed_incoming=n.loc[n.id.isin(received.cited),'id'].nunique(),
                matched_works=n.id.nunique()))
    pd.DataFrame(coverage).to_csv(args.out / 'coverage.csv', index=False)
    for name, table in comparison_stats(panel).items():
        table.to_csv(args.out / (name + '.csv'), index=False)
    diagnostics = dict(
        cache_directed_links=n_cache, all_directed_links=len(edges), latest_source_year=SOURCE_YEAR,
        last_focal_publication_year=LAST_YEAR, complete_window_width=WINDOW,
        matched_researcher_paper_links=len(matched_rows), matched_works=matched_rows.id.nunique(),
        matched_windows=len(panel), matched_professors=panel.professor_uid.nunique(),
        observed_incoming_links=len(received), outside_focal_incoming_links=len(added),
        focal_papers_with_observed_incoming=received.cited.nunique(),
        focal_papers_with_outside_focal_incoming=added.cited.nunique(),
        outside_focal_citing_papers=added.citing.nunique(),
        matched_papers_with_observed_incoming=matched_rows.loc[matched_rows.id.isin(received.cited),'id'].nunique(),
        matched_papers_with_outside_focal_incoming=matched_rows.loc[matched_rows.id.isin(added.cited),'id'].nunique(),
        graph_combined=ji | validate_distances(joint, matched, combined, args.pivots) |
            {'positive_pivot_eigenvalues':combined_rank},
        graph_out_only=ai | validate_distances(a, matched, outgoing, args.pivots) |
            {'positive_pivot_eigenvalues':outgoing_rank})
    (args.out/'diagnostics.json').write_text(json.dumps(diagnostics, indent=2)+'\n')
    print(json.dumps(diagnostics, indent=2), flush=True)


if __name__ == '__main__':
    main()
