"""Embed backward citation neighborhoods in six nested CountSketch dimensions."""
import argparse
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from numba import njit
from sklearn.cluster import MiniBatchKMeans

DIMENSIONS = (32, 64, 128, 256, 512, 1024)
BLOCK = 32
LAMBDA = 0.25


@njit
def mix(x):
    x = (x ^ (x >> 30)) * np.uint64(0xbf58476d1ce4e5b9)
    x = (x ^ (x >> 27)) * np.uint64(0x94d049bb133111eb)
    return x ^ (x >> 31)


@njit
def direct_sketch(count, source, reference, degrees):
    x = np.zeros((count, 1024), dtype=np.float32)
    for j in range(len(reference)):
        node, ref = source[j], np.uint64(reference[j])
        weight = np.float32(1.0 / degrees[node])
        for block in range(32):
            h = mix(ref + np.uint64(block + 1) * np.uint64(0x9e3779b97f4a7c15))
            bucket = block * 32 + int(h & np.uint64(31))
            x[node, bucket] += weight if (h & np.uint64(32)) else -weight
    return x


@njit
def propagate(x, order, bounds, target, degrees):
    for i in order:
        start, end = bounds[i], bounds[i + 1]
        valid = 0
        for j in range(start, end):
            if degrees[target[j]] > 0:
                valid += 1
        if valid == 0:
            continue
        for col in range(x.shape[1]):
            value = 0.0
            for j in range(start, end):
                if degrees[target[j]] > 0:
                    value += x[target[j], col]
            x[i, col] = (1 - LAMBDA) * x[i, col] + LAMBDA * value / valid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cohort', 'portfolio', 'output'):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    with sqlite3.connect(a.portfolio) as db:
        works = pd.read_sql_query('SELECT work, MIN(year) AS year FROM publication_metadata '
                                  'GROUP BY work ORDER BY work', db)
    with sqlite3.connect(a.cohort.resolve().as_uri() + '?mode=ro&immutable=1', uri=True) as db:
        db.execute('CREATE TEMP TABLE nodes(work TEXT PRIMARY KEY, ix INTEGER) WITHOUT ROWID')
        db.executemany('INSERT INTO nodes VALUES (?,?)', zip(works.work, range(len(works))))
        edges = pd.read_sql_query('''SELECT n.ix source, CAST(SUBSTR(e.reference_id,2) AS INTEGER) reference,
                      COALESCE(r.ix,-1) target
                  FROM nodes n JOIN outgoing_reference e ON e.work_id=n.work
                  LEFT JOIN nodes r ON e.reference_id=r.work''', db, dtype=np.int64)
    source, reference, target = (edges[k].to_numpy() for k in ('source', 'reference', 'target'))
    del edges
    years = works.year.to_numpy()
    gap = years[source] - years[target]
    core = target >= 0
    admissible = (reference > 0) & (~core | (gap > 0))
    internal = admissible & core
    counts = np.bincount(source[admissible], minlength=len(works))
    stats = {'unique_linked_papers': len(works), 'papers_with_admissible_references': int((counts>0).sum()),
             'edges': int(admissible.sum()), 'external_sink_edges': int((admissible & ~core).sum()),
             'retained_core_edges': int(internal.sum()), 'same_year_core_edges_omitted': int((core & (gap==0)).sum()),
             'reverse_year_core_edges_omitted': int((core & (gap<0)).sum()),
             'dimensions': DIMENSIONS, 'ancestor_mixing': LAMBDA}
    print(stats, flush=True)
    x = direct_sketch(len(works), source[admissible], reference[admissible], counts)
    s, t = source[internal], target[internal]
    order = np.argsort(s, kind='stable')
    bounds = np.searchsorted(s[order], np.arange(len(works) + 1))
    propagate(x, np.argsort(years, kind='stable'), bounds, t[order], counts)
    live = np.flatnonzero(counts)
    # Resolve exact signed-sketch cancellation consistently at all dimensions.
    norm = np.linalg.norm(x[live, :BLOCK], axis=1)
    zero = live[norm == 0]
    stats['first_block_cancellations'] = len(zero)
    if len(zero):
        first = np.full(len(works), np.iinfo(np.int64).max, dtype=np.int64)
        np.minimum.at(first, source[admissible], reference[admissible])
        sketch = direct_sketch(len(zero), np.arange(len(zero)), first[zero],
                               np.ones(len(zero), dtype=np.int64))
        x[zero, :BLOCK] = sketch[:, :BLOCK]
    anchors = x[live, :64]
    anchors /= np.linalg.norm(anchors, axis=1, keepdims=True)
    clusters = MiniBatchKMeans(n_clusters=64, n_init=3, random_state=321110,
                               batch_size=4096).fit_predict(anchors)
    with sqlite3.connect(a.portfolio) as db:
        db.execute('CREATE TABLE dag_native_coordinates(work TEXT PRIMARY KEY, cluster INTEGER NOT NULL, '
                   'vector BLOB NOT NULL) WITHOUT ROWID')
        db.executemany('INSERT INTO dag_native_coordinates VALUES (?,?,?)',
                       ((works.work.iloc[i], int(k), x[i].astype('<f4',copy=False).tobytes())
                        for i,k in zip(live, clusters, strict=True)))
    a.output.write_text(json.dumps(stats, indent=2) + '\n')
    print(stats, flush=True)


if __name__ == '__main__':
    main()
