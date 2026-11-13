"""Compare appointment phases with separate observation and matched samples."""

import argparse
from contextlib import closing
from datetime import date
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from analyze import BOOTSTRAPS, FIELDS
from career import bootstrap, main as load_references, reference_components


PHASES = {
    'pre_observed': (None, -1),
    'pre_two_year': (-2, -1),
    'early': (0, 1),
    'later_two_year': (2, 3),
    'later_observed': (2, None),
}
PAIRS = (
    ('pre_two_year', 'early'),
    ('pre_two_year', 'later_two_year'),
    ('early', 'later_two_year'),
)
METRICS = ('W', 'B', 'T', 'B_share')
KEYS = ['uid', 'field']
SEED = 20261113


def in_phase(relative_year, bounds):
    first, last = bounds
    mask = relative_year.notna()
    if first is not None:
        mask &= relative_year.ge(first)
    if last is not None:
        mask &= relative_year.le(last)
    return mask


def phase_rows(dates, linked, papers, moments, centers, center_squared, cutoff):
    """Retain each dated faculty member in every phase they can contribute to."""
    assert dates.uid.is_unique and dates.anchor.notna().all()
    anchored = dates.set_index('uid').anchor
    linked = linked[linked.year.le(cutoff)]
    papers = papers[papers.year.le(cutoff)]
    linked_year = linked.year - linked.uid.map(anchored)
    paper_year = papers.year - papers.uid.map(anchored)
    result = {}
    for name, bounds in PHASES.items():
        first, last = bounds
        frame = dates[KEYS].copy()
        frame['at_risk'] = True if first is None else dates.anchor + first <= cutoff
        bounded = first is not None and last is not None
        frame['complete'] = (dates.anchor + last <= cutoff) if bounded else False
        linked_counts = linked.loc[in_phase(linked_year, bounds)].groupby('uid').size()
        frame['linked'] = frame.uid.map(linked_counts).fillna(0).astype(int)

        records = []
        phase_papers = papers.loc[in_phase(paper_year, bounds)]
        for uid, group in phase_papers.groupby('uid', sort=True):
            ix = group.ix.to_numpy()
            w, b, t = reference_components(ix, moments, centers, center_squared)
            # Between-paper variation requires at least two papers to interpret.
            # A single paper still has a mathematically defined zero share if T > 0.
            records.append((uid, len(ix), w, b if len(ix) >= 2 else np.nan, t,
                            b / t if t > 1e-10 else np.nan))
        estimates = pd.DataFrame(records, columns=['uid', 'positioned', *METRICS])
        frame = frame.merge(estimates, on='uid', how='left', validate='one_to_one')
        frame['positioned'] = frame.positioned.fillna(0).astype(int)
        result[name] = frame
    return result


def scalar(values, rng):
    values = values.dropna().to_numpy(dtype=float)
    interval = bootstrap(values[:, None], rng)
    return {
        'n': len(values),
        'mean': float(values.mean()) if len(values) else None,
        'ci95': interval[:, 0].tolist() if interval is not None else None,
    }


def summarize(roster, rows):
    rng = np.random.default_rng(SEED)
    result = {}
    for field in FIELDS:
        roster_count = len(roster) if field == 'all' else int(roster.field.eq(field).sum())
        phases = {name: data if field == 'all' else data[data.field.eq(field)]
                  for name, data in rows.items()}
        stage_summary = {}
        for name, data in phases.items():
            risk = data[data.at_risk]
            first, last = PHASES[name]
            bounded = first is not None and last is not None
            stage_summary[name] = {
                'dated': len(data),
                'not_yet_at_risk': int((~data.at_risk).sum()),
                'at_risk': len(risk),
                'partial_calendar_window': int((~risk.complete).sum()) if bounded else None,
                'complete_calendar_window': int(risk.complete.sum()) if bounded else None,
                'no_linked_papers': int(risk.linked.eq(0).sum()),
                'linked_but_no_positioned_papers': int((risk.linked.gt(0) & risk.positioned.eq(0)).sum()),
                'one_positioned_paper': int(risk.positioned.eq(1).sum()),
                'multiple_positioned_papers': int(risk.positioned.ge(2).sum()),
                'one_paper_defined_B_share': int((risk.positioned.eq(1) & risk.B_share.notna()).sum()),
                'metrics': {metric: scalar(risk[metric], rng) for metric in METRICS},
            }
            if bounded:
                stage_summary[name]['complete_window_metrics'] = {
                    metric: scalar(risk.loc[risk.complete, metric], rng) for metric in METRICS
                }
        pair_summary = {}
        columns = KEYS + ['positioned', *METRICS]
        for left, right in PAIRS:
            a = phases[left].loc[phases[left].complete, columns]
            b = phases[right].loc[phases[right].complete, columns]
            matched = a.merge(b, on=KEYS, suffixes=('_before', '_after'), validate='one_to_one')
            metrics = {}
            for metric in METRICS:
                sample = matched
                if metric in ('B', 'B_share'):
                    sample = matched[matched.positioned_before.ge(2) & matched.positioned_after.ge(2)]
                before = sample[metric + '_before']
                after = sample[metric + '_after']
                paired = before.notna() & after.notna()
                metrics[metric] = {
                    'before': float(before[paired].mean()) if paired.any() else None,
                    'after': float(after[paired].mean()) if paired.any() else None,
                    'difference': scalar(after[paired] - before[paired], rng),
                }
            pair_summary[f'{right}_minus_{left}'] = {'calendar_eligible': len(matched), 'metrics': metrics}
        result[field] = {
            'roster': roster_count,
            'no_usable_anchor': roster_count - len(phases['early']),
            'stages': stage_summary,
            'paired': pair_summary,
        }
    return result


def publication_cutoff(cohort):
    uri = cohort.resolve(strict=True).as_uri() + '?mode=ro&immutable=1'
    with closing(sqlite3.connect(uri, uri=True)) as db:
        row = db.execute("SELECT value FROM provenance WHERE key='publication_cutoff'").fetchone()
    if row is None:
        raise ValueError('The S14 cohort is missing publication_cutoff provenance.')
    return date.fromisoformat(row[0]).year


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohort', type=Path, help='Frozen S14 cohort SQLite')
    parser.add_argument('appointments', type=Path, help='Reconciled S20 records.json')
    parser.add_argument('output', type=Path, help='New file for aggregate results')
    parser.add_argument('--cutoff', type=int, help='Last included calendar year (default: S14 cohort cutoff)')
    parser.add_argument('--anchor', choices=('official', 'baseline', 'database', 'proxy'), default='database')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists. Choose a new path.')
    archive_cutoff = publication_cutoff(args.cohort)
    cutoff = archive_cutoff if args.cutoff is None else args.cutoff
    if cutoff > archive_cutoff:
        parser.error('The requested cutoff exceeds the frozen publication coverage.')

    def collect(roster, linked, papers, centers, moments, center_squared, dates):
        selected = dates[args.anchor]
        selected = selected[selected.uid.isin(roster.uid) & selected.anchor.le(cutoff)]
        rows = phase_rows(selected, linked, papers, moments, centers, center_squared, cutoff)
        result = {
            'cutoff': cutoff,
            'archive_cutoff': archive_cutoff,
            'anchor_mode': args.anchor,
            'anchor_basis_counts': selected.basis.value_counts().to_dict(),
            'relative_year_bounds': PHASES,
            'bootstrap_reps': BOOTSTRAPS,
            'geometry': 'S14 whole-corpus shared-reference SVD, 32 coordinates, at least two positioned references per paper',
            'aggregation': 'Pool eligible papers within each faculty-phase, then weight faculty equally',
            'stage_note': 'Open-ended phases are descriptive through cutoff. Unobserved phases have no metric.',
            'pair_note': 'Only completed two-year windows; B and B/T additionally need two positioned papers in both phases.',
            'timing_note': 'Early publications may carry over prior work. Annual dates do not identify when a paper was researched or verify postdoc dates.',
            'fields': summarize(roster, rows),
        }
        with args.output.open('x', encoding='utf-8') as out:
            json.dump(result, out, indent=2, allow_nan=False)
            out.write('\n')

    load_references([str(args.cohort), str(args.appointments), str(args.output)],
                    collect_phases=collect, cutoff=cutoff)


if __name__ == '__main__':
    main()
