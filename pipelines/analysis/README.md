# analysis

Numbered processing steps. Each step writes kept / dropped / changes files under
`<private_root>/outputs/SNN_*` and appends to `<private_root>/outputs/STEP_LOG.md`.
Outputs hold raw `professor_uid` and work ids, so they stay outside this repository.

## Setup

Create `~/.config/korea-topic-mobility/paths.json` (or point `$TOPIC_MOBILITY_PATHS` at another file):

```json
{"web_db": ".../professor_affiliation_timeline_through_2026_web.sqlite",
 "koad_db": ".../identity_decisions.sqlite",
 "openalex_cache": ".../professor-career-dashboard/openalex",
 "release_work_dir": ".../topics_release",
 "private_root": "~/work/topic-mobility-private"}
```

Run every step with `python3 pipelines/analysis/sNN_*.py`. Source DBs are opened read-only (`immutable=1`).
References live in a separate `refs/work_references.sqlite` that you ATTACH.

## Paper filter

`identity_decision = 'keep'` and KOAD `work_decision.decision != 'namesake_paper'` (ATTACH
`identity_decisions.sqlite`), article/review/letter/editorial, `phd_year - 5 <= year <= 2026`.
`identity_decision` takes only `keep` / `duplicate_drop_candidate`. A filter on
`identity_decision != 'namesake_paper'` removes nothing.

S01 also excludes KOAD review rows without a seed, core coauthor, or institution
link as `trajectory_unanchored_review`. This is an unresolved identity exclusion,
not a confirmed namesake label. Topic distance and an ORCID match alone do not
decide attribution. S03 compares this rule with the permissive policy that
readmits eligible review exclusions within the S02 roster, and the strict policy
that removes all review rows. Previously reported estimates predate this rule.

## Steps

| step | script | what |
|---|---|---|
| S01 | `s01_select.py` | paper rows by the filter above; KOAD review flags kept for sensitivity |
| S02 | `s02_cells.py` | event_time = year − first appointment; pre-plot diagnostics; window choice (n<2 share ≤ 20 %) |
| S03 | `s03_raoq.py` | Rao Q̂ per professor × window from OpenAlex classification (topic one-hot, tree distance); KOAD policies P1/P2/P3 |
| S04 | `s04_curves.py` | T4 preliminary Q0 curve: raw mean, person-FE profile, slopes, n-dependence, field-wide period reference |
| S05 | `s05_refs.py` | T1 `work_references` from the OpenAlex cache (no re-fetch) |
| S06 | `s06_alias.py` | merged-id → canonical record pairs recoverable per batch |
| S07 | `s07_coverage.py` | reference coverage (acceptance ≥ 80 %), works absent from the cache |
| S08 | `s08_space.py` | T2 shared space: sublinear TF-IDF on references → TruncatedSVD, fit once on records ≤ base year |
| S09 | `s09_validate.py` | validation with signals not used in construction; Procrustes frame dependence |
| S10 | `s10_stability.py` | rotation-invariant stability (pairwise cosine, cell Q̂) across base year / seed / dims |
| S11 | `s11_panel.py` | T3 person-window panel: Q̂, R̃², D̃, θ / θ₀ / θ̃, unbiased shift `shift2_u`, persistence, integration ratio I, cluster visibility metrics |
| S11b | `s11b_checks.py` | Q̂ vs brute force, exact I decomposition, noise-floor calibration under a no-shift null |
| S12 | `s12_first_regression.py` | first regression: person-FE event-time profiles and slopes by subject, period reference |

Main space: `b2025` (one fit on all full years). Earlier base years drop most references of recent works
from the frame (S08), so they are sensitivity runs only. `theta_corr` is miscalibrated (null ratio 1.19);
use `shift2_u`, whose null mean is 0.
