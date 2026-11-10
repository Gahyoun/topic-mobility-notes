# Topic mobility research pipeline

Publication cutoff is 2025-12-31. The source includes the analysis procedures. The frozen source databases, adjudication evidence, paper vectors and researcher-level outputs are stored separately in the private research workspace.

## Stage map

| Stage | Inputs and operation |
| --- | --- |
| S01–S12 | Apply the frozen authorship decisions, build the reference cache and original classification/shared-reference comparisons. See [the original stage instructions](analysis/README.md). These scripts also reproduce legacy summaries. |
| S13 | Use the frozen OpenAlex snapshot/cache to assemble the three-year incoming-citation archive, check authorship and construct A–B–A citation paths. The API scripts describe their inputs with --help. Live API queries can return different records and are unnecessary for S27–S28. |
| S14 | Rebuild the 2025 roster and outgoing citation database from the work, identity and reference snapshots. Run from the research workspace. TOPIC_MOBILITY_OFFICIAL_DATES points to a private JSON array of [roster spelling, latest institution, doctoral year, appointment year, evidence URL, evidence note] records. |
| S15–S20 | Reconcile appointment evidence, proxy dates and publication follow-up. The original scripts under upstream/ retain their frozen workspace layout. Their queue, decision and evidence files are required private inputs. Run the scripts in the numbered order after restoring the original per-stage directories and files. |
| S21 | Compare citation reference components with paired appointment windows. |
| S22–S26 | Reproduce matched reference-geometry and time-window sensitivity analyses. |
| S27 | Rebuild the full-roster activity panel, reference-space geometry and bibliographic SQLite from the S14 cohort and the reconciled S20 evidence. |
| S28 | Construct backward-citation DAG embeddings, annual peer streams and full-dimensional research trajectories. |
| S29 | Compare four paired trajectory signals before and after first appointment using the saved S28 tensors. |

## Run S27–S29 offline

Use Python 3.11 and install the pinned requirements.

    python -m pip install -r pipelines/requirements.txt
    PYTHONPATH=pipelines/career python -m unittest discover -s pipelines/tests

Write a private inputs.json with the paths to the frozen inputs. Run:

    python pipelines/run_career.py /path/to/inputs.json /path/to/new-private-output --only-portfolio --figures

The output directory must be new and outside the checkout. Add --font followed by Korean font files when the system does not have the NanumHuman fonts used by the optional S27 figure. Omit --only-portfolio to include S22–S26.

| Input key | Frozen input |
| --- | --- |
| cohort | S14 cohort-2025.sqlite with roster_coverage, faculty_publication, reference_coverage and outgoing_reference tables |
| appointment_evidence | S20 records.json with the reconciled KRI/CV, roster and proxy timing evidence |
| metadata_db | Read-only web.sqlite with work_authorship_evidence bibliographic metadata |
| archive | S13 archived citation SQLite, for the legacy matched samples |
| appointments | S19 follow-up records, for the legacy matched samples |
| pilot_reference | S13 frozen reference comparison summary, for the legacy matched samples |

The runner records input and script SHA-256 hashes, package versions and the executed sequence in manifest.json. Dependency versions and fixed seeds constrain numerical reproducibility. Float and clustering implementations may differ across platforms. Earlier identity and appointment stages depend on the frozen human review inputs.

## S28 tensor axes

Output subdirectories are d0032, d0064, d0128, d0256, d0512 and d1024. Within a dimension:

- paper_vectors.npy is paper × coordinate in paper_axis.npz order. The paper axis also records a common cluster label.
- annual_stream.npz contains field × year × cluster counts, centers and annual velocities. The last cluster slot is the whole field, or the whole corpus when field is all. Velocity rows use target_year and the supported array.
- trajectory_tensor.npy is paired transition × velocity channel × coordinate. Read its row index in transition_index.csv. The seven channels are observed, corpus, field, local, observed minus corpus, observed minus field and observed minus local, in that order. Units are normalized embedding coordinates per year.
- faculty_annual_centers.npy and faculty_window_centers.npy use the row orders in the shared annual_index.csv and window_index.csv. A row with zero positioned works has NaN coordinates. Roster rows without a linked work are represented in faculty_roster.csv without an inferred publication start.
- window_observation.csv describes coverage and the multi label for individual windows. summary.json reports teacher-weighted bootstrap summaries and counts. display_projection.npz records the fixed two-dimensional display transform. stream_display.json and annual-stream.html are aggregated views.

The cycle audit compares the unfiltered core graph with the earlier-year graph using strongly connected components. Run career/audit_citation_cycles.py with the S14 cohort SQLite, S27 portfolio SQLite, and an output JSON path. Core edges in the embedding require a strictly earlier target publication year. Outside references are terminal nodes. Propagation and stable hashing are defined in native_dag.py. Incoming citation edges stay in S13. Each dimension uses the same eligible papers and the same 64 subject groups. Annual streams are described using unique works within each field and calendar year. Portfolio transitions use complete consecutive five-year windows, keep empty windows in the observation index, and leave the focal teacher's papers out of the matched background. The four flags are descriptive diagnostics that can co-occur.

## S29 first-appointment comparison from a saved stream

Run the event comparison directly against the private S27 SQLite and the S28 tensor root:

    python pipelines/career/appointment_dag.py /private/portfolio.sqlite /private/tensors /private/new-event-output
    python pipelines/figures/s29.py /private/new-event-output/summary.json /private/new-event-output/figure.png

First appointment comes from the database date and its recorded evidence category. The cohort retains every roster row with a supported reference link. The event year and the next two years are separate; paired post-appointment comparison starts in year +3. Each teacher uses the largest feasible equal pre/post width from 2–5 calendar years, with two consecutive blocks on each side and a common cutoff of 2025. Both calendar comparisons need two embedded papers per block and sufficient peer support to enter the common four-signal comparison. The common group is intersected across the six dimensions. The output also gives metric-specific paired counts and field-specific publication observation and censoring states.

Binary changes are paired proportion differences. The 95% bounds project simultaneous exact binomial intervals for gains and losses, with Bonferroni correction. The continuous excess and 1024-dimensional publication-count diagnostic use teacher-level resampling with fixed coordinates and peer estimates. The latter thins corresponding pre/post source and target clouds to the same count using 64 within-teacher draws without replacement. The complete per-teacher CSVs remain private.
