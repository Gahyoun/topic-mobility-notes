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

## Largest weakly connected component

Audit the archived core citation graph, the earlier-year core DAG, and the earlier-year DAG with external references as sinks. GCC membership ignores edge direction. The coordinate calculation retains citation direction, strict core year order, and external sink edges. Work-level sketching and ancestor propagation only use nodes inside a work's weak component. Restricting to the GCC therefore preserves the stored coordinates exactly. We hold the existing cluster labels fixed when comparing graph scopes.

With the completed S27–S29 outputs, run these commands from the source checkout. Use a new private directory and set COHORT to the S14 SQLite, DB to the S27 SQLite, TENSORS to the S28 tensors, and EVENT to the S29 output:

    mkdir "$GCC"
    python pipelines/career/gcc_audit.py "$COHORT" "$DB" "$GCC/coverage.json" --membership "$GCC/membership.csv"
    python pipelines/career/select_gcc.py "$DB" "$GCC/membership.csv" "$GCC/portfolio-gcc.sqlite" --event-coverage "$GCC/event-presence.json"
    python pipelines/career/native_flow.py "$GCC/portfolio-gcc.sqlite" "$GCC/tensors"
    python pipelines/career/faculty_tensors.py "$GCC/portfolio-gcc.sqlite" "$GCC/tensors"
    python pipelines/career/appointment_dag.py "$GCC/portfolio-gcc.sqlite" "$GCC/tensors" "$GCC/appointment"
    python pipelines/career/compare_gcc.py "$TENSORS" "$EVENT" "$GCC/tensors" "$GCC/appointment" "$GCC/comparison.json"
    python pipelines/figures/stream.py "$GCC/tensors" "$GCC/annual-stream.html"

The comparison verifies exact coordinate and cluster identity for the retained works in all six dimensions. The field summaries use the same faculty rows present in both scopes. The copied SQLite keeps original coverage tables and linked publications, and restricts only dag_native_coordinates. Keep the membership, row-level files and generated SQLite outside Git.

## First appointment without an elapsed-years threshold

Use the saved GCC portfolio database and the generated GCC tensor directory.

    python pipelines/career/open_followup.py "$GCC/portfolio-gcc.sqlite" "$GCC/tensors" "$GCC/open-followup"

This analysis uses all observed pre-appointment years and all publication years from the year after appointment through 2025. The appointment year is recorded separately. The first two following years contribute to the post-appointment period and are also described separately. The roster retains teachers with zero observed post-appointment publications, no elapsed post-appointment publication years, and missing appointment dates. The summary provides metric-specific counts by field and available follow-up length. Convergence and divergence describe a single event-spanning transport comparison, annual parallel-topic rates use same-year papers, and branching uses an explicit citation from a novel post-appointment topic into a retained pre-appointment topic. The balanced four-window comparison remains an optional sensitivity analysis.

### Reproduce the sample alluvial

    python pipelines/figures/sample_alluvial.py "$DB" "$GCC/open-followup/faculty-d1024.csv" "$GCC/analysis-alluvial.html" --counts "$GCC/analysis-alluvial-counts.json"

Here DB is the S27 portfolio SQLite. The chart preserves mutually exclusive coverage branches and separately counts the union and intersection of convergence/divergence, annual parallel topics, and branching availability. The HTML and optional JSON contain aggregate counts only.

### Reproduce the common-sample comparison

    python pipelines/figures/sample_selection.py "$GCC/open-followup/faculty-d1024.csv" "$GCC/selection-summary.json" --figure "$GCC/topic-sample-selection.png"

Both scopes use the same precomputed faculty-level estimates. The common sample requires convergence/divergence, paired annual parallel-topic rates, and branching availability. The binary contrasts use the weighted difference between the common and additional subsets, with exact binomial bounds. Annual parallel-topic contrasts use 10,000 stratified teacher bootstrap replicates that share the common-subset draws across the two scopes. The generated summary is aggregate. Use --font with local Korean fonts if the NanumHuman files are unavailable.

## Confirmed namesake corrections and GCC refit

Rebuild the KOAD decisions from the frozen augmented evidence database. The November 11 refit used [KOAD at `3e2a7d1`](https://github.com/Gahyoun/korea-openalex-disambiguation/tree/3e2a7d1930c188f854ffd7669ae0c48ee6ec7bec) and applied its high-confidence drop decisions plus documented wrong links and independently adjudicated professor–work pairs.

    PYTHONPATH=/path/to/koad/src python -m koad.cli classify --db "$AUGMENTED" --output "$KOAD"
    python pipelines/career/filter_namesakes.py "$ORIGINAL_COHORT" "$KOAD/identity_decisions.sqlite" "$REFERENCE_CACHE" "$METADATA" "$ADJUDICATIONS" "$CORRECTED_COHORT"

`REFERENCE_CACHE` is the S14 reference SQLite with the frozen `work_alias` table. `METADATA` is the frozen web SQLite with the private wrong-link quarantine table. `ADJUDICATIONS` is a private JSON array with `professor_uid`, `work_id`, `decision` (`keep` or `drop`), and `evidence` for each row. Preserve dated source evidence with this input. The filter resolves aliases before matching. An explicit verified keep takes precedence. Review-only decisions and bibliographic metadata remain available for audit. A work remains in the core when another valid faculty link survives, and it can remain an external reference when cited by an eligible work. The cohort copy records changes in `namesake_filter_audit`.

Use the corrected cohort in S27 and rerun the directed embedding before selecting the GCC. Use a new private output directory.

    mkdir "$REFIT" "$REFIT/gcc"
    DB="$REFIT/portfolio.sqlite"
    GCC="$REFIT/gcc"
    python pipelines/career/portfolio.py "$CORRECTED_COHORT" "$APPOINTMENT_EVIDENCE" "$METADATA" "$REFIT"
    python pipelines/career/uncertainty.py "$DB" "$REFIT/results.json"
    python pipelines/career/audit_citation_cycles.py "$CORRECTED_COHORT" "$DB" "$REFIT/cycle-audit.json"
    python pipelines/career/native_dag.py "$CORRECTED_COHORT" "$DB" "$REFIT/dag-graph.json"
    python pipelines/career/gcc_audit.py "$CORRECTED_COHORT" "$DB" "$GCC/coverage.json" --membership "$GCC/membership.csv"
    python pipelines/career/select_gcc.py "$DB" "$GCC/membership.csv" "$GCC/portfolio-gcc.sqlite" --event-coverage "$GCC/event-presence.json"
    python pipelines/career/native_flow.py "$GCC/portfolio-gcc.sqlite" "$GCC/tensors"
    python pipelines/career/faculty_tensors.py "$GCC/portfolio-gcc.sqlite" "$GCC/tensors"
    python pipelines/career/open_followup.py "$GCC/portfolio-gcc.sqlite" "$GCC/tensors" "$GCC/open-followup"
    python pipelines/figures/stream.py "$GCC/tensors" "$GCC/annual-stream.html"
    python pipelines/figures/sample_alluvial.py "$DB" "$GCC/open-followup/faculty-d1024.csv" "$GCC/analysis-alluvial.html" --counts "$GCC/alluvial-counts.json"
    python pipelines/figures/sample_selection.py "$GCC/open-followup/faculty-d1024.csv" "$GCC/selection-summary.json" --figure "$GCC/topic-sample-selection.png"

The refit uses all observed publication years through 2025 and repeats the embedding, common clustering and peer estimates. Short follow-up and no observed publications remain observation states in the roster. Use the saved outputs at all six dimensions for dimension sensitivity and keep row-level results outside the repository.

### Compare the two KOAD versions on the same faculty

Use the frozen GCC open-followup output directories before and after the confirmed namesake correction.

    python pipelines/career/koad_sensitivity.py "$BEFORE" "$AFTER" "$OUTPUT/koad-sensitivity.json" --figure "$OUTPUT/koad-sensitivity.png"

The comparison aligns rows by faculty identifier and verifies the field, appointment anchor and observation metadata. It intersects the faculty with all four signals across both versions and all six dimensions. Each estimate averages the within-faculty corrected-minus-original difference. Annual parallel-topic rates are first differenced across appointment periods within each version.

Binary intervals combine simultaneous exact binomial bounds on discordant gains and losses. Annual parallel-topic intervals use 10,000 paired faculty resamples shared across dimensions. Intervals are pointwise. The figure and JSON contain aggregates only. Use --font with a local Korean font when NanumHuman is unavailable.

### Attribute the annual-parallel change and audit bibliographic coverage

Use the publication metadata from the two full portfolio SQLite databases, the corrected cohort with its `namesake_filter_audit` table, and the two GCC coordinate and follow-up snapshots.

    python pipelines/career/koad_parallel.py "$BEFORE_DB" "$AFTER_DB" "$CORRECTED_COHORT" \
      "$BEFORE_GCC_TENSORS" "$AFTER_GCC_TENSORS" "$BEFORE" "$AFTER" \
      "$OUTPUT/koad-parallel.json" --figure "$OUTPUT/koad-parallel.png"

Pass `--before-full-tensors "$BEFORE_FULL_TENSORS"` if the original GCC archive stores the paper axes only. The script reconstructs those coordinates by exact work ID from the full tensor archive and verifies the cluster labels.

The script checks that the publication rows differ by the adjudicated faculty–work links only. On the previously fixed common sample it compares original rates, rates after removing those links with the original embedding and clustering, and the fully refitted rates. It also intersects the two GCC paper lists, then replaces cluster membership first and coordinates second. The diagnostic terms sum to the refit contrast within each faculty row. The bootstrap uses the same 10,000 paired faculty draws across stages and dimensions. Coverage is aggregated over the entire roster from the corrected cohort database. The output includes no faculty or work IDs.
