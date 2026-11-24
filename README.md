# Research notes

Password-protected static site. Every page, table and figure is encrypted at build time
(AES-256-GCM, key from PBKDF2-SHA256 with 600,000 iterations and a per-build random salt)
and decrypted in the browser after the password is entered. The published research notes are held in `data/site.json`.

[Reproduction pipelines](pipelines/) contain source code, input schemas and execution commands. Frozen data and researcher-level results are kept in the private research workspace.

The current citation design starts with eligible faculty-authored papers, denoted B,
and the observed papers A that directly cite them. A citation edge A → B means that
A cites B. The analysis uses these direct edges without expanding the reference
lists of A. Archived outgoing-reference embeddings and A–B–A return analyses have
their own definitions and reproduction instructions in the pipeline documentation.

The working B count is 167,756, as reported by the researcher. The current review
has not independently queried the private research database after all exclusions.
Public dashboard aggregates can corroborate the roster and their own publication
scope. They do not provide the work-level citation records needed to validate B,
the direct A → B edges or the research subpages' analytical sample sizes.
