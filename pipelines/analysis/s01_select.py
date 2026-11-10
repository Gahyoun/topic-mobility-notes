#!/usr/bin/env python3
"""S01 select: paper rows for the topic-mobility analysis.

Rule (same as public dashboard PR #18, plus journal types as in the reference cache):
  identity_decision = 'keep'                         (KOAD 'drop' already applied in _web DB)
  KOAD work_decision.decision != 'namesake_paper'    (removes namesake rows KOAD left as 'review')
  work_type in article/review/letter/editorial
  publication_year not null, <= 2026 (2026 partial: snapshot ends 2026-08-23)
  publication_year >= phd_year - 5 when phd_year known
  topic_id not null (needed for T4 only; rows without topic are saved separately)
  KOAD review has a seed, core-coauthor or institution link to the target career

Rows are evaluated reason-by-reason in the order above; the first failing reason is recorded.
KOAD review flags are carried so that the review-policy sensitivity can be computed later:
  koad_decision, koad_action (keep/drop/review).

Outputs (private, never in a repo): S01_select/kept.csv.gz, dropped.csv.gz, changes.md
"""
import csv, gzip, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db  # noqa: E402

STEP = "S01_select"


def main():
    out = db.OUT / STEP
    out.mkdir(parents=True, exist_ok=True)
    con = db.ro(db.WEB)
    con.execute(f"ATTACH DATABASE 'file:{db.KOAD}?mode=ro&immutable=1' AS koad")
    rows = con.execute("""
        SELECT w.professor_uid, w.work_id, w.publication_year, w.work_type, w.identity_decision,
               w.topic_id, w.subfield_id, w.field_id, w.domain_id,
               p.phd_year, p.subject,
               k.decision, k.recommended_action,
               k.is_seed, k.core_coauthor_connected, k.institution_connected
        FROM work_authorship_evidence w
        JOIN professors p USING (professor_uid)
        LEFT JOIN koad.work_decision k
               ON k.professor_uid = w.professor_uid AND k.work_id = w.work_id
    """)
    reasons, kept_n, dropped_n = Counter(), 0, 0
    review_policy = Counter()
    years = Counter()
    fk = ["professor_uid", "work_id", "year", "work_type", "topic_id", "subfield_id", "field_id",
          "domain_id", "subject", "koad_decision", "koad_action"]
    with gzip.open(out / "kept.csv.gz", "wt", newline="") as fk_, \
         gzip.open(out / "dropped.csv.gz", "wt", newline="") as fd_:
        wk = csv.writer(fk_); wd = csv.writer(fd_)
        wk.writerow(fk); wd.writerow(fk + ["reason"])
        for (uid, wid, y, wt, dec, t, s, f, d, phd, subj, kdec, kact,
             is_seed, core_coauthor_connected, institution_connected) in rows:
            wid = db.short(wid)
            # A merged profile can carry the same ORCID across unrelated works.
            unanchored_review = kact == "review" and not (
                is_seed or core_coauthor_connected or institution_connected
            )
            reason = ("identity_not_keep" if dec != "keep" else
                      "koad_namesake" if kdec == "namesake_paper" else
                      "non_journal_type" if wt not in db.JOURNAL_TYPES else
                      "no_year" if y is None else
                      "after_2026" if y > db.MAX_YEAR else
                      "before_phd_minus_5" if (phd is not None and y < phd - 5) else
                      "no_topic" if not t else
                      "trajectory_unanchored_review" if unanchored_review else None)
            rec = [uid, wid, y, wt, t, s, f, d, subj, kdec or "", kact or ""]
            if kdec is None:
                reasons["_koad_row_missing"] += 1
            if reason:
                reasons[reason] += 1; dropped_n += 1
                wd.writerow(rec + [reason])
                # S03 can readmit review exclusions under its permissive policy.
            else:
                kept_n += 1; years[y] += 1
                wk.writerow(rec)
            # review-policy sensitivity bookkeeping (on rows passing every non-KOAD rule)
            base_ok = (dec == "keep" and wt in db.JOURNAL_TYPES and y is not None and y <= db.MAX_YEAR
                       and not (phd is not None and y < phd - 5) and t)
            if base_ok:
                review_policy[(kdec, kact)] += 1
    total = kept_n + dropped_n
    lines = [f"# {STEP}", "",
             f"source rows (work_authorship_evidence) {total:,}; kept {kept_n:,}; dropped {dropped_n:,}", "",
             "## dropped by first failing reason (in rule order)"]
    order = ["identity_not_keep", "koad_namesake", "non_journal_type", "no_year",
             "after_2026", "before_phd_minus_5", "no_topic", "trajectory_unanchored_review"]
    lines += [f"- {r}: {reasons.get(r, 0):,}" for r in order]
    lines += [f"- (rows without any KOAD record: {reasons.get('_koad_row_missing', 0):,})", "",
              "## KOAD decision x action among rows passing all non-KOAD rules (review-policy base)"]
    lines += [f"- {k[0]} / {k[1]}: {v:,}" for k, v in sorted(review_policy.items(), key=str)]
    lines += ["", "## kept rows by publication_year"]
    lines += [f"- {y}: {years[y]:,}" for y in sorted(years)]
    lines += ["", "files: kept.csv.gz, dropped.csv.gz (raw uid/work ids, private only)"]
    (out / "changes.md").write_text("\n".join(lines) + "\n")
    with open(db.OUT / "STEP_LOG.md", "a") as fh:
        fh.write("\n" + "\n".join(lines[:12]) + "\n")
    print("\n".join(lines[:20]))
    print("...", "\n".join(lines[20:30]))


if __name__ == "__main__":
    main()
