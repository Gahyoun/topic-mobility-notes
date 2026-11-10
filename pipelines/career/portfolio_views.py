"""Stable queries for observing active, empty and geometry-eligible windows."""


def create_views(db):
    db.executescript("""
      CREATE VIEW IF NOT EXISTS faculty_trajectory AS
      SELECT f.uid, f.field, f.appointment_year, f.appointment_basis, f.first_year,
             f.last_year, f.observed_tail_years, f.zero_last_3, f.zero_last_5,
             w.year_start, w.year_end, w.exposure, w.state,
             w.n_papers, w.n_geometry, w.q, w.raw_q, w.multi, w.effective_groups,
             t.previous_state, t.geometry_complete, t.con_div, t.con, t.div,
             t.retained_groups, t.new_groups, t.branch_share, t.branch,
             t.location, t.scale, t.shape, t.transport,
             CASE WHEN f.first_year IS NULL THEN 'unlinked' ELSE 'linked' END AS roster_state
      FROM faculty_observation f LEFT JOIN portfolio_windows w USING(uid)
      LEFT JOIN portfolio_transitions t ON w.uid=t.uid AND w.year_start=t.year_start;

      CREATE VIEW IF NOT EXISTS cluster_topic_summary AS
      SELECT cluster, topic_name, count(DISTINCT work) AS unique_works
      FROM publication_metadata
      WHERE cluster IS NOT NULL AND topic_name IS NOT NULL
      GROUP BY cluster, topic_name;
    """)
