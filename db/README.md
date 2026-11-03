# Database queries

The original lastfirstdate pre-filter SQL query (Stage 1) belongs here as
`lastfirstdate_prefilter.sql`. It is not in this repo yet.

Until it is added, Stage 1 is implemented in Python (requirement FR-LM-04 in
`docs/SPEC.md`) and distance uses straight-line kilometres (decision D9). Once the query is
here, a test will check that the SQL and the Python filter keep the same pairs.
