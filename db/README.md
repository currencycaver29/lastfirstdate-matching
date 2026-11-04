# Database

`schema.sql` is the data the app should keep for matching, booking, billing, ratings, events
and geography (requirement FR-LM-36 in `docs/SPEC.md`). It runs on PostgreSQL and SQLite; the
tests load it into SQLite and check that its states match the engine's.

## Queries

The original lastfirstdate pre-filter SQL query (Stage 1) belongs here as
`lastfirstdate_prefilter.sql`. It is not in this repo yet.

Until it is added, Stage 1 is implemented in Python (requirement FR-LM-04 in
`docs/SPEC.md`) and distance uses straight-line kilometres (decision D9). Once the query is
here, a test will check that the SQL and the Python filter keep the same pairs.
