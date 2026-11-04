# Shared: settings and data

What every stage reads: the settings files, the member model and the app's database.

## Settings (rule R5: every tunable number lives here)

| File | Use |
|---|---|
| `config/lastfirstdate.toml` | Research defaults; every Sprint 1 to 5 result uses these |
| `config/phase1.toml` | lastfirstdate's Phase 1 rules (D14): one zone, 100 paying men, 300 women, a three-date cap, 30 calibration photos, closeness intent questions, prompts |
| `config/forecast_model.toml` | The fitted guarantee forecast (written by `forecast_fit.py`) |

- **Loading:** `config.load(path)` returns a frozen `Config`. Each `[section]` is a
  dataclass, and `validate` checks every range.
- **Adding a setting:** put it in **both** TOML files, add the dataclass field, and add a range
  check. Unknown or missing keys are errors.

## Member model

- **Profiles:** `models.Member`, one per profile. It has no attractiveness or rank field, by
  design (R2).
- **Weekly slots:** slot = weekday × 3 + part (weekday 0 is Monday; parts are lunch, evening,
  late).
- **Arrays:** `arrays.MemberArrays` holds profiles as NumPy arrays. Sets are bit masks, so
  "shared free time" is `slots[i] & slots[j]`.
- **Photo embeddings:** a separate matrix indexed by member id, so they can be encrypted and
  deleted on their own (R3).

## The app's database

- **The design:** `db/schema.sql` (FR-LM-36) is the data the app should keep. It runs on
  PostgreSQL and SQLite and has checks for the rules.
- **Keeping it in step:** booking states and sources must match `booking.STATES` and
  `booking.SOURCES` (tested in `tests/test_sprint7.py`).
- **Still missing:** the original pre-filter SQL query belongs in `db/` (see `db/README.md`).

## Tests

FR-LM-01 (`tests/test_config.py`), FR-LM-02 (`tests/test_models.py`), FR-LM-29
(`tests/test_phase1.py`), FR-LM-36 (`tests/test_sprint7.py`).
