# The Rust core (after Sprint 6)

**In one paragraph.** You asked for Rust wherever it fits. Profiling showed exactly where it
does: four loops where the engine spent its time on Python's per-call overhead, not on
arithmetic. Building every member's looks taste, the logistic fit behind Stages 2 and 3, was
the biggest, at 89 seconds for the 1,000-member city. Those four loops now run in Rust,
called from Python through PyO3. The results: looks tastes **3.6× faster** (89 s → 25 s),
daily lists **10× faster**, and the ratio gate **28× to 81× faster**. The whole test suite
dropped from 4.5 to 2.5 minutes. The answers are the same: the 50/50 scores and every
synthetic booking are byte-for-byte identical to the pure-Python engine. Writing the Rust
twin also found a weakness that both versions shared, now fixed (section 6).

**What wasn't moved, and why.** The scheduler already runs in C++ (OR-Tools), the NumPy
parts are already compiled code, and the test harness is for testing only; porting it would
break its seeded random streams. Moving those would add risk and gain nothing.

---

## 1. What got built

| Where | What |
|---|---|
| `rust/lastfirstdate-core/` | Pure Rust, no Python: the logistic fit, the calibration kernels, the fair daily lists and the gate, with 16 tests of its own. A future Rust service could reuse it as is |
| `rust/lastfirstdate-py/` | PyO3 bindings: borrow the NumPy arrays without copying, check shapes, call the core, return NumPy arrays. Imported as `lastfirstdate._core` |
| `src/lastfirstdate/native.py` | Decides when the engine calls Rust: whenever the compiled module is present. `LASTFIRSTDATE_BACKEND=python` turns it off |
| `looks.py`, `fairness.py`, `forecast.py` | Each ported function calls Rust when it can and keeps its NumPy version as the readable reference |
| `pyproject.toml` | The build tool is now maturin, so `pip install -e .` compiles the Rust core |
| `scripts/check.sh`, `.github/workflows/checks.yml` | The gates now also run `cargo fmt`, Clippy (Rust's linter, warnings are errors), the Rust tests, and a fresh build of the module |
| `tests/test_native.py` | FR-LM-35: the two versions must agree |

About 1,200 lines of Rust, with **no `unsafe` code**: every array access is bounds-checked,
and a wrong shape becomes a Python `ValueError`, not a crash.

---

## 2. How Python calls Rust

```
Python: fit_logistic(x, y, ...)                     looks.py
          │  native.enabled()?  yes
          ▼
PyO3:   _core.fit_logistic(x, y, ...)                lastfirstdate-py/src/lib.rs
          │  borrows the NumPy arrays' memory (no copy), checks shapes
          ▼
Rust:   LogisticProblem { x, y, ... }.fit(...)       lastfirstdate-core/src/logistic.rs
          │  returns Vec<f64>
          ▼
PyO3:   turns the vectors into NumPy arrays (no copy) and returns them
```

`maturin` compiles the Rust crate into a file Python can import
(`_core.cpython-313-x86_64-linux-gnu.so`) and puts it next to the Python modules. To rebuild
after editing Rust: `.venv/bin/maturin develop --release --skip-install`.

---

## 3. Where the time went, and why Rust helps there

The first profile (1,000 members, building every looks taste):

| Function | Calls | Time | Per call |
|---|---:|---:|---:|
| the uncertainty s² = x′Σx for all remaining candidates (`einsum`) | 23,349 | 55 s | 2.4 ms |
| the logistic fit (Newton's method) | 24,134 | 27 s | 1.1 ms |

Each call works on a 65 × 65 matrix (64 photo features plus an intercept). NumPy is fast on
big matrices, but on small ones every call pays Python overhead: building temporary arrays,
dispatching to BLAS, checking types. Rust compiles the whole loop into machine code, so a
call costs only its arithmetic.

**A lesson along the way: a straight port wasn't enough.** The first Rust version was a
line-by-line translation and gained only 1.4×. Rust removed the overhead, but my loops did
more arithmetic than needed. Three changes in the maths did the rest:

### The inverse (Laplace covariance): 6× less work

The fit needs Σ = H⁻¹, where H is the Hessian. H is symmetric positive definite, so it
factors as H = L Lᵀ (Cholesky) with L lower-triangular. Then:

    H⁻¹ = (L Lᵀ)⁻¹ = L⁻ᵀ L⁻¹

Inverting a triangle is cheap: row i of L⁻¹ only needs the rows above it. That plus the
product L⁻ᵀL⁻¹ costs about d³/3 ≈ 92,000 multiplications. The first port solved H x = eᵢ
for each of the 65 columns of the identity, costing about 2d³ ≈ 550,000.

### The uncertainty: half the work, using symmetry

    x′Σx = Σ_a Σ_aa x_a²  +  2 Σ_b x_b t_b,     where  t_b = Σ_{a<b} x_a Σ_ab

Σ is symmetric (Σ_ab = Σ_ba), so each off-diagonal pair is counted once and doubled. That is
d²/2 multiplications instead of d².

### Loops the processor can run 2 numbers at a time

Modern processors have SIMD instructions that add or multiply several numbers at once. A loop
like `y[b] += a · x[b]` is safe to run that way, because each `y[b]` is independent and the
rounding is unchanged. A running sum `s += x[b] · y[b]` is not, because every step needs the
previous one. So the kernels are written as `y += a·x` loops where possible. Dot products use
four separate running sums that are combined at the end.

| Version | Looks tastes, 1,000 members |
|---|---:|
| Python (NumPy) | 89 s |
| first Rust port, a direct translation | 54 s |
| + the three changes above (final version) | **25 to 30 s** (runs vary by about 15%) |
| experiment: AVX2, 4 numbers at a time | 26 s (not adopted) |

AVX2 ran in 26.4 s against 30.2 s for the run just before it, but a later normal build ran in
25.1 s, so its gain is within run-to-run noise. It would need CPU-specific code and a
fallback for older processors, so it isn't worth it. The remaining time is mostly Python
bookkeeping between calls; section 8 says what would remove it.

---

## 4. The fair daily lists and the gate

**Daily lists** (Stage 6): for every chooser, sort her candidates by score plus bonus, then
fill everyone's lists one slot at a time under the exposure cap. This is plain loops over
ints, which Python runs slowly and Rust quickly. A month's lists went from **0.64 s to
0.06 s**. That matters in production, where this runs every day for every member.

**The gate** (FR-LM-34): the Python version recomputes every man's attention share each time
it considers an applicant, costing women × men per applicant. The Rust version updates the
shares in place. When man a joins, each woman i in his pool goes from c_i to c_i + 1
possible men, so every man already in the market who could meet her loses

    1/c_i − 1/(c_i + 1)

of his share. Only those men are touched. Measured on random markets where each woman can
meet 30% of the men:

| Zone | Python | Rust | Same decision? |
|---|---:|---:|---|
| 300 women, 200 waiting | 0.09 s | 3 ms | yes |
| 1,500 women, 1,000 waiting | 8.6 s | 0.11 s | yes |
| 3,000 women, 2,000 waiting | 86 s | 1.5 s | yes |

For a city-sized waitlist, Python would take over a minute per sale; Rust answers in about a
second.

---

## 5. Proving the answers are the same

Two versions of the same algorithm are a strong check on each other (differential testing).
`tests/test_native.py` runs each function both ways on random problems:

| Check | Result |
|---|---|
| logistic fit, 6 random problems with offsets, weights, priors, warm starts | weights and covariance agree to 1 part in 10⁹ |
| uncertainty of 300 candidates | agree to 1 part in 10¹¹ |
| pair question, 5 random tastes | the same pair every time |
| daily lists, 5 random days with ties and reserved slots | identical lists and cap |
| the gate, every stopping rule, with and without men already admitted | identical admissions, deferrals and reason |
| 12 members' calibration in the default city | the same photos shown, in the same order |
| a synthetic month | exactly the same dates booked |

And on the full 1,000-member city, compared with results saved *before* any Rust existed:
the looks scores differ by at most 9 × 10⁻¹⁶ (rounding), and the 50/50 mutual scores and
every booking are **identical**. So every number in the Sprint 1–6 notes still holds.

---

## 6. What the twin found: Newton's method can run away

The first equivalence test failed on a random problem. Investigating showed that the bug was
in **both** versions. Newton's method takes the step that would be exact if the objective
were a perfect bowl:

    w ← w − H⁻¹ g

On a hard problem, the step overshoots and lands somewhere worse. Each step was bigger than
the last (1.8, 1.2, 1.6, 2.1, 4.0, 7.2, 13, 27…), and after 50 steps both versions returned
garbage, each a different garbage.

**Does this happen in the real engine?** I counted Newton steps in all 9,444 calibration
fits of a 400-member city: 3 to 6 steps each (median 4), never close to the limit of 50.
So no result was affected. But real data can be harder, for example refitting the forecast
on real cycles, and a silent divergence would quietly give a bad model.

**The fix: step halving.** If a full step would make the objective (the penalised log loss)
worse, try half the step, then a quarter, and so on. Two details matter:

- Near the optimum, the gain from the last tiny step is smaller than rounding error, so the
  comparison is noise. A first version of the fix sometimes refused that last step and
  stopped about 10⁻⁸ short. Now a step below the stopping tolerance is always taken, and a
  rise in the objective of less than one part in 10¹² counts as rounding.
- When every full step helps, which covers all 9,444 real fits, the path is exactly plain
  Newton's. That's why the results stayed identical.

Both versions now reach the optimum on the problem that diverged (a test checks the gradient
is zero there).

---

## 7. Honest limits

- **Installing needs Rust** (`rustup`), because `pip install` compiles the core. Without
  Rust, `pip install` fails, but the code still runs in pure Python straight from the source
  folder (for example with `PYTHONPATH=src`), since the engine falls back when the compiled
  module is missing.
- **Single-threaded.** Each call uses one core. That suits a server handling many members at
  once; for batch jobs, members could be spread over cores later.
- **Rounding order differs** from NumPy in the last binary digit. Here no decision changed,
  but on another machine a near-perfect tie could break the other way. NumPy already behaves
  like this across machines.
- **Python still runs the calibration loop itself** (which photo next, recording answers).
  In the profile, roughly a third of the remaining time is that bookkeeping.

---

## 8. If you want more Rust later

| Next step | Gain | Effort |
|---|---|---|
| The whole calibration loop in Rust (Stage 2 end to end) | my estimate: 25 s → about 15 s | medium: ports the photo variety pick (DPP) and the starting-taste offsets |
| A Rust service (`axum`) that serves daily lists and gate decisions to the app | production speed and memory safety on the hot path | larger: needs the app's database (Sprint 7) |
| The min-cost-flow scheduler in Rust | none in speed (OR-Tools is C++), but a good learning project, checked against the existing tests | medium |

---

## 9. What to study

1. **Rust basics used here:** ownership and borrowing (`&[f64]` borrows a slice, `&mut`
   lets a function change it), `Option` and `Result` instead of exceptions, iterators like
   `zip`, and why bounds checks make out-of-range reads impossible.
2. **PyO3:** how `#[pyfunction]` turns a Rust function into a Python one, and how
   `PyReadonlyArray2<f64>` borrows a NumPy array without copying.
3. **Cholesky factorisation:** why a symmetric positive definite matrix factors as L Lᵀ,
   and why solving with triangles is cheap.
4. **Counting operations:** d³ versus d³/3 matters more than the language. The first port
   was in Rust and still slow.
5. **SIMD intuition:** why `y += a·x` vectorises and a running sum doesn't.
6. **Incremental algorithms:** updating only what changed (the gate) instead of recomputing
   everything.
7. **Differential testing:** two independent versions checking each other catch bugs
   neither would catch alone, as section 6 shows.

## 10. Round 2: the date windows (v0.13.1)

After Sprint 13, profiling the app's calls and the test runs found one more place where Python
loops cost real time: working out which evenings two members are both free. It runs for every
candidate on every list, at every pick and in every test run.

| Python | Rust | What it does |
|---|---|---|
| `booking.window_bits` | `_core.window_bits` | each member's free windows over the next two weeks, as bits |
| `booking.earliest_common` | `_core.earliest_common` | the first window two members share |
| `app._blocked_days` | `_core.blocked_days` | days a member can't date: a date that day, a full week, or a full cap period |

**The trick in `earliest_common`.** The free windows are one 64-bit number per member, one bit
per (day, part of day). The windows two members share are `a & b`, and the earliest one is the
lowest set bit. NumPy finds it with six passes (`x & (~x + 1)`, a conversion to floating point,
a base-2 logarithm, rounding, a comparison); Rust asks the processor directly:
`trailing_zeros` is a single instruction.

**Measured** (medians; this machine is shared, so single runs vary by about 20%):

| | Python | Rust | Faster |
|---|---:|---:|---:|
| `window_bits`, two members (one call) | 60.8 µs | 0.8 µs | 75× |
| `earliest_common`, one against 200 (one call) | 9.1 µs | 3.1 µs | 3× |
| `earliest_common`, one pair (one call) | 6.5 µs | 2.8 µs | 2× |
| The app's daily lists, 400 members | 0.14 s | 0.07 s | 2.0× |
| The app's daily lists, 2,000 members | 0.56 s | 0.35 s | 1.6× |
| Test runs, three billing cycles, 400 members | 2.74 s | 2.17 s | 1.3× |

**A lesson in measuring.** The first comparison of the test runs said Rust was *slower*. It had
run first, with cold caches. Alternating the two versions four times showed the real 1.3×.

**Proof the answers are the same:** random inputs for each function (negative days, empty
windows, every input shape), and two end-to-end checks: a full day of the app's lists (windows,
positions and match values) and a billing cycle of test runs come out identical with Rust and
with Python (`tests/test_native.py`).

**What's left in Python on purpose:** the photo exercise's bookkeeping (about 20% of a test
zone's setup; members already get each answer in milliseconds) and the test harness's day loop
(test-only code). Neither is worth a second copy to maintain yet.

---

## Try it yourself

```bash
.venv/bin/python -m pytest tests/test_native.py -q                 # the twins agree
LASTFIRSTDATE_BACKEND=python .venv/bin/python -m lastfirstdate.looks_report   # pure Python
.venv/bin/python -m lastfirstdate.looks_report                     # with Rust: compare the times
cargo test --manifest-path rust/Cargo.toml -p lastfirstdate-core   # the Rust crate's own tests
```
