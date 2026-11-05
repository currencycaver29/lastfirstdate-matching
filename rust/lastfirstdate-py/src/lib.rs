//! Python bindings for `lastfirstdate-core`, imported as `lastfirstdate._core`.
//!
//! Thin wrappers only: check shapes, borrow NumPy arrays without copying, call the core,
//! return NumPy arrays. The Python modules decide when to call these (see `native.py`).

use lastfirstdate_core::{calibration, gate, lists, logistic, windows};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray, PyReadonlyArray1,
    PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// (weights, Laplace covariance)
type FitResult<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>);
/// (admitted columns, deferred columns, women per man, forecast, why it stopped)
type GateResult = (Vec<usize>, Vec<usize>, f64, f64, &'static str);

fn contiguous<'a, T: numpy::Element, D: numpy::ndarray::Dimension>(
    a: &'a PyReadonlyArray<'_, T, D>,
    name: &str,
) -> PyResult<&'a [T]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a C-contiguous array")))
}

/// MAP logistic regression by Newton's method. Returns (weights, Laplace covariance).
#[pyfunction]
#[pyo3(signature = (x, y, l2, prior_mean, offset, weight, start, max_steps, tol, max_halvings))]
#[allow(clippy::too_many_arguments)]
fn fit_logistic<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    l2: f64,
    prior_mean: PyReadonlyArray1<'py, f64>,
    offset: PyReadonlyArray1<'py, f64>,
    weight: PyReadonlyArray1<'py, f64>,
    start: PyReadonlyArray1<'py, f64>,
    max_steps: usize,
    tol: f64,
    max_halvings: usize,
) -> PyResult<FitResult<'py>> {
    let (n, d) = (x.shape()[0], x.shape()[1]);
    let problem = logistic::LogisticProblem {
        x: contiguous(&x, "x")?,
        n,
        d,
        y: contiguous(&y, "y")?,
        l2,
        prior_mean: contiguous(&prior_mean, "prior_mean")?,
        offset: contiguous(&offset, "offset")?,
        weight: contiguous(&weight, "weight")?,
    };
    let fit = problem
        .fit(contiguous(&start, "start")?, max_steps, tol, max_halvings)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let cov = fit.cov.into_pyarray(py).reshape([d, d])?;
    Ok((fit.w.into_pyarray(py), cov))
}

/// x_i' cov x_i for each row of x.
#[pyfunction]
fn quad_forms<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    cov: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (n, d) = (x.shape()[0], x.shape()[1]);
    if cov.shape() != [d, d] {
        return Err(PyValueError::new_err("cov must be d x d"));
    }
    let out = calibration::quad_forms(contiguous(&x, "x")?, n, d, contiguous(&cov, "cov")?);
    Ok(out.into_pyarray(py))
}

/// Positions (a, b) of the most informative pair question, or None.
#[pyfunction]
fn best_pair(
    features: PyReadonlyArray2<'_, f64>,
    logits: PyReadonlyArray1<'_, f64>,
    genders: PyReadonlyArray1<'_, i64>,
    cov: PyReadonlyArray2<'_, f64>,
) -> PyResult<Option<(usize, usize)>> {
    let (c, d) = (features.shape()[0], features.shape()[1]);
    if logits.len() != c || genders.len() != c || cov.shape() != [d, d] {
        return Err(PyValueError::new_err(
            "one logit and gender per candidate; cov d x d",
        ));
    }
    Ok(calibration::best_pair(
        contiguous(&features, "features")?,
        c,
        d,
        contiguous(&logits, "logits")?,
        contiguous(&genders, "genders")?,
        contiguous(&cov, "cov")?,
    ))
}

/// Round-robin lists under an exposure cap. Returns (lists, cap); lists are per chooser
/// row, in the rows' order.
#[pyfunction]
fn fair_day_lists(
    available: PyReadonlyArray2<'_, bool>,
    scores: PyReadonlyArray2<'_, f64>,
    bonus: PyReadonlyArray1<'_, f64>,
    size: usize,
    multiple: f64,
    reserved: Vec<Vec<usize>>,
) -> PyResult<(Vec<Vec<usize>>, usize)> {
    let (m, n) = (available.shape()[0], available.shape()[1]);
    if scores.shape() != [m, n] || bonus.len() != n || reserved.len() != m {
        return Err(PyValueError::new_err(
            "available and scores must be choosers x members, one bonus per member, \
             one reserved list per chooser",
        ));
    }
    if reserved.iter().flatten().any(|&j| j >= n) {
        return Err(PyValueError::new_err(
            "a reserved id is outside the member range",
        ));
    }
    let input = lists::DayInput {
        available: contiguous(&available, "available")?,
        scores: contiguous(&scores, "scores")?,
        m,
        n,
        bonus: contiguous(&bonus, "bonus")?,
        size,
        multiple,
        reserved: &reserved,
    };
    let out = lists::fair_day_lists(&input);
    Ok((out.lists, out.cap))
}

/// The ratio gate. Returns (admitted columns, deferred columns, women per man, forecast,
/// why it stopped).
#[pyfunction]
#[pyo3(signature = (eligible, n_current, coef, min_pool, min_women_per_man, fitted_min, target))]
fn gate_scan(
    eligible: PyReadonlyArray2<'_, bool>,
    n_current: usize,
    coef: PyReadonlyArray1<'_, f64>,
    min_pool: f64,
    min_women_per_man: f64,
    fitted_min: f64,
    target: f64,
) -> PyResult<GateResult> {
    let (w, c) = (eligible.shape()[0], eligible.shape()[1]);
    if n_current > c || coef.len() != gate::N_COEF {
        return Err(PyValueError::new_err(format!(
            "n_current must not exceed the columns, and the forecast needs {} coefficients",
            gate::N_COEF
        )));
    }
    let input = gate::GateInput {
        eligible: contiguous(&eligible, "eligible")?,
        n_women: w,
        n_cols: c,
        n_current,
        coef: contiguous(&coef, "coef")?,
        min_pool,
        min_women_per_man,
        fitted_min_women_per_man: fitted_min,
        target,
    };
    let out = gate::gate_scan(&input);
    let stop = match out.stop {
        gate::Stop::WaitlistEmpty => "waitlist empty",
        gate::Stop::Ratio => "ratio",
        gate::Stop::OutsideFittedRange => "fitted",
        gate::Stop::Forecast => "forecast",
    };
    Ok((
        out.admit,
        out.deferred,
        out.women_per_man,
        out.forecast,
        stop,
    ))
}

/// Free date windows over the horizon, one u64 of bits per member.
#[pyfunction]
#[pyo3(signature = (weekly, today, lead, horizon, blocked=None))]
fn window_bits<'py>(
    py: Python<'py>,
    weekly: PyReadonlyArray1<'py, i64>,
    today: i64,
    lead: usize,
    horizon: usize,
    blocked: Option<PyReadonlyArray2<'py, bool>>,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    if !(lead < horizon && horizon <= 21) {
        return Err(PyValueError::new_err(
            "need 0 <= lead_days < horizon_days <= 21",
        ));
    }
    let slots = contiguous(&weekly, "weekly")?;
    let out = match &blocked {
        Some(b) => {
            if b.shape() != [slots.len(), horizon] {
                return Err(PyValueError::new_err("blocked must be members x horizon"));
            }
            windows::window_bits(slots, today, lead, horizon, Some(contiguous(b, "blocked")?))
        }
        None => windows::window_bits(slots, today, lead, horizon, None),
    };
    Ok(out.into_pyarray(py))
}

/// The earliest shared window of each pair (index into the bits), or -1. Either side may
/// be a single member, matched against every member on the other side.
#[pyfunction]
fn earliest_common<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, u64>,
    b: PyReadonlyArray1<'py, u64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let (a, b) = (contiguous(&a, "a")?, contiguous(&b, "b")?);
    let out: Vec<i64> = match (a.len(), b.len()) {
        (1, _) => b
            .iter()
            .map(|&y| windows::earliest_common(a[0], y))
            .collect(),
        (_, 1) => a
            .iter()
            .map(|&x| windows::earliest_common(x, b[0]))
            .collect(),
        (n, m) if n == m => a
            .iter()
            .zip(b)
            .map(|(&x, &y)| windows::earliest_common(x, y))
            .collect(),
        _ => {
            return Err(PyValueError::new_err(
                "a and b must match in length, or one be 1",
            ));
        }
    };
    Ok(out.into_pyarray(py))
}

/// (members x horizon) days a member cannot date. Members are positions 0..m in `limit`.
#[pyfunction]
#[pyo3(signature = (busy_member, busy_day, week_member, week, week_count, period_member,
                    period_start, period_count, today, horizon, max_per_week, trial_days,
                    cycle_days, limit, signup))]
#[allow(clippy::too_many_arguments)]
fn blocked_days<'py>(
    py: Python<'py>,
    busy_member: PyReadonlyArray1<'py, i64>,
    busy_day: PyReadonlyArray1<'py, i64>,
    week_member: PyReadonlyArray1<'py, i64>,
    week: PyReadonlyArray1<'py, i64>,
    week_count: PyReadonlyArray1<'py, i64>,
    period_member: PyReadonlyArray1<'py, i64>,
    period_start: PyReadonlyArray1<'py, i64>,
    period_count: PyReadonlyArray1<'py, i64>,
    today: i64,
    horizon: usize,
    max_per_week: i64,
    trial_days: i64,
    cycle_days: i64,
    limit: PyReadonlyArray1<'py, i64>,
    signup: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let limit_s = contiguous(&limit, "limit")?;
    let m = limit_s.len();
    let rows = |a: &PyReadonlyArray1<'py, i64>, name: &str| -> PyResult<Vec<usize>> {
        contiguous(a, name)?
            .iter()
            .map(|&r| {
                usize::try_from(r)
                    .ok()
                    .filter(|&r| r < m)
                    .ok_or_else(|| PyValueError::new_err(format!("{name} is out of range")))
            })
            .collect()
    };
    let (bm, wm, pm) = (
        rows(&busy_member, "busy_member")?,
        rows(&week_member, "week_member")?,
        rows(&period_member, "period_member")?,
    );
    let (bd, wk, wc) = (
        contiguous(&busy_day, "busy_day")?,
        contiguous(&week, "week")?,
        contiguous(&week_count, "week_count")?,
    );
    let (ps, pc) = (
        contiguous(&period_start, "period_start")?,
        contiguous(&period_count, "period_count")?,
    );
    let signup_s = contiguous(&signup, "signup")?;
    if bd.len() != bm.len()
        || wk.len() != wm.len()
        || wc.len() != wm.len()
        || ps.len() != pm.len()
        || pc.len() != pm.len()
        || signup_s.len() != m
        || cycle_days <= 0
    {
        return Err(PyValueError::new_err(
            "blocked_days: mismatched lengths or cycle_days",
        ));
    }
    let taken = windows::Taken {
        busy_member: &bm,
        busy_day: bd,
        week_member: &wm,
        week: wk,
        week_count: wc,
        period_member: &pm,
        period_start: ps,
        period_count: pc,
    };
    let rules = windows::Rules {
        today,
        horizon,
        max_per_week,
        trial_days,
        cycle_days,
        limit: limit_s,
        signup: signup_s,
    };
    let out = windows::blocked_days(&taken, &rules);
    out.into_pyarray(py).reshape([m, horizon])
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(fit_logistic, m)?)?;
    m.add_function(wrap_pyfunction!(quad_forms, m)?)?;
    m.add_function(wrap_pyfunction!(best_pair, m)?)?;
    m.add_function(wrap_pyfunction!(fair_day_lists, m)?)?;
    m.add_function(wrap_pyfunction!(gate_scan, m)?)?;
    m.add_function(wrap_pyfunction!(window_bits, m)?)?;
    m.add_function(wrap_pyfunction!(earliest_common, m)?)?;
    m.add_function(wrap_pyfunction!(blocked_days, m)?)?;
    Ok(())
}
