//! Python bindings for `lastfirstdate-core`, imported as `lastfirstdate._core`.
//!
//! Thin wrappers only: check shapes, borrow NumPy arrays without copying, call the core,
//! return NumPy arrays. The Python modules decide when to call these (see `native.py`).

use lastfirstdate_core::{calibration, gate, lists, logistic};
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

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(fit_logistic, m)?)?;
    m.add_function(wrap_pyfunction!(quad_forms, m)?)?;
    m.add_function(wrap_pyfunction!(best_pair, m)?)?;
    m.add_function(wrap_pyfunction!(fair_day_lists, m)?)?;
    m.add_function(wrap_pyfunction!(gate_scan, m)?)?;
    Ok(())
}
