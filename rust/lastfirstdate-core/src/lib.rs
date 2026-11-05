//! Rust core of lastfirstdate's matching engine (D16, FR-LM-35).
//!
//! The Python package keeps every algorithm in readable NumPy form; this crate holds the
//! same algorithms for the loops where Python's per-call overhead dominates. Each function
//! here has a Python twin, and the test suite checks that both give the same answers.
//!
//! Matrices are dense, row-major `f64` slices with their sizes passed alongside.
//!
//! - [`linalg`]: Cholesky factorisation, solves and inverses for small symmetric matrices.
//! - [`logistic`]: MAP logistic regression by Newton's method, with the Laplace covariance
//!   (Stage 3 looks taste, the guarantee forecast, learning from dates).
//! - [`calibration`]: uncertainty of many candidates at once and the best pair question
//!   (Stage 2).
//! - [`lists`]: round-robin daily lists under an exposure cap (Stage 6).
//! - [`gate`]: the ratio gate, updated incrementally as men are admitted (FR-LM-34).
//! - [`windows`]: free date windows, the first shared one, and blocked days (FR-LM-37).

pub mod calibration;
pub mod gate;
pub mod linalg;
pub mod lists;
pub mod logistic;
pub mod windows;

/// The logistic function, with the input clipped to [-35, 35] exactly as in Python.
pub fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z.clamp(-35.0, 35.0)).exp())
}

/// Dot product of two equal-length slices.
///
/// Four running sums instead of one, so the compiler can use SIMD (several numbers per
/// instruction). The result can differ from a one-sum loop in the last binary digit.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0; 4];
    let (ca, cb) = (a.chunks_exact(4), b.chunks_exact(4));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for k in 0..4 {
            acc[k] += x[k] * y[k];
        }
    }
    let tail: f64 = ra.iter().zip(rb).map(|(x, y)| x * y).sum();
    (acc[0] + acc[2]) + (acc[1] + acc[3]) + tail
}

/// y += alpha * x, element by element (vectorises without changing the rounding).
#[inline]
pub fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += alpha * xi;
    }
}
