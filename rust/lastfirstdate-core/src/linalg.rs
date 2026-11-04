//! Small dense linear algebra for symmetric positive definite matrices.
//!
//! The looks model's Hessian is H = X' W X + l2 I. With l2 > 0 it is symmetric positive
//! definite, so it factors as H = L L' (Cholesky) with L lower triangular. Solving H x = b
//! is then two triangular solves, and the inverse (the Laplace covariance) is L^-T L^-1.
//! Cholesky needs about half the work of the general LU solve NumPy uses, and fails loudly
//! if H is not positive definite.

use crate::{axpy, dot};

/// Cholesky factor L (row-major, lower triangle filled) of a d x d symmetric positive
/// definite matrix. Only the lower triangle of `a` is read. `None` if `a` is not
/// positive definite.
pub fn cholesky(a: &[f64], d: usize) -> Option<Vec<f64>> {
    assert_eq!(a.len(), d * d, "matrix must be d x d");
    let mut l = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..=i {
            // L[i][j] uses the first j entries of rows i and j
            let s = a[i * d + j] - dot(&l[i * d..i * d + j], &l[j * d..j * d + j]);
            if i == j {
                if s <= 0.0 || !s.is_finite() {
                    return None;
                }
                l[i * d + i] = s.sqrt();
            } else {
                l[i * d + j] = s / l[j * d + j];
            }
        }
    }
    Some(l)
}

/// Solve (L L') x = b, given the Cholesky factor L.
pub fn cholesky_solve(l: &[f64], d: usize, b: &[f64]) -> Vec<f64> {
    let mut y = b.to_vec();
    // forward: L z = b
    for i in 0..d {
        y[i] = (y[i] - dot(&l[i * d..i * d + i], &y[..i])) / l[i * d + i];
    }
    // backward: L' x = z
    for i in (0..d).rev() {
        let mut s = y[i];
        for k in i + 1..d {
            s -= l[k * d + i] * y[k];
        }
        y[i] = s / l[i * d + i];
    }
    y
}

/// The inverse of (L L'), returned exactly symmetric.
///
/// (L L')^-1 = L^-T L^-1. First invert the triangle L (row i of L^-1 is
/// (e_i - sum_{k<i} L[i][k] * row k of L^-1) / L[i][i]), then multiply. Both steps cost
/// about d^3 / 6 and are written as y += a * x loops, which vectorise. Solving for the d
/// columns of the identity instead would cost about 2 d^3.
pub fn cholesky_inverse(l: &[f64], d: usize) -> Vec<f64> {
    let mut li = vec![0.0; d * d]; // L^-1, lower triangular
    let mut acc = vec![0.0; d];
    for i in 0..d {
        acc[..=i].fill(0.0);
        acc[i] = 1.0;
        for k in 0..i {
            let c = l[i * d + k];
            if c != 0.0 {
                axpy(-c, &li[k * d..k * d + k + 1], &mut acc[..=k]);
            }
        }
        let inv_diag = 1.0 / l[i * d + i];
        for (out, v) in li[i * d..i * d + i + 1].iter_mut().zip(&acc[..=i]) {
            *out = v * inv_diag;
        }
    }
    // inverse = sum over k of (row k of L^-1)' (row k of L^-1); fill the lower triangle
    let mut inv = vec![0.0; d * d];
    for k in 0..d {
        let row = &li[k * d..k * d + k + 1];
        for (a, ra) in row.iter().enumerate() {
            if *ra != 0.0 {
                axpy(*ra, &row[..=a], &mut inv[a * d..a * d + a + 1]);
            }
        }
    }
    for r in 0..d {
        for c in r + 1..d {
            inv[r * d + c] = inv[c * d + r];
        }
    }
    inv
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: [f64; 9] = [4.0, 2.0, 0.6, 2.0, 5.0, 1.0, 0.6, 1.0, 3.0];

    #[test]
    fn solve_recovers_the_right_hand_side() {
        let l = cholesky(&A, 3).unwrap();
        let x = cholesky_solve(&l, 3, &[1.0, -2.0, 0.5]);
        for (r, want) in [1.0, -2.0, 0.5].iter().enumerate() {
            let got: f64 = (0..3).map(|c| A[r * 3 + c] * x[c]).sum();
            assert!((got - want).abs() < 1e-12);
        }
    }

    #[test]
    fn inverse_times_matrix_is_identity() {
        let inv = cholesky_inverse(&cholesky(&A, 3).unwrap(), 3);
        for r in 0..3 {
            for c in 0..3 {
                let v: f64 = (0..3).map(|k| A[r * 3 + k] * inv[k * 3 + c]).sum();
                assert!((v - if r == c { 1.0 } else { 0.0 }).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn not_positive_definite_is_refused() {
        assert!(cholesky(&[1.0, 2.0, 2.0, 1.0], 2).is_none());
    }
}
