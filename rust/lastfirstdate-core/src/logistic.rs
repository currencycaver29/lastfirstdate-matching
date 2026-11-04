//! MAP logistic regression by Newton's method (twin of `looks.fit_logistic`).
//!
//! Minimises
//!
//! ```text
//! sum_k s_k * logloss(y_k, sigmoid(o_k + x_k . w))  +  (l2 / 2) |w - m|^2
//! ```
//!
//! Each Newton step solves H step = g with
//!
//! ```text
//! g = sum_k s_k (p_k - y_k) x_k + l2 (w - m)
//! H = sum_k s_k p_k (1 - p_k) x_k x_k' + l2 I
//! ```
//!
//! and stops when no weight moves by more than `tol`. If a full step would make the
//! objective worse, it is halved until it doesn't (plain Newton can overshoot and diverge on
//! hard problems). Steps below `tol` are always taken in full and a relative rise below
//! [`OBJECTIVE_NOISE`] counts as rounding, so near the optimum, and whenever every full step
//! helps, the path is exactly plain Newton's. The
//! Laplace covariance, H^-1 at the optimum, says how unsure the weights are; calibration
//! uses it to pick the next photo.

use crate::linalg::{cholesky, cholesky_inverse, cholesky_solve};
use crate::{axpy, dot, sigmoid};

/// A relative rise in the objective this small is rounding noise, not an overshoot.
pub const OBJECTIVE_NOISE: f64 = 1e-12;

/// One fitting problem: n examples with d features each.
pub struct LogisticProblem<'a> {
    /// n x d features, row-major
    pub x: &'a [f64],
    pub n: usize,
    pub d: usize,
    /// labels, 0 or 1 (any value in [0, 1] works)
    pub y: &'a [f64],
    /// strength of the pull toward `prior_mean`
    pub l2: f64,
    /// m: the weights the penalty pulls toward (length d)
    pub prior_mean: &'a [f64],
    /// o_k added to each example's logit (length n)
    pub offset: &'a [f64],
    /// s_k: how much each example counts (length n)
    pub weight: &'a [f64],
}

#[derive(Debug, Clone, PartialEq)]
pub struct Fit {
    pub w: Vec<f64>,
    /// d x d Laplace covariance, row-major
    pub cov: Vec<f64>,
    pub steps: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FitError {
    BadShape(&'static str),
    NotPositiveDefinite,
}

impl std::fmt::Display for FitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FitError::BadShape(what) => write!(f, "bad shape: {what}"),
            FitError::NotPositiveDefinite => {
                write!(f, "the Hessian is not positive definite (is l2 > 0?)")
            }
        }
    }
}

impl std::error::Error for FitError {}

impl LogisticProblem<'_> {
    fn check(&self, start: &[f64]) -> Result<(), FitError> {
        let (n, d) = (self.n, self.d);
        if self.x.len() != n * d {
            return Err(FitError::BadShape("x must be n x d"));
        }
        if self.y.len() != n || self.offset.len() != n || self.weight.len() != n {
            return Err(FitError::BadShape(
                "y, offset and weight need one value per example",
            ));
        }
        if self.prior_mean.len() != d || start.len() != d {
            return Err(FitError::BadShape(
                "prior_mean and start need one value per feature",
            ));
        }
        Ok(())
    }

    /// Gradient and Hessian at w. Only the Hessian's lower triangle is filled, which is
    /// all the Cholesky factorisation reads.
    fn gradient_hessian(&self, w: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let d = self.d;
        let mut grad = vec![0.0; d];
        let mut hess = vec![0.0; d * d];
        for k in 0..self.n {
            let row = &self.x[k * d..(k + 1) * d];
            let p = sigmoid(self.offset[k] + dot(row, w));
            let g = self.weight[k] * (p - self.y[k]);
            let h = self.weight[k] * p * (1.0 - p);
            axpy(g, row, &mut grad);
            for a in 0..d {
                let ha = h * row[a];
                if ha != 0.0 {
                    axpy(ha, &row[..=a], &mut hess[a * d..a * d + a + 1]);
                }
            }
        }
        for a in 0..d {
            grad[a] += self.l2 * (w[a] - self.prior_mean[a]);
            hess[a * d + a] += self.l2;
        }
        (grad, hess)
    }

    /// The penalised log loss being minimised (logits clipped like [`sigmoid`]).
    pub fn objective(&self, w: &[f64]) -> f64 {
        let d = self.d;
        let mut total = 0.0;
        for k in 0..self.n {
            let z = (self.offset[k] + dot(&self.x[k * d..(k + 1) * d], w)).clamp(-35.0, 35.0);
            let softplus = z.max(0.0) + (-z.abs()).exp().ln_1p(); // log(1 + e^z)
            total += self.weight[k] * (softplus - self.y[k] * z);
        }
        let penalty: f64 = w
            .iter()
            .zip(self.prior_mean)
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        total + 0.5 * self.l2 * penalty
    }

    /// Fit from `start`, taking at most `max_steps` Newton steps, each halved at most
    /// `max_halvings` times.
    pub fn fit(
        &self,
        start: &[f64],
        max_steps: usize,
        tol: f64,
        max_halvings: usize,
    ) -> Result<Fit, FitError> {
        self.check(start)?;
        let d = self.d;
        let mut w = start.to_vec();
        let mut f = self.objective(&w);
        let mut trial = vec![0.0; d];
        let mut steps = 0;
        for _ in 0..max_steps {
            steps += 1;
            let (grad, hess) = self.gradient_hessian(&w);
            let l = cholesky(&hess, d).ok_or(FitError::NotPositiveDefinite)?;
            let step = cholesky_solve(&l, d, &grad);
            if step.iter().fold(0.0_f64, |m, s| m.max(s.abs())) < tol {
                for (wa, sa) in w.iter_mut().zip(&step) {
                    *wa -= sa;
                }
                break;
            }
            let mut t = 1.0;
            let mut f_trial = f;
            for _ in 0..max_halvings {
                for ((tr, wa), sa) in trial.iter_mut().zip(&w).zip(&step) {
                    *tr = wa - t * sa;
                }
                f_trial = self.objective(&trial);
                if f_trial <= f + OBJECTIVE_NOISE * (f.abs() + 1.0) {
                    break;
                }
                t *= 0.5;
            }
            w.copy_from_slice(&trial);
            f = f_trial;
        }
        let (_, hess) = self.gradient_hessian(&w);
        let l = cholesky(&hess, d).ok_or(FitError::NotPositiveDefinite)?;
        Ok(Fit {
            w,
            cov: cholesky_inverse(&l, d),
            steps,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy() -> (Vec<f64>, Vec<f64>) {
        // two features: intercept and a slope; likes rise with the slope feature
        let mut x = Vec::new();
        let mut y = Vec::new();
        for k in 0..40 {
            let t = (k as f64 - 20.0) / 10.0;
            x.extend([1.0, t]);
            y.push(if (k * 7) % 10 < 3 + k / 8 { 1.0 } else { 0.0 });
        }
        (x, y)
    }

    #[test]
    fn gradient_is_zero_at_the_optimum() {
        let (x, y) = toy();
        let ones = vec![1.0; 40];
        let zeros = vec![0.0; 40];
        let p = LogisticProblem {
            x: &x,
            n: 40,
            d: 2,
            y: &y,
            l2: 0.5,
            prior_mean: &[0.0, 0.0],
            offset: &zeros,
            weight: &ones,
        };
        let fit = p.fit(&[0.0, 0.0], 50, 1e-10, 30).unwrap();
        let (grad, _) = p.gradient_hessian(&fit.w);
        assert!(grad.iter().all(|g| g.abs() < 1e-8), "{grad:?}");
        assert!(fit.w[1] > 0.0, "likes rise with the feature");
        assert!(fit.cov[0] > 0.0 && fit.cov[3] > 0.0);
        assert_eq!(fit.cov[1], fit.cov[2]);
    }

    #[test]
    fn step_halving_converges_where_plain_newton_overshoots() {
        // one feature, perfectly separable labels, weak penalty and a far-off start: the
        // full Newton step from w = 8 overshoots to a worse point
        let x = [-2.0, -1.0, 1.0, 2.0];
        let y = [0.0, 0.0, 1.0, 1.0];
        let p = LogisticProblem {
            x: &x,
            n: 4,
            d: 1,
            y: &y,
            l2: 0.01,
            prior_mean: &[0.0],
            offset: &[0.0; 4],
            weight: &[1.0; 4],
        };
        let fit = p.fit(&[-8.0], 50, 1e-10, 30).unwrap();
        let (grad, _) = p.gradient_hessian(&fit.w);
        assert!(grad[0].abs() < 1e-8, "gradient {grad:?} at {:?}", fit.w);
        assert!(p.objective(&fit.w) < p.objective(&[-8.0]));
    }

    #[test]
    fn a_strong_prior_keeps_the_weights_at_the_prior() {
        let (x, y) = toy();
        let ones = vec![1.0; 40];
        let zeros = vec![0.0; 40];
        let p = LogisticProblem {
            x: &x,
            n: 40,
            d: 2,
            y: &y,
            l2: 1e9,
            prior_mean: &[0.3, -0.2],
            offset: &zeros,
            weight: &ones,
        };
        let fit = p.fit(&[0.0, 0.0], 50, 1e-12, 30).unwrap();
        assert!((fit.w[0] - 0.3).abs() < 1e-6 && (fit.w[1] + 0.2).abs() < 1e-6);
    }

    #[test]
    fn shapes_are_checked() {
        let p = LogisticProblem {
            x: &[1.0, 2.0, 3.0],
            n: 2,
            d: 2,
            y: &[0.0, 1.0],
            l2: 1.0,
            prior_mean: &[0.0, 0.0],
            offset: &[0.0, 0.0],
            weight: &[1.0, 1.0],
        };
        assert!(matches!(
            p.fit(&[0.0, 0.0], 5, 1e-8, 30),
            Err(FitError::BadShape(_))
        ));
    }
}
