//! Calibration kernels (twins of `PersonalTaste.uncertainty` and `looks.select_pair`).
//!
//! Stage 2 picks the next photo (or pair) the model would learn most from. Both need the
//! model's uncertainty about a logit, s^2 = x' cov x, for many vectors x at once.

use crate::{axpy, dot, sigmoid};

/// x' cov x for one vector, for a symmetric cov. `t` is scratch space of length d.
///
/// Uses the symmetry: x' cov x = sum_a cov_aa x_a^2 + 2 sum_b x_b t_b, where
/// t_b = sum_{a<b} x_a cov_ab is built row by row with y += a * x loops. That is half the
/// multiplications of the plain double sum, and the loops vectorise.
pub fn quad_form(x: &[f64], cov: &[f64], d: usize, t: &mut [f64]) -> f64 {
    t.fill(0.0);
    let mut diagonal = 0.0;
    for (a, &xa) in x.iter().enumerate() {
        if xa == 0.0 {
            continue;
        }
        diagonal += cov[a * d + a] * xa * xa;
        axpy(xa, &cov[a * d + a + 1..(a + 1) * d], &mut t[a + 1..d]);
    }
    diagonal + 2.0 * dot(x, t)
}

/// s_i^2 = x_i' cov x_i for each of the n rows of x (n x d); cov must be symmetric.
pub fn quad_forms(x: &[f64], n: usize, d: usize, cov: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), n * d, "x must be n x d");
    assert_eq!(cov.len(), d * d, "cov must be d x d");
    let mut t = vec![0.0; d];
    (0..n)
        .map(|i| quad_form(&x[i * d..(i + 1) * d], cov, d, &mut t))
        .collect()
}

/// The most informative pair question among `c` candidates (positions into the inputs).
///
/// Every pair (a, b) with a < b and the same gender is scored by
/// log(1 + p (1 - p) s^2), with p = sigmoid(logit_a - logit_b) the chance of choosing a
/// and s^2 = (x_a - x_b)' cov (x_a - x_b). The first pair with the highest score wins,
/// in the same order as the Python loop. `None` if no two candidates share a gender.
pub fn best_pair(
    features: &[f64],
    c: usize,
    d: usize,
    logits: &[f64],
    genders: &[i64],
    cov: &[f64],
) -> Option<(usize, usize)> {
    assert_eq!(features.len(), c * d, "features must be c x d");
    assert!(
        logits.len() == c && genders.len() == c,
        "one logit and gender per candidate"
    );
    let mut best = f64::NEG_INFINITY;
    let mut pick = None;
    let mut diff = vec![0.0; d];
    let mut scratch = vec![0.0; d];
    for a in 0..c {
        let xa = &features[a * d..(a + 1) * d];
        for b in a + 1..c {
            if genders[a] != genders[b] {
                continue;
            }
            let xb = &features[b * d..(b + 1) * d];
            for ((t, ua), ub) in diff.iter_mut().zip(xa).zip(xb) {
                *t = ua - ub;
            }
            let p = sigmoid(logits[a] - logits[b]);
            let info = (p * (1.0 - p) * quad_form(&diff, cov, d, &mut scratch)).ln_1p();
            if info > best {
                best = info;
                pick = Some((a, b));
            }
        }
    }
    pick
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quad_forms_match_the_definition() {
        let cov = [2.0, 0.5, 0.5, 1.0];
        let s = quad_forms(&[1.0, 0.0, 1.0, 2.0], 2, 2, &cov);
        assert_eq!(s, vec![2.0, 2.0 + 2.0 * 0.5 * 2.0 + 4.0]);
    }

    #[test]
    fn best_pair_prefers_uncertain_coin_flips_of_the_same_gender() {
        let cov = [1.0, 0.0, 0.0, 1.0];
        // 0 and 1 are far apart but different genders; 1 and 2 share a gender
        let features = [0.0, 0.0, 5.0, 0.0, 0.0, 3.0];
        let pick = best_pair(&features, 3, 2, &[0.0, 0.0, 0.0], &[1, 2, 2], &cov);
        assert_eq!(pick, Some((1, 2)));
        assert_eq!(
            best_pair(&features, 3, 2, &[0.0; 3], &[1, 2, 3], &cov),
            None
        );
    }
}
