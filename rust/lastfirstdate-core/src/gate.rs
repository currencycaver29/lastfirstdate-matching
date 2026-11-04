//! The ratio gate, updated incrementally (twin of `forecast.gate_decision`, FR-LM-34).
//!
//! Waitlisted men are considered in order. One is admitted only if, with him added, women
//! per man stays at or above the floor and inside the forecast's fitted range, and the
//! forecast average fulfilment stays at or above the target. Applicants with fewer
//! possible matches than `min_pool` are deferred.
//!
//! The Python version recomputes every man's attention share for every applicant, which
//! costs women x men per applicant. Here the shares are updated in place: when man a joins,
//! each woman i in his pool goes from c_i to c_i + 1 possible men, so every man already in
//! the market who could meet her loses 1/c_i - 1/(c_i + 1) of share. Only those men are
//! touched, so one applicant costs about (his pool) x (men per woman) instead of
//! women x men. That keeps the gate instant for a city-sized waitlist.

use crate::sigmoid;

/// Forecast coefficients, in the order of `forecast.START_FEATURES`.
pub const N_COEF: usize = 5;

pub struct GateInput<'a> {
    /// women x columns, row-major: may woman i be matched with the man in column c
    pub eligible: &'a [bool],
    pub n_women: usize,
    pub n_cols: usize,
    /// the first `n_current` columns are men already in the market; the rest are
    /// applicants, in waitlist order
    pub n_current: usize,
    pub coef: &'a [f64],
    pub min_pool: f64,
    pub min_women_per_man: f64,
    pub fitted_min_women_per_man: f64,
    pub target: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    WaitlistEmpty,
    Ratio,
    OutsideFittedRange,
    Forecast,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GateOutcome {
    /// admitted applicants, as column indices
    pub admit: Vec<usize>,
    /// deferred applicants (pool below `min_pool`), as column indices
    pub deferred: Vec<usize>,
    pub women_per_man: f64,
    pub forecast: f64,
    pub stop: Stop,
}

/// The market as it grows: who is in, and each woman's and man's counts.
struct Market<'a> {
    input: &'a GateInput<'a>,
    pool: Vec<f64>,
    in_market: Vec<usize>,
    /// per woman: men in the market she could be matched with
    men_of: Vec<Vec<usize>>,
    /// per column: expected share of women's attention (only men in the market)
    share: Vec<f64>,
}

impl<'a> Market<'a> {
    fn new(input: &'a GateInput<'a>) -> Self {
        let (w, c) = (input.n_women, input.n_cols);
        let mut pool = vec![0.0; c];
        for i in 0..w {
            for (col, p) in pool.iter_mut().enumerate() {
                if input.eligible[i * c + col] {
                    *p += 1.0;
                }
            }
        }
        Market {
            input,
            pool,
            in_market: Vec::new(),
            men_of: vec![Vec::new(); w],
            share: vec![0.0; c],
        }
    }

    fn add(&mut self, col: usize) {
        let c = self.input.n_cols;
        let mut own = 0.0;
        for i in 0..self.input.n_women {
            if !self.input.eligible[i * c + col] {
                continue;
            }
            let before = self.men_of[i].len() as f64;
            let after = before + 1.0;
            if before >= 1.0 {
                let loss = 1.0 / before - 1.0 / after;
                for &j in &self.men_of[i] {
                    self.share[j] -= loss;
                }
            }
            self.men_of[i].push(col);
            own += 1.0 / after;
        }
        self.share[col] = own;
        self.in_market.push(col);
    }

    fn women_per_man(&self) -> f64 {
        self.input.n_women as f64 / self.in_market.len().max(1) as f64
    }

    /// Mean chance of reaching the guarantee over the men in the market (1 if none).
    fn forecast(&self) -> f64 {
        if self.in_market.is_empty() {
            return 1.0;
        }
        let b = self.input.coef;
        let log_ratio = self.women_per_man().max(1e-9).ln();
        let common = b[0] + b[1] * log_ratio + b[4] * log_ratio.min(0.0);
        let total: f64 = self
            .in_market
            .iter()
            .map(|&j| sigmoid(common + b[2] * self.pool[j].ln_1p() + b[3] * self.share[j].ln_1p()))
            .sum();
        total / self.in_market.len() as f64
    }
}

pub fn gate_scan(input: &GateInput) -> GateOutcome {
    assert_eq!(
        input.eligible.len(),
        input.n_women * input.n_cols,
        "eligible must be women x columns"
    );
    assert!(
        input.n_current <= input.n_cols,
        "more current men than columns"
    );
    assert_eq!(
        input.coef.len(),
        N_COEF,
        "the forecast has {N_COEF} start coefficients"
    );
    let mut market = Market::new(input);
    for col in 0..input.n_current {
        market.add(col);
    }
    let mut out = GateOutcome {
        admit: Vec::new(),
        deferred: Vec::new(),
        women_per_man: market.women_per_man(),
        forecast: market.forecast(),
        stop: Stop::WaitlistEmpty,
    };
    for col in input.n_current..input.n_cols {
        if market.pool[col] < input.min_pool {
            out.deferred.push(col);
            continue;
        }
        let ratio = input.n_women as f64 / (market.in_market.len() + 1) as f64;
        if ratio < input.min_women_per_man {
            out.stop = Stop::Ratio;
            break;
        }
        if ratio < input.fitted_min_women_per_man {
            out.stop = Stop::OutsideFittedRange;
            break;
        }
        market.add(col); // tentative: if the forecast fails, the scan ends here anyway
        let f = market.forecast();
        if f < input.target {
            out.stop = Stop::Forecast;
            break;
        }
        out.admit.push(col);
        out.women_per_man = ratio;
        out.forecast = f;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full(n_women: usize, n_cols: usize) -> Vec<bool> {
        vec![true; n_women * n_cols]
    }

    fn input<'a>(e: &'a [bool], w: usize, c: usize, coef: &'a [f64]) -> GateInput<'a> {
        GateInput {
            eligible: e,
            n_women: w,
            n_cols: c,
            n_current: 0,
            coef,
            min_pool: 10.0,
            min_women_per_man: 3.0,
            fitted_min_women_per_man: 0.0,
            target: 0.9,
        }
    }

    #[test]
    fn stops_at_the_ratio_floor() {
        let e = full(30, 20);
        let coef = [10.0, 0.0, 0.0, 0.0, 0.0];
        let out = gate_scan(&input(&e, 30, 20, &coef));
        assert_eq!(out.admit, (0..10).collect::<Vec<_>>());
        assert_eq!(out.stop, Stop::Ratio);
        assert_eq!(out.women_per_man, 3.0);
    }

    #[test]
    fn stops_at_the_forecast() {
        let e = full(60, 20);
        let coef = [-4.0, 4.0, 0.0, 0.0, 0.0];
        let out = gate_scan(&input(&e, 60, 20, &coef));
        assert_eq!(out.stop, Stop::Forecast);
        assert!(out.forecast >= 0.9);
    }

    #[test]
    fn defers_small_pools_and_keeps_going() {
        let mut e = full(40, 3);
        for i in 1..40 {
            e[i * 3 + 1] = false; // column 1 can only meet woman 0
        }
        let coef = [10.0, 0.0, 0.0, 0.0, 0.0];
        let out = gate_scan(&input(&e, 40, 3, &coef));
        assert_eq!(out.deferred, vec![1]);
        assert_eq!(out.admit, vec![0, 2]);
        assert_eq!(out.stop, Stop::WaitlistEmpty);
    }

    #[test]
    fn incremental_shares_match_a_fresh_count() {
        // women 0..4, men 0..3 with a sparse pattern
        let w = 5;
        let c = 3;
        let pattern = [[1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1], [0, 0, 1]];
        let e: Vec<bool> = pattern.iter().flatten().map(|v| *v == 1).collect();
        let coef = [0.0; 5];
        let inp = input(&e, w, c, &coef);
        let mut market = Market::new(&inp);
        for col in 0..c {
            market.add(col);
        }
        for col in 0..c {
            let fresh: f64 = (0..w)
                .filter(|&i| pattern[i][col] == 1)
                .map(|i| 1.0 / pattern[i].iter().sum::<i32>() as f64)
                .sum();
            assert!((market.share[col] - fresh).abs() < 1e-12);
        }
    }
}
