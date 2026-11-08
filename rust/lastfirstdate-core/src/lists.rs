//! Round-robin daily lists under an exposure cap (twin of `fairness.fair_day_lists`).
//!
//! Each chooser ranks the candidates available to her by score + guarantee bonus (ties by
//! id). Lists are then filled one slot at a time in chooser order: everyone's first slot,
//! then everyone's second, and so on, skipping anyone already on `cap` lists today, so no
//! one fills up every list (Singh and Joachims 2018, decision D7):
//!
//! ```text
//! cap = max(1, ceil(multiple * list slots today / candidates today))
//! ```

use std::cmp::Ordering;

/// One day's inputs. Rows are choosers, in the order they take turns.
pub struct DayInput<'a> {
    /// m x n: may chooser row r see candidate j today
    pub available: &'a [bool],
    /// m x n: chooser row r's score for candidate j
    pub scores: &'a [f64],
    pub m: usize,
    pub n: usize,
    /// n: guarantee bonus per candidate
    pub bonus: &'a [f64],
    pub size: usize,
    pub multiple: f64,
    /// m: slots already given out today (exploration); they come first and count
    pub reserved: &'a [Vec<usize>],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DayLists {
    pub lists: Vec<Vec<usize>>,
    pub cap: usize,
}

/// Most lists one person may appear on today.
pub fn exposure_cap(n_slots: usize, n_candidates: usize, multiple: f64) -> usize {
    if n_candidates == 0 {
        return 0;
    }
    ((multiple * n_slots as f64 / n_candidates as f64).ceil() as usize).max(1)
}

/// Candidates a chooser row may see, best first: score + bonus descending, then id.
fn ranked(available: &[bool], scores: &[f64], bonus: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..available.len()).filter(|&j| available[j]).collect();
    idx.sort_by(|&a, &b| {
        let (va, vb) = (scores[a] + bonus[a], scores[b] + bonus[b]);
        vb.partial_cmp(&va)
            .unwrap_or(Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx
}

pub fn fair_day_lists(input: &DayInput) -> DayLists {
    let (m, n, size) = (input.m, input.n, input.size);
    assert_eq!(input.available.len(), m * n, "available must be m x n");
    assert_eq!(input.scores.len(), m * n, "scores must be m x n");
    assert_eq!(input.bonus.len(), n, "one bonus per candidate");
    assert_eq!(input.reserved.len(), m, "one reserved list per chooser");

    // today's cap, from today's list slots and candidates
    let mut is_candidate = vec![false; n];
    let mut n_slots = 0;
    for r in 0..m {
        let row = &input.available[r * n..(r + 1) * n];
        let mut count = 0;
        for (j, ok) in row.iter().enumerate() {
            if *ok {
                is_candidate[j] = true;
                count += 1;
            }
        }
        n_slots += count.min(size);
    }
    let cap = exposure_cap(
        n_slots,
        is_candidate.iter().filter(|c| **c).count(),
        input.multiple,
    );

    let ranked: Vec<Vec<usize>> = (0..m)
        .map(|r| {
            ranked(
                &input.available[r * n..(r + 1) * n],
                &input.scores[r * n..(r + 1) * n],
                input.bonus,
            )
        })
        .collect();
    let mut lists: Vec<Vec<usize>> = input.reserved.to_vec();
    let mut exposure = vec![0usize; n];
    for list in &lists {
        for &j in list {
            exposure[j] += 1;
        }
    }
    let mut pointer = vec![0usize; m];
    for _ in 0..size {
        let mut progressed = false;
        for r in 0..m {
            if lists[r].len() >= size {
                continue;
            }
            let (order, mut p) = (&ranked[r], pointer[r]);
            while p < order.len() && (exposure[order[p]] >= cap || lists[r].contains(&order[p])) {
                p += 1;
            }
            if p < order.len() {
                let j = order[p];
                lists[r].push(j);
                exposure[j] += 1;
                p += 1;
                progressed = true;
            }
            pointer[r] = p;
        }
        if !progressed {
            break;
        }
    }
    DayLists { lists, cap }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cap_spreads_a_popular_candidate() {
        // 3 choosers all prefer candidate 0; lists of 1; 3 slots over 3 candidates -> cap 1
        let available = [true; 9];
        let scores = [3.0, 2.0, 1.0, 3.0, 1.0, 2.0, 3.0, 2.0, 1.0];
        let reserved = vec![Vec::new(); 3];
        let input = DayInput {
            available: &available,
            scores: &scores,
            m: 3,
            n: 3,
            bonus: &[0.0; 3],
            size: 1,
            multiple: 1.0,
            reserved: &reserved,
        };
        let out = fair_day_lists(&input);
        assert_eq!(out.cap, 1);
        assert_eq!(out.lists, vec![vec![0], vec![2], vec![1]]);
    }

    #[test]
    fn reserved_slots_come_first_and_count() {
        let available = [true; 4];
        let reserved = vec![vec![1], Vec::new()];
        let input = DayInput {
            available: &available,
            scores: &[2.0, 1.0, 2.0, 1.0],
            m: 2,
            n: 2,
            bonus: &[0.0, 0.0],
            size: 2,
            multiple: 1.0,
            reserved: &reserved,
        };
        let out = fair_day_lists(&input);
        assert_eq!(out.cap, 2);
        assert_eq!(out.lists, vec![vec![1, 0], vec![0, 1]]);
    }

    #[test]
    fn ties_go_to_the_lower_id() {
        // score + bonus = 1.5, 2.0, 1.5: candidate 1 first, then the tie by id
        let order = ranked(&[true, true, true], &[1.0, 2.0, 1.0], &[0.5, 0.0, 0.5]);
        assert_eq!(order, vec![1, 0, 2]);
    }
}
