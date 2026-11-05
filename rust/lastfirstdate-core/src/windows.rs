//! Date windows (FR-LM-37): which evenings a member is free, and the first one two members
//! share. Twins of `booking.window_bits`, `booking.earliest_common` and the blocked-day loop
//! in `app.free_windows`.
//!
//! A member's windows are one `u64`: bit `k * 3 + part` is set when they are free on day
//! `today + k` at that part of the day (lunch, evening, late), for `k` below the horizon.

use std::collections::{HashMap, HashSet};

/// Parts of the day: lunch, evening, late.
pub const N_PARTS: usize = 3;
/// What `earliest_common` returns when two members share no window.
pub const NO_WINDOW: i64 = -1;
/// The signup day of members without a subscription (`app.NO_SIGNUP`).
pub const NO_SIGNUP: i64 = i64::MIN;

/// Free windows over the next `horizon` days for each member's weekly free slots
/// (slot = weekday * 3 + part, weekday 0 = Monday). Days before `lead` are never offered;
/// `blocked` (members x horizon, row-major) marks days a member cannot date.
pub fn window_bits(
    weekly: &[i64],
    today: i64,
    lead: usize,
    horizon: usize,
    blocked: Option<&[bool]>,
) -> Vec<u64> {
    weekly
        .iter()
        .enumerate()
        .map(|(r, &slots)| {
            let mut out = 0u64;
            for k in lead..horizon {
                if blocked.is_some_and(|b| b[r * horizon + k]) {
                    continue;
                }
                let weekday = (today + k as i64).rem_euclid(7) as usize;
                let day = ((slots >> (weekday * N_PARTS)) & 0b111) as u64;
                out |= day << (k * N_PARTS);
            }
            out
        })
        .collect()
}

/// The earliest window both members are free (its bit index), or `NO_WINDOW`.
#[inline]
pub fn earliest_common(a: u64, b: u64) -> i64 {
    let common = a & b;
    if common == 0 {
        NO_WINDOW
    } else {
        i64::from(common.trailing_zeros())
    }
}

/// Every date a member already has: days with a date, dates per week, dates per cap period.
pub struct Taken<'a> {
    pub busy_member: &'a [usize],
    pub busy_day: &'a [i64],
    pub week_member: &'a [usize],
    pub week: &'a [i64],
    pub week_count: &'a [i64],
    pub period_member: &'a [usize],
    pub period_start: &'a [i64],
    pub period_count: &'a [i64],
}

/// The rules that turn dates already taken into blocked days.
pub struct Rules<'a> {
    pub today: i64,
    pub horizon: usize,
    pub max_per_week: i64,
    pub trial_days: i64,
    pub cycle_days: i64,
    /// each member's date cap per period
    pub limit: &'a [i64],
    /// each member's signup day, or `NO_SIGNUP`
    pub signup: &'a [i64],
}

/// The first day of the cap period containing `day`: a subscriber's billing cycle (the
/// trial, then back-to-back paid cycles), otherwise blocks of `cycle_days` from day 0.
pub fn period_start(day: i64, signup: i64, trial_days: i64, cycle_days: i64) -> i64 {
    if signup != NO_SIGNUP && day >= signup {
        let paid_from = signup + trial_days;
        if day < paid_from {
            signup
        } else {
            paid_from + (day - paid_from).div_euclid(cycle_days) * cycle_days
        }
    } else {
        day.div_euclid(cycle_days) * cycle_days
    }
}

/// (members x horizon, row-major) days a member cannot date: they already have a date
/// that day, their week is full, or their cap period is full.
pub fn blocked_days(taken: &Taken, rules: &Rules) -> Vec<bool> {
    let m = rules.limit.len();
    let busy: HashSet<(usize, i64)> = taken
        .busy_member
        .iter()
        .copied()
        .zip(taken.busy_day.iter().copied())
        .collect();
    let mut weeks: HashMap<(usize, i64), i64> = HashMap::new();
    for ((&r, &w), &c) in taken
        .week_member
        .iter()
        .zip(taken.week)
        .zip(taken.week_count)
    {
        *weeks.entry((r, w)).or_insert(0) += c;
    }
    let mut periods: HashMap<(usize, i64), i64> = HashMap::new();
    for ((&r, &p), &c) in taken
        .period_member
        .iter()
        .zip(taken.period_start)
        .zip(taken.period_count)
    {
        *periods.entry((r, p)).or_insert(0) += c;
    }
    let mut out = vec![false; m * rules.horizon];
    for r in 0..m {
        for k in 0..rules.horizon {
            let day = rules.today + k as i64;
            let start = period_start(day, rules.signup[r], rules.trial_days, rules.cycle_days);
            out[r * rules.horizon + k] = busy.contains(&(r, day))
                || weeks.get(&(r, day.div_euclid(7))).copied().unwrap_or(0) >= rules.max_per_week
                || periods.get(&(r, start)).copied().unwrap_or(0) >= rules.limit[r];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const MON_EVENING: i64 = 1 << 1;
    const TUE_EVENING: i64 = 1 << 4;

    #[test]
    fn first_shared_evening() {
        let bits = window_bits(&[MON_EVENING, MON_EVENING | TUE_EVENING], 0, 1, 14, None);
        // today's Monday is too soon (lead 1): the next Monday evening, day 7, part 1
        assert_eq!(earliest_common(bits[0], bits[1]), (7 * 3 + 1) as i64);
        assert_eq!(earliest_common(bits[0], 0), NO_WINDOW);
    }

    #[test]
    fn blocked_days_skip_dates_weeks_and_caps() {
        let rules = Rules {
            today: 0,
            horizon: 14,
            max_per_week: 1,
            trial_days: 30,
            cycle_days: 28,
            limit: &[3, 1],
            signup: &[NO_SIGNUP, 0],
        };
        let taken = Taken {
            busy_member: &[0],
            busy_day: &[2],
            week_member: &[0],
            week: &[1],
            week_count: &[1],
            period_member: &[1],
            period_start: &[0],
            period_count: &[1],
        };
        let b = blocked_days(&taken, &rules);
        assert!(b[2] && b[7] && b[13] && !b[1] && !b[6]); // a date on day 2; week 1 is full
        assert!(b[14..28].iter().all(|&x| x)); // member 1's trial period is full
    }

    #[test]
    fn periods_follow_the_billing_cycle() {
        assert_eq!(period_start(5, NO_SIGNUP, 30, 28), 0);
        assert_eq!(period_start(-3, NO_SIGNUP, 30, 28), -28);
        assert_eq!(period_start(10, 0, 30, 28), 0); // the trial
        assert_eq!(period_start(60, 0, 30, 28), 58); // second paid cycle starts at 30 + 28
    }
}
