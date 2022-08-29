
use math::interpolation::{Interpolable};
use std::cmp::Ordering;
use derive_more::{Add, Display, From, Into, Mul, Sub};
use serde::Serialize;
use std::fmt::Debug;
use std::ops::{Add, Sub};

pub trait Strike:
    Copy
    + Clone
    + Serialize
    + Sync
    + Send
    + Debug
    + Sized
    + Interpolable<Self>
    + Add<f64, Output = Self>
    + Sub<f64, Output = Self>
    + From<f64>
    + Into<f64>
{
    fn to_cash_strike(&self, fwd: f64, ttm: f64) -> f64;

    fn cash_to_strike_space(k: f64, fwd: f64, ttm: f64) -> Self;
}


#[derive(Copy, Clone, Serialize, Debug, Add, Sub, Display)]
pub struct LogRelStrike {
    pub x: f64,
}

impl Interpolable<LogRelStrike> for LogRelStrike {
    fn interp_diff(&self, other: LogRelStrike) -> f64 {
        self.x - other.x
    }

    fn interp_cmp(&self, other: LogRelStrike) -> Ordering {
        self.x.partial_cmp(&other.x).unwrap()
    }
}

impl Add<f64> for LogRelStrike {
    type Output = LogRelStrike;

    fn add(self, rhs: f64) -> Self::Output {
        LogRelStrike { x: self.x + rhs }
    }
}

impl Sub<f64> for LogRelStrike {
    type Output = LogRelStrike;

    fn sub(self, rhs: f64) -> Self::Output {
        LogRelStrike { x: self.x - rhs }
    }
}

impl From<f64> for LogRelStrike {
    fn from(x: f64) -> Self {
        LogRelStrike { x }
    }
}

impl Into<f64> for LogRelStrike {
    fn into(self) -> f64 {
        self.x
    }
}

impl Strike for LogRelStrike {
    fn to_cash_strike(&self, fwd: f64, _: f64) -> f64 {
        self.x.exp() * fwd
    }

    fn cash_to_strike_space(k: f64, fwd: f64, _: f64) -> Self {
        LogRelStrike { x: (k / fwd).ln() }
    }
}

impl Strike for f64 {
    fn to_cash_strike(&self, _: f64, _: f64) -> f64 {
        *self
    }

    fn cash_to_strike_space(k: f64, _: f64, _: f64) -> Self {
        k
    }
}
