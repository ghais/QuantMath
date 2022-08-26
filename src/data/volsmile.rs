use std::cmp::Ordering;
use math::interpolation::{CubicSpline, Interpolable};
use math::interpolation::Interpolate;
use math::interpolation::Extrap;
use core::qm;
use std::fmt::Debug;
use serde::{Serialize, Serializer};
use core::factories::TypeId;


pub trait Strike : Copy + Clone + Serialize + Sync + Send + Debug + Sized + Interpolable<Self> {
    fn to_cash_strike(&self, fwd: f64, ttm: f64) -> f64;

    fn cash_to_strike_space(k: f64, fwd: f64, ttm: f64) -> Self;
}

/// A VolSmile is a curve of volatilities by strike, all for a specific date.

#[derive(Debug, Clone, Serialize, Deserialize, Copy)]
struct LogRelStrike {
    x: f64
}
impl Interpolable<LogRelStrike> for LogRelStrike {
    fn interp_diff(&self, other: LogRelStrike) -> f64 {
        self.x - other.x
    }

    fn interp_cmp(&self, other: LogRelStrike) -> Ordering {
        self.x.partial_cmp(&other.x).unwrap()
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

pub trait VolSmile : Serialize + Clone + Debug{

    /// These volatilities must be converted to variances by squaring and
    /// multiplying by some t. The t to use depends on the vol surface. We
    /// assume that the VolSmile simply provides volatilities, and it is up
    /// to the VolSurface to interpret these in terms of variances.
    fn volatilities(
        &self,
        strikes: &[f64],
        volatilities: &mut[f64]) -> Result<(), qm::Error>;

    /// Convenience function to fetch a single volatility. This does not
    /// have to be implemented by every implementer of the trait, though
    /// it could be for performance reasons.
    fn volatility(&self, strike: f64) -> Result<f64, qm::Error> {
        let strikes = [strike];
        let mut vols = [f64::NAN];
        self.volatilities(&strikes, &mut vols)?;
        Ok(vols[0])
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RSVI {
    t : f64,
    fwd: f64,
    alpha: f64,
    beta: f64,
    rho: f64,
    m: f64,
    sigma: f64,
}

impl RSVI {
    fn new(t: f64, fwd: f64, alpha: f64, beta: f64, rho: f64, m: f64, sigma: f64) -> RSVI {
        RSVI {
            t,
            fwd,
            alpha,
            beta,
            rho,
            m,
            sigma,
        }
    }

    fn is_valid(&self) -> bool {
        return
            self.beta >= 0.0 &&
            self.rho.abs() < 1.0 &&
            self.sigma > 0.0 &&
            self.sigma + self.beta * self.sigma * (1.0 - self.rho * self.rho).sqrt() >= 0.0;
    }
}

impl VolSmile for RSVI {
    fn volatilities(
        &self,
        strikes: &[f64],
        volatilities: &mut[f64]) -> Result<(), qm::Error> {
        let mut i = 0;
        for strike in strikes {
            let x = LogRelStrike::cash_to_strike_space(*strike, self.fwd, self.t).x;
            let v = self.alpha+ self.beta * (self.rho* (x - self.m) + (x - self.m).powi(2) + self.sigma.powi(2)).sqrt();
            volatilities[i] = (v/self.t).sqrt();
            i += 1;
        }
        Ok(())
    }
}
/// A flat smile, where the vol is the same for all strikes. (It may be
/// different at other dates.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatSmile {
    vol: f64
}

impl VolSmile for FlatSmile {

    fn volatilities(
        &self,
        strikes: &[f64],
        volatilities: &mut[f64]) -> Result<(), qm::Error> {

        let n = strikes.len();
        assert_eq!(n, volatilities.len());

        for i in 0..n {
            volatilities[i] = self.vol;
        }
        Ok(())
    }
}

impl FlatSmile {

    /// Creates a flat smile with the given volatility.
    pub fn new(vol: f64) -> Result<FlatSmile, qm::Error> {
        Ok(FlatSmile { vol })
    }
}

/// A simple implementation of a VolSmile in terms of a cubic spline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CubicSplineSmile<K: Strike> {
    smile: CubicSpline<K>,
    fwd: f64,
    ttm: f64,
}


impl <K> VolSmile for CubicSplineSmile<K> where K : Strike  {

    fn volatilities(
        &self,
        strikes: &[f64],
        volatilities: &mut[f64]) -> Result<(), qm::Error> {

        let n = strikes.len();
        assert_eq!(n, volatilities.len());

        for i in 0..n {
            volatilities[i] = self.smile.interpolate(<K>::cash_to_strike_space(strikes[i], self.fwd, self.ttm))?;
        }
        Ok(())
    }
}

impl <K> CubicSplineSmile<K> where K: Strike {

    /// Creates a cubic spline smile that interpolates between the given
    /// pillar volatilities. The supplied vector is of (strike, volatility)
    /// pairs.
    pub fn new(pillars: &[(K, f64)], fwd: f64, ttm: f64) -> Result<CubicSplineSmile<K>, qm::Error> {
        let i = CubicSpline::new(pillars, Extrap::Natural, Extrap::Natural)?;
        Ok(CubicSplineSmile { smile: i, fwd, ttm })
    }
}


#[cfg(test)]
mod tests {
    use plotly::common::Mode;
    use super::*;
    use math::numerics::approx_eq;

    #[test]
    fn test_flat_smile() {
        let vol = 0.2;
        let smile = FlatSmile::new(vol).unwrap();

        let strikes = vec![60.0, 70.0, 80.0];
        let mut vols = vec![0.0; strikes.len()];

        smile.volatilities(&strikes, &mut vols).unwrap();

        for i in 0..vols.len() {
            assert!(approx_eq(vols[i], vol, 1e-12),
                "vol={} expected={}", vols[i], vol);
        }
    }

    #[test]
    fn test_cubic_spline_smile() {
        let points = [(70.0, 0.4), (80.0, 0.3), (90.0, 0.22), (100.0, 0.25)];
        let smile : CubicSplineSmile<f64> = CubicSplineSmile::new(&points, 1.0, 1.0).unwrap();

        let strikes = vec![60.0, 70.0, 80.0, 85.0, 90.0, 95.0, 100.0, 110.0];
        let mut vols = vec![0.0; strikes.len()];

        smile.volatilities(&strikes, &mut vols).unwrap();

        let expected = vec![0.5, 0.4, 0.3, 0.25025, 0.22, 0.2245, 0.25, 0.28];

        for i in 0..vols.len() {
            assert!(approx_eq(vols[i], expected[i], 1e-12),
                "vol={} expected={}", vols[i], expected[i]);
        }
    }

    #[test]
    fn test_svi() {
        let svi = RSVI::new(1.0, 100.0, 0.2, 0.11, 0.2, 0.0, 0.2);
        let ks = (10..200).step_by(1).map(|x| x as f64).collect::<Vec<f64>>();
        let xs : Vec<LogRelStrike> = ks.iter().map(|k| LogRelStrike::cash_to_strike_space(*k, 100.0, 1.0)).collect();
        let mut vs = vec![0.0; xs.len()];
        svi.volatilities(&ks, &mut vs).unwrap();
        use plotly::{Plot, Layout, Scatter};
        let trace1 = Scatter::new(ks, vs)
            .name("Volatility")
            .mode(Mode::Lines);

        let mut plot = Plot::new();
        plot.add_trace(trace1);
        plot.show()

    }
}
