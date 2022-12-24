use core::qm;
use core::qm::Error;
use data::volatility::strike::{LogRelStrike, Strike};
use instruments::options::PutOrCall;
use math::interpolation::CubicSpline;
use math::interpolation::Extrap;
use math::interpolation::Interpolate;
use math::optionpricing::{Bachelier, Black76};
use serde::Serialize;
use std::fmt::Debug;
use std::ops::Neg;

/// A VolSmile is a curve of volatilities by strike, all for a specific date.
pub trait VolSmile<X: Strike>: Serialize + Clone + Debug {
    /// These volatilities must be converted to variances by squaring and
    /// multiplying by some t. The t to use depends on the vol surface. We
    /// assume that the VolSmile simply provides volatilities, and it is up
    /// to the VolSurface to interpret these in terms of variances.
    fn volatilities(&self, strikes: &[X], volatilities: &mut [f64]) -> Result<(), Error>;

    /// Convenience function to fetch a single volatility. This does not
    /// have to be implemented by every implementer of the trait, though
    /// it could be for performance reasons.
    fn volatility(&self, strike: X) -> Result<f64, Error> {
        let strikes = [strike];
        let mut vols = Vec::new();
        vols.resize(1, f64::NAN);
        self.volatilities(&strikes, &mut vols)?;
        Ok(vols[0])
    }

    fn implied_density(&self, x: X, dx: X) -> f64 {
        let xh: X = x + dx.into();
        let v = self.volatility(x).unwrap();
        let vh = self.volatility(xh).unwrap();
        return (vh - v) / (dx.into());
    }
}

pub fn b76_price<X: Strike>(
    smile: &impl VolSmile<X>,
    df: f64,
    fwd: f64,
    x: X,
    ttm: f64,
    q: PutOrCall,
) -> f64 {
    let v = smile.volatility(x);
    let k = x.to_cash_strike(fwd, ttm);
    let b76 = Black76::new().unwrap();
    return b76.price(df, fwd, k, ttm, v.unwrap(), q);
}

pub fn bachelier_price<X: Strike>(
    smile: &impl VolSmile<X>,
    df: f64,
    fwd: f64,
    x: X,
    ttm: f64,
    q: PutOrCall,
) -> f64 {
    let v = smile.volatility(x);
    let k = x.to_cash_strike(fwd, ttm);
    let bachelier = Bachelier::new().unwrap();
    return bachelier.price(df, fwd, k, ttm, v.unwrap(), q);
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RSVI {
    pub t: f64,
    pub fwd: f64,
    pub a: f64,
    pub b: f64,
    pub rho: f64,
    pub m: f64,
    pub sigma: f64,
}

impl RSVI {
    pub fn new(t: f64, fwd: f64, alpha: f64, beta: f64, rho: f64, m: f64, sigma: f64) -> RSVI {
        RSVI {
            t,
            fwd,
            a: alpha,
            b: beta,
            rho,
            m,
            sigma,
        }
    }

    pub fn to_svijw(&self) -> SVIJW {
        let v = (self.a
            + self.b
                * (self.rho.neg() * self.m + f64::sqrt(self.m * self.m + self.sigma * self.sigma)))
            / self.t;
        let w = v * self.t;
        let phi = (1.0 / w.sqrt())
            * 0.5
            * self.b
            * ((self.m / f64::sqrt(self.m * self.m + self.sigma * self.sigma)).neg() + self.rho);
        let p = (1.0 / w.sqrt()) * self.b * (1.0 - self.rho);
        let c = (1.0 / w.sqrt()) * self.b * (1.0 + self.rho);
        let vv =
            (1.0 / self.t) * (self.a + self.b * self.sigma * f64::sqrt(1.0 - self.rho * self.rho));

        return SVIJW::new(self.t, self.fwd, v, phi, p, c, vv);
    }

    pub fn is_valid(&self) -> bool {
        return self.b >= 0.0
            && self.rho.abs() < 1.0
            && self.sigma > 0.0
            && self.sigma + self.b * self.sigma * (1.0 - self.rho * self.rho).sqrt() >= 0.0;
    }
}

impl VolSmile<LogRelStrike> for RSVI {
    fn volatilities(
        &self,
        strikes: &[LogRelStrike],
        volatilities: &mut [f64],
    ) -> Result<(), Error> {
        let mut i = 0;
        for strike in strikes {
            let x = strike.x;
            let v = self.a
                + self.b
                    * (self.rho * (x - self.m)
                        + ((x - self.m).powi(2) + self.sigma.powi(2)).sqrt());
            volatilities[i] = (v / self.t).sqrt();
            i += 1;
        }
        Ok(())
    }
}

impl VolSmile<f64> for RSVI {
    fn volatilities(&self, strikes: &[f64], volatilities: &mut [f64]) -> Result<(), Error> {
        let xs: Vec<LogRelStrike> = strikes
            .iter()
            .map(|k| Strike::cash_to_strike_space(*k, self.fwd, self.t))
            .collect();
        self.volatilities(&xs, volatilities)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LSSVI {
    t: f64,
    fwd: f64,
    theta: f64,
    phi: f64,
    rho: f64,
}

impl LSSVI {
    pub fn new(t: f64, fwd: f64, theta: f64, phi: f64, rho: f64) -> LSSVI {
        LSSVI {
            t,
            fwd,
            theta,
            phi,
            rho,
        }
    }
}

impl VolSmile<LogRelStrike> for LSSVI {
    fn volatilities(
        &self,
        strikes: &[LogRelStrike],
        volatilities: &mut [f64],
    ) -> Result<(), Error> {
        let mut i = 0;
        for strike in strikes {
            let x = strike.x;
            let v = 0.5
                * self.theta
                * (1.0
                    + self.rho * self.phi * x
                    + f64::sqrt(
                        1.0 + 2.0 * self.rho * self.phi * x + self.phi.powi(2) * x.powi(2),
                    ));
            volatilities[i] = (v / self.t).sqrt();
            i += 1;
        }
        Ok(())
    }
}

impl VolSmile<f64> for LSSVI {
    fn volatilities(&self, strikes: &[f64], volatilities: &mut [f64]) -> Result<(), Error> {
        let xs: Vec<LogRelStrike> = strikes
            .iter()
            .map(|k| Strike::cash_to_strike_space(*k, self.fwd, self.t))
            .collect();
        self.volatilities(&xs, volatilities)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SVIJW {
    /// Time to expiry
    t: f64,
    /// Forward
    fwd: f64,
    /// ATM Variance
    v: f64,
    /// ATM Skew
    phi: f64,
    /// The slope of the left (put) wing
    p: f64,
    /// the slope of the right (call) wing
    c: f64,
    /// The minimum iplied variance
    vv: f64,
}

impl SVIJW {
    pub fn new(t: f64, fwd: f64, v: f64, phi: f64, p: f64, c: f64, vv: f64) -> SVIJW {
        SVIJW {
            t,
            fwd,
            v,
            phi,
            p,
            c,
            vv,
        }
    }

    pub fn to_rsvi(&self) -> RSVI {
        let w = self.v * self.t;
        let ws = w.sqrt();

        let b = ws * 0.5 * (self.c + self.p);
        let rho = 1.0 - self.p * ws / b;

        let mut beta = rho - 2.0 * self.phi * w.sqrt() / b;
        if beta == 0.0 {
            beta = 0.001;
        }

        let alpha = beta.signum() * f64::sqrt(1.0 / beta.powi(2) - 1.0);

        let m = (self.v - self.vv) * self.t
            / (b * (-rho + alpha.signum() * f64::sqrt(1.0 + alpha.powi(2))
                - alpha * f64::sqrt(1.0 - rho.powi(2))));

        let sigma = alpha * m;

        let a = self.vv * self.t - b * sigma * f64::sqrt(1.0 - rho.powi(2));
        return RSVI::new(self.t, self.fwd, a, b, rho, m, sigma);
    }
}

impl VolSmile<LogRelStrike> for SVIJW {
    fn volatilities(
        &self,
        strikes: &[LogRelStrike],
        volatilities: &mut [f64],
    ) -> Result<(), Error> {
        let w = self.v * self.t;
        let ws = w.sqrt();
        let b = ws * 0.5 * (self.c + self.p);
        let rho = 1.0 - 2.0 * self.p / (self.c + self.p);
        let beta = rho - 2. * self.phi * ws / b;
        let alpha = beta.signum() * f64::sqrt(1.0 / (beta.powi(2)) - 1.0);
        let m = (self.v - self.vv) * self.t
            / (b * (-rho + alpha.signum() * f64::sqrt(1.0 + alpha * alpha)
                - alpha * f64::sqrt(1.0 - rho * rho)));
        let sigma = alpha * m;
        let a = self.vv * self.t - b * sigma * f64::sqrt(1.0 - rho * rho);
        let rsvi = RSVI::new(self.t, self.fwd, a, b, rho, m, sigma);
        rsvi.volatilities(strikes, volatilities)
    }
}

impl VolSmile<f64> for SVIJW {
    fn volatilities(&self, strikes: &[f64], volatilities: &mut [f64]) -> Result<(), Error> {
        let xs: Vec<LogRelStrike> = strikes
            .iter()
            .map(|k| Strike::cash_to_strike_space(*k, self.fwd, self.t))
            .collect();
        self.volatilities(&xs, volatilities)
    }
}

/// A flat smile, where the vol is the same for all strikes. (It may be
/// different at other dates.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatSmile {
    vol: f64,
}

impl<K> VolSmile<K> for FlatSmile
where
    K: Strike,
{
    fn volatilities(&self, strikes: &[K], volatilities: &mut [f64]) -> Result<(), Error> {
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
    pub fn new(vol: f64) -> Result<FlatSmile, Error> {
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

impl<K> VolSmile<K> for CubicSplineSmile<K>
where
    K: Strike,
{
    fn volatilities(&self, strikes: &[K], volatilities: &mut [f64]) -> Result<(), Error> {
        let n = strikes.len();
        assert_eq!(n, volatilities.len());

        for i in 0..n {
            volatilities[i] = self.smile.interpolate(strikes[i])?;
        }
        Ok(())
    }
}

impl<K> CubicSplineSmile<K>
where
    K: Strike,
{
    /// Creates a cubic spline smile that interpolates between the given
    /// pillar volatilities. The supplied vector is of (strike, volatility)
    /// pairs.
    pub fn new(pillars: &[(K, f64)], fwd: f64, ttm: f64) -> Result<CubicSplineSmile<K>, Error> {
        let i = CubicSpline::new(pillars, Extrap::Natural, Extrap::Natural)?;
        Ok(CubicSplineSmile { smile: i, fwd, ttm })
    }
}

#[cfg(test)]
mod tests {
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
            assert!(
                approx_eq(vols[i], vol, 1e-12),
                "vol={} expected={}",
                vols[i],
                vol
            );
        }
    }

    #[test]
    fn test_cubic_spline_smile() {
        let points = [(70.0, 0.4), (80.0, 0.3), (90.0, 0.22), (100.0, 0.25)];
        let smile: CubicSplineSmile<f64> = CubicSplineSmile::new(&points, 1.0, 1.0).unwrap();

        let strikes = vec![60.0, 70.0, 80.0, 85.0, 90.0, 95.0, 100.0, 110.0];
        let mut vols = vec![0.0; strikes.len()];

        smile.volatilities(&strikes, &mut vols).unwrap();

        let expected = vec![0.5, 0.4, 0.3, 0.25025, 0.22, 0.2245, 0.25, 0.28];

        for i in 0..vols.len() {
            assert!(
                approx_eq(vols[i], expected[i], 1e-12),
                "vol={} expected={}",
                vols[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_svijw() {
        // 0.04	-0.05	0.5	0.49	0.038
        let svijw = SVIJW::new(1.0, 100.0, 0.04, -0.05, 0.5, 0.49, 0.038);
        let strike = 60.0;
        let vol = svijw.volatility(strike).unwrap();
        assert_eq!(0.24733878039241802, vol);
    }

    #[test]
    fn test_svijw_to_raw() {
        let svijw = SVIJW::new(1.0, 100.0, 0.04, -0.05, 0.5, 0.49, 0.038);
        let strike = 60.0;
        let rsvi = svijw.to_rsvi();
        let svijw2 = rsvi.to_svijw();
        assert_eq!(svijw, svijw2);
    }
}
