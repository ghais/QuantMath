use std::ops::Neg;
use ndarray::s;
use num_traits::FloatConst;
use statrs::distribution::{Continuous, ContinuousCDF, Normal};

use core::qm;
use instruments::options::PutOrCall;

/// The 1976 reformulation of the Black-Scholes formula, where the price of
/// a European option is expressed in terms of the Forward and the Strike.
pub struct Black76 {
    normal: Normal
}

impl Black76 {
    pub fn new() -> Result<Black76, qm::Error> {

        // For some reason we cannot use the ? operator for this sort of error.
        // as a workaround, do it manually for now.
        match Normal::new(0.0, 1.0) {
            Ok(normal) => Ok(Black76 { normal: normal }),
            Err(e) => Err(qm::Error::new(&format!("RSStat error: {}", e)))
        }
    }

    pub fn price(&self, df: f64, fwd: f64, k: f64, vol: f64, ttm: f64, q: PutOrCall) -> f64 {
        assert!(df > 0.0, "df must be positive");
        assert!(fwd > 0.0, "Forward must be positive");
        assert!(vol >= 0.0, "Variance must be non-negative");
        assert!(k >= 0.0, "Strike must be non-negative");
        if ttm < 0.0 {
            return 0.0;
        }
        let sqrt_variance = vol * ttm.sqrt();
        if sqrt_variance == 0.0 {
            return discounted_intrinsinc(df, fwd, k, q);
        }
        let log_moneyness = (fwd / k).ln();
        let (d_plus, d_minus) = d_plus_minus(log_moneyness, sqrt_variance);
        return match q {
            PutOrCall::Put => {
                df * (self.cdf(d_plus) * fwd - self.cdf(d_minus) * k)
            }
            PutOrCall::Call => {
                df * (self.cdf(-d_minus) * k - self.cdf(-d_plus) * fwd)
            }
        }
    }
    /// Calculates the PV of a European call option under Black Scholes
    pub fn call_price(&self, df: f64, forward: f64, strike: f64, vol: f64, ttm: f64) -> f64 {
        self.price(df, forward, strike, vol, ttm, PutOrCall::Call)
    }

    /// Calculates the PV of a European put option under Black Scholes
    pub fn put_price(&self, df: f64, forward: f64, strike: f64, vol: f64, ttm: f64) -> f64 {
        self.price(df, forward, strike, vol, ttm, PutOrCall::Put)
    }

    pub fn delta(&self, df: f64, fwd: f64, k: f64, vol: f64, ttm: f64, q: PutOrCall) -> f64 {
        assert!(df > 0.0, "df must be positive");
        assert!(fwd > 0.0, "Forward must be positive");
        assert!(vol >= 0.0, "Variance must be non-negative");
        assert!(k >= 0.0, "Strike must be non-negative");
        let sqrt_variance = vol * ttm .sqrt();
        if ttm <= 0.0 {
            return 0.0
        }
        if sqrt_variance == 0.0 {
            return match q {
                PutOrCall::Put => if k > fwd {-df} else {0.0}
                PutOrCall::Call => if fwd > k {df} else {0.0}
            }
        }
        let d1 = ((fwd/k).ln() + 0.5 * sqrt_variance  * sqrt_variance)/ sqrt_variance;
        let nd1 = self.normal.cdf(d1);
        return match q {
            PutOrCall::Put => df * (nd1 - 1.0),
            PutOrCall::Call => df * nd1,
        }
    }

    pub fn vega(&self, df: f64, fwd: f64, k: f64, vol: f64, ttm: f64) -> f64 {
        assert!(df > 0.0, "df must be positive");
        assert!(fwd > 0.0, "Forward must be positive");
        assert!(vol >= 0.0, "Variance must be non-negative");
        assert!(k >= 0.0, "Strike must be non-negative");
        if ttm <= 0.0 {
            return 0.0
        }
        let sqrt_variance = vol * ttm .sqrt();
        if sqrt_variance == 0.0 {
            return 0.0;
        }
        let d1 = ((fwd/k).ln() + 0.5 * sqrt_variance  * sqrt_variance)/ sqrt_variance;
        return df * fwd * self.normal.pdf(d1) * ttm.sqrt();
    }

    pub fn gamma(&self, df: f64, fwd: f64, k: f64, vol: f64, ttm: f64) -> f64 {
        assert!(df > 0.0, "df must be positive");
        assert!(fwd > 0.0, "Forward must be positive");
        assert!(vol >= 0.0, "Variance must be non-negative");
        assert!(k >= 0.0, "Strike must be non-negative");
        if ttm <= 0.0 {
            return 0.0
        }
        let sqrt_variance = vol * ttm .sqrt();
        if sqrt_variance == 0.0 {
            return 0.0;
        }
        let d1 = ((fwd/k).ln() + 0.5 * sqrt_variance  * sqrt_variance)/ sqrt_variance;
        return df * self.normal.pdf(d1) /(fwd *  sqrt_variance)
    }

    pub fn theta(&self, df: f64, fwd: f64,k: f64, vol: f64, ttm: f64, q: PutOrCall) -> f64 {
        assert!(df > 0.0, "df must be positive");
        assert!(fwd > 0.0, "Forward must be positive");
        assert!(vol >= 0.0, "Variance must be non-negative");
        assert!(k >= 0.0, "Strike must be non-negative");
        if ttm <= 0.0 {
            return 0.0;
        }
        let sqrt_variance = vol * ttm .sqrt();
        let r = df.ln().neg() / ttm;
        if sqrt_variance == 0.0 {
            let r = df.ln().neg() / ttm;
            let term1 = r * fwd * df;
            let term2 = r * k * df;
            return match q {
                PutOrCall::Put => if k > fwd {term2 - term1} else {0.0},
                PutOrCall::Call => if fwd > k {term1 - term2} else {0.0},
            }
        }
        let (d1, d2) = d_plus_minus((fwd/k).ln(), sqrt_variance);
        let term1 = -fwd * sqrt_variance * self.normal.pdf(d1) * df / (2.0 * ttm);
        let term2 = r * fwd * df;
        let term3 = r * k * df;
        return match q {
            PutOrCall::Put => term1 - term2 * self.normal.cdf(-d1) + term3 * self.normal.cdf(-d2),
            PutOrCall::Call => term1 + term2 * self.normal.cdf(d1) - term3 * self.normal.cdf(d2)
        }
    }

    pub fn rho(&self, df: f64, fwd: f64, k: f64, vol: f64, ttm : f64, q: PutOrCall) -> f64 {
        let sqrt_variance = vol * ttm .sqrt();
        let price = self.price(df, fwd, k, vol, ttm, q);
        return -ttm * price;
    }

    pub fn implied_vol(price: f64, df: f64, fwd: f64, strike: f64, ttm: f64, q: PutOrCall) -> f64 {
        unsafe {
            implied_volatility_from_a_transformed_rational_guess(price/df, fwd, strike, ttm, q as i32 as f64)
        }
    }

    pub fn cdf(&self, x: f64) -> f64 {
        self.normal.cdf(x)
    }
}

pub struct Bachelier {
    normal: Normal
}

impl Bachelier {
    pub fn new() ->  Result<Bachelier, qm::Error> {

        // For some reason we cannot use the ? operator for this sort of error.
        // as a workaround, do it manually for now.
        match Normal::new(0.0, 1.0) {
            Ok(normal) => Ok(Bachelier { normal }),
            Err(e) => Err(qm::Error::new(&format!("RSStat error: {}", e)))
        }
    }

    pub fn price(&self, df: f64, fwd: f64, k: f64, vol: f64, ttm: f64, q: PutOrCall) -> f64 {
        assert!(df > 0.0, "df must be positive");
        assert!(vol >= 0.0, "Variance must be non-negative");
        let sqrt_variance = vol * ttm .sqrt();
        if sqrt_variance == 0.0 {
            return discounted_intrinsinc(df, fwd, k, q);
        }

        let d = (fwd - k) / sqrt_variance;
        return match q {
            PutOrCall::Call => {
                df * ((fwd - k) * self.normal.cdf(d) + sqrt_variance * self.normal.pdf(d))
            }
            PutOrCall::Put => {
                df * ((k - fwd) * self.normal.cdf(-d) + sqrt_variance * self.normal.pdf(d))
            }
        }
    }

    /// Calculates the PV of a European call option under Bachelier
    pub fn call_price(&self, df: f64, fwd: f64, k: f64, vol: f64, ttm: f64) -> f64 {
        self.price(df, fwd, k, vol, ttm, PutOrCall::Call)
    }
    /// Calculates the PV of a European put option under Bachelier
    pub fn put_price(&self, df: f64, fwd: f64, k: f64, vol: f64, ttm: f64) -> f64 {
        self.price(df, fwd, k, vol, ttm, PutOrCall::Put)
    }

    pub fn delta(&self, df: f64, fwd: f64, k:f64, vol: f64, ttm: f64,  q: PutOrCall) -> f64 {
        assert!(df > 0.0, "df must be positive");
        assert!(vol >= 0.0, "Variance must be non-negative");
        if ttm <= 0.0 {
            return 0.0;
        }
        let sqrt_variance = vol * ttm .sqrt();
        if sqrt_variance == 0.0 {
            return match q {
                PutOrCall::Call => if fwd > k {df} else {0.0},
                PutOrCall::Put => if k > fwd {-df} else {0.0},
            }
        }
        let d = (fwd - k) / sqrt_variance;
        let delta = self.normal.cdf(d);
        return match q {
            PutOrCall::Call => df * delta,
            PutOrCall::Put => df * (1.0 - delta),
        }
    }

    pub fn vega(&self, df: f64, fwd: f64, k:f64, vol: f64, ttm: f64) -> f64 {
        assert!(df > 0.0, "df must be positive");
        assert!(vol >= 0.0, "Variance must be non-negative");
        if vol == 0.0 || ttm <= 0.0 {
            return 0.0;
        }
        let sqrt_variance = vol * ttm .sqrt();
        let d = (fwd - k) / sqrt_variance;
        return df * self.normal.pdf(d) * ttm.sqrt();
    }

    pub fn gamma(&self, df: f64, fwd: f64, k:f64, vol: f64, ttm: f64) -> f64 {
        assert!(df > 0.0, "df must be positive");
        assert!(vol >= 0.0, "Variance must be non-negative");
        if ttm <= 0.0 {
            return 0.0;
        }
        let sqrt_variance = vol * ttm .sqrt();
        let d = (fwd - k) / sqrt_variance;
        return df * self.normal.pdf(d) / sqrt_variance;
    }

    pub fn theta(&self, df: f64, fwd: f64,k: f64, vol: f64, ttm: f64, q: PutOrCall) -> f64 {
        assert!(df > 0.0, "df must be positive");
        assert!(vol >= 0.0, "Variance must be non-negative");
        if ttm < 0.0 {
            return 0.0;
        }
        let sqrt_variance = vol * ttm .sqrt();
        let r = df.ln().neg() / ttm;

        if sqrt_variance == 0.0 {
            if ttm == 0.0 {
                return 0.0;
            }
            let price = self.price(df, fwd, k, vol, ttm, q);
            return r * price;
        }
        let d = (fwd - k) / sqrt_variance;
        let price = self.price(df, fwd, k, vol, ttm, q);
        let term = df * vol * self.normal.pdf(d) / (2.0 * ttm.sqrt());
        return r * price - term;
    }

    pub fn rho(&self, df: f64, fwd: f64, k: f64, vol: f64, ttm : f64, q: PutOrCall) -> f64 {
        if ttm < 0.0 {
            return 0.0;
        }
        let sqrt_variance = vol * ttm .sqrt();
        let price = self.price(df, fwd, k, vol, ttm, q);
        return -ttm * price;
    }

    pub fn implied_vol(price: f64, df: f64, fwd: f64, k: f64, ttm: f64, q: PutOrCall) -> f64 {
        const SQRT_2PI : f64 = 2.50662827463100024161235523934010;

        if (fwd - k) < 1e-14 {
            return price * SQRT_2PI / (ttm.sqrt() * df);
        }
        let mut price = price;
        if q == PutOrCall::Call && fwd > k {
            price -= df * (fwd - k);
        } else if q == PutOrCall::Put && k > fwd {
            price += df * (fwd - k);
        }
        return Bachelier::imply_vol_otm(price, df, fwd,  k, ttm);
    }

    fn imply_vol_otm(price: f64, df: f64, fwd: f64, k: f64, ttm: f64) -> f64 {
        let phi = -((price/df)/(fwd - k)).abs();
        const PHI_THRESHOLD: f64 = -0.001882039271;
        let x =
            if phi < PHI_THRESHOLD {
                let g = 1.0 / (phi - 0.5);
                let  xi_numerator = 0.032114372355 - g * g * (0.016969777977 - g * g * (2.6207332461E-3 - 9.6066952861E-5 * g * g));
                let xi_denominator = 1.0 - g * g *  (0.6635646938 - g * g * (0.14528712196 - 0.010472855461 * g * g));
                let xi = xi_numerator/ xi_denominator;
                g * (1.0 / (2.0 * f64::PI()).sqrt() + xi * g * g)
            } else {
                let h = (-(-phi).ln()).sqrt();
                let x_numerator = 9.4883409779 - h * (9.6320903635 - h * (0.58556997323 + 2.1464093351 * h));
                let x_denominator = 1.0 - h *  (0.65174820867 + h * (1.5120247828 + 6.6437847132E-5 * h));
                x_numerator / x_denominator
            };
        let normal : Normal = Normal::new(0.0, 1.0).unwrap();
        let q = ((normal.cdf(x) + normal.pdf(x) / x) - phi) / normal.pdf(x);
        let x_star_numerator = 3.0 * q * x * x * (2.0 - q * x * (2.0 + x * x));
        let x_star_denominator = 6.0 + q * x * (-12.0 + x * (6.0 * q + x * (-6.0 + q * x * (3.0 + x * x))));
        let x_star = x + x_star_numerator / x_star_denominator;
        return ((k - fwd) / (x_star * ttm.sqrt())).abs();
    }


}

/// Calculates the internal d_plus and d_minus values needed for many of the
/// Black Scholes formulae.
fn d_plus_minus(log_moneyness: f64, sqrt_variance: f64) -> (f64, f64) {
    let d_plus = log_moneyness / sqrt_variance + 0.5 * sqrt_variance;
    let d_minus = d_plus - sqrt_variance;
    (d_plus, d_minus)
}

fn discounted_intrinsinc(df: f64, fwd: f64, k: f64, q: PutOrCall) -> f64 {
    match q {
        PutOrCall::Call => {
            df * (fwd - k).max(0.0)
        }
        PutOrCall::Put => {
            df * (k - fwd).max(0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use statrs::assert_almost_eq;
    use super::*;
    use math::numerics::approx_eq;

    #[test]
    fn test_cdf() {
        // checking values against those on danielsoper.com
        let black76 = Black76::new().unwrap();
        assert_approx(black76.cdf(-4.0), 0.00003167, 1e-8, "cdf"); 
        assert_approx(black76.cdf(-3.0), 0.00134990, 1e-8, "cdf"); 
        assert_approx(black76.cdf(-2.0), 0.02275013, 1e-8, "cdf"); 
        assert_approx(black76.cdf(-1.0), 0.15865525, 1e-8, "cdf"); 
        assert_approx(black76.cdf(0.0), 0.5, 1e-8, "cdf"); 
        assert_approx(black76.cdf(1.0), 0.84134475, 1e-8, "cdf"); 
        assert_approx(black76.cdf(2.0), 0.97724987, 1e-8, "cdf"); 
        assert_approx(black76.cdf(3.0), 0.99865010, 1e-8, "cdf"); 
        assert_approx(black76.cdf(4.0), 0.99996833, 1e-8, "cdf"); 
    }

    #[test]
    fn black76_price() {

        let forward = 100.0;
        let df = 0.99;
        let vol = 0.5;
        let ttm = 1.0;
        let black76 = Black76::new().unwrap();

        for strike in [50.0, 70.0, 90.0, 100.0, 110.0, 130.0, 160.0].iter() {
            let call_price = black76.call_price(df, forward, *strike, vol, ttm);
            let put_price = black76.put_price(df, forward, *strike, vol, ttm);

            let call_intrinsic = df * (forward - *strike).max(0.0);
            let put_intrinsic = df * (*strike - forward).max(0.0);
            let parity = df * (forward - *strike) + put_price - call_price;

            assert!(call_price >= call_intrinsic);
            assert!(put_price >= put_intrinsic);
            assert_approx(parity, 0.0, 1e-12, "put/call parity");
        }
    }

    #[test]
    fn bachelier_price() {
        let forward = 100.0;
        let df = 0.99;
        let vol = 10.0;
        let ttm = 1.0;
        let bachelier = Bachelier::new().unwrap();

        for strike in [50.0, 70.0, 90.0, 100.0, 110.0, 130.0, 160.0].iter() {
            let call_price = bachelier.call_price(df, forward, *strike, vol, ttm);
            let put_price = bachelier.put_price(df, forward, *strike, vol, ttm);

            let call_intrinsic = df * (forward - *strike).max(0.0);
            let put_intrinsic = df * (*strike - forward).max(0.0);
            let parity = df * (forward - *strike) + put_price - call_price;

            assert!(call_price >= call_intrinsic);
            assert!(put_price >= put_intrinsic);
            assert_approx(parity, 0.0, 1e-12, "put/call parity");
        }
    }

    fn assert_approx(value: f64, expected: f64, tolerance: f64, message: &str) {
        assert!(approx_eq(value, expected, tolerance),
            "{}: value={} expected={}", message, value, expected);
    }


    #[test]
    fn test_lbr() {
        let t = 1.0;
        let cp = PutOrCall::Call;
        let df = 0.99;
        let f = 100.0;
        let k = 100.0;
        let vol = 0.3;
        let ttm = 1.0;
        let black76 = Black76::new().unwrap();
        let p = black76.call_price(df, f, k, vol, ttm);
        let v = Black76::implied_vol(p, df, f, k, t, cp);

        assert_almost_eq!(v, vol, 1e-12);
    }


}

extern {
    fn implied_volatility_from_a_transformed_rational_guess(
        price: f64,
        fwd: f64,
        strike: f64,
        ttm: f64,
        cp: f64
    ) -> f64;
}