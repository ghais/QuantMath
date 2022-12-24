use data::volatility::strike::LogRelStrike;
use data::volatility::volsmile::{b76_price, VolSmile, RSVI};
use instruments::options::PutOrCall;
use levenberg_marquardt::{differentiate_numerically, LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::base::dimension::Dim;
use nalgebra::base::DMatrix;
use nalgebra::{
    dimension::{U1, U2},
    storage::Owned,
    Matrix, Matrix2, OMatrix, OVector, VecStorage, Vector2, Vector5, U5,
};

use ndarray::ArrayBase;
use num_traits::Zero;
use rand::distributions::Uniform;
use rand::{distributions::Distribution, Rng};
use std::rc::Rc;

type F = f64;

#[derive(Debug, Clone)]
pub struct SviFit<'a> {
    pub prices: &'a Vec<(PutOrCall, LogRelStrike, f64)>,
    pub svi: RSVI,
    pub fwd: f64,
    pub ttm: f64,
    pub df: f64,
}

impl<'a> SviFit<'a> {
    fn into_vec(&self) -> Vector5<F> {
        Vector5::new(
            self.svi.a,
            self.svi.b,
            self.svi.rho,
            self.svi.m,
            self.svi.sigma,
        )
    }

    fn from_vec(&self, v: &Vector5<F>) -> RSVI {
        RSVI::new(self.ttm, self.fwd, v[0], v[1], v[2], v[3], v[4])
    }

    pub fn fit(&mut self) -> RSVI {
        let (fit, report) = LevenbergMarquardt::new().minimize(self.clone());
        dbg!(report);
        fit.svi
    }
}

impl<'a> LeastSquaresProblem<F, nalgebra::base::dimension::Dynamic, U5> for SviFit<'a> {
    type ResidualStorage = Owned<F, nalgebra::base::dimension::Dynamic, U1>;
    type JacobianStorage = Owned<F, nalgebra::base::dimension::Dynamic, U5>;
    type ParameterStorage = Owned<F, U5, U1>;

    fn set_params(&mut self, p: &Vector5<F>) {
        self.svi = self.from_vec(p);
    }

    fn params(&self) -> Vector5<F> {
        self.into_vec()
    }

    fn residuals(&self) -> Option<OVector<f64, nalgebra::base::dimension::Dynamic>> {
        let m = nalgebra::base::dimension::Dynamic::from_usize(self.prices.len());
        let n = Dim::from_usize(1);
        Some(
            OVector::<f64, nalgebra::base::dimension::Dynamic>::from_iterator_generic(
                m,
                n,
                self.prices.iter().map(|(q, x, p)| {
                    let price = b76_price(&self.svi, self.df, self.fwd, *x, self.ttm, *q);
                    return *p - price;
                }),
            ),
        )
    }

    fn jacobian(&self) -> Option<OMatrix<F, nalgebra::base::dimension::Dynamic, U5>> {
        let mut p = self.clone();
        differentiate_numerically(&mut p)
    }
}
