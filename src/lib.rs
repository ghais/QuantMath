extern crate statrs;
extern crate ndarray;
extern crate nalgebra;
extern crate rand;
extern crate serde;
//extern crate serde_state as serde;
#[macro_use]
extern crate serde_derive;
//#[macro_use]
//extern crate serde_derive_state;
extern crate erased_serde;
extern crate serde_tagged;
extern crate serde_json;
#[macro_use]
extern crate lazy_static;
extern crate void;
extern crate libc;
extern crate finitediff;
extern crate argmin;
extern crate num_traits;
extern crate derive_more;

// listed in dependency order, though this is not essential for compilation
pub mod core;
pub mod math;
pub mod dates;
pub mod data;
pub mod instruments;
pub mod risk;
pub mod models;
pub mod pricers;
pub mod solvers;
pub mod facade;
