pub mod calendar;
pub mod datetime;
pub mod rules;

use chrono::NaiveDate;
use core::qm;
use math::interpolation::Interpolable;
use serde::de::Error;
use serde::Deserialize;
use serde::Deserializer;
use serde::Serialize;
use serde::Serializer;
use std::cmp::Ordering;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Sub;
use std::ops::SubAssign;
use std::str::FromStr;

/// Date represents a date in risk space. In practice, this starts at the
/// open in Tokyo and finishes at the close in Chicago. (There are no
/// significant financial centres further West or East than these.) For
/// products based in only one financial centre, it can be thought of as
/// a local date.
///
/// It is a bad idea to work in any date space other than local, as it makes
/// date maths extremely difficult. For example, what date is two business
/// days after a GMT-based date, if you are working in Chicago?
///
/// It is unusual in financial maths to need to represent times of day. For
/// example, interest payments are generally calculated in integer numbers
/// of days. It is possible to borrow intraday, but the rates of interest
/// are completely different. Times are needed for volatilities -- options
/// expiring at the close are worth more than options expiring at the open.
///
/// Date contains a single number, representing a count of days from some
/// well-defined base date. We use Truncated Julian dates, which have a base
/// date of 1968-05-24T00:00:00. Julian dates in general start at midday,
/// which is not useful to us, but Truncated Julians are offset by 0.5. The
/// base date is recent enough to catch garbage dates, while still allowing
/// any conceivable financial product.
///
/// Internally, we use a 32 bit signed integer. In practice, an unsigned 16
/// bit Julian date would run out in 2147, which is late enough to cope with
/// any realistic financial product. However, it is useful to be able to use
/// signed integer arithmetic without having to worry about overflow, so
/// we use 32 bit signed integers for now.
///
/// We reserve two special dates. Negative infinite date is represented by the
/// most negative negative integer. It sorts before any date. The maximum
/// integer represents infinite date, which sorts after any date. Unknown
/// dates are represented by negative infinite date. (Consider adding a
/// third special date for this, but I don't think it is needed.)

pub type Date = NaiveDate;

impl Interpolable<NaiveDate> for NaiveDate {
    fn interp_diff(&self, other: NaiveDate) -> f64 {
        (other - *self).num_days() as f64
    }

    fn interp_cmp(&self, other: NaiveDate) -> Ordering {
        self.cmp(&other)
    }
}
