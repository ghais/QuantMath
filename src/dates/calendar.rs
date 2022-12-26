use chrono::{Datelike, Days};
use core::factories::Qrc;
use core::factories::Registry;
use core::factories::TypeId;
use dates::datetime::DateDayFraction;
use dates::Date;
use erased_serde as esd;
use num_traits::AsPrimitive;
use serde as sd;
use serde::Deserialize;
use serde_tagged as sdt;
use serde_tagged::de::BoxFnSeed;
use std::cmp::max;
use std::fmt::Debug;
use std::sync::Arc;

/// Calendars define when business holidays are scheduled. They are used for
/// business day volatility, settlement calculations, and the roll-out of
/// schedules for exotic products and swaps.

pub trait Calendar: esd::Serialize + TypeId + Sync + Send + Debug {
    /// The name of this calendar. Conventionally, the name is a three-letter
    /// upper-case string such as "TGT" or "NYS", though this is not required.
    fn name(&self) -> &str;

    /// Is the given date a holiday or weekend? The opposite of a business day.
    fn is_holiday(&self, date: Date) -> bool;

    /// Count business days between the two given dates. The dates at the ends
    /// of the range may optionally be partly included in the count, by setting
    /// a fraction between zero and one.
    fn count_business_days(
        &self,
        from: Date,
        from_fraction: f64,
        to: Date,
        to_fraction: f64,
    ) -> f64;

    /// Step forward/backward by the given number of business days. The result
    /// is always a business day. If today is a business day, a zero step
    /// means staying where we are. If today is not a business day, we slip
    /// off to the nearest business day in the direction specified by the
    /// slip_forward flag. If the step is non-zero, the overall move is
    /// equivalent to a zero step followed by the given number of one-step
    /// moves.
    ///
    /// Note that sadly we cannot use the sign of the step argument to specify
    /// the slip_forward flag, because the step may be zero. Effectively, we
    /// want to distinguish between +0 and -0. The function works correctly
    /// when the sign of step is different from the slip_forward flag, but
    /// the results are rather weird. Generally you want slip_forward = true
    /// if step > 0 and vice versa.
    fn step(&self, from: Date, step: i32, slip_forward: bool) -> Date;

    /// Returns the basis normally used for business day volatility. Roughly
    /// speaking this is the expected number of business days in a year, but
    /// conventionalised. It is 365 for EveryDayCalendar and 252 for a pure
    /// business day calendar.
    fn standard_basis(&self) -> f64;

    /// If the given date is a holiday, what weight should it receive? Returns
    /// 1.0 for a business day, 0.0 for a pure holiday with no weight, and
    /// potentially a number somewhere between for anything else.
    ///
    /// The default implementation returns zero for a holiday and one for
    /// any other day.
    fn day_weight(&self, date: Date) -> f64 {
        if self.is_holiday(date) {
            0.0
        } else {
            1.0
        }
    }

    /// Specialised function where day-counts may be fractional, for example
    /// in volatility surfaces. For most calendars, the step is truncated
    /// towards zero and an integer number of steps is taken.
    fn step_partial(&self, from: Date, step: f64, slip_forward: bool) -> Date {
        self.step(from, step.round() as i32, slip_forward)
    }

    /// Calculate the year-fraction between two date-times, given the count of
    /// business days and the standard basis.
    fn year_fraction(&self, from: DateDayFraction, to: DateDayFraction) -> f64 {
        let days = self.count_business_days(
            from.date(),
            from.day_fraction(),
            to.date(),
            to.day_fraction(),
        );
        let basis = self.standard_basis();
        days / basis
    }
}

// Get serialization to work recursively for rate curves by using the
// technology defined in core/factories. RcCalendar is a container
// class holding an Rc<Calendar>
pub type RcCalendar = Qrc<Calendar>;
pub type TypeRegistry = Registry<BoxFnSeed<RcCalendar>>;

/// Implement deserialization for subclasses of the type
impl<'de> sd::Deserialize<'de> for RcCalendar {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: sd::Deserializer<'de>,
    {
        sdt::de::external::deserialize(deserializer, get_registry())
    }
}

/// Return the type registry required for deserialization.
pub fn get_registry() -> &'static TypeRegistry {
    lazy_static! {
        static ref REG: TypeRegistry = {
            let mut reg = TypeRegistry::new();
            reg.insert(
                "EveryDayCalendar",
                BoxFnSeed::new(EveryDayCalendar::from_serial),
            );
            reg.insert(
                "WeekdayCalendar",
                BoxFnSeed::new(WeekdayCalendar::from_serial),
            );
            reg.insert(
                "WeekdayAndHolidayCalendar",
                BoxFnSeed::new(WeekdayAndHolidayCalendar::from_serial),
            );
            reg.insert(
                "VolatilityCalendar",
                BoxFnSeed::new(VolatilityCalendar::from_serial),
            );
            reg
        };
    }
    &REG
}

/// An every-day calendar assumes that all days are business days, including
/// weekends.
#[derive(Serialize, Deserialize, Debug)]
pub struct EveryDayCalendar();

impl TypeId for EveryDayCalendar {
    fn get_type_id(&self) -> &'static str {
        "EveryDayCalendar"
    }
}

impl EveryDayCalendar {
    pub fn new() -> EveryDayCalendar {
        EveryDayCalendar {}
    }

    pub fn from_serial(de: &mut esd::Deserializer) -> Result<RcCalendar, esd::Error> {
        Ok(Qrc::new(Arc::new(EveryDayCalendar::deserialize(de)?)))
    }
}

impl Calendar for EveryDayCalendar {
    fn name(&self) -> &str {
        "ALL" // a conventional string that does not clash with
              // anything in financialcalendars.com
    }

    fn is_holiday(&self, _date: Date) -> bool {
        false
    }

    fn count_business_days(
        &self,
        from: Date,
        from_fraction: f64,
        to: Date,
        to_fraction: f64,
    ) -> f64 {
        // return zero if the dates are in the wrong order
        if from > to {
            return 0.0;
        }

        let whole_days = (to - from).num_days() as f64;
        whole_days + to_fraction - from_fraction
    }

    fn step(&self, from: Date, step: i32, _slip_forward: bool) -> Date {
        if step > 0 {
            from + Days::new(step.abs() as u64)
        } else {
            from - Days::new(step.abs() as u64)
        }
    }

    fn standard_basis(&self) -> f64 {
        // This matches the standard day count convention, Act365F. Other
        // less common alternatives include 365.25 and the number of days
        // in the current actual year (365 or 366).
        365.0
    }
}

/// A weekday calendar assumes that Monday to Friday are business days, and
/// Saturday and Sunday are not.
#[derive(Serialize, Deserialize, Debug)]
pub struct WeekdayCalendar();

impl TypeId for WeekdayCalendar {
    fn get_type_id(&self) -> &'static str {
        "WeekdayCalendar"
    }
}

impl WeekdayCalendar {
    pub fn new() -> WeekdayCalendar {
        WeekdayCalendar {}
    }

    pub fn from_serial(de: &mut esd::Deserializer) -> Result<RcCalendar, esd::Error> {
        Ok(Qrc::new(Arc::new(WeekdayCalendar::deserialize(de)?)))
    }
}

impl Calendar for WeekdayCalendar {
    fn name(&self) -> &str {
        "WKD" // a conventional string that does not clash with
              // anything in financialcalendars.com
    }

    fn is_holiday(&self, date: Date) -> bool {
        date.weekday().num_days_from_monday() > 4
    }

    fn count_business_days(
        &self,
        from: Date,
        from_fraction: f64,
        to: Date,
        to_fraction: f64,
    ) -> f64 {
        // cope with starting or ending on a weekend by slipping forward
        // (at the from date) or backward (at the to date)
        let adj_from = slip_to_next_weekday(from, true);
        let adj_to = slip_to_next_weekday(to, false);

        // if after adjustment, the dates are in the wrong order,
        // just return zero.
        if adj_from > adj_to {
            return 0.0;
        }

        // days from whole weeks
        let whole_days = (adj_to - adj_from).num_days();
        let whole_weeks = whole_days / 7;
        let week_days = whole_weeks * 5;

        // days from partial weeks
        let diff = adj_to.weekday().num_days_from_monday() as i64
            - adj_from.weekday().num_days_from_monday() as i64;
        let partial_week_days = if diff >= 0 { diff } else { 5 + diff };

        // corrections at the ends if the fractions are not one
        let from_is_holiday = from.weekday().num_days_from_monday() > 4;
        let to_is_holiday = to.weekday().num_days_from_monday() > 4;
        let from_adj = if from_is_holiday { 0.0 } else { from_fraction };
        let to_adj = if to_is_holiday { 1.0 } else { to_fraction };
        let adj = to_adj - from_adj;

        (week_days + partial_week_days) as f64 + adj
    }

    fn step(&self, from: Date, step: i32, slip_forward: bool) -> Date {
        // slip off to the nearest business day in the required direction
        let adj_from = slip_to_next_weekday(from, slip_forward);
        if step == 0 {
            return adj_from; // optimise the common case of a zero step
        }

        // Step the required number of business days. We know that for every
        // five business days we have to add two weekend days.
        let week_day = adj_from.weekday().num_days_from_monday() as i32;
        let calendar_days = if step > 0 {
            // requested days plus two for each complete week
            step + ((week_day + step) / 5) * 2
        } else {
            // requested days plus two for each complete week
            // (use 4 - week_day as the start, to reverse the week days)
            step - ((4 - week_day - step) / 5) * 2
        };
        let to = if calendar_days > 0 {
            adj_from + Days::new(calendar_days as u64)
        } else {
            adj_from - Days::new(calendar_days.abs() as u64)
        };

        // Sort out the final location. We cannot use slip_to_next_weekday.
        // For example, if we step one business day from Saturday, we need
        // to end up on the Tuesday, not Monday.
        let week_day = to.weekday().num_days_from_monday();
        if week_day < 5 {
            return to; // already on a business day
        } else if step > 0 {
            return to + Days::new(2); // following Monday or Tuesday
        } else {
            return to - Days::new(2); // preceding Friday or Thursday
        }
    }

    fn standard_basis(&self) -> f64 {
        // It is normal to use a day count of Act252 for pure business day
        // vol (with zero vol time at weekends or holidays). When the holiday
        // or weekend vol time is non-zero, it is normal to adjust the basis.
        252.0
    }
}

// private helper function to step forward or backward to the nearest
// Monday to Friday
fn slip_to_next_weekday(from: Date, slip_forward: bool) -> Date {
    let week_day = from.weekday().num_days_from_monday();
    if week_day < 5 {
        return from; // already on a business day
    } else if slip_forward {
        return from + Days::new((7 - week_day) as u64); // following Monday
    } else {
        return from - Days::new((week_day - 4) as u64); // preceding Friday
    }
}

/// A calendar that assumes that Saturday and Sunday are not business days,
/// together with a specified list of business holidays. In general this list
/// is read from a file.
#[derive(Serialize, Deserialize, Debug)]
pub struct WeekdayAndHolidayCalendar {
    name: String,
    holidays: Vec<Date>, // must be in date order, with no duplicates
                         // and no weekend dates
}

impl TypeId for WeekdayAndHolidayCalendar {
    fn get_type_id(&self) -> &'static str {
        "WeekdayAndHolidayCalendar"
    }
}

impl WeekdayAndHolidayCalendar {
    pub fn new(name: &str, holidays: &[Date]) -> WeekdayAndHolidayCalendar {
        WeekdayAndHolidayCalendar {
            name: name.to_string(),
            holidays: holidays.to_vec(),
        }
    }

    pub fn from_serial<'de>(de: &mut esd::Deserializer<'de>) -> Result<RcCalendar, esd::Error> {
        Ok(Qrc::new(Arc::new(WeekdayAndHolidayCalendar::deserialize(
            de,
        )?)))
    }

    // Finds the next holiday, including the day we start from, and returns
    // its offset in the vector. Also returns a bool to say whether
    // the next holiday was on the day we asked for.
    fn remaining_holidays(&self, from: Date) -> (usize, bool) {
        // consider holding a key to the first holiday for each year (say)
        // as a way of bypassing some of this search.
        match self.holidays.binary_search(&from) {
            Ok(offset) => (offset, true),
            Err(offset) => (offset, false),
        }
    }

    // slide off a holiday or weekend onto the nearest business day,
    // stepping either forward or backward. (Step is +/- 1.)
    fn slip_to_next(&self, from: Date, step: i32) -> Date {
        // Consider a more efficient implementation here. is_holiday looks
        // at the holidays vector each time we invoke it.

        let mut date = from;
        while self.is_holiday(date) {
            if step > 0 {
                date = date + Days::new(step as u64);
            } else {
                date = date + Days::new(step.abs() as u64);
            }
        }
        date
    }

    // Count holidays, not including weekends. This method excludes any
    // holidays on the end dates of the range.
    fn count_holidays(&self, from: Date, to: Date) -> i32 {
        let (start, start_is_holiday) = self.remaining_holidays(from);
        let (end, _) = self.remaining_holidays(to);
        let count = end as i32 - start as i32;

        // subtract holidays at the start of the range, if any (the end is
        // not included anyway)
        if start_is_holiday {
            count - 1
        } else {
            count
        }
    }
}

impl Calendar for WeekdayAndHolidayCalendar {
    fn name(&self) -> &str {
        &self.name
    }

    /// TODO: is_holiday is an expensive call for WeekendAndHolidayCalendar.
    /// We still need to implement it for external users, but we should avoid
    /// calling it internally.
    fn is_holiday(&self, date: Date) -> bool {
        if date.weekday().num_days_from_monday() > 4 {
            return true;
        }

        let (_, holiday) = self.remaining_holidays(date);
        holiday
    }

    fn count_business_days(
        &self,
        from: Date,
        from_fraction: f64,
        to: Date,
        to_fraction: f64,
    ) -> f64 {
        // cope with starting or ending on a weekend or holiday
        // by slipping forward (at the from date) or backward (at the to date)
        let adj_from = self.slip_to_next(from, 1);
        let adj_to = self.slip_to_next(to, -1);

        // if after adjustment, the dates are in the wrong order,
        // just return zero.
        if adj_from > adj_to {
            return 0.0;
        }

        // days from whole weeks
        let whole_days = (adj_to - adj_from).num_days();
        let whole_weeks = whole_days / 7;
        let week_days = whole_weeks * 5;

        // days from partial weeks
        let diff = adj_to.weekday().num_days_from_monday() as i64
            - adj_from.weekday().num_days_from_monday() as i64;
        let partial_week_days = if diff >= 0 { diff } else { 5 + diff };

        // count any holidays in the range (we know the ends of the range
        // are not holidays)
        let holiday_count = self.count_holidays(adj_from, adj_to) as i64;

        // corrections at the ends if the fractions are not one
        let from_is_holiday = self.is_holiday(from);
        let to_is_holiday = self.is_holiday(to);
        let from_adj = if from_is_holiday { 0.0 } else { from_fraction };
        let to_adj = if to_is_holiday { 1.0 } else { to_fraction };
        let adj = to_adj - from_adj;

        (week_days + partial_week_days - holiday_count) as f64 + adj
    }

    fn step(&self, from: Date, step: i32, slip_forward: bool) -> Date {
        // slip off to the nearest business day in the required direction
        let direction = if slip_forward { 1 } else { -1 };
        let adj_from = self.slip_to_next(from, direction);
        if step == 0 {
            return adj_from; // optimise the common case of a zero step
        }

        // Step the required number of business days. We know that for every
        // five business days we have to add two weekend days.
        let week_day = adj_from.weekday().num_days_from_monday() as i32;
        let calendar_days = if step > 0 {
            // requested days plus two for each complete week
            step + ((week_day + step) / 5) * 2
        } else {
            // requested days plus two for each complete week
            // (use 4 - week_day as the start, to reverse the week days)
            step - ((4 - week_day - step) / 5) * 2
        };
        let to = if calendar_days > 0 {
            adj_from + Days::new(calendar_days as u64)
        } else {
            adj_from - Days::new(calendar_days.abs() as u64)
        };

        // We have a sort of Zeno's paradox here. When we step forward to
        // add the effect of a holiday, we may step over a holiday, which
        // then means we have to step forward yet again. Rather than some
        // complicated logic, we just fall back to a simple loop.
        let week_day = to.weekday().num_days_from_monday() as i32;
        let mut date = to;
        if step > 0 {
            let weekend_days = max(week_day - 4, 0);
            let mut extra = self.count_holidays(adj_from, to) + weekend_days;
            loop {
                if self.is_holiday(date) {
                    date = date + Days::new(1);
                } else if extra > 0 {
                    date = date + Days::new(1);
                    extra = extra - 1;
                } else {
                    return date;
                }
            }
        } else {
            let weekend_days = if week_day == 5 { 1 } else { 0 };
            let mut extra = self.count_holidays(to, adj_from) + weekend_days;
            loop {
                if self.is_holiday(date) {
                    date = date - Days::new(1);
                } else if extra > 0 {
                    date = date - Days::new(1);
                    extra -= 1;
                } else {
                    return date;
                }
            }
        }
    }

    fn standard_basis(&self) -> f64 {
        // It is normal to use a day count of Act252 for pure business day
        // vol (with zero vol time at weekends or holidays). When the holiday
        // or weekend vol time is non-zero, it is normal to adjust the basis.
        252.0
    }
}

/// A volatility calendar can have a non-zero weight for weekends. 25% is
/// common. This affects the basis and the business day count. It also
/// affects the step size. For example, a forward volatility model has its
/// dates rolled forward by a consistent number of business days from the
/// corresponding spot model, and this is achieved by calling the step
/// function. We may end up trying to roll by non-integer numbers of days.
/// In that case, we round to the nearest integer.
#[derive(Serialize, Deserialize, Debug)]
pub struct VolatilityCalendar {
    name: String,
    calendar: RcCalendar,
    holiday_weight: f64, // normally greater than zero and less than one
}

impl TypeId for VolatilityCalendar {
    fn get_type_id(&self) -> &'static str {
        "VolatilityCalendar"
    }
}

impl VolatilityCalendar {
    pub fn new(name: &str, calendar: RcCalendar, holiday_weight: f64) -> VolatilityCalendar {
        VolatilityCalendar {
            name: name.to_string(),
            calendar: calendar,
            holiday_weight: holiday_weight,
        }
    }

    pub fn from_serial<'de>(de: &mut esd::Deserializer<'de>) -> Result<RcCalendar, esd::Error> {
        Ok(Qrc::new(Arc::new(VolatilityCalendar::deserialize(de)?)))
    }
}

impl Calendar for VolatilityCalendar {
    fn name(&self) -> &str {
        &self.name
    }

    fn is_holiday(&self, date: Date) -> bool {
        self.calendar.is_holiday(date)
    }

    fn count_business_days(
        &self,
        from: Date,
        from_fraction: f64,
        to: Date,
        to_fraction: f64,
    ) -> f64 {
        // return zero if the dates are in the wrong order
        if from > to {
            return 0.0;
        }

        // count the business days
        let business_days = self
            .calendar
            .count_business_days(from, from_fraction, to, to_fraction);

        // count the calendar days
        let whole_days = (to - from).num_days() as f64;
        let all_days = whole_days + to_fraction - from_fraction;

        // return one for each business day and the weight for each holiday
        business_days * (1.0 - self.holiday_weight) + all_days * self.holiday_weight
    }

    fn step(&self, from: Date, step: i32, slip_forward: bool) -> Date {
        // Unlike other calendars, it makes sense to take partial steps in
        // a volatility calendar. For example, a step over a holiday is
        // a holiday_weight step, which might be 0.25. We therefore implement
        // step in terms of step_partial, rather than vice versa.
        self.step_partial(from, step as f64, slip_forward)
    }

    fn step_partial(&self, from: Date, step: f64, slip_forward: bool) -> Date {
        // Optimisation if the holiday weight is zero
        if self.holiday_weight < 1e-12 {
            return self.calendar.step(from, step as i32, slip_forward);
        }

        assert!(step == 0.0 || slip_forward == (step > 0.0));
        let total_steps_needed = step.abs();
        let dir = if slip_forward { 1 } else { -1 };

        // always step over any holidays, whatever the weight, so long as
        // there are zero or more steps needed
        let mut date = from;
        let mut steps_needed = total_steps_needed;
        let mut prev_steps = steps_needed;
        let mut prev_date = date;

        // We step in calendar days. This is bound to be the same size
        // or smaller step than we need, so we repeat until we hit the
        // right count or overshoot. We never step less than one day,
        // otherwise we could have an infinite loop.
        while steps_needed > 0.0 {
            prev_steps = steps_needed;
            prev_date = date;
            let whole_steps = (steps_needed + 1e-12).floor() as u64;
            let next_step = whole_steps.max(1);
            date = if slip_forward {
                date + Days::new(next_step)
            } else {
                date - Days::new(next_step)
            };
            let steps_taken = if slip_forward {
                self.count_business_days(from, 0.0, date, 0.0)
            } else {
                self.count_business_days(date, 1.0, from, 1.0)
            };

            /* for debugging
                        println!("from={}:{} prev_date={}:{} date={}:{} next_step={} \
                            steps_taken={} steps_needed={} total_steps_needed={}",
                            from.to_string(), from.day_of_week(),
                            prev_date.to_string(), prev_date.day_of_week(),
                            date.to_string(), date.day_of_week(),
                            next_step, steps_taken, steps_needed, total_steps_needed);
            */

            steps_needed = total_steps_needed - steps_taken;

            // the conservative stepping means we should always be on or
            // before the date we are aiming for, except when single stepping.
            assert!(steps_needed >= -1.0);
        }

        // if we significantly overshot zero, use the closest of the two
        if steps_needed < 0.0 && -steps_needed > prev_steps {
            prev_date
        } else {
            date
        }
    }

    fn standard_basis(&self) -> f64 {
        // calculate a basis on the assumption of standard basis for all
        // business days plus the weight for each standard holiday36=
        self.calendar.standard_basis() * (1.0 - self.holiday_weight) + 365.0 * self.holiday_weight
    }

    fn day_weight(&self, date: Date) -> f64 {
        if self.is_holiday(date) {
            self.holiday_weight
        } else {
            1.0
        }
    }
}
