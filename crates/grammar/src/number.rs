//! The number lexer: JSON's number grammar, and, for a number a schema
//! bounds or asks a `multipleOf` of, the exact value tracked as a scaled
//! integer so every prefix the model writes can still be completed to a
//! valid value.
//!
//! What it holds exactly, and what it gives up:
//! - `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum` and
//!   `multipleOf` on `integer` and `number` are enforced exactly, as decimal
//!   arithmetic (not the floating point a Python validator uses), on every
//!   prefix: a prefix that no completion can bring inside the constraints is
//!   refused at the digit that made it so.
//! - A constrained number is written in plain decimal notation with at most
//!   18 integer and 18 fraction digits: no exponent. Bounds and steps must be
//!   exact at 18 fraction digits and below 10^19 in magnitude, or the schema
//!   is refused by name. An unconstrained `number` may use an exponent.
//! - `integer` is written as digits only (no `1.0`, no `1e3`).
//! - An unconstrained number has at most 100 integer digits and an exponent
//!   of at most two digits, so every number the grammar admits is a finite
//!   double (JSON allows `1e999`; serde_json refuses it, JavaScript reads
//!   Infinity).

pub(crate) const SCALE_DIGITS: u32 = 18;
pub(crate) const SCALE: i128 = 1_000_000_000_000_000_000;
const MAX_DIGITS: u8 = 18;
const MAX_FREE_INT_DIGITS: u8 = 100;
const MAX_EXP_DIGITS: u8 = 2;
const BOUND_LIMIT: i128 = 10_000_000_000_000_000_000 * SCALE; // 10^19, scaled

pub(crate) const START: u8 = 0;
const MINUS: u8 = 1;
const ZERO: u8 = 2;
const INT: u8 = 3;
const DOT: u8 = 4;
const FRAC: u8 = 5;
const EXP: u8 = 6;
const EXP_SIGN: u8 = 7;
const EXP_DIG: u8 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct NumState {
    pub phase: u8,
    pub neg: bool,
    pub int_digits: u8,
    pub frac_digits: u8,
    pub exp_digits: u8,
    /// The magnitude written so far, times 10^18, as two halves (keeps the
    /// frame 8-aligned).
    pub mag: [u64; 2],
}

impl NumState {
    pub const fn start() -> Self {
        Self { phase: START, neg: false, int_digits: 0, frac_digits: 0, exp_digits: 0, mag: [0, 0] }
    }
    fn mag(&self) -> i128 {
        ((self.mag[0] as u128) | ((self.mag[1] as u128) << 64)) as i128
    }
    fn set_mag(&mut self, v: i128) {
        let u = v as u128;
        self.mag = [u as u64, (u >> 64) as u64];
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NumSpec {
    pub integer: bool,
    /// Lower bound on the value, scaled, and whether it is exclusive.
    pub lo: Option<(i128, bool)>,
    pub hi: Option<(i128, bool)>,
    /// `multipleOf`, scaled.
    pub mult: Option<i128>,
    /// The step every valid value is a multiple of (`multipleOf`, and 10^18
    /// for an integer), when the value is tracked.
    step: Option<i128>,
    pub tracked: bool,
}

pub(crate) enum NumStep {
    Inside(NumState),
    /// The byte is not part of the number, and the number written is a
    /// complete valid value: the frame pops and the byte goes to its parent.
    End,
    Reject,
}

/// An exact decimal from its JSON text (an exponent shifts the point),
/// scaled by 10^18: `None` when it has more than 18 nonzero fraction digits
/// or is not below 10^19 in magnitude.
pub(crate) fn parse_scaled(text: &str) -> Option<i128> {
    let (neg, body) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text),
    };
    let (mantissa, exp) = match body.split_once(['e', 'E']) {
        Some((m, e)) => (m, e.strip_prefix('+').unwrap_or(e).parse::<i64>().ok()?),
        None => (body, 0),
    };
    let (int, frac) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    if int.is_empty() || !int.bytes().all(|b| b.is_ascii_digit()) || !frac.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let digits = format!("{int}{frac}");
    if digits.bytes().all(|b| b == b'0') {
        return Some(0);
    }
    // Past these the value is out of range or below 10^-18 whatever its
    // digits; refusing early keeps the padding below small.
    if !(-64..=64).contains(&exp) {
        return None;
    }
    // The decimal point sits `point` digits into `digits`.
    let point = int.len() as i64 + exp;
    let (int, frac) = if point <= 0 {
        (String::new(), format!("{}{digits}", "0".repeat((-point) as usize)))
    } else if point as usize >= digits.len() {
        (format!("{digits}{}", "0".repeat(point as usize - digits.len())), String::new())
    } else {
        (digits[..point as usize].to_string(), digits[point as usize..].to_string())
    };
    let frac = frac.trim_end_matches('0');
    if frac.len() > SCALE_DIGITS as usize {
        return None;
    }
    let int = int.trim_start_matches('0');
    if int.len() > 19 {
        return None;
    }
    let mut v: i128 = 0;
    for b in int.bytes() {
        v = v * 10 + (b - b'0') as i128;
    }
    v *= SCALE;
    let mut place = SCALE / 10;
    for b in frac.bytes() {
        v += (b - b'0') as i128 * place;
        place /= 10;
    }
    if v >= BOUND_LIMIT {
        return None;
    }
    Some(if neg { -v } else { v })
}

/// A scaled value as the shortest plain decimal: no exponent, no trailing
/// fraction zeros.
pub(crate) fn format_scaled(v: i128) -> String {
    let (sign, m) = if v < 0 { ("-", -v) } else { ("", v) };
    let frac = format!("{:018}", m % SCALE);
    let frac = frac.trim_end_matches('0');
    if frac.is_empty() {
        format!("{sign}{}", m / SCALE)
    } else {
        format!("{sign}{}.{frac}", m / SCALE)
    }
}

fn gcd(a: i128, b: i128) -> i128 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Why [`NumSpec::new`] cannot hold a set of constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NumError {
    /// An integer's `multipleOf` whose step does not fit the arithmetic.
    StepTooLarge,
    /// No number satisfies the constraints.
    Empty,
}

impl NumSpec {
    pub fn new(integer: bool, lo: Option<(i128, bool)>, hi: Option<(i128, bool)>, mult: Option<i128>) -> Result<Self, NumError> {
        let tracked = lo.is_some() || hi.is_some() || mult.is_some();
        let step = match (integer, mult) {
            (true, Some(m)) => {
                let g = gcd(m, SCALE);
                Some((m / g).checked_mul(SCALE).ok_or(NumError::StepTooLarge)?)
            }
            (true, None) => Some(SCALE),
            (false, m) => m,
        };
        let spec = Self { integer, lo, hi, mult, step, tracked };
        if tracked && !spec.feasible(&NumState::start()) {
            return Err(NumError::Empty);
        }
        Ok(spec)
    }

    pub fn unconstrained(integer: bool) -> Self {
        Self { integer, lo: None, hi: None, mult: None, step: None, tracked: false }
    }

    /// Whether any magnitude in `[a, b)` (b `None` = unbounded) gives a valid
    /// value with this sign.
    fn interval_ok(&self, neg: bool, a: i128, b: Option<i128>) -> bool {
        let (mlo, mhi) = if neg {
            (self.hi.map(|(v, e)| (-v, e)), self.lo.map(|(v, e)| (-v, e)))
        } else {
            (self.lo, self.hi)
        };
        let mut low = a;
        if let Some((v, ex)) = mlo {
            low = low.max(if ex { v + 1 } else { v });
        }
        let mut high = b.map(|b| b - 1);
        if let Some((v, ex)) = mhi {
            let h = if ex { v - 1 } else { v };
            high = Some(high.map_or(h, |x| x.min(h)));
        }
        if let Some(h) = high
            && low > h
        {
            return false;
        }
        match self.step {
            None => true,
            Some(g) => {
                let first = (low + g - 1).div_euclid(g) * g;
                high.is_none_or(|h| first <= h)
            }
        }
    }

    pub fn feasible(&self, s: &NumState) -> bool {
        let mag = s.mag();
        let cap = 10i128.pow(MAX_DIGITS as u32) * SCALE;
        match s.phase {
            START => self.interval_ok(false, 0, Some(cap)) || self.interval_ok(true, 0, Some(cap)),
            MINUS => self.interval_ok(true, 0, Some(cap)),
            ZERO => self.interval_ok(s.neg, 0, Some(SCALE)),
            INT => {
                let mut lo = mag;
                let mut hi = mag + SCALE;
                for _ in s.int_digits..=MAX_DIGITS {
                    if self.interval_ok(s.neg, lo, Some(hi)) {
                        return true;
                    }
                    lo *= 10;
                    hi *= 10;
                }
                false
            }
            DOT => self.interval_ok(s.neg, mag, Some(mag + SCALE)),
            FRAC => {
                let width = 10i128.pow(SCALE_DIGITS - s.frac_digits as u32);
                self.interval_ok(s.neg, mag, Some(mag + width))
            }
            _ => true,
        }
    }

    fn valid_end(&self, s: &NumState) -> bool {
        if !matches!(s.phase, ZERO | INT | FRAC | EXP_DIG) {
            return false;
        }
        if !self.tracked {
            return true;
        }
        let m = s.mag();
        self.interval_ok(s.neg, m, Some(m + 1))
    }

    pub fn step(&self, st: NumState, b: u8) -> NumStep {
        let mut s = st;
        let exp_ok = !self.integer && !self.tracked;
        let digit = b.is_ascii_digit();
        let d = (b.wrapping_sub(b'0')) as i128;
        match st.phase {
            START | MINUS => {
                if st.phase == START && b == b'-' {
                    s.neg = true;
                    s.phase = MINUS;
                } else if b == b'0' {
                    s.phase = ZERO;
                    s.int_digits = 1;
                } else if digit {
                    s.phase = INT;
                    s.int_digits = 1;
                    if self.tracked {
                        s.set_mag(d * SCALE);
                    }
                } else {
                    return NumStep::Reject;
                }
            }
            ZERO | INT | FRAC | EXP_DIG => {
                if digit && st.phase == INT {
                    if self.tracked {
                        if st.int_digits >= MAX_DIGITS {
                            return NumStep::Reject;
                        }
                        s.set_mag(st.mag() * 10 + d * SCALE);
                    } else if st.int_digits >= MAX_FREE_INT_DIGITS {
                        return NumStep::Reject;
                    }
                    s.int_digits = st.int_digits.saturating_add(1);
                } else if digit && st.phase == FRAC {
                    if self.tracked {
                        if st.frac_digits >= MAX_DIGITS {
                            return NumStep::Reject;
                        }
                        let place = 10i128.pow(SCALE_DIGITS - st.frac_digits as u32 - 1);
                        s.set_mag(st.mag() + d * place);
                    }
                    s.frac_digits = st.frac_digits.saturating_add(1);
                } else if digit && st.phase == EXP_DIG {
                    if st.exp_digits >= MAX_EXP_DIGITS {
                        return NumStep::Reject;
                    }
                    s.exp_digits = st.exp_digits + 1;
                } else if b == b'.' && !self.integer && matches!(st.phase, ZERO | INT) {
                    s.phase = DOT;
                } else if (b == b'e' || b == b'E') && exp_ok && matches!(st.phase, ZERO | INT | FRAC) {
                    s.phase = EXP;
                } else {
                    return if self.valid_end(&st) { NumStep::End } else { NumStep::Reject };
                }
            }
            DOT => {
                if !digit {
                    return NumStep::Reject;
                }
                s.phase = FRAC;
                s.frac_digits = 1;
                if self.tracked {
                    s.set_mag(st.mag() + d * (SCALE / 10));
                }
            }
            EXP => {
                if b == b'+' || b == b'-' {
                    s.phase = EXP_SIGN;
                } else if digit {
                    s.phase = EXP_DIG;
                    s.exp_digits = 1;
                } else {
                    return NumStep::Reject;
                }
            }
            EXP_SIGN => {
                if !digit {
                    return NumStep::Reject;
                }
                s.phase = EXP_DIG;
                s.exp_digits = 1;
            }
            _ => return NumStep::Reject,
        }
        if self.tracked && !self.feasible(&s) {
            return NumStep::Reject;
        }
        NumStep::Inside(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(spec: &NumSpec, text: &str) -> bool {
        let mut s = NumState::start();
        for &b in text.as_bytes() {
            match spec.step(s, b) {
                NumStep::Inside(n) => s = n,
                _ => return false,
            }
        }
        matches!(spec.step(s, b','), NumStep::End)
    }

    fn prefix_ok(spec: &NumSpec, text: &str) -> bool {
        let mut s = NumState::start();
        for &b in text.as_bytes() {
            match spec.step(s, b) {
                NumStep::Inside(n) => s = n,
                _ => return false,
            }
        }
        true
    }

    #[test]
    fn plain_json_numbers() {
        let n = NumSpec::unconstrained(false);
        for ok in ["0", "-0", "12", "1.5", "-0.25e-3", "3E+7"] {
            assert!(run(&n, ok), "{ok}");
        }
        for bad in ["01", "-", "1.", ".5", "1e", "+1", "--1", "1e100"] {
            assert!(!run(&n, bad), "{bad}");
        }
        let i = NumSpec::unconstrained(true);
        assert!(run(&i, "-42") && !run(&i, "4.0") && !run(&i, "1e3"));
    }

    #[test]
    fn integer_bounds_are_exact_on_every_prefix() {
        let s = NumSpec::new(true, Some((SCALE, false)), Some((150 * SCALE, false)), None).unwrap();
        assert!(run(&s, "1") && run(&s, "150") && !run(&s, "151") && !run(&s, "0") && !run(&s, "-1"));
        // "2" can still become 2..=99, "16" only 16; "160" is out on its last digit.
        assert!(prefix_ok(&s, "16") && !prefix_ok(&s, "160") && !prefix_ok(&s, "-"));
    }

    #[test]
    fn decimal_bounds_and_multiple_of() {
        let half = parse_scaled("0.5").unwrap();
        let s = NumSpec::new(false, Some((0, true)), Some((5 * SCALE, false)), Some(half)).unwrap();
        assert!(run(&s, "0.5") && run(&s, "4.5") && run(&s, "5") && run(&s, "5.0"));
        assert!(!run(&s, "0") && !run(&s, "0.25") && !run(&s, "5.5"));
        assert!(!prefix_ok(&s, "1.2"), "1.2x can never be a multiple of 0.5");
        let unit = NumSpec::new(false, Some((0, false)), Some((SCALE, false)), None).unwrap();
        assert!(run(&unit, "0.731") && run(&unit, "1") && !run(&unit, "1.01") && !prefix_ok(&unit, "2"));
        assert!(!prefix_ok(&unit, "0.5e"), "a bounded number takes no exponent");
    }

    #[test]
    fn scaled_parse_refuses_what_it_cannot_hold() {
        assert_eq!(parse_scaled("0.01"), Some(SCALE / 100));
        assert_eq!(parse_scaled("-3"), Some(-3 * SCALE));
        assert_eq!(parse_scaled("0.0000000000000000001"), None);
        assert_eq!(parse_scaled("100000000000000000000"), None);
        assert_eq!(parse_scaled("1.5e2"), Some(150 * SCALE));
        assert_eq!(parse_scaled("25E-3"), Some(SCALE / 40));
        assert_eq!(parse_scaled("0e999"), Some(0));
        assert_eq!(parse_scaled("1e-19"), None);
        assert_eq!(parse_scaled("1e19"), None);
        assert_eq!(format_scaled(-3 * SCALE / 2), "-1.5");
        assert_eq!(format_scaled(7 * SCALE), "7");
    }
}
