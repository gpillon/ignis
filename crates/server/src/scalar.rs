//! `scalar` — a number that decides its own width, and may have a decimal
//! part (GitHub #255, ADR 0034, spec 10).
//!
//! [`crate::numbers`] forces **exactly** `digits` digits, so the caller has
//! to guess the magnitude and the model has to fill a field it did not
//! choose. The whole of GitHub #254 is the fallout of that field having two
//! ends. A `scalar` adds one token to the alphabet — the closing `}` — and
//! the run ends when the number is complete: `digits` becomes a **maximum**
//! and a caller with no idea of the magnitude can leave it alone.
//!
//! The same addition buys the decimal point, which is why this is a
//! primitive and not a flag: `3`, `3.5` and `-0.25` come off one schedule
//! and no grid.
//!
//! It is `number`'s sibling and not its replacement. `point` and `box` are
//! untouched, because there the width is the **scale** and not a field — a
//! coordinate on a 0-999 axis is three digits by definition, and those
//! numbers are measured.
//!
//! # Why the reader does half the work
//!
//! [`ignis_core::constrained::Schedule::step`] is a function of the index
//! alone, so a static schedule cannot say "at most one decimal point". It
//! permits what it can and this module **refuses what it must**: two
//! points, a trailing point, a bare sign, an empty run. Refused and never
//! repaired — a scalar read off a malformed run is a wrong answer wearing
//! the shape of a right one, which is the reason `run_cut_short` exists one
//! module over.
//!
//! # Three ways a run ends and only one of them answers
//!
//! The signal is the **last token**, never the length:
//!
//! | run | reading |
//! |---|---|
//! | ends with `}` | terminated — the number is what precedes it |
//! | no `}`, schedule unspent | the engine cut it off, an error |
//! | no `}`, schedule spent | more digits than were asked for, an error |
//!
//! The third row is a consequence of the schedule leaving room for every
//! structural token: a well-formed answer can **always** close itself, so a
//! run that spent the whole schedule instead wrote at least one digit past
//! its ceiling. "At the cap" is not a valid outcome here the way it is for
//! [`crate::numbers`], whose field has no terminator to miss.

use ignis_core::constrained::{Draw, Schedule};
use ignis_core::types::TokenId;
use serde::{Deserialize, Serialize};

use crate::numbers::{DigitDraw, PlanError};

/// The ceilings a scalar question may declare, inclusive at both ends.
///
/// Wider than [`crate::numbers::DIGITS`], and it can be, because the two
/// numbers are not the same kind of number. A `number`'s `digits` is a field
/// the model must **fill**, so every digit of it is a decode round spent
/// whatever the answer is, and widening it widens the run. A scalar's is a
/// ceiling the run may close early out of, so widening it costs a caller who
/// does not reach it exactly nothing — only the schedule, which is a list and
/// not a spend.
///
/// Fifteen is where `f64` stops. Every 15-significant-digit decimal
/// round-trips through an `f64` exactly and the sixteenth does not, so a
/// wider ceiling would permit a run whose spelling `value` could not carry —
/// and `text` and `value` disagreeing is the one thing this answer's two
/// fields exist to rule out.
pub const DIGITS: std::ops::Range<u32> = 1..16;

/// The ceiling when the caller declares none.
///
/// **Not the maximum.** A caller who writes nothing is not asking for the
/// widest run this endpoint can serve; they are saying they do not know the
/// magnitude, and eight digits covers the quantities that turn up in evidence
/// — a count, a duration, an amount of money — while keeping the schedule at
/// eleven steps. A caller who needs the other seven says so, and is refused
/// past fifteen.
pub const DEFAULT_DIGITS: u32 = 8;

/// The tokens a run may spend beyond its digits: the sign, the decimal
/// point, and the brace that closes the object.
///
/// The schedule is a **token** bound and not a digit bound, because
/// `Schedule::step` is a function of the index alone and cannot tell a digit
/// from a point. Which is why [`read`] counts the digits itself — the same
/// division of labour "at most one decimal point" needs.
const STRUCTURAL_TOKENS: usize = 3;

/// The literal prefilled after the prompt: the answer's opening, which
/// costs prompt tokens instead of a decode round each.
pub const PREFIX: &str = "{\"value\":";

/// The token that closes the object, and so the run.
const TERMINATOR: &str = "}";
/// The decimal point, permitted from the second step on.
const POINT: &str = ".";
/// The sign, permitted at the first step only.
const SIGN: &str = "-";

/// The system text a `scalar` question is put under.
///
/// **New text, and therefore unmeasured text** (ADR 0034) — so it was
/// measured before it shipped, and it borrows none of `number`'s figures.
///
/// It does **not** carry GitHub #254's alignment clause and cannot: an
/// instruction to pad on the left tells the model to write `003}`, which
/// fills the very field the terminator exists to avoid. The two are
/// alternatives. `number` keeps the clause because `number` keeps the
/// field.
pub fn scalar_system(digits: u32) -> String {
    format!(
        "Apply the supplied instruction to the supplied evidence and answer with a single \
         number, which may have a decimal part and may be negative. Reply with only a JSON \
         object of the form {{\"value\":N}}, using at most {digits} digits, and close the object \
         as soon as the number is complete."
    )
}

/// Everything a scalar question needs beyond its prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    /// [`PREFIX`], in tokens, appended to the rendered prompt.
    pub prefix: Vec<TokenId>,
    /// The schedule the request generates under, ending on the terminator.
    pub schedule: Schedule,
    /// The ten digit tokens, indexed by the digit they spell.
    digit_tokens: [TokenId; 10],
    point: TokenId,
    sign: TokenId,
    terminator: TokenId,
    /// The most digits the prompt asked for, which the schedule cannot
    /// enforce and [`read`] therefore does.
    max_digits: usize,
}

/// Build the plan for a scalar of at most `digits` digits.
///
/// Every token is looked up in the **loaded** tokenizer and required to be a
/// single vocabulary entry, the rule ADR 0034 states for answer labels and
/// for the same reason: a two-token `.` would put a step's draw against a
/// token that belongs to something else. On the served 27B `}` is 92, `.` is
/// 13 and `-` is 12, and no `.5`, `-3` or `0.` exists to compete with them —
/// but that is a property of *that* artifact, checked here at load, never
/// compiled in.
pub fn plan(
    digits: u32,
    encode: &dyn Fn(&str) -> Option<Vec<TokenId>>,
) -> Result<Plan, PlanError> {
    let one = |text: &str| -> Result<TokenId, PlanError> {
        let ids = encode(text).ok_or_else(|| PlanError::Unencodable(text.to_owned()))?;
        match ids.as_slice() {
            [id] => Ok(*id),
            _ => Err(PlanError::NotOneToken { text: text.to_owned(), tokens: ids.len() }),
        }
    };
    let mut digit_tokens = [0 as TokenId; 10];
    for (value, slot) in digit_tokens.iter_mut().enumerate() {
        *slot = one(&value.to_string())?;
    }
    let point = one(POINT)?;
    let sign = one(SIGN)?;
    let terminator = one(TERMINATOR)?;
    let prefix = encode(PREFIX).ok_or_else(|| PlanError::Unencodable(PREFIX.to_owned()))?;
    let max_digits = digits as usize;

    // Step 0 opens the number: a digit, or the sign before one. Neither the
    // point nor the terminator belongs there — `{"value":.5}` and
    // `{"value":}` are not numbers, and a step that permitted them would be
    // relying on the model's taste for its correctness.
    let mut opening = digit_tokens.to_vec();
    opening.push(sign);
    let mut rest = digit_tokens.to_vec();
    rest.push(point);
    rest.push(terminator);

    let mut steps = vec![opening];
    // Room for the digits *and* the structure, so a caller who asks for six
    // digits can spell `-12.3456` and still draw the brace that closes it.
    // With one token less, a full-width negative decimal could never
    // terminate and would always land at the cap.
    steps.resize(digits as usize + STRUCTURAL_TOKENS, rest);
    let schedule = Schedule::new(steps)
        .and_then(|schedule| schedule.ending_on(terminator))
        .map_err(PlanError::Schedule)?;
    Ok(Plan { prefix, schedule, digit_tokens, point, sign, terminator, max_digits })
}

/// One scalar read out of a finished run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct Reading {
    /// The number itself.
    pub value: f64,
    /// What the model actually wrote, so a caller checking a reading against
    /// its trace has the spelling that produced it.
    pub text: String,
    /// `sum((1 - p_k) * 10^place)` over the digits, in units of the value.
    ///
    /// **It cannot be accumulated as the run is walked**, which is what
    /// [`crate::numbers`] does: with a decimal point, a digit's place is not
    /// known until the point has been seen — the `5` in `3.5` is worth a
    /// tenth and the `5` in `35` is worth one. So the run is collected,
    /// parsed, and only then weighted. The point, the sign and the
    /// terminator are steps of the run and contribute nothing: two are
    /// structure and the third is the end.
    pub uncertainty: f64,
    /// The digits, in order, with the model's own confidence in each.
    pub digits: Vec<DigitDraw>,
}

/// Why a run could not be read as a scalar.
///
/// Each of these is a run the schedule allowed and a number cannot be made
/// of, which is the price of a permitted set that is a function of the step
/// index alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadError {
    /// A token no step of this plan permits.
    OffAlphabet,
    /// The engine stopped the run before it closed or reached the cap.
    CutShort,
    /// The tokens are not a number: two points, a trailing point, a sign
    /// with nothing after it, or nothing at all.
    Malformed(String),
    /// More digits than the question asked for.
    ///
    /// Its own variant rather than a [`ReadError::Malformed`] because the
    /// run *is* a number — it is the wrong number, longer than the ceiling
    /// the prompt declared, and a caller reading "not a number" would look
    /// for the fault in the wrong place.
    TooManyDigits { wrote: usize, asked: usize },
}

impl std::fmt::Display for ReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OffAlphabet => {
                write!(f, "a step committed a token outside its own permitted set")
            }
            Self::CutShort => write!(
                f,
                "the engine stopped this run before it closed its object or reached its cap, \
                 so the digits it did commit are the front of a number and not a number"
            ),
            Self::Malformed(text) => {
                write!(f, "the run spells {text:?}, which is not a number")
            }
            Self::TooManyDigits { wrote, asked } => write!(
                f,
                "the run spells {wrote} digits where the question allowed {asked}: a schedule bounds tokens and cannot tell a digit from a point, so the count is checked when the run is read"
            ),
        }
    }
}

/// Read a finished run as a scalar.
///
/// Which of the module's three outcomes this was is read off the run
/// itself: the terminator if it is there, otherwise the schedule's length
/// against the run's. Nothing else has to be passed in.
///
/// This is also where the **digit count** the prompt asked for is enforced.
/// The schedule bounds *tokens*, since a step cannot tell a digit from a
/// point, so a run may legally spell more digits than were asked for — and
/// a `digits: 1` question answering `123` would be the endpoint agreeing
/// with itself about a ceiling neither half meant.
pub fn read(plan: &Plan, drawn: &[Draw]) -> Result<Reading, ReadError> {
    let mut text = String::with_capacity(drawn.len());
    let mut digits = Vec::with_capacity(drawn.len());
    let mut terminated = false;
    for draw in drawn {
        if draw.token == plan.terminator {
            terminated = true;
            break;
        }
        if draw.token == plan.point {
            text.push('.');
            continue;
        }
        if draw.token == plan.sign {
            text.push('-');
            continue;
        }
        let digit = plan
            .digit_tokens
            .iter()
            .position(|&id| id == draw.token)
            .ok_or(ReadError::OffAlphabet)? as u32;
        text.push(char::from_digit(digit, 10).expect("a digit under ten"));
        digits.push(DigitDraw { digit, probability: f64::from(draw.probability) });
    }
    // Row three of the table: no terminator and the schedule not spent.
    if !terminated && drawn.len() < plan.schedule.len() {
        return Err(ReadError::CutShort);
    }
    if digits.len() > plan.max_digits {
        return Err(ReadError::TooManyDigits { wrote: digits.len(), asked: plan.max_digits });
    }
    // `f64::from_str` is not the well-formedness check it looks like: it
    // accepts `3.` and would report a trailing point as the number 3,
    // turning a run the schedule should never have produced into a
    // plausible answer. So the shape is **enumerated** first — sign, digits,
    // optionally a point and more digits — and the parse only runs on
    // something already known to be a number.
    if !well_formed(&text) {
        return Err(ReadError::Malformed(text));
    }
    let value: f64 = text.parse().map_err(|_| ReadError::Malformed(text.clone()))?;
    if !value.is_finite() {
        return Err(ReadError::Malformed(text.clone()));
    }
    Ok(Reading { value, text: text.clone(), uncertainty: uncertainty(&text, &digits), digits })
}

/// Whether `text` is a number this endpoint will report.
///
/// The accepted shape is written out rather than the rejected one: an
/// optional `-`, at least one digit, and at most one point with at least one
/// digit after it. Listing what is allowed is what stops a spelling nobody
/// anticipated from walking past — the same rule the answer alphabet
/// follows, and the reason `3.` is refused here although `f64` would take
/// it.
fn well_formed(text: &str) -> bool {
    let unsigned = text.strip_prefix('-').unwrap_or(text);
    let mut parts = unsigned.split('.');
    let (Some(whole), fraction, None) = (parts.next(), parts.next(), parts.next()) else {
        // Two points or more.
        return false;
    };
    let plain = |part: &str| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit());
    plain(whole) && fraction.is_none_or(plain)
}

/// The place-weighted doubt over the digits of `text`.
///
/// `text` is already known to parse, so the integer part is everything
/// before the point and each digit's place follows from where it sits
/// relative to it. Leading zeros are not skipped the way
/// [`crate::numbers`] skips them: this run has no padding to skip, because
/// nothing forced the model to fill a field.
fn uncertainty(text: &str, digits: &[DigitDraw]) -> f64 {
    let unsigned = text.strip_prefix('-').unwrap_or(text);
    let integer_len = unsigned.find('.').unwrap_or(unsigned.len());
    digits
        .iter()
        .enumerate()
        .map(|(index, draw)| {
            // Place `integer_len - 1 - index`: the last integer digit is
            // worth 1, the first digit after the point a tenth.
            let place = integer_len as i32 - 1 - index as i32;
            (1.0 - draw.probability) * 10f64.powi(place)
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One token per character, ids offset so a digit is never its own
    /// value — a plan that confused the two would pass without this.
    fn per_character(text: &str) -> Option<Vec<TokenId>> {
        Some(text.chars().map(|c| c as TokenId + 1000).collect())
    }

    fn token(c: char) -> TokenId {
        c as TokenId + 1000
    }

    fn draw(c: char, probability: f32) -> Draw {
        Draw { token: token(c), probability }
    }

    #[test]
    fn the_opening_step_is_digits_and_a_sign_and_the_rest_add_a_point_and_an_end() {
        let plan = plan(3, &per_character).expect("a plan");
        assert_eq!(plan.prefix, per_character(PREFIX).unwrap(), "the opening is prompt");
        let opening = plan.schedule.step(0).expect("a first step");
        assert!(opening.contains(&token('-')), "a scalar may be negative");
        assert!(!opening.contains(&token('.')), "`.5` is not a number this endpoint writes");
        assert!(!opening.contains(&token('}')), "`{{\"value\":}}` is not a number at all");
        let rest = plan.schedule.step(1).expect("a second step");
        assert!(rest.contains(&token('.')));
        assert!(rest.contains(&token('}')));
        assert!(!rest.contains(&token('-')), "the sign opens a number, it does not continue one");
        assert_eq!(plan.schedule.terminator(), Some(token('}')));
    }

    #[test]
    fn a_terminated_run_is_the_number_before_the_brace() {
        let plan = plan(6, &per_character).expect("a plan");
        let reading = read(&plan, &[draw('3', 0.99), draw('}', 1.0)]).expect("a number");
        assert_eq!(reading.value, 3.0);
        assert_eq!(reading.text, "3");
        assert_eq!(reading.digits.len(), 1, "the brace is not a digit");
    }

    #[test]
    fn a_decimal_digit_is_worth_a_tenth_and_not_a_unit() {
        let plan = plan(6, &per_character).expect("a plan");
        let reading = read(&plan, &[draw('3', 1.0), draw('.', 1.0), draw('5', 0.5), draw('}', 1.0)])
            .expect("a number");
        assert_eq!(reading.value, 3.5);
        assert_eq!(reading.text, "3.5");
        // Half a unit of doubt in the tenths place is 0.05, which is the
        // whole reason this is computed after parsing and not during.
        assert!((reading.uncertainty - 0.05).abs() < 1e-6, "{}", reading.uncertainty);

        let same_digits_no_point =
            read(&plan, &[draw('3', 1.0), draw('5', 0.5), draw('}', 1.0)]).expect("a number");
        assert_eq!(same_digits_no_point.value, 35.0);
        assert!(
            (same_digits_no_point.uncertainty - 0.5).abs() < 1e-6,
            "the same draw is worth ten times as much one place left: {}",
            same_digits_no_point.uncertainty
        );
    }

    #[test]
    fn a_negative_scalar_reads_as_one() {
        let plan = plan(6, &per_character).expect("a plan");
        let reading = read(
            &plan,
            &[
                draw('-', 0.9),
                draw('0', 1.0),
                draw('.', 1.0),
                draw('2', 0.8),
                draw('5', 1.0),
                draw('}', 1.0),
            ],
        )
        .expect("a number");
        assert_eq!(reading.value, -0.25);
        assert_eq!(reading.text, "-0.25");
        // The sign is not a digit and carries no place, so only the `2`'s
        // doubt counts, at a tenth.
        // A probability is an `f32`, so 0.8 is 0.800000011920929 and a
        // tenth of its complement is 0.019999998.
        assert!((reading.uncertainty - 0.02).abs() < 1e-6, "{}", reading.uncertainty);
    }

    /// A well-formed answer can always close itself, because the schedule
    /// leaves room for the sign, the point *and* the brace. So a run that
    /// closed early with its schedule unspent is the ordinary case and not
    /// a short one.
    #[test]
    fn a_closed_run_with_steps_to_spare_is_the_ordinary_case() {
        let plan = plan(2, &per_character).expect("a plan");
        let drawn: Vec<Draw> = "-1.5}".chars().map(|c| draw(c, 1.0)).collect();
        assert!(drawn.len() <= plan.schedule.len(), "the widest well-formed run still fits");
        let reading = read(&plan, &drawn).expect("a closed run");
        assert_eq!(reading.value, -1.5);
        assert_eq!(reading.digits.len(), 2, "the sign, the point and the brace are not digits");
    }

    /// The other side of that: spending the whole schedule without closing
    /// means at least one digit past the ceiling, so the third row of the
    /// module's table is always an error and never an answer.
    #[test]
    fn spending_the_whole_schedule_without_closing_is_always_too_many_digits() {
        let plan = plan(2, &per_character).expect("a plan");
        let drawn: Vec<Draw> = "-1.55".chars().map(|c| draw(c, 1.0)).collect();
        assert_eq!(drawn.len(), plan.schedule.len(), "it spent every step");
        match read(&plan, &drawn) {
            Err(ReadError::TooManyDigits { wrote: 3, asked: 2 }) => {}
            other => panic!("a two-digit question wrote three: {other:?}"),
        }
    }

    /// The schedule bounds tokens and cannot tell a digit from a point, so
    /// a run may spell more digits than the prompt asked for. The reader is
    /// what makes `digits` mean anything.
    #[test]
    fn more_digits_than_the_question_allowed_is_refused() {
        let plan = plan(1, &per_character).expect("a plan");
        let drawn: Vec<Draw> = "123}".chars().map(|c| draw(c, 1.0)).collect();
        match read(&plan, &drawn) {
            Err(ReadError::TooManyDigits { wrote: 3, asked: 1 }) => {}
            other => panic!("a one-digit question answered 123: {other:?}"),
        }
    }

    #[test]
    fn a_run_the_engine_cut_short_is_still_an_error() {
        let plan = plan(6, &per_character).expect("a plan");
        assert_eq!(read(&plan, &[draw('1', 1.0)]), Err(ReadError::CutShort));
    }

    /// The permitted sets are a function of the step index, so these are
    /// runs the schedule allows and a number cannot be made of.
    #[test]
    fn a_run_that_is_not_a_number_is_refused_and_never_repaired() {
        let plan = plan(6, &per_character).expect("a plan");
        for spelling in ["1..2}", "3.}", "-}", "}"] {
            let drawn: Vec<Draw> = spelling.chars().map(|c| draw(c, 1.0)).collect();
            match read(&plan, &drawn) {
                Err(ReadError::Malformed(_)) => {}
                other => panic!("{spelling} should not read as a number: {other:?}"),
            }
        }
    }

    #[test]
    fn a_token_from_no_step_is_named_as_such() {
        let plan = plan(6, &per_character).expect("a plan");
        assert_eq!(read(&plan, &[draw('x', 1.0), draw('}', 1.0)]), Err(ReadError::OffAlphabet));
    }

    /// The widest ceiling served, spelled out. Fifteen digits is where the
    /// range stops because it is where an `f64` stops carrying a decimal
    /// exactly, so the spelling and the value have to still agree at the
    /// edge — which is the whole reason the answer carries both.
    #[test]
    fn the_widest_ceiling_reads_every_digit_it_allows() {
        let widest = DIGITS.end - 1;
        let plan = plan(widest, &per_character).expect("a plan");
        let drawn: Vec<Draw> = "123456789012345}".chars().map(|c| draw(c, 1.0)).collect();
        let reading = read(&plan, &drawn).expect("a number");
        assert_eq!(reading.text, "123456789012345");
        assert_eq!(reading.value, 123_456_789_012_345.0);
        assert_eq!(reading.digits.len(), widest as usize);
        assert_eq!(reading.value.to_string(), reading.text, "the value spells itself back");
    }

    /// The default is what a caller who says nothing gets, and it is
    /// deliberately not the maximum: saying nothing is "I do not know the
    /// magnitude", not "give me every digit you serve".
    #[test]
    fn the_default_ceiling_leaves_room_above_it() {
        assert!(DIGITS.contains(&DEFAULT_DIGITS));
        assert!(DEFAULT_DIGITS > DIGITS.start, "a one-digit default would refuse most evidence");
        assert!(DEFAULT_DIGITS < DIGITS.end - 1, "a caller who needs more must have somewhere to go");
    }

    #[test]
    fn the_prompt_declares_the_maximum_and_asks_the_model_to_close() {
        let text = scalar_system(4);
        assert!(text.contains("at most 4 digits"), "{text}");
        assert!(text.contains("close the object as soon as the number is complete"), "{text}");
        assert!(text.contains("decimal part"), "{text}");
        assert!(
            !text.contains("padded on the left"),
            "an instruction to pad fights the terminator: {text}"
        );
    }
}
