//! `number`, `point` and `box` — the decision primitives that **generate**
//! (GitHub #242, ADR 0034, spec 06).
//!
//! Jev's three primitives read one position, so they yield one symbol. The
//! one they cannot express is a **number read digit by digit**: K positions,
//! each restricted to the digit alphabet, read in order. A point is two of
//! them and a box is four — sugar over the same primitive, not new
//! machinery.
//!
//! Two of the three prompts here are unmeasured, and each says so where it
//! is defined. What *was* measured is `point` at three digits over a
//! screenshot (`docs/findings/2026-09-19-constrained-digit-readout-points.md`,
//! `crates/server/tests/classify_pointing_gpu.rs`): inside the target button
//! on all three scenes, with the constraint reading the model rather than
//! overruling it. That text is reproduced here byte for byte at three
//! digits, and `the_measured_prompt_is_the_prompt_that_ships` is what keeps
//! it that way.
//!
//! **Where the forced text goes.** The opening literal — `{"x":` — is part
//! of the *prompt*: appended to the rendered tokens, prefilled with them,
//! costing no decode round at all. Every literal after it is a **step of
//! one** in the schedule, which is the same forcing at the price of a round
//! per token, because by then the model is generating. The digits in
//! between are steps of ten.

use std::collections::BTreeMap;
use std::ops::Range;

use ignis_core::constrained::{Draw, Schedule};
use ignis_core::types::TokenId;
use serde::{Deserialize, Serialize};

/// Digits per axis a caller may ask for.
///
/// One is a coin flip between ten answers and six is a part in a million,
/// already far past the model's own spatial resolution — the measured trace
/// puts that at about one part in a hundred, the units digit coming back at
/// p≈0.15 (`docs/findings/2026-09-19-constrained-digit-readout-points.md`).
/// The ceiling is not physics; it is the point past which the extra rounds
/// buy nothing but a longer-looking number.
///
/// **The width is a field, and any field that holds the value works**
/// (GitHub #254). A narrower one truncates, which is the only thing a
/// narrower field could do; there is nothing else to steer around.
///
/// This used to say something else, and the something else was wrong. It
/// read that the model left-aligns, so a width of exactly one more than the
/// value's own digits multiplies the answer by ten while two or more pads on
/// the left and is safe again. Measured across four truths at all six
/// widths, that rule fails on every one of them: 47 is right at four digits
/// and comes back 4700 at five, 3 comes back 300 at three. What the model
/// was doing was **choosing** an alignment, at the first digit, because
/// [`number_system`] declared a width and said nothing about which end to
/// pad. The schedule forces exactly K digits and carries no way to say the
/// number is finished, so a value narrower than its field can only be
/// rendered one way or the other, and nothing named which.
///
/// [`number_system`] names it now, and the ambiguity goes with it: 11 of 21
/// widths correct before that sentence, 21 of 21 after, with the padding
/// zeros going from contested (p = 0.48-0.91) to certain (0.92-1.00). The
/// old tell — a first digit falling to 0.65-0.71 — was a symptom of the
/// alignment being contested and not of the width being wrong, which is why
/// it never fired on the `300` that started this
/// (`docs/findings/2026-09-21-the-number-prompt-declares-an-alignment.md`).
///
/// A declared `min`/`max` would make the width a consequence of the range
/// rather than a caller's guess; spec 06 leaves that open until somebody
/// measures whether the model obeys a range other than 0-999.
pub const DIGITS: Range<u32> = 1..7;

/// Digits per axis when the caller names none: the measured width.
pub const DEFAULT_DIGITS: u32 = 3;

/// The system text a **point** question is put under, at `digits` digits per
/// axis.
///
/// At three digits this is byte-identical to the text every number in the
/// finding was measured with, which is the only reason this endpoint claims
/// those numbers. A reworded instruction is an unmeasured one, so the
/// wording is not a place to be clever: the parameterisation exists to serve
/// a caller who asked for four digits, and the test proves it changes
/// nothing at three.
pub fn point_system(digits: u32) -> String {
    let max = scale(digits);
    let n = "N".repeat(digits as usize);
    format!(
        "You are given a screenshot and an instruction. Answer with the position on the screen \
         the instruction refers to. Use a 0-{max} scale on each axis, where x=0 is the left edge, \
         x={max} the right edge, y=0 the top edge and y={max} the bottom edge. Reply with only a \
         JSON object of the form {{\"x\":{n},\"y\":{n}}}, {} digits each.",
        spelled(digits)
    )
}

/// The system text a **box** question is put under.
///
/// `point`'s text extended to four numbers by the obvious reading of it.
/// Lightly measured, and by less than `point`:
/// `docs/findings/2026-09-20-number-width-and-decide-e2e.md` ran it through
/// the served endpoint over the three committed pointing scenes and found
/// its edges within 6 px on x and 30-80 px on y of the ground truth — the
/// same y-bias `point` shows, on the same three synthetic scenes with one
/// obvious target. That is a working shape, not an accuracy figure: nothing
/// has scored `box` on a real screenshot corpus, and a caller reading across
/// from `point`'s numbers is reading across from numbers that are not about
/// `box`.
pub fn box_system(digits: u32) -> String {
    let max = scale(digits);
    let n = "N".repeat(digits as usize);
    format!(
        "You are given a screenshot and an instruction. Answer with the bounding box on the \
         screen the instruction refers to. Use a 0-{max} scale on each axis, where x=0 is the \
         left edge, x={max} the right edge, y=0 the top edge and y={max} the bottom edge. Reply \
         with only a JSON object of the form {{\"x0\":{n},\"y0\":{n},\"x1\":{n},\"y1\":{n}}}, \
         where (x0,y0) is the top-left corner and (x1,y1) the bottom-right, {} digits each.",
        spelled(digits)
    )
}

/// The system text a **number** question is put under.
///
/// **The alignment clause is load-bearing, not tidiness** (GitHub #254).
/// Without it the model picks an end to pad at the first digit and is wrong
/// on half the widths that could hold the answer; with it, every width that
/// fits is right. [`DIGITS`] has what that replaced.
///
/// **The weakest of the three, and for a sharper reason than `box`:**
/// `point`'s accuracy rests on 0-999 being the scale this model already
/// thinks in — the pointing finding establishes that from a free probe with
/// no scale declared anywhere. A bare `number` declares a range that is the
/// model's own only by coincidence, and whether it *obeys* a declared range
/// is an open follow-up in that same finding.
///
/// Its arithmetic is also not to be leaned on — asked for a total it has to
/// compute, it answered 172 for 230.50 — though it said so, with a first
/// digit at p=0.63 and a sigma of 46.7. The mechanism is
/// exposed plainly because it is the primitive the other two are made of;
/// `point`'s numbers are `point`'s and do not transfer.
pub fn number_system(digits: u32) -> String {
    let max = scale(digits);
    format!(
        "Apply the supplied instruction to the supplied evidence and answer with a single whole \
         number from 0 to {max}. Reply with only a JSON object of the form {{\"value\":{}}}, {} \
         digits, right-aligned and padded on the left with zeros.",
        "N".repeat(digits as usize),
        spelled(digits)
    )
}

/// The largest value `digits` digits can carry: the scale the prompt
/// declares, and the divisor a pixel reading is scaled by.
pub fn scale(digits: u32) -> u64 {
    10u64.pow(digits) - 1
}

fn spelled(digits: u32) -> &'static str {
    match digits {
        1 => "one",
        2 => "two",
        3 => "three",
        4 => "four",
        5 => "five",
        _ => "six",
    }
}

/// One axis of a constrained question: what the answer calls it, and which of
/// the run's draws are its digits.
///
/// The range is needed because the run is not all digits: the literal
/// separating two axes (`,"y":`) is one step per token, and those steps draw
/// tokens too. Recording the ranges at build time is what keeps the answer
/// shaper from re-deriving a layout it could get subtly wrong.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Axis {
    pub name: &'static str,
    pub digits: Range<usize>,
}

/// Everything a constrained question needs beyond its prompt: the literal
/// prefilled after it, the schedule, and where each axis sits in the run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    /// The opening literal, in tokens, appended to the rendered prompt.
    pub prefix: Vec<TokenId>,
    /// The schedule the request generates under.
    pub schedule: Schedule,
    /// The axes, in prompt order.
    pub axes: Vec<Axis>,
    /// The ten digit tokens, indexed by the digit they spell.
    pub digit_tokens: [TokenId; 10],
}

/// Why a run's plan could not be built.
///
/// Every one of these is a property of the *load*, not of the request, so
/// the endpoint reports it as a refusal that names what the tokenizer could
/// not do rather than blaming the caller for asking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanError {
    /// A digit, or a literal, does not encode at all.
    Unencodable(String),
    /// A digit is more than one token in this tokenizer, so it cannot be an
    /// alphabet member: a permitted set names single vocabulary entries.
    NotOneToken { text: String, tokens: usize },
    /// The schedule itself was refused (a step past the leaf's cap).
    Schedule(String),
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unencodable(text) => write!(f, "this model's tokenizer cannot encode {text:?}"),
            Self::NotOneToken { text, tokens } => write!(
                f,
                "this model's tokenizer spells {text:?} as {tokens} tokens, and a forced \
                 alphabet names single vocabulary entries"
            ),
            Self::Schedule(message) => write!(f, "{message}"),
        }
    }
}

/// The axis names and their literals, in prompt order.
///
/// The first element's literal is the **prefix**, which goes in the prompt;
/// every other one is forced mid-generation, a step of one per token.
pub type Layout = &'static [(&'static str, &'static str)];

pub const NUMBER_LAYOUT: Layout = &[("value", "{\"value\":")];
pub const POINT_LAYOUT: Layout = &[("x", "{\"x\":"), ("y", ",\"y\":")];
pub const BOX_LAYOUT: Layout = &[
    ("x0", "{\"x0\":"),
    ("y0", ",\"y0\":"),
    ("x1", ",\"x1\":"),
    ("y1", ",\"y1\":"),
];

/// Build a plan; `encode` is the loaded tokenizer, spelling literals the way
/// the model reads them.
///
/// The literals are encoded **standalone**, exactly as the measured test
/// encoded them, and never re-rendered through the chat template: the
/// template would put role markers around them, and the whole point of a
/// forced prefix is that the model sees text it appears to have written
/// itself.
pub fn plan(
    layout: Layout,
    digits: u32,
    encode: &dyn Fn(&str) -> Option<Vec<TokenId>>,
) -> Result<Plan, PlanError> {
    let mut digit_tokens = [0 as TokenId; 10];
    let mut alphabet = Vec::with_capacity(10);
    for digit in 0..10u32 {
        let text = digit.to_string();
        let ids = encode(&text).ok_or_else(|| PlanError::Unencodable(text.clone()))?;
        let [id] = ids[..] else {
            return Err(PlanError::NotOneToken { text, tokens: ids.len() });
        };
        digit_tokens[digit as usize] = id;
        alphabet.push(id);
    }
    let mut prefix = Vec::new();
    let mut steps: Vec<Vec<TokenId>> = Vec::new();
    let mut axes = Vec::with_capacity(layout.len());
    for (index, (name, literal)) in layout.iter().enumerate() {
        let ids = encode(literal).ok_or_else(|| PlanError::Unencodable((*literal).to_owned()))?;
        match index {
            // The opening literal is prompt, not schedule: it costs a few
            // prefilled tokens instead of one decode round each.
            0 => prefix = ids,
            // Every later one is forced mid-generation, and a forced literal
            // is a permitted set of one — drawn at probability 1, so it adds
            // nothing to the answer's uncertainty.
            _ => steps.extend(ids.into_iter().map(|id| vec![id])),
        }
        let begin = steps.len();
        steps.extend(std::iter::repeat_n(alphabet.clone(), digits as usize));
        axes.push(Axis { name, digits: begin..steps.len() });
    }
    let schedule = Schedule::new(steps).map_err(PlanError::Schedule)?;
    Ok(Plan { prefix, schedule, axes, digit_tokens })
}

/// One digit of an answer, and the model's own confidence in it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct DigitDraw {
    pub digit: u32,
    pub probability: f64,
}

/// One axis read out of a run: its value on the declared scale, the
/// uncertainty in the same units, and the digits behind both.
#[derive(Debug, Clone, PartialEq)]
pub struct Reading {
    pub value: u64,
    /// `sigma = sum((1 - p_k) * 10^place)` — in units of the **value**, not
    /// a 0-1 score.
    ///
    /// A score would be the wrong shape twice over. It would throw away
    /// *which* digit is uncertain, when that is the whole signal: the
    /// measured trace falls monotonically across a number — hundreds at
    /// 0.97-0.99, units at 0.15-0.57 — so the model is reporting its
    /// resolution, not its mood. And it would be uncomparable with the
    /// answer, when what a caller wants is "3145 ± 87 pixels".
    ///
    /// **It is the model's self-declared uncertainty, and it is not a
    /// guarantee.** On the measured sample it covers the true error on
    /// **four of six axes**: the two it misses are `large`'s y (sigma 7.8,
    /// error 20.4) and `small`'s y (sigma 4.6, error 7.7), both on the axis
    /// whose first digit the model was least sure of
    /// (`docs/findings/2026-09-19-constrained-digit-readout-points.md`
    /// carries the derivation). A caller treating it as a bound will be
    /// wrong about one reading in three; a caller treating it as the
    /// model's own statement of resolution will not.
    pub sigma: f64,
    pub digits: Vec<DigitDraw>,
}

/// Read every axis out of a finished run.
///
/// `drawn` must be the whole trace, in order, exactly as long as the plan's
/// schedule: a short run is one the engine cut off, and a place-weighted sum
/// over one would be a plausible-looking wrong number rather than a missing
/// one. The caller checks the length; this indexes.
pub fn read(plan: &Plan, drawn: &[Draw]) -> Option<BTreeMap<String, Reading>> {
    let mut readings = BTreeMap::new();
    for axis in &plan.axes {
        let mut value = 0u64;
        let mut sigma = 0f64;
        let width = axis.digits.len();
        let mut digits = Vec::with_capacity(width);
        for (place, index) in axis.digits.clone().enumerate() {
            let draw = drawn.get(index)?;
            let digit = plan.digit_tokens.iter().position(|&id| id == draw.token)? as u32;
            // The leftmost digit is the most significant: step `0` of the
            // loop is worth `10^(width - 1)`.
            let weight = 10f64.powi((width - 1 - place) as i32);
            value = value * 10 + u64::from(digit);
            sigma += (1.0 - f64::from(draw.probability)) * weight;
            digits.push(DigitDraw { digit, probability: f64::from(draw.probability) });
        }
        readings.insert(axis.name.to_owned(), Reading { value, sigma, digits });
    }
    Some(readings)
}

impl Reading {
    /// Rescale this reading onto an image `pixels` pixels wide (or tall) on
    /// its own axis, returning the value and its uncertainty in pixels.
    ///
    /// A method rather than a free function taking the value and the sigma
    /// apart: they are always this reading's two fields, and separating
    /// them is how an x sigma ends up rescaled by a y side.
    pub fn to_pixels(&self, digits: u32, pixels: u32) -> (i64, f64) {
        to_pixels(self.value, self.sigma, digits, pixels)
    }
}

/// Rescale a reading from the declared 0-`scale` axis onto an image `pixels`
/// pixels wide (or tall) on that axis.
///
/// The server does this rather than handing a caller a normalized pair,
/// because per-axis normalization on a non-square image is the mistake
/// everyone makes once: the model answers 0-999 on *both* axes, so a
/// 1600x900 screenshot needs two different divisors, and a caller who used
/// one has an answer that is right along x and wrong along y.
///
/// The top of the range maps to `pixels`, which is one past the last
/// addressable column — the same arithmetic `classify_pointing_gpu.rs`
/// measured with, kept so the served reading and the finding's are the same
/// number. A caller clicking at exactly `x == width` is off the image by
/// one; nothing in the measured data ever read a top-of-range digit, so this
/// is a stated edge rather than an observed one.
pub fn to_pixels(value: u64, sigma: f64, digits: u32, pixels: u32) -> (i64, f64) {
    let per_unit = f64::from(pixels) / scale(digits) as f64;
    ((value as f64 * per_unit).round() as i64, sigma * per_unit)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The text every number in the finding was measured with, copied from
    /// `crates/server/tests/classify_pointing_gpu.rs` rather than retyped.
    const MEASURED: &str = "You are given a screenshot and an instruction. Answer with the position on the screen the instruction refers to. Use a 0-999 scale on each axis, where x=0 is the left edge, x=999 the right edge, y=0 the top edge and y=999 the bottom edge. Reply with only a JSON object of the form {\"x\":NNN,\"y\":NNN}, three digits each.";

    #[test]
    fn the_measured_prompt_is_the_prompt_that_ships() {
        assert_eq!(
            point_system(DEFAULT_DIGITS),
            MEASURED,
            "the parameterisation exists for a caller who asked for a different \
             width; at three digits it must change nothing, or this endpoint's \
             accuracy figures are about a prompt it does not send"
        );
    }

    /// The alignment clause is the fix for GitHub #254 and it is the whole
    /// fix: there is no code path that pads, only a prompt that says which
    /// end to pad. So it is pinned the way `point_system`'s text is pinned,
    /// at every width, because a reword that dropped it would put the
    /// endpoint back to answering 300 for a truth of 3 with nothing failing.
    #[test]
    fn every_width_tells_the_model_which_end_to_pad() {
        for digits in DIGITS {
            let text = number_system(digits);
            assert!(
                text.contains("right-aligned and padded on the left with zeros"),
                "{digits} digits: {text}"
            );
            assert!(text.contains(&format!("0 to {}", scale(digits))), "{text}");
        }
    }

    /// A stand-in tokenizer: one token per character, ids offset so a digit
    /// is never its own value — a plan that confused the two would pass.
    fn per_character(text: &str) -> Option<Vec<TokenId>> {
        Some(text.chars().map(|c| c as TokenId + 1000).collect())
    }

    #[test]
    fn a_point_plan_is_digits_a_forced_literal_and_digits() {
        let plan = plan(POINT_LAYOUT, 3, &per_character).expect("a plan");
        let separator = per_character(",\"y\":").expect("the literal").len();
        assert_eq!(
            plan.prefix,
            per_character("{\"x\":").unwrap(),
            "the opening literal is prompt, not schedule: it costs no round"
        );
        assert_eq!(plan.schedule.len(), 3 + separator + 3);
        assert_eq!(plan.axes[0].digits, 0..3);
        assert_eq!(plan.axes[1].digits, 3 + separator..6 + separator);
        for step in &plan.schedule.steps()[plan.axes[0].digits.clone()] {
            assert_eq!(step.len(), 10, "a digit step permits the ten digits");
        }
        for step in &plan.schedule.steps()[3..3 + separator] {
            assert_eq!(step.len(), 1, "a forced literal is a set of one");
        }
    }

    #[test]
    fn a_reading_is_place_weighted_and_so_is_its_uncertainty() {
        let plan = plan(NUMBER_LAYOUT, 3, &per_character).expect("a plan");
        let draw = |digit: usize, probability: f32| Draw {
            token: plan.digit_tokens[digit],
            probability,
        };
        // 7, 6, 7 — the measured `large` scene's x axis.
        let drawn = [draw(7, 0.993), draw(6, 0.981), draw(7, 0.574)];
        let readings = read(&plan, &drawn).expect("every draw is a digit");
        let reading = &readings["value"];
        assert_eq!(reading.value, 767);
        let expected = (1.0 - 0.993f64) * 100.0 + (1.0 - 0.981f64) * 10.0 + (1.0 - 0.574f64);
        assert!(
            (reading.sigma - expected).abs() < 1e-3,
            "sigma is the place-weighted sum, so an uncertain hundreds digit is \
             worth a hundred times an uncertain units one: {} vs {expected}",
            reading.sigma
        );
        assert_eq!(reading.digits.len(), 3);
        assert_eq!(reading.digits[0].digit, 7);
    }

    #[test]
    fn a_forced_literal_adds_nothing_to_the_uncertainty() {
        // Two axes, with the separator's draws between them: an axis range
        // that swallowed those would read a literal's token as a digit and
        // fail, which is what the ranges are for.
        let plan = plan(POINT_LAYOUT, 2, &per_character).expect("a plan");
        let digit = |d: usize, p: f32| Draw { token: plan.digit_tokens[d], probability: p };
        let mut drawn = vec![digit(3, 1.0), digit(1, 1.0)];
        for step in &plan.schedule.steps()[plan.axes[0].digits.end..plan.axes[1].digits.start] {
            drawn.push(Draw { token: step[0], probability: 1.0 });
        }
        drawn.extend([digit(2, 1.0), digit(4, 0.5)]);
        let readings = read(&plan, &drawn).expect("the literal's draws are skipped");
        assert_eq!(readings["x"].value, 31);
        assert_eq!(readings["y"].value, 24);
        assert_eq!(readings["x"].sigma, 0.0);
        assert!((readings["y"].sigma - 0.5).abs() < 1e-9, "only the units digit was unsure");
    }

    #[test]
    fn pixels_come_from_the_axis_the_reading_is_on() {
        // A non-square image: the same normalized value on two axes is two
        // different pixel positions, which is the mistake this exists to
        // prevent.
        let (x, _) = to_pixels(500, 0.0, 3, 1600);
        let (y, _) = to_pixels(500, 0.0, 3, 900);
        assert_eq!((x, y), (801, 450), "500 of 999 is 800.8 px across and 450.5 px down");
        let (_, sigma) = to_pixels(0, 10.0, 3, 999);
        assert!((sigma - 10.0).abs() < 1e-9, "a 0-999 axis on a 999px image is 1:1");
    }

    #[test]
    fn a_digit_that_is_not_one_token_is_refused_rather_than_approximated() {
        let two_tokens = |text: &str| Some(vec![1 as TokenId; text.len() + 1]);
        let error = plan(NUMBER_LAYOUT, 3, &two_tokens).expect_err("digits are not single tokens");
        assert!(matches!(error, PlanError::NotOneToken { .. }), "{error}");
    }
}
