//! YaRN RoPE scaling as a load option (GitHub #227).
//!
//! Like speculation and vision, the rotary table is engine residency: chosen
//! at load, frozen for the life of that load, because a sequence's cached
//! keys are rotated with it and a table that changed underneath them would
//! make every retained page mean something else.
//!
//! The checkpoint is trained over 262,144 positions. Past that the linear
//! table's lowest-frequency pairs are extrapolating into angles the model
//! never saw, and quality falls off a cliff — while the engine itself will
//! happily serve the context (the GQA attention op's envelope is 1,048,576
//! visible keys under `hq-e8-2b`). YaRN is the table that closes that gap:
//! the high-frequency pairs keep extrapolating, the low-frequency ones are
//! interpolated by the factor, and a ramp blends the band between.
//!
//! [`RopeScaling::default`] is no scaling — the linear table the engine has
//! always used, byte for byte. That is not a stylistic default: the vendored
//! `ops::rope` gives an `attention_factor` of exactly 1 its legacy FP32 angle
//! route and anything else an FP64-reduced one, so "YaRN at factor 1" would
//! quietly move the outputs of a server nobody asked to scale.

/// The trained position envelope: the rotary positions the checkpoints are
/// trained over. Past it a context needs a scaled table, and `auto`
/// ([`RopeSetting::Auto`]) scales by the ratio -- for both models, against
/// this one envelope. The leaf names it per model, the same 262,144: the
/// 27B's `kTextOriginalPositions` (`kernel/src/rope_scaling.h`) and
/// Flash-Next's `flash_next::kOriginalPositions`
/// (`kernel/src/flash_next/flash_next_internal.h`); a checkpoint trained over
/// another envelope would need its own here too.
pub const TRAINED_POSITION_ENVELOPE: u32 = 262_144;

/// The widest factor a load accepts (`kMaxYarnFactor`,
/// `kernel/src/rope_scaling.h`). 64x the trained envelope is already past
/// every attention envelope the engine can serve, so this is a typo guard,
/// not a capability claim.
pub const MAX_YARN_FACTOR: f32 = 64.0;

/// The attention-temperature coefficient YaRN's own paper uses, and HF's
/// default: the attention factor is `temperature * ln(factor) + 1`.
pub const DEFAULT_YARN_TEMPERATURE: f32 = 0.1;

/// The ramp's fast bound: the pair below which frequencies extrapolate
/// unchanged.
pub const DEFAULT_YARN_BETA_FAST: f32 = 32.0;

/// The ramp's slow bound: the pair past which frequencies are fully
/// interpolated.
pub const DEFAULT_YARN_BETA_SLOW: f32 = 1.0;

/// The rotary table a load runs on.
#[derive(Debug, Clone, Copy, Default)]
pub struct RopeScaling {
    /// 0 (the default) or 1 is no scaling; a value in `(1, MAX_YARN_FACTOR]`
    /// selects YaRN at that factor.
    factor: f32,
    temperature: f32,
    beta_fast: f32,
    beta_slow: f32,
}

/// What `--model-rope-scaling` says: a table spelled out, or `auto`, which
/// is decided per load from the context it serves ([`Self::resolve`]).
/// `auto` is the default, so the 27B asked for 1,048,576 tokens of context
/// runs YaRN at 4 without being told, and at the trained 262,144 the linear
/// table it always has. (Both models run the table the server's config
/// resolves, ADR 0057, ADR 0059.)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RopeSetting {
    #[default]
    Auto,
    Fixed(RopeScaling),
}

impl RopeSetting {
    /// `auto`, or what [`RopeScaling::parse`] takes.
    pub fn parse(value: &str) -> Result<Self, RopeScalingError> {
        if value.trim() == "auto" {
            return Ok(Self::Auto);
        }
        RopeScaling::parse(value).map(Self::Fixed)
    }

    /// Whether the table is left to the context ([`Self::resolve`]) rather
    /// than spelled out.
    pub fn is_auto(&self) -> bool {
        matches!(self, Self::Auto)
    }

    /// The table a load of `max_context` tokens runs on. A spelled-out
    /// setting is itself. `auto` is [`RopeScaling::NONE`] within
    /// [`TRAINED_POSITION_ENVELOPE`] (never "YaRN at 1": see the module doc)
    /// and YaRN at the exact ratio past it, with the default temperature and
    /// ramp. A context past [`MAX_YARN_FACTOR`] times the envelope is
    /// refused ([`RopeScalingError::Context`]), never quietly the linear
    /// table.
    pub fn resolve(&self, max_context: u32) -> Result<RopeScaling, RopeScalingError> {
        match self {
            Self::Fixed(scaling) => Ok(*scaling),
            Self::Auto if max_context <= TRAINED_POSITION_ENVELOPE => Ok(RopeScaling::NONE),
            // In tokens, not as a factor: one token past the widest context
            // is a ratio that rounds to the widest factor in `f32`.
            Self::Auto if u64::from(max_context) > widest_auto_context() => {
                Err(RopeScalingError::Context(max_context))
            }
            Self::Auto => {
                RopeScaling::yarn((f64::from(max_context) / f64::from(TRAINED_POSITION_ENVELOPE)) as f32)
            }
        }
    }
}

/// The widest context `auto` scales: [`MAX_YARN_FACTOR`] times the trained
/// position envelope.
fn widest_auto_context() -> u64 {
    MAX_YARN_FACTOR as u64 * u64::from(TRAINED_POSITION_ENVELOPE)
}

/// Round-trips through [`RopeSetting::parse`].
impl std::fmt::Display for RopeSetting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::Fixed(scaling) => scaling.fmt(f),
        }
    }
}

/// Why a `--model-rope-scaling` value is not usable.
#[derive(Debug, Clone, PartialEq)]
pub enum RopeScalingError {
    /// The value is neither `auto`, `none` nor `yarn:F[,...]`.
    Shape(String),
    /// A field is not a finite number.
    Field { field: &'static str, got: String },
    /// A field is present twice, or is not one of `t`, `bf`, `bs`.
    UnknownField(String),
    /// The factor is outside `{0, 1} ∪ (1, MAX_YARN_FACTOR]`.
    Factor(f32),
    /// The attention temperature is not positive.
    Temperature(f32),
    /// The ramp is not `beta_fast > beta_slow > 0`.
    Ramp { fast: f32, slow: f32 },
    /// `auto` at a context, in tokens, past what [`MAX_YARN_FACTOR`] scales
    /// the trained position envelope to.
    Context(u32),
}

impl std::fmt::Display for RopeScalingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shape(got) => write!(f, "expected `auto`, `none` or `yarn:F[,t=..][,bf=..][,bs=..]`, got `{got}`"),
            Self::Field { field, got } => write!(f, "{field} must be a finite number, got `{got}`"),
            Self::UnknownField(got) => {
                write!(f, "unknown or repeated field `{got}` (expected `t=`, `bf=` or `bs=`)")
            }
            Self::Factor(got) => write!(
                f,
                "the YaRN factor must be in (1, {MAX_YARN_FACTOR}] (or 1 for no scaling), got {got}"
            ),
            Self::Temperature(got) => {
                write!(f, "the attention temperature must be positive, got {got}")
            }
            Self::Ramp { fast, slow } => {
                write!(f, "the ramp needs bf > bs > 0, got bf={fast} and bs={slow}")
            }
            Self::Context(got) => write!(
                f,
                "`auto` scales a context of at most {} tokens, {MAX_YARN_FACTOR}x the {TRAINED_POSITION_ENVELOPE} trained positions, got {got}",
                widest_auto_context()
            ),
        }
    }
}

impl std::error::Error for RopeScalingError {}

/// Two scalings are the same when their four scalars are bit-identical.
/// Written out rather than derived because `f32` is not [`Eq`], and
/// `EngineShape` — which carries this — derives it.
impl PartialEq for RopeScaling {
    fn eq(&self, other: &Self) -> bool {
        self.bits() == other.bits()
    }
}

/// Sound because the comparison above is over the raw bits, so it is
/// reflexive even for the NaN no constructor here will produce.
impl Eq for RopeScaling {}

impl RopeScaling {
    /// No scaling: the linear table, unchanged from before #227.
    pub const NONE: Self = Self {
        factor: 0.0,
        temperature: DEFAULT_YARN_TEMPERATURE,
        beta_fast: DEFAULT_YARN_BETA_FAST,
        beta_slow: DEFAULT_YARN_BETA_SLOW,
    };

    /// YaRN at `factor` with the default temperature and ramp.
    pub fn yarn(factor: f32) -> Result<Self, RopeScalingError> {
        Self::new(factor, DEFAULT_YARN_TEMPERATURE, DEFAULT_YARN_BETA_FAST, DEFAULT_YARN_BETA_SLOW)
    }

    /// A fully spelled-out scaling. The ranges are the leaf's own
    /// (`ignis::rope_scaling_rejection`), checked here so an operator's typo
    /// fails at startup rather than after the weights are on the card.
    pub fn new(
        factor: f32,
        temperature: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> Result<Self, RopeScalingError> {
        if !factor.is_finite() || factor < 0.0 || (factor > 0.0 && factor < 1.0) || factor > MAX_YARN_FACTOR
        {
            return Err(RopeScalingError::Factor(factor));
        }
        let scaling = Self { factor, temperature, beta_fast, beta_slow };
        if !scaling.is_yarn() {
            // No scaling, however it was spelled: a factor of 1 and a factor
            // of 0 build the same linear table, and the ramp is unread, so a
            // caller that left it at whatever is not refused for it either.
            return Ok(Self::NONE);
        }
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(RopeScalingError::Temperature(temperature));
        }
        if !beta_fast.is_finite() || !beta_slow.is_finite() || beta_slow <= 0.0 || beta_fast <= beta_slow
        {
            return Err(RopeScalingError::Ramp { fast: beta_fast, slow: beta_slow });
        }
        Ok(scaling)
    }

    /// `none` or `yarn:F[,t=<c>][,bf=<n>][,bs=<n>]` — the reference's own
    /// `--model-rope-scaling` grammar, so an operator's ninfer command line reads
    /// the same here.
    pub fn parse(value: &str) -> Result<Self, RopeScalingError> {
        let text = value.trim();
        if text == "none" {
            return Ok(Self::NONE);
        }
        let Some(rest) = text.strip_prefix("yarn:") else {
            return Err(RopeScalingError::Shape(value.to_owned()));
        };
        let mut fields = rest.split(',');
        let factor = parse_field(fields.next().unwrap_or_default(), "the YaRN factor")?;
        let mut temperature = None;
        let mut beta_fast = None;
        let mut beta_slow = None;
        for field in fields {
            let Some((key, raw)) = field.split_once('=') else {
                return Err(RopeScalingError::UnknownField(field.to_owned()));
            };
            let slot = match key.trim() {
                "t" => &mut temperature,
                "bf" => &mut beta_fast,
                "bs" => &mut beta_slow,
                _ => return Err(RopeScalingError::UnknownField(field.to_owned())),
            };
            if slot.is_some() {
                return Err(RopeScalingError::UnknownField(field.to_owned()));
            }
            *slot = Some(parse_field(raw, field_name(key.trim()))?);
        }
        Self::new(
            factor,
            temperature.unwrap_or(DEFAULT_YARN_TEMPERATURE),
            beta_fast.unwrap_or(DEFAULT_YARN_BETA_FAST),
            beta_slow.unwrap_or(DEFAULT_YARN_BETA_SLOW),
        )
    }

    /// Whether this load builds a YaRN table rather than the linear one.
    pub fn is_yarn(&self) -> bool {
        self.factor > 1.0
    }

    /// The four scalars, in the order the ABI carries them.
    pub fn factor(&self) -> f32 {
        self.factor
    }

    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    pub fn beta_fast(&self) -> f32 {
        self.beta_fast
    }

    pub fn beta_slow(&self) -> f32 {
        self.beta_slow
    }

    fn bits(&self) -> [u32; 4] {
        [
            self.factor.to_bits(),
            self.temperature.to_bits(),
            self.beta_fast.to_bits(),
            self.beta_slow.to_bits(),
        ]
    }
}

/// Round-trips through [`RopeScaling::parse`], which is what puts it in the
/// request log and in `make config` output as the operator typed it.
impl std::fmt::Display for RopeScaling {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if !self.is_yarn() {
            return f.write_str("none");
        }
        write!(f, "yarn:{}", self.factor)?;
        if self.temperature != DEFAULT_YARN_TEMPERATURE {
            write!(f, ",t={}", self.temperature)?;
        }
        if self.beta_fast != DEFAULT_YARN_BETA_FAST {
            write!(f, ",bf={}", self.beta_fast)?;
        }
        if self.beta_slow != DEFAULT_YARN_BETA_SLOW {
            write!(f, ",bs={}", self.beta_slow)?;
        }
        Ok(())
    }
}

fn field_name(key: &str) -> &'static str {
    match key {
        "t" => "the attention temperature",
        "bf" => "beta_fast",
        _ => "beta_slow",
    }
}

fn parse_field(raw: &str, field: &'static str) -> Result<f32, RopeScalingError> {
    let text = raw.trim();
    match text.parse::<f32>() {
        Ok(value) if value.is_finite() => Ok(value),
        _ => Err(RopeScalingError::Field { field, got: text.to_owned() }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_is_the_default_and_the_linear_table() {
        assert_eq!(RopeScaling::default().factor(), 0.0);
        assert!(!RopeScaling::default().is_yarn());
        assert_eq!(RopeScaling::parse("none").unwrap(), RopeScaling::NONE);
        assert_eq!(RopeScaling::NONE.to_string(), "none");
    }

    #[test]
    fn a_factor_of_one_is_no_scaling() {
        // Not "YaRN at 1": the leaf must take the linear branch, because an
        // attention factor of exactly 1 is what keeps ops::rope on its
        // legacy FP32 angle route.
        let one = RopeScaling::parse("yarn:1").unwrap();
        assert!(!one.is_yarn());
        assert_eq!(one, RopeScaling::NONE);
        assert_eq!(one.to_string(), "none");
    }

    #[test]
    fn the_reference_grammar_parses() {
        let plain = RopeScaling::parse("yarn:4").unwrap();
        assert!(plain.is_yarn());
        assert_eq!(plain.factor(), 4.0);
        assert_eq!(plain.temperature(), DEFAULT_YARN_TEMPERATURE);
        assert_eq!(plain.beta_fast(), DEFAULT_YARN_BETA_FAST);
        assert_eq!(plain.beta_slow(), DEFAULT_YARN_BETA_SLOW);

        let full = RopeScaling::parse("yarn:2.5,t=0.25,bf=16,bs=2").unwrap();
        assert_eq!(full.factor(), 2.5);
        assert_eq!(full.temperature(), 0.25);
        assert_eq!(full.beta_fast(), 16.0);
        assert_eq!(full.beta_slow(), 2.0);
        assert_eq!(full.to_string(), "yarn:2.5,t=0.25,bf=16,bs=2");
        assert_eq!(RopeScaling::parse(&full.to_string()).unwrap(), full);
    }

    #[test]
    fn a_malformed_value_names_what_is_wrong() {
        assert!(matches!(RopeScaling::parse("linear:4"), Err(RopeScalingError::Shape(_))));
        assert!(matches!(RopeScaling::parse("yarn:banana"), Err(RopeScalingError::Field { .. })));
        assert!(matches!(RopeScaling::parse("yarn:4,x=1"), Err(RopeScalingError::UnknownField(_))));
        assert!(matches!(RopeScaling::parse("yarn:4,bf"), Err(RopeScalingError::UnknownField(_))));
        // A repeated field is a typo, not a last-one-wins.
        assert!(matches!(
            RopeScaling::parse("yarn:4,t=0.2,t=0.3"),
            Err(RopeScalingError::UnknownField(_))
        ));
    }

    #[test]
    fn the_ranges_are_the_leafs_own() {
        assert!(matches!(RopeScaling::parse("yarn:0.5"), Err(RopeScalingError::Factor(_))));
        assert!(matches!(RopeScaling::parse("yarn:-1"), Err(RopeScalingError::Factor(_))));
        assert!(matches!(RopeScaling::parse("yarn:65"), Err(RopeScalingError::Factor(_))));
        assert!(RopeScaling::parse("yarn:64").is_ok());
        assert!(matches!(
            RopeScaling::parse("yarn:4,t=0"),
            Err(RopeScalingError::Temperature(_))
        ));
        assert!(matches!(
            RopeScaling::parse("yarn:4,bf=1,bs=32"),
            Err(RopeScalingError::Ramp { .. })
        ));
        assert!(matches!(
            RopeScaling::parse("yarn:4,bs=0"),
            Err(RopeScalingError::Ramp { .. })
        ));
    }

    #[test]
    fn a_ramp_without_a_factor_is_not_refused() {
        // Nothing reads the ramp without a factor, so a caller that zeroes
        // the struct is asking for no scaling, not for a broken ramp.
        let none = RopeScaling::new(0.0, 0.0, 0.0, 0.0).unwrap();
        assert!(!none.is_yarn());
        assert_eq!(none, RopeScaling::NONE);
    }

    #[test]
    fn auto_is_no_scaling_within_the_trained_position_envelope() {
        for context in [1, 16_384, 262_144] {
            let resolved = RopeSetting::Auto.resolve(context).unwrap();
            // NONE exactly, not YaRN at 1: the leaf's route depends on it.
            assert_eq!(resolved, RopeScaling::NONE, "{context}");
            assert!(!resolved.is_yarn());
        }
    }

    #[test]
    fn auto_is_yarn_at_the_exact_ratio_past_the_trained_position_envelope() {
        for (context, factor) in [(524_288, 2.0), (786_432, 3.0), (1_048_576, 4.0), (589_824, 2.25)] {
            let resolved = RopeSetting::Auto.resolve(context).unwrap();
            assert!(resolved.is_yarn(), "{context}");
            assert_eq!(resolved.factor(), factor, "{context}");
            assert_eq!(resolved, RopeScaling::yarn(factor).unwrap(), "default temperature and ramp");
        }
        assert_eq!(RopeSetting::Auto.resolve(262_145).unwrap().factor(), 262_145.0 / 262_144.0);
    }

    /// Past 64x the envelope (16,777,216 tokens) `auto` has no table to give:
    /// refused naming the context and the most it scales, never quietly the
    /// linear one. The ratio one token past it rounds to 64.0 in `f32`, so
    /// the boundary is the token count, not the factor.
    #[test]
    fn auto_past_the_widest_factor_is_refused_naming_the_context_and_the_most_it_scales() {
        assert_eq!(RopeSetting::Auto.resolve(16_777_216).unwrap().factor(), 64.0);
        let refused = RopeSetting::Auto.resolve(16_777_217).unwrap_err();
        assert_eq!(refused, RopeScalingError::Context(16_777_217));
        let message = refused.to_string();
        assert!(message.contains("16777217") && message.contains("16777216"), "{message}");
        // A table spelled out is not the context's to refuse.
        assert_eq!(RopeSetting::parse("yarn:2").unwrap().resolve(u32::MAX).unwrap().factor(), 2.0);
    }

    #[test]
    fn auto_resolves_to_the_same_load_as_the_explicit_factor() {
        assert_eq!(RopeSetting::Auto.resolve(1_048_576), RopeSetting::parse("yarn:4").unwrap().resolve(1_048_576));
    }

    #[test]
    fn an_explicit_setting_ignores_the_context() {
        let none = RopeSetting::parse("none").unwrap();
        assert_eq!(none.resolve(1_048_576).unwrap(), RopeScaling::NONE);
        let two = RopeSetting::parse("yarn:2").unwrap();
        assert_eq!(two.resolve(16_384).unwrap().factor(), 2.0);
        assert!(!two.is_auto() && !none.is_auto());
    }

    #[test]
    fn the_setting_round_trips_and_defaults_to_auto() {
        assert_eq!(RopeSetting::default(), RopeSetting::Auto);
        assert!(RopeSetting::Auto.is_auto());
        assert_eq!(RopeSetting::parse(" auto ").unwrap(), RopeSetting::Auto);
        assert_eq!(RopeSetting::Auto.to_string(), "auto");
        for text in ["none", "yarn:4", "yarn:2.5,t=0.25"] {
            assert_eq!(RopeSetting::parse(text).unwrap().to_string(), text);
        }
        assert!(matches!(RopeSetting::parse("yarn:0.5"), Err(RopeScalingError::Factor(_))));
        let shape = RopeSetting::parse("autom").unwrap_err();
        assert!(matches!(shape, RopeScalingError::Shape(_)));
        assert!(shape.to_string().starts_with("expected `auto`, `none` or `yarn:F"), "{shape}");
    }
}
