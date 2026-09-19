//! YaRN RoPE scaling as a load option (GitHub #227, spec
//! `.scratch/rope-scaling/specs/01-yarn.md`).
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

/// The widest factor a load accepts (`kMaxYarnFactor`,
/// `kernel/src/rope_scaling.h`). 64x the trained envelope is already past
/// every attention envelope the engine can serve, so this is a typo guard,
/// not a capability claim.
pub const MAX_YARN_FACTOR: f32 = 64.0;

/// The attention-temperature coefficient YaRN's own paper uses, and HF's
/// default: the q-side factor is `temperature * ln(factor) + 1`.
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

/// Why a `--rope-scaling` value is not usable.
#[derive(Debug, Clone, PartialEq)]
pub enum RopeScalingError {
    /// The value is neither `none` nor `yarn:F[,...]`.
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
}

impl std::fmt::Display for RopeScalingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shape(got) => write!(f, "expected `none` or `yarn:F[,t=..][,bf=..][,bs=..]`, got `{got}`"),
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
        }
    }
}

impl std::error::Error for RopeScalingError {}

/// Two scalings are the same when their four scalars are bit-identical.
/// Written out rather than derived because `f32` is not [`Eq`] — and this
/// type travels inside the engine shape, which is compared whole when a
/// reload decides whether anything actually changed.
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
            // The ramp is unread without a factor; a caller that left it at
            // whatever is not refused for it.
            return Ok(Self { factor, ..Self::NONE });
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
    /// `--rope-scaling` grammar, so an operator's ninfer command line reads
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
}
