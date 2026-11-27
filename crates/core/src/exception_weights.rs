//! How the 27B holds its nine BF16 exception tensors on the device (GitHub
//! #341, spec prefill-5090/01 lever 1): as stored in the artifact, or
//! converted at load to FP8 row-scale (ADR 0065), which a call past 8 rows --
//! a prefill chunk, a verify round of two lanes or more -- multiplies on the
//! FP8 tensor cores (activations quantized per token, W8A8), and a decode
//! round or a one-lane verify round with BF16 activations (W8A16). A load
//! argument, fixed for the life of the load:
//! the binder places the converted tensors (`ignis_artifact::ModelScope::fp8_exceptions`)
//! and the leaf dispatches on their format.

use std::fmt;

/// How a load holds them: `bf16` (as stored) or `fp8`. The engine's
/// default is `bf16`, so a load bound directly keeps the stored numerics
/// unless it names FP8; the server's `model.exception_weights` is an
/// [`ExceptionWeightsSetting`], `auto` by default, which its config
/// resolution decides by the card (ADR 0065).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExceptionWeights {
    /// As the artifact stores them.
    #[default]
    Bf16,
    /// Converted at load to FP8 row-scale.
    Fp8,
}

impl ExceptionWeights {
    /// The spellings `parse` accepts, canonical first.
    pub const NAMES: [&'static str; 2] = ["bf16", "fp8"];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Fp8 => "fp8",
        }
    }

    /// Whether the load converts them.
    pub const fn is_fp8(self) -> bool {
        matches!(self, Self::Fp8)
    }

    /// Case-insensitive.
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "bf16" => Ok(Self::Bf16),
            "fp8" => Ok(Self::Fp8),
            other => Err(format!("unknown exception weights `{other}` (expected `bf16` or `fp8`)")),
        }
    }
}

impl fmt::Display for ExceptionWeights {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// `--model-exception-weights` as configured (GitHub #682, ADR 0065):
/// `auto`, the default, or the weights spelled out. The server's config
/// resolution decides `auto` beside the card it reads: `fp8` on a card that
/// runs the block-scaled FP8 MMA (compute capability 12.0), `bf16` on any
/// other card and when no card is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExceptionWeightsSetting {
    #[default]
    Auto,
    Fixed(ExceptionWeights),
}

impl ExceptionWeightsSetting {
    /// The spellings `parse` accepts, canonical first.
    pub const NAMES: [&'static str; 3] = ["auto", "bf16", "fp8"];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Fixed(weights) => weights.as_str(),
        }
    }

    /// Case-insensitive.
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            other => ExceptionWeights::parse(other)
                .map(Self::Fixed)
                .map_err(|_| format!("unknown exception weights `{other}` (expected `auto`, `bf16` or `fp8`)")),
        }
    }
}

impl fmt::Display for ExceptionWeightsSetting {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The engine's default stays `bf16` (GitHub #682): `auto` is the
    /// configured setting's default, which the server resolves.
    #[test]
    fn bf16_is_the_engines_default_and_auto_the_settings() {
        assert_eq!(ExceptionWeights::default(), ExceptionWeights::Bf16);
        assert!(!ExceptionWeights::Bf16.is_fp8() && ExceptionWeights::Fp8.is_fp8());
        assert_eq!(ExceptionWeightsSetting::default(), ExceptionWeightsSetting::Auto);
        for name in ExceptionWeightsSetting::NAMES {
            assert_eq!(ExceptionWeightsSetting::parse(name).map(ExceptionWeightsSetting::as_str), Ok(name));
        }
        assert_eq!(ExceptionWeightsSetting::parse("FP8"), Ok(ExceptionWeightsSetting::Fixed(ExceptionWeights::Fp8)));
        assert!(ExceptionWeightsSetting::parse("nvfp4").unwrap_err().contains("`auto`, `bf16` or `fp8`"));
    }

    #[test]
    fn parse_is_case_insensitive_and_refuses_the_rest() {
        assert_eq!(ExceptionWeights::parse("FP8"), Ok(ExceptionWeights::Fp8));
        for name in ExceptionWeights::NAMES {
            assert_eq!(ExceptionWeights::parse(name).map(ExceptionWeights::as_str), Ok(name));
        }
        assert!(ExceptionWeights::parse("nvfp4").unwrap_err().contains("fp8"));
    }
}
