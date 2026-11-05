//! The precision of the 27B's prompt attention band pass (GitHub #335, spec
//! 27b-16gb/03 solution 6): the 8-bit band by default, the FP16-accumulate fallback
//! behind a load option. A load argument, fixed for the life of the load:
//! `kernel/src/gqa_layer.cu` hands it to every hq-e8-2b prefill chunk.
//!
//! Spec #503 (GitHub #505) adds the integer band of the 8.x cards, `int8`, and
//! the **prefill linear** route beside it ([`PrefillLinear`]).

use std::fmt;

/// `--model-prefill-attention`: the 8-bit band (the default), `fp16` or
/// `int8`. `8bit` and `fp8` are two spellings of the 8-bit band (spec #503):
/// [`Self::parse`] reads both as [`Self::Fp8`], which is the block-scaled FP8
/// band on an RTX 50 or RTX PRO card and the integer band on an 8.x card,
/// where the server's decision makes it [`Self::Int8`]. It renders `fp8`
/// ([`Self::as_str`]), so a report reads `fp8` on Blackwell and `int8` on 8.x.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrefillAttention {
    /// The band pass in 8 bits: Q·Kᵀ in INT8, P·V in E4M3.
    #[default]
    Fp8,
    /// The FP16-accumulate fallback.
    Fp16,
    /// The integer band: Q·Kᵀ in INT8, P·V on the u8×s8 MMA. The 8-bit band of
    /// the 8.x cards (`8bit` and `fp8` resolve to it there); on any other
    /// card only when named.
    Int8,
}

impl PrefillAttention {
    /// The spellings `parse` accepts: `8bit` and `fp8` name the same band,
    /// rendered `fp8`.
    pub const NAMES: [&'static str; 4] = ["8bit", "fp8", "fp16", "int8"];

    /// `struct ignis_model_load_options::prefill_attention`: 0 = 8-bit,
    /// 1 = FP16 accumulate, 2 = the integer band.
    pub const fn abi_code(self) -> u32 {
        match self {
            Self::Fp8 => 0,
            Self::Fp16 => 1,
            Self::Int8 => 2,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fp8 => "fp8",
            Self::Fp16 => "fp16",
            Self::Int8 => "int8",
        }
    }

    /// Case-insensitive.
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "8bit" | "fp8" => Ok(Self::Fp8),
            "fp16" => Ok(Self::Fp16),
            "int8" => Ok(Self::Int8),
            other => Err(format!("unknown prefill attention `{other}` (expected `8bit`, `fp8`, `fp16` or `int8`)")),
        }
    }
}

impl fmt::Display for PrefillAttention {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// `--model-prefill-linear`: how the 27B's trellis prefill multiplies its
/// matrices (spec #503, GitHub #505).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrefillLinear {
    /// The fused prefill: the trellis decoded inside the GEMM's inner loop,
    /// the multiply in f16. The field's default, which an RTX 50 or RTX PRO
    /// card runs; an 8.x card left to it runs the dense prefill (GitHub #515).
    #[default]
    Fp16,
    /// The dense prefill: each matrix decoded once per chunk into one INT8
    /// buffer and multiplied on the integer MMA.
    Int8,
}

impl PrefillLinear {
    /// The spellings `parse` accepts, canonical first.
    pub const NAMES: [&'static str; 2] = ["fp16", "int8"];

    /// `struct ignis_model_load_options::prefill_linear`: 0 = the fused
    /// prefill, 1 = the dense prefill.
    pub const fn abi_code(self) -> u32 {
        match self {
            Self::Fp16 => 0,
            Self::Int8 => 1,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Int8 => "int8",
        }
    }

    /// Case-insensitive.
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "fp16" => Ok(Self::Fp16),
            "int8" => Ok(Self::Int8),
            other => Err(format!("unknown prefill linear `{other}` (expected `fp16` or `int8`)")),
        }
    }
}

impl fmt::Display for PrefillLinear {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The 27B's gated delta net prefill route (GitHub #514), which no load option
/// names: the leaf decides it once per load from the card's generation, or from
/// the measurement switch `IGNIS_GDN_PREFILL`, and reports it
/// ([`crate::model_load::Model::gdn_prefill`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GdnPrefill {
    /// The vendored chunked kernels: what an RTX 50 or RTX PRO card runs.
    Vendored,
    /// ignis's own kernels in full-rate forms: what an 8.x card runs.
    Ours,
}

impl GdnPrefill {
    /// `enum ignis_gdn_prefill`: 0 the vendored kernels, 1 ours; `None` for
    /// any other code.
    pub const fn from_abi(code: u32) -> Option<Self> {
        match code {
            0 => Some(Self::Vendored),
            1 => Some(Self::Ours),
            _ => None,
        }
    }

    /// The switch's own spellings.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Vendored => "vendored",
            Self::Ours => "ours",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_is_the_default_and_the_abi_codes_are_the_headers() {
        assert_eq!(PrefillAttention::default(), PrefillAttention::Fp8);
        assert_eq!(PrefillAttention::Fp8.abi_code(), 0);
        assert_eq!(PrefillAttention::Fp16.abi_code(), 1);
        assert_eq!(PrefillAttention::Int8.abi_code(), 2);
        assert_eq!(PrefillLinear::default(), PrefillLinear::Fp16);
        assert_eq!((PrefillLinear::Fp16.abi_code(), PrefillLinear::Int8.abi_code()), (0, 1));
    }

    /// The leaf's codes, and the switch's spellings.
    #[test]
    fn the_gdn_prefill_routes_are_the_headers_codes() {
        assert_eq!(GdnPrefill::from_abi(0), Some(GdnPrefill::Vendored));
        assert_eq!(GdnPrefill::from_abi(1), Some(GdnPrefill::Ours));
        assert_eq!(GdnPrefill::from_abi(2), None);
        assert_eq!((GdnPrefill::Vendored.as_str(), GdnPrefill::Ours.as_str()), ("vendored", "ours"));
    }

    #[test]
    fn the_integer_band_and_the_8bit_alias_parse() {
        assert_eq!(PrefillAttention::parse("INT8"), Ok(PrefillAttention::Int8));
        assert_eq!(PrefillAttention::parse("8bit"), Ok(PrefillAttention::Fp8));
        assert_eq!(PrefillAttention::parse("fp8"), Ok(PrefillAttention::Fp8));
        assert_eq!(PrefillLinear::parse("Int8"), Ok(PrefillLinear::Int8));
        for name in PrefillLinear::NAMES {
            assert_eq!(PrefillLinear::parse(name).map(PrefillLinear::as_str), Ok(name));
        }
        assert!(PrefillLinear::parse("fp8").unwrap_err().contains("int8"));
    }

    /// Every spelling reads back as itself, but `8bit`: it and `fp8` are two
    /// spellings of the 8-bit band, which renders `fp8`.
    #[test]
    fn parse_is_case_insensitive_and_refuses_the_rest() {
        assert_eq!(PrefillAttention::parse("FP16"), Ok(PrefillAttention::Fp16));
        for name in PrefillAttention::NAMES {
            let rendered = if name == "8bit" { "fp8" } else { name };
            assert_eq!(PrefillAttention::parse(name).map(PrefillAttention::as_str), Ok(rendered), "{name}");
        }
        assert!(PrefillAttention::parse("bf16").unwrap_err().contains("fp16"));
    }
}
