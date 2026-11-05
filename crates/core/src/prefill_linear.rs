//! How the 27B's trellis body multiplies a prefill chunk (GitHub #509, spec #503): the fused
//! prefill, which decodes the trellis inside its GEMM on fp16 tensor cores, or the dense
//! prefill, which decodes each matrix once per chunk into INT8 and multiplies it by INT8
//! activations on our own integer GEMM (`kernel/include/ignis_dense_prefill.h`). A load
//! argument, fixed for the life of the load: `kernel/src/trellis_body.cu` dispatches every
//! body call wider than the decode route on it.

use std::fmt;

/// `--model-prefill-linear`: `fp16` (the fused prefill, the default) or `int8` (the dense
/// prefill).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrefillLinear {
    /// The fused prefill.
    #[default]
    Fp16,
    /// The dense prefill: W8, A8 per token, an s8 GEMM into int32.
    Int8,
}

impl PrefillLinear {
    /// The spellings `parse` accepts, canonical first.
    pub const NAMES: [&'static str; 2] = ["fp16", "int8"];

    /// `struct ignis_model_load_options::prefill_linear` (enum ignis_prefill_linear): 0 = the
    /// fused prefill, 1 = the dense prefill.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fused_prefill_is_the_default_and_the_abi_codes_are_the_headers() {
        assert_eq!(PrefillLinear::default(), PrefillLinear::Fp16);
        assert_eq!(PrefillLinear::Fp16.abi_code(), 0);
        assert_eq!(PrefillLinear::Int8.abi_code(), 1);
    }

    #[test]
    fn parse_is_case_insensitive_and_refuses_the_rest() {
        assert_eq!(PrefillLinear::parse(" INT8 "), Ok(PrefillLinear::Int8));
        for name in PrefillLinear::NAMES {
            assert_eq!(PrefillLinear::parse(name).map(PrefillLinear::as_str), Ok(name));
        }
        assert!(PrefillLinear::parse("w8a8").unwrap_err().contains("int8"));
    }
}
