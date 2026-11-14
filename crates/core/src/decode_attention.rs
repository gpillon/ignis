//! The attention route of the 27B's decode and verify rounds over hq-e8-2b
//! (GitHub #572, spec #570), which no load option names: the leaf decides it
//! once per load from the card, or the measurement switch
//! `IGNIS_HQ_DECODE_ATTENTION` names today's route, and reports it
//! ([`crate::model_load::Model::hq_decode_attention`]).

/// `enum ignis_hq_decode_attention` (`kernel/include/ignis_model.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HqDecodeAttention {
    /// The vendored kernel's small-T route: every card but Blackwell, a BF16
    /// load, and what `IGNIS_HQ_DECODE_ATTENTION=vendored` names.
    Vendored,
    /// ignis's warp-specialized route (`kernel/src/hq_verify_ws.cuh`): a
    /// Blackwell card's hq-e8-2b load unless the switch names the other.
    Ours,
}

impl HqDecodeAttention {
    /// The leaf's code: 0 the vendored route, 1 ours; `None` for any other.
    pub const fn from_abi(code: u32) -> Option<Self> {
        match code {
            0 => Some(Self::Vendored),
            1 => Some(Self::Ours),
            _ => None,
        }
    }

    /// The switch's own spellings, which the load's `ignis.runtime.decode_routes`
    /// line reports.
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

    /// The leaf's codes, and the switch's spellings.
    #[test]
    fn the_hq_decode_attention_routes_are_the_headers_codes() {
        assert_eq!(HqDecodeAttention::from_abi(0), Some(HqDecodeAttention::Vendored));
        assert_eq!(HqDecodeAttention::from_abi(1), Some(HqDecodeAttention::Ours));
        assert_eq!(HqDecodeAttention::from_abi(2), None);
        assert_eq!(
            (HqDecodeAttention::Vendored.as_str(), HqDecodeAttention::Ours.as_str()),
            ("vendored", "ours")
        );
    }
}
