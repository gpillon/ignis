//! The 27B MTP head's proposal readout (GitHub #620, spec #618): a test-only
//! measurement seam beside the program (`kernel/include/ignis_mtp_readout.h`).
//!
//! What the head proposes at a sequence's frontier -- its first draft, the
//! draft's scores, the head's hidden -- the continuation hidden the head reads,
//! the head's attention state at any position it holds, and the load's output
//! head over any hidden. Enough for a test to hold the head to a
//! double-precision recomputation from the stored weights. No serving path
//! calls it (`crates/core/tests/replay_seams_test_only.rs`).

#[cfg(feature = "cuda")]
use crate::model_load::Model;
#[cfg(feature = "cuda")]
use crate::seq::{Seq, SeqPool};

/// The leaf's struct, 1:1 with `struct ignis_mtp_readout`: plain `repr(C)`
/// with no kernel behind it, so its layout pin runs in the CPU suite.
#[repr(C)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct IgnisMtpReadout {
    size: u32,
    out_draft: i32,
    out_scores: *mut f32,
    out_head_hidden: *mut u16,
    out_continuation: *mut u16,
}

#[cfg(feature = "cuda")]
mod ffi {
    use std::os::raw::c_char;

    use super::IgnisMtpReadout;
    use crate::model_load::ffi::IgnisModel;
    use crate::seq::ffi::{IgnisSeq, IgnisSeqPool};

    unsafe extern "C" {
        pub fn ignis_mtp_readout(
            model: *mut IgnisModel,
            pool: *mut IgnisSeqPool,
            seq: *mut IgnisSeq,
            readout: *mut IgnisMtpReadout,
        ) -> i32;
        pub fn ignis_mtp_readout_scores(model: *mut IgnisModel, hidden: *const u16, out_scores: *mut f32) -> i32;
        pub fn ignis_mtp_readout_state(
            pool: *const IgnisSeqPool,
            seq: *const IgnisSeq,
            first_position: i32,
            count: i32,
            out_keys: *mut u16,
            out_values: *mut u16,
        ) -> i32;
        pub fn ignis_mtp_readout_ring(
            pool: *const IgnisSeqPool,
            seq: *const IgnisSeq,
            out_target_words: *mut u32,
            out_head_words: *mut u32,
        ) -> i32;
        pub fn ignis_mtp_readout_last_error() -> *const c_char;
    }
}

#[cfg(feature = "cuda")]
fn last_error() -> String {
    let message = unsafe { std::ffi::CStr::from_ptr(ffi::ignis_mtp_readout_last_error()) };
    message.to_string_lossy().into_owned()
}

/// The head's KV width per position: 4 KV heads of 256.
pub const KV_WIDTH: usize = 4 * 256;

/// The head's first draft at a sequence's frontier.
#[derive(Debug, Clone, PartialEq)]
pub struct MtpProposal {
    /// The argmax of [`Self::scores`].
    pub draft: i32,
    /// The target's output head over [`Self::head_hidden`], `[vocab]`.
    pub scores: Vec<f32>,
    /// The head's hidden at the frontier, after its own final norm, BF16
    /// bits `[hidden]`.
    pub head_hidden: Vec<u16>,
    /// The sequence's continuation hidden, BF16 bits `[hidden]`.
    pub continuation: Vec<u16>,
}

/// The head's first draft for `sequence` at its frontier, from its
/// continuation hidden and pending token. Computing it writes the head's key
/// and value at the head's frontier -- a position no committed state holds,
/// which the sequence's next step writes again from the same inputs -- hence
/// `&mut`.
#[cfg(feature = "cuda")]
pub fn propose(model: &Model, pool: &SeqPool, sequence: &mut Seq<'_>) -> Result<MtpProposal, String> {
    let config = crate::compute::ModelConfig::qwen38_27b();
    let mut scores = vec![0f32; config.vocab as usize];
    let mut head_hidden = vec![0u16; config.hidden as usize];
    let mut continuation = vec![0u16; config.hidden as usize];
    let mut readout = IgnisMtpReadout {
        size: std::mem::size_of::<IgnisMtpReadout>() as u32,
        out_draft: -1,
        out_scores: scores.as_mut_ptr(),
        out_head_hidden: head_hidden.as_mut_ptr(),
        out_continuation: continuation.as_mut_ptr(),
    };
    let rc = unsafe { ffi::ignis_mtp_readout(model.handle(), pool.handle(), sequence.handle(), &mut readout) };
    if rc != 0 {
        return Err(last_error());
    }
    Ok(MtpProposal {
        draft: readout.out_draft,
        scores,
        head_hidden,
        continuation,
    })
}

/// `hidden` (BF16 bits `[hidden]`) scored by the load's output head,
/// `[vocab]`: on the 16 GB artifact its trellis head, which no host code
/// decodes.
#[cfg(feature = "cuda")]
pub fn score(model: &Model, hidden: &[u16]) -> Result<Vec<f32>, String> {
    let config = crate::compute::ModelConfig::qwen38_27b();
    if hidden.len() != config.hidden as usize {
        return Err(format!("mtp readout: a hidden of {} values, not {}", hidden.len(), config.hidden));
    }
    let mut scores = vec![0f32; config.vocab as usize];
    let rc = unsafe { ffi::ignis_mtp_readout_scores(model.handle(), hidden.as_ptr(), scores.as_mut_ptr()) };
    if rc != 0 {
        return Err(last_error());
    }
    Ok(scores)
}

/// The head's keys and values of `sequence` at positions
/// `first..first + count`, each `[count][KV_WIDTH]` BF16 bits as its layer
/// stores them (keys after their norm and rotation). A BF16 pool only.
#[cfg(feature = "cuda")]
pub fn head_state(
    pool: &SeqPool,
    sequence: &Seq<'_>,
    first: u32,
    count: u32,
) -> Result<(Vec<u16>, Vec<u16>), String> {
    let elements = count as usize * KV_WIDTH;
    let mut keys = vec![0u16; elements];
    let mut values = vec![0u16; elements];
    let (first, count) = (
        i32::try_from(first).map_err(|e| e.to_string())?,
        i32::try_from(count).map_err(|e| e.to_string())?,
    );
    let rc = unsafe {
        ffi::ignis_mtp_readout_state(pool.handle(), sequence.handle(), first, count, keys.as_mut_ptr(), values.as_mut_ptr())
    };
    if rc != 0 {
        return Err(last_error());
    }
    Ok((keys, values))
}

/// GitHub #621: under hq-e8-2b, `sequence`'s residual window ring words --
/// the target's, then the MTP head's own -- as the device holds them, each
/// comparable with [`crate::hq_ring::HqRing::words`]. A pool built for the
/// head refuses a snapshot until the head's sections are listed (#622), so
/// this is the read of them meanwhile.
#[cfg(feature = "cuda")]
pub fn ring_words(
    pool: &SeqPool,
    sequence: &Seq<'_>,
) -> Result<(crate::hq_ring::RingWords, crate::hq_ring::RingWords), String> {
    let mut target = [0u32; crate::hq_ring::RING_WORDS];
    let mut head = [0u32; crate::hq_ring::RING_WORDS];
    let rc = unsafe {
        ffi::ignis_mtp_readout_ring(pool.handle(), sequence.handle(), target.as_mut_ptr(), head.as_mut_ptr())
    };
    if rc != 0 {
        return Err(last_error());
    }
    Ok((target, head))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The mirror is the leaf's layout (`static_assert` in
    /// kernel/include/ignis_mtp_readout.h).
    #[test]
    fn the_readout_mirror_is_the_leaf_structs_layout() {
        assert_eq!(std::mem::size_of::<IgnisMtpReadout>(), 32);
        assert_eq!(std::mem::offset_of!(IgnisMtpReadout, out_scores), 8);
        assert_eq!(std::mem::offset_of!(IgnisMtpReadout, out_continuation), 24);
    }
}
