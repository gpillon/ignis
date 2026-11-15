//! The 27B MTP head's proposal readout (GitHub #620, spec #618): a test-only
//! measurement seam beside the program (`kernel/include/ignis_mtp_readout.h`).
//!
//! What the head proposes at a sequence's frontier -- its first draft, the
//! draft's scores, the head's hidden -- the continuation hidden the head reads,
//! the head's attention state at any position it holds, and the load's output
//! head over any hidden. Enough for a test to hold the head to a
//! double-precision recomputation from the stored weights. GitHub #626: and
//! the head's whole chain at the frontier ([`chain`]), every draft with the
//! head's probability of it and its leading candidates, which the replay of
//! recorded agent turns under MTP reads before each teacher-forced round. No
//! serving path calls it (`crates/core/tests/replay_seams_test_only.rs`).

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

/// GitHub #626: the leaf's chain readout struct, 1:1 with `struct
/// ignis_mtp_chain_readout`.
#[repr(C)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct IgnisMtpChainReadout {
    size: u32,
    window: u32,
    extent: u32,
    reserved: u32,
    out_drafts: *mut i32,
    out_scores: *mut u16,
}

#[cfg(feature = "cuda")]
mod ffi {
    use std::os::raw::c_char;

    use super::{IgnisMtpChainReadout, IgnisMtpReadout};
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
        pub fn ignis_mtp_readout_round_hidden(model: *mut IgnisModel, lanes: u32, out_hidden: *mut u16) -> i32;
        pub fn ignis_mtp_readout_chain(
            model: *mut IgnisModel,
            pool: *mut IgnisSeqPool,
            seq: *mut IgnisSeq,
            readout: *const IgnisMtpChainReadout,
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

/// GitHub #621: the last verify round's target post-final-norm hidden of
/// every column, for its `lanes` lanes at the load's `window`: BF16 bits
/// `[lanes][window + 1][hidden]`, lane b's column j the hidden at its
/// frontier plus j -- what the round's commit built the head's rows from.
/// Read it before the next round or prefill.
#[cfg(feature = "cuda")]
pub fn round_hidden(model: &Model, lanes: u32, window: u32) -> Result<Vec<u16>, String> {
    let config = crate::compute::ModelConfig::qwen38_27b();
    let mut hidden = vec![0u16; lanes as usize * (window as usize + 1) * config.hidden as usize];
    let rc = unsafe { ffi::ignis_mtp_readout_round_hidden(model.handle(), lanes, hidden.as_mut_ptr()) };
    if rc != 0 {
        return Err(last_error());
    }
    Ok(hidden)
}

/// GitHub #626: the head's leading candidates the chain readout keeps at each
/// draft position, as many as the DFlash2 readout's (`dflash2_readout::TOP_K`).
pub const TOP_K: usize = 16;

/// Scores more than this far below a row's best are left out of its
/// partition function: at most `vocab * e^-40` of it, under 1e-12 on the
/// 27B's vocabulary, past what an f32 probability holds.
const TAIL_SPAN: f64 = 40.0;

fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// GitHub #626: one draft position of the head's chain -- the draft, the
/// head's probability of it, and its leading candidates.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainStep {
    pub draft: i32,
    /// The draft's share of the softmax at temperature 1 over the step's
    /// scores, the whole vocabulary: the head's confidence in it.
    pub probability: f32,
    /// The step's best-scored ids, descending, the lower id first on a tie:
    /// the first is the draft.
    pub candidate_ids: Vec<i32>,
    /// Their probabilities, as [`Self::probability`] is the draft's.
    pub candidate_probabilities: Vec<f32>,
}

impl ChainStep {
    /// The step whose draft the device chose as `draft` from `scores` (the
    /// target's output head over the head's hidden, BF16 bits `[vocab]`),
    /// with its `top` best candidates. The device's draft is the scores'
    /// argmax, the lowest id on a tie; one this ranking does not put first is
    /// refused, an inconsistency between the two.
    pub fn from_scores(draft: i32, scores: &[u16], top: usize) -> Result<Self, String> {
        if top == 0 {
            return Err("mtp chain readout: no candidates asked for".to_string());
        }
        // Largest first; a later id enters only above an equal one, so the
        // lower id stays first on a tie.
        let mut best: Vec<(f32, i32)> = Vec::with_capacity(top + 1);
        let mut floor = f32::NEG_INFINITY;
        for (id, &bits) in scores.iter().enumerate() {
            let value = bf16(bits);
            if value.is_nan() || (best.len() == top && value <= floor) {
                continue;
            }
            let at = best.partition_point(|&(held, _)| held >= value);
            best.insert(at, (value, id as i32));
            best.truncate(top);
            if best.len() == top {
                floor = best[top - 1].0;
            }
        }
        let Some(&(max, first)) = best.first() else {
            return Err("mtp chain readout: a step with no scores".to_string());
        };
        if first != draft {
            return Err(format!("mtp chain readout: the draft {draft} is not the scores' best, {first}"));
        }
        let max = f64::from(max);
        let partition: f64 = scores
            .iter()
            .map(|&bits| f64::from(bf16(bits)) - max)
            .filter(|&delta| delta > -TAIL_SPAN)
            .map(f64::exp)
            .sum();
        let log_partition = max + partition.ln();
        let probability = |value: f32| (f64::from(value) - log_partition).exp() as f32;
        Ok(Self {
            draft,
            probability: probability(best[0].0),
            candidate_ids: best.iter().map(|&(_, id)| id).collect(),
            candidate_probabilities: best.iter().map(|&(value, _)| probability(value)).collect(),
        })
    }
}

/// GitHub #626: the head's chain at a sequence's frontier, one
/// [`ChainStep`] per draft position up to the extent read.
#[derive(Debug, Clone, PartialEq)]
pub struct MtpChain {
    pub steps: Vec<ChainStep>,
    /// Each step's scores, BF16 bits `[extent][vocab]`, when asked for.
    pub scores: Option<Vec<u16>>,
}

impl MtpChain {
    pub fn drafts(&self) -> Vec<i32> {
        self.steps.iter().map(|step| step.draft).collect()
    }

    pub fn probabilities(&self) -> Vec<f32> {
        self.steps.iter().map(|step| step.probability).collect()
    }
}

/// GitHub #626: the drafts the head's chain proposes at `sequence`'s
/// frontier for a round at `extent` (1 to the load's `window`), each with
/// the head's probability of it and its [`TOP_K`] leading candidates -- the
/// round's own chain run once, eagerly, at width 1, without the round
/// (kernel/include/ignis_mtp_readout.h `ignis_mtp_readout_chain`). Its drafts
/// are the ones the sequence's next verify round at that extent proposes, and
/// it leaves nothing that round reads: it writes the head's rows at and past
/// the head's frontier, which every later step writes before it reads them,
/// and under hq-e8-2b gives the head's residual window and ring words back
/// as they were -- hence `&mut`. With `with_scores` it returns each step's
/// scores too.
#[cfg(feature = "cuda")]
pub fn chain(
    model: &Model,
    pool: &SeqPool,
    sequence: &mut Seq<'_>,
    extent: u32,
    window: u32,
    with_scores: bool,
) -> Result<MtpChain, String> {
    let vocab = crate::compute::ModelConfig::qwen38_27b().vocab as usize;
    // Sized for what the leaf may write; an extent past the window it
    // refuses before writing anything.
    let steps = extent.min(window) as usize;
    let mut drafts = vec![-1i32; steps];
    let mut scores = vec![0u16; steps * vocab];
    let readout = IgnisMtpChainReadout {
        size: std::mem::size_of::<IgnisMtpChainReadout>() as u32,
        window,
        extent,
        reserved: 0,
        out_drafts: drafts.as_mut_ptr(),
        out_scores: scores.as_mut_ptr(),
    };
    let rc = unsafe { ffi::ignis_mtp_readout_chain(model.handle(), pool.handle(), sequence.handle(), &readout) };
    if rc != 0 {
        return Err(last_error());
    }
    // A thread a step: the host's reduction of the rows is what a replay's
    // round waits on.
    let reduced: Vec<Result<ChainStep, String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = drafts
            .iter()
            .zip(scores.chunks_exact(vocab))
            .map(|(&draft, row)| scope.spawn(move || ChainStep::from_scores(draft, row, TOP_K)))
            .collect();
        handles.into_iter().map(|handle| handle.join().expect("a step's reduction panicked")).collect()
    });
    let steps = reduced.into_iter().collect::<Result<Vec<_>, _>>()?;
    Ok(MtpChain { steps, scores: with_scores.then_some(scores) })
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

    /// GitHub #626: the chain readout's mirror is the leaf's too.
    #[test]
    fn the_chain_readout_mirror_is_the_leaf_structs_layout() {
        assert_eq!(std::mem::size_of::<IgnisMtpChainReadout>(), 32);
        assert_eq!(std::mem::offset_of!(IgnisMtpChainReadout, window), 4);
        assert_eq!(std::mem::offset_of!(IgnisMtpChainReadout, extent), 8);
        assert_eq!(std::mem::offset_of!(IgnisMtpChainReadout, out_drafts), 16);
        assert_eq!(std::mem::offset_of!(IgnisMtpChainReadout, out_scores), 24);
    }

    fn bf16_row(values: &[f32]) -> Vec<u16> {
        values
            .iter()
            .map(|v| {
                let bits = v.to_bits();
                assert_eq!(bits & 0xffff, 0, "{v} is not a BF16 value");
                (bits >> 16) as u16
            })
            .collect()
    }

    /// A worked example: the softmax at T = 1 over the whole row, the draft
    /// the first of the two tied best ids, the candidates descending with
    /// the lower id first on a tie.
    #[test]
    fn a_steps_probability_is_its_drafts_share_of_the_whole_vocabularys_softmax() {
        let row = bf16_row(&[1.0, 3.0, 2.0, 3.0, 0.0]);
        let step = ChainStep::from_scores(1, &row, 4).unwrap();
        assert_eq!(step.draft, 1);
        assert!((f64::from(step.probability) - 0.391_695_768_812_087).abs() < 1e-7, "{}", step.probability);
        assert_eq!(step.candidate_ids, vec![1, 3, 2, 0]);
        let expected = [0.391_695_768_812_087, 0.391_695_768_812_087, 0.144_096_820_539_809, 0.053_010_257_814_766];
        for (got, want) in step.candidate_probabilities.iter().zip(expected) {
            assert!((f64::from(*got) - want).abs() < 1e-7, "{got} against {want}");
        }
        // Past the vocabulary, every id is a candidate.
        let step = ChainStep::from_scores(1, &row, 8).unwrap();
        assert_eq!(step.candidate_ids, vec![1, 3, 2, 0, 4]);
        let total: f32 = step.candidate_probabilities.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "{total}");
    }

    /// A score far below the row's best adds nothing a probability can hold.
    #[test]
    fn a_far_tail_changes_no_probability() {
        let step = ChainStep::from_scores(0, &bf16_row(&[10.0, -60.0, 9.5]), 2).unwrap();
        assert!((f64::from(step.probability) - 0.622_459_331_201_854_6).abs() < 1e-7, "{}", step.probability);
        assert!((f64::from(step.candidate_probabilities[1]) - 0.377_540_668_798_145_5).abs() < 1e-7);
    }

    /// The device's draft is the argmax, the lowest id on a tie: a draft the
    /// host's ranking does not put first is an inconsistency, refused.
    #[test]
    fn a_draft_that_is_not_the_best_score_is_refused() {
        let row = bf16_row(&[1.0, 3.0, 2.0, 3.0, 0.0]);
        for draft in [3, 2, -1, 5] {
            let err = ChainStep::from_scores(draft, &row, 4).expect_err("not the argmax");
            assert!(err.contains("not the scores' best"), "{err}");
        }
    }

    /// Over a vocabulary-sized row the candidates are the row's best 16, in
    /// the order a full sort gives.
    #[test]
    fn the_candidates_are_the_rows_best_sixteen_in_order() {
        // A deterministic spread with repeats: BF16 keeps 8 bits of mantissa.
        let values: Vec<f32> = (0..248_320u32)
            .map(|i| {
                let x = (i.wrapping_mul(2_654_435_761) >> 20) as f32 / 256.0 - 8.0;
                f32::from_bits(x.to_bits() & 0xffff_0000)
            })
            .collect();
        let row = bf16_row(&values);
        let mut order: Vec<usize> = (0..values.len()).collect();
        order.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
        let step = ChainStep::from_scores(order[0] as i32, &row, TOP_K).unwrap();
        let expected: Vec<i32> = order[..TOP_K].iter().map(|&i| i as i32).collect();
        assert_eq!(step.candidate_ids, expected);
        for pair in step.candidate_probabilities.windows(2) {
            assert!(pair[0] >= pair[1], "{pair:?}");
        }
        assert_eq!(step.probability, step.candidate_probabilities[0]);
    }
}
