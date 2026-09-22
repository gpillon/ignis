//! Pointing in one pass: the **attention readout** and the **pointing head**
//! (spec `docs/specs/decide/13-point-by-attention-head.md`, GitHub #260,
//! ADR 0038).
//!
//! One attention head of the served model, read at the position after the
//! forced `{"x":`, lands inside the target more often than the ten-round
//! digit chain does. Three things live here, and they are the host's whole
//! part of it:
//!
//! - **What a job asks the leaf to read** ([`AttentionQuery`]): one query
//!   head of one GQA layer, dotted with the keys of one span of the prompt.
//!   What comes back across the `Compute` seam is one pre-softmax score per
//!   key of that span and nothing else — never a full attention row, never a
//!   logits row.
//! - **Which head points** ([`calibrated_head`]): a constant chosen with
//!   labelled scenes and cross-validation, keyed to the artifact's content
//!   hash. Unlike the answer alphabet it cannot be computed at load, and like
//!   the alphabet it is refused against any other load.
//! - **How a map becomes a point** ([`read_head_map`]): TAG's region rule,
//!   the one every number in the findings was measured with.

use crate::identity::ArtifactHash;

/// One head of one GQA layer: the head `/v1/decide` reads to answer a
/// `point` in one pass.
///
/// `gqa_ordinal` counts GQA layers only (0..16 on the 27B: backbone layer
/// `4 * ordinal + 3`), and `query_head` is a query head of that layer
/// (0..24), not a KV head — six query heads share each KV head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PointingHead {
    /// The GQA layer, counted among GQA layers only.
    pub gqa_ordinal: u32,
    /// The query head within that layer.
    pub query_head: u32,
}

impl std::fmt::Display for PointingHead {
    /// The findings' name for a head: `L39.h10` is GQA ordinal 9's backbone
    /// layer 39, query head 10.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "L{}.h{}", 4 * self.gqa_ordinal + 3, self.query_head)
    }
}

/// The served NVFP4 27B artifact (`qwen3_8_27b_nvfp4full-v2.ninfer`)'s
/// content hash, as [`ignis_artifact::Reader::content_hash`] computes it.
///
/// Structural, not a digest of the weights (see [`ArtifactHash`]): a
/// re-quantization that kept every name, format and offset would match it.
/// That is the proxy the whole engine keys retained state on, and it is the
/// right strength here too — a *different* artifact cannot match.
const SERVED_NVFP4_27B: [u8; 32] = [
    0x4b, 0xdc, 0x7b, 0x13, 0x03, 0x17, 0x29, 0x52, 0xcf, 0x50, 0xad, 0x0b, 0xb3, 0x74, 0x8d, 0xd9,
    0xc3, 0x2d, 0x2f, 0xae, 0x30, 0xda, 0x8b, 0x09, 0xd8, 0x45, 0x0e, 0x83, 0xb6, 0x54, 0x25, 0x13,
];

/// The calibration table: artifact content hash → the head that points.
///
/// One entry today. L39.h10 was chosen on the served artifact by
/// cross-validation over labelled synthetic scenes (every fold of every arm
/// picked it) and checked on sets it was never chosen on
/// (`docs/findings/2026-09-21-one-attention-head-points.md`,
/// `2026-09-21-the-head-points-in-the-engine.md`). Recalibrating for a new
/// artifact is the procedure in `docs/specs/decide/13-point-by-attention-head.md`
/// § Further Notes, run with `crates/server/tests/attention_head_point_gpu.rs`.
const CALIBRATED: &[([u8; 32], PointingHead)] = &[(
    SERVED_NVFP4_27B,
    PointingHead {
        gqa_ordinal: 9,
        query_head: 10,
    },
)];

/// The pointing head calibrated for `artifact`, or `None` when nobody
/// calibrated one for it — and then `point` answers with the digit chain.
///
/// Looked up by the full 32-byte hash: a head chosen for one model is never
/// read on another, however alike their names.
pub fn calibrated_head(artifact: ArtifactHash) -> Option<PointingHead> {
    CALIBRATED
        .iter()
        .find(|(hash, _)| hash == artifact.as_bytes())
        .map(|&(_, head)| head)
}

/// Every artifact hash the table names — for the test that fails when the
/// served artifact changes without a recalibration.
pub fn calibrated_artifacts() -> impl Iterator<Item = ArtifactHash> {
    CALIBRATED.iter().map(|(hash, _)| ArtifactHash::from_bytes(*hash))
}

/// What one prefill job asks the leaf to read out of one layer's attention:
/// `head`'s query at the job's **last position**, dotted with the keys at
/// prompt positions `[key_begin, key_begin + key_count)`.
///
/// The third thing the `Compute` seam carries (ADR 0038), shaped like the
/// answer-token readout (ADR 0034): the job names what to read, the outcome
/// carries back exactly that — one `q · k / sqrt(head_dim)` per key, before
/// any softmax — and the full row never crosses.
///
/// The keys are the ones the layer's attention read, as it read them: the
/// cache's pages under BF16, the prompt route's materialized planes under
/// hq-e8-2b (codec decodes, with the residual window's rows exact). A leaf
/// that cannot read them says so, and the question fails; it never answers
/// from some other copy of the keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionQuery {
    /// The layer and query head read.
    pub head: PointingHead,
    /// The absolute prompt position of the first key read.
    pub key_begin: u32,
    /// How many consecutive keys are read — one score comes back for each.
    pub key_count: u32,
}

/// The fewest tokens the chunk carrying an [`AttentionQuery`] may hold.
///
/// The leaf reads the keys where the layer's attention materialized them,
/// and under hq-e8-2b only the prompt route materializes them: a chunk of 8
/// tokens or fewer takes the small-T route, which decodes keys in registers
/// and leaves nothing to read. The scheduler keeps an attention readout's
/// last chunk above that, and the leaf fails the question if it ever meets
/// one that is not — a readout is a failed question, never a wrong point.
pub const ATTENTION_MIN_CHUNK_TOKENS: u32 = 9;

/// The threshold of TAG's region rule, on the min-max normalized map.
pub const REGION_THRESHOLD: f64 = 0.5;

/// One head's map, read into a point by [`read_head_map`].
///
/// `x` and `y` are in **grid units**: the cell in row `r`, column `c` has its
/// centre at `(c + 0.5, r + 0.5)`, so a reading spans `[0, cols]` by
/// `[0, rows]` and maps to any image the grid covers by its own side on each
/// axis ([`HeadReading::pixels`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeadReading {
    /// The region's weighted centre along the grid's columns.
    pub x: f64,
    /// The region's weighted centre along the grid's rows.
    pub y: f64,
    /// The grid's rows (merged image tokens down the image).
    pub rows: usize,
    /// The grid's columns (merged image tokens across the image).
    pub cols: usize,
    /// How many cells the region holds — one on most real maps: the head's
    /// map is peaked enough that TAG's rule is an argmax with a little
    /// smoothing.
    pub cells: usize,
    /// The region's share of the softmax over the span, in `[0, 1]`.
    ///
    /// The confidence spec 13 exposes: it separates hits from misses on the
    /// synthetic sets (AUC 0.65-0.84), where the region's spread does not.
    /// Not a calibrated probability.
    pub share: f64,
}

impl HeadReading {
    /// The point in pixels of an image `width` by `height` that the grid
    /// covers: each axis scaled by its own side, which is the processor's
    /// own grid-to-pixel scale (it resizes the whole image onto the grid,
    /// neither cropping nor padding).
    pub fn pixels(&self, width: u32, height: u32) -> (f64, f64) {
        (
            self.x * f64::from(width) / self.cols as f64,
            self.y * f64::from(height) / self.rows as f64,
        )
    }

    /// One grid cell in pixels of that image, per axis: the map's
    /// resolution, which is what a head point's `uncertainty` is.
    pub fn cell_pixels(&self, width: u32, height: u32) -> (f64, f64) {
        (
            f64::from(width) / self.cols as f64,
            f64::from(height) / self.rows as f64,
        )
    }

    /// The point on a `0..=scale` axis each way — the question's own scale
    /// (0-999 at three digits), rounded to the nearest unit.
    pub fn normalized(&self, scale: u64) -> (u64, u64) {
        let on = |fraction: f64| (fraction * scale as f64).round().clamp(0.0, scale as f64) as u64;
        (on(self.x / self.cols as f64), on(self.y / self.rows as f64))
    }
}

/// Read one head's `scores` over an image span into a point: TAG's region
/// rule over a `rows` by `cols` grid, row-major.
///
/// The map is `m = exp(s - max s)` — the softmax over the span, up to one
/// positive factor — min-max normalized; the cells at or above
/// [`REGION_THRESHOLD`] form 4-connected regions, the one with the highest
/// mean wins (the first in raster order on a tie), and the answer is its
/// centre weighted by the normalized map. A flat map, which min-max cannot
/// normalize, reads as its first cell, as the rule every finding used does;
/// its share, one cell's part of a uniform softmax, says what that is worth.
///
/// `None` when `scores` is not the grid's size, is empty, or holds a
/// non-finite score: a map the leaf did not produce whole is not a map.
pub fn read_head_map(scores: &[f32], rows: usize, cols: usize) -> Option<HeadReading> {
    if scores.is_empty() || rows.checked_mul(cols)? != scores.len() {
        return None;
    }
    if scores.iter().any(|s| !s.is_finite()) {
        return None;
    }
    let top = scores.iter().fold(f64::NEG_INFINITY, |a, &s| a.max(f64::from(s)));
    let mass: Vec<f64> = scores.iter().map(|&s| (f64::from(s) - top).exp()).collect();
    let total: f64 = mass.iter().sum();
    let lo = mass.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = mass.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let reading = |cells: &[usize], weights: &dyn Fn(usize) -> f64| {
        let weight: f64 = cells.iter().map(|&k| weights(k)).sum();
        let (mut x, mut y) = (0.0, 0.0);
        for &k in cells {
            x += ((k % cols) as f64 + 0.5) * weights(k) / weight;
            y += ((k / cols) as f64 + 0.5) * weights(k) / weight;
        }
        HeadReading {
            x,
            y,
            rows,
            cols,
            cells: cells.len(),
            share: (cells.iter().map(|&k| mass[k]).sum::<f64>() / total).clamp(0.0, 1.0),
        }
    };
    if hi <= lo {
        return Some(reading(&[0], &|_| 1.0));
    }
    let normalized: Vec<f64> = mass.iter().map(|&m| (m - lo) / (hi - lo)).collect();
    let on: Vec<bool> = normalized.iter().map(|&v| v >= REGION_THRESHOLD).collect();
    let mut seen = vec![false; normalized.len()];
    let mut best: Option<(f64, Vec<usize>)> = None;
    for start in 0..normalized.len() {
        if !on[start] || seen[start] {
            continue;
        }
        seen[start] = true;
        let mut stack = vec![start];
        let mut region = Vec::new();
        while let Some(k) = stack.pop() {
            region.push(k);
            let (row, col) = (k / cols, k % cols);
            let neighbours = [
                (row + 1 < rows).then(|| k + cols),
                (row > 0).then(|| k - cols),
                (col + 1 < cols).then(|| k + 1),
                (col > 0).then(|| k - 1),
            ];
            for n in neighbours.into_iter().flatten() {
                if on[n] && !seen[n] {
                    seen[n] = true;
                    stack.push(n);
                }
            }
        }
        let mean = region.iter().map(|&k| normalized[k]).sum::<f64>() / region.len() as f64;
        // Strictly greater: the region met first in raster order keeps a tie.
        if best.as_ref().is_none_or(|(high, _)| mean > *high) {
            best = Some((mean, region));
        }
    }
    let (_, region) = best?;
    Some(reading(&region, &|k| normalized[k]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_served_artifact_is_calibrated_to_l39_h10() {
        let head = calibrated_head(ArtifactHash::from_bytes(SERVED_NVFP4_27B));
        assert_eq!(
            head,
            Some(PointingHead {
                gqa_ordinal: 9,
                query_head: 10
            })
        );
        assert_eq!(head.unwrap().to_string(), "L39.h10");
    }

    /// Keyed by the whole hash: an artifact that differs in its last byte —
    /// or one nobody measured — has no head, and `point` answers by chain.
    #[test]
    fn any_other_artifact_has_no_pointing_head() {
        let mut other = SERVED_NVFP4_27B;
        other[31] ^= 1;
        assert_eq!(calibrated_head(ArtifactHash::from_bytes(other)), None);
        assert_eq!(calibrated_head(ArtifactHash::UNKNOWN), None);
    }
}
