//! Pointing in one pass: the **attention readout**, the **pointing head** and
//! the **head set** (specs `docs/specs/decide/13-point-by-attention-head.md`
//! and `14-point-and-box-from-the-head-set.md`, GitHub #260 and #263, ADR 0038
//! and ADR 0039).
//!
//! One attention head of the served model, read at the position after the
//! forced `{"x":`, names the object the question asked for; 96 heads of the
//! layers after it each mark a part of that object, and the extent of the
//! parts near the first head's point is the object's box. Four things live
//! here, and they are the host's whole part of it:
//!
//! - **What a job asks the leaf to read** ([`AttentionQuery`]): one query
//!   head of one GQA layer, dotted with the keys of one span of the prompt,
//!   and optionally a head set read beside it. What comes back across the
//!   `Compute` seam ([`AttentionScores`]) is one pre-softmax score per key of
//!   that span for the pointing head, one key index per head of the set, and
//!   nothing else — never a full attention row, never a logits row.
//! - **Which heads point** ([`calibration`]): constants chosen with labelled
//!   scenes, keyed to the artifact's content hash. Unlike the answer
//!   alphabet they cannot be computed at load, and like the alphabet they
//!   are refused against any other load.
//! - **How a map becomes a point** ([`read_head_map`]): TAG's region rule,
//!   the one every number in the findings was measured with.
//! - **How a map and a head set become a box** ([`read_anchored`]): spec
//!   14's anchored reading, ported from `tools/pointing-scenes/ensemble_score.py`.

use std::sync::Arc;

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

/// A **head set** (spec 14, ADR 0039): the heads read together beside the
/// pointing head, one cell each, and the **fallback cells** their argmax
/// skips.
///
/// Every head of a set reads a different *part* of the object the question
/// asked for (`docs/findings/2026-09-22-the-heads-outline-the-object.md`);
/// the anchored reading ([`read_anchored`]) keeps the cells near the pointing
/// head's point and spans them into the object's extent.
#[derive(Debug, PartialEq, Eq)]
pub struct HeadSet {
    /// The heads, in the set's order — the order the leaf's argmax indices
    /// come back in.
    pub heads: &'static [PointingHead],
    /// The fallback cells measured on particular grids, beyond the first and
    /// the last image cell every grid excludes.
    pub fallback: &'static [GridCells],
}

/// Cells of one image grid, row-major.
#[derive(Debug, PartialEq, Eq)]
pub struct GridCells {
    /// The grid's rows.
    pub rows: u32,
    /// The grid's columns.
    pub cols: u32,
    /// The cells, as row-major indices into the grid.
    pub cells: &'static [u32],
}

impl HeadSet {
    /// The **fallback cells** of a `rows` by `cols` grid, ascending: the
    /// image cells heads peak on when nothing matches. Always the first and
    /// the last, and on a grid the calibration measured, the cells it found
    /// there too.
    ///
    /// Spec 14 § Fallback cells: excluding only the first and the last on
    /// the 32x32 grid costs one large object in 60; the extra cells are
    /// measured, not guessed, which is why they are listed per grid.
    pub fn fallback_cells(&self, rows: u32, cols: u32) -> Vec<u32> {
        let last = (rows * cols).saturating_sub(1);
        let mut cells = vec![0, last];
        if let Some(measured) = self.fallback.iter().find(|grid| (grid.rows, grid.cols) == (rows, cols)) {
            cells.extend(measured.cells.iter().copied().filter(|&cell| cell <= last));
        }
        cells.sort_unstable();
        cells.dedup();
        cells
    }
}

/// A head at backbone layer `layer`, query head `head` — the findings'
/// `L<layer>.h<head>`, which is how the head set was recorded. A layer that
/// is not a GQA layer (`4k + 3`) fails the build rather than rounding to a
/// neighbour.
const fn layer_head(layer: u32, head: u32) -> PointingHead {
    assert!(layer % 4 == 3, "the 27B's GQA layers are 3, 7, ..., 63");
    PointingHead {
        gqa_ordinal: (layer - 3) / 4,
        query_head: head,
    }
}

/// The served NVFP4 27B's head set: 96 heads of GQA layers 31 to 63, chosen
/// by `tools/pointing-scenes/ensemble_score.py select` (spec 14's selection
/// rule: layer 31 or deeper, at least 0.3 of the softmax mass over the boxes
/// of distractor scenes and at least 0.8 of that on the asked box) and
/// recorded in `tools/pointing-scenes/pointing-heads-4bdc7b13.json`.
const SERVED_NVFP4_27B_SET: HeadSet = HeadSet {
    heads: &[
        layer_head(31, 1),
        layer_head(31, 10),
        layer_head(31, 12),
        layer_head(31, 17),
        layer_head(31, 23),
        layer_head(35, 2),
        layer_head(35, 4),
        layer_head(35, 6),
        layer_head(35, 8),
        layer_head(35, 16),
        layer_head(35, 17),
        layer_head(35, 18),
        layer_head(35, 22),
        layer_head(35, 23),
        layer_head(39, 0),
        layer_head(39, 2),
        layer_head(39, 7),
        layer_head(39, 8),
        layer_head(39, 10),
        layer_head(39, 11),
        layer_head(39, 12),
        layer_head(39, 15),
        layer_head(39, 16),
        layer_head(39, 17),
        layer_head(39, 22),
        layer_head(39, 23),
        layer_head(43, 1),
        layer_head(43, 3),
        layer_head(43, 5),
        layer_head(43, 6),
        layer_head(43, 7),
        layer_head(43, 9),
        layer_head(43, 14),
        layer_head(43, 15),
        layer_head(43, 17),
        layer_head(43, 18),
        layer_head(43, 19),
        layer_head(43, 20),
        layer_head(43, 22),
        layer_head(43, 23),
        layer_head(47, 0),
        layer_head(47, 1),
        layer_head(47, 2),
        layer_head(47, 3),
        layer_head(47, 4),
        layer_head(47, 5),
        layer_head(47, 9),
        layer_head(47, 10),
        layer_head(47, 13),
        layer_head(47, 14),
        layer_head(47, 16),
        layer_head(47, 17),
        layer_head(47, 18),
        layer_head(47, 20),
        layer_head(47, 21),
        layer_head(47, 23),
        layer_head(51, 0),
        layer_head(51, 2),
        layer_head(51, 4),
        layer_head(51, 6),
        layer_head(51, 12),
        layer_head(51, 13),
        layer_head(51, 14),
        layer_head(51, 15),
        layer_head(51, 16),
        layer_head(51, 17),
        layer_head(51, 18),
        layer_head(51, 19),
        layer_head(51, 20),
        layer_head(51, 21),
        layer_head(51, 22),
        layer_head(51, 23),
        layer_head(55, 0),
        layer_head(55, 2),
        layer_head(55, 4),
        layer_head(55, 6),
        layer_head(55, 7),
        layer_head(55, 8),
        layer_head(55, 9),
        layer_head(55, 10),
        layer_head(55, 11),
        layer_head(55, 13),
        layer_head(55, 16),
        layer_head(55, 18),
        layer_head(55, 20),
        layer_head(55, 22),
        layer_head(55, 23),
        layer_head(59, 0),
        layer_head(59, 1),
        layer_head(59, 2),
        layer_head(59, 3),
        layer_head(59, 4),
        layer_head(59, 5),
        layer_head(59, 8),
        layer_head(59, 14),
        layer_head(63, 18),
    ],
    // (0,0), (0,1), (6,31) and (31,31): measured on the 32x32 grid's blank
    // priors (spec 14 § Calibration).
    fallback: &[GridCells {
        rows: 32,
        cols: 32,
        cells: &[0, 1, 223, 1023],
    }],
};

/// What one artifact is calibrated with: the pointing head, and the head set
/// read beside it when one was chosen (spec 14).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Calibration {
    /// The pointing head: which object, and the answer's confidence.
    pub head: PointingHead,
    /// The head set, or `None` on a load calibrated with a pointing head
    /// only — spec 13's reading, unchanged.
    pub set: Option<&'static HeadSet>,
}

/// The calibration table: artifact content hash → the heads that point.
///
/// One entry today. L39.h10 was chosen on the served artifact by
/// cross-validation over labelled synthetic scenes (every fold of every arm
/// picked it) and checked on sets it was never chosen on
/// (`docs/findings/2026-09-21-one-attention-head-points.md`,
/// `2026-09-21-the-head-points-in-the-engine.md`); its head set by spec 14's
/// selection rule (`docs/findings/2026-09-23-an-anchored-head-set-points-and-boxes.md`).
/// Recalibrating for a new artifact is the procedure in
/// `tools/pointing-scenes/README.md`, run with
/// `crates/server/tests/attention_head_point_gpu.rs`.
const CALIBRATED: &[([u8; 32], Calibration)] = &[(
    SERVED_NVFP4_27B,
    Calibration {
        head: PointingHead {
            gqa_ordinal: 9,
            query_head: 10,
        },
        set: Some(&SERVED_NVFP4_27B_SET),
    },
)];

/// The calibration recorded for `artifact`, or `None` when nobody
/// calibrated it — and then `point` answers with the digit chain.
///
/// Looked up by the full 32-byte hash: heads chosen for one model are never
/// read on another, however alike their names.
pub fn calibration(artifact: ArtifactHash) -> Option<Calibration> {
    CALIBRATED
        .iter()
        .find(|(hash, _)| hash == artifact.as_bytes())
        .map(|&(_, calibration)| calibration)
}

/// The pointing head calibrated for `artifact`, or `None` when nobody
/// calibrated one for it.
pub fn calibrated_head(artifact: ArtifactHash) -> Option<PointingHead> {
    calibration(artifact).map(|calibration| calibration.head)
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
///
/// With a [`SetQuery`] (spec 14, ADR 0039) the job also names a head set:
/// every GQA layer holding one of its heads is read the same way, and one
/// argmax key index per head comes back beside the pointing head's scores
/// ([`AttentionScores`]). A read any armed layer cannot make leaves the
/// whole readout unread — never a partial set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttentionQuery {
    /// The layer and query head read.
    pub head: PointingHead,
    /// The absolute prompt position of the first key read.
    pub key_begin: u32,
    /// How many consecutive keys are read — one score comes back for each.
    pub key_count: u32,
    /// The head set read beside the pointing head, or `None` for the
    /// pointing head alone — which then costs what it cost before spec 14:
    /// one armed layer.
    pub set: Option<SetQuery>,
}

/// The head set half of an [`AttentionQuery`] (spec 14, ADR 0039).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetQuery {
    /// The heads, at most [`MAX_SET_HEADS`]; one key index per head comes
    /// back, in this order.
    pub heads: Arc<[PointingHead]>,
    /// Span-relative key indices no head's argmax may land on — the
    /// **fallback cells** ([`HeadSet::fallback_cells`]), at most
    /// [`MAX_EXCLUDED_KEYS`]. The pointing head's own scores still cover
    /// every key: its region rule reads the whole map.
    pub excluded: Arc<[u32]>,
}

impl SetQuery {
    /// The query for `set` over a `rows` by `cols` image grid.
    pub fn for_grid(set: &HeadSet, rows: u32, cols: u32) -> Self {
        Self {
            heads: Arc::from(set.heads),
            excluded: Arc::from(set.fallback_cells(rows, cols)),
        }
    }
}

/// The most heads a [`SetQuery`] may name: every query head of every GQA
/// layer (16 x 24). What the leaf reserves its per-head results for.
pub const MAX_SET_HEADS: usize = 16 * 24;

/// The most keys a [`SetQuery`] may exclude. The leaf carries them as
/// launch arguments, not memory; the served calibration excludes four.
pub const MAX_EXCLUDED_KEYS: usize = 32;

/// What an [`AttentionQuery`] reads back across the `Compute` seam (ADR 0038,
/// ADR 0039): the pointing head's scores, and one key index per head of the
/// set when it named one. Nothing else crosses — at 4096 px that is 64 KB
/// and 384 bytes, where every head's row would be 6.3 MB.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionScores {
    /// The pointing head's `q · k / sqrt(head_dim)`, one per key of the
    /// span, in order, before any softmax.
    pub scores: Arc<[f32]>,
    /// One span-relative key index per head of the query's set, in the
    /// set's order: where that head's score peaks over the span minus the
    /// excluded keys (the larger index on a tie). `None` when the query
    /// named no set.
    pub set_argmax: Option<Arc<[u32]>>,
}

impl AttentionScores {
    /// The pointing head's scores alone, as a query with no set reads them.
    pub fn pointing(scores: impl Into<Arc<[f32]>>) -> Self {
        Self {
            scores: scores.into(),
            set_argmax: None,
        }
    }
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

/// How far from the pointing head's point a head set's cell is kept: this
/// many times the median distance of the set's cells (spec 14).
pub const KEEP_RADIUS: f64 = 2.0;

/// The extent is the [`EXTENT_QUANTILE`] to `1 - EXTENT_QUANTILE` span of
/// the kept cells on each axis, grown half a cell (spec 14).
pub const EXTENT_QUANTILE: f64 = 0.1;

/// A box in pixels of the submitted image: `(x0, y0)` its top-left corner,
/// `(x1, y1)` its bottom-right.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Extent {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

impl Extent {
    /// The extent's centre: the anchored reading's point.
    pub fn centre(&self) -> (f64, f64) {
        ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)
    }
}

/// The **anchored reading** of a head set (spec 14): what [`read_anchored`]
/// returns.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnchoredReading {
    /// The pointing head's own reading — the anchor that names which
    /// object, and whose `cells` and `share` stay the answer's confidence.
    pub anchor: HeadReading,
    /// The **extent**: the object's box, clamped to the image.
    pub extent: Extent,
    /// How many of the set's cells the extent was spanned over.
    pub kept: usize,
}

impl AnchoredReading {
    /// The point: the extent's centre.
    pub fn point(&self) -> (f64, f64) {
        self.extent.centre()
    }
}

/// Read a head set anchored on the pointing head into a box and its centre
/// (spec 14): `anchor_scores` is the pointing head's map over a `rows` by
/// `cols` grid, row-major, `set_argmax` one cell per head of the set, and
/// the image `width` by `height` pixels the grid covers.
///
/// The rule every number in spec 14 was measured with,
/// `tools/pointing-scenes/ensemble_score.py read_anchored`, in the same
/// arithmetic:
///
/// 1. the anchor is [`read_head_map`]'s point, in pixels;
/// 2. each head's cell centre is kept when its distance to the anchor is at
///    most [`KEEP_RADIUS`] times the larger of the cells' median distance
///    and one cell's longer side — so at least half of them always are;
/// 3. the extent is the [`EXTENT_QUANTILE`] and `1 - EXTENT_QUANTILE`
///    quantiles of the kept centres on each axis (NumPy's default linear
///    interpolation at position `(n - 1) * q`), grown half a cell and
///    clamped to the image; the point is its centre.
///
/// `None` when the anchor's map does not read ([`read_head_map`]), the set
/// is empty, or a cell is outside the grid: a set the leaf did not read
/// whole is not a set.
pub fn read_anchored(
    anchor_scores: &[f32],
    set_argmax: &[u32],
    rows: usize,
    cols: usize,
    width: u32,
    height: u32,
) -> Option<AnchoredReading> {
    let anchor = read_head_map(anchor_scores, rows, cols)?;
    if set_argmax.is_empty() || set_argmax.iter().any(|&cell| cell as usize >= rows * cols) {
        return None;
    }
    let (width, height) = (f64::from(width), f64::from(height));
    let (cw, ch) = (width / cols as f64, height / rows as f64);
    let (ax, ay) = (anchor.x * cw, anchor.y * ch);
    let cells: Vec<(f64, f64)> = set_argmax
        .iter()
        .map(|&cell| {
            let cell = cell as usize;
            (((cell % cols) as f64 + 0.5) * cw, ((cell / cols) as f64 + 0.5) * ch)
        })
        .collect();
    let distances: Vec<f64> = cells.iter().map(|&(x, y)| (x - ax).hypot(y - ay)).collect();
    let scale = median(&distances).max(cw.max(ch));
    let (mut xs, mut ys): (Vec<f64>, Vec<f64>) = cells
        .iter()
        .zip(&distances)
        .filter(|&(_, &d)| d <= KEEP_RADIUS * scale)
        .map(|(&cell, _)| cell)
        .unzip();
    xs.sort_by(f64::total_cmp);
    ys.sort_by(f64::total_cmp);
    let extent = Extent {
        x0: (quantile(&xs, EXTENT_QUANTILE) - cw / 2.0).max(0.0),
        y0: (quantile(&ys, EXTENT_QUANTILE) - ch / 2.0).max(0.0),
        x1: (quantile(&xs, 1.0 - EXTENT_QUANTILE) + cw / 2.0).min(width),
        y1: (quantile(&ys, 1.0 - EXTENT_QUANTILE) + ch / 2.0).min(height),
    };
    Some(AnchoredReading {
        anchor,
        extent,
        kept: xs.len(),
    })
}

/// `numpy.median`: the middle value, or the mean of the two middle values.
fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    match n % 2 {
        1 => sorted[n / 2],
        _ => (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0,
    }
}

/// `numpy.quantile(sorted, q)` with its default linear method: the virtual
/// index `(n - 1) * q`, interpolated between its two order statistics the
/// way NumPy's `_lerp` does — from below under half the gap, from above at
/// or past it — so the port agrees with the reference to the last bit.
fn quantile(sorted: &[f64], q: f64) -> f64 {
    let position = (sorted.len() - 1) as f64 * q;
    let below = position.floor() as usize;
    let above = (below + 1).min(sorted.len() - 1);
    let t = position - below as f64;
    let (a, b) = (sorted[below], sorted[above]);
    let gap = b - a;
    match t >= 0.5 {
        true => b - gap * (1.0 - t),
        false => a + gap * t,
    }
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
        assert_eq!(calibration(ArtifactHash::from_bytes(other)), None);
    }

    /// Spec 14 § Calibration: the served artifact's set is the table the
    /// spec records — 96 heads over GQA layers 31 to 63, per layer as
    /// listed, the pointing head among them.
    #[test]
    fn the_served_artifact_carries_the_recorded_head_set() {
        let calibration = calibration(ArtifactHash::from_bytes(SERVED_NVFP4_27B)).expect("calibrated");
        let set = calibration.set.expect("the served artifact has a head set");
        assert_eq!(set.heads.len(), 96);
        assert!(set.heads.len() <= MAX_SET_HEADS);
        assert!(set.heads.contains(&calibration.head), "the pointing head is one of the set");
        let per_layer: Vec<(u32, usize)> = (7..16)
            .map(|ordinal| (4 * ordinal + 3, set.heads.iter().filter(|h| h.gqa_ordinal == ordinal).count()))
            .collect();
        assert_eq!(
            per_layer,
            [(31, 5), (35, 9), (39, 12), (43, 14), (47, 16), (51, 16), (55, 15), (59, 8), (63, 1)]
        );
        assert!(set.heads.iter().all(|h| h.gqa_ordinal >= 7 && h.query_head < 24), "layer 31 or deeper");
        let mut unique = set.heads.to_vec();
        unique.sort_by_key(|h| (h.gqa_ordinal, h.query_head));
        unique.dedup();
        assert_eq!(unique.len(), set.heads.len(), "no head twice");
    }

    /// The fallback cells: the first and the last on every grid, and the
    /// measured ones only on the grid they were measured on.
    #[test]
    fn fallback_cells_are_per_grid() {
        let set = &SERVED_NVFP4_27B_SET;
        assert_eq!(set.fallback_cells(32, 32), [0, 1, 223, 1023]);
        assert_eq!(set.fallback_cells(15, 27), [0, 404]);
        assert_eq!(set.fallback_cells(32, 31), [0, 991], "a grid of the same area is another grid");
        assert_eq!(set.fallback_cells(1, 1), [0], "one cell is both the first and the last");
        // What the leaf is handed: the measured cells with the first and the
        // last added — past MAX_EXCLUDED_KEYS the whole prefill is refused.
        for grid in set.fallback {
            assert!(set.fallback_cells(grid.rows, grid.cols).len() <= MAX_EXCLUDED_KEYS);
        }
        let query = SetQuery::for_grid(set, 32, 32);
        assert_eq!(query.heads.len(), 96);
        assert_eq!(&*query.excluded, &[0, 1, 223, 1023]);
    }
}
