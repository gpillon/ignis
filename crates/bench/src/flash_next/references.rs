//! A converter reference set (`references/<set>/`, layout.md §10) read
//! whole: the windows, their fed tokens, and per stored position the BF16
//! and quantized streams' 64 most probable next tokens; plus the parts of
//! the converter's record (`work/converter.json`, or the sidecar it is
//! merged into) the acceptance limits come from.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use super::mmlu::Paired;

/// Entries per stored row (`scoring.TOP`).
pub const TOP: usize = 64;

/// One window of a set, as `manifest.json` lists it.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Window {
    pub index: usize,
    /// The domain the window is scored under.
    pub kind: String,
    pub source: String,
    /// Tokens stored for the window (EOS padding past `valid` included).
    pub length: usize,
    /// Positions stored for the window: `0..valid`.
    pub valid: usize,
    /// The window's first row in the set's per-position files.
    pub first_position: usize,
}

#[derive(Deserialize)]
struct Manifest {
    windows: Vec<Window>,
    positions: usize,
    top: usize,
}

/// One stored row of a stream: its 64 most probable next tokens,
/// descending (ties by the lower id), and their log-probabilities (nats).
#[derive(Debug, Clone, Copy)]
pub struct TopRow<'a> {
    pub ids: &'a [i32],
    pub lp: &'a [f32],
}

impl TopRow<'_> {
    /// The log-probability of `id` when it is among the row's 64.
    pub fn log_prob(&self, id: u32) -> Option<f32> {
        self.ids.iter().position(|&i| i as i64 == id as i64).map(|k| self.lp[k])
    }

    /// The row's most probable token.
    pub fn argmax(&self) -> u32 {
        self.ids[0] as u32
    }
}

/// A reference set in memory (about 140 MB for `test2048`).
#[derive(Debug, Clone)]
pub struct ReferenceSet {
    pub dir: PathBuf,
    pub windows: Vec<Window>,
    tokens: Vec<u32>,
    token_start: Vec<usize>,
    bf16_ids: Vec<i32>,
    bf16_lp: Vec<f32>,
    q_ids: Vec<i32>,
    q_lp: Vec<f32>,
    q_argmax: Vec<i32>,
    q_at_bf16: Option<Vec<f32>>,
}

/// The optional file with the quantized stream's log-probs at the BF16
/// stream's top-64 ids, which makes the converter's quantization-only
/// top-64 KLD reproducible from the set alone.
pub const Q_AT_BF16_FILE: &str = "q_lp_at_bf16_ids.f32";

impl ReferenceSet {
    /// Reads `dir` (`references/<set>`), refusing a set whose files do not
    /// have the sizes its manifest implies.
    pub fn read(dir: &Path) -> Result<Self, String> {
        let manifest_path = dir.join("manifest.json");
        let manifest: Manifest = serde_json::from_str(
            &std::fs::read_to_string(&manifest_path).map_err(|e| format!("read {}: {e}", manifest_path.display()))?,
        )
        .map_err(|e| format!("parse {}: {e}", manifest_path.display()))?;
        if manifest.top != TOP {
            return Err(format!("{}: top {} rows, this scorer reads {TOP}", manifest_path.display(), manifest.top));
        }
        let mut token_start = Vec::with_capacity(manifest.windows.len());
        let (mut tokens_total, mut positions) = (0usize, 0usize);
        for (i, w) in manifest.windows.iter().enumerate() {
            if w.index != i || w.first_position != positions || w.valid == 0 || w.valid > w.length {
                return Err(format!(
                    "{}: window {i} is out of order (index {}, first_position {} after {positions} positions, valid {} of {})",
                    manifest_path.display(),
                    w.index,
                    w.first_position,
                    w.valid,
                    w.length
                ));
            }
            token_start.push(tokens_total);
            tokens_total += w.length;
            positions += w.valid;
        }
        if positions != manifest.positions {
            return Err(format!(
                "{}: the windows hold {positions} positions, the manifest says {}",
                manifest_path.display(),
                manifest.positions
            ));
        }
        let p = positions;
        let rows = p * TOP;
        let set = ReferenceSet {
            dir: dir.to_path_buf(),
            tokens: read_words(dir, "tokens.u32", tokens_total, u32::from_le_bytes)?,
            token_start,
            bf16_ids: read_words(dir, "bf16_top64_ids.i32", rows, i32::from_le_bytes)?,
            bf16_lp: read_words(dir, "bf16_top64_lp.f32", rows, f32::from_le_bytes)?,
            q_ids: read_words(dir, "q_top64_ids.i32", rows, i32::from_le_bytes)?,
            q_lp: read_words(dir, "q_top64_lp.f32", rows, f32::from_le_bytes)?,
            q_argmax: read_words(dir, "q_argmax.i32", p, i32::from_le_bytes)?,
            q_at_bf16: if dir.join(Q_AT_BF16_FILE).exists() {
                Some(read_words(dir, Q_AT_BF16_FILE, rows, f32::from_le_bytes)?)
            } else {
                None
            },
            windows: manifest.windows,
        };
        for name in ["bf16_lse.f32", "q_lse.f32"] {
            check_size(dir, name, p)?;
        }
        if let Some(bad) = set.bf16_ids.iter().chain(&set.q_ids).chain(&set.q_argmax).find(|&&id| id < 0) {
            return Err(format!("{}: a stored token id is negative ({bad})", dir.display()));
        }
        Ok(set)
    }

    /// Stored positions (`P`).
    pub fn positions(&self) -> usize {
        self.q_argmax.len()
    }

    /// The window's stored tokens (`length` of them).
    pub fn tokens(&self, window: &Window) -> &[u32] {
        let start = self.token_start[window.index];
        &self.tokens[start..start + window.length]
    }

    /// The BF16 stream's row `row` of the set.
    pub fn bf16(&self, row: usize) -> TopRow<'_> {
        TopRow { ids: &self.bf16_ids[row * TOP..(row + 1) * TOP], lp: &self.bf16_lp[row * TOP..(row + 1) * TOP] }
    }

    /// The quantized stream's row `row` of the set.
    pub fn quantized(&self, row: usize) -> TopRow<'_> {
        TopRow { ids: &self.q_ids[row * TOP..(row + 1) * TOP], lp: &self.q_lp[row * TOP..(row + 1) * TOP] }
    }

    /// The quantized stream's argmax at row `row` (ties by the lower id).
    pub fn quantized_argmax(&self, row: usize) -> u32 {
        self.q_argmax[row] as u32
    }

    /// The quantized stream's log-probs at the BF16 row's ids, when the set
    /// stores them ([`Q_AT_BF16_FILE`]).
    pub fn quantized_at_bf16(&self, row: usize) -> Option<&[f32]> {
        self.q_at_bf16.as_deref().map(|v| &v[row * TOP..(row + 1) * TOP])
    }
}

fn check_size(dir: &Path, name: &str, words: usize) -> Result<(), String> {
    let path = dir.join(name);
    let meta = std::fs::metadata(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if meta.len() != (words * 4) as u64 {
        return Err(format!("{}: {} bytes, the manifest implies {}", path.display(), meta.len(), words * 4));
    }
    Ok(())
}

/// Reads a file of `words` little-endian 4-byte values.
fn read_words<T>(dir: &Path, name: &str, words: usize, decode: fn([u8; 4]) -> T) -> Result<Vec<T>, String> {
    check_size(dir, name, words)?;
    let path = dir.join(name);
    let bytes = std::fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok(bytes.as_chunks::<4>().0.iter().map(|&b| decode(b)).collect())
}

/// A domain's figures in the converter's record.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DomainFigures {
    /// The exact KL over the full vocabulary.
    pub kld: f64,
    /// The top-64 scorer's figure: what an engine's figure is compared with.
    pub kld_top64: f64,
    pub top1: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct KldSection {
    pub quantized: BTreeMap<String, DomainFigures>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LongKldSection {
    pub q: BTreeMap<String, DomainFigures>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MmluFigures {
    pub n: usize,
    pub bf16: Option<f64>,
    pub quantized: Option<f64>,
    pub mcnemar: BTreeMap<String, Paired>,
}

/// The converter's record (layout.md §9), the parts the acceptance limits
/// read: `work/converter.json`, or the sidecar its fields are merged into.
#[derive(Debug, Clone, Deserialize)]
pub struct ConverterRecord {
    pub status: String,
    /// The 2048-token test windows, per stream and domain.
    pub kld: KldSection,
    /// The 8192-token windows (the quantized stream only).
    pub kld_long8192: LongKldSection,
    pub mmlu: MmluFigures,
}

impl ConverterRecord {
    pub fn read(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))
    }
}
