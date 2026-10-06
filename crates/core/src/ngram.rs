//! Flash-Next's n-gram embedding, host side (spec flash-next/04, GitHub #302).
//!
//! Each token reads `heads()` rows of the n-gram table: its 2-gram hashed by
//! the first `heads_per_ngram` heads, its 3-gram by the next, every head into
//! its own prime-sized range of the table. [`NgramHasher`] computes those row
//! ids bit for bit as the checkpoint's modeling code does
//! (`Qwen4ExpTextNGramEmbedding`, transformers), from the hash buffers the
//! checkpoint stores; `crates/core/tests/ngram_hashing.rs` holds it to ids
//! recorded from that code.
//!
//! The table itself (320M rows, INT4) never reaches the device or RAM whole:
//! [`plan_gather`] says, for a step's row ids, which come from the RAM
//! hot-row cache ([`HotRows`]) and which aligned reads of the artifact's table
//! range fetch the rest.

use crate::compute::NgramGeometry;

/// The hash buffers the checkpoint stores beside its n-gram table
/// (`layer_multipliers`, `ngram_heads_vocab_sizes`, `ngram_heads_offsets`),
/// as `i64` like the checkpoint's own tensors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NgramHashBuffers {
    /// One odd multiplier per position of the longest n-gram, the current
    /// token's first.
    pub layer_multipliers: Vec<i64>,
    /// Each head's table range size (a prime).
    pub head_vocab_sizes: Vec<i64>,
    /// Each head's first row in the table.
    pub head_offsets: Vec<i64>,
}

const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX_M1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_M2: u64 = 0x94D0_49BB_1331_11EB;
const PRIME_1: u64 = 10_007;

fn splitmix64(value: u64) -> u64 {
    let mut value = value.wrapping_add(SPLITMIX_GAMMA);
    value = (value ^ (value >> 30)).wrapping_mul(SPLITMIX_M1);
    value = (value ^ (value >> 27)).wrapping_mul(SPLITMIX_M2);
    value ^ (value >> 31)
}

fn is_prime(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    if value % 2 == 0 {
        return value == 2;
    }
    let mut divisor = 3;
    while divisor * divisor <= value {
        if value % divisor == 0 {
            return false;
        }
        divisor += 2;
    }
    true
}

fn next_prime_after(value: u64) -> u64 {
    let mut prime = value + 1;
    while !is_prime(prime) {
        prime += 1;
    }
    prime
}

impl NgramHashBuffers {
    /// The buffers the modeling code builds from the config
    /// (`_build_layer_multipliers`, `_find_nth_prime_after`) for the first
    /// (and Flash-Next's only) PLE layer. The engine hashes with the stored
    /// buffers; this is the cross-check that the stored ones are what the
    /// checkpoint means.
    pub fn derive(geometry: &NgramGeometry, vocab: u64) -> Self {
        let ple_layer_index = 0u64;
        let multiplier_max = i64::MAX as u64 / vocab.max(1);
        let half_bound = (multiplier_max / 2).max(1);
        let base_seed = geometry.seed.wrapping_add(PRIME_1.wrapping_mul(ple_layer_index));
        let layer_multipliers = (0..geometry.ngram_size)
            .map(|index| {
                let value = base_seed.wrapping_add(SPLITMIX_GAMMA.wrapping_mul(index + 1));
                (2 * (splitmix64(value) % half_bound) + 1) as i64
            })
            .collect();
        // Head `h` (of this PLE layer's `heads()`) takes the
        // `(ple_layer_index * heads + h + 1)`-th prime after `base - 1`.
        let heads = geometry.heads();
        let mut prime = geometry.vocab_size_base - 1;
        for _ in 0..ple_layer_index * heads {
            prime = next_prime_after(prime);
        }
        let mut head_vocab_sizes = Vec::with_capacity(heads as usize);
        let mut head_offsets = Vec::with_capacity(heads as usize);
        let mut total = 0i64;
        for _ in 0..heads {
            prime = next_prime_after(prime);
            head_vocab_sizes.push(prime as i64);
            head_offsets.push(total);
            total += prime as i64;
        }
        Self { layer_multipliers, head_vocab_sizes, head_offsets }
    }

    /// The table's rows: every head's range.
    pub fn table_rows(&self) -> u64 {
        self.head_vocab_sizes.iter().map(|&size| size as u64).sum()
    }
}

/// One sequence's hashing state: its last `ngram_size - 1` tokens, EOS
/// before its first token (the checkpoint's empty context).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NgramContext {
    recent: Vec<u32>,
}

impl NgramContext {
    /// A sequence that has hashed no token yet.
    pub fn new(hasher: &NgramHasher) -> Self {
        Self { recent: vec![hasher.eos; hasher.geometry.context_tokens() as usize] }
    }

    /// A context holding `recent`, oldest first: one read back from a
    /// snapshot blob (spec flash-next/05). Refused unless it holds exactly
    /// the tokens [`NgramContext::new`] would.
    pub fn from_recent(hasher: &NgramHasher, recent: &[u32]) -> Result<Self, String> {
        let tokens = hasher.geometry.context_tokens() as usize;
        if recent.len() != tokens {
            return Err(format!("an n-gram context of {} tokens, not {tokens}", recent.len()));
        }
        Ok(Self { recent: recent.to_vec() })
    }

    /// The tokens it holds, oldest first: the last ones hashed, EOS where
    /// there were none.
    pub fn recent(&self) -> &[u32] {
        &self.recent
    }
}

/// Hashes tokens to their n-gram table rows, bit-exact with the checkpoint.
#[derive(Debug, Clone)]
pub struct NgramHasher {
    geometry: NgramGeometry,
    buffers: NgramHashBuffers,
    eos: u32,
}

impl NgramHasher {
    /// A hasher for `geometry` with the checkpoint's stored `buffers`.
    /// Refuses buffers whose lengths do not fit the geometry, or a head
    /// range that is not positive.
    pub fn new(geometry: NgramGeometry, buffers: NgramHashBuffers) -> Result<Self, String> {
        if geometry.ngram_size < 2 {
            return Err(format!("n-gram size {} hashes no n-gram", geometry.ngram_size));
        }
        if buffers.layer_multipliers.len() as u64 != geometry.ngram_size {
            return Err(format!(
                "{} layer multipliers for n-grams of {} tokens",
                buffers.layer_multipliers.len(),
                geometry.ngram_size
            ));
        }
        let heads = geometry.heads() as usize;
        if buffers.head_vocab_sizes.len() != heads || buffers.head_offsets.len() != heads {
            return Err(format!(
                "{} head sizes and {} head offsets for {heads} heads",
                buffers.head_vocab_sizes.len(),
                buffers.head_offsets.len()
            ));
        }
        if let Some(size) = buffers.head_vocab_sizes.iter().find(|&&size| size <= 0) {
            return Err(format!("a head's table range is {size} rows"));
        }
        if let Some(offset) = buffers.head_offsets.iter().find(|&&offset| offset < 0) {
            return Err(format!("a head's table range starts at row {offset}"));
        }
        Ok(Self { geometry, buffers, eos: geometry.eos_token_id })
    }

    /// Row ids per token.
    pub fn heads(&self) -> usize {
        self.geometry.heads() as usize
    }

    /// The table's rows rounded up to the checkpoint's divisor: the stored
    /// table's row count.
    pub fn padded_table_rows(&self) -> u64 {
        self.buffers.table_rows().div_ceil(self.geometry.vocab_divisor) * self.geometry.vocab_divisor
    }

    /// Appends `heads()` row ids per token of `tokens` to `out` and moves
    /// `context` past them. `tokens` continue the sequence `context` holds.
    ///
    /// A token's n-gram reads back through its predecessors until an EOS: an
    /// EOS ends a segment, so a position past one reads EOS instead (the
    /// checkpoint's `_shift_right_ignore_eos`). The arithmetic is the
    /// checkpoint's `int64` one: wrapping products, XOR, floor remainder.
    pub fn hash(&self, context: &mut NgramContext, tokens: &[u32], out: &mut Vec<u64>) {
        let n = self.geometry.ngram_size as usize;
        let per_ngram = self.geometry.heads_per_ngram as usize;
        let multipliers = &self.buffers.layer_multipliers;
        let mut history: Vec<u32> = Vec::with_capacity(context.recent.len() + tokens.len());
        history.extend_from_slice(&context.recent);
        history.extend_from_slice(tokens);
        let first = context.recent.len();
        let mut shifted = vec![0i64; n];
        out.reserve(tokens.len() * self.heads());
        for position in first..history.len() {
            shifted[0] = i64::from(history[position]);
            let mut segment_ended = false;
            for back in 1..n {
                let previous = history[position - back];
                shifted[back] = i64::from(if segment_ended { self.eos } else { previous });
                segment_ended |= previous == self.eos;
            }
            let mut mixed = shifted[0].wrapping_mul(multipliers[0]);
            for size in 2..=n {
                mixed ^= shifted[size - 1].wrapping_mul(multipliers[size - 1]);
                let heads = (size - 2) * per_ngram..(size - 1) * per_ngram;
                for head in heads {
                    let row = mixed.rem_euclid(self.buffers.head_vocab_sizes[head])
                        + self.buffers.head_offsets[head];
                    out.push(row as u64);
                }
            }
        }
        let kept = history.len() - context.recent.len();
        context.recent.copy_from_slice(&history[kept..]);
    }
}

/// Where the n-gram table's rows are in the artifact file: row `r` is
/// `row_bytes` at `base_offset + r * row_stride` (spec flash-next/01: rows are
/// directly addressable, no index).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TableLayout {
    pub base_offset: u64,
    pub row_stride: u64,
    pub row_bytes: u64,
    pub rows: u64,
}

impl TableLayout {
    fn row_offset(&self, row: u64) -> u64 {
        self.base_offset + row * self.row_stride
    }
}

/// The bytes of the cache's index per hot row: its row id.
const HOT_ROW_INDEX_BYTES: u64 = std::mem::size_of::<u32>() as u64;

/// The RAM hot-row cache: the head of the artifact's hot list (rows ranked
/// most frequent first) that fits a byte budget, its index included. Slot
/// `s` holds row `rows()[s]`, in ascending row order, so a row's slot is a
/// binary search and the index costs 4 bytes a row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HotRows {
    rows: Vec<u32>,
    row_bytes: u64,
}

impl HotRows {
    /// The first distinct rows of `ranked` whose bytes and index entries fit
    /// `budget_bytes` whole.
    pub fn from_ranked(ranked: &[u64], budget_bytes: u64, row_bytes: u64) -> Result<Self, String> {
        if row_bytes == 0 {
            return Err("a hot-row cache of 0-byte rows".to_string());
        }
        let capacity = (budget_bytes / (row_bytes + HOT_ROW_INDEX_BYTES)) as usize;
        let mut seen = std::collections::HashSet::with_capacity(capacity.min(ranked.len()));
        let mut rows = Vec::with_capacity(capacity.min(ranked.len()));
        for &row in ranked {
            if rows.len() == capacity {
                break;
            }
            let row = u32::try_from(row).map_err(|_| format!("hot row {row} is past a 32-bit row id"))?;
            if seen.insert(row) {
                rows.push(row);
            }
        }
        rows.sort_unstable();
        Ok(Self { rows, row_bytes })
    }

    /// Rows held.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the cache holds no row.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// The RAM the cache occupies: its rows and its index.
    pub fn bytes(&self) -> u64 {
        self.rows.len() as u64 * (self.row_bytes + HOT_ROW_INDEX_BYTES)
    }

    /// The rows held, in slot order: what the cache loads at start.
    pub fn rows(&self) -> &[u32] {
        &self.rows
    }

    /// The slot holding `row`, if the cache holds it.
    pub fn slot(&self, row: u64) -> Option<usize> {
        let row = u32::try_from(row).ok()?;
        self.rows.binary_search(&row).ok()
    }
}

/// How the table's file range is read: unbuffered reads start on and span
/// whole `alignment`-byte sectors, and none is longer than `max_read_bytes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadPolicy {
    pub alignment: u64,
    pub max_read_bytes: u64,
}

/// One aligned read of the artifact file. Near the end of the file it may
/// come back short; every row it was planned for still lies inside what it
/// returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlignedRead {
    pub offset: u64,
    pub len: u64,
}

/// Where one requested row's bytes come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowSource {
    /// The hot-row cache's slot.
    Hot { slot: usize },
    /// `offset` bytes into the plan's read `read`.
    Read { read: usize, offset: usize },
}

/// A step's rows: one [`RowSource`] per requested row, in request order, and
/// the reads that fetch every row the cache does not hold, each row once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatherPlan {
    pub reads: Vec<AlignedRead>,
    pub sources: Vec<RowSource>,
    /// The width of every row the plan gathers: the table's, and the cache's.
    pub row_bytes: usize,
}

impl GatherPlan {
    /// The bytes the plan reads from the file.
    pub fn read_bytes(&self) -> u64 {
        self.reads.iter().map(|read| read.len).sum()
    }

    /// Copies every requested row, in request order, into `out`
    /// ([`GatherPlan::row_bytes`] each) from the cache's bytes (`hot`, slot
    /// order) and each read's returned bytes (`reads`, plan order).
    pub fn gather(&self, hot: &[u8], reads: &[&[u8]], out: &mut [u8]) -> Result<(), String> {
        let row_bytes = self.row_bytes;
        if reads.len() != self.reads.len() {
            return Err(format!("{} reads returned for a plan of {}", reads.len(), self.reads.len()));
        }
        if out.len() != self.sources.len() * row_bytes {
            return Err(format!("{} output bytes for {} rows of {row_bytes}", out.len(), self.sources.len()));
        }
        for (i, source) in self.sources.iter().enumerate() {
            let (data, start) = match *source {
                RowSource::Hot { slot } => (hot, slot * row_bytes),
                RowSource::Read { read, offset } => (reads[read], offset),
            };
            let row = data.get(start..start + row_bytes).ok_or_else(|| {
                format!("request {i}: {source:?} needs bytes [{start}, {}) of {}", start + row_bytes, data.len())
            })?;
            out[i * row_bytes..(i + 1) * row_bytes].copy_from_slice(row);
        }
        Ok(())
    }
}

/// Plans a step's n-gram rows (`rows`, any order, repeats allowed): the
/// cache's rows come from RAM, the rest from aligned reads of the table's
/// file range. Every missing row is read once; rows whose sectors touch or
/// overlap share a read up to `policy.max_read_bytes`.
pub fn plan_gather(rows: &[u64], hot: &HotRows, layout: &TableLayout, policy: ReadPolicy) -> Result<GatherPlan, String> {
    if hot.row_bytes != layout.row_bytes {
        return Err(format!(
            "the hot-row cache holds {}-byte rows, the table {}-byte ones",
            hot.row_bytes, layout.row_bytes
        ));
    }
    let alignment = policy.alignment;
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(format!("read alignment {alignment} is not a power of two"));
    }
    // The widest span one row can need: its bytes starting one byte short of
    // a sector's end.
    let widest_row = (layout.row_bytes + alignment - 2) / alignment * alignment + alignment;
    if policy.max_read_bytes < widest_row {
        return Err(format!(
            "max_read_bytes {} cannot hold a {}-byte row at alignment {alignment} ({widest_row} bytes)",
            policy.max_read_bytes, layout.row_bytes
        ));
    }
    let span = |row: u64| {
        let start = layout.row_offset(row);
        let end = start + layout.row_bytes;
        (start / alignment * alignment, end.div_ceil(alignment) * alignment)
    };

    let mut missing: Vec<u64> = Vec::new();
    for &row in rows {
        if row >= layout.rows {
            return Err(format!("n-gram row {row} is outside the table's {} rows", layout.rows));
        }
        if hot.slot(row).is_none() {
            missing.push(row);
        }
    }
    missing.sort_unstable();
    missing.dedup();

    let mut reads: Vec<AlignedRead> = Vec::new();
    let mut read_of: Vec<usize> = Vec::with_capacity(missing.len());
    for &row in &missing {
        let (start, end) = span(row);
        match reads.last_mut() {
            Some(read) if start <= read.offset + read.len && end - read.offset <= policy.max_read_bytes => {
                read.len = read.len.max(end - read.offset);
            }
            _ => reads.push(AlignedRead { offset: start, len: end - start }),
        }
        read_of.push(reads.len() - 1);
    }

    let sources = rows
        .iter()
        .map(|&row| match hot.slot(row) {
            Some(slot) => RowSource::Hot { slot },
            None => {
                let index = missing.binary_search(&row).expect("every missing row was planned");
                let read = read_of[index];
                RowSource::Read { read, offset: (layout.row_offset(row) - reads[read].offset) as usize }
            }
        })
        .collect();
    Ok(GatherPlan { reads, sources, row_bytes: layout.row_bytes as usize })
}
