//! The hq-e8-2b residual window's rule, on the host (GitHub #257, spec
//! `docs/specs/runtime/06-hq-residual-window.md`).
//!
//! Under hq-e8-2b the vendored attention kernels read a key exact instead of
//! decoding it from three places: the current prefill chunk, the first
//! [`SINK_KEYS`] keys (sink rows, no validity bit), and a
//! [`RECENT_KEYS`]-slot **ring** of recent keys, where key `k` lives in slot
//! `k % 512` and is read exact iff the slot's bit is set. The bit carries no
//! position: it says "this slot holds the row of the last key appended
//! congruent to it". So what a kernel reads for a key depends on the order of
//! every append and every clear the slot has seen — which is what [`HqRing`]
//! replays, write for write, the way `kernel/include/ignis_hq_ring.h` and the
//! vendored append kernels do it.
//!
//! Two uses: the GPU tests compare the ring words the device holds after a
//! prefill, a verify round or a clone with [`HqRing::words`], and the
//! consumed-key tap classifies each key the prompt route read with
//! [`prompt_source`]. Pure arithmetic, so its own tests run on the CPU.
//!
//! **The prompt route reads the ring before its own chunk's append**
//! ([`ring_before_chunk`]): an **Ignis-patched** behaviour (ADR 0037, GitHub
//! #258, spec `docs/specs/runtime/07-hq-prefill-ring-read-before-append.md`).
//! The reference (ninfer `a00648cb`) enqueues the chunk's append (which
//! dual-writes the chunk's last 512 keys into their ring slots and sets their
//! bits) *before* the scratch decode that reads the ring for the 512 keys
//! before the chunk, so a key there whose slot the chunk has just rewritten is
//! served the chunk's row — [`PromptSource::Clobbered`], a key from the
//! query's future. That is [`prompt_source`] over [`ring_after_prefill`] of
//! the chunks up to and including the query's; ignis attends first, and
//! after the call the ring is the same either way.

/// Sink keys the window keeps exact, with no validity bit
/// (`kGqaHqSinkKeys`).
pub const SINK_KEYS: u64 = 32;
/// Ring slots (`kGqaHqRecentKeys`).
pub const RECENT_KEYS: u64 = 512;
/// Validity words per slot row: one bit per ring slot.
pub const RING_WORDS: usize = (RECENT_KEYS / 32) as usize;

/// One slot row's validity words, as the device lays them out: bit `r % 32`
/// of word `r / 32` is ring slot `r`.
pub type RingWords = [u32; RING_WORDS];

/// A model of one slot's ring: for each ring slot, the key that last wrote
/// it and whether its bit is set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HqRing {
    last: Vec<Option<u64>>,
    valid: Vec<bool>,
}

impl Default for HqRing {
    fn default() -> Self {
        Self::new()
    }
}

impl HqRing {
    /// A zeroed slot: what `ignis_seq_alloc` hands a sequence.
    pub fn new() -> Self {
        Self {
            last: vec![None; RECENT_KEYS as usize],
            valid: vec![false; RECENT_KEYS as usize],
        }
    }

    fn write(&mut self, key: u64) {
        // Sink keys go to their own rows and never set a bit
        // (`hq_ring_mark_valid` returns for them).
        if key < SINK_KEYS {
            return;
        }
        let slot = (key % RECENT_KEYS) as usize;
        self.last[slot] = Some(key);
        self.valid[slot] = true;
    }

    /// A prefill chunk's append, `len` keys from `start`: the fill kernel
    /// (`gqa_attention_prefill_fill_hq_kernel`) writes the chunk's sink keys
    /// and its last [`RECENT_KEYS`] keys — an earlier key of a wider chunk is
    /// superseded within it and skipped — and marks every ring slot it
    /// writes.
    pub fn append_prefill(&mut self, start: u64, len: u64) {
        let end = start + len;
        for key in start.max(end.saturating_sub(RECENT_KEYS))..end {
            self.write(key);
        }
    }

    /// A decode or verify pass's append of `len` columns from `start`: every
    /// column is written and marked (the fused append of
    /// `gqa_attention_small_t_tc_partial_bf16_kernel`).
    pub fn append_decode(&mut self, start: u64, len: u64) {
        for key in start..start + len {
            self.write(key);
        }
    }

    /// The clear a verify round applies over its uncommitted columns
    /// (`ignis_hq_ring_invalidate_mask`): the slots keys `[first, end)` were
    /// appended to lose their bits. At most one lap of the ring.
    pub fn invalidate(&mut self, first: u64, end: u64) {
        for key in first..end.min(first + RECENT_KEYS) {
            self.valid[(key % RECENT_KEYS) as usize] = false;
        }
    }

    /// The validity words the device should hold.
    pub fn words(&self) -> RingWords {
        let mut words = [0u32; RING_WORDS];
        for (slot, &valid) in self.valid.iter().enumerate() {
            if valid {
                words[slot / 32] |= 1 << (slot % 32);
            }
        }
        words
    }

    /// The key whose row ring slot `key % 512` holds, when its bit is set —
    /// what a kernel reading `key` from the ring is served.
    pub fn served(&self, key: u64) -> Option<u64> {
        let slot = (key % RECENT_KEYS) as usize;
        if self.valid[slot] { self.last[slot] } else { None }
    }
}

/// Where the hq prompt route takes one key from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptSource {
    /// The chunk being prefilled: staged exact from its own rows.
    Fresh,
    /// One of the [`SINK_KEYS`] sink keys: exact, from the sink rows.
    Sink,
    /// One of the [`RECENT_KEYS`] keys before the chunk, served its own
    /// exact row from the ring.
    Ring,
    /// One of the keys before the chunk served key `by`'s exact row instead
    /// of its own: what the reference's launch order does to the keys whose
    /// ring slots the chunk's own append rewrote. On ignis's order a ring the
    /// appends built never holds another key's row there, so a capture
    /// classified this way is off the rule.
    Clobbered { by: u64 },
    /// Everything else, and a ring key whose bit is clear: the codec.
    Codec,
}

/// [`PromptSource`] of the key at `position` for the query chunk starting at
/// `chunk_start`, with `ring` the slot's ring as that chunk's scratch decode
/// reads it — the rule of `gqa_attention_prefill_hq_scratch_kernel` with the
/// fresh-chunk pass on. On ignis's order that ring is [`ring_before_chunk`];
/// on the reference's it is the ring after the chunk's own append too.
pub fn prompt_source(position: u64, chunk_start: u64, ring: &HqRing) -> PromptSource {
    if position >= chunk_start {
        PromptSource::Fresh
    } else if position < SINK_KEYS {
        PromptSource::Sink
    } else if position + RECENT_KEYS >= chunk_start {
        match ring.served(position) {
            Some(key) if key == position => PromptSource::Ring,
            Some(key) => PromptSource::Clobbered { by: key },
            None => PromptSource::Codec,
        }
    } else {
        PromptSource::Codec
    }
}

/// The ring a fresh slot holds after a prefill of `chunks` (`(start, len)`
/// in order), each appended as the fill kernel appends it.
pub fn ring_after_prefill(chunks: &[(u64, u64)]) -> HqRing {
    let mut ring = HqRing::new();
    for &(start, len) in chunks {
        ring.append_prefill(start, len);
    }
    ring
}

/// The ring the prompt route reads for the chunk starting at `chunk_start`,
/// of a prefill run as `chunks` (`(start, len)` in order): the ring every
/// chunk before it left, since ignis attends a chunk before it appends it
/// (ADR 0037, GitHub #258).
pub fn ring_before_chunk(chunks: &[(u64, u64)], chunk_start: u64) -> HqRing {
    let before = chunks.iter().take_while(|&&(start, _)| start < chunk_start).count();
    ring_after_prefill(&chunks[..before])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count(words: &RingWords) -> u32 {
        words.iter().map(|w| w.count_ones()).sum()
    }

    #[test]
    fn a_prefill_chunk_writes_its_last_512_keys_and_never_a_sink_bit() {
        let mut ring = HqRing::new();
        ring.append_prefill(0, 100);
        // Keys 32..100 set bits; the sinks 0..32 set none.
        assert_eq!(count(&ring.words()), 68);
        assert_eq!(ring.served(40), Some(40));
        assert_eq!(ring.served(10), None, "a sink key has no ring slot");
        // A 1024-wide chunk writes only its last 512 keys, every slot once.
        let ring = ring_after_prefill(&[(0, 1024)]);
        assert_eq!(count(&ring.words()), 512);
        assert_eq!(ring.served(600), Some(600));
        assert_eq!(ring.served(100), Some(612), "slot 100 holds the later key");
    }

    fn tally(chunk_start: u64, total: u64, ring: &HqRing) -> (usize, usize) {
        let sources: Vec<_> = (0..total).map(|p| prompt_source(p, chunk_start, ring)).collect();
        let clobbered = sources.iter().filter(|s| matches!(s, PromptSource::Clobbered { .. })).count();
        let ring_rows = sources.iter().filter(|s| **s == PromptSource::Ring).count();
        (clobbered, ring_rows)
    }

    #[test]
    fn the_prompt_route_reads_the_ring_the_chunks_before_it_left() {
        // Two chunks, the second 122 wide at 1024 -- the 1024 px pointing
        // prompt's shape: 1146 tokens, the query chunk from 1024.
        let chunks = [(0, 1024), (1024, 122)];
        let ring = ring_before_chunk(&chunks, 1024);
        assert_eq!(ring, ring_after_prefill(&chunks[..1]), "the query chunk's own append is not in it");
        // Every key of the ring window [512, 1024) is served its own row.
        assert_eq!(prompt_source(512, 1024, &ring), PromptSource::Ring);
        assert_eq!(prompt_source(633, 1024, &ring), PromptSource::Ring);
        assert_eq!(prompt_source(1023, 1024, &ring), PromptSource::Ring);
        assert_eq!(prompt_source(1024, 1024, &ring), PromptSource::Fresh);
        assert_eq!(prompt_source(31, 1024, &ring), PromptSource::Sink);
        assert_eq!(prompt_source(32, 1024, &ring), PromptSource::Codec);
        assert_eq!(prompt_source(511, 1024, &ring), PromptSource::Codec);
        assert_eq!(tally(1024, 1146, &ring), (0, 512));
        // A full-width chunk after the first keeps the whole window too.
        let chunks = [(0, 1024), (1024, 1024)];
        assert_eq!(tally(1024, 2048, &ring_before_chunk(&chunks, 1024)), (0, 512));
        // The first chunk has nothing before it: no ring row at all.
        assert_eq!(tally(0, 1024, &ring_before_chunk(&chunks, 0)), (0, 0));
    }

    #[test]
    fn the_references_order_serves_ring_keys_the_chunks_own_rows() {
        // The reference appends the query chunk before it attends, so the
        // ring it reads includes that append (spec runtime/07).
        let ring = ring_after_prefill(&[(0, 1024), (1024, 122)]);
        // The first 122 keys of the ring window [512, 1024) share their slots
        // with the chunk's keys [1024, 1146) and are served those.
        assert_eq!(prompt_source(512, 1024, &ring), PromptSource::Clobbered { by: 1024 });
        assert_eq!(prompt_source(633, 1024, &ring), PromptSource::Clobbered { by: 1145 });
        assert_eq!(prompt_source(634, 1024, &ring), PromptSource::Ring);
        assert_eq!(tally(1024, 1146, &ring), (122, 390));
        // A full-width chunk after the first clobbers the whole window.
        let ring = ring_after_prefill(&[(0, 1024), (1024, 1024)]);
        assert_eq!(tally(1024, 2048, &ring), (512, 0));
    }

    #[test]
    fn a_verify_round_clears_what_it_did_not_commit() {
        let mut ring = ring_after_prefill(&[(0, 1000)]);
        assert_eq!(count(&ring.words()), 512);
        // Eight columns at 1000, three committed.
        ring.append_decode(1000, 8);
        ring.invalidate(1003, 1008);
        assert_eq!(count(&ring.words()), 507);
        assert_eq!(ring.served(1002), Some(1002));
        assert_eq!(ring.served(1003 - 512), None, "the older key falls back to the codec");
        // The next round re-appends from the frontier and sets them again.
        ring.append_decode(1003, 8);
        assert_eq!(count(&ring.words()), 512);
        assert_eq!(ring.served(1005), Some(1005));
    }

    #[test]
    fn the_words_match_the_device_bit_layout() {
        let mut ring = HqRing::new();
        ring.append_decode(32 + 512 * 3, 1); // slot 32: word 1, bit 0
        ring.append_decode(511, 1); // slot 511: word 15, bit 31
        let words = ring.words();
        assert_eq!(words[1], 1);
        assert_eq!(words[15], 1 << 31);
        assert_eq!(count(&words), 2);
    }
}
