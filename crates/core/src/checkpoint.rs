//! Prompt checkpoints — the device tier of cross-request state reuse
//! (GitHub #186, ADR 0029; spec `.scratch/kv-reuse/specs/01-cross-request-reuse.md`
//! §"Prompt checkpoint on the device").
//!
//! A **prompt checkpoint** is the whole-sequence state of a request at its
//! **generation opener** — the byte offset where the rendered prompt's last
//! `<|im_start|>assistant\n` ends — retained after the request finishes so
//! that a later request whose prompt *extends* it resumes there instead of
//! re-prefilling. It is the one point every later turn of a conversation
//! provably shares: the next turn re-renders the assistant message
//! differently from how it was generated, so nothing past the opener ever
//! matches (ADR 0029).
//!
//! This module is the CPU half: the ledger of which checkpoints exist, what
//! they cost, and which one a prompt matches. It owns no device memory and
//! moves no bytes — exactly as [`crate::host::HostTier`] owns no host memory
//! — so the whole policy is testable without a GPU (ADR 0006). The device
//! image itself lives behind the [`Compute`](crate::scheduler::Compute) seam.
//!
//! Three properties shape the type below, and each is a rule the scheduler
//! leans on rather than a detail:
//!
//! - **Reuse copies, never consumes.** [`CheckpointPool::claim`] hands out a
//!   match and leaves the entry in place, so a retry, a regenerate and two
//!   forks from the same history all hit (ADR 0029).
//! - **Retained state is free until the room is needed.** The pool has a byte
//!   budget ([`CheckpointPool::admits`]); when it is full a capture is simply
//!   **not taken** — [`CheckpointPool::retain`] never evicts to make room for
//!   itself. What *does* remove an entry is a live request needing the pages
//!   ([`CheckpointPool::discard_victim`]): retained state is the first victim
//!   on the device, always, and it never delays or refuses an admission (ADR
//!   0023 as amended by ADR 0029).
//! - **Entries are addressed by content, never by session.** The match is the
//!   longest retained entry whose tokens are a prefix of the request's prompt.
//!   There is no API field for it and no `RequestId` in the key: a wrong
//!   session id could hand one conversation another's state, a content match
//!   can only hand over identical history (ADR 0029).
//!
//! **What later slices change here.** #187 (lineage) adds the
//! latest/turn-opening pair and supersession; #188 (retained prefix) adds a
//! second kind of retained entry, matched against the same prompt; #190
//! (KV-RAM) makes [`CheckpointPool::discard_victim`] a *spill* rather than a
//! discard, and gives the match a second tier to choose between — which is
//! why [`CheckpointMatch`] already names its [`ReuseSource`]; #189 (identity)
//! replaces [`CheckpointEntry::tokens`] with a hash chain over the token ids
//! and adds the compatibility identity a blob is refused on, with #193
//! filling the media slot in that key. None of those need the shape here to
//! change.

use crate::gdn::GdnState;
use crate::prefix::PrefixId;
use crate::types::{RequestId, TokenId};

/// An opaque handle to a retained prompt checkpoint.
pub type CheckpointId = u64;

/// Where a request's reused state came from — the `reuse_source` field of the
/// request log (spec §12).
///
/// `Device` is Tier 0 (the checkpoint's pages and image never left the card)
/// and `KvRam` is Tier 1, which #190 fills. A request that reused nothing
/// reports no source at all, which is why "none" is the absence of this value
/// rather than a variant of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReuseSource {
    /// Tier 0: the checkpoint was still resident on the device.
    Device,
    /// Tier 1: the checkpoint came back from KV-RAM (GitHub #190).
    KvRam,
}

impl ReuseSource {
    /// The request log's wire spelling (`device` / `kv_ram`). The gate spec
    /// maps these onto ninfer's `vram_resident` / `host_ram`.
    pub fn as_str(self) -> &'static str {
        match self {
            ReuseSource::Device => "device",
            ReuseSource::KvRam => "kv_ram",
        }
    }
}

/// A retained prompt checkpoint: one finished request's state at its
/// generation opener.
///
/// The entry does not own the history it covers. The whole KV pages up to the
/// opener are the pages of the **shared prefix** the request published under
/// it ([`CheckpointEntry::prefix`]), held by the same refcount every
/// concurrent sibling uses — so the pages are charged to the pool exactly
/// once whether a live sibling, a retained checkpoint, or both are holding
/// them. What the entry itself costs is [`CheckpointEntry::bytes`]: the
/// device image of the mutable sections plus the copy of the partial tail
/// page, which is what the pool's byte budget bounds.
#[derive(Debug, Clone)]
pub struct CheckpointEntry {
    /// The entry's opaque id.
    pub id: CheckpointId,
    /// The request whose prefill captured it. The compute backend keys its
    /// device image by this id, exactly as it keys a shared prefix by its
    /// publisher (P4-10, GitHub #126) — the request is long gone, but its id
    /// is never reused, so it stays a valid name for the bytes it left
    /// behind. It is a *backend handle*, never part of the match key: ADR
    /// 0029's "keys never contain a `RequestId`" is about
    /// [`CheckpointEntry::tokens`], which is what #189 turns into a hash
    /// chain.
    pub publisher: RequestId,
    /// The prompt tokens up to and including the generation opener: the match
    /// key. A request whose prompt starts with these reuses this entry.
    pub tokens: Vec<TokenId>,
    /// The shared-prefix entry whose pages carry this checkpoint's history.
    /// The checkpoint counts as one of its claimants, which is what keeps the
    /// pages alive after every live request has gone.
    pub prefix: PrefixId,
    /// Whole KV pages that shared prefix holds (`tokens` floored to pages).
    pub pages: u32,
    /// Device bytes the entry's own image occupies in the retained pool.
    pub bytes: u64,
    /// The GDN state at the opener (core-02: a claimant resumes its recurrent
    /// state from a recorded boundary).
    pub gdn: GdnState,
    /// The scheduling tick this entry was last captured or claimed at — the
    /// LRU order [`CheckpointPool::discard_victim`] discards in.
    pub use_tick: u64,
}

/// A successful match: what the claimant needs to be allocated against the
/// checkpoint instead of prefilling its head.
#[derive(Debug, Clone)]
pub struct CheckpointMatch {
    /// The matched entry's id.
    pub id: CheckpointId,
    /// The request that captured the entry — what the prefill job carries so
    /// the backend can find the device image.
    pub publisher: RequestId,
    /// The tier the match came from (`device` today; #190 adds `kv_ram`).
    pub source: ReuseSource,
    /// The shared-prefix entry carrying the checkpoint's whole KV pages. The
    /// claimant takes a reference to it like any other holder, so the pages
    /// stay charged to the pool exactly once however many live requests and
    /// retained checkpoints are standing on them.
    pub prefix: PrefixId,
    /// Leading prompt tokens the claimant skips: the whole prompt up to the
    /// opener, **not** a whole number of pages — the claimant copies the
    /// partial tail page rather than sharing it (CONTEXT.md, "publish point").
    pub tokens: u32,
    /// The whole KV pages the entry's shared prefix holds. The claimant's own
    /// reservation shrinks by these: they are charged to the pool once.
    pub pages: u32,
    /// The GDN state to resume from.
    pub gdn: GdnState,
}

/// The byte-budgeted device pool of retained prompt checkpoints (ADR 0029,
/// `--retained-pool-bytes`).
///
/// A byte budget rather than an entry count for the reason the KV-RAM tier
/// uses one (`crate::host`): a checkpoint's image is dominated by the GDN
/// recurrent state and the drafter window, a fixed floor that a short
/// conversation pays exactly as a long one does, so counting entries would
/// misprice the thing being bounded.
#[derive(Debug)]
pub struct CheckpointPool {
    capacity_bytes: u64,
    used_bytes: u64,
    entries: Vec<CheckpointEntry>,
    next_id: CheckpointId,
    reused_tok: u64,
    captures: u64,
    skipped_captures: u64,
    discards: u64,
}

impl CheckpointPool {
    /// A pool bounded by `capacity_bytes` of device image. `0` disables
    /// retention entirely (every capture is skipped) — a legal, explicit
    /// choice, the way a `0` KV-RAM budget disables the host tier.
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            used_bytes: 0,
            entries: Vec::new(),
            next_id: 0,
            reused_tok: 0,
            captures: 0,
            skipped_captures: 0,
            discards: 0,
        }
    }

    /// The pool's byte budget.
    pub fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }

    /// Device image bytes the retained entries occupy.
    pub fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    /// Retained entries.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// The KV pages the retained entries hold through their shared prefixes.
    ///
    /// Counted once per **prefix**, not once per entry: two checkpoints taken
    /// at different openers inside the same prompt head hold the same shared
    /// prefix, and its pages are charged to the KV pool exactly once
    /// ([`crate::prefix::PrefixCache`] is what carries that charge). Summing
    /// per entry would tell the admission machine that more of the pool is
    /// reclaimable than actually is.
    pub fn retained_pages(&self) -> u32 {
        let mut counted: Vec<PrefixId> = Vec::with_capacity(self.entries.len());
        let mut pages = 0;
        for entry in &self.entries {
            if counted.contains(&entry.prefix) {
                continue;
            }
            counted.push(entry.prefix);
            pages += entry.pages;
        }
        pages
    }

    /// The distinct shared prefixes the retained entries stand on.
    pub fn retained_prefixes(&self) -> Vec<PrefixId> {
        let mut out: Vec<PrefixId> = Vec::with_capacity(self.entries.len());
        for entry in &self.entries {
            if !out.contains(&entry.prefix) {
                out.push(entry.prefix);
            }
        }
        out
    }

    /// How many retained entries stand on `prefix`.
    ///
    /// A conversation can leave two checkpoints inside one prompt head, so a
    /// caller asking "would giving these up return the pages" has to compare
    /// this against the prefix's own refcount rather than assume one holder.
    pub fn retained_holders(&self, prefix: PrefixId) -> u32 {
        self.entries.iter().filter(|e| e.prefix == prefix).count() as u32
    }

    /// Whether a checkpoint over exactly `tokens` is already retained.
    ///
    /// A second capture at the same point would be a second image of the same
    /// state: the same skip for a claimant, paid for twice out of the byte
    /// budget. The caller checks this before it asks the backend to capture,
    /// the way [`crate::prefix::PrefixCache::register`] declines a duplicate
    /// prompt head.
    pub fn holds(&self, tokens: &[TokenId]) -> bool {
        self.entries.iter().any(|e| e.tokens == tokens)
    }

    /// Whether the byte budget can take one more image of `bytes`.
    ///
    /// This is the whole of "a full pool skips the capture": a capture that
    /// does not fit is **not taken**, and nothing is evicted to make room for
    /// it. Retention is a bet; a bet never costs certain work (ADR 0029).
    pub fn admits(&self, bytes: u64) -> bool {
        bytes > 0 && self.used_bytes.saturating_add(bytes) <= self.capacity_bytes
    }

    /// Retain a checkpoint captured at `tokens`' end (the generation opener).
    ///
    /// Returns the new entry's id, or `None` when the byte budget cannot take
    /// it — the caller must have asked [`CheckpointPool::admits`] before it
    /// let the backend capture anything, so a `None` here is the pool
    /// refusing a capture that raced its own budget, not an error path.
    #[allow(clippy::too_many_arguments)]
    pub fn retain(
        &mut self,
        publisher: RequestId,
        tokens: Vec<TokenId>,
        prefix: PrefixId,
        pages: u32,
        bytes: u64,
        gdn: GdnState,
        use_tick: u64,
    ) -> Option<CheckpointId> {
        if !self.admits(bytes) {
            self.skipped_captures += 1;
            return None;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.used_bytes += bytes;
        self.captures += 1;
        self.entries.push(CheckpointEntry {
            id,
            publisher,
            tokens,
            prefix,
            pages,
            bytes,
            gdn,
            use_tick,
        });
        Some(id)
    }

    /// The longest retained checkpoint whose tokens are a prefix of `tokens`,
    /// or `None` when nothing matches. A pure query: it decides nothing and
    /// changes nothing.
    ///
    /// It is separate from [`CheckpointPool::record_claim`] because the
    /// scheduler has to weigh this match against a *sibling prefix* match
    /// before it takes either — longest reuse wins (ADR 0029), and a losing
    /// match must leave no trace in the LRU order or the reuse counter.
    pub fn best_match(&self, tokens: &[TokenId]) -> Option<CheckpointMatch> {
        self.entries
            .iter()
            .filter(|e| e.tokens.len() <= tokens.len() && tokens.starts_with(&e.tokens))
            // The longest reuse wins (ADR 0029). Two device entries cannot
            // tie — two entries with the same token content are the same
            // conversation at the same point — but #190 has to break a
            // device-vs-KV-RAM tie in favour of the device, which is why the
            // tier is carried on the match rather than inferred from it.
            .max_by_key(|e| e.tokens.len())
            .map(|best| CheckpointMatch {
                id: best.id,
                publisher: best.publisher,
                source: ReuseSource::Device,
                prefix: best.prefix,
                tokens: best.tokens.len() as u32,
                pages: best.pages,
                gdn: best.gdn.clone(),
            })
    }

    /// Record that `id` was claimed: refresh its LRU tick and count the
    /// skipped tokens.
    ///
    /// **Non-consuming**: the entry stays. N claimants of one checkpoint all
    /// hit (ADR 0029) — which is what makes regenerate, retry and two forks
    /// of one history free rather than a race.
    pub fn record_claim(&mut self, id: CheckpointId, use_tick: u64) {
        let Some(entry) = self.entries.iter_mut().find(|e| e.id == id) else {
            return;
        };
        entry.use_tick = use_tick;
        self.reused_tok += entry.tokens.len() as u64;
    }

    /// [`CheckpointPool::best_match`] followed by
    /// [`CheckpointPool::record_claim`], for a caller with nothing to weigh
    /// the match against.
    pub fn claim(&mut self, tokens: &[TokenId], use_tick: u64) -> Option<CheckpointMatch> {
        let matched = self.best_match(tokens)?;
        self.record_claim(matched.id, use_tick);
        Some(matched)
    }

    /// Discard the least recently used entry and return it, so the caller can
    /// release its image and its shared-prefix claim. `None` when the pool is
    /// empty.
    ///
    /// This is the **first victim** path (ADR 0023 as amended): a live request
    /// that cannot materialize takes retained pages back before admission
    /// considers evicting anybody. Until #190, released means discarded.
    pub fn discard_victim(&mut self) -> Option<CheckpointEntry> {
        self.discard_victim_on(None)
    }

    /// [`CheckpointPool::discard_victim`], restricted to entries standing on
    /// one of `prefixes` (`None` means any).
    ///
    /// The restriction is what keeps the first-victim path from emptying the
    /// pool for nothing. A retained entry's pages come back only when the
    /// prefix under it has no holder left at all, so an entry whose prefix a
    /// live request is *also* standing on frees nothing when it goes — and
    /// giving it up would cost the next turn its reuse while buying this
    /// request no page. The caller works out which prefixes qualify, because
    /// only it can see the live holders.
    pub fn discard_victim_on(&mut self, prefixes: Option<&[PrefixId]>) -> Option<CheckpointEntry> {
        let pos = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| prefixes.is_none_or(|allowed| allowed.contains(&e.prefix)))
            .min_by_key(|(_, e)| (e.use_tick, e.id))
            .map(|(i, _)| i)?;
        Some(self.remove_at(pos))
    }

    /// Discard the entry `id`, if it is still retained.
    pub fn discard(&mut self, id: CheckpointId) -> Option<CheckpointEntry> {
        let pos = self.entries.iter().position(|e| e.id == id)?;
        Some(self.remove_at(pos))
    }

    /// Cumulative prompt tokens claimants skipped through a retained
    /// checkpoint. Cumulative skips, not net savings — read it the way
    /// [`crate::prefix::PrefixCache::reused_tok`] is read.
    pub fn reused_tok(&self) -> u64 {
        self.reused_tok
    }

    /// Checkpoints taken, checkpoints the byte budget refused, and
    /// checkpoints discarded — the three counters #190's per-tier
    /// hit / miss / spill / discard reporting is built from.
    pub fn counters(&self) -> CheckpointCounters {
        CheckpointCounters {
            captures: self.captures,
            skipped_captures: self.skipped_captures,
            discards: self.discards,
            reused_tok: self.reused_tok,
        }
    }

    fn remove_at(&mut self, pos: usize) -> CheckpointEntry {
        let entry = self.entries.remove(pos);
        self.used_bytes = self.used_bytes.saturating_sub(entry.bytes);
        self.discards += 1;
        entry
    }
}

/// The retained pool's lifetime counters. #190 maps them onto ADR 0017's
/// per-tier hit / miss / spill / discard contract, which happens wholly on
/// the asynchronous telemetry side: nothing on the inference path knows
/// these are projected anywhere.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CheckpointCounters {
    /// Checkpoints captured and retained.
    pub captures: u64,
    /// Captures the byte budget refused (nothing was evicted for them).
    pub skipped_captures: u64,
    /// Entries discarded, whether for a live request's pages or superseded.
    pub discards: u64,
    /// Prompt tokens skipped through a retained checkpoint.
    pub reused_tok: u64,
}

/// Device bytes held back from the retained pool's derived default: what
/// the leaf must still be able to allocate at serving time (GitHub #186).
///
/// Retention is a bet, and a bet must never be the reason certain work
/// fails. The leaf allocates out of the same free device memory while it
/// serves — a media encode transient, a graph capture, the scratch a wider
/// chunk needs — so the derived budget starts by giving that back.
pub const RETAINED_POOL_RESERVE_BYTES: u64 = 1024 * 1024 * 1024;

/// The most the retained pool's default ever claims (GitHub #186).
///
/// Past a few dozen checkpoints the marginal one buys little — a
/// conversation keeps at most two (ADR 0029) and a burst shares one prefix —
/// while the headroom it costs buys a lot. An operator who wants more says
/// so with `--retained-pool-bytes`.
pub const MAX_AUTO_RETAINED_POOL_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// The retained pool's default byte budget, derived from the device memory
/// still free once the model, its KV pool and every other reservation have
/// landed (GitHub #186, ADR 0029; the operator's `--retained-pool-bytes`
/// overrides it).
///
/// Half of what is left after [`RETAINED_POOL_RESERVE_BYTES`], capped at
/// [`MAX_AUTO_RETAINED_POOL_BYTES`]. Half rather than all, because "free"
/// is measured once, at startup, and the number it returns is not a promise
/// about the rest of the run.
///
/// A caller that cannot measure free memory passes 0 and gets 0: nothing is
/// retained, which is the same engine that existed before checkpoints did.
/// That is deliberate — an earlier attempt at sizing the *KV* pool from a
/// naive free-VRAM guess OOM'd on the real artifact
/// (`crates/runtime/src/cuda_leaf.rs`'s module doc), and the lesson taken
/// from it is that a derived device budget guesses downwards or not at all.
pub fn auto_retained_pool_bytes(free_device_bytes: u64) -> u64 {
    (free_device_bytes.saturating_sub(RETAINED_POOL_RESERVE_BYTES) / 2)
        .min(MAX_AUTO_RETAINED_POOL_BYTES)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A GDN state resumable at `position` (a recorded boundary).
    fn gdn_at(position: usize) -> GdnState {
        let mut gdn = GdnState::new();
        gdn.checkpoint(position);
        gdn
    }

    /// One image's worth of device bytes, for a pool sized in whole images.
    const IMAGE: u64 = 1_000;

    fn pool_of(images: u64) -> CheckpointPool {
        CheckpointPool::new(IMAGE * images)
    }

    /// Retain a checkpoint over `tokens`, at `tick`, holding `pages` pages.
    fn retain(
        pool: &mut CheckpointPool,
        publisher: RequestId,
        tokens: &[TokenId],
        pages: u32,
        tick: u64,
    ) -> Option<CheckpointId> {
        pool.retain(
            publisher,
            tokens.to_vec(),
            publisher, // one prefix per publisher in these tests
            pages,
            IMAGE,
            gdn_at(tokens.len()),
            tick,
        )
    }

    #[test]
    fn a_prompt_that_extends_a_checkpoint_matches_it() {
        // The whole point: turn N leaves a checkpoint at its opener, and turn
        // N+1 — whose prompt is turn N's prompt plus the assistant's reply and
        // the new user message — starts with it.
        let mut pool = pool_of(4);
        let turn_n: Vec<TokenId> = (1..=100).collect();
        retain(&mut pool, 7, &turn_n, 1, 1).unwrap();
        let turn_n_plus_1: Vec<TokenId> = (1..=160).collect();
        let matched = pool.claim(&turn_n_plus_1, 2).expect("turn N+1 extends turn N");
        assert_eq!(matched.tokens, 100, "everything up to turn N's opener");
        assert_eq!(matched.publisher, 7);
        assert_eq!(matched.source, ReuseSource::Device);
        assert_eq!(pool.reused_tok(), 100);
    }

    #[test]
    fn a_prompt_that_only_shares_a_head_does_not_match() {
        // Reuse happens at recorded points, never at an arbitrary longest
        // common prefix (ADR 0029): GDN state exists only where it was
        // captured. A prompt that diverges *before* the checkpoint's opener
        // has no state to resume from.
        let mut pool = pool_of(4);
        let retained: Vec<TokenId> = (1..=100).collect();
        retain(&mut pool, 7, &retained, 1, 1).unwrap();
        let mut divergent: Vec<TokenId> = (1..=99).collect();
        divergent.push(999);
        divergent.extend(200..260);
        assert!(
            pool.claim(&divergent, 2).is_none(),
            "a prompt that diverges before the opener reuses nothing"
        );
        assert_eq!(pool.reused_tok(), 0);
    }

    #[test]
    fn the_longest_retained_checkpoint_wins() {
        // A conversation leaves a checkpoint per turn; the newest one is the
        // longest, and it is the one that saves the most prefill.
        let mut pool = pool_of(4);
        let turn1: Vec<TokenId> = (1..=100).collect();
        let turn2: Vec<TokenId> = (1..=180).collect();
        retain(&mut pool, 1, &turn1, 1, 1).unwrap();
        retain(&mut pool, 2, &turn2, 2, 2).unwrap();
        let turn3: Vec<TokenId> = (1..=250).collect();
        let matched = pool.claim(&turn3, 3).unwrap();
        assert_eq!(matched.tokens, 180, "the longest reuse wins");
        assert_eq!(matched.publisher, 2);
        assert_eq!(matched.pages, 2);
    }

    #[test]
    fn a_claim_does_not_consume_the_entry() {
        // Retry, regenerate and two forks from one history all hit (ADR
        // 0029): the entry survives every claimant.
        let mut pool = pool_of(4);
        let retained: Vec<TokenId> = (1..=100).collect();
        retain(&mut pool, 7, &retained, 1, 1).unwrap();
        let prompt: Vec<TokenId> = (1..=140).collect();
        for tick in 2..=4 {
            assert!(pool.claim(&prompt, tick).is_some(), "claim {tick} hits");
        }
        assert_eq!(pool.entry_count(), 1, "the entry survives its claimants");
        assert_eq!(pool.reused_tok(), 300, "each claim counts its skip");
    }

    #[test]
    fn a_full_pool_skips_the_capture_and_evicts_nothing() {
        // The byte budget is a hard stop, not a pressure signal: when it is
        // full the checkpoint is simply not taken, and the entries already in
        // it are untouched.
        let mut pool = pool_of(2);
        let a: Vec<TokenId> = (1..=100).collect();
        let b: Vec<TokenId> = (500..=600).collect();
        let c: Vec<TokenId> = (900..=1000).collect();
        retain(&mut pool, 1, &a, 1, 1).unwrap();
        retain(&mut pool, 2, &b, 1, 2).unwrap();
        assert!(!pool.admits(IMAGE), "the budget is spent");
        assert_eq!(retain(&mut pool, 3, &c, 1, 3), None, "the capture is skipped");
        assert_eq!(pool.entry_count(), 2, "nothing was evicted to take it");
        assert_eq!(pool.used_bytes(), 2 * IMAGE);
        assert_eq!(pool.counters().skipped_captures, 1);
        assert_eq!(pool.counters().discards, 0, "a skipped capture discards nothing");
    }

    #[test]
    fn a_zero_budget_retains_nothing() {
        let mut pool = pool_of(0);
        let a: Vec<TokenId> = (1..=100).collect();
        assert!(!pool.admits(IMAGE));
        assert_eq!(retain(&mut pool, 1, &a, 1, 1), None);
        assert_eq!(pool.entry_count(), 0);
    }

    #[test]
    fn the_first_victim_is_the_least_recently_used_entry() {
        // A live request that needs the pages takes them back from the entry
        // whose conversation has been quiet longest.
        let mut pool = pool_of(4);
        let a: Vec<TokenId> = (1..=100).collect();
        let b: Vec<TokenId> = (500..=600).collect();
        retain(&mut pool, 1, &a, 3, 1).unwrap();
        retain(&mut pool, 2, &b, 5, 2).unwrap();
        // Claiming `a` refreshes it, so `b` is now the older entry.
        let prompt: Vec<TokenId> = (1..=140).collect();
        pool.claim(&prompt, 9).unwrap();
        assert_eq!(pool.retained_pages(), 8);
        let victim = pool.discard_victim().expect("a victim");
        assert_eq!(victim.publisher, 2, "the least recently used entry goes");
        assert_eq!(victim.pages, 5, "the caller gets its pages back to release");
        assert_eq!(pool.used_bytes(), IMAGE, "its image bytes came back");
        assert_eq!(pool.retained_pages(), 3);
        assert_eq!(pool.discard_victim().map(|e| e.publisher), Some(1));
        assert!(pool.discard_victim().is_none(), "an empty pool has no victim");
        assert_eq!(pool.used_bytes(), 0);
    }

    #[test]
    fn a_discarded_entry_is_no_longer_matched() {
        let mut pool = pool_of(4);
        let a: Vec<TokenId> = (1..=100).collect();
        let id = retain(&mut pool, 1, &a, 1, 1).unwrap();
        let prompt: Vec<TokenId> = (1..=140).collect();
        assert!(pool.claim(&prompt, 2).is_some());
        assert_eq!(pool.discard(id).map(|e| e.id), Some(id));
        assert!(pool.claim(&prompt, 3).is_none(), "a discarded entry is gone");
        assert_eq!(pool.discard(id).map(|e| e.id), None, "and gone once");
    }

    #[test]
    fn a_claim_seeds_the_claimant_gdn_state_at_the_opener() {
        // core-02: the claimant resumes its recurrent state at a recorded
        // boundary — the opener, not a page boundary.
        let mut pool = pool_of(4);
        let a: Vec<TokenId> = (1..=100).collect();
        retain(&mut pool, 1, &a, 1, 1).unwrap();
        let prompt: Vec<TokenId> = (1..=140).collect();
        let matched = pool.claim(&prompt, 2).unwrap();
        assert!(matched.gdn.is_valid_snapshot_point(100));
        assert_eq!(matched.gdn.position(), 100);
    }

    #[test]
    fn a_losing_match_leaves_no_trace() {
        // The scheduler weighs a checkpoint match against a sibling-prefix
        // match before it takes either. A checkpoint that loses that
        // comparison must not have moved in the LRU order or counted itself
        // as reuse — which is why `best_match` is a pure query.
        let mut pool = pool_of(4);
        let older: Vec<TokenId> = (1..=100).collect();
        let newer: Vec<TokenId> = (500..=600).collect();
        retain(&mut pool, 1, &older, 1, 1).unwrap();
        retain(&mut pool, 2, &newer, 1, 5).unwrap();
        let prompt: Vec<TokenId> = (1..=140).collect();
        let peek = pool.best_match(&prompt).expect("a match");
        assert_eq!(peek.tokens, 100);
        assert_eq!(pool.reused_tok(), 0, "peeking is not reuse");
        // The LRU order is untouched: entry 1 is still the older one.
        assert_eq!(pool.discard_victim().map(|e| e.publisher), Some(1));
    }

    #[test]
    fn a_duplicate_capture_point_is_already_held() {
        // Two requests with the same prompt reach the same opener. The second
        // has nothing new to retain, and a second image of the same state
        // would cost the byte budget twice for the same skip.
        let mut pool = pool_of(4);
        let a: Vec<TokenId> = (1..=100).collect();
        retain(&mut pool, 1, &a, 1, 1).unwrap();
        assert!(pool.holds(&a));
        assert!(!pool.holds(&a[..99]), "a different point is not held");
    }

    #[test]
    fn retained_pages_count_a_shared_prefix_once() {
        // Two checkpoints taken inside the same prompt head hold the same
        // shared prefix, whose pages the KV pool is charged for exactly once.
        // Summing per entry would promise admission pages it cannot get back.
        let mut pool = pool_of(4);
        let early: Vec<TokenId> = (1..=100).collect();
        let late: Vec<TokenId> = (1..=120).collect();
        pool.retain(1, early, 42, 6, IMAGE, gdn_at(100), 1).unwrap();
        pool.retain(2, late, 42, 6, IMAGE, gdn_at(120), 2).unwrap();
        assert_eq!(pool.entry_count(), 2);
        assert_eq!(pool.retained_pages(), 6, "one prefix, one charge");
    }

    #[test]
    fn the_derived_budget_gives_back_the_reserve_and_halves_the_rest() {
        // The shape of a real load: a 32 GiB card with the 27B artifact,
        // its KV pool and its reservations already down leaves single-digit
        // gigabytes. 5 GiB free holds 1 GiB back for the leaf's own
        // serving-time allocations and retains half of the remaining 4.
        const GIB: u64 = 1024 * 1024 * 1024;
        assert_eq!(auto_retained_pool_bytes(5 * GIB), 2 * GIB);
        // At ~150 MiB an image (the 27B geometry's mutable sections plus one
        // KV page), that is a few dozen live conversations.
        assert!(auto_retained_pool_bytes(5 * GIB) / (150 * 1024 * 1024) >= 10);
    }

    #[test]
    fn the_derived_budget_is_capped_and_never_negative() {
        // A card with room to spare does not get a proportionally huge bet:
        // the cap binds well before half of it.
        assert_eq!(
            auto_retained_pool_bytes(u64::MAX),
            MAX_AUTO_RETAINED_POOL_BYTES
        );
        assert_eq!(
            auto_retained_pool_bytes(64 * 1024 * 1024 * 1024),
            MAX_AUTO_RETAINED_POOL_BYTES
        );
        // Less free than the reserve, or a backend that cannot measure free
        // memory at all: retain nothing, which is the engine that existed
        // before checkpoints did.
        assert_eq!(auto_retained_pool_bytes(RETAINED_POOL_RESERVE_BYTES), 0);
        assert_eq!(auto_retained_pool_bytes(0), 0);
    }

    #[test]
    fn reuse_source_spells_the_two_tiers() {
        assert_eq!(ReuseSource::Device.as_str(), "device");
        assert_eq!(ReuseSource::KvRam.as_str(), "kv_ram");
    }
}
