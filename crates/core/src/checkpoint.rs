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
//!   longest retained entry whose [`MatchKey`] — a hash chain over the token
//!   ids, with each media item's identity mixed in (GitHub #189,
//!   [`crate::identity`]) — is a prefix key of the request's prompt. There is
//!   no API field for it and no `RequestId` in the key: a wrong session id
//!   could hand one conversation another's state, a content match can only
//!   hand over identical history (ADR 0029).
//!
//! An entry also carries the [`BlobIdentity`] its state was produced under,
//! and the pool matches only entries whose identity this load accepts. Inside
//! one process every entry was captured under the pool's own identity and the
//! filter never fires; it exists because the entries a tier below the device
//! holds do not all come from this process — that is the whole of the Tier 2
//! seam, and a filter that only runs when a blob is adopted would be a filter
//! that was never exercised.
//!
//! - **A conversation keeps at most two.** #187 added the third rule: every
//!   capture joins the lineage of the entry its request claimed
//!   ([`CheckpointCapture::claimed`]), and that lineage is immediately cut
//!   back to its latest entry and its newest **turn-opening** one. A tool
//!   loop therefore costs one pair, not one entry per iteration, and the
//!   turn-opening entry is what a *new* user message still matches after
//!   history drops the earlier thinking.
//!
//! **What later slices change here.** #188 (retained prefix) adds a
//! second kind of retained entry, matched against the same prompt; #190
//! (KV-RAM) fills the second [`ResidencyTier`] of the pool's [`TierList`]:
//! [`CheckpointPool::discard_victim`] becomes a *spill* to
//! [`TierList::below`], and [`CheckpointPool::best_match`] already weighs the
//! tiers in order with each one's restore floor. None of those need the shape
//! here to change.

use crate::gdn::GdnState;
use crate::identity::{BlobHeader, BlobIdentity, IdentityMismatch, MatchKey, PromptContent};
use crate::prefix::PrefixId;
use crate::types::RequestId;

/// An opaque handle to a retained prompt checkpoint.
pub type CheckpointId = u64;

/// An opaque handle to a **lineage** (GitHub #187): the chain of prompt
/// checkpoints one conversation left behind.
///
/// A conversation has no name. The clients send no session id, and ADR 0029
/// refuses to invent one — so what ties turn N+1's checkpoint to turn N's is
/// the only link that actually exists: turn N+1 *claimed* turn N's entry.
/// Every capture inherits the lineage of the entry its request resumed from,
/// and a capture that resumed from nothing starts one. That is the whole of
/// "per conversation" here: a conversation is a claim chain, and the pool
/// bounds each chain rather than the pool as a whole.
pub type LineageId = u64;

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

/// One residency tier in the ordered list retained state lives across
/// (CONTEXT.md, "Residency tier"; GitHub #189).
///
/// A tier is a place plus the price of reaching it. The price is what makes
/// the list more than an enum: state on the device is already where it is
/// needed, so the device costs nothing to prefer; a tier below it costs a
/// transfer, and a match there is only worth taking when it reuses enough
/// *more* to pay for the crossing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidencyTier {
    /// Where state in this tier lives.
    pub source: ReuseSource,
    /// Prompt tokens a match here must reuse **beyond** the best match in
    /// every tier above it before it is taken.
    ///
    /// 0 in the first tier, which has nothing above it, and at least 1 in
    /// every tier below — which is also what makes a tie go to the tier above
    /// (ADR 0029). #190 sets KV-RAM's to one prefill chunk.
    pub restore_floor_tokens: u32,
}

/// The ordered list of [`ResidencyTier`]s this load retains state across:
/// device first, then whatever is below it.
///
/// **Why a list and not a branch.** Tier 2 (KV-disk) is prepared, not built
/// (ADR 0029). The thing it would otherwise force a rewrite of is every place
/// that says "device, or else KV-RAM": a third place turns each of those into
/// a three-way branch, and the branches drift. Written over a list, adding a
/// tier is adding an entry — the selection below and the demotion order are
/// already expressed in terms of "the tier above" and "the tier below".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TierList(Vec<ResidencyTier>);

impl Default for TierList {
    fn default() -> Self {
        Self::device_only()
    }
}

impl TierList {
    /// The device alone — today's engine, and what a pool built without a
    /// tier list gets.
    pub fn device_only() -> Self {
        Self(vec![ResidencyTier {
            source: ReuseSource::Device,
            restore_floor_tokens: 0,
        }])
    }

    /// An explicit ordered list, cheapest tier first.
    ///
    /// Panics on a list that could not be a residency order: empty, a tier
    /// named twice, a first tier with a nonzero floor (there is nothing above
    /// it to beat), or a lower tier with a zero floor (which would let it win
    /// a tie against the tier above it, and ties go upward).
    pub fn new(tiers: Vec<ResidencyTier>) -> Self {
        assert!(!tiers.is_empty(), "a residency order needs a tier");
        assert_eq!(
            tiers[0].restore_floor_tokens, 0,
            "the first tier has nothing above it to beat"
        );
        for (rank, tier) in tiers.iter().enumerate() {
            assert!(
                !tiers[..rank].iter().any(|t| t.source == tier.source),
                "{} appears twice in the residency order",
                tier.source.as_str()
            );
            assert!(
                rank == 0 || tier.restore_floor_tokens > 0,
                "a tier below the first must beat it by something"
            );
        }
        Self(tiers)
    }

    /// The tiers, cheapest first.
    pub fn tiers(&self) -> &[ResidencyTier] {
        &self.0
    }

    /// `source`'s position in the order, or `None` when this load does not
    /// carry that tier at all.
    pub fn rank(&self, source: ReuseSource) -> Option<usize> {
        self.0.iter().position(|t| t.source == source)
    }

    /// Where state released by `source` goes next, or `None` to discard it —
    /// the demotion order, and the one question the first-victim path and
    /// #190's spill both ask.
    pub fn below(&self, source: ReuseSource) -> Option<ReuseSource> {
        let rank = self.rank(source)?;
        self.0.get(rank + 1).map(|t| t.source)
    }

    /// Whether a match of `tokens` prompt tokens found in `source` is worth
    /// taking over `best`, the best match so far and where it came from.
    ///
    /// The whole selection policy, in one place: longest reuse wins inside a
    /// tier, and between two tiers the **lower** one has to beat the upper by
    /// its own restore floor — whichever of the two the caller happens to be
    /// holding. A tie therefore always goes upward, and a `source` this list
    /// does not carry never wins anything.
    ///
    /// The two cross-tier branches below are the same rule read from each
    /// end, and they have to stay each other's mirror: a caller that offered
    /// its candidates in one order and a caller that offered them in the
    /// other must choose the same match.
    pub fn replaces(
        &self,
        source: ReuseSource,
        tokens: u32,
        best: Option<(ReuseSource, u32)>,
    ) -> bool {
        if tokens == 0 {
            return false;
        }
        let Some(rank) = self.rank(source) else {
            return false;
        };
        let Some((best_source, best_tokens)) = best else {
            return true;
        };
        match self.rank(best_source) {
            None => true,
            // The candidate is above the standing best: it takes the match
            // unless the best below it clears *its* floor over the candidate.
            Some(best_rank) if rank < best_rank => {
                tokens.saturating_add(self.0[best_rank].restore_floor_tokens) > best_tokens
            }
            Some(best_rank) if rank == best_rank => tokens > best_tokens,
            // The candidate is below: it has to clear its own floor.
            Some(_) => tokens >= best_tokens.saturating_add(self.0[rank].restore_floor_tokens),
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
    /// [`CheckpointEntry::key`], which is computed from prompt content alone
    /// and cannot reach this field.
    pub publisher: RequestId,
    /// The content of the prompt up to and including the generation opener:
    /// the match key (GitHub #189). A request whose prompt's prefix key at
    /// [`CheckpointEntry::tokens`] equals this reuses this entry.
    ///
    /// Sixteen bytes instead of the token ids themselves, which is what lets
    /// a tier below the device name what it holds without holding a
    /// conversation ([`BlobHeader`]).
    pub key: MatchKey,
    /// Prompt tokens the key covers — everything up to the generation opener.
    pub tokens: u32,
    /// What the state was produced under (GitHub #189). Matched against the
    /// pool's own identity before the entry is ever offered to a claimant, so
    /// a blob from another artifact, KV format, blob layout or drafter is
    /// refused rather than written into a sequence.
    pub identity: BlobIdentity,
    /// Which residency tier the entry is in. Every entry a capture produces is
    /// on the device; #190 moves entries down the pool's [`TierList`].
    pub tier: ReuseSource,
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
    /// The conversation this entry belongs to (GitHub #187): inherited from
    /// the entry the capturing request claimed, fresh when it claimed none.
    pub lineage: LineageId,
    /// Whether this entry is a **turn-opening checkpoint**: the first one
    /// captured after its conversation's last real user message (GitHub #187,
    /// ADR 0029).
    ///
    /// It is the only one a *new* user message can still match. Once history
    /// drops the earlier thinking, every checkpoint taken further along in the
    /// turn covers tokens the next render no longer produces — so the pool
    /// keeps this one whatever else it supersedes, and the pair it keeps is
    /// "the latest, for the tool loop; the turn opener, for the human".
    pub turn_opening: bool,
    /// The scheduling tick this entry was last captured or claimed at — the
    /// LRU order [`CheckpointPool::discard_victim`] discards in.
    pub use_tick: u64,
}

impl CheckpointEntry {
    /// What this entry would say about itself to a tier that does not hold its
    /// history: its identity and its content, and nothing that names a request
    /// or a process (GitHub #189).
    pub fn header(&self) -> BlobHeader {
        BlobHeader {
            identity: self.identity,
            key: self.key,
            tokens: self.tokens,
        }
    }
}

/// One checkpoint offered to the pool: everything an entry is made of except
/// the id and the tick the pool assigns.
///
/// A struct rather than nine positional arguments because two of them —
/// [`CheckpointCapture::identity`] and [`CheckpointCapture::tier`] — are the
/// difference between a capture this load just took and a blob adopted from
/// somewhere else, and a caller that has to count commas to tell them apart
/// will eventually get it wrong.
#[derive(Debug, Clone)]
pub struct CheckpointCapture {
    /// The request whose prefill captured it — the backend's handle on the
    /// device image, never part of the key.
    pub publisher: RequestId,
    /// The content key at the generation opener.
    pub key: MatchKey,
    /// Prompt tokens the key covers.
    pub tokens: u32,
    /// The shared-prefix entry carrying the whole KV pages under the opener.
    pub prefix: PrefixId,
    /// Whole KV pages that shared prefix holds.
    pub pages: u32,
    /// Device bytes the entry's own image occupies.
    pub bytes: u64,
    /// The GDN state at the opener.
    pub gdn: GdnState,
    /// What the state was produced under.
    pub identity: BlobIdentity,
    /// The residency tier the entry lands in.
    pub tier: ReuseSource,
    /// The retained entry the capturing request resumed from, or `None` when
    /// it resumed from nothing (GitHub #187). This is what names the
    /// conversation: the new entry joins that entry's lineage, or opens one of
    /// its own.
    pub claimed: Option<CheckpointId>,
    /// Whether this capture opens a new turn — whether a real user message
    /// lies between [`Self::claimed`]'s opener and this one's. `true` for a
    /// capture that claimed nothing (a conversation's first checkpoint is its
    /// first turn's), and `false` when the frontend could not say, because a
    /// wrong `true` retires a lineage's turn opener and sends the next user
    /// message cold.
    pub turn_opening: bool,
}

/// What [`CheckpointPool::retain`] did: the new entry, and the entries of its
/// lineage that giving it a place superseded.
///
/// The superseded entries are **returned rather than dropped** because the
/// pool owns no device memory: their images and their holds on the shared
/// pages below them are the caller's to release, exactly as
/// [`CheckpointPool::discard_victim`]'s are. A `retain` whose result is
/// ignored leaks a device image per tool-call iteration.
#[derive(Debug)]
#[must_use = "the superseded entries still hold a device image and a shared prefix"]
pub struct Retained {
    /// The new entry's id.
    pub id: CheckpointId,
    /// Entries of the same lineage this capture displaced, oldest first.
    pub superseded: Vec<CheckpointEntry>,
}

/// Why the pool would not take a checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetainRefused {
    /// The byte budget cannot hold the image. The ordinary, expected answer:
    /// retention is a bet, and a full pool simply does not take one (ADR
    /// 0029). Nothing is evicted to make room.
    Budget,
    /// The state was produced under an identity this load cannot accept — a
    /// blob from another artifact, KV format, blob layout or drafter. It is
    /// refused here, before a single byte is written into a sequence.
    Identity(IdentityMismatch),
    /// The tier named is not one this load carries.
    UnknownTier(ReuseSource),
}

impl std::fmt::Display for RetainRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RetainRefused::Budget => write!(f, "the retained pool's byte budget is full"),
            RetainRefused::Identity(mismatch) => write!(f, "{mismatch}"),
            RetainRefused::UnknownTier(source) => {
                write!(f, "this load has no {} tier", source.as_str())
            }
        }
    }
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
    identity: BlobIdentity,
    tiers: TierList,
    entries: Vec<CheckpointEntry>,
    next_id: CheckpointId,
    next_lineage: LineageId,
    reused_tok: u64,
    captures: u64,
    skipped_captures: u64,
    refused_blobs: u64,
    discards: u64,
}

impl CheckpointPool {
    /// A pool bounded by `capacity_bytes` of device image, holding state
    /// produced under [`BlobIdentity::UNSET`] on the device alone. `0`
    /// disables retention entirely (every capture is skipped) — a legal,
    /// explicit choice, the way a `0` KV-RAM budget disables the host tier.
    pub fn new(capacity_bytes: u64) -> Self {
        Self::with_identity(capacity_bytes, BlobIdentity::UNSET)
    }

    /// A pool for state produced under `identity`, on the device alone.
    pub fn with_identity(capacity_bytes: u64, identity: BlobIdentity) -> Self {
        Self::with_tiers(capacity_bytes, identity, TierList::device_only())
    }

    /// A pool for state produced under `identity`, across `tiers`.
    pub fn with_tiers(capacity_bytes: u64, identity: BlobIdentity, tiers: TierList) -> Self {
        Self {
            capacity_bytes,
            used_bytes: 0,
            identity,
            tiers,
            entries: Vec::new(),
            next_id: 0,
            next_lineage: 0,
            reused_tok: 0,
            captures: 0,
            skipped_captures: 0,
            refused_blobs: 0,
            discards: 0,
        }
    }

    /// The pool's byte budget.
    pub fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }

    /// The identity state retained here was produced under: what a blob has
    /// to carry to be written into a sequence on this load.
    pub fn identity(&self) -> BlobIdentity {
        self.identity
    }

    /// The residency tiers this pool retains state across, cheapest first.
    pub fn tiers(&self) -> &TierList {
        &self.tiers
    }

    /// Image bytes the retained entries occupy.
    ///
    /// **One budget, the device's** — [`CheckpointPool::capacity_bytes`] is
    /// `--retained-pool-bytes`, and every entry is charged to it whatever tier
    /// it names. That is right while the device is the only tier that holds
    /// anything, and it is the first thing #190 has to split: KV-RAM has its
    /// own budget, and an entry that left the card must give its device bytes
    /// back or the pool will refuse captures for room nothing is using.
    pub fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    /// Retained entries.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// The retained entries, in capture order — what each one covers, what it
    /// costs, which tier it is in and what it was produced under.
    pub fn entries(&self) -> &[CheckpointEntry] {
        &self.entries
    }

    /// The KV pages of history the retained entries cover, counted once per
    /// **prefix** rather than once per entry: two checkpoints taken at
    /// different openers inside the same prompt head hold the same shared
    /// prefix, and its pages are charged to the KV pool exactly once
    /// ([`crate::prefix::PrefixCache`] is what carries that charge).
    ///
    /// **Diagnostic, not the admission machine's number.** Since #187 an entry
    /// may stand on a *chained* prefix, whose reach includes its ancestors' —
    /// so two entries on two links of one chain count the shared links twice
    /// here. What the admission path reads instead is
    /// `ConcreteScheduler::reclaimable_retained_pages`, which sums each
    /// prefix's **own** pages and only for prefixes no live request holds:
    /// pages that would genuinely come back, never a promise that would not.
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

    /// Whether a checkpoint over exactly the content `key` names is already
    /// retained.
    ///
    /// A second capture at the same point would be a second image of the same
    /// state: the same skip for a claimant, paid for twice out of the byte
    /// budget. The caller checks this before it asks the backend to capture,
    /// the way [`crate::prefix::PrefixCache::register`] declines a duplicate
    /// prompt head.
    pub fn holds(&self, key: MatchKey) -> bool {
        self.entries.iter().any(|e| e.key == key)
    }

    /// Whether the byte budget can take one more image of `bytes`.
    ///
    /// This is the whole of "a full pool skips the capture": a capture that
    /// does not fit is **not taken**, and nothing is evicted to make room for
    /// it. Retention is a bet; a bet never costs certain work (ADR 0029).
    pub fn admits(&self, bytes: u64) -> bool {
        bytes > 0 && self.used_bytes.saturating_add(bytes) <= self.capacity_bytes
    }

    /// A content key this load may match against — one whose state was
    /// produced under an identity this pool accepts.
    ///
    /// Refuses before anything is admitted or offered, which is the whole of
    /// "a mismatch is refused before any byte reaches a sequence" (ADR 0029).
    pub fn accepts(&self, header: &BlobHeader) -> Result<(), IdentityMismatch> {
        self.identity.accepts(&header.identity)
    }

    /// Retain a checkpoint: an image this load's backend just captured, or a
    /// blob adopted from a tier that outlives it.
    ///
    /// Returns the new entry's id, or why the pool would not take it. A
    /// [`RetainRefused::Budget`] is the ordinary answer to a full pool — the
    /// caller must have asked [`CheckpointPool::admits`] before it let the
    /// backend capture anything, so it means the capture raced its own budget
    /// rather than that something went wrong. A [`RetainRefused::Identity`] is
    /// the Tier 2 refusal: the bytes exist and this load cannot use them.
    pub fn retain(
        &mut self,
        capture: CheckpointCapture,
        use_tick: u64,
    ) -> Result<Retained, RetainRefused> {
        if let Err(mismatch) = self.identity.accepts(&capture.identity) {
            self.refused_blobs += 1;
            return Err(RetainRefused::Identity(mismatch));
        }
        if self.tiers.rank(capture.tier).is_none() {
            self.refused_blobs += 1;
            return Err(RetainRefused::UnknownTier(capture.tier));
        }
        if !self.admits(capture.bytes) {
            self.skipped_captures += 1;
            return Err(RetainRefused::Budget);
        }
        // GitHub #187. A claimed entry that is already gone — taken by the
        // first-victim path, or superseded by a sibling that finished first —
        // leaves this capture with no conversation to join. It opens one,
        // which is the same answer a first turn gets and for the same reason:
        // there is nothing to supersede either way.
        let inherited = capture
            .claimed
            .and_then(|claimed| self.entries.iter().find(|e| e.id == claimed))
            .map(|e| e.lineage);
        let lineage = match inherited {
            Some(lineage) => lineage,
            None => {
                let fresh = self.next_lineage;
                self.next_lineage += 1;
                fresh
            }
        };
        let id = self.next_id;
        self.next_id += 1;
        self.used_bytes += capture.bytes;
        self.captures += 1;
        self.entries.push(CheckpointEntry {
            id,
            publisher: capture.publisher,
            key: capture.key,
            tokens: capture.tokens,
            identity: capture.identity,
            tier: capture.tier,
            prefix: capture.prefix,
            pages: capture.pages,
            bytes: capture.bytes,
            gdn: capture.gdn,
            lineage,
            turn_opening: capture.turn_opening,
            use_tick,
        });
        Ok(Retained {
            id,
            superseded: self.cut_lineage_back(lineage),
        })
    }

    /// Cut `lineage` back to the two entries ADR 0029 lets a conversation
    /// keep — its latest and its newest turn-opening one — and return the rest
    /// for the caller to release (GitHub #187).
    ///
    /// Newest by id rather than by [`CheckpointEntry::use_tick`]: ids are
    /// assigned in capture order, and what "latest" means here is *furthest
    /// along the conversation*, which a claim refreshing an older entry's LRU
    /// tick must not change.
    fn cut_lineage_back(&mut self, lineage: LineageId) -> Vec<CheckpointEntry> {
        let of_lineage = |e: &&CheckpointEntry| e.lineage == lineage;
        let latest = self.entries.iter().filter(of_lineage).map(|e| e.id).max();
        let opening = self
            .entries
            .iter()
            .filter(of_lineage)
            .filter(|e| e.turn_opening)
            .map(|e| e.id)
            .max();
        let superseded: Vec<CheckpointId> = self
            .entries
            .iter()
            .filter(of_lineage)
            .filter(|e| Some(e.id) != latest && Some(e.id) != opening)
            .map(|e| e.id)
            .collect();
        superseded
            .into_iter()
            .filter_map(|id| self.discard(id))
            .collect()
    }

    /// The best retained checkpoint whose content is a prefix of `prompt`, or
    /// `None` when nothing matches. A pure query: it decides nothing and
    /// changes nothing.
    ///
    /// "Best" is the [`TierList`]'s answer, not this function's: longest reuse
    /// wins inside a tier, a tier above wins a tie, and a tier below has to
    /// clear its own restore floor (ADR 0029). Entries produced under an
    /// identity this load does not accept are not candidates at all.
    ///
    /// It is separate from [`CheckpointPool::record_claim`] because the
    /// scheduler has to weigh this match against a *sibling prefix* match
    /// before it takes either — longest reuse wins, and a losing match must
    /// leave no trace in the LRU order or the reuse counter.
    pub fn best_match(&self, prompt: &PromptContent<'_>) -> Option<CheckpointMatch> {
        let length = prompt.tokens();
        // An entry longer than the prompt cannot be a prefix of it, and one
        // this load would refuse to write must never be offered.
        let candidates: Vec<&CheckpointEntry> = self
            .entries
            .iter()
            .filter(|e| e.tokens <= length && self.identity.accepts(&e.identity).is_ok())
            .collect();
        if candidates.is_empty() {
            return None;
        }
        // One forward pass over the prompt answers every candidate's length —
        // the reason the key is a chain rather than a digest of the whole.
        let lengths: Vec<u32> = candidates.iter().map(|e| e.tokens).collect();
        let keys = prompt.keys_at(&lengths);
        let mut best: Option<&CheckpointEntry> = None;
        for tier in self.tiers.tiers() {
            for (n, entry) in candidates.iter().enumerate() {
                if entry.tier != tier.source || entry.key != keys[n] {
                    continue;
                }
                let standing = best.map(|b| (b.tier, b.tokens));
                if self.tiers.replaces(tier.source, entry.tokens, standing) {
                    best = Some(entry);
                }
            }
        }
        best.map(|best| CheckpointMatch {
            id: best.id,
            publisher: best.publisher,
            source: best.tier,
            prefix: best.prefix,
            tokens: best.tokens,
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
        self.reused_tok += u64::from(entry.tokens);
    }

    /// [`CheckpointPool::best_match`] followed by
    /// [`CheckpointPool::record_claim`], for a caller with nothing to weigh
    /// the match against.
    pub fn claim(&mut self, prompt: &PromptContent<'_>, use_tick: u64) -> Option<CheckpointMatch> {
        let matched = self.best_match(prompt)?;
        self.record_claim(matched.id, use_tick);
        Some(matched)
    }

    /// Discard the least recently used entry and return it, so the caller can
    /// release its image and its shared-prefix claim. `None` when the pool is
    /// empty.
    ///
    /// This is the **first victim** path (ADR 0023 as amended): a live request
    /// that cannot materialize takes retained pages back before admission
    /// considers evicting anybody. Until #190, released means discarded —
    /// where it goes instead is [`TierList::below`]'s answer.
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
        self.discard_victim_in(ReuseSource::Device, prefixes)
    }

    /// [`CheckpointPool::discard_victim_on`], in one named tier.
    ///
    /// The first-victim path is about the **device**: what a live request
    /// needs back is VRAM, and an entry that already left the card frees none
    /// of it. #190 calls this for its own tier when KV-RAM is the budget under
    /// pressure.
    pub fn discard_victim_in(
        &mut self,
        tier: ReuseSource,
        prefixes: Option<&[PrefixId]>,
    ) -> Option<CheckpointEntry> {
        let pos = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.tier == tier)
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

    /// Checkpoints taken, checkpoints the byte budget refused, blobs refused
    /// on their identity, and checkpoints discarded — the counters #190's
    /// per-tier hit / miss / spill / discard reporting is built from.
    pub fn counters(&self) -> CheckpointCounters {
        CheckpointCounters {
            captures: self.captures,
            skipped_captures: self.skipped_captures,
            refused_blobs: self.refused_blobs,
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
    /// Blobs refused on their compatibility identity (GitHub #189): state
    /// produced under another artifact, KV format, blob layout or drafter.
    /// Zero for the life of a process that adopts nothing from outside it,
    /// which is every process until Tier 2 exists.
    pub refused_blobs: u64,
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
    use crate::identity::{ArtifactHash, MediaKey};
    use crate::kv_format::KvFormat;
    use crate::types::TokenId;

    /// A GDN state resumable at `position` (a recorded boundary).
    fn gdn_at(position: usize) -> GdnState {
        let mut gdn = GdnState::new();
        gdn.checkpoint(position);
        gdn
    }

    /// One image's worth of device bytes, for a pool sized in whole images.
    const IMAGE: u64 = 1_000;

    /// A load's identity, distinguishable from another by `byte`.
    fn load(byte: u8) -> BlobIdentity {
        BlobIdentity::of_load(
            ArtifactHash::from_bytes([byte; 32]),
            KvFormat::HqE8_2b,
            None,
            2,
        )
    }

    fn pool_of(images: u64) -> CheckpointPool {
        CheckpointPool::with_identity(IMAGE * images, load(1))
    }

    /// Retain a checkpoint over `tokens`, at `tick`, holding `pages` pages.
    fn retain(
        pool: &mut CheckpointPool,
        publisher: RequestId,
        tokens: &[TokenId],
        pages: u32,
        tick: u64,
    ) -> Result<CheckpointId, RetainRefused> {
        let capture = CheckpointCapture {
            publisher,
            key: PromptContent::text(tokens).key_at(tokens.len() as u32),
            tokens: tokens.len() as u32,
            prefix: publisher, // one prefix per publisher in these tests
            pages,
            bytes: IMAGE,
            gdn: gdn_at(tokens.len()),
            identity: pool.identity(),
            tier: ReuseSource::Device,
            claimed: None,
            turn_opening: true,
        };
        pool.retain(capture, tick).map(|retained| retained.id)
    }

    /// The pool's answer for a text-only prompt.
    fn claim(
        pool: &mut CheckpointPool,
        tokens: &[TokenId],
        tick: u64,
    ) -> Option<CheckpointMatch> {
        pool.claim(&PromptContent::text(tokens), tick)
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
        let matched = claim(&mut pool, &turn_n_plus_1, 2).expect("turn N+1 extends turn N");
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
            claim(&mut pool, &divergent, 2).is_none(),
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
        let matched = claim(&mut pool, &turn3, 3).unwrap();
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
            assert!(claim(&mut pool, &prompt, tick).is_some(), "claim {tick} hits");
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
        assert_eq!(
            retain(&mut pool, 3, &c, 1, 3),
            Err(RetainRefused::Budget),
            "the capture is skipped"
        );
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
        assert_eq!(retain(&mut pool, 1, &a, 1, 1), Err(RetainRefused::Budget));
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
        claim(&mut pool, &prompt, 9).unwrap();
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
        assert!(claim(&mut pool, &prompt, 2).is_some());
        assert_eq!(pool.discard(id).map(|e| e.id), Some(id));
        assert!(claim(&mut pool, &prompt, 3).is_none(), "a discarded entry is gone");
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
        let matched = claim(&mut pool, &prompt, 2).unwrap();
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
        let peek = pool
            .best_match(&PromptContent::text(&prompt))
            .expect("a match");
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
        let key_at = |at: u32| PromptContent::text(&a).key_at(at);
        assert!(pool.holds(key_at(100)));
        assert!(!pool.holds(key_at(99)), "a different point is not held");
    }

    #[test]
    fn retained_pages_count_a_shared_prefix_once() {
        // Two checkpoints taken inside the same prompt head hold the same
        // shared prefix, whose pages the KV pool is charged for exactly once.
        // Summing per entry would promise admission pages it cannot get back.
        let mut pool = pool_of(4);
        let early: Vec<TokenId> = (1..=100).collect();
        let late: Vec<TokenId> = (1..=120).collect();
        for (publisher, tokens) in [(1, early), (2, late)] {
            let capture = CheckpointCapture {
                publisher,
                key: PromptContent::text(&tokens).key_at(tokens.len() as u32),
                tokens: tokens.len() as u32,
                prefix: 42, // both checkpoints stand on one prompt head
                pages: 6,
                bytes: IMAGE,
                gdn: gdn_at(tokens.len()),
                identity: pool.identity(),
                tier: ReuseSource::Device,
                claimed: None,
                turn_opening: true,
            };
            let _ = pool.retain(capture, publisher).unwrap();
        }
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

    // ── GitHub #189: identity, the media slot, and the tier list ─────────

    #[test]
    fn a_blob_taken_under_another_load_is_refused() {
        // The Tier 2 refusal, at the one place a blob can enter the ledger: a
        // restart under `--kv-format bf16`, or a disk tier handing back what
        // yesterday's artifact left. Nothing is retained, so nothing can be
        // matched, so no byte of it ever reaches a sequence.
        let mut pool = pool_of(4);
        let tokens: Vec<TokenId> = (1..=100).collect();
        let foreign = CheckpointCapture {
            publisher: 7,
            key: PromptContent::text(&tokens).key_at(100),
            tokens: 100,
            prefix: 7,
            pages: 1,
            bytes: IMAGE,
            gdn: gdn_at(100),
            identity: load(2),
            tier: ReuseSource::Device,
            claimed: None,
            turn_opening: true,
        };
        let refused = pool.retain(foreign, 1).expect_err("another artifact");
        assert_eq!(
            refused,
            RetainRefused::Identity(
                load(1)
                    .accepts(&load(2))
                    .expect_err("the identities differ")
            )
        );
        assert_eq!(pool.entry_count(), 0, "nothing was retained");
        assert_eq!(pool.used_bytes(), 0, "and nothing was charged for");
        assert_eq!(pool.counters().refused_blobs, 1);
        assert_eq!(
            pool.counters().skipped_captures,
            0,
            "a refusal is not a full pool"
        );
        let prompt: Vec<TokenId> = (1..=140).collect();
        assert!(claim(&mut pool, &prompt, 2).is_none());
    }

    #[test]
    fn an_entry_from_another_load_is_never_matched() {
        // Defence in depth for the same rule. Even if a future tier put an
        // entry in the ledger without going through `retain`, the match
        // filters on the entry's own identity — the field exists on the entry
        // exactly so that being in the pool is not the same as being usable.
        let mut pool = pool_of(4);
        let tokens: Vec<TokenId> = (1..=100).collect();
        retain(&mut pool, 7, &tokens, 1, 1).unwrap();
        let prompt: Vec<TokenId> = (1..=140).collect();
        assert!(claim(&mut pool, &prompt, 2).is_some(), "its own load matches");
        // The same bytes, re-labelled as another load's.
        pool.entries[0].identity = load(2);
        assert!(
            pool.best_match(&PromptContent::text(&prompt)).is_none(),
            "a foreign entry is not a candidate"
        );
    }

    #[test]
    fn the_match_key_does_not_depend_on_who_captured_it() {
        // ADR 0029: keys never contain a `RequestId`. Two requests that
        // reached the same opener leave the same key, and the publisher stays
        // what it is — the backend's handle on the device image.
        let mut pool = pool_of(4);
        let tokens: Vec<TokenId> = (1..=100).collect();
        retain(&mut pool, 7, &tokens, 1, 1).unwrap();
        retain(&mut pool, 99, &tokens, 1, 2).unwrap();
        assert_eq!(
            pool.entries[0].key, pool.entries[1].key,
            "same content, same key"
        );
        assert_ne!(pool.entries[0].publisher, pool.entries[1].publisher);
        assert!(
            pool.holds(pool.entries[0].key),
            "the second capture is a duplicate the caller would decline"
        );
    }

    #[test]
    fn two_prompts_with_equal_tokens_and_different_media_never_match() {
        // Spec §11 (ADR 0029): "I am never answered about a picture I did not
        // send." The two prompts are the same token ids — the placeholders are
        // one token repeated — and differ only in the images behind them.
        let mut pool = pool_of(4);
        let tokens: Vec<TokenId> = (1..=100).collect();
        let mine = [MediaKey {
            begin: 10,
            count: 16,
            digest: [0xAA; 32],
            grid: [1, 8, 8],
        }];
        let yours = [MediaKey {
            digest: [0xBB; 32],
            ..mine[0]
        }];
        let capture = CheckpointCapture {
            publisher: 7,
            key: PromptContent::new(&tokens, &mine).key_at(100),
            tokens: 100,
            prefix: 7,
            pages: 1,
            bytes: IMAGE,
            gdn: gdn_at(100),
            identity: pool.identity(),
            tier: ReuseSource::Device,
            claimed: None,
            turn_opening: true,
        };
        let _ = pool.retain(capture, 1).unwrap();
        let longer: Vec<TokenId> = (1..=140).collect();
        assert!(
            pool.claim(&PromptContent::new(&longer, &yours), 2).is_none(),
            "another image is another conversation"
        );
        assert!(
            pool.claim(&PromptContent::text(&longer), 3).is_none(),
            "and no image at all is a third one"
        );
        assert!(
            pool.claim(&PromptContent::new(&longer, &mine), 4).is_some(),
            "the same image still matches"
        );
    }

    #[test]
    fn a_device_only_load_has_nothing_below_the_device() {
        let pool = pool_of(4);
        let tiers = pool.tiers();
        assert_eq!(tiers.tiers().len(), 1);
        assert_eq!(tiers.rank(ReuseSource::Device), Some(0));
        assert_eq!(tiers.rank(ReuseSource::KvRam), None);
        assert_eq!(
            tiers.below(ReuseSource::Device),
            None,
            "until #190, released means discarded"
        );
    }

    #[test]
    fn a_second_tier_is_an_entry_in_the_list() {
        // What #190 adds, and the whole of what it has to add: KV-RAM below
        // the device with a restore floor. The policy below is already
        // written over the list.
        let tiers = TierList::new(vec![
            ResidencyTier {
                source: ReuseSource::Device,
                restore_floor_tokens: 0,
            },
            ResidencyTier {
                source: ReuseSource::KvRam,
                restore_floor_tokens: 1024,
            },
        ]);
        assert_eq!(tiers.rank(ReuseSource::KvRam), Some(1));
        assert_eq!(tiers.below(ReuseSource::Device), Some(ReuseSource::KvRam));
        assert_eq!(tiers.below(ReuseSource::KvRam), None, "the last tier discards");
    }

    #[test]
    fn a_lower_tier_has_to_clear_its_restore_floor() {
        let tiers = TierList::new(vec![
            ResidencyTier {
                source: ReuseSource::Device,
                restore_floor_tokens: 0,
            },
            ResidencyTier {
                source: ReuseSource::KvRam,
                restore_floor_tokens: 1024,
            },
        ]);
        let device_best = Some((ReuseSource::Device, 4_000));
        assert!(
            !tiers.replaces(ReuseSource::KvRam, 4_000, device_best),
            "a tie goes to the device"
        );
        assert!(
            !tiers.replaces(ReuseSource::KvRam, 5_023, device_best),
            "one token short of a chunk is not worth the crossing"
        );
        assert!(tiers.replaces(ReuseSource::KvRam, 5_024, device_best));
        // Read from the other end it is the same rule, and it has to be: a
        // caller that found the KV-RAM match first must choose what a caller
        // that found the device match first chooses. The device takes the tie,
        // and keeps the match until KV-RAM's lead is worth the crossing.
        for (device_tokens, device_wins) in
            [(5_024, true), (4_001, true), (4_000, false), (3_000, false)]
        {
            assert_eq!(
                tiers.replaces(
                    ReuseSource::Device,
                    device_tokens,
                    Some((ReuseSource::KvRam, 5_024))
                ),
                device_wins,
                "{device_tokens} on the device against 5024 in KV-RAM"
            );
            assert_eq!(
                tiers.replaces(
                    ReuseSource::KvRam,
                    5_024,
                    Some((ReuseSource::Device, device_tokens))
                ),
                !device_wins,
                "the same comparison offered the other way round"
            );
        }
        // Inside one tier it is simply the longest reuse.
        assert!(tiers.replaces(ReuseSource::Device, 4_001, device_best));
        assert!(!tiers.replaces(ReuseSource::Device, 4_000, device_best));
        // Nothing beats nothing, and a tier this load does not carry never
        // wins.
        assert!(tiers.replaces(ReuseSource::Device, 1, None));
        assert!(!tiers.replaces(ReuseSource::Device, 0, None));
        assert!(!TierList::device_only().replaces(ReuseSource::KvRam, 9_000, None));
    }

    #[test]
    fn the_first_victim_is_taken_from_the_device() {
        // The first-victim path is about VRAM: an entry that already left the
        // card frees none of it, so it is not a candidate however old it is.
        let tiers = TierList::new(vec![
            ResidencyTier {
                source: ReuseSource::Device,
                restore_floor_tokens: 0,
            },
            ResidencyTier {
                source: ReuseSource::KvRam,
                restore_floor_tokens: 1024,
            },
        ]);
        let mut pool = CheckpointPool::with_tiers(IMAGE * 4, load(1), tiers);
        let spilled: Vec<TokenId> = (1..=100).collect();
        let resident: Vec<TokenId> = (500..=600).collect();
        for (publisher, tokens, tier) in [
            (1, &spilled, ReuseSource::KvRam),
            (2, &resident, ReuseSource::Device),
        ] {
            let capture = CheckpointCapture {
                publisher,
                key: PromptContent::text(tokens).key_at(tokens.len() as u32),
                tokens: tokens.len() as u32,
                prefix: publisher,
                pages: 1,
                bytes: IMAGE,
                gdn: gdn_at(tokens.len()),
                identity: pool.identity(),
                tier,
                claimed: None,
                turn_opening: true,
            };
            let _ = pool.retain(capture, publisher).unwrap();
        }
        let victim = pool.discard_victim().expect("a device victim");
        assert_eq!(
            victim.publisher, 2,
            "the older entry is in KV-RAM and frees no device page"
        );
        assert!(pool.discard_victim().is_none());
        assert_eq!(
            pool.discard_victim_in(ReuseSource::KvRam, None)
                .map(|e| e.publisher),
            Some(1),
            "#190's own budget asks for its own tier"
        );
    }

    #[test]
    fn a_tier_this_load_does_not_carry_is_refused() {
        let mut pool = pool_of(4);
        let tokens: Vec<TokenId> = (1..=100).collect();
        let capture = CheckpointCapture {
            publisher: 7,
            key: PromptContent::text(&tokens).key_at(100),
            tokens: 100,
            prefix: 7,
            pages: 1,
            bytes: IMAGE,
            gdn: gdn_at(100),
            identity: pool.identity(),
            tier: ReuseSource::KvRam,
            claimed: None,
            turn_opening: true,
        };
        assert_eq!(
            pool.retain(capture, 1).err(),
            Some(RetainRefused::UnknownTier(ReuseSource::KvRam))
        );
        assert_eq!(pool.entry_count(), 0);
    }

    #[test]
    fn an_entry_names_itself_to_a_tier_that_does_not_hold_its_history() {
        // The Tier 2 seam: sixteen bytes of key and the identity it was taken
        // under, and nothing that names a request or a process.
        let mut pool = pool_of(4);
        let tokens: Vec<TokenId> = (1..=100).collect();
        retain(&mut pool, 7, &tokens, 1, 1).unwrap();
        let header = pool.entries[0].header();
        assert_eq!(header.tokens, 100);
        assert_eq!(header.key, PromptContent::text(&tokens).key_at(100));
        assert_eq!(header.identity, load(1));
        assert_eq!(pool.accepts(&header), Ok(()));
        let stale = BlobHeader {
            identity: load(2),
            ..header
        };
        assert!(pool.accepts(&stale).is_err());
    }
}
