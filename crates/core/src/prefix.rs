//! Sibling prefix caching — `core-07`.
//!
//! Concurrent requests sharing a prompt prefix skip the redundant
//! prefill: the first request to prefill a prefix registers its prompt
//! (truncated to whole KV pages) in the prefix cache; every sibling that
//! shares the prefix **claims** the cached entry (one reference each) and
//! prefills only its tail. When the last claimant completes (or is
//! re-queued), the entry is dropped and its pages return to the pool.
//!
//! The cache honors the GDN boundary (core-02): a prefix is reusable only
//! at a recorded checkpoint / frontier boundary, so registration rejects
//! a GDN position that is mid-prefill (a mid-prefill position is not
//! resumable for GDN layers).
//!
//! The `sibling_prefix_reused_tok` counter (design §5; exposed via
//! telemetry by `server-02`) accumulates every prompt token a sibling
//! skipped through a cached prefix. The concrete scheduler drives the
//! cache and exposes the counter for the telemetry writer.
//!
//! Retention (v1): an entry lives while at least one live request
//! references it (the registrant counts as a claimant). A budgeted LRU
//! retention of *unused* entries (caching past liveness) is a documented
//! v1.1+ extension — v1 drops an entry as soon as its last claimant is
//! gone, so the cache never pins pages no live request needs.
//!
//! **What this file is, since P4-10 (GitHub #126, ADR 0024).** The pages a
//! prefix covers are owned by the *leaf*, which shares them in place between
//! claimants and frees them when the last holder releases. So the refcount
//! here is admission accounting rather than truth — the two parallel ledgers
//! v1 kept over the same pages are now one, and this one answers a different
//! question: what the pool is charged, and which entry a prompt matches.
//! What makes the match reach the device is [`PrefixEntry::publisher`], the
//! request whose prefill produced the pages: every claim carries it to the
//! backend, which keys the leaf's prefix by it.

use std::time::Instant;

use crate::gdn::GdnState;
use crate::identity::{MatchKey, PromptContent};
use crate::types::{RequestClass, RequestId};

/// An opaque handle to a cached prefix entry (a claimant's reference to a
/// shared prefix).
pub type PrefixId = u64;

/// An opaque handle to a retained prefix whose blob lives in KV-RAM (GitHub
/// #190).
pub type SpilledPrefixId = u64;

/// What keeps a **retained prefix** alive with no live claimant (GitHub #188),
/// and what KV-RAM ranks it by once the device gives it up (#190).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Retention {
    /// The scheduling tick it was retained or last claimed at: the device's
    /// LRU order.
    pub at: u64,
    /// The class of the request that published it.
    pub class: RequestClass,
    /// When it was retained or last claimed, in wall time.
    pub used_at: Instant,
}

/// A retained prefix whose materialized blob is in KV-RAM (GitHub #190): what
/// a prompt has to match to bring it back, and what it comes back as.
#[derive(Debug, Clone)]
pub struct SpilledPrefix {
    pub id: SpilledPrefixId,
    /// The backend's name for the blob, with [`SpilledPrefix::length_tokens`]
    /// — the name the prefix it came from was published under, and the one it
    /// is published under again.
    pub publisher: RequestId,
    /// What the prefix is: the content key of its head (GitHub #193).
    pub key: MatchKey,
    pub length_tokens: u32,
    pub gdn: GdnState,
    pub class: RequestClass,
    /// The device entry that is this same prefix, while one exists — spilled
    /// and still being released, or brought back. A prompt matches that entry
    /// on the device instead, and giving it up again copies nothing.
    pub on_device: Option<PrefixId>,
}

/// A cached sibling prefix: the shared KV pages of a prompt head (whole
/// pages only — a partial page cannot be split between two requests), its
/// GDN state at the prefix's end (the reusable boundary, core-02), and
/// the reference count of live claimants.
#[derive(Debug, Clone)]
pub struct PrefixEntry {
    /// The entry's opaque id (a claimant's handle).
    pub id: PrefixId,
    /// The request whose prefill published this prefix. The compute backend
    /// keys the *leaf's* prefix — the one that actually owns the pages — by
    /// this id (P4-10, GitHub #126), so a claim has to carry it through.
    pub publisher: RequestId,
    /// What the cached prefix *is*: the content key of the shared prompt head
    /// (page-aligned) — its token ids and the media items inside it (GitHub
    /// #193, the #189 key). Token ids alone would let two prompts differing
    /// only in their pictures share one head, since every placeholder of
    /// every image is the same token.
    pub key: MatchKey,
    /// Tokens in the cached prefix (= `total_pages * page_tokens`).
    pub length_tokens: u32,
    /// The KV pages this entry itself holds in the pool — charged exactly
    /// once, for every claimant.
    ///
    /// Its **own** pages, not its history's: a chained entry (GitHub #187)
    /// covers its parent's pages too, and those are the parent's charge. So
    /// this is what comes back when the entry drops, and
    /// [`PrefixCache::total_pages_of`] is what a claimant shares.
    pub pages: u32,
    /// The entry this one extends (GitHub #187), or `None` for one that
    /// covers its whole head itself.
    ///
    /// A request that resumed from retained state and then prefilled past it
    /// has no head of its own to publish: the pages below its generation
    /// opener are partly the entry it claimed. So it publishes a **chained**
    /// entry — its own new pages, plus the reference on its parent that it
    /// was holding itself until the publish. That is what lets every
    /// iteration of a tool loop leave a checkpoint instead of only the first
    /// (ADR 0029; the leaf's `ignis_seq_prefix::parent` is the same idea).
    pub parent: Option<PrefixId>,
    /// The GDN state at the prefix's end (the resumable boundary, core-02;
    /// a claimant seeds its recurrent state from this).
    pub gdn: GdnState,
    /// Live claimants (the registrant counts as one; the entry is dropped
    /// when the last claimant releases).
    pub refcount: u32,
    /// What keeps this entry alive as a **retained prefix** (GitHub #188, ADR
    /// 0029); `None` for an ordinary sibling prefix, which dies with its
    /// claimants.
    ///
    /// One of [`PrefixEntry::refcount`]'s references is the retention itself
    /// while this is `Some`, which is what carries the entry past its
    /// publisher's completion — and why the flag lives beside the refcount
    /// rather than in a second ledger that could drift from it.
    pub retained: Option<Retention>,
}

/// The result of a successful claim: the entry's id, the leading prompt
/// tokens skipped (the matched prefix), the pages the entry holds (the
/// caller's charge split — its own reservation shrinks by these), and the
/// GDN state to seed the claimant's recurrent state from (core-02:
/// resume at the shared boundary).
#[derive(Debug, Clone)]
pub struct PrefixClaim {
    /// The claimed entry's id.
    pub id: PrefixId,
    /// The request that published the entry (P4-10, GitHub #126): what the
    /// prefill job carries so the backend allocates the claimant's sequence
    /// against the leaf's prefix rather than a fresh one.
    pub publisher: RequestId,
    /// The leading prompt tokens skipped via the cached prefix.
    pub tokens: u32,
    /// The pages of the whole head the entry covers, its chain included
    /// (GitHub #187) — the caller reduces its own reservation by these, since
    /// every one of them is a page it now shares rather than reserves.
    pub pages: u32,
    /// The claimant's seeded GDN state (resumed at the cached boundary).
    pub gdn: GdnState,
}

/// The sibling prefix cache (core-07): concurrent requests sharing a
/// prompt prefix skip the redundant prefill.
///
/// See the module docs for the retention model (v1: liveness-based).
pub struct PrefixCache {
    /// The KV page size in tokens (sets the shareable boundary: prefixes
    /// are truncated to whole pages).
    page_tokens: u32,
    entries: Vec<PrefixEntry>,
    next_id: PrefixId,
    /// Retained prefixes whose blob is in KV-RAM (GitHub #190).
    spilled: Vec<SpilledPrefix>,
    next_spilled: SpilledPrefixId,
    /// The cumulative `sibling_prefix_reused_tok` counter (telemetry,
    /// design §5): every prompt token skipped through a cached prefix.
    reused_tok: u64,
}

impl PrefixCache {
    /// A cache with KV pages of `page_tokens` tokens each.
    pub fn new(page_tokens: u32) -> Self {
        assert!(page_tokens > 0, "KV pages must hold tokens");
        Self {
            page_tokens,
            entries: Vec::new(),
            next_id: 0,
            spilled: Vec::new(),
            next_spilled: 0,
            reused_tok: 0,
        }
    }

    /// The KV page size in tokens.
    pub fn page_tokens(&self) -> u32 {
        self.page_tokens
    }

    /// Register a request's prefilled prompt as a reusable prefix
    /// (core-07): the prompt is truncated to a whole-page boundary (a
    /// partial page cannot be split between two requests) and cached with
    /// the request's GDN state at the prefix's end. The registrant counts
    /// as the first claimant (refcount 1); the entry's pages are charged
    /// to the pool by the caller (the charge split: the registrant's own
    /// reservation shrinks by `pages`).
    ///
    /// Returns the entry's id and its page count on success. Skipped
    /// (returns `None`) when the prompt holds fewer than one page
    /// (nothing shareable), the prefix is already cached (a duplicate
    /// prompt — the caller claims the existing entry instead), or the GDN
    /// state is not at a recorded boundary (core-02: a mid-prefill
    /// position is not resumable, so the prefix is not reusable).
    ///
    /// `publisher` is the request whose prefill produced the pages. P4-10
    /// (GitHub #126) made that identity load-bearing rather than
    /// bookkeeping: the leaf's prefix — which actually owns the physical
    /// pages — is keyed by it, so every claim carries it to the backend.
    /// `parent` (GitHub #187) is the entry the publisher was standing on, or
    /// `None` for one publishing a head it warmed entirely itself. A chained
    /// entry owns only the pages past its parent's end, and **inherits the
    /// publisher's reference on the parent** rather than taking a new one:
    /// from here on the publisher reaches the parent through this entry, so
    /// the chain is held by exactly one reference per link and the pages
    /// under it are still charged once. It is refused when it would cover
    /// nothing its parent does not.
    ///
    /// `head` is the published head with its media (GitHub #193), and it is
    /// also refused when the page floor lands **inside** a media item: half an
    /// image's placeholders would be shared, and a claimant's prefill would
    /// start in the middle of an item whose encode it never runs. The
    /// scheduler never publishes there (`RequestInput::prefix_floor`); this is
    /// the ledger refusing to record it if it ever did.
    pub fn register<'a>(
        &mut self,
        publisher: RequestId,
        head: impl Into<PromptContent<'a>>,
        gdn: &GdnState,
        parent: Option<PrefixId>,
    ) -> Option<(PrefixId, u32)> {
        let head = head.into();
        let length = (head.tokens() / self.page_tokens) * self.page_tokens;
        if length == 0 || head.splits_media(length) {
            return None; // fewer than one page, or an image cut in two
        }
        self.insert(publisher, head.key_at(length), length, gdn, parent)
    }

    /// Register a retained prefix brought back from KV-RAM (GitHub #190) under
    /// the key and length it was spilled with. It has no parent: a blob is a
    /// whole head.
    pub fn register_returned(&mut self, spilled: &SpilledPrefix) -> Option<(PrefixId, u32)> {
        self.insert(
            spilled.publisher,
            spilled.key,
            spilled.length_tokens,
            &spilled.gdn,
            None,
        )
    }

    /// [`PrefixCache::register`] once the head is named: `length` whole pages
    /// of content `key`.
    fn insert(
        &mut self,
        publisher: RequestId,
        key: MatchKey,
        length: u32,
        gdn: &GdnState,
        parent: Option<PrefixId>,
    ) -> Option<(PrefixId, u32)> {
        if length == 0 || length % self.page_tokens != 0 {
            return None; // fewer than one page: nothing shareable
        }
        let below = match parent {
            Some(parent) => {
                let pages = self.total_pages_of(parent);
                if !self.contains(parent) || pages * self.page_tokens >= length {
                    return None;
                }
                pages
            }
            None => 0,
        };
        // core-02: a prefix is reusable only at a recorded GDN boundary
        // (a mid-prefill position is not resumable for GDN layers).
        if !gdn.is_valid_snapshot_point(gdn.position()) {
            return None;
        }
        // A duplicate prompt (the same truncated prefix is already
        // cached): the caller claims the existing entry — no second set
        // of pages.
        if self.find(key, length).is_some() {
            return None;
        }
        let entry = PrefixEntry {
            id: self.next_id,
            publisher,
            key,
            length_tokens: length,
            pages: length / self.page_tokens - below,
            parent,
            gdn: gdn.clone(),
            refcount: 1, // the registrant is the first claimant
            retained: None,
        };
        let (id, pages) = (entry.id, entry.pages);
        self.next_id += 1;
        self.entries.push(entry);
        Some((id, pages))
    }

    /// Claim the longest cached prefix of `tokens` (page-aligned): bumps
    /// the entry's refcount (the claimant pins the shared pages for its
    /// lifetime) and records the skipped tokens in the
    /// `sibling_prefix_reused_tok` counter. Returns the claim (the
    /// entry's id, the matched token count, the entry's pages, and the
    /// GDN state to resume from), or `None` when nothing cached matches
    /// the prompt head.
    pub fn claim<'a>(&mut self, prompt: impl Into<PromptContent<'a>>) -> Option<PrefixClaim> {
        self.claim_longer_than(prompt, 0)
    }

    /// [`PrefixCache::claim`], restricted to entries covering **more** than
    /// `floor` tokens.
    ///
    /// GitHub #186: a request may also match a retained prompt checkpoint,
    /// which reaches further into its prompt than any page-aligned prefix of
    /// the same head. Longest reuse wins (ADR 0029), so the checkpoint's span
    /// is the floor this has to beat — and it has to be a floor rather than a
    /// comparison made afterwards, because claiming here is not free: it pins
    /// the entry and counts the skip.
    pub fn claim_longer_than<'a>(
        &mut self,
        prompt: impl Into<PromptContent<'a>>,
        floor: u32,
    ) -> Option<PrefixClaim> {
        let match_entry = self.longest_match(prompt.into(), floor)?;
        let id = match_entry.id;
        let claim = PrefixClaim {
            id,
            publisher: match_entry.publisher,
            tokens: match_entry.length_tokens,
            // The whole head, chain included (GitHub #187): what the claimant
            // shares is every page below its own first, whichever entry of
            // the chain happens to own each one.
            pages: self.total_pages_of(id),
            gdn: match_entry.gdn.clone(),
        };
        if let Some(e) = self.entries.iter_mut().find(|e| e.id == id) {
            e.refcount += 1;
        }
        self.reused_tok += u64::from(claim.tokens);
        Some(claim)
    }

    /// The leading tokens of `tokens` the longest registered prefix would let
    /// a claimant skip, or 0 — without claiming it (GitHub #190). What a
    /// KV-RAM checkpoint has to beat by its restore floor before a restore
    /// across the bus is worth more than sharing pages already on the device.
    pub fn longest_match_tokens<'a>(&self, prompt: impl Into<PromptContent<'a>>) -> u32 {
        self.longest_match(prompt.into(), 0)
            .map_or(0, |e| e.length_tokens)
    }

    /// The longest registered prefix of `prompt` reaching past `floor`.
    ///
    /// What matches is the entry's **content key** at its length (GitHub
    /// #193): the prompt's token ids *and* the media items inside that head.
    /// Only entries no longer than the prompt are asked about — `keys_at`
    /// answers a length past the end with the whole prompt's key, which must
    /// never stand in for a longer entry's.
    fn longest_match(&self, prompt: PromptContent<'_>, floor: u32) -> Option<&PrefixEntry> {
        let candidates = self.entries.iter().filter(|e| e.length_tokens > floor);
        longest_keyed(prompt, candidates, |e| (e.key, e.length_tokens))
    }

    /// Take one more reference to `entry` on behalf of something that is not
    /// a live request (GitHub #186): a retained prompt checkpoint holds the
    /// whole KV pages under its opener through the very prefix the capturing
    /// request published, so the pages outlive every live claimant without
    /// being charged to the pool a second time.
    ///
    /// Unlike [`PrefixCache::claim`] this counts no skipped tokens — nothing
    /// was prefilled or skipped, the reference is pure retention — and it
    /// matches no prompt: the caller already knows which entry it means.
    /// Returns `false` when the entry is gone.
    pub fn retain(&mut self, entry: PrefixId) -> bool {
        match self.entries.iter_mut().find(|e| e.id == entry) {
            Some(e) => {
                e.refcount += 1;
                true
            }
            None => false,
        }
    }

    /// Keep `entry` alive with no live claimant — make it a **retained
    /// prefix** (GitHub #188, ADR 0029).
    ///
    /// The retention takes one reference of its own, so the entry survives
    /// the request that published it and the next subagent of a burst claims
    /// it exactly as a concurrent sibling would. The spec says the prefix
    /// "becomes retained at refcount 0"; taking the reference at the publish
    /// instead is the same fact with no window in it — the entry is never one
    /// release away from a drop that would have to be undone.
    ///
    /// `retention` seeds the LRU order [`PrefixCache::lru_retained`] gives
    /// entries up in, and what KV-RAM ranks the prefix by if it gets there.
    /// Returns `false` when the entry is gone or is already retained: a second
    /// retention would be a second reference nothing ever releases, and the
    /// pages would never come back.
    pub fn retain_published(&mut self, entry: PrefixId, retention: Retention) -> bool {
        match self.entries.iter_mut().find(|e| e.id == entry) {
            Some(e) if e.retained.is_none() => {
                e.retained = Some(retention);
                e.refcount += 1;
                true
            }
            _ => false,
        }
    }

    /// Whether `entry` is a **retained prefix** (GitHub #188): kept alive by
    /// a retention reference rather than by a live request.
    pub fn is_retained(&self, entry: PrefixId) -> bool {
        self.entries
            .iter()
            .any(|e| e.id == entry && e.retained.is_some())
    }

    /// Give up the retention on `entry`, so its reference is the caller's to
    /// release (GitHub #188): the first-victim path.
    ///
    /// Returns `false` when the entry is gone or was never retained — the
    /// caller must not then release a reference the cache does not hold, which
    /// would drop the entry out from under a live claimant. It is two steps
    /// rather than one because releasing is the *scheduler's* act: only it
    /// knows what the freed pages are charged against and what handle the
    /// backend still holds.
    pub fn unretain(&mut self, entry: PrefixId) -> bool {
        match self.entries.iter_mut().find(|e| e.id == entry) {
            Some(e) if e.retained.is_some() => {
                e.retained = None;
                true
            }
            _ => false,
        }
    }

    /// Move `entry` to the back of the retained LRU order, if it is retained
    /// (GitHub #188). A no-op on anything else, because a claimant cannot tell
    /// a retained entry from a live sibling's and must not have to.
    pub fn touch_retained(&mut self, entry: PrefixId, tick: u64, used_at: Instant) {
        if let Some(retention) = self
            .entries
            .iter_mut()
            .find(|e| e.id == entry)
            .and_then(|e| e.retained.as_mut())
        {
            retention.at = tick;
            retention.used_at = used_at;
        }
    }

    /// The retained prefixes whose pages would actually come back if the
    /// retention were given up: those whose only holder *is* the retention
    /// (GitHub #188).
    ///
    /// The same question [`PrefixCache::refcount_of`] answers for a retained
    /// checkpoint, asked about the other kind of retained state: a prefix a
    /// live request is still standing on frees nothing when its retention
    /// goes, so giving it up would cost the next burst its reuse and buy this
    /// request no page.
    pub fn reclaimable_retained(&self) -> Vec<PrefixId> {
        self.entries
            .iter()
            .filter(|e| e.retained.is_some() && e.refcount == 1)
            .map(|e| e.id)
            .collect()
    }

    /// The least recently used reclaimable retained prefix — the first-victim
    /// path's choice among them (GitHub #188), or `None` when nothing can be
    /// given up.
    pub fn lru_retained(&self) -> Option<PrefixId> {
        self.entries
            .iter()
            .filter(|e| e.retained.is_some() && e.refcount == 1)
            .min_by_key(|e| (e.retained.map(|r| r.at), e.id))
            .map(|e| e.id)
    }

    /// The entry `id`'s live holders, or 0 when it is gone (GitHub #186).
    ///
    /// The pages come back only at zero, so this is what answers "would
    /// letting go of this actually free anything" — the question the
    /// first-victim path has to ask before it gives up a bet, since a prefix
    /// a live request is still standing on frees nothing when its retained
    /// holder releases.
    pub fn refcount_of(&self, entry: PrefixId) -> u32 {
        self.entries
            .iter()
            .find(|e| e.id == entry)
            .map_or(0, |e| e.refcount)
    }

    /// The entry `id`'s **own** pages, or 0 when it is gone — the pages that
    /// come back to the pool when it drops.
    ///
    /// Not what it covers: a chained entry's parent's pages come back only
    /// when the parent does. The first-victim path reads this one, so that
    /// giving up a retained entry never promises admission pages another link
    /// of the chain is still holding (GitHub #186, #187).
    pub fn pages_of(&self, entry: PrefixId) -> u32 {
        self.entries
            .iter()
            .find(|e| e.id == entry)
            .map_or(0, |e| e.pages)
    }

    /// The pages of the whole head `entry` covers — its own and every
    /// ancestor's (GitHub #187), or 0 when it is gone.
    ///
    /// This is what a claimant shares and what a retained checkpoint records
    /// as the history below its opener: both are about *how much is warm*,
    /// which the chain answers together, not about who gives what back.
    pub fn total_pages_of(&self, entry: PrefixId) -> u32 {
        let mut pages = 0;
        let mut at = Some(entry);
        while let Some(id) = at {
            let Some(e) = self.entries.iter().find(|e| e.id == id) else {
                break;
            };
            pages += e.pages;
            at = e.parent;
        }
        pages
    }

    /// Release one reference to `entry` (the claimant completed or was
    /// re-queued). When the last reference drops, the entry (and its own
    /// pages) is released.
    ///
    /// Returns every entry that dropped as a result — the pages each freed,
    /// **the request that published them and the head's length**, so the
    /// caller can drop the backend's handle on that leaf prefix too (P4-10;
    /// one request may publish two heads, #187 x #188, so the publisher alone
    /// does not name one) — innermost first,
    /// and empty while the entry is still pinned by other claimants. It is a
    /// list rather than one entry because a chained entry (GitHub #187) holds
    /// its parent's reference: letting go of the child may let go of the
    /// parent, and of its parent in turn.
    pub fn release(&mut self, entry: PrefixId) -> Vec<(u32, RequestId, u32)> {
        let mut dropped = Vec::new();
        let mut at = Some(entry);
        while let Some(id) = at {
            let Some(pos) = self.entries.iter().position(|e| e.id == id) else {
                break;
            };
            self.entries[pos].refcount -= 1;
            if self.entries[pos].refcount != 0 {
                break;
            }
            let entry = self.entries.remove(pos);
            dropped.push((entry.pages, entry.publisher, entry.length_tokens));
            // A KV-RAM copy of it stays, and is what a prompt matches now.
            for spilled in self.spilled.iter_mut().filter(|s| s.on_device == Some(entry.id)) {
                spilled.on_device = None;
            }
            at = entry.parent;
        }
        dropped
    }

    /// The entry `id`, if it is live.
    pub fn entry(&self, id: PrefixId) -> Option<&PrefixEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    // ── retained prefixes in KV-RAM (GitHub #190) ────────────────────────

    /// Record that retained entry `entry`'s blob was written to KV-RAM. It
    /// stays linked to the device entry until that entry drops. `None` when
    /// the entry is gone or is not retained.
    pub fn record_spill(&mut self, entry: PrefixId) -> Option<SpilledPrefixId> {
        let e = self.entries.iter().find(|e| e.id == entry)?;
        let retention = e.retained?;
        let id = self.next_spilled;
        self.next_spilled += 1;
        self.spilled.push(SpilledPrefix {
            id,
            publisher: e.publisher,
            key: e.key,
            length_tokens: e.length_tokens,
            gdn: e.gdn.clone(),
            class: retention.class,
            on_device: Some(entry),
        });
        Some(id)
    }

    /// The KV-RAM copy of device entry `entry`, if it has one.
    pub fn spilled_copy_of(&self, entry: PrefixId) -> Option<SpilledPrefixId> {
        self.spilled
            .iter()
            .find(|s| s.on_device == Some(entry))
            .map(|s| s.id)
    }

    /// The longest prefix of `prompt` held **only** in KV-RAM — one still on
    /// the device is matched there — or `None`. Matched by content key, as
    /// the device entries are (GitHub #193).
    pub fn longest_spilled_match<'a>(
        &self,
        prompt: impl Into<PromptContent<'a>>,
    ) -> Option<&SpilledPrefix> {
        let candidates = self.spilled.iter().filter(|s| s.on_device.is_none());
        longest_keyed(prompt.into(), candidates, |s| (s.key, s.length_tokens))
    }

    /// The spilled prefix `id`, if its blob is still held.
    pub fn spilled(&self, id: SpilledPrefixId) -> Option<&SpilledPrefix> {
        self.spilled.iter().find(|s| s.id == id)
    }

    /// Link spilled prefix `id` to the device entry it was brought back as.
    pub fn record_return(&mut self, id: SpilledPrefixId, entry: PrefixId) {
        if let Some(spilled) = self.spilled.iter_mut().find(|s| s.id == id) {
            spilled.on_device = Some(entry);
        }
    }

    /// Forget spilled prefix `id` — its blob is being freed.
    pub fn forget_spilled(&mut self, id: SpilledPrefixId) -> Option<SpilledPrefix> {
        let pos = self.spilled.iter().position(|s| s.id == id)?;
        Some(self.spilled.remove(pos))
    }

    /// Retained prefixes with a blob in KV-RAM.
    pub fn spilled_count(&self) -> usize {
        self.spilled.len()
    }

    /// Whether the cache holds a live entry for `entry` (a claimant's
    /// reference is still valid).
    pub fn contains(&self, entry: PrefixId) -> bool {
        self.entries.iter().any(|e| e.id == entry)
    }

    /// The pages held by the cache (live entries' pages — these pages are
    /// in the pool while the entries live; the pool's accounting counts
    /// them once, for every claimant).
    pub fn pinned_pages(&self) -> u32 {
        self.entries.iter().map(|e| e.pages).sum()
    }

    /// The cumulative `sibling_prefix_reused_tok` counter (telemetry,
    /// design §5): every prompt token skipped through a cached prefix.
    ///
    /// Cumulative skips, not net savings. A request that claims a prefix and
    /// is later evicted and re-queued re-prefills that head from scratch
    /// (`Request::requeue`), and this counter is not reduced: the skip
    /// happened, and eviction is a separate event with its own cost. Read it
    /// as "prefill the cache let requests avoid at the time", not as "prefill
    /// this engine never performed".
    pub fn reused_tok(&self) -> u64 {
        self.reused_tok
    }

    /// The number of live entries (cached prefixes with ≥ 1 claimant).
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// The shareable head of a prompt of `prompt_tokens` tokens: the whole
    /// pages of it, which is what a prefix may ever cover.
    ///
    /// The scheduler needs this **before** it prefills (P4-10, GitHub #126),
    /// not after: the leaf publishes a prefix at the chunk boundary that
    /// lands on it, because what a claimant clones is the mutable state at
    /// the prefix's end. So the chunk decomposition has to be told where that
    /// **publish point** is, and this is where it comes from. 0 means
    /// "nothing shareable" — a prompt shorter than one page.
    pub fn shareable_head_tokens(&self, prompt_tokens: usize) -> u32 {
        ((prompt_tokens / self.page_tokens as usize) * self.page_tokens as usize) as u32
    }

    /// The cached prefix with content `key` at `length` tokens (an exact,
    /// page-aligned match) already in the cache, or `None`.
    fn find(&self, key: MatchKey, length: u32) -> Option<&PrefixEntry> {
        self.entries
            .iter()
            .find(|e| e.key == key && e.length_tokens == length)
    }
}

/// The longest of `candidates` that is a prefix of `prompt` by content —
/// whose key equals the prompt's key at its length (GitHub #193). The longest
/// match wins: a cached prefix of a cached prefix is a shorter match.
///
/// Only candidates no longer than the prompt are asked about: `keys_at`
/// answers a length past the end with the whole prompt's key, which must
/// never stand in for a longer entry's.
fn longest_keyed<'e, T>(
    prompt: PromptContent<'_>,
    candidates: impl Iterator<Item = &'e T>,
    name: impl Fn(&T) -> (MatchKey, u32),
) -> Option<&'e T> {
    let candidates: Vec<&T> = candidates.filter(|c| name(c).1 <= prompt.tokens()).collect();
    if candidates.is_empty() {
        return None;
    }
    let lengths: Vec<u32> = candidates.iter().map(|c| name(c).1).collect();
    let keys = prompt.keys_at(&lengths);
    candidates
        .into_iter()
        .zip(keys)
        .filter(|(c, key)| name(c).0 == *key)
        .map(|(c, _)| c)
        .max_by_key(|c| name(c).1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gdn::GdnState;
    use crate::identity::MediaKey;
    use crate::types::TokenId;

    fn retention(at: u64) -> Retention {
        Retention {
            at,
            class: RequestClass::Agent,
            used_at: Instant::now(),
        }
    }

    /// A GDN state resumable at `position` (a recorded boundary): valid
    /// for prefix reuse.
    fn gdn_boundary(position: usize) -> GdnState {
        let mut gdn = GdnState::new();
        gdn.checkpoint(position);
        gdn
    }

    /// A GDN state mid-prefill (the position is not a recorded boundary):
    /// *invalid* for prefix reuse (a mid-prefill position is not
    /// resumable for GDN layers).
    fn gdn_mid_prefill(position: usize) -> GdnState {
        let mut gdn = GdnState::new();
        gdn.advance(position);
        gdn
    }

    /// A 64-token prompt (= 4 pages at 16 tokens/page).
    fn prompt64() -> Vec<TokenId> {
        (1..=64).collect()
    }

    /// A 96-token prompt (= 6 pages): its first 64 tokens match
    /// `prompt64`'s (a longer prompt sharing the shorter head).
    fn prompt96() -> Vec<TokenId> {
        (1..=96).collect()
    }

    #[test]
    fn register_truncates_to_a_page_boundary() {
        // A 40-token prompt (2.5 pages) shares its first 32 tokens (2
        // whole pages) — the half page is never split between requests.
        let mut cache = PrefixCache::new(16);
        let tokens: Vec<TokenId> = (1..=40).collect();
        let (_, pages) = cache.register(7, &tokens, &gdn_boundary(0), None).unwrap();
        assert_eq!(pages, 2, "a 40-token prompt holds 2 shared pages");
        assert_eq!(cache.pinned_pages(), 2);
        // A sibling whose prompt starts with the cached prefix matches it
        // (the match is the cached prefix, not the whole prompt); a claim
        // adds no new pages.
        let claim = cache.claim(&tokens).unwrap();
        assert_eq!(claim.tokens, 32, "the sibling skips the shared 32 tokens");
        assert_eq!(cache.pinned_pages(), 2, "a claim adds no new pages");
    }

    #[test]
    fn register_rejects_a_subpage_prefix() {
        // A 15-token prompt (less than one 16-token page) holds no
        // shareable page: nothing is cached.
        let mut cache = PrefixCache::new(16);
        let tokens: Vec<TokenId> = (1..=15).collect();
        assert!(
            cache.register(7, &tokens, &gdn_boundary(0), None).is_none(),
            "a sub-page prompt holds no shareable page"
        );
        assert_eq!(cache.entry_count(), 0);
        assert!(cache.claim(&tokens).is_none());
    }

    #[test]
    fn register_rejects_a_mid_prefill_gdn_position() {
        // core-02: a mid-prefill GDN position (not a recorded boundary)
        // is not resumable, so its prefix is not reusable.
        let mut cache = PrefixCache::new(16);
        let tokens = prompt64();
        assert_eq!(
            cache.register(7, &tokens, &gdn_mid_prefill(128), None),
            None,
            "a mid-prefill position is not a reusable prefix"
        );
        assert_eq!(cache.entry_count(), 0);
        // The same prompt with a valid boundary registers fine.
        let (id, pages) = cache.register(7, &tokens, &gdn_boundary(0), None).unwrap();
        assert_eq!(pages, 4);
        let _ = id;
    }

    #[test]
    fn a_duplicate_prompt_claims_the_existing_entry() {
        // The same prompt (page-aligned) is cached once: a second
        // registration of it is a no-op (the caller claims the existing
        // entry — no second set of pages).
        let mut cache = PrefixCache::new(16);
        let tokens = prompt64();
        let (_, pages) = cache.register(7, &tokens, &gdn_boundary(0), None).unwrap();
        assert_eq!(pages, 4);
        assert!(
            cache.register(7, &tokens, &gdn_boundary(0), None).is_none(),
            "a duplicate prompt is not re-registered"
        );
        assert_eq!(cache.entry_count(), 1);
        // The duplicate's claim still pins the shared prefix (no new
        // pages: the entry's charge covers every claimant).
        assert!(cache.claim(&tokens).is_some());
        assert_eq!(cache.pinned_pages(), 4);
    }

    #[test]
    fn claim_returns_the_longest_match() {
        // Two cached prefixes, one a prefix of the other: a 96-token
        // prompt matches the longer (6-page) entry, not the shorter one.
        let mut cache = PrefixCache::new(16);
        cache.register(7, &prompt64(), &gdn_boundary(0), None).unwrap();
        cache.register(7, &prompt96(), &gdn_boundary(0), None).unwrap();
        let claim = cache.claim(&prompt96()).unwrap();
        assert_eq!(claim.tokens, 96, "the longest cached prefix matches");
        // A 64-token prompt matches only the shorter entry.
        let short = cache.claim(&prompt64()).unwrap();
        assert_eq!(short.tokens, 64);
    }

    #[test]
    fn claim_bumps_refcount_and_counts_reused_tokens() {
        let mut cache = PrefixCache::new(16);
        cache.register(7, &prompt64(), &gdn_boundary(0), None).unwrap();
        assert_eq!(cache.reused_tok(), 0, "registration is not a reuse");
        let claim = cache.claim(&prompt64()).unwrap();
        assert_eq!(claim.pages, 4);
        assert_eq!(cache.reused_tok(), 64, "a claim counts its skipped tokens");
        cache.claim(&prompt64()).unwrap();
        assert_eq!(
            cache.reused_tok(),
            128,
            "each claim counts its skipped tokens"
        );
    }

    #[test]
    fn release_drops_the_entry_at_zero_refcount() {
        let mut cache = PrefixCache::new(16);
        let (id, pages) = cache.register(7, &prompt64(), &gdn_boundary(0), None).unwrap();
        assert_eq!(pages, 4);
        cache.claim(&prompt64()).unwrap(); // refcount 2
        // Releases down to one claimant keep the entry pinned.
        assert_eq!(cache.release(id), vec![], "other claimants still pin it");
        assert_eq!(cache.pinned_pages(), 4);
        // The last release drops the entry, frees its pages and names the
        // request that published them (P4-10: the backend's own handle on
        // the leaf's prefix is dropped by that id).
        assert_eq!(cache.release(id), vec![(4, 7, 64)]);
        assert_eq!(cache.pinned_pages(), 0);
        assert_eq!(cache.entry_count(), 0);
        // A dropped entry cannot be claimed or released again.
        assert!(cache.claim(&prompt64()).is_none());
        assert_eq!(cache.release(id), vec![]);
    }

    #[test]
    fn a_claim_names_the_request_that_published_the_prefix() {
        // P4-10 (GitHub #126): the leaf's prefix — the thing that actually
        // owns the pages — is keyed by its publisher, so a claim that could
        // not name it would leave the backend unable to share anything.
        let mut cache = PrefixCache::new(16);
        cache.register(42, &prompt64(), &gdn_boundary(0), None).unwrap();
        let claim = cache.claim(&prompt64()).unwrap();
        assert_eq!(claim.publisher, 42);
    }

    #[test]
    fn the_shareable_head_is_whole_pages_of_the_prompt() {
        // What the scheduler asks before it prefills, so the chunk that
        // lands on the prefix can be cut at exactly the publish point.
        let cache = PrefixCache::new(16);
        assert_eq!(cache.shareable_head_tokens(40), 32, "2.5 pages share 2");
        assert_eq!(
            cache.shareable_head_tokens(64),
            64,
            "an exact multiple shares all of it"
        );
        assert_eq!(
            cache.shareable_head_tokens(15),
            0,
            "a sub-page prompt shares nothing"
        );
    }

    #[test]
    fn a_claim_floor_skips_entries_that_reach_no_further() {
        // GitHub #186: a request whose retained prompt checkpoint already
        // covers 64 tokens gains nothing from a 64-token prefix, and claiming
        // it would pin an entry and count a skip that never happened.
        let mut cache = PrefixCache::new(16);
        cache.register(7, &prompt64(), &gdn_boundary(0), None).unwrap();
        cache.register(8, &prompt96(), &gdn_boundary(0), None).unwrap();
        assert!(
            cache.claim_longer_than(&prompt96(), 96).is_none(),
            "nothing reaches past the floor"
        );
        assert_eq!(cache.reused_tok(), 0, "a refused claim counts no skip");
        let claim = cache.claim_longer_than(&prompt96(), 64).unwrap();
        assert_eq!(claim.tokens, 96, "only the entry past the floor is taken");
    }

    #[test]
    fn a_retention_reference_outlives_the_last_live_claimant() {
        // GitHub #186: a retained prompt checkpoint holds the pages under its
        // opener through this very entry, so the entry survives the request
        // that published it — and its pages are still charged exactly once.
        let mut cache = PrefixCache::new(16);
        let (id, pages) = cache.register(7, &prompt64(), &gdn_boundary(0), None).unwrap();
        assert_eq!(pages, 4);
        assert!(cache.retain(id), "the checkpoint takes a reference");
        assert_eq!(cache.pages_of(id), 4);
        // The publisher completes: its own reference goes, the entry stays.
        assert_eq!(cache.release(id), vec![], "the checkpoint still holds it");
        assert_eq!(cache.pinned_pages(), 4);
        assert!(
            cache.claim(&prompt64()).is_some(),
            "a later sibling can still claim it"
        );
        assert_eq!(cache.release(id), vec![]);
        // The checkpoint is discarded last: now the pages come back.
        assert_eq!(cache.release(id), vec![(4, 7, 64)]);
        assert_eq!(cache.pinned_pages(), 0);
        assert!(!cache.retain(id), "a dropped entry cannot be retained");
        assert_eq!(cache.pages_of(id), 0);
    }

    // ── Chained prefixes (GitHub #187) ──────────────────────────────────

    #[test]
    fn a_chained_prefix_covers_the_head_its_publisher_claimed() {
        // GitHub #187: a request standing on a 4-page prefix prefills past it
        // and publishes the 6-page head it now covers. The new entry owns only
        // the 2 pages it warmed itself — the 4 below stay the parent's, so the
        // pool is charged for them exactly once — but what it *covers*, and
        // what a claimant of it shares, is all 6.
        let mut cache = PrefixCache::new(16);
        let (parent, _) = cache.register(7, &prompt64(), &gdn_boundary(0), None).unwrap();
        let (child, own) = cache
            .register(8, &prompt96(), &gdn_boundary(0), Some(parent))
            .unwrap();
        assert_eq!(own, 2, "only the pages the publisher warmed itself");
        assert_eq!(cache.total_pages_of(child), 6, "the whole head it covers");
        assert_eq!(cache.pages_of(child), 2, "its own charge");
        assert_eq!(cache.pinned_pages(), 6, "one charge per page, chain included");

        let claim = cache.claim(&prompt96()).unwrap();
        assert_eq!(claim.id, child, "the longer, chained head wins");
        assert_eq!(claim.tokens, 96);
        assert_eq!(claim.pages, 6, "a claimant shares every page below its own first");
    }

    #[test]
    fn releasing_a_chained_prefix_cascades_to_its_parent() {
        // The child inherits the publishing request's reference on the parent
        // — from the publish on, that request reaches the parent *through*
        // the child — so the parent outlives its own registrant and comes
        // back exactly when the child does, never before.
        let mut cache = PrefixCache::new(16);
        let (parent, _) = cache.register(7, &prompt64(), &gdn_boundary(0), None).unwrap();
        cache.claim(&prompt96()).expect("request 8 claims the parent");
        let (child, _) = cache
            .register(8, &prompt96(), &gdn_boundary(0), Some(parent))
            .unwrap();
        // The parent's own registrant completes. The child still holds it.
        assert_eq!(cache.release(parent), vec![], "the child still holds the parent");
        assert_eq!(cache.pinned_pages(), 6);
        // Request 8 completes, releasing the child: both entries drop, and
        // each names the request whose leaf prefix the backend must let go of.
        assert_eq!(cache.release(child), vec![(2, 8, 96), (4, 7, 64)]);
        assert_eq!(cache.pinned_pages(), 0);
        assert_eq!(cache.entry_count(), 0);
    }

    #[test]
    fn a_chained_prefix_that_reaches_no_further_is_not_registered() {
        // A publish point at or below what the publisher already shares would
        // be a second entry over pages it does not own — no new history, and
        // a parent reference nothing balances.
        let mut cache = PrefixCache::new(16);
        let (parent, _) = cache.register(7, &prompt64(), &gdn_boundary(0), None).unwrap();
        assert_eq!(
            cache.register(8, &prompt64(), &gdn_boundary(0), Some(parent)),
            None,
            "the child covers nothing the parent does not"
        );
        assert_eq!(cache.entry_count(), 1);
    }

    #[test]
    fn a_retained_prefix_outlives_its_publisher_and_a_later_request_claims_it() {
        // GitHub #188: the burst. The first subagent publishes its system and
        // tools block and finishes; the entry does not drop, and the next
        // subagent — which arrived after its sibling was gone, and whose
        // prompt extends nothing of it but the block — claims it as if they
        // had been concurrent.
        let mut cache = PrefixCache::new(16);
        let (id, pages) = cache.register(7, &prompt64(), &gdn_boundary(0), None).unwrap();
        assert_eq!(pages, 4);
        assert!(cache.retain_published(id, retention(1)), "the publish is retained");
        assert!(cache.is_retained(id));
        // The publisher completes. Its reference goes; the retention stays.
        assert_eq!(cache.release(id), vec![], "the retention still holds it");
        assert_eq!(cache.pinned_pages(), 4);
        let claim = cache.claim(&prompt96()).expect("a later request claims it");
        assert_eq!(claim.tokens, 64, "it skips the whole retained block");
        assert_eq!(claim.publisher, 7, "named by the request that warmed it");
        assert_eq!(cache.pinned_pages(), 4, "a claim adds no new pages");
    }

    #[test]
    fn a_retained_prefix_is_reclaimable_only_while_nothing_live_stands_on_it() {
        // The first-victim path asks this before it gives up a bet: a prefix a
        // live request is also standing on frees not one page when the
        // retention goes, and giving it up would cost the next burst its reuse
        // while buying this request nothing.
        let mut cache = PrefixCache::new(16);
        let (id, _) = cache.register(7, &prompt64(), &gdn_boundary(0), None).unwrap();
        cache.retain_published(id, retention(1));
        assert_eq!(
            cache.reclaimable_retained(),
            Vec::<PrefixId>::new(),
            "the publisher is still live"
        );
        cache.release(id); // the publisher completes
        assert_eq!(cache.reclaimable_retained(), vec![id], "now only the retention holds it");
        cache.claim(&prompt64()).unwrap(); // a live claimant arrives
        assert_eq!(
            cache.reclaimable_retained(),
            Vec::<PrefixId>::new(),
            "a live claimant is standing on it again"
        );
    }

    #[test]
    fn giving_up_a_retention_hands_the_reference_back_once() {
        // `unretain` says whether the caller now owes a release. Saying yes
        // twice would release a reference the cache never held and drop the
        // entry out from under a live claimant.
        let mut cache = PrefixCache::new(16);
        let (id, _) = cache.register(7, &prompt64(), &gdn_boundary(0), None).unwrap();
        cache.retain_published(id, retention(1));
        assert!(
            !cache.retain_published(id, retention(2)),
            "retaining twice would pin the pages forever"
        );
        cache.release(id); // the publisher completes; the retention holds it
        assert!(cache.unretain(id), "the caller now owes one release");
        assert!(!cache.is_retained(id));
        assert!(!cache.unretain(id), "and owes it exactly once");
        assert_eq!(cache.release(id), vec![(4, 7, 64)], "that release frees the pages");
        assert_eq!(cache.pinned_pages(), 0);
        assert!(!cache.unretain(id), "a dropped entry is not retained");
        assert!(!cache.retain_published(id, retention(3)), "nor can it be retained again");
    }

    #[test]
    fn the_least_recently_used_retention_is_the_victim_and_a_claim_refreshes_it() {
        // LRU over the retained prefixes, refreshed when one is used: a system
        // block a burst is still arriving against must not be given up ahead
        // of one nothing has touched since the server started.
        let mut cache = PrefixCache::new(16);
        let (old, _) = cache.register(7, &prompt64(), &gdn_boundary(0), None).unwrap();
        let other: Vec<TokenId> = (500..=595).collect();
        let (new, _) = cache.register(8, &other, &gdn_boundary(0), None).unwrap();
        cache.retain_published(old, retention(1));
        cache.retain_published(new, retention(2));
        cache.release(old);
        cache.release(new);
        assert_eq!(cache.lru_retained(), Some(old), "the older retention goes first");
        // A claim on the older one moves it to the back of the order.
        let claim = cache.claim(&prompt64()).unwrap();
        cache.touch_retained(claim.id, 9, Instant::now());
        cache.release(claim.id);
        assert_eq!(cache.lru_retained(), Some(new), "the used one is no longer oldest");
        // Only reclaimable retentions are offered: a live claimant hides one.
        cache.claim(&other).unwrap();
        assert_eq!(cache.lru_retained(), Some(old));
        assert!(cache.unretain(old));
        cache.release(old);
        assert_eq!(cache.lru_retained(), None, "nothing else can be given up");
    }

    #[test]
    fn touching_something_that_is_not_retained_changes_nothing() {
        // Every claim touches, because the claimant cannot tell a retained
        // entry from a live sibling's — and must not have to.
        let mut cache = PrefixCache::new(16);
        let (id, _) = cache.register(7, &prompt64(), &gdn_boundary(0), None).unwrap();
        cache.touch_retained(id, 5, Instant::now());
        assert!(!cache.is_retained(id), "touching does not retain");
        cache.touch_retained(999, 5, Instant::now()); // a gone entry
        assert_eq!(cache.entry_count(), 1);
    }

    #[test]
    fn a_claim_seeds_the_claimant_gdn_state() {
        // The claimant's recurrent state resumes at the shared boundary
        // (core-02): the claim hands over the entry's GDN state (a
        // recorded boundary the claimant can snapshot / resume at).
        let mut cache = PrefixCache::new(16);
        cache.register(7, &prompt64(), &gdn_boundary(0), None).unwrap();
        let claim = cache.claim(&prompt64()).unwrap();
        assert!(claim.gdn.is_valid_snapshot_point(claim.gdn.position()));
    }

    #[test]
    fn peeking_at_the_longest_match_claims_nothing() {
        // GitHub #190: what a KV-RAM checkpoint has to beat is read without
        // pinning the entry or counting a skip it may never take.
        let mut cache = PrefixCache::new(16);
        cache.register(7, &prompt64(), &gdn_boundary(64), None).unwrap();
        assert_eq!(cache.longest_match_tokens(&prompt96()), 64);
        assert_eq!(cache.longest_match_tokens(&(500..600).collect::<Vec<_>>()), 0);
        let entry = cache.claim(&prompt96()).unwrap().id;
        assert_eq!(cache.refcount_of(entry), 2, "the publisher and this one claim, not the peeks");
        assert_eq!(cache.reused_tok(), 64);
    }

    #[test]
    fn a_spilled_prefix_is_matched_in_kv_ram_only_once_the_device_copy_is_gone() {
        // GitHub #190.
        let mut cache = PrefixCache::new(16);
        let (id, _) = cache.register(7, &prompt64(), &gdn_boundary(64), None).unwrap();
        assert!(cache.record_spill(id).is_none(), "only a retained prefix spills");
        assert!(cache.retain_published(id, retention(1)));
        let spilled = cache.record_spill(id).unwrap();
        assert_eq!(cache.spilled_copy_of(id), Some(spilled));
        assert!(
            cache.longest_spilled_match(&prompt96()).is_none(),
            "while it is on the device it is matched there"
        );

        assert!(cache.unretain(id));
        cache.release(id); // the retention
        cache.release(id); // the publisher
        assert_eq!(cache.entry_count(), 0);
        let matched = cache.longest_spilled_match(&prompt96()).unwrap();
        assert_eq!((matched.id, matched.publisher, matched.length_tokens), (spilled, 7, 64));
        assert!(cache.longest_spilled_match(&(500..600).collect::<Vec<_>>()).is_none());

        // Brought back, it is the device's to match again; forgotten, nobody's.
        let (back, _) = cache.register(7, &prompt64(), &gdn_boundary(64), None).unwrap();
        cache.record_return(spilled, back);
        assert!(cache.longest_spilled_match(&prompt96()).is_none());
        assert_eq!(cache.forget_spilled(spilled).map(|s| s.id), Some(spilled));
        assert_eq!(cache.spilled_count(), 0);
    }

    // ── media-aware identity (GitHub #193) ───────────────────────────────

    /// An image of `count` placeholders at `begin`, content `digest`, in an
    /// `[1, 8, 8]` grid — two of these differing only in `digest` are the
    /// same size and expand to the same placeholder ids.
    fn image(begin: u32, count: u32, digest: u8) -> MediaKey {
        MediaKey {
            begin,
            count,
            digest: [digest; 32],
            grid: [1, 8, 8],
        }
    }

    #[test]
    fn equal_token_ids_with_different_images_never_match() {
        // Two same-size images: the token ids are identical, placeholders
        // included, and only the digests differ.
        let mut cache = PrefixCache::new(16);
        let tokens = prompt96();
        let mine = [image(20, 16, 0xAA)];
        let yours = [image(20, 16, 0xBB)];
        cache
            .register(7, PromptContent::new(&tokens, &mine), &gdn_boundary(0), None)
            .unwrap();
        let theirs = PromptContent::new(&tokens, &yours);
        assert_eq!(cache.longest_match_tokens(theirs), 0);
        assert!(cache.claim(theirs).is_none(), "another picture never matches");
        assert!(cache.claim(&tokens).is_none(), "nor does no picture at all");
        let claim = cache.claim(PromptContent::new(&tokens, &mine)).unwrap();
        assert_eq!(claim.tokens, 96, "the same picture does");
    }

    #[test]
    fn a_head_before_the_image_is_shared_by_any_picture() {
        // The history before an item's first placeholder is the same history
        // whatever the picture: a system block is shared by a burst sending
        // different images.
        let mut cache = PrefixCache::new(16);
        let tokens = prompt96();
        let mine = [image(40, 16, 0xAA)];
        let yours = [image(40, 16, 0xBB)];
        let mine = PromptContent::new(&tokens, &mine);
        cache.register(7, mine.head(32), &gdn_boundary(0), None).unwrap();
        cache.register(7, mine, &gdn_boundary(0), None).unwrap();
        let claim = cache.claim(PromptContent::new(&tokens, &yours)).unwrap();
        assert_eq!(claim.tokens, 32, "only the head before the image");
    }

    #[test]
    fn the_same_ids_with_different_images_are_two_entries_not_a_duplicate() {
        // Two siblings in one batch, each publishing its own picture: the
        // second is not "the prompt already cached" — it would be handed the
        // first one's pages.
        let mut cache = PrefixCache::new(16);
        let tokens = prompt64();
        let (a, b) = ([image(4, 16, 0xAA)], [image(4, 16, 0xBB)]);
        assert!(cache.register(7, PromptContent::new(&tokens, &a), &gdn_boundary(0), None).is_some());
        assert!(cache.register(8, PromptContent::new(&tokens, &b), &gdn_boundary(0), None).is_some());
        assert!(cache.register(9, PromptContent::new(&tokens, &b), &gdn_boundary(0), None).is_none());
        assert_eq!(cache.entry_count(), 2);
        assert_eq!(cache.claim(PromptContent::new(&tokens, &b)).unwrap().publisher, 8);
    }

    #[test]
    fn a_prefix_never_ends_inside_an_image() {
        // A 70-token head floors to 64, which is inside the image at 50..80:
        // refused, rather than sharing half of a picture's placeholders.
        let mut cache = PrefixCache::new(16);
        let tokens = prompt96();
        let items = [image(50, 30, 0xAA)];
        let prompt = PromptContent::new(&tokens, &items);
        assert!(cache.register(7, prompt.head(70), &gdn_boundary(0), None).is_none());
        assert_eq!(cache.entry_count(), 0);
        // Ending exactly where the image begins is a legal head, and so is
        // ending past it.
        assert!(cache.register(7, prompt.head(48), &gdn_boundary(0), None).is_some());
        assert!(cache.register(7, prompt, &gdn_boundary(0), None).is_some());
    }

    #[test]
    fn a_spilled_prefix_is_matched_by_its_images_too() {
        let mut cache = PrefixCache::new(16);
        let tokens = prompt64();
        let (mine, yours) = ([image(8, 16, 0xAA)], [image(8, 16, 0xBB)]);
        let (id, _) = cache
            .register(7, PromptContent::new(&tokens, &mine), &gdn_boundary(0), None)
            .unwrap();
        cache.retain_published(id, retention(1));
        cache.record_spill(id).unwrap();
        cache.release(id);
        assert!(cache.unretain(id));
        cache.release(id);
        assert!(cache.longest_spilled_match(PromptContent::new(&tokens, &yours)).is_none());
        let spilled = cache
            .longest_spilled_match(PromptContent::new(&tokens, &mine))
            .cloned()
            .expect("the same picture matches in KV-RAM");
        // And it comes back under the same name.
        let (back, _) = cache.register_returned(&spilled).unwrap();
        assert!(cache.claim(PromptContent::new(&tokens, &yours)).is_none());
        assert_eq!(cache.claim(PromptContent::new(&tokens, &mine)).unwrap().id, back);
    }
}
