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

use crate::gdn::GdnState;
use crate::types::{RequestId, TokenId};

/// An opaque handle to a cached prefix entry (a claimant's reference to a
/// shared prefix).
pub type PrefixId = u64;

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
    /// The cached prefix tokens (the shared prompt head, page-aligned).
    pub tokens: Vec<TokenId>,
    /// Tokens in the cached prefix (= `pages * page_tokens`).
    pub length_tokens: u32,
    /// The KV pages the entry holds in the pool (the shared prefix pages;
    /// charged to the pool exactly once, for every claimant).
    pub pages: u32,
    /// The GDN state at the prefix's end (the resumable boundary, core-02;
    /// a claimant seeds its recurrent state from this).
    pub gdn: GdnState,
    /// Live claimants (the registrant counts as one; the entry is dropped
    /// when the last claimant releases).
    pub refcount: u32,
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
    /// The entry's pages (the caller reduces its own reservation by these
    /// — the entry now owns the shared pages).
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
    pub fn register(
        &mut self,
        publisher: RequestId,
        tokens: &[TokenId],
        gdn: &GdnState,
    ) -> Option<(PrefixId, u32)> {
        let length = (tokens.len() / self.page_tokens as usize) * self.page_tokens as usize;
        if length == 0 {
            return None; // fewer than one page: nothing shareable
        }
        // core-02: a prefix is reusable only at a recorded GDN boundary
        // (a mid-prefill position is not resumable for GDN layers).
        if !gdn.is_valid_snapshot_point(gdn.position()) {
            return None;
        }
        // A duplicate prompt (the same truncated prefix is already
        // cached): the caller claims the existing entry — no second set
        // of pages.
        if self.find(&tokens[..length]).is_some() {
            return None;
        }
        let entry = PrefixEntry {
            id: self.next_id,
            publisher,
            tokens: tokens[..length].to_vec(),
            length_tokens: length as u32,
            pages: (length / self.page_tokens as usize) as u32,
            gdn: gdn.clone(),
            refcount: 1, // the registrant is the first claimant
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
    pub fn claim(&mut self, tokens: &[TokenId]) -> Option<PrefixClaim> {
        let mut best: Option<&PrefixEntry> = None;
        for entry in &self.entries {
            if entry.tokens.len() > tokens.len() {
                continue;
            }
            if !tokens.starts_with(&entry.tokens) {
                continue;
            }
            // The longest match wins (a cached prefix of a cached prefix
            // is a shorter match).
            let better = match best {
                Some(b) => entry.tokens.len() > b.tokens.len(),
                None => true,
            };
            if better {
                best = Some(entry);
            }
        }
        let match_entry = best?;
        let id = match_entry.id;
        let claim = PrefixClaim {
            id,
            publisher: match_entry.publisher,
            tokens: match_entry.length_tokens,
            pages: match_entry.pages,
            gdn: match_entry.gdn.clone(),
        };
        if let Some(e) = self.entries.iter_mut().find(|e| e.id == id) {
            e.refcount += 1;
        }
        self.reused_tok += u64::from(claim.tokens);
        Some(claim)
    }

    /// Release one reference to `entry` (the claimant completed or was
    /// re-queued). When the last reference drops, the entry (and its
    /// pages) is released: returns the pages freed **and the request that
    /// published them** (so the caller can drop the backend's handle on the
    /// leaf's prefix too, P4-10), `None` while the entry is still pinned by
    /// other claimants.
    pub fn release(&mut self, entry: PrefixId) -> Option<(u32, RequestId)> {
        let pos = self.entries.iter().position(|e| e.id == entry)?;
        self.entries[pos].refcount -= 1;
        if self.entries[pos].refcount == 0 {
            let dropped = self.entries.remove(pos);
            Some((dropped.pages, dropped.publisher))
        } else {
            None
        }
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

    /// The cached prefix `key` (an exact, page-aligned match) already in
    /// the cache, or `None`.
    fn find(&self, key: &[TokenId]) -> Option<&PrefixEntry> {
        self.entries.iter().find(|e| e.tokens == key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gdn::GdnState;

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
        let (_, pages) = cache.register(7, &tokens, &gdn_boundary(0)).unwrap();
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
            cache.register(7, &tokens, &gdn_boundary(0)).is_none(),
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
            cache.register(7, &tokens, &gdn_mid_prefill(128)),
            None,
            "a mid-prefill position is not a reusable prefix"
        );
        assert_eq!(cache.entry_count(), 0);
        // The same prompt with a valid boundary registers fine.
        let (id, pages) = cache.register(7, &tokens, &gdn_boundary(0)).unwrap();
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
        let (_, pages) = cache.register(7, &tokens, &gdn_boundary(0)).unwrap();
        assert_eq!(pages, 4);
        assert!(
            cache.register(7, &tokens, &gdn_boundary(0)).is_none(),
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
        cache.register(7, &prompt64(), &gdn_boundary(0)).unwrap();
        cache.register(7, &prompt96(), &gdn_boundary(0)).unwrap();
        let claim = cache.claim(&prompt96()).unwrap();
        assert_eq!(claim.tokens, 96, "the longest cached prefix matches");
        // A 64-token prompt matches only the shorter entry.
        let short = cache.claim(&prompt64()).unwrap();
        assert_eq!(short.tokens, 64);
    }

    #[test]
    fn claim_bumps_refcount_and_counts_reused_tokens() {
        let mut cache = PrefixCache::new(16);
        cache.register(7, &prompt64(), &gdn_boundary(0)).unwrap();
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
        let (id, pages) = cache.register(7, &prompt64(), &gdn_boundary(0)).unwrap();
        assert_eq!(pages, 4);
        cache.claim(&prompt64()).unwrap(); // refcount 2
        // Releases down to one claimant keep the entry pinned.
        assert_eq!(cache.release(id), None, "other claimants still pin it");
        assert_eq!(cache.pinned_pages(), 4);
        // The last release drops the entry, frees its pages and names the
        // request that published them (P4-10: the backend's own handle on
        // the leaf's prefix is dropped by that id).
        assert_eq!(cache.release(id), Some((4, 7)));
        assert_eq!(cache.pinned_pages(), 0);
        assert_eq!(cache.entry_count(), 0);
        // A dropped entry cannot be claimed or released again.
        assert!(cache.claim(&prompt64()).is_none());
        assert_eq!(cache.release(id), None);
    }

    #[test]
    fn a_claim_names_the_request_that_published_the_prefix() {
        // P4-10 (GitHub #126): the leaf's prefix — the thing that actually
        // owns the pages — is keyed by its publisher, so a claim that could
        // not name it would leave the backend unable to share anything.
        let mut cache = PrefixCache::new(16);
        cache.register(42, &prompt64(), &gdn_boundary(0)).unwrap();
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
    fn a_claim_seeds_the_claimant_gdn_state() {
        // The claimant's recurrent state resumes at the shared boundary
        // (core-02): the claim hands over the entry's GDN state (a
        // recorded boundary the claimant can snapshot / resume at).
        let mut cache = PrefixCache::new(16);
        cache.register(7, &prompt64(), &gdn_boundary(0)).unwrap();
        let claim = cache.claim(&prompt64()).unwrap();
        assert!(claim.gdn.is_valid_snapshot_point(claim.gdn.position()));
    }
}