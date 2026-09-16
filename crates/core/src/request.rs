//! Request state machine + basic admission — `core-03`.
//!
//! The request lifecycle is `admitted → prefilling → running → done`
//! (`CONTEXT.md`). This module models a request in flight: its lifecycle
//! state, its class (for admission / backfill), the decode lane it holds, and
//! its token count, plus — since `core-05` — the admission state
//! machine's bookkeeping: the KV resources the request reserves, its
//! remaining service work (the protection's donor ordering, temporal
//! credit, and frontier distance all run on these quanta), and the
//! backfill class / protection epoch it was admitted under.
//!
//! [`admit_candidates`] / [`basic_admission`] below carry the *basic*
//! lane assignment (class priority + FIFO) that `core-03` shipped; the
//! **full** admission state machine (protection, backfill class, temporal
//! credit, frontier distance — `core-05`, ADR 0004) drives lane
//! assignment in the concrete scheduler (`concrete.rs` + `admission.rs`)
//! and supersedes them there.

use crate::admission::AdmissionResources;
use crate::gdn::GdnState;
use crate::prefix::PrefixId;
use crate::types::{
    BackfillClass, LaneId, RequestClass, RequestId, RequestInput, RequestState, SpecCounters,
    TokenId,
};

/// A request in flight in the engine: its lifecycle state, its class (for
/// admission / backfill), and the decode lane it holds while running.
#[derive(Debug, Clone)]
pub struct Request {
    pub id: RequestId,
    /// Admission / backfill class (drives lane priority and eviction order).
    pub class: RequestClass,
    /// The submitted prompt (tokenized + templated) + generation params.
    pub input: RequestInput,
    /// The lifecycle state (admitted → prefilling → running → done).
    pub state: RequestState,
    /// The resident decode lane held while `Running` (else `None`).
    pub lane: Option<LaneId>,
    /// Whether the leaf currently holds a materialized sequence for this
    /// request (P4-07, GitHub #125): KV pages, GDN slot and conv taps
    /// reserved — true from the first successful prefill chunk (durable
    /// across every later chunk and the eventual decode-lane deal) until
    /// completion, cancellation, or eviction. **Independent of `lane`**: a
    /// `Prefilling` request is resident with no lane at all, which is
    /// exactly the gap this ticket closes (`admission.rs`'s
    /// `resident_slots` dimension is charged/released off this flag, never
    /// off `lane`).
    pub resident: bool,
    /// Tokens generated so far for this request.
    pub tokens: u32,
    /// The speculative rounds this request ran, summed (P5-06, GitHub
    /// #154); `None` until its first one.
    pub spec: Option<SpecCounters>,
    /// The resources this request reserves while it holds a lane (core-05:
    /// the admission state machine's KV reservation, charged at deal and
    /// released at completion — the pool never over-allocates).
    pub resources: AdmissionResources,
    /// Remaining service work (quanta; 1 quantum per decode token): drives
    /// the protection's donor ordering, temporal credit, and frontier
    /// distance (core-05).
    pub remaining_work: u64,
    /// The protection epoch this request was admitted under as a backfill
    /// (core-05; 0 = a plain deal, no protection involved).
    pub backfill_epoch: u64,
    /// The class this request was admitted under by the admission state
    /// machine (core-05; [`BackfillClass::None`] for plain deals).
    pub backfill_class: BackfillClass,
    /// The GDN (linear-attention) recurrent-state tracker (core-02): the
    /// checkpoint / frontier boundaries at which the state is resumable.
    /// The host tier (core-06) snapshots this — a snapshot is only valid at
    /// a recorded boundary (a mid-prefill snapshot is invalid for GDN
    /// layers).
    pub gdn: GdnState,
    /// The sibling prefix entry this request reuses (core-07): `Some(id)`
    /// when the request's prefill skipped the cached shared head (the
    /// claim pins the shared pages for the request's lifetime — released
    /// on completion or re-queue).
    pub prefix_entry: Option<PrefixId>,
    /// The request that published the entry in [`Self::prefix_entry`]
    /// (P4-10, GitHub #126). The compute backend keys the *leaf's* prefix —
    /// which owns the physical pages — by its publisher, so a claim has to
    /// carry that identity down with it.
    pub prefix_publisher: Option<RequestId>,
    /// Leading prompt tokens reused from the shared prefix (core-07; 0 = a
    /// full prefill, nothing skipped).
    pub shared_prefix_tokens: u32,
    /// Whole KV pages of this request's history that belong to a shared
    /// prefix rather than to the request itself (GitHub #186) — whether it
    /// published that prefix, claimed it as a concurrent sibling, or arrived
    /// at it through a retained prompt checkpoint. One fact, three ways in,
    /// because what the checkpoint machinery needs to know is the same in all
    /// three: where the request's *own* first KV page begins, and therefore
    /// whether its generation opener falls inside it.
    pub shared_pages: u32,
    /// The retained prompt checkpoint this request claimed (GitHub #186, ADR
    /// 0029), named by the request that captured it — what the prefill job
    /// carries so the backend can find the device image. `None` for a request
    /// that reused no retained state.
    pub checkpoint_publisher: Option<RequestId>,
    /// The pool's own handle on the entry [`Self::checkpoint_publisher`] names
    /// (GitHub #187), as opposed to the backend's.
    ///
    /// It is what ties this request's own capture to the conversation it is
    /// continuing: the lineage the new entry joins, and the entry it
    /// supersedes when the turn has not changed. There is no session id to do
    /// that with (ADR 0029), so the claim edge is the link.
    pub checkpoint_entry: Option<crate::checkpoint::CheckpointId>,
    /// Leading prompt tokens reused from a retained prompt checkpoint (GitHub
    /// #186): everything up to that checkpoint's generation opener, which is
    /// **not** a whole number of pages. 0 = nothing reused.
    pub checkpoint_tokens: u32,
    /// The residency tier [`Request::checkpoint_tokens`] came from, for the
    /// request log's `reuse_source`.
    pub reuse_source: Option<crate::checkpoint::ReuseSource>,
    /// Whether this request has already had its own prompt checkpoint
    /// captured (GitHub #186). A request captures at most one: the opener is
    /// a single point in its prompt, and a second capture would be a second
    /// image of state that has since moved on.
    pub checkpoint_captured: bool,
    /// The tiers this request's retained-state lookup found nothing in (GitHub
    /// #190), reported when its first chunk lands. A request waiting for room
    /// looks again on every tick, so the lookup is not the place to count it.
    pub pending_retained_misses: crate::checkpoint::TierSet,
    /// The KV-RAM checkpoint this request chose and has not yet restored from
    /// (GitHub #190). While set, the blob is claimed in KV-RAM and nothing
    /// discards it: it is the only copy of the state the request's first job
    /// is about to be built on.
    pub kv_ram_claim: Option<crate::checkpoint::CheckpointId>,
    /// Leading prompt tokens this request's sequence holds as its own pages
    /// because they were restored from a **materialized** blob rather than
    /// prefilled or shared (GitHub #190): a KV-RAM checkpoint, or a snapshot
    /// taken while it held a shared prefix. No publish point at or below it
    /// can be reached any more — the prefill that would have stopped there is
    /// already behind the sequence — so [`Request::publish_point`] skips them.
    pub standalone_tokens: u32,
    /// The whole KV pages of this request's own prompt — what it *could*
    /// publish as a shared prefix (P4-10, GitHub #126), or 0 for a prompt
    /// shorter than one page.
    ///
    /// Known before the first chunk, not after the last, because the chunk
    /// that lands on it has to stop there: the mutable state a claimant
    /// clones is the state at the prefix's end. Read it through
    /// [`Request::publish_point`], which also answers whether this request
    /// publishes at all.
    pub publish_tokens: u32,
    /// Prompt tokens already sent to the compute backend during prefill
    /// (P3-01, ADR 0018): advances by at most the scheduler's serving chunk
    /// width per `advance()`. `Prefilling` is durable and carries this as
    /// its progress — a request can sit here across many ticks before
    /// [`Request::prefill_complete`] is true. Seeded to `shared_prefix_tokens`
    /// when a sibling-prefix claim skips the shared head (core-07): those
    /// tokens are already warm, so progress starts past them.
    pub prefill_progress: u32,
    /// Marked by [`crate::concrete::ConcreteScheduler::cancel`] (P3-01, ADR
    /// 0018): cancel is abort, not suspend. A cancelled request is never
    /// dealt another chunk or decode round — the next `advance()` releases
    /// it via [`Request::abort`] before running its phases (whatever chunk
    /// was already in flight has, by construction, already returned: compute
    /// calls are synchronous, so there is nothing to interrupt).
    pub cancelled: bool,
    /// Consecutive `prefill_step` calls that failed with this request in
    /// the batch (GitHub #166). Reset by a successful chunk; at
    /// `MAX_PREFILL_ATTEMPTS` the scheduler ends the request with
    /// [`crate::types::FinishReason::Error`] instead of retrying it forever.
    pub prefill_failures: u32,
}

impl Request {
    /// A freshly-admitted request: state `Admitted`, no lane yet.
    /// `resources` is the request's KV reservation (the admission state
    /// machine, core-05) and `remaining_work` its service work in quanta.
    pub fn new(
        id: RequestId,
        class: RequestClass,
        input: RequestInput,
        resources: AdmissionResources,
        remaining_work: u64,
    ) -> Self {
        Self {
            id,
            class,
            input,
            state: RequestState::Admitted,
            lane: None,
            resident: false,
            tokens: 0,
            spec: None,
            resources,
            remaining_work,
            backfill_epoch: 0,
            backfill_class: BackfillClass::None,
            gdn: GdnState::new(),
            prefix_entry: None,
            prefix_publisher: None,
            shared_prefix_tokens: 0,
            shared_pages: 0,
            checkpoint_publisher: None,
            checkpoint_entry: None,
            checkpoint_tokens: 0,
            reuse_source: None,
            checkpoint_captured: false,
            pending_retained_misses: crate::checkpoint::TierSet::default(),
            kv_ram_claim: None,
            standalone_tokens: 0,
            publish_tokens: 0,
            prefill_progress: 0,
            cancelled: false,
            prefill_failures: 0,
        }
    }

    /// Whether a transition `from → to` is a valid lifecycle step.
    ///
    /// The lifecycle is a strict pipeline: `Admitted → Prefilling → Running
    /// → Done`, plus — since core-06 — two host-tier detours through
    /// `Evicted` (P4-07, GitHub #125 added the first: a half-prefilled
    /// request is GPU-resident, and so evictable, well before it ever holds
    /// a decode lane):
    /// - `Prefilling → Evicted → Prefilling` — a half-prefilled request is
    ///   suspended and later restored to resume prefilling from its
    ///   snapshotted chunk boundary (not from zero); it re-earns a decode
    ///   lane the normal way, once prefill completes.
    /// - `Running → Evicted → Running` — a request evicted from a decode
    ///   lane is suspended and later restored straight back onto one,
    ///   resuming generation without re-prefilling.
    ///
    /// Either detour may also end in the re-queue `Evicted → Admitted` (the
    /// host-tier snapshot was discarded, so the request re-prefills from
    /// the start). No skipping, no other backwards steps.
    pub fn valid_transition(from: RequestState, to: RequestState) -> bool {
        match from {
            RequestState::Admitted => to == RequestState::Prefilling,
            RequestState::Prefilling => {
                to == RequestState::Running || to == RequestState::Evicted
            }
            RequestState::Running => to == RequestState::Done || to == RequestState::Evicted,
            RequestState::Evicted => {
                to == RequestState::Running
                    || to == RequestState::Prefilling
                    || to == RequestState::Admitted
            }
            RequestState::Done => false,
        }
    }

    /// Advance the request to `next`, enforcing the valid lifecycle. Returns
    /// `true` (and applies) only when the transition is valid.
    ///
    /// A request only enters `Running` while holding a resident decode
    /// lane — `assign_lane` first (it also transitions to `Running`), so a
    /// lane-less request can never run.
    pub fn advance(&mut self, next: RequestState) -> bool {
        if !Self::valid_transition(self.state, next) {
            return false;
        }
        if next == RequestState::Running && self.lane.is_none() {
            return false;
        }
        self.state = next;
        true
    }

    /// The **publish point** (CONTEXT.md): the prefill position at which this
    /// request publishes its prompt head as a shared prefix, or 0 for a
    /// request that publishes nothing (P4-10, GitHub #126).
    ///
    /// A request holding a claim publishes a **chained** head (GitHub #187):
    /// the pages it warmed past the entry it resumed from, over that entry.
    /// Publishing its whole head instead would own the shared pages twice, and
    /// publishing nothing — which is what #186 did — is what stopped a
    /// conversation past its first page from ever taking a second checkpoint.
    ///
    /// **This is a sequence, not a number.** A request may publish at more
    /// than one boundary in its prompt, each as a chained entry over the last,
    /// and what it publishes *next* is the first boundary past what it already
    /// shares. #187 supplies one boundary, the generation opener's page floor,
    /// which is what makes the request's own checkpoint capturable. #188 adds
    /// the end of the system-and-tools block **before** it — the only head a
    /// subagent burst can share — and the two compose rather than exclude each
    /// other: publish the block, chain the opener's page over it, capture
    /// there. (Merge note: `boundaries` below becomes
    /// `[self.retained_prefix_point(page_tokens), self.publish_tokens]` and
    /// `ConcreteScheduler::submit` stops flooring `publish_tokens` to the
    /// block, which is the whole of the #187 × #188 join.)
    ///
    /// A boundary past the first is published only when the request has a
    /// generation opener to capture at. Without that gate every concurrent
    /// sibling of a burst would pay a chunk split for a chained prefix nobody
    /// is going to claim — no subagent's prompt extends its sibling's.
    ///
    /// One function so that the chunk decomposition (where to cut) and the
    /// registration (when to publish) cannot disagree about it.
    pub fn publish_point(&self, page_tokens: u32) -> u32 {
        // Pages already shared, or restored as the request's own from a
        // materialized blob: a boundary inside either is behind it.
        let shared = self
            .shared_pages
            .saturating_mul(page_tokens)
            .max(self.standalone_tokens);
        // Ascending, and it has to be: "the first boundary past what I already
        // share" is only the next one if they are in prompt order. They always
        // are — the system block ends before the last generation opener, and
        // flooring to pages is monotone.
        let boundaries = [self.retained_prefix_point(page_tokens), self.publish_tokens];
        for at in boundaries {
            // Already published, or covered by the entry this request claimed.
            if at <= shared {
                continue;
            }
            // A chained publish has to earn its chunk split — and so does a
            // head published over history restored from a materialized blob
            // (GitHub #190), which is a resumed request's publish too.
            let resumed = self.prefix_entry.is_some() || self.standalone_tokens > 0;
            if resumed && self.input.opener_tokens.is_none() {
                continue;
            }
            return at;
        }
        0
    }

    /// The **retained-prefix boundary** (GitHub #188, ADR 0029): the end of
    /// this prompt's first system-and-tools block floored to whole KV pages,
    /// or 0 for a request that publishes no retained prefix.
    ///
    /// 0 has one meaning with three ways in, and each is a real case:
    ///
    /// - **the frontend reported none** — a render that does not open with a
    ///   system block, or one whose boundary does not tokenize to an exact
    ///   token prefix (three of the seven reference-recorded renders carry
    ///   neither a system message nor tools);
    /// - **the block is under one page** — a shared prefix is whole KV pages,
    ///   so there is nothing to publish there. The request falls back to
    ///   #186's opener page and takes its prompt checkpoint as before;
    /// - **an image starts in the block's first page** — the boundary never
    ///   ends inside a media item's placeholders (GitHub #193), so it walks
    ///   back to the page holding the item's first one, and that can be
    ///   page 0. A system block carries no images today; this is the rule
    ///   holding if one ever does.
    ///
    /// Unlike [`Request::publish_point`] this is a property of the *prompt*,
    /// not of what the request is currently holding, so it stays answerable
    /// after a claim. That is what lets the registration site ask "is the head
    /// I just published the retained one?" instead of re-deriving a condition
    /// that could disagree with the one the chunk was cut at.
    pub fn retained_prefix_point(&self, page_tokens: u32) -> u32 {
        if page_tokens == 0 {
            return 0;
        }
        self.input
            .system_block_tokens
            .map_or(0, |block| self.input.prefix_floor(block, page_tokens))
    }

    /// The **capture point** (GitHub #186, ADR 0029): the prefill position at
    /// which this request's state is captured as a prompt checkpoint, or 0 for
    /// a request that captures none.
    ///
    /// It is the generation opener the frontend reported, and it is offered
    /// only when the opener falls inside the request's **own first KV page**
    /// — that is, when the whole pages under it are exactly the shared prefix
    /// the request already holds. That is not a formality: what a later
    /// claimant shares is those whole pages, and what it copies is the partial
    /// tail page, which has to be a page the publisher owns rather than one
    /// other holders are also writing.
    ///
    /// The three ways that condition fails, and what each means:
    ///
    /// - **no shared prefix** (`prefix_entry` is `None`) — a prompt head
    ///   shorter than one page, or a head another request in the same batch
    ///   already took. There is nothing to hang the checkpoint's history on.
    /// - **the opener's pages are not the shared ones** — the request is
    ///   standing on a prefix that stops somewhere other than its own
    ///   opener's page, so its own first page is not the opener's. Two real
    ///   cases used to end here, and #187 removed both: a conversation's
    ///   *second* turn, which resumed from an earlier checkpoint and prefilled
    ///   past it, and every **concurrent sibling** that claimed another
    ///   request's shared prefix and whose own opener lies further on. Neither
    ///   is refused now — each publishes a **chained** prefix at its own
    ///   opener's page floor first ([`Request::publish_point`]), which makes
    ///   this condition true rather than weakening it. What still lands here
    ///   is a request whose claim already reaches *past* its own opener's
    ///   page, which has nothing of its own to cut at.
    /// - **already captured** — one checkpoint per request.
    ///
    /// One function, so that the chunk decomposition (where to cut) and the
    /// capture (when to take it) cannot disagree about it, exactly as
    /// [`Request::publish_point`] is one function.
    pub fn checkpoint_point(&self, page_tokens: u32) -> u32 {
        if self.checkpoint_captured || page_tokens == 0 {
            return 0;
        }
        let Some(opener) = self.input.opener_tokens else {
            return 0;
        };
        // The opener must leave at least one prompt token after it, and not
        // only because a checkpoint at the prompt's end would be useless to
        // the next turn. It is what makes the acceptance criterion "the
        // penalty-count row at the opener is zero" true *by construction*
        // rather than by luck: only the chunk that **ends** the prompt is
        // dealt the request's real sampling parameters (see
        // `ConcreteScheduler::advance`), so an opener strictly inside the
        // prompt guarantees the capturing chunk is an intermediate one,
        // therefore greedy, therefore has sampled nothing and has left the
        // count row alone. An opener at the very end would be captured from
        // a chunk that had just sampled.
        if opener == 0 || opener as usize >= self.input.tokens.len() {
            return 0;
        }
        if self.prefix_entry.is_none() || opener / page_tokens != self.shared_pages {
            return 0;
        }
        opener
    }

    /// Whether the checkpoint this request is about to capture **opens a new
    /// turn** of its conversation (GitHub #187, ADR 0029) — whether a real
    /// user message lies past the entry it resumed from.
    ///
    /// A request that resumed from nothing is the start of a conversation as
    /// far as anything here can tell, so its checkpoint opens that
    /// conversation's first turn. Otherwise the frontend's
    /// last-real-user-query offset decides it, against the claimed entry's
    /// reach: **at or past** it means the human spoke again and this is a new
    /// turn; before it means the prompt grew by an assistant message and a
    /// tool result, which is one more iteration of the same turn. At or past,
    /// not strictly past, because the offset is where the user message
    /// *begins* — one beginning exactly where the claimed entry ends is
    /// history that entry never covered.
    ///
    /// A frontend that could not report the offset answers **false**. A wrong
    /// `false` costs a superseded entry that would have been kept; a wrong
    /// `true` retires the turn-opening entry the next user message was going
    /// to match, which is the reuse this whole slice exists to protect.
    pub fn opens_a_turn(&self) -> bool {
        match self.checkpoint_entry {
            None => true,
            Some(_) => self
                .input
                .user_turn_tokens
                .is_some_and(|user| user >= self.checkpoint_tokens),
        }
    }

    /// Whether this request has finished prefill: every prompt token has
    /// been sent to the compute backend (P3-01, ADR 0018). `Prefilling` is
    /// durable — a request may sit here across many `advance()` calls
    /// (each sending at most one serving chunk) before this is true. Only a
    /// complete request may be dealt a decode lane.
    pub fn prefill_complete(&self) -> bool {
        self.prefill_progress as usize >= self.input.tokens.len()
    }

    /// Abort the request regardless of its current lifecycle state (P3-01,
    /// ADR 0018: cancel is abort, not suspend — there is no suspend/resume
    /// primitive for a cancelled request). Unlike [`Request::advance`], this
    /// is not gated by [`Request::valid_transition`]: cancellation is a
    /// deliberate exit from the pipeline, not a pipeline step. Fails
    /// (returns `false`) when the request is already `Done`.
    pub fn abort(&mut self) -> bool {
        if self.state == RequestState::Done {
            return false;
        }
        self.state = RequestState::Done;
        self.lane = None;
        true
    }

    /// Assign a resident decode lane (only valid from `Prefilling` →
    /// `Running`). Fails (returns `false`) when the request is not in the
    /// `Prefilling` state — a request must finish prefill before it holds a
    /// lane.
    pub fn assign_lane(&mut self, lane: LaneId) -> bool {
        if self.state != RequestState::Prefilling {
            return false;
        }
        self.lane = Some(lane);
        self.state = RequestState::Running;
        true
    }

    /// Evict a `Running` request from its decode lane to the host KV-RAM
    /// tier (core-06): transition `Running → Evicted` and release the lane.
    /// Fails (returns `false`) when the request is not `Running`. Restores
    /// back onto a lane via [`Request::restore_lane`] — see
    /// [`Request::evict_prefilling`] for the half-prefilled counterpart.
    pub fn evict(&mut self) -> bool {
        if self.state != RequestState::Running {
            return false;
        }
        self.state = RequestState::Evicted;
        self.lane = None; // the lane is released (the caller reclaims it).
        true
    }

    /// Evict a half-prefilled `Prefilling` request to the host KV-RAM tier
    /// (P4-07, GitHub #125): transition `Prefilling → Evicted`. Holds no
    /// lane to release — a `Prefilling` request is device-resident (KV
    /// pages, GDN slot, conv taps) well before it ever holds one, which is
    /// exactly what makes it evictable at all. Fails (returns `false`) when
    /// the request is not `Prefilling`. Restores back into `Prefilling` via
    /// [`Request::restore_prefilling`], never straight to `Running` — its
    /// prefill was not complete when it was evicted, and a decode lane is
    /// earned the normal way once it is.
    pub fn evict_prefilling(&mut self) -> bool {
        if self.state != RequestState::Prefilling {
            return false;
        }
        self.state = RequestState::Evicted;
        true
    }

    /// Restore a previously-evicted `Running` request onto a decode lane
    /// (core-06): transition `Evicted → Running` and (re-)acquire the lane.
    /// Fails (returns `false`) when the request is not in the `Evicted`
    /// state. Resumes from where it was evicted (no re-prefill).
    pub fn restore_lane(&mut self, lane: LaneId) -> bool {
        if self.state != RequestState::Evicted {
            return false;
        }
        self.lane = Some(lane);
        self.state = RequestState::Running;
        true
    }

    /// Restore a previously-evicted half-prefilled request into `Prefilling`
    /// (P4-07, GitHub #125): transition `Evicted → Prefilling`, no lane
    /// involved. The caller (the scheduler's snapshot restore) is
    /// responsible for putting [`Request::prefill_progress`] back at the
    /// chunk boundary the snapshot was taken at, so prefill resumes there
    /// rather than from zero. Fails (returns `false`) when the request is
    /// not in the `Evicted` state.
    pub fn restore_prefilling(&mut self) -> bool {
        if self.state != RequestState::Evicted {
            return false;
        }
        self.state = RequestState::Prefilling;
        true
    }

    /// Re-queue a discarded (evicted) request for re-prefill (core-06):
    /// transition `Evicted → Admitted` and release the lane (there is none
    /// — the request is suspended). Fails (returns `false`) when the
    /// request is not in the `Evicted` state. A re-queued request re-
    /// prefills from the start (its snapshot was discarded from the host
    /// tier, so its warmed KV is gone); the caller resets its token /
    /// service-work counters. The GDN recurrent state is also reset to a
    /// fresh (position-0, no-boundary) state: the re-prefilled stream starts
    /// over, so the old (stale) checkpoint boundaries — which sit at
    /// positions the new stream has not reached — would otherwise let a
    /// later snapshot be accepted at a position the stream never reached
    /// (a restore would resume GDN at a point it has not computed).
    pub fn requeue(&mut self) -> bool {
        if self.state != RequestState::Evicted {
            return false;
        }
        self.state = RequestState::Admitted;
        self.lane = None;
        self.gdn = GdnState::new(); // fresh recurrent state (re-prefill starts over)
        // core-07: the re-prefilled stream starts over — the old prefix
        // claim (released by the caller) is stale; a fresh prefill may
        // re-claim a live entry.
        self.prefix_entry = None;
        self.prefix_publisher = None;
        self.shared_prefix_tokens = 0;
        self.shared_pages = 0;
        // GitHub #186: so is the retained-checkpoint claim, and for the same
        // reason — the pages and the image it resumed from were released with
        // the entry the caller let go of. `checkpoint_captured` is *not*
        // reset: whatever this request already retained is a real entry in
        // the pool, which a re-prefill must not duplicate.
        self.checkpoint_publisher = None;
        self.checkpoint_entry = None;
        self.checkpoint_tokens = 0;
        self.reuse_source = None;
        // GitHub #190: so is whatever a materialized blob restored — the
        // caller released the KV-RAM claim, and the sequence is gone.
        self.kv_ram_claim = None;
        self.standalone_tokens = 0;
        // `publish_tokens` is untouched: it is a property of the prompt, not
        // of a run. With the claim gone the re-prefill is free to publish
        // that head again (P4-10, GitHub #126) — the pages it had went back
        // with the entry the caller released.
        // P3-01: the re-prefilled stream starts over at position 0 too —
        // whatever chunk progress it had before eviction is gone with the
        // KV it warmed.
        self.prefill_progress = 0;
        self.prefill_failures = 0;
        true
    }

    /// Record a GDN checkpoint / frontier boundary at `position` (core-02):
    /// the recurrent state becomes resumable there. The host tier (core-06)
    /// may snapshot the request's state only at a recorded boundary.
    pub fn checkpoint(&mut self, position: usize) {
        self.gdn.checkpoint(position);
    }

    /// The prompt tokens this request was submitted with.
    pub fn prompt_tokens(&self) -> &[TokenId] {
        &self.input.tokens
    }
}

/// **Basic admission** (core-03): assign free decode lanes to `Prefilling`
/// requests, returning the (request, lane) pairs dealt, ordered by class
/// priority (Interactive before Agent — the derived `Ord` on
/// [`RequestClass`]) then FIFO by request id. The lane pool is consumed in
/// the order the caller provides it (the concrete scheduler sorts it
/// ascending so the deal order is deterministic).
pub fn admit_candidates(
    requests: &mut [Request],
    free_lanes: &mut Vec<LaneId>,
) -> Vec<(RequestId, LaneId)> {
    // Candidate requests: those that have finished prefill and need a lane.
    let mut pending: Vec<usize> = (0..requests.len())
        .filter(|&i| requests[i].state == RequestState::Prefilling)
        .collect();
    // Class priority first (Interactive < Agent), then FIFO by id.
    pending.sort_by_key(|&i| (requests[i].class, requests[i].id));

    let mut dealt = Vec::new();
    for &i in &pending {
        if let Some(lane) = free_lanes.pop()
            && requests[i].assign_lane(lane)
        {
            dealt.push((requests[i].id, lane));
        }
    }
    dealt
}

/// **Basic admission** (core-03): assign free decode lanes to `Prefilling`
/// requests. Requests are ordered by class priority (Interactive before
/// Agent — the derived `Ord` on [`RequestClass`]) then FIFO by request id.
/// A `Prefilling` request that gets a lane transitions to `Running`.
///
/// Returns the number of lanes assigned. This is the *basic* lane
/// assignment — the full admission state machine (protection, backfill
/// class, temporal credit, frontier distance — core-05, ADR 0004) drives
/// lane assignment in the concrete scheduler and supersedes it there.
pub fn basic_admission(requests: &mut [Request], free_lanes: &mut Vec<LaneId>) -> usize {
    admit_candidates(requests, free_lanes).len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DecodeParams;

    fn req(id: RequestId, class: RequestClass, state: RequestState) -> Request {
        let mut r = Request::new(
            id,
            class,
            RequestInput {
                model: "qwen3.8-27b".into(),
                tokens: vec![1, 2, 3],
                params: DecodeParams::default(),
                multimodal: None,
                opener_tokens: None,
                user_turn_tokens: None,
                system_block_tokens: None,
            },
            AdmissionResources::default(),
            4,
        );
        r.state = state;
        r
    }

    #[test]
    fn lifecycle_is_a_strict_pipeline() {
        let mut r = req(0, RequestClass::Agent, RequestState::Admitted);
        assert!(r.advance(RequestState::Prefilling));
        // Entering Running requires a held lane: assign_lane first (it also
        // transitions to Running), then the pipeline continues to Done.
        assert!(r.assign_lane(3));
        assert_eq!(r.state, RequestState::Running);
        assert!(r.advance(RequestState::Done));
        // No valid transition out of Done.
        assert!(!r.advance(RequestState::Prefilling));
        // A lane-less request cannot enter Running.
        let mut r = req(1, RequestClass::Agent, RequestState::Prefilling);
        assert!(!r.advance(RequestState::Running));
        assert_eq!(r.state, RequestState::Prefilling);
    }

    #[test]
    fn invalid_transitions_are_rejected() {
        // Cannot skip from Admitted straight to Running.
        let mut r = req(0, RequestClass::Agent, RequestState::Admitted);
        assert!(!r.advance(RequestState::Running));
        // Cannot go backwards.
        let mut r = req(0, RequestClass::Agent, RequestState::Prefilling);
        assert!(!r.advance(RequestState::Admitted));
    }

    #[test]
    fn lane_assignment_requires_prefilling_state() {
        // A request still Admitted cannot grab a lane.
        let mut r = req(0, RequestClass::Agent, RequestState::Admitted);
        assert!(!r.assign_lane(3));
        assert_eq!(r.lane, None);
        // From Prefilling it can.
        let mut r = req(0, RequestClass::Agent, RequestState::Prefilling);
        assert!(r.assign_lane(3));
        assert_eq!(r.lane, Some(3));
        assert_eq!(r.state, RequestState::Running);
    }

    #[test]
    fn basic_admission_orders_by_class_then_fifo() {
        let mut requests = vec![
            req(10, RequestClass::Agent, RequestState::Prefilling),
            req(1, RequestClass::Agent, RequestState::Prefilling),
            req(2, RequestClass::Interactive, RequestState::Prefilling),
            req(3, RequestClass::Interactive, RequestState::Prefilling),
        ];
        let mut free = vec![5, 7];
        let n = basic_admission(&mut requests, &mut free);
        assert_eq!(n, 2);
        // Both Interactive requests hold a lane (class priority: Interactive
        // before Agent). Which physical lane each gets is not part of the
        // contract — free lanes are fungible, so assert on the set, not on
        // the deal order.
        assert!(requests[2].lane.is_some());
        assert!(requests[3].lane.is_some());
        // The Agent requests stay unassigned.
        assert!(requests[0].lane.is_none());
        assert!(requests[1].lane.is_none());
        assert!(free.is_empty());

        // FIFO within a class: with a single lane, the lower-id Interactive
        // request is admitted before the higher-id one (and before Agents).
        let mut requests = vec![
            req(9, RequestClass::Interactive, RequestState::Prefilling),
            req(4, RequestClass::Interactive, RequestState::Prefilling),
            req(0, RequestClass::Agent, RequestState::Prefilling),
        ];
        let mut free = vec![3];
        let n = basic_admission(&mut requests, &mut free);
        assert_eq!(n, 1);
        assert!(requests[1].lane.is_some()); // Interactive id 4 won
        assert!(requests[0].lane.is_none()); // Interactive id 9 deferred
        assert!(requests[2].lane.is_none()); // Agent deferred
        assert!(free.is_empty());
    }

    // ── P4-07 (GitHub #125): half-prefilled eviction ─────────────────────

    #[test]
    fn a_prefilling_request_can_be_evicted_and_restored_to_prefilling() {
        let mut r = req(0, RequestClass::Agent, RequestState::Prefilling);
        assert!(r.evict_prefilling());
        assert_eq!(r.state, RequestState::Evicted);
        assert_eq!(r.lane, None, "a Prefilling request held no lane to release");
        assert!(r.restore_prefilling());
        assert_eq!(r.state, RequestState::Prefilling);
        assert_eq!(r.lane, None, "restoring to Prefilling acquires no lane");
    }

    #[test]
    fn evict_prefilling_only_accepts_the_prefilling_state() {
        let mut r = req(0, RequestClass::Agent, RequestState::Admitted);
        assert!(!r.evict_prefilling(), "Admitted holds no device state to evict");
        let mut r = req(0, RequestClass::Agent, RequestState::Running);
        assert!(!r.evict_prefilling(), "Running evicts via `evict`, not this");
    }

    #[test]
    fn restore_prefilling_only_accepts_the_evicted_state() {
        let mut r = req(0, RequestClass::Agent, RequestState::Prefilling);
        assert!(!r.restore_prefilling(), "not suspended: nothing to restore");
    }

    #[test]
    fn both_eviction_detours_are_valid_transitions() {
        assert!(Request::valid_transition(
            RequestState::Prefilling,
            RequestState::Evicted
        ));
        assert!(Request::valid_transition(
            RequestState::Evicted,
            RequestState::Prefilling
        ));
        assert!(Request::valid_transition(
            RequestState::Running,
            RequestState::Evicted
        ));
        assert!(Request::valid_transition(
            RequestState::Evicted,
            RequestState::Running
        ));
        // Still no skipping: an evicted-while-prefilling request cannot
        // resume straight into Admitted's *sibling* states.
        assert!(!Request::valid_transition(
            RequestState::Admitted,
            RequestState::Evicted
        ));
    }
}
