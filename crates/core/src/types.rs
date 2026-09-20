//! Shared types for the ignis engine's scheduling & state layer.
//!
//! This is the **stable public contract** between the engine (`ignis-core`)
//! and its consumers (the server crate, tests, the kernel leaf). The types
//! here are deliberately small and `Send + Sync` friendly. See
//! `docs/design/ignis-v1.md` §2 and `CONTEXT.md` for the domain vocabulary.

/// Opaque id for a request admitted by the scheduler.
pub type RequestId = u64;

/// Index of a resident decode lane (0..`N_DECODE_LANES`).
pub type LaneId = usize;

/// A single model token, in the loaded model's tokenizer id-space.
pub type TokenId = u32;

/// Fixed number of resident decode lanes (v1: 8, sized for a ~10-subagent
/// concurrent coding workload; overflow goes to the host KV-RAM tier).
pub const N_DECODE_LANES: usize = 8;

/// Why a request's generation stopped (GitHub #61 / P1-25) — the reasons
/// the OpenAI surface reports as `finish_reason`: `stop` (the model's own
/// EOS token), `length` (`max_tokens`, or a scheduler-side reservation cap,
/// reached first), or `error` (the engine gave up on the request).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// The model generated its end-of-sequence token.
    Stop,
    /// `max_tokens` or the engine's KV reservation cap was reached before
    /// EOS.
    Length,
    /// The compute backend failed the request's prefill
    /// `MAX_PREFILL_ATTEMPTS` times in a row (GitHub #166): the request is
    /// ended rather than retried on every advance.
    Error,
}

/// The engine's operating mode (what the scheduler + telemetry report).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineMode {
    /// Normal operation: decode lanes + admission active.
    Serving,
    /// No requests in flight.
    Idle,
}

/// A request as submitted to the scheduler: the prompt is **already
/// tokenized and chat-templated** (the server does that from the artifact's
/// frontend objects). The scheduler works on tokens, not raw messages.
#[derive(Debug, Clone)]
pub struct RequestInput {
    /// Model to route to. v1 loads a single model; validated against
    /// `Scheduler::model_id`.
    pub model: String,
    /// Tokenized + templated prompt.
    pub tokens: Vec<TokenId>,
    /// Generation parameters.
    pub params: DecodeParams,
    /// The prompt's positions, `rope_delta` and media items (GitHub #178),
    /// or `None` for a text-only request — today's path, unchanged.
    pub multimodal: Option<std::sync::Arc<crate::vision::Multimodal>>,
    /// The **generation opener** (GitHub #186, ADR 0029): how many leading
    /// prompt tokens end at the rendered prompt's last
    /// `<|im_start|>assistant\n`, the point where the prompt hands over to the
    /// model, and the last position every later turn of the conversation
    /// provably shares.
    ///
    /// `None` when the frontend could not report one — a prompt with no
    /// opener, or an opener whose byte offset does not tokenize to an exact
    /// token prefix of the prompt. In that case no checkpoint is taken at all,
    /// rather than one taken at a point the tokenizer disagrees about.
    ///
    /// It is a *structural* fact about the rendered prompt, known only to
    /// whoever rendered it and not recoverable from token ids. #188 adds the
    /// end of the system-and-tools block beside it for the same reason.
    pub opener_tokens: Option<u32>,
    /// The **last real user query** (GitHub #187, ADR 0029): how many leading
    /// prompt tokens end where the rendered prompt's last genuine
    /// `<|im_start|>user\n` message *begins* — genuine meaning not a
    /// `<tool_response>`, exactly the distinction the chat template's own
    /// `last_query_index` scan makes.
    ///
    /// It answers one question, at capture time: does a real user message lie
    /// between the checkpoint this request resumed from and the one it is
    /// taking? If it does, this capture opens a new turn and becomes its
    /// conversation's **turn-opening checkpoint**; if it does not, the request
    /// is another iteration of the same tool loop and supersedes what it
    /// claimed ([`crate::checkpoint::CheckpointPool::retain`]).
    ///
    /// `None` when the frontend could not report one — no user message in the
    /// render, or an offset that does not tokenize to an exact token prefix.
    /// A capture then counts as *not* turn-opening, because a wrong `true`
    /// retires the very entry a new user message would have matched.
    pub user_turn_tokens: Option<u32>,
    /// The **system block boundary** (GitHub #188, ADR 0029): how many leading
    /// prompt tokens end the rendered prompt's first
    /// `<|im_start|>system … <|im_end|>\n` — the reasoning instructions, the
    /// tools and the system message, which the template renders as one block.
    ///
    /// The mirror of [`Self::opener_tokens`], and the point a **retained
    /// prefix** is published at. The opener is the last position a
    /// conversation's own later turns share; this is the first position two
    /// *unrelated* requests share, which is all a burst of subagents has: no
    /// subagent's prompt extends its sibling's, so no prompt checkpoint can
    /// ever match between them.
    ///
    /// `None` when the frontend could not report one — a render that does not
    /// open with a system block, or a boundary whose byte offset does not
    /// tokenize to an exact token prefix of the prompt. Then nothing is
    /// published there, rather than a prefix published at a point the
    /// tokenizer disagrees about.
    pub system_block_tokens: Option<u32>,
    /// The **answer tokens** this request is a **decision** over (GitHub
    /// #238, ADR 0034), or `None` for every ordinary request.
    ///
    /// Carrying them here is what makes a decision a *kind* of request
    /// rather than a mode of the engine: the scheduler reads this to decide
    /// that the request ends where its prefill ends. A decision never takes
    /// a decode lane, never enters [`RequestState::Running`], generates
    /// nothing, and finishes with the **readout** of these tokens' logits
    /// at its prompt's last position.
    ///
    /// It also changes what the request reserves. [`DecodeParams::max_tokens`]
    /// is meaningless here — nothing is generated — so the whole-sequence
    /// reservation is the prompt alone, and a decision is admitted on a
    /// prompt that fits however large a `max_tokens` came with it.
    pub decision: Option<std::sync::Arc<[TokenId]>>,
}

impl RequestInput {
    /// Whether this request is a **decision** (GitHub #238): it reads its
    /// answer tokens out at the end of prefill and generates nothing.
    pub fn is_decision(&self) -> bool {
        self.decision.is_some()
    }

    /// How many leading prompt tokens this request may **match** retained
    /// state over: all of them, or one short of the prompt for a request
    /// that must be left something to prefill.
    ///
    /// Two kinds of request must, for the same shape of reason. A
    /// **multimodal** claimant learns its own rope delta only from a prefill
    /// span (ADR 0029), and a **decision** (GitHub #238) reads its answer
    /// off a forward pass — a chunk with nothing left to prefill runs none,
    /// so there would be no logits at the last position to read. A
    /// decision's rendered prompt *ends* at the generation opener, so an
    /// exact repeat would otherwise match state covering every token it has:
    /// the empty-last-chunk trap, prevented rather than repaired, at the
    /// cost of one token's prefill.
    ///
    /// Not to be confused with [`RequestInput::publish_reach`], which
    /// answers the other half of the same picture and does **not** have the
    /// same membership.
    pub fn reuse_reach(&self) -> usize {
        match self.is_decision() || self.multimodal.is_some() {
            true => self.tokens.len().saturating_sub(1),
            false => self.tokens.len(),
        }
    }

    /// How many leading prompt tokens this request may **publish** as a
    /// shared prefix: all of them, or one short of the prompt for a
    /// **decision** (GitHub #238).
    ///
    /// The same arithmetic as [`RequestInput::reuse_reach`] over a
    /// deliberately different set, which is why they are two functions. A
    /// multimodal request publishes its whole head — it is a *claimant* that
    /// needs a tail to prefill, not a publisher — and folding the two
    /// together silently cut every multimodal prefix by a page.
    ///
    /// A decision is in this set because the two halves compose: what it
    /// publishes must be something it can also claim, and it may claim only
    /// `reuse_reach`. Publishing at its own prompt length would create an
    /// entry no decision could ever match, and — through the capture that
    /// rides a page-aligned publish — a prompt checkpoint covering the whole
    /// prompt.
    pub fn publish_reach(&self) -> usize {
        match self.is_decision() {
            true => self.tokens.len().saturating_sub(1),
            false => self.tokens.len(),
        }
    }

    /// The last whole-page boundary at or before `at` that a prefix of this
    /// prompt may end at: the plain page floor, walked back out of any media
    /// item it would land inside (GitHub #193,
    /// [`crate::vision::Multimodal::floor_outside_media`]).
    pub fn prefix_floor(&self, at: u32, page_tokens: u32) -> u32 {
        match &self.multimodal {
            Some(multimodal) => multimodal.floor_outside_media(at, page_tokens),
            None if page_tokens == 0 => 0,
            None => (at / page_tokens) * page_tokens,
        }
    }
}

/// Sampling / decoding parameters for a request.
///
/// The default remains **greedy + fixed seed** (ADR 0007). Positive
/// temperature enables the leaf's stochastic sampler, where the remaining
/// fields are applied per request (P3-04, GitHub #101).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeParams {
    /// Cap on generated tokens (`None` = until EOS / model max).
    pub max_tokens: Option<u32>,
    /// Sampling temperature (`0` = greedy).
    pub temperature: f32,
    /// Nucleus-sampling probability mass (`1` disables top-p filtering).
    pub top_p: f32,
    /// Candidate cap (`0` selects the leaf's 20-candidate default). This is
    /// an ignis extension, not part of the OpenAI Chat Completions standard.
    pub top_k: i32,
    /// One-time penalty for tokens already present in this sequence.
    pub presence_penalty: f32,
    /// Per-occurrence penalty for tokens already present in this sequence.
    pub frequency_penalty: f32,
    /// Sampling seed (fixed for reproducibility / the self-check).
    pub seed: u64,
    /// Keep decoding when the model emits its EOS token. This is reserved
    /// for bounded measurement streams that are cancelled by their window.
    pub ignore_eos: bool,
}

impl Default for DecodeParams {
    fn default() -> Self {
        Self {
            max_tokens: None,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: 0,
            ignore_eos: false,
        }
    }
}

/// The lifecycle of a request inside the engine (`CONTEXT.md`: "request state
/// machine (admit → prefill → decode → done / evict)").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestState {
    /// Accepted, queued, no lane yet.
    Admitted,
    /// Prefill in progress (global prefill lane / batched prefill).
    Prefilling,
    /// Holds a resident decode lane, generating.
    Running,
    /// Evicted to the host KV-RAM tier (core-06, GitHub #125): suspended
    /// with its state retained in host RAM. Reachable from `Running`
    /// (restores back onto a lane, no re-prefill) and from `Prefilling`
    /// (a half-prefilled request — GPU-resident but holding no decode
    /// lane — restores back into `Prefilling` at the chunk boundary it
    /// was snapshotted at, and re-earns a lane the normal way once its
    /// prefill completes).
    Evicted,
    /// Finished (reached `max_tokens` / EOS).
    Done,
}

/// Admission / backfill class for the admission state machine (ADR 0004).
/// Drives protection, backfill priority, and eviction ordering.
///
/// GitHub #120: reachable from HTTP as the **Lane tag** (`CONTEXT.md`: "the
/// request's own statement of its class"), the `class` ignis extension field
/// (an unrecognized or absent wire value maps to `Interactive` — [`Default`]
/// mirrors that so every call site that has not yet heard otherwise gets the
/// same safe default).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum RequestClass {
    /// Foreground interactive request — highest priority, protected from
    /// eviction while active.
    #[default]
    Interactive,
    /// Background agent subtask — the backfill class that fills lanes left
    /// free by interactive traffic.
    Agent,
}

impl RequestClass {
    /// Parse the `class` ignis extension's wire value (GitHub #120):
    /// `"agent"` (case-insensitive) is [`RequestClass::Agent`]; anything
    /// else — including a value nobody has defined yet — maps to
    /// [`RequestClass::Interactive`], the documented safe default. Same
    /// mapping for the "@<lane>" suffix on the `model` field, ignis's second
    /// entry point for this same class.
    pub fn from_extension(value: &str) -> Self {
        if value.eq_ignore_ascii_case("agent") {
            RequestClass::Agent
        } else {
            RequestClass::Interactive
        }
    }

    /// The class a **decision** request is admitted under (GitHub #238,
    /// ADR 0034): the **Lane tag** it stated, or [`RequestClass::Agent`]
    /// when it stated none — where every other route defaults to
    /// [`RequestClass::Interactive`].
    ///
    /// `CONTEXT.md`'s **Lane tag** entry carries why, and carries it once:
    /// the short of it is that a decision has no residency an **eviction
    /// priority** could take and no decode lane to protect, so
    /// `Interactive` would spend a protection it cannot use on behalf of a
    /// conversation that could.
    ///
    /// A tag that was *stated* but not understood still means
    /// `Interactive` — that rule belongs to the tag, not to the route —
    /// which is why this takes an `Option` rather than a sentinel string.
    pub fn for_decision(tag: Option<&str>) -> Self {
        tag.map_or(RequestClass::Agent, RequestClass::from_extension)
    }

    /// The wire string for the `class` extension and the canonical
    /// `ignis.request.*` events (GitHub #120) — the inverse of
    /// [`RequestClass::from_extension`] for the two recognized classes.
    pub fn as_extension_str(&self) -> &'static str {
        match self {
            RequestClass::Interactive => "interactive",
            RequestClass::Agent => "agent",
        }
    }

    /// The class's rank in the **eviction** ordering (ADR 0023): lower ranks
    /// first, so `Agent` (the backfill class — a better victim) is `0` and
    /// `Interactive` is `1`. This is the single definition of "class" in
    /// "request class" for both eviction levels
    /// ([`crate::admission::retained_lane_is_better_victim`] on the GPU,
    /// [`crate::host::HostTier`]'s discard ordering on the host tier) — it
    /// is the *opposite* direction from the derived [`Ord`] on this type,
    /// which orders `Interactive` first for lane-dealing *admission*
    /// priority ([`crate::request::admit_candidates`]); the two orderings
    /// answer different questions and are not interchangeable.
    #[must_use]
    pub fn eviction_rank(&self) -> u8 {
        match self {
            RequestClass::Agent => 0,
            RequestClass::Interactive => 1,
        }
    }
}

/// The backfill class a request was admitted under by the admission state
/// machine (ADR 0004: "the port must preserve invariant behavior
/// (protection promotion, credit decay, frontier distance)").
///
/// A request is only admitted with a non-`None` class while a protection is
/// active (the protected head is blocked by the active set): the class says
/// *how* the request may use the donor's future capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackfillClass {
    /// Admitted without borrowing a protection's future — a normal lane
    /// deal (the head of a blocked queue, or any deal with no protection).
    #[default]
    None,
    /// A *permanent* backfill: it fits the protected head's **future**
    /// capacity (head + non-donor incumbents + all persistent backfills
    /// admitted under this protection + this candidate ≤ pool capacity).
    /// It never borrows the donor's reserved pages.
    Persistent,
    /// A *temporary* borrower: it does not fit the head's future capacity,
    /// but its own service work fits within the protected head's
    /// **frontier distance** (the projected distance to the last still-active
    /// frozen donor) and the protection's **temporal credit**, so it
    /// completes before the donor's future capacity is needed.
    Temporal,
}

/// A request's speculative decode rounds (P5-06, GitHub #154, spec 05): how
/// many verify rounds it ran, how many drafts those rounds proposed, and how
/// many of them were committed. The reference's counters, same names.
///
/// One round is [`SpecCounters::round`]; a request's total is the sum of its
/// rounds. Carried per round on [`crate::DecodeOutcome`] and per request on
/// [`SchedEvent::Done`] — never per token.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpecCounters {
    /// Verify rounds run.
    pub rounds: u32,
    /// Draft tokens proposed to those rounds.
    pub drafted: u32,
    /// Draft tokens committed (the run past its anchor).
    pub accepted: u32,
    /// Rounds that proposed a draft at position `j + 1` (GitHub #160): the
    /// reference's per-slot acceptance profile (`pos=[...]` on its request
    /// log line). A block drafter decaying toward the end of its block and
    /// one that is simply wrong show the same `accepted / drafted`; these
    /// two tell them apart.
    pub drafted_at: [u32; DRAFT_POSITIONS],
    /// Rounds that committed the draft at position `j + 1`.
    pub accepted_at: [u32; DRAFT_POSITIONS],
}

/// Draft positions a round can carry: the widest window a load admits
/// (`MAX_DRAFT_TOKENS`, spec 05).
pub const DRAFT_POSITIONS: usize = crate::speculation::MAX_DRAFT_TOKENS as usize;

impl SpecCounters {
    /// One verify round that proposed `drafted` tokens and committed
    /// `accepted` of them (the first `accepted` of the `drafted`, a run).
    pub fn round(drafted: u32, accepted: u32) -> Self {
        let mut drafted_at = [0; DRAFT_POSITIONS];
        let mut accepted_at = [0; DRAFT_POSITIONS];
        for (j, slot) in drafted_at.iter_mut().enumerate() {
            *slot = u32::from((j as u32) < drafted);
        }
        for (j, slot) in accepted_at.iter_mut().enumerate() {
            *slot = u32::from((j as u32) < accepted);
        }
        Self {
            rounds: 1,
            drafted,
            accepted,
            drafted_at,
            accepted_at,
        }
    }

    /// The per-position acceptance as the reference prints it: the
    /// percentage of rounds that committed position `j + 1` among those that
    /// proposed it, slot 1 first, `-` for a slot never proposed, trailing
    /// never-proposed slots dropped (`"94,72,46"` at a window of 3).
    pub fn acceptance_profile(&self) -> String {
        let proposed = self
            .drafted_at
            .iter()
            .rposition(|&drafted| drafted > 0)
            .map_or(0, |last| last + 1);
        (0..proposed)
            .map(|j| {
                if self.drafted_at[j] == 0 {
                    "-".to_string()
                } else {
                    let percent = 100.0 * f64::from(self.accepted_at[j]) / f64::from(self.drafted_at[j]);
                    format!("{}", percent.round() as u32)
                }
            })
            .collect::<Vec<_>>()
            .join(",")
    }
}

impl std::ops::Add for SpecCounters {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut drafted_at = self.drafted_at;
        let mut accepted_at = self.accepted_at;
        for (mine, theirs) in drafted_at.iter_mut().zip(other.drafted_at) {
            *mine = mine.saturating_add(theirs);
        }
        for (mine, theirs) in accepted_at.iter_mut().zip(other.accepted_at) {
            *mine = mine.saturating_add(theirs);
        }
        Self {
            rounds: self.rounds.saturating_add(other.rounds),
            drafted: self.drafted.saturating_add(other.drafted),
            accepted: self.accepted.saturating_add(other.accepted),
            drafted_at,
            accepted_at,
        }
    }
}

/// An event emitted by a scheduler step. This is what the server streams to
/// clients and what the telemetry writer logs (ADR 0007: the telemetry
/// counters are derived from these events).
#[derive(Debug, Clone)]
pub enum SchedEvent {
    /// A token was committed for a request — one event per token of a
    /// round's run, in order (P5-06, GitHub #154).
    Token { request: RequestId, token: TokenId },
    /// A request completed (`tokens` = total generated this request).
    Done {
        request: RequestId,
        tokens: u32,
        /// Why generation stopped (GitHub #61 / P1-25) — the server maps
        /// this straight to the OpenAI `finish_reason` field.
        reason: FinishReason,
        /// The request's speculative rounds, summed (P5-06, GitHub #154):
        /// `None` when it ran none — a load without speculation — so the
        /// request log never reports placeholder zeros.
        spec: Option<SpecCounters>,
        /// The **readout** a **decision** finished with (GitHub #238, ADR
        /// 0034), and `None` for every other request.
        ///
        /// This is the whole answer. A decision emits no
        /// [`SchedEvent::Token`] and its `tokens` is 0, so without this
        /// field its completion would carry nothing at all — which is why
        /// the readout rides the finish event rather than a second one.
        readout: Option<crate::decision::Readout>,
    },
    /// A request was admitted onto a decode lane. `backfill` is the class
    /// the admission state machine admitted it under (ADR 0004): `None` for
    /// a normal deal, `Persistent` / `Temporal` for a backfill admitted
    /// while a protection is active (see [`BackfillClass`]).
    Admitted {
        request: RequestId,
        lane: LaneId,
        backfill: BackfillClass,
    },
    /// The admission state machine froze a protection (ADR 0004): the
    /// `head` request is blocked by the active set, so the machine froze
    /// the active incumbents and selected `donors` — the earliest-completion
    /// prefix whose release makes the head feasible. No donor is evicted
    /// while the protection is open; backfills admitted under it may not
    /// borrow the donor's reserved pages (see [`BackfillClass`]).
    Protected {
        epoch: u64,
        head: RequestId,
        donors: Vec<RequestId>,
    },
    /// A request was evicted from a decode lane to the host KV-RAM tier
    /// (sibling prefix reuse will restore it instead of re-prefilling).
    /// `snapshot_micros` is the wall time [`Compute::evict`](crate::scheduler::Compute::evict)
    /// took (GitHub #125) — the request log's own attribution of the
    /// tier's cost, alongside [`SchedEvent::Restored::restore_micros`].
    Evicted {
        request: RequestId,
        snapshot_micros: u64,
    },
    /// A request was restored from the host KV-RAM tier (its KV + GDN state
    /// came back from host RAM — no re-prefill, core-06). `lane` is the
    /// decode lane it was restored onto, or `None` for a half-prefilled
    /// request restored back into `Prefilling` (P4-07, GitHub #125), which
    /// holds no lane until its prefill completes. `restore_micros` is the
    /// wall time [`Compute::restore`](crate::scheduler::Compute::restore)
    /// took.
    Restored {
        request: RequestId,
        lane: Option<LaneId>,
        restore_micros: u64,
    },
    /// A request was re-queued for re-prefill (core-06): its host-tier
    /// snapshot was discarded (the tier was full), so it goes back to
    /// `Admitted` and re-prefills from the start.
    Requeued { request: RequestId },
    /// A **live** snapshot was dropped out of the KV-RAM tier to make room
    /// for another one (GitHub #224): the tier's own eviction, the mirror of
    /// [`SchedEvent::Evicted`]'s departure from the device. The owning
    /// request loses every prefilled token and re-prefills from the start.
    ///
    /// A [`SchedEvent::Requeued`] always accompanies this one, but requeue
    /// is *not* the same fact: it also fires when a restore fails, which
    /// costs the same re-prefill without KV-RAM having evicted anything. The
    /// tier's eviction counter is projected from this event and never from
    /// `Requeued`, so the two causes stay apart.
    SnapshotDropped { request: RequestId },
    /// A request's prefill reused a cached prefix (core-07): the `tokens`
    /// leading prompt tokens were skipped — the shared KV prefix is already
    /// warm, so no redundant prefill. Telemetry accumulates these into the
    /// `sibling_prefix_reused_tok` counter (design §5, `server-02`).
    ///
    /// Since GitHub #188 the entry may be a **retained prefix**, whose
    /// publisher has already finished, rather than a live sibling's. The claim
    /// is the same act on the same object, so it is the same event; `retained`
    /// is what tells the two apart (GitHub #190), so sibling reuse and
    /// cross-request reuse are never summed into one number.
    PrefixReused {
        request: RequestId,
        tokens: u32,
        /// The claimed entry's publisher had already finished: this was
        /// reuse of retained state, not of a concurrent sibling's prefix.
        retained: bool,
    },
    /// A request's prefill resumed from **retained state** left by an earlier,
    /// already-finished request (GitHub #186, ADR 0029): the `tokens` leading
    /// prompt tokens — everything up to that state's generation opener — were
    /// not prefilled at all.
    ///
    /// Emitted once per request, on the chunk that actually landed the claim,
    /// so a claim whose prefill batch failed and was retried is reported when
    /// it succeeds and never twice. This is where the request log's
    /// `reuse_source`, `reused_prompt_tokens` and `restore_ms` come from; a
    /// request that reused nothing emits none of it, which is what "source
    /// `none`" means.
    StateReused {
        request: RequestId,
        /// The residency tier the state came from (`device`; #190 adds
        /// `kv_ram`).
        source: crate::checkpoint::ReuseSource,
        /// Leading prompt tokens skipped.
        tokens: u32,
        /// What the restore itself cost — the device-to-device clone of the
        /// checkpoint into this request's slot, measured by the backend.
        restore_micros: u64,
        /// Which kind of retained state the prefill resumed from (GitHub
        /// #216), named by the site that holds the claim rather than left to
        /// be guessed from the tier it came out of.
        kind: crate::checkpoint::RetainedKind,
    },
    /// Something happened to retained state in one residency tier (GitHub
    /// #190) — a hit, a miss, a spill, a discard or a restore, as
    /// [`crate::checkpoint::RetainedStateOperation`] defines each. It names
    /// no request: a spill or a discard happens to state no live request
    /// owns, usually while making room for some other request. `kind` names
    /// which of the two kinds of retained state moved (GitHub #216) — a
    /// prompt checkpoint or a shared prefix — so the two are not read back as
    /// one thing. Every site that reports one holds the blob or entry the
    /// kind comes from, so it is never inferred from `source` or `operation`.
    RetainedState {
        operation: crate::checkpoint::RetainedStateOperation,
        source: crate::checkpoint::ReuseSource,
        kind: crate::checkpoint::RetainedKind,
    },
    /// A publish or a capture `request` would have left was not taken (GitHub
    /// #215, ADR 0030): no retained slot was free and no retained state could
    /// give one up, or a checkpoint's tail page found no KV page. The request
    /// runs regardless; it only leaves no reuse behind.
    RetainedSlotSkipped {
        request: RequestId,
        skip: crate::retained_slot::RetainedSkip,
        /// The retained slots held when it was skipped, of `capacity`.
        in_use: u32,
        capacity: u32,
    },
    /// How many of the load's retained slots are held (GitHub #215), reported
    /// on the step it changed.
    RetainedSlots { in_use: u32, capacity: u32 },
    /// One chunked-prefill step landed for `request` (P3-01, ADR 0018;
    /// P3-06 request log): `chunk_tokens` is this chunk's width,
    /// `prefilled_tokens` the cumulative prompt tokens sent to the compute
    /// backend so far (`== prompt_tokens` once prefill completes).
    /// Diagnosis-only, like `PrefixReused` — never a scheduling signal, and
    /// never a substitute for the live/live gate measurement (ADR 0011).
    PrefillChunk {
        request: RequestId,
        chunk_tokens: u32,
        prefilled_tokens: u32,
        /// Wall time this chunk spent encoding a media item (GitHub #192),
        /// in microseconds — 0 on a text chunk and on one that reuses a
        /// live embedding. Telemetry sums it per request, so a slow
        /// multimodal TTFT is attributable to the encode and not only to
        /// preprocessing.
        encode_micros: u64,
    },
}

/// Errors from submitting a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubmitError {
    /// The engine cannot admit the request right now (all lanes + admission
    /// capacity in use). The caller should retry or queue.
    Full,
    /// The request named a model the engine does not load.
    UnknownModel(String),
    /// The request's KV reservation (prompt + token budget in pages)
    /// exceeds the whole pool — it can never be admitted, even alone.
    /// Rejected at submit rather than left to block the queue forever.
    Oversized,
    /// The request's sequence (prompt + `max_tokens`, or a prompt that
    /// leaves no room to generate when `max_tokens` is absent) is longer
    /// than the engine's per-sequence limit, `max_context` (GitHub #166).
    /// The leaf cannot reserve more than that for one sequence however
    /// empty the pool is, so it is rejected at submit.
    ContextExceeded {
        /// Prompt tokens plus the requested generation budget.
        requested: u64,
        /// The per-sequence limit (`SchedulerConfig::max_sequence_tokens`).
        limit: u32,
    },
}

impl std::fmt::Display for SubmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SubmitError::Full => write!(f, "engine cannot admit the request (full)"),
            SubmitError::UnknownModel(m) => write!(f, "unknown model: {m}"),
            SubmitError::Oversized => write!(
                f,
                "request KV reservation exceeds the whole pool (oversized)"
            ),
            SubmitError::ContextExceeded { requested, limit } => write!(
                f,
                "request needs {requested} tokens (prompt + max_tokens), over the {limit}-token context limit"
            ),
        }
    }
}

impl std::error::Error for SubmitError {}

/// Errors from a compute step (prefill / decode) driven by the scheduler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComputeError {
    /// The kernel leaf reported a CUDA / argument error (return code).
    Kernel(i32),
    /// A request was asked to generate beyond its `max_tokens` / EOS while
    /// still scheduled (a soft stop, not a fault).
    Stopped,
}

impl std::fmt::Display for ComputeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComputeError::Kernel(rc) => write!(f, "kernel error (rc = {rc})"),
            ComputeError::Stopped => write!(f, "request stopped (max_tokens / EOS)"),
        }
    }
}

impl std::error::Error for ComputeError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognized_extension_values_parse_case_insensitively() {
        assert_eq!(RequestClass::from_extension("agent"), RequestClass::Agent);
        assert_eq!(RequestClass::from_extension("Agent"), RequestClass::Agent);
        assert_eq!(RequestClass::from_extension("AGENT"), RequestClass::Agent);
        assert_eq!(
            RequestClass::from_extension("interactive"),
            RequestClass::Interactive
        );
    }

    #[test]
    fn an_unrecognized_or_empty_value_maps_to_the_safe_default() {
        assert_eq!(
            RequestClass::from_extension("classifier"),
            RequestClass::Interactive
        );
        assert_eq!(RequestClass::from_extension(""), RequestClass::Interactive);
    }

    #[test]
    fn default_is_interactive() {
        assert_eq!(RequestClass::default(), RequestClass::Interactive);
    }

    #[test]
    fn as_extension_str_round_trips_through_from_extension() {
        for class in [RequestClass::Interactive, RequestClass::Agent] {
            assert_eq!(RequestClass::from_extension(class.as_extension_str()), class);
        }
    }

    #[test]
    fn a_round_marks_the_positions_it_proposed_and_the_run_it_committed() {
        // GitHub #160: seven drafts, the first three committed.
        let round = SpecCounters::round(7, 3);
        assert_eq!(round.rounds, 1);
        assert_eq!(round.drafted_at, [1; DRAFT_POSITIONS]);
        assert_eq!(round.accepted_at, [1, 1, 1, 0, 0, 0, 0]);
        // A short lane at extent 2, nothing accepted.
        let short = SpecCounters::round(2, 0);
        assert_eq!(short.drafted_at, [1, 1, 0, 0, 0, 0, 0]);
        assert_eq!(short.accepted_at, [0; DRAFT_POSITIONS]);
        // Extent 0: proposes nothing, so no position is counted.
        assert_eq!(SpecCounters::round(0, 0).drafted_at, [0; DRAFT_POSITIONS]);
    }

    #[test]
    fn rounds_add_per_position_and_print_the_reference_profile() {
        // GitHub #160: two full rounds committing 7 and 1, one round at
        // extent 3 committing 2 -- the sum's profile is what the reference's
        // request log prints as `pos=[...]`, rounded, slot 1 first.
        let total = SpecCounters::round(7, 7) + SpecCounters::round(7, 1) + SpecCounters::round(3, 2);
        assert_eq!(total.rounds, 3);
        assert_eq!(total.drafted, 17);
        assert_eq!(total.accepted, 10);
        assert_eq!(total.drafted_at, [3, 3, 3, 2, 2, 2, 2]);
        assert_eq!(total.accepted_at, [3, 2, 1, 1, 1, 1, 1]);
        assert_eq!(total.acceptance_profile(), "100,67,33,50,50,50,50");
        // A window of 3 prints three slots, never the unused tail.
        let narrow = SpecCounters::round(3, 3) + SpecCounters::round(3, 0);
        assert_eq!(narrow.acceptance_profile(), "50,50,50");
        // A request that ran no drafting round has no profile.
        assert_eq!(SpecCounters::round(0, 0).acceptance_profile(), "");
        assert_eq!(SpecCounters::default().acceptance_profile(), "");
    }
}
