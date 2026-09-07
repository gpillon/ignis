//! The server's engine: a dedicated model thread owning the core
//! [`Scheduler`] exclusively, plus per-request event routing (GitHub #69).
//!
//! Concurrency model (v2, GitHub #69 — replaces the shared-mutex v1):
//! - **The model thread** — a single, dedicated `std::thread`, spawned once
//!   when the [`Engine`] is constructed and living for the server's whole
//!   life, owns the [`Scheduler`] and the per-request route table
//!   (`streams`) as plain, unshared, thread-owned state. Nothing outside
//!   this thread ever touches either — no `Arc<Mutex<..>>` around them, so
//!   nothing on the async/HTTP side can ever contend a lock with a GPU-bound
//!   `Scheduler::advance()` call.
//! - **The command channel** — [`Engine::submit`] is the one call that
//!   crosses the thread boundary: it sends a command and awaits a one-shot
//!   reply. The model thread drains every queued command *before* each
//!   `advance()`, so command latency is bounded by "at most one decode
//!   step," not by however long the current generation runs.
//! - **`model_id`** — immutable for the server's life, captured once at
//!   construction and read lock-free off the `Engine` handle; it never
//!   touches the model thread.
//! - **Telemetry** — the model thread only ever pushes lightweight facts
//!   (a routed event, a submission notice, a per-step tick) onto an
//!   unbounded channel; a separate async task owns the [`Telemetry`] value
//!   and does all the sink I/O and counter math off the model thread, then
//!   publishes the computed counters into a wait-free [`ArcSwap`] snapshot.

use std::collections::HashMap;
use std::sync::mpsc as std_mpsc;
use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwap;
use ignis_core::{
    FinishReason, RequestClass, RequestId, RequestInput, SchedEvent, Scheduler, SubmitError,
    TokenId,
};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::sync::oneshot;

use crate::telemetry::{
    IntervalCounters, IntervalStatsProvider, NullSink, SystemClock, Telemetry, TelemetryClock,
    TelemetrySink,
};

/// A per-request event stream: the `SchedEvent`s the engine routed to one
/// request. The stream closes (the sender is dropped) when the request
/// completes, so [`UnboundedReceiver::recv`] yielding `None` marks
/// end-of-stream.
pub type EventStream = UnboundedReceiver<SchedEvent>;

/// A request's event route (the engine's side of its stream).
pub type EventRoute = UnboundedSender<SchedEvent>;

/// A command sent from the async/HTTP side to the model thread. `submit` is
/// the only one today (design §"the command channel") — `model_id` is
/// static, cloneable state on the `Engine` handle, and `is_idle` never left
/// the model thread's own loop.
enum Command {
    Submit {
        input: RequestInput,
        class: RequestClass,
        reply: oneshot::Sender<Result<(RequestId, EventStream), SubmitError>>,
    },
}

/// A message on the telemetry consumer's inbox. The model thread only ever
/// sends the first three variants — lightweight facts, never sink I/O,
/// never counter math (mirroring the call sites [`Telemetry`] has: a
/// submission, a routed event, a per-step tick). `SetSink`/`SetStats` are a
/// different kind of message on the same channel: an async-side
/// reconfiguration request the model thread never sends, used by
/// [`Engine::with_telemetry`] / [`Engine::with_stats`] to reach the
/// already-running consumer without ever sharing a lock with the model
/// thread.
enum TelemetryFact {
    Submitted(RequestId),
    Routed(SchedEvent),
    Tick,
    SetSink(Arc<dyn TelemetrySink>),
    SetStats(Arc<dyn IntervalStatsProvider>),
}

/// The server-side engine: a cheap, cloneable handle onto the model thread
/// (GitHub #69) that owns the core [`Scheduler`] exclusively for the
/// server's whole life.
pub struct Engine {
    /// The loaded model id — immutable for the server's life, so it is
    /// captured once here instead of crossing the command channel.
    model_id: String,
    commands: std_mpsc::Sender<Command>,
    facts: UnboundedSender<TelemetryFact>,
    /// The latest interval counters, published wait-free by the telemetry
    /// consumer after each tick (design §"telemetry: computed off the model
    /// thread, published wait-free").
    counters: Arc<ArcSwap<IntervalCounters>>,
}

impl Clone for Engine {
    fn clone(&self) -> Self {
        Self {
            model_id: self.model_id.clone(),
            commands: self.commands.clone(),
            facts: self.facts.clone(),
            counters: Arc::clone(&self.counters),
        }
    }
}

impl Engine {
    /// Wrap a concrete scheduler in a server engine (telemetry off — a no-op
    /// sink, so no JSONL is written; use [`Engine::with_sinks`] to enable it).
    /// Spawns the model thread immediately (server startup, for the
    /// process's whole life).
    pub fn new(scheduler: Box<dyn Scheduler>) -> Self {
        Self::with_sinks(scheduler, Arc::new(NullSink), Arc::new(SystemClock))
    }

    /// Wrap a concrete scheduler in a server engine whose telemetry writes
    /// through `sink`, with `clock` supplying the request-line `ms` /
    /// `tok_s` (a fixed clock keeps tests deterministic — ADR 0006). Spawns
    /// the model thread and the async telemetry consumer immediately.
    pub fn with_sinks(
        scheduler: Box<dyn Scheduler>,
        sink: Arc<dyn TelemetrySink>,
        clock: Arc<dyn TelemetryClock>,
    ) -> Self {
        Self::with_sinks_and_driver(scheduler, sink, clock).0
    }

    /// Same as [`Engine::with_sinks`], but also returns the model thread's
    /// [`std::thread::JoinHandle`] (GitHub #71). Production (`main.rs`)
    /// never needs it — the process exits with the thread still running.
    /// A caller that needs the scheduler's GPU-resident state (weights, KV
    /// cache) fully released before proceeding — e.g. between GPU
    /// integration tests sharing one process — must: drop every clone of
    /// the returned `Engine` (so the command channel disconnects and
    /// `model_thread_loop` returns), then join the handle. Joining blocks
    /// until the model thread has actually exited and dropped the
    /// `Scheduler` it owned, so the next caller never races the GPU
    /// teardown of the previous one.
    pub fn with_sinks_and_driver(
        scheduler: Box<dyn Scheduler>,
        sink: Arc<dyn TelemetrySink>,
        clock: Arc<dyn TelemetryClock>,
    ) -> (Self, std::thread::JoinHandle<()>) {
        let model_id = scheduler.model_id().to_string();
        let (command_tx, command_rx) = std_mpsc::channel();
        let (facts_tx, facts_rx) = unbounded_channel();
        let counters = Arc::new(ArcSwap::from_pointee(IntervalCounters::default()));

        // The model thread: a single, dedicated OS thread that owns the
        // Scheduler + route table exclusively for the server's whole life.
        let facts_tx_for_thread = facts_tx.clone();
        let driver = std::thread::Builder::new()
            .name("ignis-model".into())
            .spawn(move || model_thread_loop(scheduler, command_rx, facts_tx_for_thread))
            .expect("spawning the model thread must not fail");

        // The telemetry consumer: an async task that owns `Telemetry` and
        // does all sink I/O / counter math off the model thread.
        let telemetry = Telemetry::new(sink, clock);
        tokio::spawn(telemetry_task(telemetry, facts_rx, Arc::clone(&counters)));

        (
            Self {
                model_id,
                commands: command_tx,
                facts: facts_tx,
                counters,
            },
            driver,
        )
    }

    /// Route the engine's telemetry through `sink` (keeping the existing
    /// clock and any live counter source). Reconfigures the already-running
    /// telemetry consumer through the facts channel — never touches the
    /// model thread.
    pub fn with_telemetry(self, sink: Arc<dyn TelemetrySink>) -> Self {
        let _ = self.facts.send(TelemetryFact::SetSink(sink));
        self
    }

    /// Use `provider` as the live counter source for the interval line (the
    /// §5 blocker seam: a real `Scheduler::stats` accessor, once core ships
    /// it, overrides the event-derived estimator).
    pub fn with_stats(self, provider: Arc<dyn IntervalStatsProvider>) -> Self {
        let _ = self.facts.send(TelemetryFact::SetStats(provider));
        self
    }

    /// The loaded model id (for `GET /v1/models`) — immutable for the
    /// server's life, read lock-free off this handle (never touches the
    /// model thread).
    pub fn model_id(&self) -> String {
        self.model_id.clone()
    }

    /// The latest interval counters, published wait-free by the telemetry
    /// consumer (a snapshot; never blocks on, or is blocked by, the model
    /// thread or the telemetry consumer).
    pub fn interval_counters(&self) -> IntervalCounters {
        *self.counters.load_full()
    }

    /// Submit a request and attach its event stream. Sends a command to the
    /// model thread and awaits its one-shot reply — the model thread drains
    /// every queued command before its next `advance()`, so this completes
    /// promptly (at most one decode step of latency) even while the model
    /// thread is busy decoding other requests.
    ///
    /// Returns the request's id and its event stream. The stream delivers
    /// every [`SchedEvent`] the engine emits for that request (tokens,
    /// completions, evictions, restorations) and closes when the request
    /// completes.
    pub async fn submit(
        &self,
        input: RequestInput,
        class: RequestClass,
    ) -> Result<(RequestId, EventStream), SubmitError> {
        let (reply, reply_rx) = oneshot::channel();
        self.commands
            .send(Command::Submit { input, class, reply })
            .expect("the model thread outlives every Engine handle");
        reply_rx
            .await
            .expect("the model thread replies to every submit before it can exit")
    }
}

/// The model thread's loop (GitHub #69): drains every queued command
/// (non-blocking), performs exactly one `Scheduler::advance()` when
/// anything is in flight, and blocks on the command channel (no busy-spin)
/// when idle. Returns — a clean shutdown — once every [`Engine`] handle has
/// been dropped (the command channel disconnects).
fn model_thread_loop(
    mut scheduler: Box<dyn Scheduler>,
    commands: std_mpsc::Receiver<Command>,
    facts: UnboundedSender<TelemetryFact>,
) {
    let mut streams: HashMap<RequestId, EventRoute> = HashMap::new();
    loop {
        loop {
            match commands.try_recv() {
                Ok(command) => handle_command(command, &mut *scheduler, &mut streams, &facts),
                Err(std_mpsc::TryRecvError::Empty) => break,
                // Every Engine handle was dropped: clean shutdown (no
                // in-flight request is silently dropped — the process is
                // exiting anyway; nothing left to notify).
                Err(std_mpsc::TryRecvError::Disconnected) => return,
            }
        }
        if !scheduler.is_idle() {
            let events = scheduler.advance();
            route_events(&events, &mut streams, &facts);
            let _ = facts.send(TelemetryFact::Tick);
            continue;
        }
        // Idle: block on the next command instead of busy-spinning. An
        // unbounded `recv()` (rather than a timed wait) is deliberate: with
        // nothing in flight, there is no periodic work to come back for —
        // the only thing that can end the idle period is a new command, so
        // waking on exactly that costs strictly less than a bounded wait
        // that would poll on a timer for no reason (an idle server costs
        // ~no CPU either way, but this is the tighter of the two).
        match commands.recv() {
            Ok(command) => handle_command(command, &mut *scheduler, &mut streams, &facts),
            Err(_) => return,
        }
    }
}

/// Handle one command against the thread-owned scheduler/route table.
fn handle_command(
    command: Command,
    scheduler: &mut dyn Scheduler,
    streams: &mut HashMap<RequestId, EventRoute>,
    facts: &UnboundedSender<TelemetryFact>,
) {
    match command {
        Command::Submit { input, class, reply } => {
            let result = scheduler.submit(input, class).map(|id| {
                let (route, stream) = unbounded_channel();
                streams.insert(id, route);
                let _ = facts.send(TelemetryFact::Submitted(id));
                (id, stream)
            });
            // A dropped receiver (the caller gave up) is not an error here.
            let _ = reply.send(result);
        }
    }
}

/// Route one step's emitted events into their registered per-request
/// streams, and push a copy of each onto the telemetry facts channel.
fn route_events(
    events: &[SchedEvent],
    streams: &mut HashMap<RequestId, EventRoute>,
    facts: &UnboundedSender<TelemetryFact>,
) {
    for event in events {
        // Every event names the request it belongs to — except `Protected`
        // (an admission *batch* event: protection established for the
        // protected head in this step). It isn't per-request stream
        // content, so it is not routed to a stream, but the rest of the
        // batch (which may follow it) still is.
        if let Some(request) = event_request(event) {
            // The route may be gone (the handler already reaped the
            // stream): a failed send is a no-op, not an error.
            if let Some(route) = streams.get(&request) {
                let _ = route.send(event.clone());
            }
            if let SchedEvent::Done { .. } = event {
                // Completion: close the stream (the dropped sender ends the
                // receiver — the `Done` itself was just delivered).
                streams.remove(&request);
            }
        }
        let _ = facts.send(TelemetryFact::Routed(event.clone()));
    }
}

/// The request an event belongs to, or `None` for `Protected` (an
/// admission-batch marker with no single owning request).
fn event_request(event: &SchedEvent) -> Option<RequestId> {
    match event {
        SchedEvent::Token { request, .. }
        | SchedEvent::Done { request, .. }
        | SchedEvent::Admitted { request, .. }
        | SchedEvent::Evicted { request }
        | SchedEvent::Restored { request, .. }
        | SchedEvent::Requeued { request }
        | SchedEvent::PrefixReused { request, .. } => Some(*request),
        SchedEvent::Protected { .. } => None,
    }
}

/// The async telemetry consumer (GitHub #69): owns `Telemetry` and drains
/// the facts channel, calling the exact same methods the old inline driver
/// called — all sink I/O and counter math happens here, off the model
/// thread. After each tick, publishes the computed counters into `counters`
/// (the wait-free `ArcSwap` snapshot).
async fn telemetry_task(
    mut telemetry: Telemetry,
    mut facts: UnboundedReceiver<TelemetryFact>,
    counters: Arc<ArcSwap<IntervalCounters>>,
) {
    while let Some(fact) = facts.recv().await {
        match fact {
            TelemetryFact::Submitted(id) => telemetry.note_submit(id),
            TelemetryFact::Routed(event) => match event {
                SchedEvent::Admitted { request, .. } => telemetry.on_admitted(request),
                SchedEvent::Token { request, .. } => telemetry.on_token(request),
                SchedEvent::Evicted { request } => telemetry.on_evicted(request),
                SchedEvent::Done { request, tokens, .. } => telemetry.on_done(request, tokens),
                _ => {}
            },
            TelemetryFact::Tick => {
                let snapshot = telemetry.emit_interval();
                counters.store(Arc::new(snapshot));
            }
            TelemetryFact::SetSink(sink) => telemetry.set_sink(sink),
            TelemetryFact::SetStats(provider) => telemetry.with_stats(provider),
        }
    }
}

/// Drive a submitted request's stream to completion: collect the generated
/// tokens until the request's [`SchedEvent::Done`] (or its stream closes),
/// returning them alongside why the request stopped (the OpenAI
/// `finish_reason`, GitHub #61 / P1-25). Bounded by `timeout` — a wedged
/// engine must not hang the client forever.
pub async fn collect_tokens(
    rx: &mut EventStream,
    timeout: Duration,
) -> Result<(Vec<TokenId>, FinishReason), CollectError> {
    let deadline = tokio::time::Instant::now() + timeout;
    let mut tokens = Vec::new();
    loop {
        match tokio::time::timeout_at(deadline, rx.recv()).await {
            Ok(Some(event)) => match event {
                SchedEvent::Token { token, .. } => tokens.push(token),
                SchedEvent::Done { reason, .. } => return Ok((tokens, reason)),
                // Other events for this request (admissions, evictions,
                // restorations) do not change the generated-token list —
                // keep draining.
                _ => {}
            },
            // The stream closed without a Done (the engine gave up on the
            // request) or the timeout fired: either way, not completed.
            Ok(None) | Err(_) => return Err(CollectError::NotCompleted),
        }
    }
}

/// The request's stream ended without a completion.
#[derive(Debug)]
pub enum CollectError {
    /// The stream closed without a `Done` before the timeout — the engine
    /// did not finish the request in time (a fault or a wedged compute
    /// step).
    NotCompleted,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use ignis_core::mock::{GatedCompute, MockCompute};
    use ignis_core::{ConcreteScheduler, Compute, DecodeParams, EngineMode, SchedulerConfig};

    /// A test engine: the concrete scheduler over a deterministic mock
    /// compute (ADR 0006 — CPU-only).
    fn test_engine() -> (Engine, Arc<MockCompute>) {
        let compute = Arc::new(MockCompute::new());
        let scheduler = ConcreteScheduler::with_config(
            SchedulerConfig {
                model: "test-model".into(),
                ..SchedulerConfig::default()
            },
            compute.clone(),
        );
        (Engine::new(Box::new(scheduler)), compute)
    }

    fn input(model: &str, tokens: Vec<TokenId>, max_tokens: Option<u32>) -> RequestInput {
        RequestInput {
            model: model.into(),
            tokens,
            params: DecodeParams {
                max_tokens,
                ..DecodeParams::default()
            },
        }
    }

    /// Drain a request's stream to its `Done`, returning the tokens
    /// generated along the way plus why it stopped.
    async fn drain_to_done(rx: &mut EventStream) -> (Vec<TokenId>, FinishReason) {
        let mut tokens = Vec::new();
        loop {
            match rx.recv().await.expect("the stream must reach a Done before closing") {
                SchedEvent::Token { token, .. } => tokens.push(token),
                SchedEvent::Done { reason, .. } => return (tokens, reason),
                _ => {}
            }
        }
    }

    #[tokio::test]
    async fn tokens_route_to_the_request_stream() {
        let (engine, compute) = test_engine();
        let (id, mut rx) = engine
            .submit(input("test-model", vec![1, 2, 3], Some(2)), RequestClass::Interactive)
            .await
            .expect("submit");
        let (tokens, reason) = drain_to_done(&mut rx).await;
        // The mock's deterministic stream (seed 0): pin the exact tokens,
        // not just "some tokens" — proves the routed stream carries the
        // engine's real token ids in order.
        assert_eq!(tokens, vec![compute.token_for(id, 0), compute.token_for(id, 1)]);
        assert_eq!(reason, FinishReason::Length);
    }

    #[tokio::test]
    async fn unknown_model_is_rejected_at_submit() {
        let (engine, _) = test_engine();
        let err = engine
            .submit(input("no-such-model", vec![1], Some(1)), RequestClass::Interactive)
            .await
            .expect_err("submit must fail for an unknown model");
        assert!(matches!(err, SubmitError::UnknownModel(_)));
    }

    #[tokio::test]
    async fn the_stream_closes_on_completion() {
        let (engine, _) = test_engine();
        let (_id, mut rx) = engine
            .submit(input("test-model", vec![1], Some(1)), RequestClass::Interactive)
            .await
            .expect("submit");
        let (tokens, _reason) = drain_to_done(&mut rx).await;
        assert_eq!(tokens.len(), 1);
        // The engine removed the route on Done: the stream is fully closed.
        assert!(rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn an_unbounded_request_completes_at_its_reservation_cap() {
        // `max_tokens: None`: the request reserves `max_sequence_tokens`
        // (8192) and completes exactly there (the reservation is a hard
        // cap — pins that un-capped requests still terminate, so their
        // streams never hang).
        let (engine, _) = test_engine();
        let (_id, mut rx) = engine
            .submit(input("test-model", vec![1], None), RequestClass::Interactive)
            .await
            .expect("submit");
        let (tokens, reason) = drain_to_done(&mut rx).await;
        assert_eq!(tokens.len(), 8192);
        assert_eq!(reason, FinishReason::Length);
    }

    /// A scheduler that emits a single `[Protected, Token, Done]` batch for
    /// one request on its first `advance()` (then nothing). Used to pin
    /// that a `Protected` (admission-batch) event does *not* drop the
    /// `Token`/`Done` events that follow it in the same step's batch.
    struct ProtectedBatchScheduler {
        emitted: bool,
    }

    impl ProtectedBatchScheduler {
        const MODEL: &'static str = "fake-model";
        const ID: RequestId = 42;
    }

    impl Scheduler for ProtectedBatchScheduler {
        fn submit(
            &mut self,
            _input: RequestInput,
            _class: RequestClass,
        ) -> Result<RequestId, SubmitError> {
            Ok(Self::ID)
        }
        fn advance(&mut self) -> Vec<SchedEvent> {
            if self.emitted {
                return Vec::new();
            }
            self.emitted = true;
            // The `Protected` event is an admission-batch marker (protection
            // opened for the head) — in the real scheduler it can precede
            // the decode-phase `Token`/`Done` of the *same* step. The
            // router must skip it and still route what follows.
            vec![
                SchedEvent::Protected {
                    epoch: 1,
                    head: Self::ID,
                    donors: Vec::new(),
                },
                SchedEvent::Token {
                    request: Self::ID,
                    token: 7,
                },
                SchedEvent::Done {
                    request: Self::ID,
                    tokens: 1,
                    reason: FinishReason::Stop,
                },
            ]
        }
        fn is_idle(&self) -> bool {
            self.emitted
        }
        fn model_id(&self) -> &str {
            Self::MODEL
        }
        fn mode(&self) -> EngineMode {
            EngineMode::Serving
        }
    }

    /// A scheduler that emits a single `[PrefixReused, Token, Done]` batch
    /// for one request on its first `advance()` (then nothing). Pins that a
    /// `PrefixReused` (core-07) event is *routed* to the request's stream
    /// (it carries a `request` id, unlike the `Protected` batch marker) and
    /// does not drop the `Token`/`Done` events that follow it.
    struct PrefixReuseBatchScheduler {
        emitted: bool,
    }

    impl Scheduler for PrefixReuseBatchScheduler {
        fn submit(
            &mut self,
            _input: RequestInput,
            _class: RequestClass,
        ) -> Result<RequestId, SubmitError> {
            Ok(ProtectedBatchScheduler::ID)
        }
        fn advance(&mut self) -> Vec<SchedEvent> {
            if self.emitted {
                return Vec::new();
            }
            self.emitted = true;
            // The `PrefixReused` event (a sibling's prefill skipped the
            // cached prefix, core-07) carries the request's id, so the
            // router forwards it to the request's stream (it is not a
            // per-request *content* event, but it is per-request — unlike
            // the `Protected` batch marker, which has no request and is
            // skipped). The subsequent `Token`/`Done` must still route.
            vec![
                SchedEvent::PrefixReused {
                    request: ProtectedBatchScheduler::ID,
                    tokens: 32,
                },
                SchedEvent::Token {
                    request: ProtectedBatchScheduler::ID,
                    token: 7,
                },
                SchedEvent::Done {
                    request: ProtectedBatchScheduler::ID,
                    tokens: 1,
                    reason: FinishReason::Stop,
                },
            ]
        }
        fn is_idle(&self) -> bool {
            self.emitted
        }
        fn model_id(&self) -> &str {
            ProtectedBatchScheduler::MODEL
        }
        fn mode(&self) -> EngineMode {
            EngineMode::Serving
        }
    }

    #[tokio::test]
    async fn a_prefix_reused_event_is_routed_to_the_request_stream() {
        let engine = Engine::new(Box::new(PrefixReuseBatchScheduler { emitted: false }));
        let (id, mut rx) = engine
            .submit(input("fake-model", vec![1], Some(1)), RequestClass::Interactive)
            .await
            .expect("submit");
        assert_eq!(id, ProtectedBatchScheduler::ID);
        let mut saw_reused = false;
        let mut saw_token = false;
        let mut saw_done = false;
        while let Some(event) = rx.recv().await {
            match event {
                SchedEvent::PrefixReused { request, .. } if request == id => saw_reused = true,
                SchedEvent::Token { .. } => saw_token = true,
                SchedEvent::Done { .. } => saw_done = true,
                _ => {}
            }
        }
        assert!(
            saw_reused,
            "a PrefixReused event must be routed to the request's stream"
        );
        assert!(saw_token, "the batch's Token must be routed after a PrefixReused");
        assert!(saw_done, "the batch's Done must be routed after a PrefixReused");
    }

    #[tokio::test]
    async fn a_protected_event_does_not_drop_the_same_batches_events() {
        let engine = Engine::new(Box::new(ProtectedBatchScheduler { emitted: false }));
        let (id, mut rx) = engine
            .submit(input("fake-model", vec![1], Some(1)), RequestClass::Interactive)
            .await
            .expect("submit");
        assert_eq!(id, ProtectedBatchScheduler::ID);
        let mut saw_token = false;
        let mut saw_done = false;
        while let Some(event) = rx.recv().await {
            match event {
                SchedEvent::Token { .. } => saw_token = true,
                SchedEvent::Done { .. } => saw_done = true,
                _ => {}
            }
        }
        assert!(saw_token, "the batch's Token must be routed after a Protected");
        assert!(saw_done, "the batch's Done must be routed after a Protected");
    }

    // ── the primary seam: the isolated model thread (GitHub #69) ──────────

    #[tokio::test]
    async fn a_submit_completes_promptly_while_another_request_is_held_mid_decode() {
        let (gated, controller) = GatedCompute::new(Arc::new(MockCompute::new()));
        let scheduler = ConcreteScheduler::with_config(
            SchedulerConfig {
                model: "test-model".into(),
                ..SchedulerConfig::default()
            },
            gated.clone() as Arc<dyn Compute>,
        );
        let engine = Engine::new(Box::new(scheduler));

        // Arm the gate before anything is submitted: the very first
        // decode_step call (for request A, below) blocks until released.
        gated.arm();
        let (id_a, rx_a) = engine
            .submit(
                input("test-model", vec![1, 2, 3], Some(8000)),
                RequestClass::Interactive,
            )
            .await
            .expect("submit A");

        // Blocks until the model thread is confirmed stuck inside A's first
        // decode_step call — it cannot service anything else right now.
        controller.wait_entered();

        // `model_id` never touches the model thread — it answers instantly
        // even while the thread is stuck inside a compute call.
        assert_eq!(engine.model_id(), "test-model");

        // Submit B while A is held. If the old shared-mutex design were
        // still in place, this would have to wait for A's *entire*
        // generation; here it only has to wait for the model thread to
        // finish draining its command queue, which happens the instant A's
        // held step returns.
        let engine_b = engine.clone();
        let submit_b = tokio::spawn(async move {
            engine_b
                .submit(input("test-model", vec![9], Some(1)), RequestClass::Interactive)
                .await
        });
        // Let the spawned task actually enqueue its command before we
        // release the hold (deterministic: one yield, not a sleep).
        tokio::task::yield_now().await;

        controller.release();

        let (id_b, mut rx_b) = submit_b
            .await
            .expect("the submit task must not panic")
            .expect("submit B");
        assert_ne!(id_a, id_b);

        // Both requests still complete normally once the gate stops
        // interfering with anything further.
        let (_tokens_b, _reason_b) = drain_to_done(&mut rx_b).await;
        // Drop A's stream without draining it to completion — the model
        // thread's shutdown must not hang on an abandoned request.
        drop(rx_a);
    }

    /// Story 22 (GitHub #69): the model thread shuts down cleanly — no
    /// hung thread — once every `Engine` handle (and so every clone of the
    /// command sender) is dropped. A white-box test against
    /// `model_thread_loop` directly, since nothing on the public `Engine`
    /// surface can observe the thread itself exiting.
    #[test]
    fn the_model_thread_exits_once_every_command_sender_is_dropped() {
        let compute = Arc::new(MockCompute::new());
        let scheduler = ConcreteScheduler::with_config(
            SchedulerConfig {
                model: "test-model".into(),
                ..SchedulerConfig::default()
            },
            compute,
        );
        let (command_tx, command_rx) = std_mpsc::channel();
        let (facts_tx, _facts_rx) = unbounded_channel();
        let (done_tx, done_rx) = std_mpsc::channel();
        std::thread::spawn(move || {
            model_thread_loop(Box::new(scheduler), command_rx, facts_tx);
            // Reached only once `model_thread_loop` returns.
            let _ = done_tx.send(());
        });

        // Drop the only command sender — the model thread's next command
        // channel check must see it disconnected and return.
        drop(command_tx);

        done_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("the model thread must exit once every command sender is dropped");
    }
}
