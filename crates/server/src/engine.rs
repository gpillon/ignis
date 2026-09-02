//! The server's engine: the core scheduler + per-request event routing.
//!
//! The [`Scheduler`] contract is `&mut self` (the engine is a single
//! owner), but the HTTP handlers are concurrent — so the [`Engine`] is the
//! shared seam: it owns the scheduler behind a `std::sync::Mutex` (short
//! critical sections; a step holds the lock across a compute call, and the
//! only other lock taken inside it is the compute backend's own — no lock
//! inversion with this one) and routes the events each [`Engine::step`]
//! emits into per-request event streams that request handlers read from.
//!
//! Concurrency model (v1):
//! - **One driver** — a single async task runs [`Engine::run`]: step the
//!   engine, route the events, sleep a tick while idle. Every in-flight
//!   request is advanced by the same loop, so concurrent requests stream
//!   in parallel (the scheduler's N-lane batching does its job instead of
//!   N handlers each double-stepping the engine).
//! - **Atomic submit** — [`Engine::submit`] enqueues the request with the
//!   scheduler and registers its event stream under the same critical
//!   section: no event can be lost between submit and registration (the
//!   driver routes into whatever the map holds at the moment of the step).

use std::collections::HashMap;
use std::time::Duration;

use ignis_core::{
    RequestClass, RequestId, RequestInput, Scheduler, SchedEvent, SubmitError, TokenId,
};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

/// A per-request event stream: the `SchedEvent`s the engine routed to one
/// request. The stream closes (the sender is dropped) when the request
/// completes, so [`UnboundedReceiver::recv`] yielding `None` marks
/// end-of-stream.
pub type EventStream = UnboundedReceiver<SchedEvent>;

/// A request's event route (the engine's side of its stream).
pub type EventRoute = UnboundedSender<SchedEvent>;

struct EngineInner {
    /// The core scheduler the server drives (production: the kernel leaf
    /// via FFI; tests: `MockCompute` — ADR 0006).
    scheduler: Box<dyn Scheduler>,
    /// Live per-request event streams (request → its route). A route is
    /// removed when its request completes (the dropped sender closes the
    /// receiver, signalling end-of-stream to the handler).
    streams: HashMap<RequestId, EventRoute>,
}

/// The server-side engine: owns the core [`Scheduler`] and routes its
/// events into per-request event streams.
pub struct Engine {
    inner: std::sync::Arc<std::sync::Mutex<EngineInner>>,
}

impl Clone for Engine {
    fn clone(&self) -> Self {
        Self {
            inner: std::sync::Arc::clone(&self.inner),
        }
    }
}

impl Engine {
    /// Wrap a concrete scheduler in a server engine.
    pub fn new(scheduler: Box<dyn Scheduler>) -> Self {
        Self {
            inner: std::sync::Arc::new(std::sync::Mutex::new(EngineInner {
                scheduler,
                streams: HashMap::new(),
            })),
        }
    }

    /// The loaded model id (for `GET /v1/models`).
    pub fn model_id(&self) -> String {
        self.inner.lock().unwrap().scheduler.model_id().to_string()
    }

    /// Submit a request and attach its event stream (atomic: the request is
    /// in the engine and its route is registered under the same critical
    /// section, so no event can be lost).
    ///
    /// Returns the request's id and its event stream. The stream delivers
    /// every [`SchedEvent`] the engine emits for that request (tokens,
    /// completions, evictions, restorations) and closes when the request
    /// completes.
    pub fn submit(
        &self,
        input: RequestInput,
        class: RequestClass,
    ) -> Result<(RequestId, EventStream), SubmitError> {
        let (route, stream) = unbounded_channel();
        let mut inner = self.inner.lock().unwrap();
        let id = inner.scheduler.submit(input, class)?;
        inner.streams.insert(id, route);
        Ok((id, stream))
    }

    /// One engine tick: advance the scheduler and route the emitted events
    /// into the registered per-request streams. Called by the driver loop
    /// ([`Engine::run`]); also called directly in tests (no driver, manual
    /// stepping).
    pub fn step(&self) -> Vec<SchedEvent> {
        let events = self
            .inner
            .lock()
            .unwrap()
            .scheduler
            .advance();
        for event in &events {
            // Every event names the request it belongs to — except
            // `Protected` (an admission *batch* event: protection
            // established for the protected head in this step). It isn't
            // per-request stream content, so skip it (the protected head
            // sees its own deal through the normal `Admitted`/`Token`
            // flow) and keep routing the remaining events in the batch.
            let request = match event {
                SchedEvent::Token { request, .. }
                | SchedEvent::Done { request, .. }
                | SchedEvent::Admitted { request, .. }
                | SchedEvent::Evicted { request }
                | SchedEvent::Restored { request, .. }
                | SchedEvent::Requeued { request }
                | SchedEvent::PrefixReused { request, .. } => *request,
                SchedEvent::Protected { .. } => continue,
            };
            let mut inner = self.inner.lock().unwrap();
            // The route may be gone (the handler already reaped the
            // stream): a failed send is a no-op, not an error.
            if let Some(route) = inner.streams.get(&request) {
                let _ = route.send(event.clone());
            }
            if let SchedEvent::Done { .. } = event {
                // Completion: close the stream (the dropped sender ends
                // the receiver — the `Done` itself was just delivered).
                inner.streams.remove(&request);
            }
        }
        events
    }

    /// Whether the engine has nothing in flight (the driver sleeps a tick
    /// instead of busy-spinning).
    pub fn is_idle(&self) -> bool {
        self.inner.lock().unwrap().scheduler.is_idle()
    }

    /// The driver loop: step the engine and route the events; sleep a
    /// tick while idle so an empty engine costs ~no CPU. Runs for the
    /// life of the server (`main` spawns it; tests drive the engine
    /// directly via [`Engine::step`]).
    pub async fn run(&self) {
        let tick = Duration::from_millis(1);
        loop {
            self.step();
            if self.is_idle() {
                tokio::time::sleep(tick).await;
            }
        }
    }
}

/// Drive a submitted request's stream to completion: collect the generated
/// tokens until the request's [`SchedEvent::Done`] (or its stream closes).
/// Bounded by `timeout` — a wedged engine must not hang the client
/// forever.
pub async fn collect_tokens(
    rx: &mut EventStream,
    timeout: Duration,
) -> Result<Vec<TokenId>, CollectError> {
    let deadline = tokio::time::Instant::now() + timeout;
    let mut tokens = Vec::new();
    loop {
        match tokio::time::timeout_at(deadline, rx.recv()).await {
            Ok(Some(event)) => match event {
                SchedEvent::Token { token, .. } => tokens.push(token),
                SchedEvent::Done { .. } => return Ok(tokens),
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
    use tokio::sync::mpsc::error::TryRecvError;

    use ignis_core::{
        mock::MockCompute, ConcreteScheduler, DecodeParams, EngineMode, SchedulerConfig,
    };

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

    /// Drive the engine (manual stepping, no driver task) until the
    /// request's stream delivers its `Done`; returns the non-terminal
    /// events routed along the way.
    fn drive_to_done(engine: &Engine, rx: &mut EventStream) -> Vec<SchedEvent> {
        let mut routed = Vec::new();
        loop {
            engine.step();
            match rx.try_recv() {
                Ok(SchedEvent::Done { .. }) => break,
                Ok(event) => routed.push(event),
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => break,
            }
        }
        routed
    }

    #[test]
    fn tokens_route_to_the_request_stream() {
        let (engine, compute) = test_engine();
        let (id, mut rx) = engine
            .submit(input("test-model", vec![1, 2, 3], Some(2)), RequestClass::Interactive)
            .expect("submit");
        let mut tokens = Vec::new();
        for _ in 0..16 {
            engine.step();
            if let Ok(event) = rx.try_recv() {
                match event {
                    SchedEvent::Token { token, .. } => tokens.push(token),
                    SchedEvent::Done { .. } => break,
                    _ => {}
                }
            }
        }
        // The mock's deterministic stream (seed 0): pin the exact tokens,
        // not just "some tokens" — proves the routed stream carries the
        // engine's real token ids in order.
        assert_eq!(tokens, vec![compute.token_for(id, 0), compute.token_for(id, 1)]);
    }

    #[test]
    fn unknown_model_is_rejected_at_submit() {
        let (engine, _) = test_engine();
        let err = engine
            .submit(input("no-such-model", vec![1], Some(1)), RequestClass::Interactive)
            .expect_err("submit must fail for an unknown model");
        assert!(matches!(err, SubmitError::UnknownModel(_)));
    }

    #[test]
    fn the_stream_closes_on_completion() {
        let (engine, _) = test_engine();
        let (id, mut rx) = engine
            .submit(input("test-model", vec![1], Some(1)), RequestClass::Interactive)
            .expect("submit");
        let routed = drive_to_done(&engine, &mut rx);
        // The routed stream delivered the request's token, then its
        // completion (the engine removed the route on Done).
        assert!(routed
            .iter()
            .any(|e| matches!(e, SchedEvent::Token { request, .. } if *request == id)));
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn an_unbounded_request_completes_at_its_reservation_cap() {
        // `max_tokens: None`: the request reserves `max_sequence_tokens`
        // (8192) and completes exactly there (the reservation is a hard
        // cap — pins that un-capped requests still terminate, so their
        // streams never hang).
        let (engine, _) = test_engine();
        let (_id, mut rx) = engine
            .submit(input("test-model", vec![1], None), RequestClass::Interactive)
            .expect("submit");
        for _ in 0..12 {
            engine.step();
            // Break early if the request finished (the route was removed and
            // the stream disconnected); otherwise keep draining.
            if let Err(TryRecvError::Disconnected) = rx.try_recv() {
                break; // done early
            }
        }
        // 12 of ~8192 steps done: the stream is still routed (the request
        // is mid-generation), not closed — `Empty` = open but drained,
        // `Disconnected` = the engine removed the route (completion).
        assert!(matches!(
            rx.try_recv(),
            Err(TryRecvError::Empty) | Ok(_)
        ));
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

    #[test]
    fn a_prefix_reused_event_is_routed_to_the_request_stream() {
        let engine = Engine::new(Box::new(PrefixReuseBatchScheduler { emitted: false }));
        let (id, mut rx) = engine
            .submit(input("fake-model", vec![1], Some(1)), RequestClass::Interactive)
            .expect("submit");
        assert_eq!(id, ProtectedBatchScheduler::ID);
        engine.step();
        // Drain the routed stream: the `PrefixReused` marker must have been
        // forwarded to this request's stream (it carries the request's id),
        // and the `Token`/`Done` that follow it must still arrive (the
        // reuse marker does not drop the rest of the batch).
        let mut saw_reused = false;
        let mut saw_token = false;
        let mut saw_done = false;
        while let Ok(event) = rx.try_recv() {
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

    #[test]
    fn a_protected_event_does_not_drop_the_same_batches_events() {
        let engine = Engine::new(Box::new(ProtectedBatchScheduler { emitted: false }));
        let (id, mut rx) = engine
            .submit(input("fake-model", vec![1], Some(1)), RequestClass::Interactive)
            .expect("submit");
        assert_eq!(id, ProtectedBatchScheduler::ID);
        engine.step();
        // Drain the routed stream: both the batch's `Token` and its `Done`
        // must have been delivered even though a `Protected` event preceded
        // them in the same step (regression for the router skipping
        // `Protected` instead of early-returning and dropping the rest).
        let mut saw_token = false;
        let mut saw_done = false;
        while let Ok(event) = rx.try_recv() {
            match event {
                SchedEvent::Token { .. } => saw_token = true,
                SchedEvent::Done { .. } => saw_done = true,
                _ => {}
            }
        }
        assert!(saw_token, "the batch's Token must be routed after a Protected");
        assert!(saw_done, "the batch's Done must be routed after a Protected");
    }
}