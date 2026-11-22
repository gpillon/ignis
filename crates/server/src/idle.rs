//! Idle unload (spec model-switch/02, ADR 0049, GitHub #317): a model nobody
//! uses for `model.idle_unload` seconds leaves the card — VRAM, pinned host
//! pools, KV-RAM — while the process, the API, `/metrics` and the Playground
//! stay up, and the next request that needs it loads it again.
//!
//! **What counts as use.** A request **in flight** on a route that needs the
//! model, counted from its admission past the API key until its response
//! body — its stream — ends ([`Usage::admit`], `crate::api`'s `count_use`
//! layer), and per job on a Responses socket, never per connection, so an
//! idle socket does not pin the model. A count, not a timestamp of admission:
//! between admission and the engine's submit lie the template render and the
//! media decode, which a timestamp alone would see as idle. And any work
//! still waiting or running on the model ([`crate::ActiveModel::in_flight`],
//! a plain read): a text model's lanes, the image model's jobs, queued or
//! running (spec image-model/05). The routes that only report state — `GET /v1/models`, the
//! config and catalog reads, `/metrics`, preflights — are not counted: a
//! page polling them neither keeps the model loaded nor loads it again.
//!
//! **The clock** ([`crate::Server::spawn_idle_clock`]) ticks about once a
//! second and reads `model.idle_unload` off the live knobs at every tick, so
//! a live `PATCH` takes effect with no reload. The model is unloaded when the
//! knob is above zero, the status is `serving`, nothing is in flight, the
//! model has no work, and the last use is at least the knob ago. While the
//! knob is zero, or the model is in use, or anything but serving, the last
//! use is now: the clock starts when the model is serving and unused, never
//! from a timestamp gone stale meanwhile. The unload itself and the load that
//! ends it are the model switch's (`crate::model_switch`), on its slot.
//!
//! **On request** (GitHub #723, ADR 0105). `POST /v1/ignis/models/unload`
//! unloads the model the clock's way at once, under the same checks
//! ([`unload_unused`]), and `POST /v1/ignis/models/load` loads it again the
//! way a request that needs it does.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::{ModelStatus, Server};

/// How often the idle clock looks: the unload lands within a tick of its
/// due time, and a tick costs a lock and two atomic reads.
pub const IDLE_TICK: Duration = Duration::from_secs(1);

/// The requests using the model now, and when it was last used.
#[derive(Debug)]
pub struct Usage {
    state: Mutex<UsageState>,
}

#[derive(Debug)]
struct UsageState {
    /// Requests admitted on a model route whose response has not ended.
    in_flight: usize,
    /// When the model was last used: the latest admission, end of a
    /// response, or tick that found it in use.
    last_use: Instant,
}

impl Default for Usage {
    /// Nothing in flight, last used now.
    fn default() -> Self {
        Self { state: Mutex::new(UsageState { in_flight: 0, last_use: Instant::now() }) }
    }
}

impl Usage {
    /// One request admitted on a route that uses the model: counted until
    /// the returned guard is dropped — when its response body ends, or its
    /// socket job does.
    pub fn admit(self: &Arc<Self>) -> InUse {
        let mut state = self.state.lock().expect("usage lock");
        state.in_flight += 1;
        state.last_use = Instant::now();
        InUse(Arc::clone(self))
    }

    /// The model is in use now.
    pub fn touch(&self) {
        self.state.lock().expect("usage lock").last_use = Instant::now();
    }

    /// The requests in flight now.
    pub fn in_flight(&self) -> usize {
        self.state.lock().expect("usage lock").in_flight
    }

    /// Run `unload` with how long the model has gone unused, when nothing is
    /// in flight and that is at least `after`, **holding the count's lock**:
    /// an admission waits for it, so the request it admits either kept the
    /// model from being unloaded or finds the status `unloaded` already
    /// stored, and loads the model again — it never reaches an engine being
    /// torn down.
    fn when_idle<T>(&self, after: Duration, unload: impl FnOnce(Duration) -> Option<T>) -> Option<T> {
        let state = self.state.lock().expect("usage lock");
        let idle = state.last_use.elapsed();
        if state.in_flight > 0 || idle < after {
            return None;
        }
        unload(idle)
    }
}

/// A request counted as using the model ([`Usage::admit`]), until dropped.
#[derive(Debug)]
pub struct InUse(Arc<Usage>);

impl Drop for InUse {
    fn drop(&mut self) {
        let mut state = self.0.state.lock().expect("usage lock");
        state.in_flight = state.in_flight.saturating_sub(1);
        state.last_use = Instant::now();
    }
}

/// A response body that holds its request's [`InUse`] until it ends: hyper
/// drops it once its last frame is written, or when the client hangs up
/// mid-stream. Every frame, the size hint and the end are the body's own.
pub(crate) struct Counted {
    body: axum::body::Body,
    _in_use: InUse,
}

impl Counted {
    pub(crate) fn new(body: axum::body::Body, in_use: InUse) -> Self {
        Self { body, _in_use: in_use }
    }
}

impl http_body::Body for Counted {
    type Data = axum::body::Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        std::pin::Pin::new(&mut self.get_mut().body).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.body.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.body.size_hint()
    }
}

/// The running idle clock; dropping it stops the clock. An unload it began
/// still ends whole — its teardown, then its slot freed
/// (`crate::model_switch::unload`).
#[derive(Debug)]
pub struct Clock(tokio::task::JoinHandle<()>);

impl Drop for Clock {
    fn drop(&mut self) {
        self.0.abort();
    }
}

impl Server {
    /// Start the idle clock (spec model-switch/02), looking every `tick`
    /// ([`IDLE_TICK`] when serving): the serve loops start it, a test with a
    /// shorter tick. `None` without a [`crate::model_switch::Switcher`] —
    /// nothing could load the model again.
    pub fn spawn_idle_clock(&self, tick: Duration) -> Option<Clock> {
        self.switcher.as_ref()?;
        let server = self.clone();
        Some(Clock(tokio::spawn(async move {
            let mut ticks = tokio::time::interval(tick);
            ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                ticks.tick().await;
                tick_once(&server).await;
            }
        })))
    }
}

/// One look of the clock: unload the model when it is due, else move its
/// last use to now when it is in use or the clock is off.
async fn tick_once(server: &Server) {
    let after = server.live().idle_unload;
    let model = server.active();
    let serving = matches!(*server.status(), ModelStatus::Serving);
    let busy = serving && !model.in_flight().is_idle();
    if after.is_zero() || !serving || busy || model.source.is_none() {
        server.usage.touch();
        return;
    }
    unload_unused(server, after, |_| true, crate::model_switch::UnloadCause::Idle).await;
}

/// Unload the active model when `meant` says it is the one to unload,
/// nothing is in flight or waiting on it, and its last use is at least
/// `after` ago, all read **under the count's lock** ([`Usage::when_idle`])
/// so no request is admitted onto the engine being torn down; then wait for
/// the teardown. `cause` turns how long the model went unused into why it
/// was unloaded, for its log line. True when this call unloaded it; false
/// when it was in use, not serving, not loaded from an artifact, or a
/// switch, a load or another unload held the switch's slot
/// ([`crate::model_switch::claim_unload_if`]). The clock's due unload is
/// this, and so is `POST /v1/ignis/models/unload` (GitHub #723), with no
/// wait.
pub(crate) async fn unload_unused(
    server: &Server,
    after: Duration,
    meant: impl FnOnce(&crate::ActiveModel) -> bool,
    cause: impl FnOnce(Duration) -> crate::model_switch::UnloadCause,
) -> bool {
    // The model's work is read before the switch's lock is taken: an image
    // model's job queue has a lock of its own.
    let due = server.usage.when_idle(after, |idle| {
        if !server.active().in_flight().is_idle() {
            return None;
        }
        crate::model_switch::claim_unload_if(server, meant).map(|unloading| (unloading, idle))
    });
    let Some((unloading, idle)) = due else {
        return false;
    };
    crate::model_switch::unload(server, unloading, cause(idle)).await;
    true
}
