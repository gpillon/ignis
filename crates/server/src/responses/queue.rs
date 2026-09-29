//! The socket admission queue (GitHub #282): one server-wide FIFO that a
//! WebSocket request enters when the engine answers it "full", instead of
//! failing as an HTTP request does with a 503.
//!
//! Only the head of the queue tries again, and it tries each time a request
//! leaves the scheduler ([`Engine::freed`]); an admitted head leaves the
//! queue and the next one tries at once. So requests are admitted in arrival
//! order across every connection, and one busy client cannot starve
//! another. A submission that finds others already waiting joins them rather
//! than overtaking them. Only "full" queues: every other refusal is answered
//! at once.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use tokio::sync::watch;

use ignis_core::{RequestClass, RequestId, RequestInput, SubmitError};

use crate::engine::{Engine, EventStream, RequestNotes};
use crate::metrics::Metrics;

/// What every Responses socket on this server shares: the admission queue,
/// and the open-socket count behind its gauge.
#[derive(Default)]
pub struct Hub {
    pub(crate) queue: Arc<AdmissionQueue>,
    sockets: AtomicU64,
    metrics: OnceLock<Arc<Metrics>>,
}

impl Hub {
    /// Keep the two gauges in `metrics` (`--metrics`, ADR 0017).
    pub(crate) fn install_metrics(&self, metrics: Arc<Metrics>) {
        let _ = self.queue.metrics.set(Arc::clone(&metrics));
        let _ = self.metrics.set(metrics);
    }

    /// A socket opened (`opened`) or closed.
    pub(crate) fn socket(&self, opened: bool) {
        let open = if opened {
            self.sockets.fetch_add(1, Ordering::Relaxed) + 1
        } else {
            self.sockets.fetch_sub(1, Ordering::Relaxed) - 1
        };
        if let Some(metrics) = self.metrics.get() {
            metrics.set_responses_sockets(open);
        }
    }
}

/// The server-wide FIFO of socket requests waiting for admission.
#[derive(Default)]
pub(crate) struct AdmissionQueue {
    waiting: Mutex<VecDeque<u64>>,
    next_ticket: AtomicU64,
    /// Changes whenever a ticket leaves, so the next head can notice.
    moved: watch::Sender<()>,
    metrics: OnceLock<Arc<Metrics>>,
}

/// A place in the queue. Dropping it leaves the queue: a request cancelled,
/// or whose connection closed, while it waited.
pub(crate) struct Ticket {
    queue: Arc<AdmissionQueue>,
    number: u64,
}

impl Ticket {
    /// The ticket's number, unique for the server's life: what names a
    /// response the scheduler has not given a request id yet.
    pub(crate) fn number(&self) -> u64 {
        self.number
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        self.queue.leave(self.number);
    }
}

/// What a socket request's submission came to.
pub(crate) enum Admission {
    /// The engine took it.
    Admitted(RequestId, EventStream),
    /// The engine was full, or others were already waiting: it waits.
    Queued(Ticket),
    /// The engine refused it for good (unknown model, context, size).
    Refused(SubmitError),
}

impl AdmissionQueue {
    /// Submit `input`, or queue it when the engine is full.
    pub(crate) async fn submit(
        self: &Arc<Self>,
        engine: &Engine,
        input: &RequestInput,
        class: RequestClass,
        notes: RequestNotes,
    ) -> Admission {
        if self.waiting.lock().expect("queue lock").is_empty() {
            match engine.submit_with_notes(input.clone(), class, notes).await {
                Ok((request, stream)) => return Admission::Admitted(request, stream),
                Err(SubmitError::Full) => {}
                Err(refused) => return Admission::Refused(refused),
            }
        }
        Admission::Queued(self.join())
    }

    /// Wait for `ticket`'s turn and the engine's room, then submit. The
    /// ticket has left the queue when this returns, admitted or refused.
    pub(crate) async fn wait(
        &self,
        ticket: &Ticket,
        engine: &Engine,
        input: &RequestInput,
        class: RequestClass,
        notes: RequestNotes,
    ) -> Result<(RequestId, EventStream), SubmitError> {
        let mut moved = self.moved.subscribe();
        let mut freed = engine.freed();
        loop {
            moved.borrow_and_update();
            if self.waiting.lock().expect("queue lock").front() != Some(&ticket.number) {
                let _ = moved.changed().await;
                continue;
            }
            // Marked seen before the attempt: a request that leaves between
            // the refusal and the wait below still wakes it.
            freed.borrow_and_update();
            match engine.submit_with_notes(input.clone(), class, notes).await {
                Err(SubmitError::Full) => {
                    if freed.changed().await.is_err() {
                        self.leave(ticket.number);
                        return Err(SubmitError::Full);
                    }
                }
                result => {
                    self.leave(ticket.number);
                    return result;
                }
            }
        }
    }

    fn join(self: &Arc<Self>) -> Ticket {
        let number = self.next_ticket.fetch_add(1, Ordering::Relaxed);
        let mut waiting = self.waiting.lock().expect("queue lock");
        waiting.push_back(number);
        self.publish(waiting.len());
        Ticket { queue: Arc::clone(self), number }
    }

    /// Take `number` out of the queue, if it is still in it.
    fn leave(&self, number: u64) {
        let mut waiting = self.waiting.lock().expect("queue lock");
        let Some(at) = waiting.iter().position(|&n| n == number) else {
            return;
        };
        waiting.remove(at);
        self.publish(waiting.len());
        drop(waiting);
        self.moved.send_replace(());
    }

    fn publish(&self, queued: usize) {
        if let Some(metrics) = self.metrics.get() {
            metrics.set_responses_queued(queued as u64);
        }
    }
}
