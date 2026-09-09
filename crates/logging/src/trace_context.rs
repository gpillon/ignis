//! Trace correlation (GitHub #81, ADR 0012, spec §19): a [`LogRecord`]'s
//! `trace_id`/`span_id` come from the `tracing` span an event is nested in,
//! never independently generated. Ignis reuses the request's existing
//! `RequestId` (a `u64`, `ignis_core::RequestId`) as the OTel `trace_id`:
//! whichever span in an event's active scope carries a [`REQUEST_ID_FIELD`]
//! field — recorded at creation, or later via `Span::record` (a span may be
//! opened before the id-bearing value exists, e.g. the HTTP root span at
//! ingress, before the scheduler has assigned one) — supplies it. An event
//! with no active span, or an active span whose scope never records
//! `request_id` anywhere in it, gets neither field: this module never
//! fabricates either identifier (spec §19's "MUST NOT").
//!
//! `span_id` is the *leaf* span's own id (the innermost span active when the
//! event fired) — the id changes as a request moves between admission,
//! prefill, decode-round, and completion spans, while `trace_id` (derived
//! from `request_id`) stays constant across all of them, exactly as OTel's
//! data model intends (spec §19's example: one `trace_id`, a fresh `span_id`
//! per unit of work).
//!
//! Both fields are gated together: a `span_id` is only surfaced when a
//! `trace_id` was also found. An active span that never resolves to a
//! `request_id` (e.g. some future non-request-scoped span) is therefore
//! logged with neither field, same as no active span at all — this crate's
//! notion of "trace" is request correlation, not "some span happened to be
//! open."
//!
//! `JsonLayer` and `PrettyLayer` both call [`on_new_span`]/[`on_record`]/
//! [`resolve`] rather than re-deriving this, so the two never disagree about
//! what "the active trace context" means.

use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record as SpanRecord};
use tracing::{Event, Subscriber};
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;

/// The field name every span that wants trace correlation carries —
/// already the attribute name request-lifecycle events log under
/// (`ignis_server::telemetry`'s `ignis.request.*` events, GitHub #79), so
/// this reuses one vocabulary for "which request" rather than inventing a
/// second `trace.request_id`-shaped key.
pub const REQUEST_ID_FIELD: &str = "request_id";

/// Per-span cached state: the `request_id` this span's own fields declared,
/// if any (never an ancestor's — [`resolve`] walks the scope itself).
#[derive(Clone, Copy, Debug, Default)]
struct SpanCorrelation {
    request_id: Option<u64>,
}

#[derive(Default)]
struct RequestIdVisitor(Option<u64>);

impl Visit for RequestIdVisitor {
    fn record_u64(&mut self, field: &Field, value: u64) {
        if field.name() == REQUEST_ID_FIELD {
            self.0 = Some(value);
        }
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        if field.name() == REQUEST_ID_FIELD
            && let Ok(v) = u64::try_from(value)
        {
            self.0 = Some(v);
        }
    }

    // `request_id` is always logged as a plain integer (`RequestId = u64`);
    // any other value shape (debug/str/bool/...) is not this field and is
    // ignored rather than coerced.
    fn record_debug(&mut self, _field: &Field, _value: &dyn std::fmt::Debug) {}
}

/// `Layer::on_new_span`: capture `request_id` when a newly created span
/// already carries it (the common case — call sites that know the id up
/// front set it directly, not via `Empty` + a later `record`).
pub fn on_new_span<S>(attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>)
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    let mut visitor = RequestIdVisitor::default();
    attrs.record(&mut visitor);
    if let Some(span) = ctx.span(id) {
        span.extensions_mut()
            .insert(SpanCorrelation { request_id: visitor.0 });
    }
}

/// `Layer::on_record`: a span declared `request_id` as
/// [`tracing::field::Empty`] at creation (the HTTP root span, opened before
/// the scheduler assigns an id) and is only now recording it.
pub fn on_record<S>(id: &Id, values: &SpanRecord<'_>, ctx: Context<'_, S>)
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    let mut visitor = RequestIdVisitor::default();
    values.record(&mut visitor);
    let Some(request_id) = visitor.0 else { return };
    let Some(span) = ctx.span(id) else { return };
    let mut ext = span.extensions_mut();
    match ext.get_mut::<SpanCorrelation>() {
        Some(existing) => existing.request_id = Some(request_id),
        None => ext.insert(SpanCorrelation { request_id: Some(request_id) }),
    }
}

/// The `(trace_id, span_id)` pair for `event`: `(None, None)` when no
/// active span carries `request_id` anywhere in scope — including "no
/// active span at all" (spec §19's "MUST NOT fabricate").
pub fn resolve<S>(event: &Event<'_>, ctx: &Context<'_, S>) -> (Option<String>, Option<String>)
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    let Some(scope) = ctx.event_scope(event) else {
        return (None, None);
    };
    let mut leaf_id: Option<Id> = None;
    let mut request_id: Option<u64> = None;
    for span in scope {
        if leaf_id.is_none() {
            leaf_id = Some(span.id());
        }
        if let Some(correlation) = span.extensions().get::<SpanCorrelation>()
            && let Some(rid) = correlation.request_id
        {
            request_id = Some(rid);
            break;
        }
    }
    match (request_id, leaf_id) {
        (Some(rid), Some(leaf)) => {
            (Some(format_trace_id(rid)), Some(format_span_id(leaf.into_u64())))
        }
        _ => (None, None),
    }
}

/// OTel `trace_id` shape: 32 lowercase hex characters (128 bits). Ignis's
/// trace id is a `RequestId` (64 bits, ADR 0012) zero-extended into that
/// width — not 128 fresh random bits, but genuinely the request's own id,
/// which is the whole point (spec §19: never a fabricated replacement).
fn format_trace_id(request_id: u64) -> String {
    format!("{request_id:032x}")
}

/// OTel `span_id` shape: 16 lowercase hex characters (64 bits) — `tracing`'s
/// own span id is already a non-zero `u64`, so this only pads/formats it.
fn format_span_id(span_id: u64) -> String {
    format!("{span_id:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::Layer;

    /// A minimal layer that delegates to this module's own hooks — enough
    /// to drive [`resolve`] in a test without pulling in `JsonLayer`.
    struct ProbeLayer {
        captured: std::sync::Arc<std::sync::Mutex<Vec<(Option<String>, Option<String>)>>>,
    }

    impl<S> Layer<S> for ProbeLayer
    where
        S: Subscriber + for<'a> LookupSpan<'a>,
    {
        fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
            on_new_span(attrs, id, ctx);
        }

        fn on_record(&self, id: &Id, values: &SpanRecord<'_>, ctx: Context<'_, S>) {
            on_record(id, values, ctx);
        }

        fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
            self.captured.lock().unwrap().push(resolve(event, &ctx));
        }
    }

    fn probe(f: impl FnOnce()) -> Vec<(Option<String>, Option<String>)> {
        let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let layer = ProbeLayer { captured: captured.clone() };
        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, f);
        let captured = captured.lock().unwrap();
        captured.clone()
    }

    #[test]
    fn an_event_with_no_active_span_gets_neither_id() {
        let results = probe(|| tracing::info!("no span here"));
        assert_eq!(results, vec![(None, None)]);
    }

    #[test]
    fn a_span_declaring_request_id_at_creation_correlates_its_events() {
        let results = probe(|| {
            let span = tracing::info_span!("ignis.admission", request_id = 42u64);
            let _guard = span.enter();
            tracing::info!("inside admission");
        });
        let (trace_id, span_id) = &results[0];
        assert_eq!(trace_id.as_deref(), Some("0000000000000000000000000000002a"));
        assert!(span_id.is_some());
    }

    #[test]
    fn a_span_recording_request_id_later_still_correlates() {
        let results = probe(|| {
            let span = tracing::info_span!("ignis.http.request", request_id = tracing::field::Empty);
            let _guard = span.enter();
            span.record("request_id", 7u64);
            tracing::info!("after record");
        });
        let (trace_id, _span_id) = &results[0];
        assert_eq!(trace_id.as_deref(), Some("00000000000000000000000000000007"));
    }

    #[test]
    fn a_child_span_inherits_the_parents_request_id() {
        let results = probe(|| {
            let root = tracing::info_span!("ignis.http.request", request_id = 9u64);
            let _root_guard = root.enter();
            let child = tracing::info_span!("ignis.prefill");
            let _child_guard = child.enter();
            tracing::info!("inside child");
        });
        let (trace_id, _span_id) = &results[0];
        assert_eq!(
            trace_id.as_deref(),
            Some("00000000000000000000000000000009"),
            "a child span with no request_id of its own still resolves the ancestor's"
        );
    }

    #[test]
    fn distinct_spans_for_the_same_request_share_trace_id_but_not_span_id() {
        let results = probe(|| {
            {
                let s = tracing::info_span!("ignis.admission", request_id = 3u64);
                let _g = s.enter();
                tracing::info!("admission event");
            }
            {
                let s = tracing::info_span!("ignis.prefill", request_id = 3u64);
                let _g = s.enter();
                tracing::info!("prefill event");
            }
        });
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, results[1].0, "same request -> same trace_id");
        assert_ne!(results[0].1, results[1].1, "different spans -> different span_id");
    }

    #[test]
    fn an_active_span_without_request_id_anywhere_in_scope_gets_neither_id() {
        let results = probe(|| {
            let span = tracing::info_span!("ignis.unrelated");
            let _guard = span.enter();
            tracing::info!("no request id in scope");
        });
        assert_eq!(results, vec![(None, None)]);
    }
}
