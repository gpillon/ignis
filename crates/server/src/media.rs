//! Media acquisition (GitHub #179, spec `.scratch/vision/specs/01-image-input.md`
//! §Wire contract): every `image_url` part of a chat request becomes a
//! [`PreparedMedia`] — its bytes acquired from a base64 `data:` URI or an
//! HTTP(S) URL, decoded, resized and packed — before the request is admitted
//! to a lane.
//!
//! The acquisition policy is the reference's (ninfer
//! `product/media_acquire/acquire.cpp`): a strict base64 decoder, a byte cap
//! checked before decoding and while streaming, credential-free HTTP(S)
//! only, the host resolved first and private addresses refused unless the
//! operator opts in, the connection pinned to the resolved address, bounded
//! redirects each re-checked, no proxy. Every refusal is an HTTP 400 carrying
//! the reference's code name (the reference answers fetch failures 502/504;
//! the spec keeps every media error a 400).
//!
//! Preparation runs on a bounded host pool off the async runtime, behind a
//! host [`MediaCache`] keyed by content digest with single-flight for
//! concurrent identical misses. Dropping the acquisition future — a client
//! disconnect — stops the work at its next checkpoint.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ignis_artifact::vision::{
    Budget, InvalidMedia, PreparedMedia, ProcessorError, ProcessorOptions, VisionProcessor,
};
use sha2::{Digest, Sha256};
use tokio::sync::{watch, Semaphore};

use crate::template::{ChatMessage, MessageContent};

/// Why a request's media could not be acquired: an OpenAI error `code` and a
/// message naming the offending message and part. Raised before admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaRejection {
    /// The HTTP status (400, or 504 when the request deadline passed while
    /// its media were being prepared).
    pub status: u16,
    /// `invalid_media`, `media_budget_exceeded`, `media_fetch_failed`,
    /// `media_fetch_timeout` or `request_timeout`.
    pub code: &'static str,
    pub message: String,
}

/// What acquiring a request's media cost, for its `ignis.request.*` events.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MediaStats {
    /// Media items in the request.
    pub items: u32,
    /// Merged vision tokens across the items.
    pub vision_tokens: u64,
    /// Acquired (encoded) bytes across the items.
    pub media_bytes: u64,
    /// Wall time from the first acquisition to the last prepared item.
    pub preprocess_seconds: f64,
    /// Items served from the media cache (including single-flight joins).
    pub cache_hits: u32,
    /// Items this request prepared itself.
    pub cache_misses: u32,
}

/// Turns one image's acquired bytes into its prepared patches — the unit of
/// work the media cache stores and the host pool runs. [`VisionProcessor`] in
/// production; a test double where a test needs to hold or count the work.
pub trait Preparer: Send + Sync + 'static {
    /// Decode, resize and pack media item `item` (0-based, prompt order).
    /// `cancelled` turns true once no request wants the result; an
    /// implementation may return early (with any error) when it does.
    fn prepare_media(
        &self,
        item: usize,
        bytes: &[u8],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<PreparedMedia, ProcessorError>;
}

/// The processor has no checkpoint inside one image's decode and resize;
/// the pool checks before and after it.
impl Preparer for VisionProcessor {
    fn prepare_media(
        &self,
        item: usize,
        bytes: &[u8],
        _cancelled: &dyn Fn() -> bool,
    ) -> Result<PreparedMedia, ProcessorError> {
        VisionProcessor::prepare_media(self, item, bytes)
    }
}

/// Why acquiring one item's bytes failed, before it is placed in a request.
#[derive(Debug, Clone, PartialEq, Eq)]
enum AcquireError {
    /// Malformed source, disallowed address, empty body.
    Invalid(String),
    /// The item would exceed the request's media byte budget.
    Budget(String),
    /// The remote could not be reached or answered an error.
    FetchFailed(String),
    /// The remote did not answer within the fetch timeouts.
    FetchTimeout(String),
    /// The request deadline passed.
    Deadline,
}

impl AcquireError {
    fn rejection(self, at: &str) -> MediaRejection {
        let (status, code, why) = match self {
            Self::Invalid(why) => (400, "invalid_media", why),
            Self::Budget(why) => (400, "media_budget_exceeded", why),
            Self::FetchFailed(why) => (400, "media_fetch_failed", why),
            Self::FetchTimeout(why) => (400, "media_fetch_timeout", why),
            Self::Deadline => (504, "request_timeout", "the request deadline passed while its media were acquired".to_owned()),
        };
        MediaRejection { status, code, message: format!("{at}: {why}") }
    }
}

/// One image part of a request, in prompt order.
#[derive(Debug, Clone, PartialEq, Eq)]
struct MediaPart {
    /// `message {i} content part {j}`, as every content-part error names it.
    at: String,
    url: String,
}

/// The request's `image_url` parts in prompt order (message order, then part
/// order — the order the chat template renders their placeholders in).
/// `check_content_parts` has already refused every malformed part.
fn image_parts(messages: &[ChatMessage]) -> Vec<MediaPart> {
    let mut out = Vec::new();
    for (i, message) in messages.iter().enumerate() {
        let MessageContent::Parts(parts) = &message.content else {
            continue;
        };
        for (j, part) in parts.iter().enumerate() {
            if let ("image_url", Some(url)) = (part.kind.as_deref().unwrap_or_default(), &part.url) {
                out.push(MediaPart { at: format!("message {i} content part {j}"), url: url.clone() });
            }
        }
    }
    out
}

/// Where an image part's bytes come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MediaSource<'a> {
    /// A `data:` URI (the whole value).
    Data(&'a str),
    /// An `http`/`https` URL.
    Http(&'a str),
}

fn media_source(url: &str) -> Result<MediaSource<'_>, AcquireError> {
    let scheme = url.split_once(':').map(|(scheme, _)| scheme.to_ascii_lowercase());
    match scheme.as_deref() {
        Some("data") => Ok(MediaSource::Data(url)),
        Some("http" | "https") => Ok(MediaSource::Http(url)),
        _ => Err(AcquireError::Invalid("image_url must be a base64 data URI or an HTTP(S) URL".to_owned())),
    }
}

/// A base64 `data:` URI's bytes, the reference's way: the header must name
/// `;base64`, an encoded length that cannot fit `max_bytes` is refused before
/// decoding, and an empty payload is invalid.
fn decode_data_uri(value: &str, max_bytes: u64) -> Result<Vec<u8>, AcquireError> {
    let malformed = || AcquireError::Invalid("media data source must be a base64 data URI".to_owned());
    let (header, payload) = value.split_once(',').ok_or_else(malformed)?;
    if !header.starts_with("data:") || !header.contains(";base64") {
        return Err(malformed());
    }
    let over = || AcquireError::Budget(format!("media data exceeds the request media byte budget ({max_bytes} bytes)"));
    if payload.len() as u64 > (max_bytes / 3 + 1) * 4 {
        return Err(over());
    }
    let bytes = decode_base64(payload)?;
    if bytes.len() as u64 > max_bytes {
        return Err(over());
    }
    if bytes.is_empty() {
        return Err(AcquireError::Invalid("media source is empty".to_owned()));
    }
    Ok(bytes)
}

/// The reference's strict standard-alphabet base64: ASCII whitespace is
/// skipped, `=` ends the data (anything but padding after it is an error),
/// and a dangling six-bit group is an error.
fn decode_base64(text: &str) -> Result<Vec<u8>, AcquireError> {
    let value = |c: u8| match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    };
    let mut out = Vec::with_capacity(text.len() * 3 / 4);
    let (mut bits, mut count, mut padded) = (0u32, 0u32, false);
    for c in text.bytes() {
        if c == b'=' {
            padded = true;
            continue;
        }
        if matches!(c, b' ' | b'\n' | b'\r' | b'\t') {
            continue;
        }
        let digit = value(c)
            .filter(|_| !padded)
            .ok_or_else(|| AcquireError::Invalid("malformed base64 media data".to_owned()))?;
        bits = (bits << 6) | digit as u32;
        count += 6;
        if count >= 8 {
            count -= 8;
            out.push((bits >> count) as u8);
        }
    }
    if count >= 6 {
        return Err(AcquireError::Invalid("malformed base64 media padding".to_owned()));
    }
    Ok(out)
}

// ── HTTP(S) fetch ───────────────────────────────────────────────────────────

/// Which resolved addresses a fetch may connect to.
pub type AddressFilter = Arc<dyn Fn(IpAddr) -> bool + Send + Sync>;

/// The network half of acquisition, fixed at load.
#[derive(Clone)]
pub struct MediaPolicy {
    /// `--media-allow-private-network`: fetch from private, loopback,
    /// link-local, multicast and CGNAT addresses too.
    pub allow_private_network: bool,
    /// `--media-cache-mib`, in bytes: prepared patches retained for reuse
    /// (0 retains nothing).
    pub cache_bytes: u64,
    /// Redirects followed, each re-checked (the reference's 3).
    pub max_redirects: u32,
    /// Connect timeout, bounded further by the request deadline.
    pub connect_timeout: Duration,
    /// Whole-fetch timeout, bounded further by the request deadline.
    pub fetch_timeout: Duration,
    /// Host preparation workers.
    pub workers: usize,
    address_filter: Option<AddressFilter>,
}

impl MediaPolicy {
    /// The reference's policy (5 s connect, 60 s total, 3 redirects) with the
    /// operator's two choices.
    pub fn new(allow_private_network: bool, cache_bytes: u64) -> Self {
        Self {
            allow_private_network,
            cache_bytes,
            max_redirects: 3,
            connect_timeout: Duration::from_secs(5),
            fetch_timeout: Duration::from_secs(60),
            workers: std::thread::available_parallelism().map_or(1, |n| n.get()).min(16),
            address_filter: None,
        }
    }

    /// Replace the private-address rule with `filter` (tests: a loopback
    /// test server that redirects to a loopback address the filter refuses).
    pub fn with_address_filter(mut self, filter: AddressFilter) -> Self {
        self.address_filter = Some(filter);
        self
    }

    fn address_allowed(&self, address: IpAddr) -> bool {
        match &self.address_filter {
            Some(filter) => filter(address),
            None => self.allow_private_network || !private_address(address),
        }
    }
}

impl fmt::Debug for MediaPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MediaPolicy")
            .field("allow_private_network", &self.allow_private_network)
            .field("cache_bytes", &self.cache_bytes)
            .field("max_redirects", &self.max_redirects)
            .field("workers", &self.workers)
            .finish_non_exhaustive()
    }
}

/// The reference's disallowed ranges (`acquire.cpp` `private_ipv4` /
/// `private_address`): IPv4 0/8, 10/8, 127/8, 169.254/16, 172.16/12,
/// 192.168/16, CGNAT 100.64/10, benchmarking 198.18/15 and everything from
/// 224/4 up; IPv6 unspecified, loopback, link-local, multicast, unique-local,
/// and a v4-mapped address by its IPv4 rule.
pub fn private_address(address: IpAddr) -> bool {
    let v4 = |a: Ipv4Addr| {
        let a = u32::from(a);
        a >> 24 == 0
            || a >> 24 == 10
            || a >> 24 == 127
            || a >> 16 == 0xa9fe
            || a >> 20 == 0xac1
            || a >> 16 == 0xc0a8
            || a >> 22 == 0x0191
            || a >> 17 == 0x633f
            || a >> 24 >= 224
    };
    match address {
        IpAddr::V4(a) => v4(a),
        IpAddr::V6(a) => {
            if let Some(mapped) = a.to_ipv4_mapped() {
                return v4(mapped);
            }
            let first = a.segments()[0];
            a.is_unspecified()
                || a.is_loopback()
                || first & 0xffc0 == 0xfe80
                || first & 0xff00 == 0xff00
                || first & 0xfe00 == 0xfc00
        }
    }
}

/// Resolve `host` and pick the address to pin: the first allowed IPv4, else
/// the last allowed IPv6 (the reference's choice).
async fn resolve_allowed(host: &str, port: u16, policy: &MediaPolicy) -> Result<SocketAddr, AcquireError> {
    let literal = host.trim_start_matches('[').trim_end_matches(']');
    let addresses: Vec<SocketAddr> = match literal.parse::<IpAddr>() {
        Ok(ip) => vec![SocketAddr::new(ip, port)],
        Err(_) => tokio::net::lookup_host((host, port))
            .await
            .map_err(|e| AcquireError::FetchFailed(format!("failed to resolve media URL host: {e}")))?
            .collect(),
    };
    let mut selected = None;
    for address in addresses.into_iter().filter(|a| policy.address_allowed(a.ip())) {
        selected = Some(address);
        if address.is_ipv4() {
            break;
        }
    }
    selected.ok_or_else(|| AcquireError::Invalid("media URL resolves only to disallowed network addresses".to_owned()))
}

fn fetch_error(error: reqwest::Error) -> AcquireError {
    if error.is_timeout() {
        AcquireError::FetchTimeout(format!("media URL fetch timed out: {error}"))
    } else {
        AcquireError::FetchFailed(format!("failed to fetch media URL: {error}"))
    }
}

/// An HTTP(S) image's bytes, at most `max_bytes` of them.
async fn fetch(url: &str, policy: &MediaPolicy, max_bytes: u64, deadline: Instant) -> Result<Vec<u8>, AcquireError> {
    let invalid = || AcquireError::Invalid("media URL must be credential-free HTTP(S)".to_owned());
    let mut url = reqwest::Url::parse(url).map_err(|_| AcquireError::Invalid("invalid media URL".to_owned()))?;
    for hop in 0..=policy.max_redirects {
        if !matches!(url.scheme(), "http" | "https") || !url.username().is_empty() || url.password().is_some() {
            return Err(invalid());
        }
        let host = url.host_str().filter(|h| !h.is_empty()).ok_or_else(invalid)?.to_owned();
        let port = url.port_or_known_default().ok_or_else(invalid)?;
        let address = resolve_allowed(&host, port, policy).await?;
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(AcquireError::Deadline);
        }
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(policy.connect_timeout.min(remaining))
            .timeout(policy.fetch_timeout.min(remaining))
            .resolve(host.trim_start_matches('[').trim_end_matches(']'), address)
            .user_agent("ignis/vision")
            .build()
            .map_err(fetch_error)?;
        let mut response = client.get(url.clone()).send().await.map_err(fetch_error)?;
        let status = response.status();
        if status.is_success() {
            let over = || AcquireError::Budget(format!("media URL exceeds the request media byte budget ({max_bytes} bytes)"));
            if response.content_length().is_some_and(|n| n > max_bytes) {
                return Err(over());
            }
            let mut body = Vec::new();
            while let Some(chunk) = response.chunk().await.map_err(fetch_error)? {
                if (body.len() + chunk.len()) as u64 > max_bytes {
                    return Err(over());
                }
                body.extend_from_slice(&chunk);
            }
            if body.is_empty() {
                return Err(AcquireError::Invalid("media source contains no data".to_owned()));
            }
            return Ok(body);
        }
        if !status.is_redirection() || hop == policy.max_redirects {
            return Err(AcquireError::FetchFailed(format!("media URL returned HTTP {}", status.as_u16())));
        }
        let location = response
            .headers()
            .get(reqwest::header::LOCATION)
            .and_then(|v| v.to_str().ok())
            .filter(|v| !v.is_empty())
            .ok_or_else(|| AcquireError::FetchFailed("media URL redirect has no location".to_owned()))?;
        url = url
            .join(location)
            .map_err(|_| AcquireError::FetchFailed("media URL redirect location is not a URL".to_owned()))?;
    }
    Err(AcquireError::FetchFailed("too many media URL redirects".to_owned()))
}

// ── the media cache ─────────────────────────────────────────────────────────

/// What a cached payload was prepared as. Images only until video lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Modality {
    Image,
}

type CacheKey = ([u8; 32], Modality);

/// A preparation's outcome: the payload and the seconds it took to build.
type BuildResult = Result<(Arc<PreparedMedia>, f64), BuildError>;

#[derive(Debug, Clone)]
enum BuildError {
    Processor(ProcessorError),
    /// Every request waiting on it went away.
    Cancelled,
}

/// One in-flight preparation, shared by the request that started it and any
/// request that sent the same bytes meanwhile.
struct Flight {
    /// Requests still waiting; changed only under the cache lock. The build
    /// stops at its next checkpoint once it reaches zero.
    interested: AtomicUsize,
    done: watch::Sender<Option<BuildResult>>,
}

struct Entry {
    media: Arc<PreparedMedia>,
    bytes: u64,
    last_used: u64,
}

#[derive(Default)]
struct CacheState {
    ready: HashMap<CacheKey, Entry>,
    /// `last_used` → key, oldest first.
    lru: BTreeMap<u64, CacheKey>,
    retained_bytes: u64,
    clock: u64,
    inflight: HashMap<CacheKey, Arc<Flight>>,
}

impl CacheState {
    fn touch(&mut self, key: &CacheKey) -> Option<Arc<PreparedMedia>> {
        self.clock += 1;
        let entry = self.ready.get_mut(key)?;
        self.lru.remove(&entry.last_used);
        entry.last_used = self.clock;
        self.lru.insert(self.clock, *key);
        Some(Arc::clone(&entry.media))
    }

    fn retain(&mut self, key: CacheKey, media: Arc<PreparedMedia>, capacity: u64) {
        let bytes = media.patches.len() as u64 * 2;
        if bytes > capacity || self.ready.contains_key(&key) {
            return;
        }
        while self.retained_bytes + bytes > capacity {
            let Some((_, oldest)) = self.lru.pop_first() else { break };
            if let Some(evicted) = self.ready.remove(&oldest) {
                self.retained_bytes -= evicted.bytes;
            }
        }
        self.clock += 1;
        self.lru.insert(self.clock, key);
        self.ready.insert(key, Entry { media, bytes, last_used: self.clock });
        self.retained_bytes += bytes;
    }
}

/// Prepared patch payloads keyed by (content digest, modality), LRU under a
/// byte capacity, with single-flight for concurrent identical misses.
struct MediaCache {
    capacity: u64,
    state: Mutex<CacheState>,
}

/// How a request got an item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Disposition {
    Hit,
    Joined,
    Produced,
}

/// A request's claim on one item's preparation. Dropping it withdraws the
/// request's interest.
struct Ticket {
    cache: Arc<MediaCache>,
    disposition: Disposition,
    hit: Option<Arc<PreparedMedia>>,
    flight: Option<Arc<Flight>>,
}

impl Drop for Ticket {
    fn drop(&mut self) {
        if let Some(flight) = &self.flight {
            let _state = self.cache.state.lock().unwrap_or_else(|e| e.into_inner());
            flight.interested.fetch_sub(1, Ordering::SeqCst);
        }
    }
}

impl Ticket {
    async fn wait(&self) -> BuildResult {
        if let Some(media) = &self.hit {
            return Ok((Arc::clone(media), 0.0));
        }
        let flight = self.flight.as_ref().expect("a ticket without a hit has a flight");
        let mut done = flight.done.subscribe();
        let result = done.wait_for(Option::is_some).await.expect("the flight's sender lives in the ticket");
        result.clone().expect("waited for Some")
    }
}

impl MediaCache {
    fn new(capacity: u64) -> Self {
        Self { capacity, state: Mutex::new(CacheState::default()) }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, CacheState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Claim `bytes` (digest `digest`): a hit, a join onto an identical
    /// in-flight preparation, or a new preparation on `pool`.
    fn begin(
        self: &Arc<Self>,
        digest: [u8; 32],
        item: usize,
        bytes: Vec<u8>,
        preparer: &Arc<dyn Preparer>,
        pool: &Arc<Semaphore>,
    ) -> Ticket {
        let key = (digest, Modality::Image);
        let mut state = self.lock();
        let ticket = |disposition, hit, flight| Ticket { cache: Arc::clone(self), disposition, hit, flight };
        if let Some(media) = state.touch(&key) {
            return ticket(Disposition::Hit, Some(media), None);
        }
        if let Some(flight) = state.inflight.get(&key) {
            flight.interested.fetch_add(1, Ordering::SeqCst);
            return ticket(Disposition::Joined, None, Some(Arc::clone(flight)));
        }
        let flight = Arc::new(Flight { interested: AtomicUsize::new(1), done: watch::Sender::new(None) });
        state.inflight.insert(key, Arc::clone(&flight));
        drop(state);
        tokio::spawn(produce(Arc::clone(self), key, item, bytes, Arc::clone(&flight), Arc::clone(preparer), Arc::clone(pool)));
        ticket(Disposition::Produced, None, Some(flight))
    }

    /// End `flight`, if nobody is waiting: remove it and publish
    /// cancellation. Decided under the lock joiners take, so a request can
    /// never join a flight that has already given up.
    fn abandon_if_unwanted(&self, key: &CacheKey, flight: &Flight) -> bool {
        let mut state = self.lock();
        if flight.interested.load(Ordering::SeqCst) != 0 {
            return false;
        }
        state.inflight.remove(key);
        flight.done.send_replace(Some(Err(BuildError::Cancelled)));
        true
    }
}

/// Build one flight on the pool: wait for a worker, prepare (checking
/// interest before, during and after), retain the payload and publish it.
async fn produce(
    cache: Arc<MediaCache>,
    key: CacheKey,
    item: usize,
    bytes: Vec<u8>,
    flight: Arc<Flight>,
    preparer: Arc<dyn Preparer>,
    pool: Arc<Semaphore>,
) {
    let Ok(permit) = pool.acquire_owned().await else { return };
    if cache.abandon_if_unwanted(&key, &flight) {
        return;
    }
    let worker = {
        let (cache, flight) = (Arc::clone(&cache), Arc::clone(&flight));
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            loop {
                let started = Instant::now();
                let stopped = AtomicBool::new(false);
                let cancelled = || {
                    let unwanted = flight.interested.load(Ordering::SeqCst) == 0;
                    if unwanted {
                        stopped.store(true, Ordering::SeqCst);
                    }
                    unwanted
                };
                let built = preparer.prepare_media(item, &bytes, &cancelled);
                if cache.abandon_if_unwanted(&key, &flight) {
                    return;
                }
                // A build that bailed on a moment with no waiters, joined
                // since: build again for the request that joined.
                if built.is_err() && stopped.load(Ordering::SeqCst) {
                    continue;
                }
                let seconds = started.elapsed().as_secs_f64();
                let result = built.map(|media| (Arc::new(media), seconds)).map_err(BuildError::Processor);
                let mut state = cache.lock();
                if let Ok((media, _)) = &result {
                    state.retain(key, Arc::clone(media), cache.capacity);
                }
                state.inflight.remove(&key);
                flight.done.send_replace(Some(result));
                return;
            }
        })
    };
    if worker.await.is_err() {
        // The preparer panicked: fail the flight rather than strand waiters.
        let mut state = cache.lock();
        state.inflight.remove(&key);
        let error = ProcessorError::InvalidMedia {
            item,
            reason: InvalidMedia::Undecodable("media preparation failed".to_owned()),
        };
        flight.done.send_replace(Some(Err(BuildError::Processor(error))));
    }
}

// ── the acquirer ────────────────────────────────────────────────────────────

/// A request's prepared media, in prompt order, and what they cost.
#[derive(Debug, Clone)]
pub struct AcquiredMedia {
    pub media: Vec<PreparedMedia>,
    pub stats: MediaStats,
}

/// Acquires and prepares a request's media (see the module doc). One per
/// load, shared by every handler.
pub struct MediaAcquirer {
    preparer: Arc<dyn Preparer>,
    limits: ProcessorOptions,
    policy: MediaPolicy,
    cache: Arc<MediaCache>,
    pool: Arc<Semaphore>,
}

impl fmt::Debug for MediaAcquirer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MediaAcquirer").field("limits", &self.limits).field("policy", &self.policy).finish()
    }
}

impl MediaAcquirer {
    /// An acquirer preparing with `preparer` under the per-request `limits`
    /// (the processor's own options) and the network/cache `policy`.
    pub fn new(preparer: Arc<dyn Preparer>, limits: ProcessorOptions, policy: MediaPolicy) -> Self {
        let pool = Arc::new(Semaphore::new(policy.workers.max(1)));
        let cache = Arc::new(MediaCache::new(policy.cache_bytes));
        Self { preparer, limits, policy, cache, pool }
    }

    /// Requests currently waiting on an in-flight preparation, counted once
    /// per item they wait for.
    pub fn waiting_requests(&self) -> usize {
        self.cache.lock().inflight.values().map(|f| f.interested.load(Ordering::SeqCst)).sum()
    }

    /// Acquire and prepare every `image_url` part of `messages`, before
    /// `deadline`. Budgets are enforced as the items arrive, before any
    /// device work; dropping the returned future stops the work.
    pub async fn acquire(&self, messages: &[ChatMessage], deadline: Instant) -> Result<AcquiredMedia, MediaRejection> {
        let parts = image_parts(messages);
        let limits = &self.limits;
        let budget = |budget: Budget, limit: u64, requested: u64| MediaRejection {
            status: 400,
            code: "media_budget_exceeded",
            message: ProcessorError::BudgetExceeded { budget, limit, requested }.to_string(),
        };
        let max_items = (limits.max_raw_patches / 4).min(limits.max_vision_tokens);
        if parts.len() as u64 > max_items {
            return Err(budget(Budget::MediaItems, max_items, parts.len() as u64));
        }
        let mut stats = MediaStats { items: parts.len() as u32, ..MediaStats::default() };
        let mut remaining = limits.max_encoded_media_bytes;
        let mut tickets = Vec::with_capacity(parts.len());
        for (item, part) in parts.iter().enumerate() {
            let acquired = match media_source(&part.url) {
                Err(error) => Err(error),
                Ok(_) if remaining == 0 => {
                    Err(AcquireError::Budget("request media exceed the aggregate media byte budget".to_owned()))
                }
                Ok(MediaSource::Data(value)) => decode_data_uri(value, remaining),
                Ok(MediaSource::Http(url)) => tokio::time::timeout_at(deadline.into(), fetch(url, &self.policy, remaining, deadline))
                    .await
                    .unwrap_or(Err(AcquireError::Deadline)),
            };
            let bytes = acquired.map_err(|error| error.rejection(&part.at))?;
            remaining -= bytes.len() as u64;
            stats.media_bytes += bytes.len() as u64;
            let (digest, bytes) = tokio::task::spawn_blocking(move || (<[u8; 32]>::from(Sha256::digest(&bytes)), bytes))
                .await
                .expect("hashing does not panic");
            tickets.push(self.cache.begin(digest, item, bytes, &self.preparer, &self.pool));
        }
        let (mut raw_patches, mut vision_tokens) = (0u64, 0u64);
        let mut media = Vec::with_capacity(tickets.len());
        for ((item, part), ticket) in parts.iter().enumerate().zip(&tickets) {
            let result = tokio::time::timeout_at(deadline.into(), ticket.wait())
                .await
                .map_err(|_| AcquireError::Deadline.rejection(&part.at))?;
            let (prepared, seconds) = match result {
                Ok(done) => done,
                Err(BuildError::Processor(error)) => {
                    let error = match error {
                        ProcessorError::InvalidMedia { reason, .. } => ProcessorError::InvalidMedia { item, reason },
                        other => other,
                    };
                    return Err(MediaRejection { status: 400, code: error.code(), message: format!("{}: {error}", part.at) });
                }
                Err(BuildError::Cancelled) => unreachable!("a flight is never cancelled while this request waits on it"),
            };
            match ticket.disposition {
                Disposition::Produced => {
                    stats.cache_misses += 1;
                    stats.preprocess_seconds += seconds;
                }
                Disposition::Hit | Disposition::Joined => stats.cache_hits += 1,
            }
            raw_patches += prepared.grid.raw_patches();
            vision_tokens += prepared.grid.vision_tokens();
            if raw_patches > limits.max_raw_patches {
                return Err(budget(Budget::RawPatches, limits.max_raw_patches, raw_patches));
            }
            if vision_tokens > limits.max_vision_tokens {
                return Err(budget(Budget::VisionTokens, limits.max_vision_tokens, vision_tokens));
            }
            media.push(prepared);
        }
        drop(tickets);
        stats.vision_tokens = vision_tokens;
        let media = media.into_iter().map(|m| Arc::try_unwrap(m).unwrap_or_else(|m| (*m).clone())).collect();
        Ok(AcquiredMedia { media, stats })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::template::ContentPart;

    fn b64(bytes: &[u8]) -> String {
        const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in bytes.chunks(3) {
            let n = chunk.iter().enumerate().fold(0u32, |n, (i, &b)| n | (b as u32) << (16 - 8 * i));
            for i in 0..4 {
                out.push(if i <= chunk.len() { ALPHABET[(n >> (18 - 6 * i) & 63) as usize] as char } else { '=' });
            }
        }
        out
    }

    #[test]
    fn a_data_uri_decodes_its_base64_payload() {
        for bytes in [&b"a"[..], b"ab", b"abc", b"\x89PNG\r\n\x1a\n\x00\xff"] {
            let uri = format!("data:image/png;base64,{}", b64(bytes));
            assert_eq!(decode_data_uri(&uri, 1 << 20).unwrap(), bytes, "{uri}");
        }
        // Whitespace inside the payload is skipped, as the reference does.
        assert_eq!(decode_data_uri("data:image/png;base64,YW\nJj", 16).unwrap(), b"abc");
    }

    #[test]
    fn a_malformed_data_uri_is_invalid_media() {
        for uri in [
            "data:image/png,YWJj",          // not base64
            "data:image/png;base64",        // no comma
            "data:image/png;base64,YW=Jj",  // data after padding
            "data:image/png;base64,YW*j",   // outside the alphabet
            "data:image/png;base64,Y",      // a dangling six-bit group
            "data:image/png;base64,",       // empty
        ] {
            let error = decode_data_uri(uri, 1 << 20).unwrap_err();
            assert!(matches!(error, AcquireError::Invalid(_)), "{uri}: {error:?}");
            assert_eq!(error.rejection("message 0 content part 1").code, "invalid_media");
        }
    }

    #[test]
    fn the_byte_budget_admits_exactly_its_limit_and_refuses_one_over() {
        let bytes = vec![7u8; 300];
        let uri = format!("data:image/png;base64,{}", b64(&bytes));
        assert_eq!(decode_data_uri(&uri, 300).unwrap(), bytes);
        let error = decode_data_uri(&uri, 299).unwrap_err();
        assert!(matches!(error, AcquireError::Budget(_)), "{error:?}");
        let rejection = error.rejection("message 1 content part 0");
        assert_eq!((rejection.status, rejection.code), (400, "media_budget_exceeded"));
        assert!(rejection.message.starts_with("message 1 content part 0: "), "{}", rejection.message);
        // An encoded length that cannot fit is refused before decoding —
        // even when the payload is not valid base64.
        let huge = format!("data:image/png;base64,{}", "*".repeat(1000));
        assert!(matches!(decode_data_uri(&huge, 10), Err(AcquireError::Budget(_))));
    }

    #[test]
    fn only_data_and_http_sources_are_accepted() {
        assert_eq!(media_source("data:image/png;base64,YQ==").unwrap(), MediaSource::Data("data:image/png;base64,YQ=="));
        assert_eq!(media_source("HTTPS://x/y.png").unwrap(), MediaSource::Http("HTTPS://x/y.png"));
        for url in ["file:///etc/passwd", "ftp://x/y", "/tmp/a.png", "x.png"] {
            assert!(matches!(media_source(url), Err(AcquireError::Invalid(_))), "{url}");
        }
    }

    #[test]
    fn image_parts_come_back_in_prompt_order_naming_their_position() {
        let part = |kind: &str, url: Option<&str>, text: Option<&str>| ContentPart {
            kind: Some(kind.to_owned()),
            url: url.map(str::to_owned),
            text: text.map(str::to_owned),
        };
        let message = |role: &str, content| ChatMessage {
            role: role.to_owned(),
            content,
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        };
        let messages = [
            message("system", MessageContent::Text("s".to_owned())),
            message("user", MessageContent::Parts(vec![part("text", None, Some("a")), part("image_url", Some("data:1"), None)])),
            message("tool", MessageContent::Parts(vec![part("image_url", Some("data:2"), None)])),
        ];
        assert_eq!(
            image_parts(&messages),
            [
                MediaPart { at: "message 1 content part 1".to_owned(), url: "data:1".to_owned() },
                MediaPart { at: "message 2 content part 0".to_owned(), url: "data:2".to_owned() },
            ]
        );
    }
}
