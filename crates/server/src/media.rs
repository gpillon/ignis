//! Media acquisition (GitHub #179, spec of GitHub #174,
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

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ignis_artifact::vision::{
    Budget, DecodeFailure, Grid, InvalidMedia, PreparedMedia, ProcessorError, ProcessorOptions, VisionProcessor,
};
use ignis_core::image::{ImageSize, RgbaImage};
use sha2::{Digest, Sha256};
use tokio::sync::{watch, Semaphore};

use crate::template::{ChatMessage, ContentPart, MessageContent};

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
    /// The part refused, `message {i} content part {j}`; `None` for a limit
    /// on the whole request. The refusal's log event names it in place of
    /// the message, which for a fetch names the URL (GitHub #673).
    pub part: Option<String>,
    /// For `media_budget_exceeded`, the budget passed, in numbers: what the
    /// refusal's log event carries (GitHub #673).
    pub budget: Option<BudgetExceeded>,
}

/// A media budget a request passed: which, its limit and the amount the
/// request reached when it was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetExceeded {
    pub budget: Budget,
    pub limit: u64,
    pub requested: u64,
}

impl MediaRejection {
    /// A processor's refusal of the part at `at`.
    fn processor(at: &str, error: ProcessorError) -> Self {
        let budget = match error {
            ProcessorError::BudgetExceeded { budget, limit, requested } => Some(BudgetExceeded { budget, limit, requested }),
            _ => None,
        };
        Self { status: 400, code: error.code(), message: format!("{at}: {error}"), part: Some(at.to_owned()), budget }
    }

    /// A request-wide budget the request's media passed.
    fn over(budget: Budget, limit: u64, requested: u64) -> Self {
        let error = ProcessorError::BudgetExceeded { budget, limit, requested };
        let budget = Some(BudgetExceeded { budget, limit, requested });
        Self { status: 400, code: error.code(), message: error.to_string(), part: None, budget }
    }
}

/// What a dropped image is rendered as, where it was in the prompt (GitHub
/// #673, ADR 0102): the model reads that an image was there.
pub const OMITTED_IMAGE: &str = "[image omitted: past this request's vision budget]";

/// What acquiring a request's media cost, for its `ignis.request.*` events.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MediaStats {
    /// Media items in the request the model sees: the dropped ones are not
    /// among them.
    pub items: u32,
    /// Merged vision tokens across the items, as admitted.
    pub vision_tokens: u64,
    /// What those tokens would be under the artifact's own pixel bounds,
    /// without `--vision-max-tokens`: above `vision_tokens` exactly when the
    /// budget shrank an image (GitHub #248).
    pub natural_vision_tokens: u64,
    /// Acquired (encoded) bytes across the items.
    pub media_bytes: u64,
    /// The request's oldest images, dropped past its budgets and rendered as
    /// [`OMITTED_IMAGE`] (GitHub #673): never prepared, and not in `items`.
    pub dropped_items: u32,
    /// The merged vision tokens the dropped images would have taken.
    pub dropped_vision_tokens: u64,
    /// The dropped images' encoded bytes.
    pub dropped_bytes: u64,
    /// Seconds spent decoding, resizing and packing the items this request
    /// prepared itself (a cache hit costs none).
    pub preprocess_seconds: f64,
    /// Items served from the media cache (including single-flight joins).
    pub cache_hits: u32,
    /// Items this request prepared itself.
    pub cache_misses: u32,
    /// The **first** item's submitted pixel size, `(width, height)`, before
    /// any resize (GitHub #242); `None` when the request carried no media.
    /// The first the model sees: a dropped image is no answer's frame.
    ///
    /// The first because that is the one a spatial question is about: a
    /// `point` is put to one screenshot, and a `state` carrying several
    /// images has no single frame for its answer to be in. It rides here
    /// rather than on the request input because it is a property of what was
    /// *submitted*, which the engine neither knows nor needs — the engine
    /// sees patches.
    pub source_pixels: Option<(u32, u32)>,
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
    /// The item would exceed the byte cap it was acquired under: why, and the
    /// bytes it had reached when it was refused.
    Budget(String, u64),
    /// The remote could not be reached or answered an error.
    FetchFailed(String),
    /// The remote did not answer within the fetch timeouts.
    FetchTimeout(String),
    /// The request deadline passed.
    Deadline,
}

impl AcquireError {
    /// The refusal of the part at `at`. A byte cap is reported against the
    /// request's byte budget `limit`, the item's bytes on top of `before`
    /// (what the request had already spent of it under the same cap).
    fn rejection(self, at: &str, limit: u64, before: u64) -> MediaRejection {
        let mut budget = None;
        let (status, code, why) = match self {
            Self::Invalid(why) => (400, "invalid_media", why),
            Self::Budget(why, reached) => {
                budget = Some(BudgetExceeded { budget: Budget::EncodedBytes, limit, requested: before + reached });
                (400, "media_budget_exceeded", why)
            }
            Self::FetchFailed(why) => (400, "media_fetch_failed", why),
            Self::FetchTimeout(why) => (400, "media_fetch_timeout", why),
            Self::Deadline => (504, "request_timeout", "the request deadline passed while its media were acquired".to_owned()),
        };
        MediaRejection { status, code, message: format!("{at}: {why}"), part: Some(at.to_owned()), budget }
    }
}

/// One image part of a request, in prompt order.
#[derive(Debug, Clone, PartialEq, Eq)]
struct MediaPart {
    /// `message {i} content part {j}`, as every content-part error names it.
    at: String,
    url: String,
}

/// Where the request's `image_url` parts sit, in prompt order (message
/// order, then part order — the order the chat template renders their
/// placeholders in): each one's message index, part index and URL.
/// `check_content_parts` has already refused every malformed part.
fn image_positions(messages: &[ChatMessage]) -> impl Iterator<Item = (usize, usize, &str)> {
    messages.iter().enumerate().flat_map(|(i, message)| {
        let parts = match &message.content {
            MessageContent::Parts(parts) => parts.as_slice(),
            MessageContent::Text(_) | MessageContent::Null => &[],
        };
        parts.iter().enumerate().filter_map(move |(j, part)| match (part.kind.as_deref().unwrap_or_default(), &part.url) {
            ("image_url", Some(url)) => Some((i, j, url.as_str())),
            _ => None,
        })
    })
}

/// The request's `image_url` parts in prompt order ([`image_positions`]).
fn image_parts(messages: &[ChatMessage]) -> Vec<MediaPart> {
    image_positions(messages)
        .map(|(i, j, url)| MediaPart { at: format!("message {i} content part {j}"), url: url.to_owned() })
        .collect()
}

/// The processor a `--vision-enabled` load prepares images with: the artifact's
/// pixel bounds and the reference's limits, with the vision-token budget
/// capped by the load's envelope and context (`min(max_context,
/// --vision-max-tokens)`, spec §Scope) and the raw patches it implies.
pub fn load_processor(
    frontend: &ignis_artifact::FrontendSet,
    vision: ignis_core::Vision,
    max_context: u32,
) -> Result<VisionProcessor, ProcessorError> {
    let options = bounded_options(frontend.vision_processor()?.options().clone(), vision, max_context);
    VisionProcessor::new(frontend.tokenizer(), options)
}

/// `processor` at a load's context `max_context` rather than the one it was
/// built at: the context a 27B load runs, which its adaptation may have
/// lowered (ADR 0063 §3). Its request budget, and so its item bound, follow
/// the envelope the leaf planned; and an item bound the load holds below the
/// artifact's (`vision`'s, GitHub #557: Flash-Next's encode in its prefill
/// scratch, ADR 0087) lowers the pixel bound to its area, so a larger image
/// is shrunk to fit, aspect kept, as the request budget shrinks one.
pub fn processor_at(processor: VisionProcessor, vision: ignis_core::Vision, max_context: u32) -> VisionProcessor {
    let mut options = bounded_options(processor.options().clone(), vision, max_context);
    if vision.item_max_tokens() > 0 {
        let factor = ignis_artifact::vision::FACTOR as u64;
        options.max_pixels = options.max_pixels.min(u64::from(vision.item_max_tokens()) * factor * factor);
    }
    processor.with_options(options)
}

/// `options` with the vision-token budget of a load of `max_context` tokens
/// and the raw patches it implies; the pixel bounds stay.
fn bounded_options(mut options: ProcessorOptions, vision: ignis_core::Vision, max_context: u32) -> ProcessorOptions {
    options.max_vision_tokens = u64::from(vision.envelope_tokens(max_context));
    options.max_raw_patches = options.max_vision_tokens * 4;
    options
}

/// Whether any message carries an `image_url` part.
pub fn has_media(messages: &[ChatMessage]) -> bool {
    messages.iter().any(|message| match &message.content {
        MessageContent::Parts(parts) => {
            parts.iter().any(|part| part.kind.as_deref() == Some("image_url") && part.url.is_some())
        }
        MessageContent::Text(_) | MessageContent::Null => false,
    })
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
    let over = |reached: u64| {
        AcquireError::Budget(format!("media data exceeds the request media byte budget ({max_bytes} bytes)"), reached)
    };
    if payload.len() as u64 > (max_bytes / 3 + 1) * 4 {
        // What the payload would decode to, had it been decoded.
        return Err(over(payload.len() as u64 / 4 * 3));
    }
    let bytes = decode_base64(payload)?;
    if bytes.len() as u64 > max_bytes {
        return Err(over(bytes.len() as u64));
    }
    if bytes.is_empty() {
        return Err(AcquireError::Invalid("media source is empty".to_owned()));
    }
    Ok(bytes)
}

/// [`decode_base64`]'s bytes, `None` where it refuses `text`: for a caller
/// with no media error to report (a Responses reasoning blob).
pub(crate) fn base64_bytes(text: &str) -> Option<Vec<u8>> {
    decode_base64(text).ok()
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
/// 224/4 up; IPv6 unspecified, loopback, link-local, multicast, unique-local.
///
/// Two departures, both closing holes: the reference's benchmarking check
/// (`>> 17 == 0x633f`) matches the public 198.126/15 instead of 198.18/15;
/// and an IPv6 address that carries an IPv4 one — v4-mapped, v4-compatible,
/// NAT64 `64:ff9b::/96`, 6to4 `2002::/16` — is judged by its IPv4 rule, with
/// deprecated site-local `fec0::/10` refused too.
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
            || a >> 17 == 0x6309
            || a >> 24 >= 224
    };
    match address {
        IpAddr::V4(a) => v4(a),
        IpAddr::V6(a) => {
            let s = a.segments();
            let embedded = |hi: u16, lo: u16| Ipv4Addr::from((u32::from(hi) << 16) | u32::from(lo));
            if let Some(mapped) = a.to_ipv4_mapped() {
                return v4(mapped);
            }
            if a.is_unspecified() || a.is_loopback() {
                return true;
            }
            if s[..6] == [0; 6] || (s[0] == 0x64 && s[1] == 0xff9b && s[2..6] == [0; 4]) {
                return v4(embedded(s[6], s[7]));
            }
            if s[0] == 0x2002 {
                return v4(embedded(s[1], s[2]));
            }
            s[0] & 0xffc0 == 0xfe80 || s[0] & 0xffc0 == 0xfec0 || s[0] & 0xff00 == 0xff00 || s[0] & 0xfe00 == 0xfc00
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

/// An HTTP(S) image's bytes, at most `max_bytes` of them. The fetch timeout
/// is one total across resolution and every redirect, and never outlasts the
/// request `deadline`; running out of either is a fetch timeout.
async fn fetch(url: &str, policy: &MediaPolicy, max_bytes: u64, deadline: Instant) -> Result<Vec<u8>, AcquireError> {
    let deadline = deadline.min(Instant::now() + policy.fetch_timeout);
    let timed_out = || AcquireError::FetchTimeout("media URL fetch timed out".to_owned());
    let invalid = || AcquireError::Invalid("media URL must be credential-free HTTP(S)".to_owned());
    let mut url = reqwest::Url::parse(url).map_err(|_| AcquireError::Invalid("invalid media URL".to_owned()))?;
    for hop in 0..=policy.max_redirects {
        if !matches!(url.scheme(), "http" | "https") || !url.username().is_empty() || url.password().is_some() {
            return Err(invalid());
        }
        let host = url.host_str().filter(|h| !h.is_empty()).ok_or_else(invalid)?.to_owned();
        let port = url.port_or_known_default().ok_or_else(invalid)?;
        let address = tokio::time::timeout_at(deadline.into(), resolve_allowed(&host, port, policy))
            .await
            .map_err(|_| timed_out())??;
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(timed_out());
        }
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(policy.connect_timeout.min(remaining))
            .timeout(remaining)
            .resolve(host.trim_start_matches('[').trim_end_matches(']'), address)
            .user_agent("ignis/vision")
            .build()
            .map_err(fetch_error)?;
        let mut response = client.get(url.clone()).send().await.map_err(fetch_error)?;
        let status = response.status();
        if status.is_success() {
            let over = |reached: u64| {
                AcquireError::Budget(format!("media URL exceeds the request media byte budget ({max_bytes} bytes)"), reached)
            };
            if let Some(length) = response.content_length().filter(|&n| n > max_bytes) {
                return Err(over(length));
            }
            let mut body = Vec::new();
            while let Some(chunk) = response.chunk().await.map_err(fetch_error)? {
                if (body.len() + chunk.len()) as u64 > max_bytes {
                    return Err(over((body.len() + chunk.len()) as u64));
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
    /// The kept images, prepared.
    pub media: Vec<PreparedMedia>,
    pub stats: MediaStats,
    /// How many of the request's image parts, the oldest in prompt order,
    /// were dropped past its budgets (GitHub #673): they precede every kept
    /// one, and [`Self::omit_dropped`] renders them as text.
    pub dropped: usize,
}

impl AcquiredMedia {
    /// `messages` as the model reads them (GitHub #673, ADR 0102): each
    /// dropped image part replaced, where it was, by a text part reading
    /// [`OMITTED_IMAGE`], so the chat template renders that text there and
    /// the vision placeholders of the kept images alone, in the order
    /// [`Self::media`] holds them. Borrowed when nothing was dropped.
    pub fn omit_dropped<'m>(&self, messages: &'m [ChatMessage]) -> Cow<'m, [ChatMessage]> {
        if self.dropped == 0 {
            return Cow::Borrowed(messages);
        }
        let dropped: Vec<(usize, usize)> = image_positions(messages).take(self.dropped).map(|(i, j, _)| (i, j)).collect();
        let mut messages = messages.to_vec();
        for (i, j) in dropped {
            let MessageContent::Parts(parts) = &mut messages[i].content else {
                unreachable!("an image part is in a parts message");
            };
            let part = &mut parts[j];
            *part = ContentPart {
                kind: Some("text".to_owned()),
                url: None,
                text: Some(OMITTED_IMAGE.to_owned()),
                cache_control: part.cache_control.take(),
            };
        }
        Cow::Owned(messages)
    }
}

/// What an image weighs against a request's budgets, read before anything is
/// prepared (GitHub #673) — or, as a limit, what the budgets allow.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Weight {
    raw_patches: u64,
    vision_tokens: u64,
    bytes: u64,
}

impl Weight {
    fn of(grid: Grid, bytes: usize) -> Self {
        Self { raw_patches: grid.raw_patches(), vision_tokens: grid.vision_tokens(), bytes: bytes as u64 }
    }

    fn plus(self, other: Self) -> Self {
        Self {
            raw_patches: self.raw_patches + other.raw_patches,
            vision_tokens: self.vision_tokens + other.vision_tokens,
            bytes: self.bytes + other.bytes,
        }
    }

    fn within(self, limit: Self) -> bool {
        self.raw_patches <= limit.raw_patches && self.vision_tokens <= limit.vision_tokens && self.bytes <= limit.bytes
    }

    fn half(self) -> Self {
        Self { raw_patches: self.raw_patches / 2, vision_tokens: self.vision_tokens / 2, bytes: self.bytes / 2 }
    }
}

/// How many of a request's oldest images are dropped so that the rest are
/// within `limit` (GitHub #673, ADR 0102). The images, in prompt order, form
/// blocks: a block takes images until the next would carry it past half of
/// `limit` on any count, so an image past half alone is a block of its own.
/// Over `limit`, the fewest whole blocks are dropped, oldest first, that bring
/// the rest within it, never the last block; within it, none.
///
/// A block's bounds depend only on the images before it, so on a history
/// that grows at its end a closed block never changes and is dropped once:
/// the prompt changes at its first image once a block, and between drops the
/// next request's prompt extends this one's.
fn dropped_images(weights: &[Weight], limit: Weight) -> usize {
    let total = |weights: &[Weight]| weights.iter().fold(Weight::default(), |sum, &weight| sum.plus(weight));
    if total(weights).within(limit) {
        return 0;
    }
    let half = limit.half();
    let (mut starts, mut block) = (Vec::new(), Weight::default());
    for (index, &weight) in weights.iter().enumerate() {
        if index == 0 || !block.plus(weight).within(half) {
            starts.push(index);
            block = weight;
        } else {
            block = block.plus(weight);
        }
    }
    let last = *starts.last().expect("a request over its budget has an image");
    // Never past the last block: should it not fit alone, the preparation's
    // own check refuses the request, as before the rule.
    starts.into_iter().find(|&start| total(&weights[start..]).within(limit)).unwrap_or(last)
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
    /// `deadline`, all before any device work; dropping the returned future
    /// stops the work.
    ///
    /// Images whose total passes the request's budgets lose their oldest, in
    /// whole blocks ([`dropped_images`], GitHub #673): every part's bytes are
    /// acquired and weighed from its header first, and only the kept ones are
    /// prepared. What would refuse an image alone refuses the request
    /// wherever the image is.
    pub async fn acquire(&self, messages: &[ChatMessage], deadline: Instant) -> Result<AcquiredMedia, MediaRejection> {
        let parts = image_parts(messages);
        let limits = &self.limits;
        let max_items = (limits.max_raw_patches / 4).min(limits.max_vision_tokens);
        if parts.len() as u64 > max_items {
            return Err(MediaRejection::over(Budget::MediaItems, max_items, parts.len() as u64));
        }
        // Every part's bytes. An inline image is held to the byte budget
        // alone; the request's fetches together, as before the rule, since a
        // fetch has cost its bytes once made.
        let budget = limits.max_encoded_media_bytes;
        let mut fetched = 0u64;
        let mut sources = Vec::with_capacity(parts.len());
        for part in &parts {
            let source = media_source(&part.url);
            let fetching = matches!(source, Ok(MediaSource::Http(_)));
            let fetch_cap = budget - fetched;
            let acquired = match source {
                Err(error) => Err(error),
                Ok(MediaSource::Data(value)) => decode_data_uri(value, budget),
                // Spent: refused unfetched, as before the rule, at least a
                // byte more than the fetches before it.
                Ok(MediaSource::Http(_)) if fetch_cap == 0 => {
                    Err(AcquireError::Budget("request media exceed the aggregate media byte budget".to_owned(), 1))
                }
                Ok(MediaSource::Http(url)) => tokio::time::timeout_at(deadline.into(), fetch(url, &self.policy, fetch_cap, deadline))
                    .await
                    .unwrap_or_else(|_| Err(AcquireError::FetchTimeout("media URL fetch timed out".to_owned()))),
            };
            let bytes = acquired.map_err(|error| error.rejection(&part.at, budget, if fetching { fetched } else { 0 }))?;
            if fetching {
                fetched += bytes.len() as u64;
            }
            sources.push(bytes);
        }
        // What each weighs, from its header, and which are dropped.
        let weights = parts
            .iter()
            .zip(&sources)
            .enumerate()
            .map(|(item, (part, bytes))| {
                limits
                    .header_grid(item, bytes)
                    .map(|grid| Weight::of(grid, bytes.len()))
                    .map_err(|error| MediaRejection::processor(&part.at, error))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let limit = Weight { raw_patches: limits.max_raw_patches, vision_tokens: limits.max_vision_tokens, bytes: budget };
        let dropped = dropped_images(&weights, limit);
        let mut stats = MediaStats {
            items: (parts.len() - dropped) as u32,
            dropped_items: dropped as u32,
            dropped_vision_tokens: weights[..dropped].iter().map(|w| w.vision_tokens).sum(),
            dropped_bytes: weights[..dropped].iter().map(|w| w.bytes).sum(),
            ..MediaStats::default()
        };
        // Only the kept images are hashed, cached and prepared.
        let mut tickets = Vec::with_capacity(parts.len() - dropped);
        for (item, bytes) in sources.into_iter().enumerate().skip(dropped) {
            stats.media_bytes += bytes.len() as u64;
            // The digest is CPU work on up to the whole byte budget: on the
            // pool, like the preparation it keys.
            let permit = Arc::clone(&self.pool).acquire_owned().await.expect("the pool is never closed");
            let (digest, bytes) = tokio::task::spawn_blocking(move || {
                let _permit = permit;
                (<[u8; 32]>::from(Sha256::digest(&bytes)), bytes)
            })
            .await
            .expect("hashing does not panic");
            tickets.push((item, self.cache.begin(digest, item, bytes, &self.preparer, &self.pool)));
        }
        let (mut raw_patches, mut vision_tokens, mut natural_vision_tokens) = (0u64, 0u64, 0u64);
        let mut media = Vec::with_capacity(tickets.len());
        for (item, ticket) in &tickets {
            let (item, part) = (*item, &parts[*item]);
            let result = tokio::time::timeout_at(deadline.into(), ticket.wait())
                .await
                .map_err(|_| AcquireError::Deadline.rejection(&part.at, budget, 0))?;
            let (prepared, seconds) = match result {
                Ok(done) => done,
                Err(BuildError::Processor(error)) => {
                    let error = match error {
                        ProcessorError::InvalidMedia { reason, .. } => ProcessorError::InvalidMedia { item, reason },
                        other => other,
                    };
                    return Err(MediaRejection::processor(&part.at, error));
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
            if media.is_empty() {
                stats.source_pixels = Some(prepared.source_pixels);
            }
            raw_patches += prepared.grid.raw_patches();
            vision_tokens += prepared.grid.vision_tokens();
            natural_vision_tokens += prepared.natural_vision_tokens;
            // A backstop: the weights said the kept images fit.
            if raw_patches > limits.max_raw_patches {
                return Err(MediaRejection::over(Budget::RawPatches, limits.max_raw_patches, raw_patches));
            }
            if vision_tokens > limits.max_vision_tokens {
                return Err(MediaRejection::over(Budget::VisionTokens, limits.max_vision_tokens, vision_tokens));
            }
            media.push(prepared);
        }
        drop(tickets);
        stats.vision_tokens = vision_tokens;
        stats.natural_vision_tokens = natural_vision_tokens;
        let media = media.into_iter().map(|m| Arc::try_unwrap(m).unwrap_or_else(|m| (*m).clone())).collect();
        Ok(AcquiredMedia { media, stats, dropped })
    }
}

// ── the image model's condition images ──────────────────────────────────

/// An edit's condition image (spec image-model/05): PNG, JPEG or WebP bytes
/// decoded to RGBA, PNG and WebP with their alpha, JPEG opaque, refused past
/// `max_pixels` before its frame is allocated (`image.max_condition_mipx`).
/// The message says why it is not one.
pub fn decode_condition_image(bytes: &[u8], max_pixels: u64) -> Result<RgbaImage, String> {
    match ignis_artifact::vision::decode_image_rgba(bytes, max_pixels) {
        Ok(image) => Ok(RgbaImage { size: ImageSize::new(image.width, image.height), pixels: image.rgba }),
        Err(DecodeFailure::Invalid(InvalidMedia::Undecodable(why))) => Err(format!("cannot decode image: {why}")),
        Err(DecodeFailure::Invalid(InvalidMedia::Unsupported(why))) => Err(format!("unsupported image: {why}")),
        // The decoders never raise it: the vision processor's resize does.
        Err(DecodeFailure::Invalid(InvalidMedia::AspectRatio { width, height })) => {
            Err(format!("unsupported image: {width}x{height}"))
        }
        Err(DecodeFailure::Pixels(pixels)) => Err(format!(
            "the image has {pixels} pixels, over the {max_pixels} a condition image may decode to (image.max_condition_mipx)"
        )),
    }
}

// `resize_rgba`, `clip8`, `lanczos_weights` and `PRECISION_BITS` are a port
// of python-pillow/Pillow 12.3.0 (tag commit bb1d8e8), MIT-CMU licensed:
// `src/libImaging/Resample.c` (`ImagingResample`, `precompute_coeffs`,
// `normalize_coeffs_8bpc`, the 8-bit horizontal and vertical passes, `clip8`,
// `PRECISION_BITS`), `src/libImaging/Convert.c` (`rgbA2rgba`, the
// premultiply, and `rgba2rgbA`, its inverse) and
// `src/libImaging/ImagingUtils.h` (`MULDIV255`). Pillow's copyright and
// permission notice are in the repository's root NOTICE.

/// `image` resampled to `size` the way diffusers' `VaeImageProcessor.resize`
/// does it, through Pillow (spec image-model/03 §Image preprocessing):
/// Pillow's `ImagingResample` with its Lanczos-3 filter, step for step —
/// RGBA premultiplied to `RGBa` (Pillow resizes RGBA that way, so a
/// transparent pixel's colour never bleeds into its neighbours), the
/// horizontal pass then the vertical, each only when its side changes, each
/// in Pillow's 8-bit fixed point (the filter widened by the downscale factor,
/// 22-bit weights summing to 1, each pass rounded and clamped to a byte), and
/// `RGBa` back to RGBA.
pub fn resize_rgba(image: &RgbaImage, size: ImageSize) -> RgbaImage {
    // Pillow copies an image asked for at its own size.
    if image.size == size {
        return image.clone();
    }
    // Pillow's `rgbA2rgba`, through its MULDIV255: c · a / 255, rounded.
    let premultiply = |c: u8, a: u8| {
        let t = u32::from(c) * u32::from(a) + 128;
        ((t + (t >> 8)) >> 8) as u8
    };
    let mut pixels: Vec<u8> = image
        .pixels
        .chunks_exact(4)
        .flat_map(|px| [premultiply(px[0], px[3]), premultiply(px[1], px[3]), premultiply(px[2], px[3]), px[3]])
        .collect();
    let (mut width, height) = (image.size.width as usize, image.size.height as usize);
    if size.width as usize != width {
        let columns = lanczos_weights(width, size.width as usize);
        let mut out = Vec::with_capacity(columns.len() * height * 4);
        for row in pixels.chunks_exact(width * 4) {
            for (first, weights) in &columns {
                for c in 0..4 {
                    out.push(clip8(weights.iter().enumerate().map(|(k, w)| i64::from(row[(first + k) * 4 + c]) * w).sum()));
                }
            }
        }
        (pixels, width) = (out, size.width as usize);
    }
    if size.height as usize != height {
        let rows = lanczos_weights(height, size.height as usize);
        let mut out = Vec::with_capacity(rows.len() * width * 4);
        for (first, weights) in &rows {
            for x in 0..width * 4 {
                out.push(clip8(weights.iter().enumerate().map(|(k, w)| i64::from(pixels[(first + k) * width * 4 + x]) * w).sum()));
            }
        }
        pixels = out;
    }
    // Pillow's `rgba2rgbA`: c · 255 / a, truncated; alpha 0 and 255 kept.
    for px in pixels.chunks_exact_mut(4) {
        let alpha = u32::from(px[3]);
        if alpha != 0 && alpha != 255 {
            for c in &mut px[..3] {
                *c = (u32::from(*c) * 255 / alpha).min(255) as u8;
            }
        }
    }
    RgbaImage { size, pixels }
}

/// Pillow's fixed-point precision for 8-bit samples (`PRECISION_BITS`).
const PRECISION_BITS: u32 = 32 - 8 - 2;

/// A sum of samples times 22-bit weights back to a byte: Pillow's `clip8`,
/// from its rounding offset.
fn clip8(sum: i64) -> u8 {
    ((sum + (1 << (PRECISION_BITS - 1))) >> PRECISION_BITS).clamp(0, 255) as u8
}

/// Pillow's `precompute_coeffs` and `normalize_coeffs_8bpc` for Lanczos-3
/// from `input` samples to `out`: per output sample, its first input sample
/// and the 22-bit weights of the run starting there.
fn lanczos_weights(input: usize, out: usize) -> Vec<(usize, Vec<i64>)> {
    let sinc = |x: f64| if x == 0.0 { 1.0 } else { (x * std::f64::consts::PI).sin() / (x * std::f64::consts::PI) };
    let lanczos = |x: f64| if (-3.0..3.0).contains(&x) { sinc(x) * sinc(x / 3.0) } else { 0.0 };
    let scale = input as f64 / out as f64;
    let filter_scale = scale.max(1.0);
    let (support, ss) = (3.0 * filter_scale, 1.0 / filter_scale);
    (0..out)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let first = (center - support + 0.5).max(0.0) as usize;
            let last = ((center + support + 0.5) as usize).min(input);
            let weights: Vec<f64> = (first..last).map(|x| lanczos((x as f64 - center + 0.5) * ss)).collect();
            let total: f64 = weights.iter().sum();
            let fixed = |w: f64| {
                let w = if total == 0.0 { w } else { w / total } * f64::from(1u32 << PRECISION_BITS);
                (if w < 0.0 { w - 0.5 } else { w + 0.5 }) as i64
            };
            (first, weights.iter().map(|&w| fixed(w)).collect())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::template::ContentPart;

    /// A processor over a tokenizer that knows the model contract's two
    /// placeholders, under `options`.
    fn processor(options: ProcessorOptions) -> VisionProcessor {
        use ignis_artifact::vision::{IMAGE_PAD, IMAGE_PAD_ID, VIDEO_PAD, VIDEO_PAD_ID};
        let added = |id: u32, content: &str| {
            serde_json::json!({"id": id, "content": content, "single_word": false, "lstrip": false,
                               "rstrip": false, "normalized": false, "special": true})
        };
        let json = serde_json::json!({
            "version": "1.0",
            "added_tokens": [added(IMAGE_PAD_ID, IMAGE_PAD), added(VIDEO_PAD_ID, VIDEO_PAD)],
            "pre_tokenizer": {"type": "Whitespace"},
            "model": {"type": "WordLevel", "vocab": {"x": 0, IMAGE_PAD: IMAGE_PAD_ID, VIDEO_PAD: VIDEO_PAD_ID}, "unk_token": "x"},
        });
        let tokenizer = ignis_artifact::Tokenizer::from_bytes(json.to_string().as_bytes()).expect("tokenizer");
        VisionProcessor::new(&tokenizer, options).expect("the contract's placeholders")
    }

    /// ADR 0063 §3 (review of #336 part D): a processor built at the
    /// configured context and served at the one a 27B load adapted to takes
    /// that context's envelope -- its request budget, its item bound, the
    /// raw patches they imply -- and keeps the artifact's pixel bounds.
    #[test]
    fn a_processor_at_an_adapted_context_takes_its_envelope() {
        let artifact = ProcessorOptions {
            min_pixels: 32 * 32,
            max_pixels: 1 << 28,
            max_encoded_media_bytes: 1 << 20,
            max_decoded_pixels: 1 << 28,
            max_raw_patches: 1 << 16,
            max_vision_tokens: 1 << 14,
        };
        let vision = ignis_core::Vision::new(131_072).unwrap();
        let configured = processor_at(processor(artifact.clone()), vision, 131_072);
        assert_eq!((configured.options().max_vision_tokens, configured.options().max_item_tokens()), (131_072, 131_072));
        let adapted = processor_at(configured, vision, 65_536);
        let options = adapted.options();
        assert_eq!((options.max_vision_tokens, options.max_item_tokens(), options.max_raw_patches), (65_536, 65_536, 262_144));
        assert_eq!((options.min_pixels, options.max_pixels), (artifact.min_pixels, artifact.max_pixels));
        let wider = processor_at(adapted, vision, 196_608);
        assert_eq!(wider.options().max_vision_tokens, 131_072, "the vision envelope below the context");
    }

    /// GitHub #557 (ADR 0087): a load whose encoder holds a smaller item than
    /// the artifact's bound -- Flash-Next's, in its prefill scratch -- serves
    /// that bound: the processor's pixel bound lowered to its area, so an
    /// image at the artifact's bound is shrunk to fit, aspect kept, never
    /// refused. A bound at or past the artifact's changes nothing.
    #[test]
    fn a_processor_serves_the_item_bound_its_load_holds() {
        let artifact = ProcessorOptions {
            min_pixels: 256 * 256,
            max_pixels: 4096 * 4096,
            max_encoded_media_bytes: 1 << 20,
            max_decoded_pixels: 1 << 28,
            max_raw_patches: 1 << 17,
            max_vision_tokens: 1 << 15,
        };
        let vision = ignis_core::Vision::default();
        let unbounded = processor_at(processor(artifact.clone()), vision, 262_144);
        assert_eq!(unbounded.options().max_item_tokens(), 16_384, "the artifact's 4096 x 4096");
        let at_the_artifacts = processor_at(processor(artifact.clone()), vision.with_item_max_tokens(16_384), 262_144);
        assert_eq!(at_the_artifacts.options(), unbounded.options());

        let held = processor_at(processor(artifact.clone()), vision.with_item_max_tokens(5_000), 262_144);
        let options = held.options();
        assert_eq!(options.max_item_tokens(), 5_000);
        assert_eq!((options.min_pixels, options.max_vision_tokens), (artifact.min_pixels, artifact.max_vision_tokens));
        let size = options.resize_target(4096, 2048).expect("shrunk, not refused");
        let tokens = u64::from(size.height / 32) * u64::from(size.width / 32);
        assert!(tokens <= 5_000 && tokens > 4_000, "{size:?}: {tokens} tokens");
        let aspect = f64::from(size.height) / f64::from(size.width);
        assert!((aspect - 2.0).abs() < 0.05, "aspect kept: {size:?}");
    }

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
            assert_eq!(error.rejection("message 0 content part 1", 1 << 20, 0).code, "invalid_media");
        }
    }

    #[test]
    fn the_byte_budget_admits_exactly_its_limit_and_refuses_one_over() {
        let bytes = vec![7u8; 300];
        let uri = format!("data:image/png;base64,{}", b64(&bytes));
        assert_eq!(decode_data_uri(&uri, 300).unwrap(), bytes);
        let error = decode_data_uri(&uri, 299).unwrap_err();
        assert!(matches!(error, AcquireError::Budget(_, 300)), "{error:?}");
        let rejection = error.rejection("message 1 content part 0", 299, 0);
        assert_eq!((rejection.status, rejection.code), (400, "media_budget_exceeded"));
        assert!(rejection.message.starts_with("message 1 content part 0: "), "{}", rejection.message);
        let numbers = BudgetExceeded { budget: Budget::EncodedBytes, limit: 299, requested: 300 };
        assert_eq!(rejection.budget, Some(numbers), "the refusal carries its numbers (GitHub #673)");
        // An encoded length that cannot fit is refused before decoding —
        // even when the payload is not valid base64.
        let huge = format!("data:image/png;base64,{}", "*".repeat(1000));
        assert!(matches!(decode_data_uri(&huge, 10), Err(AcquireError::Budget(_, 750))));
    }

    #[test]
    fn the_address_classifier_refuses_private_ranges_and_their_ipv6_wrappings() {
        let private = |s: &str| private_address(s.parse().unwrap());
        for address in [
            "0.1.2.3", "10.0.0.1", "127.0.0.1", "169.254.1.1", "172.16.0.1", "172.31.255.255", "192.168.1.1",
            "100.64.0.1", "100.127.255.255", "198.18.0.1", "198.19.255.255", "224.0.0.1", "255.255.255.255",
            "::", "::1", "fe80::1", "fec0::1", "ff02::1", "fc00::1", "fd12::1",
            "::ffff:10.0.0.1", "::10.0.0.1", "64:ff9b::7f00:1", "2002:c0a8:0101::1",
        ] {
            assert!(private(address), "{address} must be refused");
        }
        for address in [
            "8.8.8.8", "172.32.0.1", "100.128.0.1", "198.17.255.255", "198.20.0.1", "198.126.0.1",
            "2001:4860::8888", "::ffff:8.8.8.8", "64:ff9b::808:808", "2002:0808:0808::1",
        ] {
            assert!(!private(address), "{address} is public");
        }
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
            cache_control: None,
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

    /// The condition images' resize is Pillow's byte for byte: the fixtures
    /// are Pillow 12.3's `Image.resize(.., LANCZOS)` of the vision fixtures —
    /// an RGBA image down on both sides, a crop of it up, and an opaque one —
    /// as diffusers resizes a condition image.
    #[test]
    fn the_condition_resize_is_pillows_lanczos_byte_for_byte() {
        let decode = |bytes: &[u8]| decode_condition_image(bytes, 64 << 20).unwrap();
        let alpha = decode(include_bytes!("../../artifact/tests/fixtures/vision/images/alpha.png"));
        let crop = RgbaImage {
            size: ImageSize::new(100, 60),
            pixels: (70..130).flat_map(|y| alpha.pixels[(y * 300 + 100) * 4..][..400].to_vec()).collect(),
        };
        let grid = decode(include_bytes!("../../artifact/tests/fixtures/vision/images/on_grid.png"));
        for (source, pillow) in [
            (&alpha, &include_bytes!("../tests/fixtures/pillow_resize/alpha_160x96.png")[..]),
            (&crop, &include_bytes!("../tests/fixtures/pillow_resize/alpha_crop_160x96.png")[..]),
            (&grid, &include_bytes!("../tests/fixtures/pillow_resize/on_grid_224x160.png")[..]),
        ] {
            let pillow = decode(pillow);
            assert!(resize_rgba(source, pillow.size) == pillow, "{} to {}", source.size, pillow.size);
        }
        assert!(resize_rgba(&alpha, alpha.size) == alpha, "the same size is the same image");
    }

    /// A transparent pixel's colour does not bleed into an opaque one; one
    /// resized side leaves the other untouched.
    #[test]
    fn the_condition_resize_never_bleeds_a_transparent_colour() {
        // Left half transparent red, right half opaque blue.
        let pixels = (0..64 * 32).flat_map(|i| if i % 64 < 32 { [255, 0, 0, 0] } else { [0, 0, 255, 255] }).collect();
        let image = RgbaImage { size: ImageSize::new(64, 32), pixels };
        let resized = resize_rgba(&image, ImageSize::new(40, 32));
        assert!(resized.pixels.chunks_exact(4).all(|px| px[0] == 0), "no red anywhere");
        assert!(resized.pixels.chunks_exact(4).any(|px| px[3] == 0) && resized.pixels.chunks_exact(4).any(|px| px[3] == 255));
        let rows: Vec<&[u8]> = resized.pixels.chunks_exact(40 * 4).collect();
        assert!(rows.iter().all(|row| *row == rows[0]), "the height did not change, nor did the rows");
    }

    #[test]
    fn a_condition_image_that_is_not_png_jpeg_or_webp_is_refused_saying_so() {
        assert!(decode_condition_image(b"GIF89a....", 64 << 20).unwrap_err().starts_with("unsupported image"));
        assert!(decode_condition_image(b"\x89PNG\r\n\x1a\nxx", 64 << 20).unwrap_err().starts_with("cannot decode image"));
    }
}

