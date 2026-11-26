//! The image routes (spec image-model/05 §Image routes, §The condition
//! budget, §The queue, GitHub #322) over the router, on a fake image runner
//! (ADR 0006): a generation's RGBA PNGs and its seed (AC 3), the field rules
//! and `auto` (AC 4), an edit's condition images and the budget (AC 5), the
//! streamed events (AC 6), and the queue — one job at a time, in order, full
//! past its depth, a disconnect cancelling its job (AC 7).
//!
//! The runner is [`Recording`]: `FakeImageRunner`'s numbers, every call
//! logged, each denoising step behind a [`Gate`] a test can hold shut, and
//! the [`Faults`] a test may arm.

use std::io::Cursor;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use http_body_util::BodyExt;
use ignis_core::image::{
    ConditionId, ConditionUnfit, FakeImageRunner, ImageCondition, ImageError, ImagePlan, ImageRunner, ImageSize, RgbaImage, TargetLatents,
};
use ignis_server::deadline::TimeoutReply;
use ignis_server::images::ImageModel;
use ignis_server::{ActiveModel, Server};
use serde_json::{json, Value};
use tower::ServiceExt;

const IMAGE: &str = "ignis-image-2.1-turbo";

/// How long a test waits for something that should happen at once.
const PATIENT: Duration = Duration::from_secs(30);

/// Shut, a denoising step waits at it; open, every step passes. A step
/// waits [`PATIENT`] at most, so a failed test never leaves a thread behind.
#[derive(Default)]
struct Gate {
    open: Mutex<bool>,
    changed: Condvar,
}

impl Gate {
    fn shut() -> Arc<Self> {
        Arc::new(Self::default())
    }

    fn opened() -> Arc<Self> {
        let gate = Self::shut();
        gate.open();
        gate
    }

    fn pass(&self) {
        let deadline = Instant::now() + PATIENT;
        let mut open = self.open.lock().unwrap();
        while !*open && Instant::now() < deadline {
            open = self.changed.wait_timeout(open, Duration::from_millis(50)).unwrap().0;
        }
    }

    fn open(&self) {
        *self.open.lock().unwrap() = true;
        self.changed.notify_all();
    }
}

/// A call the runner took.
#[derive(Debug, Clone, PartialEq)]
enum Call {
    /// The prompt, each condition image's size, and whether it carried any
    /// alpha below 255.
    Encode { prompt: String, sizes: Vec<ImageSize>, alpha: Vec<bool> },
    Step,
    Decode,
    Release,
}

/// What a test may make the runner do wrong.
#[derive(Default)]
struct Faults {
    /// The denoising step, counted from 1 across the runner's life, that
    /// panics, once; 0 is none.
    panic_at_step: AtomicUsize,
    /// How many tokens `condition_tokens` counts short of the prefix
    /// `encode_condition` refuses by: a runner whose count drifted.
    undercount: AtomicUsize,
    /// What the next `encode_condition` answers instead of encoding, once.
    encode_error: Mutex<Option<ImageError>>,
    /// What `condition_tokens` refuses every condition with while set, in
    /// the runner's words: a size its processor would not keep.
    count_refusal: Mutex<Option<String>>,
    /// The most latent cells a condition image may have before
    /// `condition_tokens` says the runner has no room for it (its arena); 0
    /// is no limit.
    room_cells: AtomicUsize,
    /// How many denoising steps, across the runner's life, pass without
    /// waiting at the gate: a job that gets that far, then waits.
    free_steps: AtomicUsize,
}

/// `FakeImageRunner`, logging each call, each step behind `gate`.
struct Recording {
    inner: FakeImageRunner,
    log: Arc<Mutex<Vec<Call>>>,
    gate: Arc<Gate>,
    faults: Arc<Faults>,
    steps: usize,
}

impl ImageRunner for Recording {
    fn plan(&self) -> &ImagePlan {
        self.inner.plan()
    }

    fn condition_tokens(&self, prompt: &str, images: &[ImageSize]) -> Result<u64, ConditionUnfit> {
        if let Some(why) = self.faults.count_refusal.lock().unwrap().clone() {
            return Err(ConditionUnfit::Refused(why));
        }
        let room = self.faults.room_cells.load(Ordering::SeqCst);
        if let Some(size) = images.iter().find(|size| room > 0 && size.latent_cells() > room) {
            return Err(ConditionUnfit::NoRoom(format!("no room in the arena for a {size} condition image")));
        }
        Ok(self.inner.condition_tokens(prompt, images)?.saturating_sub(self.faults.undercount.load(Ordering::SeqCst) as u64))
    }

    fn encode_condition(&mut self, condition: &ImageCondition) -> Result<ConditionId, ImageError> {
        self.log.lock().unwrap().push(Call::Encode {
            prompt: condition.prompt.clone(),
            sizes: condition.images.iter().map(|image| image.size).collect(),
            alpha: condition.images.iter().map(|image| image.pixels.chunks_exact(4).any(|px| px[3] != 255)).collect(),
        });
        if let Some(error) = self.faults.encode_error.lock().unwrap().take() {
            return Err(error);
        }
        self.inner.encode_condition(condition)
    }

    fn denoise_step(
        &mut self,
        condition: ConditionId,
        latents: &mut TargetLatents,
        sigma: f32,
        sigma_next: f32,
        velocity: Option<&mut Vec<f32>>,
    ) -> Result<(), ImageError> {
        if self.steps >= self.faults.free_steps.load(Ordering::SeqCst) {
            self.gate.pass();
        }
        self.steps += 1;
        if self.faults.panic_at_step.compare_exchange(self.steps, 0, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
            panic!("a leaf fault at step {}", self.steps);
        }
        self.log.lock().unwrap().push(Call::Step);
        self.inner.denoise_step(condition, latents, sigma, sigma_next, velocity)
    }

    fn vae_decode(&mut self, latents: &TargetLatents) -> Result<RgbaImage, ImageError> {
        self.log.lock().unwrap().push(Call::Decode);
        self.inner.vae_decode(latents)
    }

    fn release_condition(&mut self, condition: ConditionId) {
        self.log.lock().unwrap().push(Call::Release);
        self.inner.release_condition(condition);
    }

    fn condition_bytes(&self) -> u64 {
        self.inner.condition_bytes()
    }
}

/// A server on the image model, its runner [`Recording`] behind `gate`, with
/// small sizes unless a test asks for the defaults.
struct Harness {
    server: Server,
    app: axum::Router,
    log: Arc<Mutex<Vec<Call>>>,
    held: Arc<AtomicUsize>,
    faults: Arc<Faults>,
}

impl Harness {
    fn new(gate: Arc<Gate>) -> Self {
        Self::planning(gate, ignis_core::image::fake_plan())
    }

    /// [`Harness::new`], its load planning `plan`.
    fn planning(gate: Arc<Gate>, plan: ImagePlan) -> Self {
        let inner = FakeImageRunner::new(plan);
        let held = inner.held_conditions();
        let log = Arc::default();
        let faults = Arc::<Faults>::default();
        let runner = Recording { inner, log: Arc::clone(&log), gate, faults: Arc::clone(&faults), steps: 0 };
        let server = Server::from_active(ActiveModel::image(ImageModel::new(IMAGE, Box::new(runner))));
        server.update_live(|live| live.image.sizes = vec![ImageSize::new(64, 64), ImageSize::new(128, 64), ImageSize::new(96, 64)]);
        let app = server.app();
        Self { server, app, log, held, faults }
    }

    fn calls(&self) -> Vec<Call> {
        self.log.lock().unwrap().clone()
    }

    fn encodes(&self) -> Vec<String> {
        self.calls().into_iter().filter_map(|call| if let Call::Encode { prompt, .. } = call { Some(prompt) } else { None }).collect()
    }

    fn jobs(&self) -> Arc<ignis_server::images::ImageJobs> {
        Arc::clone(&self.server.active().image_model().expect("the image model").jobs)
    }
}

fn json_request(uri: &str, body: &Value) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

/// A part of a multipart form.
enum Part<'a> {
    Text(&'a str),
    File(&'a [u8]),
}

const BOUNDARY: &str = "ignis-images-test-boundary";

fn form_request(parts: &[(&str, Part)]) -> Request<Body> {
    let mut body = Vec::new();
    for (name, part) in parts {
        body.extend(format!("--{BOUNDARY}\r\n").bytes());
        match part {
            Part::Text(text) => {
                body.extend(format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n{text}\r\n").bytes());
            }
            Part::File(bytes) => {
                body.extend(
                    format!("Content-Disposition: form-data; name=\"{name}\"; filename=\"image\"\r\nContent-Type: application/octet-stream\r\n\r\n")
                        .bytes(),
                );
                body.extend_from_slice(bytes);
                body.extend(b"\r\n");
            }
        }
    }
    body.extend(format!("--{BOUNDARY}--\r\n").bytes());
    Request::builder()
        .method(Method::POST)
        .uri("/v1/images/edits")
        .header(header::CONTENT_TYPE, format!("multipart/form-data; boundary={BOUNDARY}"))
        .body(Body::from(body))
        .unwrap()
}

async fn send(app: &axum::Router, request: Request<Body>) -> (StatusCode, Value) {
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body = serde_json::from_slice(&bytes).unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, body)
}

/// Standard base64, padded.
fn unbase64(text: &str) -> Vec<u8> {
    let value = |c: u8| match c {
        b'A'..=b'Z' => c - b'A',
        b'a'..=b'z' => c - b'a' + 26,
        b'0'..=b'9' => c - b'0' + 52,
        b'+' => 62,
        b'/' => 63,
        _ => panic!("not base64: {c}"),
    };
    let mut out = Vec::new();
    for chunk in text.as_bytes().chunks(4) {
        let digits: Vec<u8> = chunk.iter().filter(|&&c| c != b'=').map(|&c| value(c)).collect();
        let n = digits.iter().enumerate().fold(0u32, |n, (i, &d)| n | u32::from(d) << (18 - 6 * i));
        out.extend((0..digits.len() - 1).map(|i| (n >> (16 - 8 * i)) as u8));
    }
    out
}

/// A response image: the PNG's header and pixels, untransformed.
struct Png {
    size: ImageSize,
    color: png::ColorType,
    depth: png::BitDepth,
    pixels: Vec<u8>,
}

fn png_of(b64: &Value) -> Png {
    let bytes = unbase64(b64.as_str().expect("b64_json"));
    let mut reader = png::Decoder::new(Cursor::new(bytes)).read_info().expect("a PNG");
    let mut pixels = vec![0; reader.output_buffer_size().unwrap()];
    let frame = reader.next_frame(&mut pixels).unwrap();
    Png { size: ImageSize::new(frame.width, frame.height), color: frame.color_type, depth: frame.bit_depth, pixels }
}

const ALPHA_PNG: &[u8] = include_bytes!("../../artifact/tests/fixtures/vision/images/alpha.png");
const RGB_PNG: &[u8] = include_bytes!("../../artifact/tests/fixtures/vision/images/on_grid.png");
const JPEG: &[u8] = include_bytes!("../../artifact/tests/fixtures/vision/images/odd_width.jpg");
const WEBP: &[u8] = include_bytes!("../../artifact/tests/fixtures/vision/images/lossless.webp");
const ALPHA_WEBP: &[u8] = include_bytes!("fixtures/images/alpha_96x64.webp");

/// AC 3: `n` RGBA 8-bit PNGs as `b64_json`, the seed returned, the same seed
/// twice the same bytes; one prefix for the job, released at its end.
#[tokio::test]
async fn a_generation_returns_n_rgba_pngs_with_its_seed_and_the_same_seed_the_same_bytes() {
    let harness = Harness::new(Gate::opened());
    let body = json!({ "prompt": "a red fox in snow", "n": 3, "size": "128x64", "seed": 11 });
    let (status, first) = send(&harness.app, json_request("/v1/images/generations", &body)).await;
    assert_eq!(status, StatusCode::OK, "{first}");
    assert_eq!(first["seed"], 11);
    assert_eq!((first["size"].as_str(), first["output_format"].as_str()), (Some("128x64"), Some("png")));
    assert!(first["created"].is_u64() && first.get("condition_sizes").is_none(), "{first}");
    let images = first["data"].as_array().expect("data");
    assert_eq!(images.len(), 3);
    for image in images {
        let png = png_of(&image["b64_json"]);
        assert_eq!((png.size, png.color, png.depth), (ImageSize::new(128, 64), png::ColorType::Rgba, png::BitDepth::Eight));
        assert!(image.get("revised_prompt").is_none());
    }
    assert_ne!(images[0]["b64_json"], images[1]["b64_json"], "image i starts from seed + i");
    let (_, again) = send(&harness.app, json_request("/v1/images/generations", &body)).await;
    assert_eq!(again["data"], first["data"], "the same seed, the same bytes");
    let calls = harness.calls();
    assert_eq!(calls.iter().filter(|call| matches!(call, Call::Encode { .. })).count(), 2, "one condition prefix per job");
    assert_eq!(calls.iter().filter(|call| **call == Call::Step).count(), 2 * 3 * 8);
    assert_eq!(harness.held.load(Ordering::SeqCst), 0, "every prefix released");
    assert!(harness.jobs().in_flight().is_idle());
}

/// #735: changing PNG compression live applies to generations and edits,
/// including every image of a job, without changing their RGBA pixels.
#[tokio::test]
async fn image_png_compression_changes_live_without_changing_generated_or_edited_pixels() {
    let harness = Harness::new(Gate::opened());
    let flags: Vec<String> = ["--profile", "none", "--image-sizes", "64x64;128x64;96x64"]
        .into_iter().map(str::to_owned).collect();
    let ignis_server::config::ConfigOutcome::Config(config) =
        ignis_server::config::resolve(&flags, |_| None).unwrap() else { panic!("config") };
    let state = Arc::new(ignis_server::config_http::ConfigState::new(config.clone(), Arc::new(ignis_server::config::file::RealFiles)));
    let server = harness.server.with_config(state);
    server.apply_config(&config);
    let app = server.app();
    let mut reference = [Vec::<Vec<u8>>::new(), Vec::<Vec<u8>>::new()];
    for (mode, level) in [("fast", 0), ("balanced", 2), ("fast", 0)] {
        if !reference[0].is_empty() {
            let patch = Request::builder().method(Method::PATCH).uri("/v1/config")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(json!({ "image": { "png_compression": mode } }).to_string())).unwrap();
            let (status, applied) = send(&app, patch).await;
            assert_eq!(status, StatusCode::OK, "{applied}");
            assert_eq!(applied["applied"], "live", "no model reload");
            assert_eq!(applied["config"]["image"]["png_compression"], mode);
        }
        let generation = json_request("/v1/images/generations", &json!({ "prompt": "a fox", "size": "64x64", "n": 2, "seed": 11 }));
        let edit = form_request(&[("prompt", Part::Text("a fox")), ("size", Part::Text("64x64")),
            ("n", Part::Text("2")), ("seed", Part::Text("11")), ("image[]", Part::File(ALPHA_PNG))]);
        for (route, request) in [generation, edit].into_iter().enumerate() {
            let (status, answer) = send(&app, request).await;
            assert_eq!(status, StatusCode::OK, "{answer}");
            let data = answer["data"].as_array().unwrap();
            assert_eq!(data.len(), 2);
            for (index, image) in data.iter().enumerate() {
                let bytes = unbase64(image["b64_json"].as_str().unwrap());
                let mut offset = 8;
                while &bytes[offset + 4..offset + 8] != b"IDAT" {
                    let length = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
                    offset += length + 12;
                }
                assert_eq!(bytes[offset + 9] >> 6, level, "{mode}, route {route}, image {index}");
                let decoded = png_of(&image["b64_json"]);
                assert_eq!((decoded.size, decoded.color, decoded.depth),
                    (ImageSize::new(64, 64), png::ColorType::Rgba, png::BitDepth::Eight));
                assert!(decoded.pixels.chunks_exact(4).any(|pixel| pixel[3] < 255), "alpha kept");
                if reference[route].len() <= index {
                    reference[route].push(decoded.pixels);
                } else {
                    assert_eq!(decoded.pixels, reference[route][index], "compression preserves RGBA");
                }
            }
        }
    }
    let bad = Request::builder().method(Method::PATCH).uri("/v1/config")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(r#"{"image":{"png_compression":"slow","max_n":1}}"#)).unwrap();
    let (status, error) = send(&app, bad).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert_eq!(server.live().image.max_n, 10, "an invalid compression applies none of the patch");
}

/// AC 3: `background=opaque` is alpha 255, the generated image composited on
/// white; `transparent` is the alpha as generated.
#[tokio::test]
async fn an_opaque_background_is_alpha_255_composited_on_white() {
    let harness = Harness::new(Gate::opened());
    let generate = |background: &str| json!({ "prompt": "a glass", "size": "64x64", "seed": 5, "background": background });
    let (_, auto) = send(&harness.app, json_request("/v1/images/generations", &generate("auto"))).await;
    let (_, transparent) = send(&harness.app, json_request("/v1/images/generations", &generate("transparent"))).await;
    let (_, opaque) = send(&harness.app, json_request("/v1/images/generations", &generate("opaque"))).await;
    assert_eq!(transparent["data"], auto["data"]);
    assert_eq!(opaque["background"], "opaque");
    let generated = png_of(&auto["data"][0]["b64_json"]);
    assert!(generated.pixels.chunks_exact(4).any(|px| px[3] < 255), "the fake's alpha shows");
    let mut expected = RgbaImage { size: generated.size, pixels: generated.pixels };
    ignis_server::images::job::on_white(&mut expected);
    let opaque = png_of(&opaque["data"][0]["b64_json"]);
    assert!(opaque.pixels.chunks_exact(4).all(|px| px[3] == 255));
    assert_eq!(opaque.pixels, expected.pixels);
}

/// AC 3: a request without a seed gets one drawn, and returned; the server's
/// fixed seedless seed, when set, is that seed.
#[tokio::test]
async fn a_seedless_request_gets_a_seed_drawn_and_returned() {
    let harness = Harness::new(Gate::opened());
    let body = json!({ "prompt": "a lighthouse", "size": "64x64" });
    let (_, first) = send(&harness.app, json_request("/v1/images/generations", &body)).await;
    let (_, second) = send(&harness.app, json_request("/v1/images/generations", &body)).await;
    let seed = first["seed"].as_u64().expect("a drawn seed");
    assert_ne!(Some(seed), second["seed"].as_u64(), "two seedless requests, two seeds");
    let replay = json!({ "prompt": "a lighthouse", "size": "64x64", "seed": seed });
    assert_eq!(send(&harness.app, json_request("/v1/images/generations", &replay)).await.1["data"], first["data"]);
    let fixed = Harness { server: harness.server.clone().with_seedless_seed(77), ..harness };
    let app = fixed.server.app();
    assert_eq!(send(&app, json_request("/v1/images/generations", &body)).await.1["seed"], 77);
}

/// AC 4: a size outside `image.sizes` is a 400 listing them; `auto` is
/// 2048x2048 for a generation and, for an edit, the size nearest the last
/// condition image's aspect ratio; `url`, a non-PNG `output_format` and a
/// `mask` are 400s.
#[tokio::test]
async fn sizes_auto_and_the_refused_fields() {
    let harness = Harness::new(Gate::opened());
    let (status, refused) = send(&harness.app, json_request("/v1/images/generations", &json!({ "prompt": "x", "size": "1024x1024" }))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(refused["error"]["param"], "size");
    assert!(refused["error"]["message"].as_str().unwrap().contains("64x64, 128x64, 96x64"), "{refused}");
    for (field, value) in [("response_format", "url"), ("output_format", "jpeg"), ("output_format", "webp")] {
        let (status, refused) = send(&harness.app, json_request("/v1/images/generations", &json!({ "prompt": "x", field: value }))).await;
        assert_eq!((status, refused["error"]["param"].as_str()), (StatusCode::BAD_REQUEST, Some(field)), "{field}: {refused}");
    }
    let (status, refused) = send(&harness.app, form_request(&[("prompt", Part::Text("x")), ("image[]", Part::File(ALPHA_PNG)), ("mask", Part::File(ALPHA_PNG))])).await;
    assert_eq!((status, refused["error"]["param"].as_str()), (StatusCode::BAD_REQUEST, Some("mask")), "{refused}");
    assert!(harness.calls().is_empty(), "nothing refused reached the model");

    // An edit's `auto`: the last image's ratio. 300x200 is 3:2 -> 96x64;
    // 321x240 (4:3) first, then 400x300 (4:3) last -> 96x64 too; a 2:1 last
    // image -> 128x64.
    let wide = RgbaImage { size: ImageSize::new(200, 100), pixels: vec![255; 200 * 100 * 4] };
    let wide = encode(&wide);
    for (images, size) in [(vec![ALPHA_PNG], "96x64"), (vec![wide.as_slice(), ALPHA_PNG], "96x64"), (vec![ALPHA_PNG, wide.as_slice()], "128x64")] {
        let mut parts = vec![("prompt", Part::Text("make it blue")), ("seed", Part::Text("1"))];
        parts.extend(images.iter().map(|bytes| ("image[]", Part::File(bytes))));
        let (status, body) = send(&harness.app, form_request(&parts)).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["size"], size);
        assert_eq!(png_of(&body["data"][0]["b64_json"]).size.to_string(), size);
    }

    // A generation's `auto`, at the checkpoint's presets.
    harness.server.update_live(|live| live.image.sizes = ignis_server::config::DEFAULT_IMAGE_SIZES.to_vec());
    let (status, body) = send(&harness.app, json_request("/v1/images/generations", &json!({ "prompt": "x", "size": "auto", "seed": 2 }))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["size"], "2048x2048");
    assert_eq!(png_of(&body["data"][0]["b64_json"]).size, ImageSize::new(2048, 2048));
}

/// Spec image-model/05 §Sizes against the load's plan: a size in
/// `image.sizes` past the largest target the load makes is a 400 listing the
/// sizes it serves, before anything is queued; `auto` picks among those.
#[tokio::test]
async fn a_size_past_the_loads_largest_target_is_refused_and_auto_never_picks_it() {
    // 96x64 is 24 latent cells; 128x64, 32, is past it.
    let plan = ImagePlan { max_target_cells: ImageSize::new(96, 64).latent_cells(), ..ignis_core::image::fake_plan() };
    let harness = Harness::planning(Gate::opened(), plan);
    let (status, refused) = send(&harness.app, json_request("/v1/images/generations", &json!({ "prompt": "x", "size": "128x64" }))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{refused}");
    assert_eq!(refused["error"]["param"], "size");
    let message = refused["error"]["message"].as_str().unwrap();
    assert!(message.contains("past the largest image") && message.contains("24 latent cells") && message.contains("64x64, 96x64 (image.sizes)"), "{message}");
    assert!(harness.calls().is_empty() && harness.jobs().in_flight().is_idle(), "nothing queued for it");

    // A 2:1 last condition image would get 128x64; 96x64 is the nearest served.
    let wide = encode(&RgbaImage { size: ImageSize::new(200, 100), pixels: vec![255; 200 * 100 * 4] });
    let (status, body) = send(&harness.app, form_request(&[("prompt", Part::Text("x")), ("image[]", Part::File(&wide))])).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["size"], "96x64");
    harness.server.update_live(|live| live.image.sizes = vec![ImageSize::new(128, 64)]);
    let (status, refused) = send(&harness.app, json_request("/v1/images/generations", &json!({ "prompt": "x" }))).await;
    assert_eq!((status, refused["error"]["param"].as_str()), (StatusCode::BAD_REQUEST, Some("size")), "{refused}");
    assert!(refused["error"]["message"].as_str().unwrap().contains("none this image model makes"), "{refused}");
}

/// An RGBA image as a PNG.
fn encode(image: &RgbaImage) -> Vec<u8> {
    ignis_server::images::job::encode_png(image, ignis_server::images::job::PngCompression::Fast)
}

/// AC 5: 1 to 10 condition images in PNG, JPEG and WebP, with and without
/// alpha, are accepted and 11 refused; ten 1024² images are over the fake's
/// budget, so all are scaled by one factor, and `condition_sizes` reports the
/// sizes the runner got.
#[tokio::test]
async fn an_edit_takes_1_to_10_condition_images_scaled_by_one_factor_over_the_budget() {
    let harness = Harness::new(Gate::opened());
    let one = [("prompt", Part::Text("make it blue")), ("image", Part::File(JPEG))];
    let (status, body) = send(&harness.app, form_request(&one)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body.get("condition_sizes").is_none(), "one image fits at 1024²: {body}");
    // 321x240 at 1024² of area: 1184x896.
    assert!(matches!(&harness.calls()[0], Call::Encode { sizes, alpha, .. } if *sizes == [ImageSize::new(1184, 896)] && *alpha == [false]));

    let sources = [ALPHA_PNG, RGB_PNG, JPEG, WEBP, ALPHA_WEBP, ALPHA_PNG, RGB_PNG, JPEG, WEBP, ALPHA_WEBP];
    let mut parts = vec![("prompt", Part::Text("all of them in a row")), ("size", Part::Text("64x64"))];
    parts.extend(sources.iter().map(|bytes| ("image[]", Part::File(bytes))));
    let (status, body) = send(&harness.app, form_request(&parts)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let reported: Vec<String> = body["condition_sizes"].as_array().expect("scaled, so reported").iter().map(|s| s.as_str().unwrap().to_owned()).collect();
    let Some(Call::Encode { sizes, alpha, .. }) = harness.calls().into_iter().filter(|call| matches!(call, Call::Encode { .. })).last() else {
        panic!("no encode");
    };
    assert_eq!(reported, sizes.iter().map(ToString::to_string).collect::<Vec<_>>(), "condition_sizes are the sizes encoded");
    assert_eq!(alpha, [true, false, false, false, true, true, false, false, false, true], "PNG and WebP keep their alpha");
    let tokens: usize = 5 + sizes.iter().map(|size| size.latent_cells()).sum::<usize>();
    assert!(tokens as u64 <= ignis_core::image::fake_plan().condition_budget_tokens, "{tokens}");
    assert!(sizes.iter().all(|size| size.latent_cells() < 4096 && size.latent_cells() >= 1024), "below 1024², above the 512 floor: {sizes:?}");
    // One factor: the same images (the same ratio) got the same size.
    for i in 0..5 {
        assert_eq!(sizes[i], sizes[i + 5], "{i}");
    }

    let mut eleven = vec![("prompt", Part::Text("x"))];
    eleven.extend((0..11).map(|_| ("image[]", Part::File(JPEG))));
    let (status, refused) = send(&harness.app, form_request(&eleven)).await;
    assert_eq!((status, refused["error"]["param"].as_str()), (StatusCode::BAD_REQUEST, Some("image")), "{refused}");
    assert!(refused["error"]["message"].as_str().unwrap().contains("1 to 10"), "{refused}");
    let (status, refused) = send(&harness.app, form_request(&[("prompt", Part::Text("x"))])).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "no image: {refused}");
    let gif = include_bytes!("../../artifact/tests/fixtures/vision/images/animated.gif");
    let (status, refused) = send(&harness.app, form_request(&[("prompt", Part::Text("x")), ("image[]", Part::File(JPEG)), ("image[]", Part::File(gif))])).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(refused["error"]["message"].as_str().unwrap().starts_with("condition image 1: unsupported image"), "{refused}");
    assert_eq!(harness.held.load(Ordering::SeqCst), 0);
}

/// AC 5: below the floor an edit is refused, naming the budget — here the
/// one `image.condition_budget_tokens` lowered the plan's to.
#[tokio::test]
async fn an_edit_below_the_floor_is_refused_naming_the_budget() {
    let harness = Harness::new(Gate::opened());
    harness.server.update_live(|live| live.image.condition_budget_tokens = 2000);
    let parts = [("prompt", Part::Text("x")), ("image[]", Part::File(JPEG)), ("image[]", Part::File(JPEG))];
    let (status, refused) = send(&harness.app, form_request(&parts)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{refused}");
    assert_eq!(refused["error"]["code"], "condition_budget_exceeded");
    assert_eq!(refused["error"]["param"], "image");
    assert!(refused["error"]["message"].as_str().unwrap().contains("condition budget of 2000 tokens"), "{refused}");
    // Raising the floor's budget lets the same request through, scaled.
    harness.server.update_live(|live| live.image.condition_budget_tokens = 4000);
    let (status, body) = send(&harness.app, form_request(&parts)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["condition_sizes"].as_array().map(Vec::len), Some(2));
    assert!(harness.encodes().len() == 1 && harness.jobs().in_flight().is_idle());
}

/// Review-322b F2: a condition image past `image.max_condition_mipx` is
/// refused before its frame is decoded, naming the field; at the cap it is
/// taken. `image.max_condition_decodes` edits decode at a time (review-322cf
/// L1): while that many decodes are held, an edit waits for its turn and
/// reaches nothing; with the limit raised, one more decodes beside them.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn condition_images_decode_one_edit_at_a_time_up_to_the_cap() {
    let harness = Harness::new(Gate::opened());
    harness.server.update_live(|live| live.image.max_condition_mipx = 1);
    // 1024x1025 is one row past 1 Mi pixels; 1024x1024 is the cap itself.
    let past = encode(&RgbaImage { size: ImageSize::new(1024, 1025), pixels: vec![200; 1024 * 1025 * 4] });
    let at = encode(&RgbaImage { size: ImageSize::new(1024, 1024), pixels: vec![200; 1024 * 1024 * 4] });
    let (status, refused) = send(&harness.app, form_request(&[("prompt", Part::Text("x")), ("image[]", Part::File(&past))])).await;
    assert_eq!((status, refused["error"]["param"].as_str()), (StatusCode::BAD_REQUEST, Some("image")), "{refused}");
    let message = refused["error"]["message"].as_str().unwrap();
    assert!(message.contains("1049600 pixels") && message.contains("image.max_condition_mipx"), "{message}");
    let (status, body) = send(&harness.app, form_request(&[("prompt", Part::Text("x")), ("image[]", Part::File(&at))])).await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let held = harness.server.image_decodes.enter(1).await;
    let app = harness.app.clone();
    let waiting = tokio::spawn(async move { send(&app, form_request(&[("prompt", Part::Text("held")), ("image[]", Part::File(JPEG))])).await });
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(!harness.encodes().contains(&"held".to_owned()) && !waiting.is_finished(), "the edit waits for its turn to decode");
    drop(held);
    assert_eq!(waiting.await.unwrap().0, StatusCode::OK);
    assert_eq!(harness.encodes().last().map(String::as_str), Some("held"));

    harness.server.update_live(|live| live.image.max_condition_decodes = 2);
    let held = harness.server.image_decodes.enter(2).await;
    let (status, body) = send(&harness.app, form_request(&[("prompt", Part::Text("beside")), ("image[]", Part::File(JPEG))])).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(harness.server.image_decodes.running(), 1, "its decode ended; the held one runs on");
    drop(held);
}

/// The edit route's body limit is the media one on every load: a form past
/// a text load's 16 MiB is read.
#[tokio::test]
async fn an_edit_form_past_the_text_body_limit_is_read() {
    let harness = Harness::new(Gate::opened());
    let padding = "p".repeat(ignis_server::api::TEXT_REQUEST_BODY_LIMIT + 1);
    let parts = [("prompt", Part::Text("x")), ("image[]", Part::File(JPEG)), ("padding", Part::Text(&padding))];
    let (status, body) = send(&harness.app, form_request(&parts)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

/// A streamed response's events: `(event, data)` in order.
async fn events_of(response: axum::response::Response) -> Vec<(String, Value)> {
    assert_eq!(response.headers()[header::CONTENT_TYPE], "text/event-stream");
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    text.split("\n\n")
        .filter(|block| !block.trim().is_empty())
        .map(|block| {
            let field = |name: &str| block.lines().find_map(|line| line.strip_prefix(name)).map(str::trim).unwrap_or_default().to_owned();
            (field("event:"), serde_json::from_str(&field("data:")).expect("a JSON event"))
        })
        .collect()
}

/// AC 6: the phases in order — encoding once, then per image one denoising
/// event per step and its decoding — and one completed per image, right
/// after its decoding; the images are the ones the same request gets
/// without streaming. An edit's events carry the edit's names.
#[tokio::test]
async fn streaming_emits_the_phases_in_order_and_one_completed_per_image() {
    let harness = Harness::new(Gate::opened());
    let body = json!({ "prompt": "two moons", "n": 2, "size": "64x64", "seed": 3, "stream": true });
    let response = harness.app.clone().oneshot(json_request("/v1/images/generations", &body)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let events = events_of(response).await;
    let mut expected = vec![("image_generation.progress".to_owned(), "encoding".to_owned(), 0, 1)];
    for image in 0..2 {
        expected.extend((1..=8).map(|step| ("image_generation.progress".to_owned(), "denoising".to_owned(), image, step)));
        expected.push(("image_generation.progress".to_owned(), "decoding".to_owned(), image, 1));
        expected.push(("image_generation.completed".to_owned(), String::new(), image, 0));
    }
    let got: Vec<(String, String, u64, u64)> = events
        .iter()
        .map(|(event, data)| {
            assert_eq!(data["type"], event.as_str(), "the event names its type");
            (event.clone(), data["phase"].as_str().unwrap_or_default().to_owned(), data["image"].as_u64().unwrap(), data["step"].as_u64().unwrap_or(0))
        })
        .collect();
    assert_eq!(got, expected.into_iter().map(|(e, p, i, s)| (e, p, i as u64, s as u64)).collect::<Vec<_>>());
    for (_, data) in events.iter().filter(|(_, data)| data["phase"] == "denoising") {
        assert_eq!(data["total"], 8);
    }
    let completed: Vec<&Value> = events.iter().filter(|(event, _)| event.ends_with("completed")).map(|(_, data)| data).collect();
    assert_eq!((completed[0]["seed"].as_u64(), completed[0]["size"].as_str()), (Some(3), Some("64x64")));
    let mut whole = body.clone();
    whole["stream"] = json!(false);
    let (_, whole) = send(&harness.app, json_request("/v1/images/generations", &whole)).await;
    for (image, data) in completed.iter().enumerate() {
        assert_eq!(data["b64_json"], whole["data"][image]["b64_json"], "image {image}");
    }

    let parts = [("prompt", Part::Text("x")), ("image[]", Part::File(JPEG)), ("size", Part::Text("64x64")), ("stream", Part::Text("true"))];
    let events = events_of(harness.app.clone().oneshot(form_request(&parts)).await.unwrap()).await;
    let names: Vec<&str> = events.iter().map(|(event, _)| event.as_str()).collect();
    assert_eq!(names.first(), Some(&"image_edit.progress"));
    assert_eq!(names.last(), Some(&"image_edit.completed"));
    assert_eq!(names.len(), 1 + 8 + 1 + 1);
}

/// Wait (bounded) until `condition` holds.
async fn until(what: &str, mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + PATIENT;
    while !condition() {
        assert!(Instant::now() < deadline, "never: {what}");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// AC 7: one job runs at a time, the waiting ones in the order they came,
/// and one more past `image.queue_depth` is `503 engine_full` with
/// `Retry-After`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_queue_runs_one_job_at_a_time_in_order_and_is_full_past_its_depth() {
    let gate = Gate::shut();
    let harness = Harness::new(Arc::clone(&gate));
    harness.server.update_live(|live| live.image.queue_depth = 2);
    let jobs = harness.jobs();
    let request = |prompt: &str| json_request("/v1/images/generations", &json!({ "prompt": prompt, "size": "64x64", "seed": 1 }));
    let mut answers = Vec::new();
    for (prompt, in_flight) in [("first", (0, 1)), ("second", (1, 1)), ("third", (2, 1))] {
        answers.push(tokio::spawn(harness.app.clone().oneshot(request(prompt))));
        until(prompt, || {
            let now = jobs.in_flight();
            (now.waiting, now.running) == in_flight
        })
        .await;
    }
    let full = harness.app.clone().oneshot(request("fourth")).await.unwrap();
    assert_eq!(full.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(full.headers()[header::RETRY_AFTER], "1");
    let body: Value = serde_json::from_slice(&full.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["error"]["code"], "engine_full", "{body}");
    assert_eq!(harness.encodes(), ["first"], "the others wait");

    gate.open();
    for answer in answers {
        assert_eq!(answer.await.unwrap().unwrap().status(), StatusCode::OK);
    }
    assert_eq!(harness.encodes(), ["first", "second", "third"], "in the order they came");
    // One at a time: each job's calls are contiguous.
    let job_of_each_call: Vec<usize> = harness
        .calls()
        .iter()
        .scan(0, |job, call| {
            if matches!(call, Call::Encode { .. }) {
                *job += 1;
            }
            Some(*job)
        })
        .collect();
    assert!(job_of_each_call.windows(2).all(|pair| pair[0] <= pair[1]));
    let releases = harness.calls().iter().enumerate().filter(|(_, call)| **call == Call::Release).map(|(at, _)| job_of_each_call[at]).collect::<Vec<_>>();
    assert_eq!(releases, [1, 2, 3], "each job released its prefix before the next began");
    assert!(jobs.in_flight().is_idle());
}

/// AC 7: a client that disconnects cancels its job — a streamed one between
/// two denoising steps, its prefix released; a waiting one before it
/// starts, leaving the queue.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_disconnect_cancels_its_job_between_steps_or_before_it_starts() {
    let gate = Gate::shut();
    let harness = Harness::new(Arc::clone(&gate));
    let jobs = harness.jobs();
    let streamed = json!({ "prompt": "streamed", "n": 2, "size": "64x64", "stream": true });
    let response = harness.app.clone().oneshot(json_request("/v1/images/generations", &streamed)).await.unwrap();
    let mut body = response.into_body();
    // The encoding event and the first step's: the job is inside that step.
    let first = body.frame().await.expect("an event").unwrap().into_data().unwrap();
    assert!(String::from_utf8_lossy(&first).contains("encoding"), "{first:?}");
    until("the first step is running", || harness.held.load(Ordering::SeqCst) == 1).await;

    let waiting = tokio::spawn(harness.app.clone().oneshot(json_request("/v1/images/generations", &json!({ "prompt": "queued", "size": "64x64" }))));
    until("the second job waits", || jobs.in_flight().waiting == 1).await;
    waiting.abort();
    until("the waiting job left the queue", || jobs.in_flight().waiting == 0).await;

    drop(body);
    gate.open();
    until("the cancelled job released its prefix", || harness.held.load(Ordering::SeqCst) == 0 && jobs.in_flight().is_idle()).await;
    let calls = harness.calls();
    let steps = calls.iter().filter(|call| **call == Call::Step).count();
    assert!(steps < 8, "cancelled between steps, not run to the end: {steps} steps");
    assert!(!calls.contains(&Call::Decode), "no image was decoded");
    assert_eq!(calls.last(), Some(&Call::Release));
    assert_eq!(harness.encodes(), ["streamed"], "the queued job never started");
}

/// ADR 0058: a job is held to `server.request_timeout` from its start, never
/// from its queue. Past it the job ends between two steps, its prefix
/// released and the runner free, and its request is `504 request_timeout` —
/// always an error, whatever `server.timeout_reply` says: a cut image is no
/// image. The job queued behind it, whose own deadline only began with its
/// turn, runs to its end. The same reply under either `reply`.
async fn check_a_job_past_its_deadline_is_a_504_and_the_one_behind_it_runs(reply: TimeoutReply) {
    let gate = Gate::shut();
    let harness = Harness::new(Arc::clone(&gate));
    harness.server.update_live(|live| {
        live.request_timeout = Duration::from_secs(1);
        live.timeout_reply = reply;
    });
    let jobs = harness.jobs();
    let request = |prompt: &str| json_request("/v1/images/generations", &json!({ "prompt": prompt, "n": 2, "size": "64x64" }));
    let cut = tokio::spawn(harness.app.clone().oneshot(request("cut")));
    until("the job runs", || harness.held.load(Ordering::SeqCst) == 1).await;
    let behind = tokio::spawn(harness.app.clone().oneshot(request("behind")));
    until("the second job waits", || jobs.in_flight().waiting == 1).await;
    // Time passing is the point here, not a synchronization.
    tokio::time::sleep(Duration::from_millis(1100)).await;
    gate.open();
    let response = cut.await.unwrap().unwrap();
    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    let body: Value = serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["error"]["code"], "request_timeout", "{body}");
    assert!(body["error"]["message"].as_str().unwrap().contains("--server-request-timeout"), "{body}");
    let (status, body) = {
        let response = behind.await.unwrap().unwrap();
        let status = response.status();
        (status, response.into_body().collect().await.unwrap().to_bytes())
    };
    assert_eq!(status, StatusCode::OK, "{body:?}");
    let calls = harness.calls();
    let first_job = calls.iter().position(|call| *call == Call::Release).expect("the cut job released its prefix");
    assert_eq!(calls[..first_job].iter().filter(|call| **call == Call::Step).count(), 1, "cut after the step it was in: {calls:?}");
    assert!(!calls[..first_job].contains(&Call::Decode), "no image was decoded");
    until("the runner is free", || harness.held.load(Ordering::SeqCst) == 0 && jobs.in_flight().is_idle()).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_job_past_its_deadline_is_a_504_and_the_one_behind_it_runs() {
    check_a_job_past_its_deadline_is_a_504_and_the_one_behind_it_runs(TimeoutReply::Partial).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_job_past_its_deadline_is_a_504_under_the_error_reply_too() {
    check_a_job_past_its_deadline_is_a_504_and_the_one_behind_it_runs(TimeoutReply::Error).await;
}

/// ADR 0058: a streamed job is held to `server.stream_timeout`. The images
/// it completed have gone out; then its `error` event says
/// `request_timeout`, under either `reply`.
async fn check_a_streamed_job_past_its_deadline_sends_its_images_then_an_error(reply: TimeoutReply) {
    let gate = Gate::shut();
    let harness = Harness::new(Arc::clone(&gate));
    harness.server.update_live(|live| {
        live.request_timeout = Duration::from_secs(600);
        live.stream_timeout = Duration::from_secs(1);
        live.timeout_reply = reply;
    });
    // The whole first image, then the job waits in the second's first step.
    harness.faults.free_steps.store(8, Ordering::SeqCst);
    let body = json!({ "prompt": "two moons", "n": 2, "size": "64x64", "stream": true });
    let response = harness.app.clone().oneshot(json_request("/v1/images/generations", &body)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let events = tokio::spawn(events_of(response));
    until("the second image's first step runs", || harness.calls().iter().filter(|call| **call == Call::Step).count() == 8).await;
    // Time passing is the point here, not a synchronization.
    tokio::time::sleep(Duration::from_millis(1100)).await;
    gate.open();
    let events = events.await.unwrap();
    let completed: Vec<u64> = events.iter().filter(|(event, _)| event.ends_with("completed")).map(|(_, data)| data["image"].as_u64().unwrap()).collect();
    assert_eq!(completed, [0], "the first image went out");
    let (last, data) = events.last().unwrap();
    assert_eq!(last, "error", "{events:?}");
    assert_eq!(data["error"]["code"], "request_timeout", "{data}");
    assert!(data["error"]["message"].as_str().unwrap().contains("--server-stream-timeout"), "{data}");
    until("the prefix released", || harness.held.load(Ordering::SeqCst) == 0 && harness.jobs().in_flight().is_idle()).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_streamed_job_past_its_deadline_sends_its_images_then_a_request_timeout_error() {
    check_a_streamed_job_past_its_deadline_sends_its_images_then_an_error(TimeoutReply::Partial).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_streamed_job_past_its_deadline_ends_with_the_same_error_under_the_error_reply() {
    check_a_streamed_job_past_its_deadline_sends_its_images_then_an_error(TimeoutReply::Error).await;
}

/// `0` is no deadline for a streamed job either, held to
/// `server.stream_timeout` alone: one held past its `request_timeout` makes
/// its image, and no `error` event follows it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_zero_stream_timeout_leaves_a_streamed_job_to_its_images() {
    let gate = Gate::shut();
    let harness = Harness::new(Arc::clone(&gate));
    harness.server.update_live(|live| {
        live.request_timeout = Duration::from_millis(10);
        live.stream_timeout = Duration::ZERO;
    });
    let body = json!({ "prompt": "x", "size": "64x64", "stream": true });
    let response = harness.app.clone().oneshot(json_request("/v1/images/generations", &body)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let events = tokio::spawn(events_of(response));
    until("the job runs", || harness.held.load(Ordering::SeqCst) == 1).await;
    // Time passing is the point here, not a synchronization.
    tokio::time::sleep(Duration::from_millis(100)).await;
    gate.open();
    let events = events.await.unwrap();
    let (last, _) = events.last().unwrap();
    assert_eq!(last, "image_generation.completed", "{events:?}");
}

/// `0` is no deadline for a job either: one held well past a zero-length
/// deadline makes its image.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_zero_request_timeout_leaves_a_job_to_its_images() {
    let gate = Gate::shut();
    let harness = Harness::new(Arc::clone(&gate));
    harness.server.update_live(|live| live.request_timeout = Duration::ZERO);
    let answer = tokio::spawn(harness.app.clone().oneshot(json_request("/v1/images/generations", &json!({ "prompt": "x", "size": "64x64" }))));
    until("the job runs", || harness.held.load(Ordering::SeqCst) == 1).await;
    // Time passing is the point here, not a synchronization.
    tokio::time::sleep(Duration::from_millis(100)).await;
    gate.open();
    assert_eq!(answer.await.unwrap().unwrap().status(), StatusCode::OK);
}

/// A model switch's stop (`ImageJobs::close`) ends the running job between
/// steps: a whole request is answered `503 model_stopped`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stopped_model_ends_its_job_with_model_stopped() {
    let gate = Gate::shut();
    let harness = Harness::new(Arc::clone(&gate));
    let jobs = harness.jobs();
    let answer = tokio::spawn(harness.app.clone().oneshot(json_request("/v1/images/generations", &json!({ "prompt": "x", "size": "64x64" }))));
    until("the job runs", || harness.held.load(Ordering::SeqCst) == 1).await;
    jobs.close();
    gate.open();
    let response = answer.await.unwrap().unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert!(response.headers().contains_key(header::RETRY_AFTER));
    let body: Value = serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["error"]["code"], "model_stopped");
    until("the prefix released", || harness.held.load(Ordering::SeqCst) == 0).await;
}

/// Review-322b F1: a runner call that panics mid-job neither aborts the
/// process nor keeps the job's prefix — the lock it poisoned still releases
/// it — and its request is answered `500 engine_error`; streamed, with an
/// `error` event in place of the images it never got (L7). Then the model is
/// failed (322g's note): its queue closed and its runner dropped before the
/// request hears the end, so the request after it is `503 model_stopped`.
/// This server cannot load the model again; one with a switcher unloads it
/// and loads it for the next request (`model_kind.rs`).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_runner_that_panics_mid_job_is_an_engine_error_and_fails_the_model() {
    let harness = Harness::new(Gate::opened());
    let body = json!({ "prompt": "a fault", "n": 2, "size": "64x64", "seed": 4 });
    harness.faults.panic_at_step.store(3, Ordering::SeqCst);
    let (status, refused) = send(&harness.app, json_request("/v1/images/generations", &body)).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{refused}");
    assert_eq!(refused["error"]["code"], "engine_error");
    assert_eq!(harness.calls().last(), Some(&Call::Release), "the prefix released through the poisoned lock");
    assert_eq!(harness.held.load(Ordering::SeqCst), 0);
    let model = harness.server.active().image_model().expect("the image model").clone();
    assert!(model.jobs.is_closed() && model.with_runner(|_| ()).is_none(), "the model failed: its queue closed, its runner gone");
    until("the job ended", || harness.jobs().in_flight().is_idle()).await;
    let (status, stopped) = send(&harness.app, json_request("/v1/images/generations", &body)).await;
    assert_eq!((status, stopped["error"]["code"].as_str()), (StatusCode::SERVICE_UNAVAILABLE, Some("model_stopped")), "{stopped}");

    // Streamed, on a model of its own: the first image is sent, then the
    // step of the second panics (steps 1 to 8 are the first image's).
    let harness = Harness::new(Gate::opened());
    harness.faults.panic_at_step.store(8 + 3, Ordering::SeqCst);
    let mut streamed = body.clone();
    streamed["stream"] = json!(true);
    let events = events_of(harness.app.clone().oneshot(json_request("/v1/images/generations", &streamed)).await.unwrap()).await;
    let completed = events.iter().filter(|(event, _)| event == "image_generation.completed").count();
    let (last, error) = events.last().expect("events");
    assert_eq!((completed, last.as_str()), (1, "error"), "{events:?}");
    assert_eq!(error["error"]["code"], "engine_error", "{error}");
    until("the prefix released", || harness.held.load(Ordering::SeqCst) == 0 && harness.jobs().in_flight().is_idle()).await;
    assert!(harness.jobs().is_closed());
}

/// Review-322b L2: a runner whose prefix count drifted below the one its
/// encode refuses by — the fit passes, the encode refuses `OverBudget` — is
/// still the request's size: `400 condition_budget_exceeded`, streamed as the
/// `error` event's code, the prefix never held.
#[tokio::test]
async fn a_prefix_the_runner_refuses_past_the_budget_is_a_400() {
    // The JPEG at 1024² of area is 1184x896, 4,144 cells, and the prompt one
    // token: 4,145, counted as 4,144 — the plan's whole budget.
    let plan = ImagePlan { condition_budget_tokens: 4144, ..ignis_core::image::fake_plan() };
    let harness = Harness::planning(Gate::opened(), plan);
    harness.faults.undercount.store(1, Ordering::SeqCst);
    let parts = [("prompt", Part::Text("x")), ("image[]", Part::File(JPEG))];
    let (status, refused) = send(&harness.app, form_request(&parts)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{refused}");
    assert_eq!((refused["error"]["code"].as_str(), refused["error"]["param"].as_str()), (Some("condition_budget_exceeded"), Some("image")), "{refused}");
    let streamed = [("prompt", Part::Text("x")), ("image[]", Part::File(JPEG)), ("stream", Part::Text("true"))];
    let events = events_of(harness.app.clone().oneshot(form_request(&streamed)).await.unwrap()).await;
    let (last, error) = events.last().expect("events");
    assert_eq!(last, "error", "{events:?}");
    assert_eq!(error["error"]["code"], "condition_budget_exceeded", "{error}");
    assert_eq!(harness.held.load(Ordering::SeqCst), 0);
}

/// Spec image-model/04: what the image leaf refuses is the request's — a
/// prompt whose RoPE positions run past the table, a condition image the
/// encoder's processor would resize — `400 image_refused` in the leaf's
/// words, whole or streamed; what it fails at is the server's, `500
/// engine_error`. Neither leaves a prefix held.
#[tokio::test]
async fn what_the_leaf_refuses_is_a_400_and_what_it_fails_at_a_500() {
    let harness = Harness::new(Gate::opened());
    let body = json!({ "prompt": "a very long prompt", "size": "64x64", "seed": 2 });
    let refusal = "the condition's RoPE positions run past [-1024, 8192)";
    *harness.faults.encode_error.lock().unwrap() = Some(ImageError::Refused(refusal.into()));
    let (status, refused) = send(&harness.app, json_request("/v1/images/generations", &body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{refused}");
    assert_eq!((refused["error"]["type"].as_str(), refused["error"]["code"].as_str()), (Some("invalid_request_error"), Some("image_refused")));
    assert_eq!(refused["error"]["message"], refusal);

    *harness.faults.encode_error.lock().unwrap() = Some(ImageError::Refused(refusal.into()));
    let mut streamed = body.clone();
    streamed["stream"] = json!(true);
    let events = events_of(harness.app.clone().oneshot(json_request("/v1/images/generations", &streamed)).await.unwrap()).await;
    let (last, error) = events.last().expect("events");
    assert_eq!(last, "error", "{events:?}");
    assert_eq!((error["error"]["code"].as_str(), error["error"]["message"].as_str()), (Some("image_refused"), Some(refusal)), "{error}");

    *harness.faults.encode_error.lock().unwrap() = Some(ImageError::Leaf("the DiT's blocks: an illegal address".into()));
    let (status, failed) = send(&harness.app, json_request("/v1/images/generations", &body)).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{failed}");
    assert_eq!(failed["error"]["code"], "engine_error");
    assert_eq!(harness.held.load(Ordering::SeqCst), 0);
    let (status, _) = send(&harness.app, json_request("/v1/images/generations", &body)).await;
    assert_eq!(status, StatusCode::OK, "the model serves on");
}

/// Review-321d F1: what the runner's count refuses — a condition image its
/// processor would not keep at 1024² of area — is `400 image_refused` in its
/// words, before anything is encoded; what it has no room for is scaled down
/// as for the budget, and past its room even at the floor is `400
/// condition_budget_exceeded`, naming the room.
#[tokio::test]
async fn what_the_count_refuses_is_image_refused_and_what_it_has_no_room_for_is_scaled() {
    let harness = Harness::new(Gate::opened());
    let refusal = "condition image 0 is 16192x64: the encoder's processor refuses it (at most 200:1)";
    *harness.faults.count_refusal.lock().unwrap() = Some(refusal.into());
    let parts = [("prompt", Part::Text("x")), ("image[]", Part::File(JPEG))];
    let (status, refused) = send(&harness.app, form_request(&parts)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{refused}");
    assert_eq!((refused["error"]["code"].as_str(), refused["error"]["message"].as_str()), (Some("image_refused"), Some(refusal)), "{refused}");
    assert!(harness.encodes().is_empty(), "refused before anything is encoded");
    *harness.faults.count_refusal.lock().unwrap() = None;

    // The JPEG at 1024² of area is 1184x896, 4,144 cells: past a room of
    // 2,000, scaled to fit it.
    harness.faults.room_cells.store(2000, Ordering::SeqCst);
    let (status, body) = send(&harness.app, form_request(&parts)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let Some(Call::Encode { sizes, .. }) = harness.calls().into_iter().find(|call| matches!(call, Call::Encode { .. })) else { panic!("no encode") };
    assert!(sizes[0].latent_cells() <= 2000 && sizes[0].latent_cells() > 1000, "{sizes:?}");
    assert_eq!(body["condition_sizes"], json!([sizes[0].to_string()]), "{body}");

    // At the 512 floor it is 608x448, 1,064 cells: past a room of 1,000.
    let at_floor = ignis_server::images::budget::condition_size(512.0 * 512.0, 321.0 / 240.0);
    assert_eq!((at_floor, at_floor.latent_cells()), (ImageSize::new(608, 448), 1064));
    harness.faults.room_cells.store(1000, Ordering::SeqCst);
    let (status, refused) = send(&harness.app, form_request(&parts)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{refused}");
    assert_eq!((refused["error"]["code"].as_str(), refused["error"]["param"].as_str()), (Some("condition_budget_exceeded"), Some("image")), "{refused}");
    let message = refused["error"]["message"].as_str().unwrap();
    assert!(message.contains("512x512 pixels of area") && message.contains("no room in the arena for a 608x448"), "{message}");
    assert_eq!(harness.encodes().len(), 1, "only the scaled edit was encoded");
    assert_eq!(harness.held.load(Ordering::SeqCst), 0);
}

/// Spec image-model/05 §Prompt enhancement: on a server that cannot switch
/// models and has no enhancer loaded, an enhanced request is made from its
/// prompt as sent, the answer saying so — streamed too, in each completed
/// event.
#[tokio::test]
async fn an_enhanced_request_on_a_server_that_cannot_switch_uses_its_prompt_as_sent() {
    let harness = Harness::new(Gate::opened());
    let body = json!({ "prompt": "a heron", "size": "64x64", "prompt_enhancement": true });
    let (status, answer) = send(&harness.app, json_request("/v1/images/generations", &body)).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    assert!(answer["data"][0].get("revised_prompt").is_none(), "{answer}");
    let warning = answer["warnings"][0].as_str().unwrap_or_default();
    assert!(warning.contains("qwen3.8-27b did not rewrite the prompt") && warning.contains("cannot switch models"), "{answer}");
    let mut streamed = body.clone();
    streamed["stream"] = json!(true);
    let events = events_of(harness.app.clone().oneshot(json_request("/v1/images/generations", &streamed)).await.unwrap()).await;
    let (last, completed) = events.last().expect("events");
    assert_eq!(last, "image_generation.completed");
    assert_eq!(completed["warnings"][0], warning, "{completed}");
    assert_eq!(harness.encodes(), ["a heron", "a heron"]);
}

/// The image routes are model routes: with a text model loaded and no
/// `model` naming the image model, `409 model_kind_mismatch`; they count as
/// the model's use for idle unload. The edit's body limit is the media one
/// on a text load without vision too: a form past 16 MiB is read, and
/// refused for the kind.
#[tokio::test]
async fn with_a_text_model_loaded_an_image_request_is_a_kind_mismatch() {
    let scheduler = ignis_core::ConcreteScheduler::with_config(
        ignis_core::SchedulerConfig { model: "test-model".into(), ..ignis_core::SchedulerConfig::default() },
        Arc::new(ignis_core::MockCompute::new()),
    );
    let engine = ignis_server::engine::Engine::new(Box::new(scheduler));
    let server = Server::new(engine, Box::new(ignis_server::template::SimpleTemplateProvider));
    let app = server.app();
    let (status, refused) = send(&app, json_request("/v1/images/generations", &json!({ "prompt": "x" }))).await;
    assert_eq!(status, StatusCode::CONFLICT, "{refused}");
    assert_eq!(refused["error"]["code"], "model_kind_mismatch");
    let (status, refused) = send(&app, form_request(&[("prompt", Part::Text("x")), ("image[]", Part::File(JPEG))])).await;
    assert_eq!((status, refused["error"]["code"].as_str()), (StatusCode::CONFLICT, Some("model_kind_mismatch")));
    let padding = "p".repeat(ignis_server::api::TEXT_REQUEST_BODY_LIMIT + 1);
    let parts = [("prompt", Part::Text("x")), ("image[]", Part::File(JPEG)), ("padding", Part::Text(&padding))];
    let (status, refused) = send(&app, form_request(&parts)).await;
    assert_eq!((status, refused["error"]["code"].as_str()), (StatusCode::CONFLICT, Some("model_kind_mismatch")), "{refused}");
}
