//! The image routes (spec image-model/05 §Image routes, ADR 0050):
//! `POST /v1/images/generations` (JSON) and `POST /v1/images/edits`
//! (multipart), OpenAI's Images API on the image model, with the job queue
//! behind them ([`super::job`]).
//!
//! A request's fields are checked against `image.*`, and an edit's condition
//! images decoded and resized to their unscaled size ([`Conditions`]),
//! before anything loads or switches — then the kind check
//! and the implicit switch ([`crate::api::switch_to_named_model`]), the
//! queue's admission, the condition budget ([`super::budget`]) and the
//! resize of the images it scales, and the job. Two 400s come after the switch, since they need the
//! loaded image model: a size past what its load makes (and `auto` with none
//! of `image.sizes` made), and a prefix over the condition budget, which the
//! runner's tokenizer counts. Its answer is one JSON body, or with `stream` its events as
//! Server-Sent Events, one per progress unit and one per finished image.

use std::convert::Infallible;
use std::pin::Pin;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

use axum::extract::{Multipart, State};
use axum::http::{HeaderValue, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_core::Stream;
use ignis_core::compute::ModelKind;
use ignis_core::image::{ImageCondition, ImageError, ImageSize, RgbaImage};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use utoipa::ToSchema;

use super::budget::{self, Fitted};
use super::enhance::Rewritten;
use super::job::{self, Background, ImageJob, JobContext, JobEnd, JobEvent, Phase, Progress};
use super::{ImageModel, Running, Waiting};
use crate::api::{
    bad_request_param, check_kind, check_served_kind, error_response, error_response_naming, implicit_switch_refused,
    load_refused, refused_field, split_model_lane, unknown_model, unswitchable_named, ApiError, ENGINE_ERROR_MESSAGE,
};
use crate::config::schema::ImageGroup;
use crate::model_switch::{ImplicitRefusal, Steps, SwitchOutcome, SwitchRefusal, SwitchStep};
use crate::openai_fields::{self, Surface};
use crate::{ModelSource, ModelStatus, Server};

/// `POST /v1/images/generations`'s body, and the fields of an edit's form
/// beside its images.
#[derive(Debug, Default, Deserialize, ToSchema)]
pub(crate) struct GenerationRequest {
    /// Absent, or the loaded image model's id: served as it is. Another known
    /// image model is switched to first (`switch.allow_implicit`); a text
    /// model is `409 model_kind_mismatch`.
    model: Option<String>,
    /// What to make. Required; passed to the model as it is.
    #[schema(required = true, value_type = String)]
    prompt: Option<String>,
    /// How many images, 1 to `image.max_n` (default 1): made one after
    /// another from one condition prefix.
    #[schema(minimum = 1)]
    n: Option<u32>,
    /// One of `image.sizes` (`<width>x<height>`), or `auto` (the default).
    size: Option<String>,
    /// `b64_json` only: the images come back inline. `url` is refused.
    response_format: Option<String>,
    /// `png` only.
    output_format: Option<String>,
    /// `auto` (the default) or `transparent`: the alpha channel as generated.
    /// `opaque`: composited on white, alpha 255.
    background: Option<String>,
    /// ignis: image `i` starts from the noise of `seed + i`. Absent, one is
    /// drawn and returned. The same seed gives the same bytes on the same
    /// build and machine.
    seed: Option<u64>,
    /// ignis: rewrite the prompt with the text model `image.prompt_enhancer`
    /// first; absent, `image.prompt_enhancement` decides.
    prompt_enhancement: Option<bool>,
    /// `true`: Server-Sent Events, the progress and each image as it is made.
    #[serde(default)]
    stream: bool,
    /// Every field not named above, for `crate::openai_fields` to classify.
    #[serde(flatten)]
    #[schema(ignore)]
    other: serde_json::Map<String, serde_json::Value>,
}

/// `POST /v1/images/edits`'s form (`multipart/form-data`): the condition
/// images and the generation's fields, each a text part.
#[derive(Debug, Default, ToSchema)]
pub(crate) struct EditForm {
    /// The condition images, 1 to `image.max_condition_images`, in order:
    /// PNG, JPEG or WebP, with or without alpha. One may be sent as `image`.
    #[serde(rename = "image[]")]
    #[schema(value_type = Vec<String>, format = Binary)]
    images: Vec<Vec<u8>>,
    /// Not supported: `400`.
    #[schema(value_type = Option<String>, format = Binary)]
    mask: Option<Vec<u8>>,
    #[serde(flatten)]
    fields: GenerationRequest,
}

/// The images, as one JSON body.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ImagesResponse {
    /// Unix seconds.
    created: u64,
    data: Vec<ImageData>,
    /// ignis: the job's seed, the request's or the one drawn for it; image
    /// `i` started from `seed + i`.
    seed: u64,
    /// The images' size, `auto` resolved.
    size: String,
    background: &'static str,
    output_format: &'static str,
    /// ignis: each condition image's size as encoded, in order — only when
    /// the condition budget scaled them below 1024² of area.
    #[serde(skip_serializing_if = "Option::is_none")]
    condition_sizes: Option<Vec<String>>,
    /// ignis: what the request should know of how it was served — prompt
    /// enhancement that did not rewrite the prompt, or rewrote an edit's
    /// without seeing its images. Absent when there is nothing to say.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    warnings: Vec<String>,
}

/// One image.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ImageData {
    /// The PNG, RGBA 8-bit, base64.
    b64_json: String,
    /// The prompt as prompt enhancement rewrote it, when it did.
    #[serde(skip_serializing_if = "Option::is_none")]
    revised_prompt: Option<String>,
}

/// One Server-Sent Event of a streamed image request: `event:` is its
/// `type`, `data:` the object. `image_generation.*` on `/v1/images/generations`,
/// `image_edit.*` on `/v1/images/edits` (OpenAI's names for each route's
/// events).
#[derive(Debug, Serialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum ImageStreamEvent {
    Progress(ProgressEvent),
    Completed(CompletedEvent),
    Error(ErrorEvent),
}

/// `image_generation.progress` / `image_edit.progress`: a unit of the
/// request's work starting — `encoding` once, then per image each
/// `denoising` step and its `decoding`. `switching` and `loading` come first
/// when the request waits on a switch or a load, `enhancing` when its prompt
/// is rewritten.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ProgressEvent {
    r#type: &'static str,
    #[serde(flatten)]
    progress: Progress,
}

/// `image_generation.completed` / `image_edit.completed`: one per image, as
/// soon as it is made. No partial images are sent.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct CompletedEvent {
    r#type: &'static str,
    /// Which image, 0-based.
    image: u32,
    b64_json: String,
    created_at: u64,
    size: String,
    background: &'static str,
    output_format: &'static str,
    /// The job's seed: this image started from `seed + image`.
    seed: u64,
    /// The prompt as prompt enhancement rewrote it, when it did.
    #[serde(skip_serializing_if = "Option::is_none")]
    revised_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    condition_sizes: Option<Vec<String>>,
    /// As in the whole answer.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    warnings: Vec<String>,
}

/// `error`: the job ended before its last image, the status already sent.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ErrorEvent {
    r#type: &'static str,
    #[serde(flatten)]
    error: ApiError,
}

/// Which route a request came by: the names of its streamed events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Route {
    Generations,
    Edits,
}

impl Route {
    /// The field a prefix past the condition budget is laid to: an edit's
    /// images, a generation's prompt.
    fn condition_param(self) -> &'static str {
        match self {
            Self::Generations => "prompt",
            Self::Edits => "image",
        }
    }

    fn progress(self) -> &'static str {
        match self {
            Self::Generations => "image_generation.progress",
            Self::Edits => "image_edit.progress",
        }
    }

    fn completed(self) -> &'static str {
        match self {
            Self::Generations => "image_generation.completed",
            Self::Edits => "image_edit.completed",
        }
    }
}

/// A request whose fields passed their rules.
#[derive(Debug)]
struct ImageRequest {
    model: Option<String>,
    prompt: String,
    n: u32,
    /// `None`: `auto`.
    size: Option<ImageSize>,
    background: Background,
    seed: Option<u64>,
    /// Whether its prompt is rewritten first: the request's
    /// `prompt_enhancement`, else `image.prompt_enhancement`.
    enhance: bool,
    stream: bool,
}

/// The fields' rules (spec image-model/05 §Image routes) against `limits`,
/// and OpenAI's fields the routes do not read against their table
/// (`crate::openai_fields`).
fn checked(fields: GenerationRequest, limits: &ImageGroup) -> Result<ImageRequest, Response> {
    if let Err(refusal) = openai_fields::check(&fields.other, Surface::Images) {
        return Err(refused_field(refusal));
    }
    let Some(prompt) = fields.prompt else {
        return Err(bad_request_param("prompt is required", "prompt"));
    };
    let n = fields.n.unwrap_or(1);
    if n == 0 || n > limits.max_n {
        return Err(bad_request_param(&format!("n must be between 1 and {} (image.max_n)", limits.max_n), "n"));
    }
    let size = match fields.size.as_deref() {
        None | Some("auto") => None,
        Some(asked) => match ImageSize::parse(asked).ok().filter(|size| limits.sizes.contains(size)) {
            Some(size) => Some(size),
            None => {
                let sizes: Vec<String> = limits.sizes.iter().map(ToString::to_string).collect();
                return Err(bad_request_param(
                    &format!("size `{asked}` is not served: it must be auto or one of {} (image.sizes)", sizes.join(", ")),
                    "size",
                ));
            }
        },
    };
    match fields.response_format.as_deref() {
        None | Some("b64_json") => {}
        Some("url") => {
            return Err(bad_request_param(
                "response_format url is not served: the images come back inline as b64_json, never stored to be fetched",
                "response_format",
            ))
        }
        Some(other) => return Err(bad_request_param(&format!("response_format `{other}` is not b64_json"), "response_format")),
    }
    match fields.output_format.as_deref() {
        None | Some("png") => {}
        Some(other) => return Err(bad_request_param(&format!("output_format `{other}` is not served: png only"), "output_format")),
    }
    let background = match fields.background.as_deref() {
        None => Background::Auto,
        Some(asked) => Background::parse(asked).ok_or_else(|| {
            bad_request_param(&format!("background `{asked}` is not auto, transparent or opaque"), "background")
        })?,
    };
    Ok(ImageRequest {
        model: fields.model,
        prompt,
        n,
        size,
        background,
        seed: fields.seed,
        enhance: fields.prompt_enhancement.unwrap_or(limits.prompt_enhancement),
        stream: fields.stream,
    })
}

/// `auto` (spec image-model/05 §Image routes): among `sizes`, the one whose
/// aspect ratio is nearest `ratio` — 1 for a generation, the last condition
/// image's for an edit — the larger on a tie, so 2048x2048 for a generation
/// with the default sizes.
fn auto_size(sizes: &[ImageSize], ratio: f64) -> Option<ImageSize> {
    // On the log scale, where 2:1 and 1:2 are as far from 1:1.
    let distance = |size: &ImageSize| ((f64::from(size.width) / f64::from(size.height)).ln() - ratio.ln()).abs();
    sizes.iter().copied().min_by(|a, b| {
        distance(a).total_cmp(&distance(b)).then_with(|| b.latent_cells().cmp(&a.latent_cells()))
    })
}

/// `503` with `code` and `Retry-After: 1`.
fn unavailable(type_: &str, code: &str, message: impl Into<String>) -> Response {
    let mut response = error_response(StatusCode::SERVICE_UNAVAILABLE, type_, code, message);
    response.headers_mut().insert(axum::http::header::RETRY_AFTER, HeaderValue::from_static("1"));
    response
}

/// What a request whose job the image model's stop ended is told.
const MODEL_STOPPED_MESSAGE: &str =
    "the image model was stopped (a model switch or a reload) before the job finished; retry once GET /v1/models reports `serving`";

/// The image model was stopped (a switch, a reload) before the job could
/// finish: `503 model_stopped`, with `Retry-After`.
fn model_stopped() -> Response {
    unavailable("server_error", "model_stopped", MODEL_STOPPED_MESSAGE)
}

/// Why a job that ended before its last image did, as its request is told:
/// a status and a body for a whole answer, the body alone for a stream that
/// already answered `200`.
struct Refusal {
    status: StatusCode,
    error: ApiError,
}

impl Refusal {
    /// A job that ended as `end` — `None` when its thread went away without
    /// saying why, a panic: `503 model_stopped` for a stopped model, `504
    /// request_timeout` past its deadline (ADR 0058: an image cut short is no
    /// image, so never a partial answer), `400 condition_budget_exceeded` for
    /// a prefix the runner found past the budget (the fit trusts the runner's
    /// count, so a drift between the two is still the request's size, not an
    /// engine fault), `400 image_refused` in the runner's words for a request
    /// it refuses (spec image-model/04), `500 engine_error` otherwise.
    fn of(end: Option<JobEnd>, route: Route) -> Self {
        match end {
            Some(JobEnd::Stopped) => Self {
                status: StatusCode::SERVICE_UNAVAILABLE,
                error: ApiError::new("server_error", "model_stopped", MODEL_STOPPED_MESSAGE),
            },
            Some(JobEnd::TimedOut(timeout)) => Self { status: StatusCode::GATEWAY_TIMEOUT, error: timeout.error() },
            Some(JobEnd::Failed(over @ ImageError::OverBudget { .. })) => Self {
                status: StatusCode::BAD_REQUEST,
                error: ApiError::naming("invalid_request_error", "condition_budget_exceeded", &over.to_string(), Some(route.condition_param())),
            },
            Some(JobEnd::Failed(ImageError::Refused(why))) => Self {
                status: StatusCode::BAD_REQUEST,
                error: ApiError::new("invalid_request_error", "image_refused", &why),
            },
            Some(JobEnd::Failed(_) | JobEnd::Cancelled) | None => Self {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                error: ApiError::new("engine_error", "engine_error", ENGINE_ERROR_MESSAGE),
            },
        }
    }

    fn into_response(self) -> Response {
        let retry = self.status == StatusCode::SERVICE_UNAVAILABLE;
        let mut response = (self.status, Json(self.error)).into_response();
        if retry {
            response.headers_mut().insert(axum::http::header::RETRY_AFTER, HeaderValue::from_static("1"));
        }
        response
    }
}

/// What a request carries past its checks into the job.
struct Prepared {
    model: ImageModel,
    job: ImageJob,
    fitted: Fitted,
}

#[utoipa::path(
    post,
    path = "/v1/images/generations",
    tag = "images",
    operation_id = "create_image",
    summary = "Images from a prompt",
    description = "OpenAI's image generation on the image model (`ignis-image-2.1-turbo`, ADR 0050): `n` RGBA PNGs from `prompt`, as `b64_json`.

`size` is one of `image.sizes` (by default the checkpoint's presets 2048x2048, 2400x1792, 1792x2400, 2528x1696, 1696x2528, 2752x1536, 1536x2752, and 1024x1024) or `auto`, 2048x2048. `response_format` is `b64_json` only (`url` is a 400), `output_format` `png` only. `background: opaque` composites the image on white with alpha 255; `auto` and `transparent` keep the alpha as generated.

ignis adds `seed` (image `i` starts from `seed + i`; absent, one is drawn and returned in `seed`) and `prompt_enhancement`. OpenAI's other fields are listed at the end.

One job runs at a time and up to `image.queue_depth` wait, in order; past that the answer is `503 engine_full` with `Retry-After`. A client that disconnects cancels its job, while it waits or between two denoising steps. A job is held to a deadline from its start, never from its queue: `--server-request-timeout`, or `--server-stream-timeout` when it streams (ADR 0058, 0 = none). Past it the job ends between two denoising steps and the request is `504 request_timeout`, or, streamed, its `error` event after the images already completed; always an error, whatever `--server-timeout-reply` says.

`stream: false` answers one JSON body. `stream: true` answers `text/event-stream`, each event an `event:` named by its `type` and a `data:` object: `image_generation.progress` (`{phase, image, step, total}`) as each unit of work starts -- `switching`, `loading` and `enhancing` first when the request waits on a switch, a load or a rewrite, then `encoding` once, then per image one `denoising` per step and its `decoding` -- and one `image_generation.completed` per image with its `b64_json`, as soon as it is made. A job that ends early sends one `error` event (`{type, error}`) instead of the rest; so does a refusal that comes once a stream waiting on a switch, a load or a rewrite has begun.

`model` absent, or the loaded image model's id, is served as it is; another known image model is switched to first when `switch.allow_implicit` is on. With a text model loaded and no `model` naming the image model, the answer is `409 model_kind_mismatch`.

Prompt enhancement (`prompt_enhancement: true`, or `image.prompt_enhancement` when absent): the text model `image.prompt_enhancer` rewrites the prompt first with Qwen's own system prompts (a prompt holding a CJK ideograph in Chinese, any other in English; reasoning off; seeded with `seed`), and the images are made from the rewrite, returned as each image's `revised_prompt`. With the enhancer loaded it rewrites, and the request then switches to the image model it names. Otherwise the request visits the enhancer: a switch to it and one on to the image model, during which every other request is answered `503 model_switching`. From the image model it first takes its turn in the queue, so the jobs admitted before it finish there; the ones admitted behind it run in the visit's drain, as in any switch's. A visit back to the model it left does not change `GET /v1/models`' `last_switch`; one from another text model is the request's switch to the image model, and each of its two switches is written there as a switch is. A rewrite that cannot be had -- an enhancer the server cannot switch to, a switch to it that does not land, a rewrite that fails -- leaves the prompt as sent, and `warnings` says why.",
    request_body = GenerationRequest,
    responses(
        (status = 200, description = "The images. `application/json` when `stream` is false or absent; `text/event-stream` when it is true.", content(
            (ImagesResponse = "application/json"),
            (ImageStreamEvent = "text/event-stream"),
        )),
        (status = 400, description = "A field outside its rule: no `prompt`, `n` outside 1..`image.max_n`, a `size` not in `image.sizes`, or past the largest image the loaded model makes (the message lists the sizes), `response_format: url`, an `output_format` other than `png`, an unknown `background`, `partial_images` above 0; or a prompt longer than the condition budget (`condition_budget_exceeded`); or a request the image model refuses, in its words (`image_refused`: a prompt whose positions run past the model's RoPE table).", body = ApiError),
        (status = 401, description = "The server was started with `--server-api-key` and the request carried no matching bearer token.", body = ApiError),
        (status = 404, description = "The request named a model this server neither loads nor may switch to.", body = ApiError),
        (status = 409, description = "The model that would serve the request is a text model (`model_kind_mismatch`, naming both kinds and the loaded id). Or `model` names a catalog entry this card cannot load, and no entry that answers to it can: by its class (`model_above_card_class`, naming the entry's class and the card's, ADR 0064) or by its GPU generation, alone or with its class (`model_above_gpu_generation`, naming the generation the entry needs and the card's name, compute capability and generation, ADR 0079): refused before anything loads or switches, the loaded model serving on.", body = ApiError),
        (status = 500, description = "The image model failed a call (`engine_error`).", body = ApiError),
        (status = 503, description = "The queue is full (`engine_full`, with `Retry-After`); a model switch is under way (`model_switching`, with `Retry-After`) or did not land (`model_switch_failed`, also when a prompt enhancement visit could not switch back to the image model); the image model was stopped before the job finished (`model_stopped`, with `Retry-After`); or the model was unloaded while idle and loading it again failed (`model_load_failed`); or the server began to stop while the request waited for a switch or a load (`server_stopping`, with `Retry-After`).", body = ApiError),
        (status = 504, description = "The job ran past `--server-request-timeout` from its start (`request_timeout`), whatever `--server-timeout-reply` says: an image cut short is no image.", body = ApiError),
    ),
)]
pub(crate) async fn create_image(State(server): State<Arc<Server>>, Json(fields): Json<GenerationRequest>) -> Response {
    let request = match checked(fields, &server.live().image) {
        Ok(request) => request,
        Err(refused) => return refused,
    };
    serve(server, request, Conditions::default(), Route::Generations).await
}

#[utoipa::path(
    post,
    path = "/v1/images/edits",
    tag = "images",
    operation_id = "create_image_edit",
    summary = "Images from a prompt and condition images",
    description = "OpenAI's image edit on the image model: `n` RGBA PNGs from `prompt` and 1 to `image.max_condition_images` condition images (`image[]`, or `image` for one; PNG, JPEG or WebP, with or without alpha), as a `multipart/form-data` form whose other parts are the generation's fields as text. `mask` is not supported (400).

Each condition image is resized to 1024x1024 pixels of area at its own aspect ratio, each side a multiple of 32 (diffusers' `calculate_dimensions`). When the prompt and the images exceed the condition budget -- the load's, lowered by `image.condition_budget_tokens`; a condition image costs one token per 16x16 pixels, 4,096 at 1024x1024 -- every image is scaled down by the same factor until they fit, and `condition_sizes` reports the sizes encoded; below `image.condition_floor_px` squared of area the request is a 400 naming the budget.

`size: auto` is the size in `image.sizes` whose aspect ratio is nearest the last condition image's. Streaming, the events are `image_edit.progress` and `image_edit.completed`, otherwise as for a generation. Prompt enhancement rewrites the instruction with Qwen's edit prompt, showing the enhancer the condition images when its vision tower is loaded; without one it rewrites from the text alone, and `warnings` says so. Every other field, the queue and the switch are as for `POST /v1/images/generations`.",
    request_body(content = EditForm, content_type = "multipart/form-data"),
    responses(
        (status = 200, description = "The images. `application/json` when `stream` is false or absent; `text/event-stream` when it is true.", content(
            (ImagesResponse = "application/json"),
            (ImageStreamEvent = "text/event-stream"),
        )),
        (status = 400, description = "As for a generation; or no condition image, more than `image.max_condition_images`, one that is not a PNG, JPEG or WebP, a `mask`, or condition images that do not fit the condition budget, or the image model's room for them, even at `image.condition_floor_px` (`condition_budget_exceeded`); or a condition image the image model refuses, in its words (`image_refused`: one its encoder's processor would not keep at its condition size, a panorama past 200:1).", body = ApiError),
        (status = 401, description = "The server was started with `--server-api-key` and the request carried no matching bearer token.", body = ApiError),
        (status = 404, description = "The request named a model this server neither loads nor may switch to.", body = ApiError),
        (status = 409, description = "The model that would serve the request is a text model (`model_kind_mismatch`). Or `model` names a catalog entry this card cannot load, and no entry that answers to it can: by its class (`model_above_card_class`, naming the entry's class and the card's, ADR 0064) or by its GPU generation, alone or with its class (`model_above_gpu_generation`, naming the generation the entry needs and the card's name, compute capability and generation, ADR 0079): refused before anything loads or switches, the loaded model serving on.", body = ApiError),
        (status = 500, description = "The image model failed a call (`engine_error`).", body = ApiError),
        (status = 503, description = "As for a generation: `engine_full`, `model_switching`, `model_switch_failed`, `model_stopped`, `model_load_failed`, `server_stopping`.", body = ApiError),
        (status = 504, description = "As for a generation: the job ran past its deadline (`request_timeout`).", body = ApiError),
    ),
)]
pub(crate) async fn create_image_edit(State(server): State<Arc<Server>>, multipart: Multipart) -> Response {
    let form = match read_form(multipart).await {
        Ok(form) => form,
        Err(refused) => return refused,
    };
    let limits = server.live().image.clone();
    if form.mask.is_some() {
        return bad_request_param("mask is not supported: an edit is conditioned on whole images", "mask");
    }
    let request = match checked(form.fields, &limits) {
        Ok(request) => request,
        Err(refused) => return refused,
    };
    let count = form.images.len();
    if count == 0 || count > limits.max_condition_images as usize {
        return bad_request_param(
            &format!(
                "an edit takes 1 to {} condition images (image.max_condition_images), got {count}",
                limits.max_condition_images
            ),
            "image",
        );
    }
    let max_pixels = u64::from(limits.max_condition_mipx) << 20;
    let decoding = server.image_decodes.enter(limits.max_condition_decodes).await;
    let decoded = tokio::task::spawn_blocking(move || {
        // Held by the decode itself, so a request that goes away mid-decode
        // does not let another begin beside it.
        let _decoding = decoding;
        decode_conditions(&form.images, max_pixels)
    })
    .await
    .expect("decoding condition images does not panic");
    match decoded {
        Ok(conditions) => serve(server, request, conditions, Route::Edits).await,
        Err(why) => bad_request_param(&why, "image"),
    }
}

/// An edit's condition images, decoded: each resized as soon as it is to the
/// size it is encoded at unless the condition budget scales it (diffusers'
/// size at 1024² of area, [`budget::condition_size`]), so that no full frame
/// outlives its own decode — not through a load, a switch or a visit; and
/// each one's size as sent, which the budget and `auto` read.
#[derive(Debug, Default)]
struct Conditions {
    /// Each image's size as sent, in order.
    sources: Vec<ImageSize>,
    /// Each image at its unscaled encoded size, in order.
    images: Vec<RgbaImage>,
}

/// Decode an edit's condition images in order ([`Conditions`]), each at most
/// `max_pixels`; `Err` names the first that is not one. Blocking.
fn decode_conditions(images: &[Vec<u8>], max_pixels: u64) -> Result<Conditions, String> {
    let area = f64::from(budget::CONDITION_SIDE) * f64::from(budget::CONDITION_SIDE);
    let mut conditions = Conditions::default();
    for (i, bytes) in images.iter().enumerate() {
        let image = crate::media::decode_condition_image(bytes, max_pixels).map_err(|why| format!("condition image {i}: {why}"))?;
        let ratio = f64::from(image.size.width) / f64::from(image.size.height);
        conditions.images.push(crate::media::resize_rgba(&image, budget::condition_size(area, ratio)));
        conditions.sources.push(image.size);
    }
    Ok(conditions)
}

/// An edit's form: the image parts in order, the mask's presence, and the
/// text parts as the generation's fields — one the routes do not read as
/// the JSON value it spells (a string when it spells none), for
/// `crate::openai_fields` to classify as a JSON body's would be.
async fn read_form(mut multipart: Multipart) -> Result<EditForm, Response> {
    let malformed = |error: axum::extract::multipart::MultipartError| {
        bad_request_param(&format!("the multipart form could not be read: {}", error.body_text()), "image")
    };
    let mut form = EditForm::default();
    while let Some(field) = multipart.next_field().await.map_err(malformed)? {
        let name = field.name().unwrap_or_default().to_owned();
        match name.as_str() {
            "image" | "image[]" => form.images.push(field.bytes().await.map_err(malformed)?.to_vec()),
            "mask" => form.mask = Some(field.bytes().await.map_err(malformed)?.to_vec()),
            _ => {
                let text = field.text().await.map_err(malformed)?;
                set_field(&mut form.fields, &name, text)?;
            }
        }
    }
    Ok(form)
}

/// One text part of an edit's form into `fields`, parsed as the JSON body
/// would be.
fn set_field(fields: &mut GenerationRequest, name: &str, text: String) -> Result<(), Response> {
    fn parse<T: std::str::FromStr>(name: &str, text: &str, what: &str) -> Result<T, Response> {
        text.trim().parse().map_err(|_| bad_request_param(&format!("{name} must be {what}, got `{text}`"), name))
    }
    match name {
        "model" => fields.model = Some(text),
        "prompt" => fields.prompt = Some(text),
        "n" => fields.n = Some(parse(name, &text, "an integer")?),
        "size" => fields.size = Some(text),
        "response_format" => fields.response_format = Some(text),
        "output_format" => fields.output_format = Some(text),
        "background" => fields.background = Some(text),
        "seed" => fields.seed = Some(parse(name, &text, "an unsigned integer")?),
        "prompt_enhancement" => fields.prompt_enhancement = Some(parse(name, &text, "true or false")?),
        "stream" => fields.stream = parse(name, &text, "true or false")?,
        _ => {
            let value = serde_json::from_str(&text).unwrap_or(serde_json::Value::String(text));
            fields.other.insert(name.to_owned(), value);
        }
    }
    Ok(())
}

/// What a request's channel carries, in order: the progress of what comes
/// before its job (a load, a switch), its answer once it is admitted, then
/// the job's own events — or, once a stream has answered `200`, why it was
/// refused.
#[derive(Debug)]
enum Streamed {
    Admitted(Answer),
    Job(JobEvent),
    Refused(ApiError),
}

impl From<JobEvent> for Streamed {
    fn from(event: JobEvent) -> Self {
        Self::Job(event)
    }
}

/// A progress event of what comes before the job: step `step` of `total`.
fn progress(phase: Phase, step: u32, total: u32) -> Streamed {
    Streamed::Job(JobEvent::Progress(Progress { phase, image: 0, step, total }))
}

/// What tells a request's channel the steps of the switches it waits on:
/// `switching` as leg `k` of `legs` begins, `loading` as its load does.
fn switch_steps(events: mpsc::UnboundedSender<Streamed>, legs: u32) -> Steps {
    let leg = AtomicU32::new(0);
    Arc::new(move |step| {
        let event = match step {
            SwitchStep::Switching => progress(Phase::Switching, leg.fetch_add(1, Ordering::SeqCst) + 1, legs),
            SwitchStep::Loading => progress(Phase::Loading, leg.load(Ordering::SeqCst).max(1), legs),
        };
        let _ = events.send(event);
    })
}

/// Serve a checked request on the image model: refused first, before
/// anything loads or switches, when a text model would serve it
/// ([`check_kind`]) or it names a model the server can neither serve nor
/// switch to ([`unswitchable_named`]); then [`admit`]ted, and answered whole
/// or streamed. A streamed request that waits on a load or a switch starts
/// its stream at once, so that it hears those phases as they happen; what
/// refuses it past that point is its `error` event — and so does one whose
/// prompt is rewritten first. One the loaded image model takes as it is
/// starts streaming once admitted, so every refusal before its job is still
/// a status.
async fn serve(server: Arc<Server>, mut request: ImageRequest, conditions: Conditions, route: Route) -> Response {
    // The id this card serves the request under (ADR 0064): an alias
    // rewritten, an entry above the card's class refused before anything.
    request.model = match crate::api::model_on_this_card(&server, request.model.take()) {
        Ok(model) => model,
        Err(refusal) => return crate::api::above_card(&refusal),
    };
    let named = split_model_lane(request.model.clone()).0;
    if let Err(refused) = check_kind(&server, ModelKind::Image, named.as_deref()) {
        return refused;
    }
    if let Some(named) = unswitchable_named(&server, ModelKind::Image, named.as_deref()) {
        return unknown_model(&server, named);
    }
    let (events, receiver) = mpsc::unbounded_channel();
    let stream = request.stream;
    if stream && (request.enhance || waits_before_the_job(&server, named.as_deref())) {
        tokio::spawn(async move {
            let refused = tokio::select! {
                admitted = admit(server, request, conditions, route, &events) => admitted.err(),
                // The client went away: nobody is left to admit it for.
                () = events.closed() => None,
            };
            if let Some(refused) = refused {
                let _ = events.send(Streamed::Refused(error_of(refused).await));
            }
        });
        return Sse::new(ImageSse::new(receiver, route)).into_response();
    }
    let admitted = admit(server, request, conditions, route, &events).await;
    // The job holds a sender of its own: the channel closes once it ends.
    drop(events);
    if let Err(refused) = admitted {
        return refused;
    }
    if stream {
        return Sse::new(ImageSse::new(receiver, route)).into_response();
    }
    collect(receiver, route).await
}

/// Whether a request naming `named` waits on something before its job: a
/// load (no model is serving) or a switch (the loaded model is not the image
/// model it is served by).
fn waits_before_the_job(server: &Server, named: Option<&str>) -> bool {
    let loaded = server.active();
    !matches!(*server.status(), ModelStatus::Serving)
        || loaded.kind() != ModelKind::Image
        || named.is_some_and(|named| !named.is_empty() && named != loaded.model_id())
}

/// The body of `refused`, a JSON error answer: what a stream that already
/// answered `200` sends in its `error` event instead.
async fn error_of(refused: Response) -> ApiError {
    let bytes = axum::body::to_bytes(refused.into_body(), usize::MAX).await.unwrap_or_default();
    serde_json::from_slice(&bytes).unwrap_or_else(|_| ApiError::new("engine_error", "engine_error", ENGINE_ERROR_MESSAGE))
}

/// A request past its first checks, to its job: the load of a model the idle
/// clock unloaded — the enhancer itself when it is the one unloaded and the
/// prompt is to be rewritten — prompt enhancement ([`enhance`]) and the implicit switch
/// to the image model it names, each phase sent to `events` as it starts
/// (ADR 0049's rules); then, on the
/// image model, pinned for the rest of the request, the size against what
/// its load makes (`auto` resolved among those), the queue's admission, the
/// condition budget and the resize, and the job — its answer sent to
/// `events` before any event of its own. `Err` is the refusal, sent instead.
async fn admit(
    server: Arc<Server>,
    request: ImageRequest,
    conditions: Conditions,
    route: Route,
    events: &mpsc::UnboundedSender<Streamed>,
) -> Result<(), Response> {
    let named = split_model_lane(request.model.clone()).0;
    let began = std::time::Instant::now();
    // What `ensure_loaded` waits on: the load of an unloaded model, or the
    // one another request began.
    let status = server.status();
    if status.is_unloaded() || matches!(*status, ModelStatus::Loading) {
        let _ = events.send(progress(Phase::Loading, 1, 1));
    }
    // A request to be rewritten on an unloaded text model loads the enhancer
    // rather than the image model it names: the rewrite on it, then one
    // switch — two loads, not three. The unloaded model itself when it is
    // the enhancer; else, naming an image model, the enhancer the server can
    // switch to.
    let unloaded = server.active();
    let enhancer = super::enhance::enhancer_on_this_card(&server, &server.live().image.prompt_enhancer);
    let rewrites_first = request.enhance && status.is_unloaded() && unloaded.kind() == ModelKind::Text;
    let load = match rewrites_first {
        true if unloaded.model_id() == enhancer => None,
        true if named.as_deref().is_some_and(|named| !named.is_empty())
            && server.switcher.as_ref().is_some_and(|switcher| switcher.text_model(&enhancer).is_some()) =>
        {
            Some(enhancer.as_str())
        }
        _ => named.as_deref(),
    };
    let loading = std::time::Instant::now();
    crate::model_switch::ensure_loaded(&server, load).await.map_err(|refusal| load_refused(&server, refusal))?;
    let mut switch = loading.elapsed();
    // Drawn now, so that the rewrite is seeded with it too.
    let seed = request.seed.or(server.seedless_seed).unwrap_or_else(crate::api::fresh_seed);
    let enhancing = std::time::Instant::now();
    let enhanced = match request.enhance {
        true => enhance(&server, request.prompt.clone(), &conditions.images, named.as_deref(), seed, events).await?,
        false => Enhanced::sent(request.prompt.clone()),
    };
    let enhance_time = enhancing.elapsed();
    let switching = std::time::Instant::now();
    crate::model_switch::implicit_switch_telling(&server, named.as_deref(), Some(switch_steps(events.clone(), 1)))
        .await
        .map_err(implicit_switch_refused)?;
    switch += switching.elapsed();
    // One model for the rest of the request (spec model-switch/01), the
    // image model (spec image-model/05); the server itself for the job,
    // which unloads its model should it fail.
    let unpinned = Arc::clone(&server);
    let server = Arc::new(server.pinned());
    check_served_kind(&server, ModelKind::Image, named.as_deref())?;
    let model = server.active().image_model().expect("the kind check passed: the image model").clone();
    let limits = server.live().image.clone();
    // `image.sizes` less what this load cannot make: what `auto` picks from.
    let sizes = super::served_sizes(&limits.sizes, model.plan());
    let ratio = conditions.sources.last().map_or(1.0, |size| f64::from(size.width) / f64::from(size.height));
    let size = match request.size {
        Some(size) if !model.plan().takes(size) => {
            let served: Vec<String> = sizes.iter().map(ToString::to_string).collect();
            return Err(bad_request_param(
                &format!(
                    "size `{size}` is past the largest image {} makes ({} latent cells of 16x16 pixels): it must be auto or one of {} (image.sizes)",
                    model.model_id(),
                    model.plan().max_target_cells,
                    served.join(", ")
                ),
                "size",
            ));
        }
        Some(size) => size,
        None => match auto_size(&sizes, ratio) {
            Some(size) => size,
            None => return Err(bad_request_param("size auto has no size to pick: image.sizes names none this image model makes", "size")),
        },
    };
    if let Some(metrics) = &unpinned.metrics {
        model.jobs.report_to(metrics);
    }
    let waiting = admitted(&model, limits.queue_depth)?;
    // ADR 0058: counted from the job's turn, never from its queue.
    let timeout = crate::deadline::Timeout::of(&server.live(), request.stream);
    let context = JobContext {
        kind: match route {
            Route::Generations => "generation",
            Route::Edits => "edit",
        },
        condition_images: conditions.sources.len() as u32,
        enhancement: match (request.enhance, enhanced.revised.is_some()) {
            (false, _) => "off",
            (true, true) => "rewritten",
            (true, false) => "as_sent",
        },
        began,
        queued_at: std::time::Instant::now(),
        enhance: enhance_time,
        switch,
        timeout,
    };
    let ImageRequest { n, background, .. } = request;
    let Enhanced { prompt, revised, warnings } = enhanced;
    let prepared = {
        let model = model.clone();
        tokio::task::spawn_blocking(move || prepare(model, &limits, prompt, conditions, size, n, seed, background, route))
            .await
            .expect("fitting condition images does not panic")
    };
    let Prepared { model, job, fitted } = prepared?;
    let _ = events.send(Streamed::Admitted(Answer {
        n,
        created: crate::api::now(),
        seed,
        size: size.to_string(),
        background: background.as_str(),
        condition_sizes: fitted.scaled.then(|| fitted.sizes.iter().map(ToString::to_string).collect()),
        revised_prompt: revised,
        warnings,
    }));
    job::spawn(unpinned, model, job, waiting, events.clone(), context);
    Ok(())
}

/// A place at the back of `model`'s queue, at most `depth`
/// (`image.queue_depth`) waiting behind the job running; else the refusal,
/// `503 model_stopped` for a closed queue and `503 engine_full` for a full
/// one.
fn admitted(model: &ImageModel, depth: u32) -> Result<Waiting, Response> {
    model.jobs.admit(depth).ok_or_else(|| {
        if model.jobs.is_closed() {
            return model_stopped();
        }
        unavailable("engine_full", "engine_full", format!("the image queue is full: one job runs and {depth} wait (image.queue_depth); retry"))
    })
}

/// A turn on `model` for a request about to visit the enhancer from it
/// (review-322cf M1): admitted as a job is ([`admitted`]), then its turn once
/// every job admitted before it has run, so that the visit's drain never
/// cuts one of them. `503 model_stopped` when the queue closes first.
async fn queue_turn(model: &ImageModel, depth: u32) -> Result<Running, Response> {
    admitted(model, depth)?.turn().await.ok_or_else(model_stopped)
}

/// The prompt a job runs on, and what its answer says of it.
#[derive(Debug)]
struct Enhanced {
    prompt: String,
    /// The rewrite, when there was one: each image's `revised_prompt`.
    revised: Option<String>,
    warnings: Vec<String>,
}

impl Enhanced {
    /// `prompt` as sent, no rewrite asked for.
    fn sent(prompt: String) -> Self {
        Self { prompt, revised: None, warnings: Vec::new() }
    }

    /// `prompt` as sent, since `enhancer` did not rewrite it, for `why`.
    fn without(prompt: String, enhancer: &str, why: &str) -> Self {
        tracing::warn!(
            name: "ignis.image.enhancement_failed",
            enhancer,
            reason = why,
            "prompt enhancement did not rewrite the prompt; the job runs on the prompt as sent"
        );
        let warning = format!("prompt enhancement with {enhancer} did not rewrite the prompt ({why}); the prompt was used as sent");
        Self { prompt, revised: None, warnings: vec![warning] }
    }

    /// `prompt` as `enhancer` rewrote it, or as sent when it did not.
    fn of(prompt: String, enhancer: &str, rewritten: Result<Rewritten, String>) -> Self {
        match rewritten {
            Ok(Rewritten { prompt: revised, warnings }) => Self { prompt: revised.clone(), revised: Some(revised), warnings },
            Err(why) => Self::without(prompt, enhancer, &why),
        }
    }
}

/// Prompt enhancement (spec image-model/05 §Prompt enhancement), on
/// `image.prompt_enhancer`, before the switch to the image model:
///
/// - the enhancer serving: the rewrite on it, then the request's switch to
///   the image model — no switch of its own;
/// - otherwise a visit to it ([`crate::model_switch::visit`]): from the image
///   model and back, `last_switch` left as it was, or from another text
///   model on to the image model the request names (`named`), each switch
///   written there — two switches, the gate closed throughout. From the
///   image model the request first
///   takes its turn in the queue ([`queue_turn`]): the jobs admitted before
///   it run to their end there, and the visit begins with none running; the
///   ones admitted behind it run in the visit's drain, as for any switch.
///
/// Each phase is sent to `events`. A rewrite that cannot be had — an
/// enhancer the server cannot switch to, a visit that does not get there,
/// a rewrite that fails — leaves the prompt as sent, with a warning; `Err`
/// is a visit that left no image model to serve the request.
async fn enhance(
    server: &Server,
    prompt: String,
    images: &[RgbaImage],
    named: Option<&str>,
    seed: u64,
    events: &mpsc::UnboundedSender<Streamed>,
) -> Result<Enhanced, Response> {
    let enhancer = super::enhance::enhancer_on_this_card(server, &server.live().image.prompt_enhancer);
    // The model the check reads is the one the rewrite runs on.
    let pinned = server.pinned();
    let loaded = pinned.active();
    if loaded.kind() == ModelKind::Text && loaded.model_id() == enhancer {
        let _ = events.send(progress(Phase::Enhancing, 1, 1));
        let rewritten = super::enhance::rewrite(&pinned, &prompt, images, seed).await;
        return Ok(Enhanced::of(prompt, &enhancer, rewritten));
    }
    let Some(switcher) = server.switcher.as_ref() else {
        return Ok(Enhanced::without(prompt, &enhancer, "this server cannot switch models"));
    };
    let Some(via) = switcher.text_model(&enhancer) else {
        return Ok(Enhanced::without(prompt, &enhancer, "image.prompt_enhancer names no text model this server can switch to"));
    };
    // Where the visit ends: the image model the request names, else the one
    // loaded.
    let named = named.filter(|named| !named.is_empty() && *named != loaded.model_id());
    let then = named
        .and_then(|named| switcher.known(named).map(|artifact| ModelSource { artifact, model: named.to_owned() }))
        .or_else(|| loaded.source.clone().filter(|_| loaded.kind() == ModelKind::Image));
    let Some(then) = then else {
        return Ok(Enhanced::without(prompt, &enhancer, "the image model has no artifact to load again after a visit"));
    };
    let to = then.model.clone();
    let turn = match loaded.image_model() {
        Some(model) => Some(queue_turn(model, server.live().image.queue_depth).await?),
        None => None,
    };
    let mut visit = match crate::model_switch::visit(server, via, then, Some(switch_steps(events.clone(), 2))) {
        Ok(visit) => visit,
        Err(SwitchRefusal::InProgress(running)) => return Err(implicit_switch_refused(ImplicitRefusal::Switching(running))),
        // GitHub #329: a request whose turn came after the stop — behind a
        // job the drain let finish — begins no visit, and is not served
        // without the rewrite it asked for either.
        Err(SwitchRefusal::Stopping) => return Err(implicit_switch_refused(ImplicitRefusal::Stopping { to })),
        Err(refused) => return Ok(Enhanced::without(prompt, &enhancer, &format!("a visit to it could not begin ({refused:?})"))),
    };
    // The gate is closed: what waits behind this request runs in the drain.
    drop(turn);
    // Once the serve stops, neither of the visit's loads is waited for
    // (GitHub #329): the visit runs on, detached, until the process ends.
    let Some(arrived) = switcher.unless_stopping(visit.arrived()).await else {
        return Err(implicit_switch_refused(ImplicitRefusal::Stopping { to }));
    };
    let rewritten = match arrived {
        Some(visited) => {
            let _ = events.send(progress(Phase::Enhancing, 1, 1));
            Some(super::enhance::rewrite(&visited, &prompt, images, seed).await)
        }
        None => None,
    };
    let Some(on) = switcher.unless_stopping(visit.on()).await else {
        return Err(implicit_switch_refused(ImplicitRefusal::Stopping { to }));
    };
    match (on, rewritten) {
        (SwitchOutcome::Switched, Some(rewritten)) => Ok(Enhanced::of(prompt, &enhancer, rewritten)),
        // The way there did not land, and the model it left serves again.
        (SwitchOutcome::Refused { reason } | SwitchOutcome::Restored { reason }, None) => {
            Ok(Enhanced::without(prompt, &enhancer, &format!("switching to it did not land: {reason}")))
        }
        (SwitchOutcome::Switched, None) => Ok(Enhanced::without(prompt, &enhancer, "the visit never handed it over")),
        // Nothing serves, or the way on did not land: no image model.
        (SwitchOutcome::Refused { reason } | SwitchOutcome::Restored { reason } | SwitchOutcome::Failed { reason }, _) => {
            Err(implicit_switch_refused(ImplicitRefusal::Failed { to, reason }))
        }
    }
}

/// The condition budget and the resize, then the job: blocking, since the
/// prefix's tokens are the runner's to count — it holds the image model's
/// tokenizer — and the resize is per pixel. The fit holds the runner for its
/// few host-side counts (a bisection's), behind the call a running job is in.
#[allow(clippy::too_many_arguments)]
fn prepare(
    model: ImageModel,
    limits: &ImageGroup,
    prompt: String,
    conditions: Conditions,
    size: ImageSize,
    n: u32,
    seed: u64,
    background: Background,
    route: Route,
) -> Result<Prepared, Response> {
    let budget = budget::budget(model.plan(), limits.condition_budget_tokens);
    let fitted = model.with_runner(|runner| {
        budget::fit(|sizes| runner.condition_tokens(&prompt, sizes), &conditions.sources, budget, limits.condition_floor_px)
    });
    let Some(fitted) = fitted else {
        return Err(model_stopped());
    };
    let fitted = fitted.map_err(|unfit| match unfit {
        budget::NotFitted::Refused(why) => error_response_naming(StatusCode::BAD_REQUEST, "invalid_request_error", "image_refused", why, None),
        over => error_response_naming(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "condition_budget_exceeded",
            over.to_string(),
            Some(route.condition_param()),
        ),
    })?;
    // Each image was decoded at its unscaled size: only one the budget scaled
    // is resized again, from that.
    let images = conditions
        .images
        .into_iter()
        .zip(&fitted.sizes)
        .map(|(image, &size)| if image.size == size { image } else { crate::media::resize_rgba(&image, size) })
        .collect();
    let job = ImageJob { condition: ImageCondition { prompt, images }, size, n, seed, background };
    Ok(Prepared { model, job, fitted })
}

/// What every answer to one admitted request says beside its images.
#[derive(Debug)]
struct Answer {
    /// The images the job makes.
    n: u32,
    created: u64,
    seed: u64,
    size: String,
    background: &'static str,
    condition_sizes: Option<Vec<String>>,
    revised_prompt: Option<String>,
    warnings: Vec<String>,
}

/// A request that did not stream: the job's `n` images, collected, or why
/// it ended without them.
async fn collect(mut events: mpsc::UnboundedReceiver<Streamed>, route: Route) -> Response {
    let mut answer = None;
    let mut data = Vec::new();
    while let Some(event) = events.recv().await {
        match event {
            Streamed::Admitted(admitted) => answer = Some(admitted),
            Streamed::Job(JobEvent::Progress(_)) => {}
            Streamed::Job(JobEvent::Image { png, .. }) => data.push(png),
            Streamed::Job(JobEvent::Ended(end)) => return Refusal::of(Some(end), route).into_response(),
            // A whole answer's refusals are statuses (`serve`): none comes here.
            Streamed::Refused(error) => return Refusal { status: StatusCode::INTERNAL_SERVER_ERROR, error }.into_response(),
        }
    }
    match answer {
        Some(answer) if data.len() == answer.n as usize => Json(ImagesResponse {
            created: answer.created,
            data: data.iter().map(|png| ImageData { b64_json: job::base64(png), revised_prompt: answer.revised_prompt.clone() }).collect(),
            seed: answer.seed,
            size: answer.size,
            background: answer.background,
            output_format: "png",
            condition_sizes: answer.condition_sizes,
            warnings: answer.warnings,
        })
        .into_response(),
        // The job's thread went away without saying why: it panicked.
        _ => Refusal::of(None, route).into_response(),
    }
}

/// A streamed request: its channel as Server-Sent Events. Dropped — the
/// client gone — it drops the channel, which cancels the job.
struct ImageSse {
    events: mpsc::UnboundedReceiver<Streamed>,
    route: Route,
    /// The request's answer, once it is admitted.
    answer: Option<Answer>,
    /// The images sent so far.
    made: u32,
    /// The last event was sent.
    ended: bool,
}

impl ImageSse {
    fn new(events: mpsc::UnboundedReceiver<Streamed>, route: Route) -> Self {
        Self { events, route, answer: None, made: 0, ended: false }
    }

    fn event(event: &ImageStreamEvent) -> Event {
        let kind = match event {
            ImageStreamEvent::Progress(progress) => progress.r#type,
            ImageStreamEvent::Completed(completed) => completed.r#type,
            ImageStreamEvent::Error(error) => error.r#type,
        };
        Event::default().event(kind).data(serde_json::to_string(event).expect("an image event serializes"))
    }
}

impl Stream for ImageSse {
    type Item = Result<Event, Infallible>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if this.ended {
                return Poll::Ready(None);
            }
            let event = match this.events.poll_recv(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(Streamed::Admitted(answer))) => {
                    this.answer = Some(answer);
                    continue;
                }
                Poll::Ready(Some(Streamed::Job(JobEvent::Progress(progress)))) => {
                    ImageStreamEvent::Progress(ProgressEvent { r#type: this.route.progress(), progress })
                }
                Poll::Ready(Some(Streamed::Job(JobEvent::Image { index, png }))) => {
                    let answer = this.answer.as_ref().expect("a job's images come after its request's answer");
                    this.made += 1;
                    ImageStreamEvent::Completed(CompletedEvent {
                        r#type: this.route.completed(),
                        image: index,
                        b64_json: job::base64(&png),
                        created_at: answer.created,
                        size: answer.size.clone(),
                        background: answer.background,
                        output_format: "png",
                        seed: answer.seed,
                        revised_prompt: answer.revised_prompt.clone(),
                        condition_sizes: answer.condition_sizes.clone(),
                        warnings: answer.warnings.clone(),
                    })
                }
                Poll::Ready(Some(Streamed::Refused(error))) => {
                    this.ended = true;
                    ImageStreamEvent::Error(ErrorEvent { r#type: "error", error })
                }
                // The job made its images: the stream ends.
                Poll::Ready(None) if this.answer.as_ref().is_some_and(|answer| answer.n == this.made) => return Poll::Ready(None),
                // What the request would have been answered had it not
                // streamed: the status is already 200. A channel that closes
                // before the last image without saying why is a job's thread
                // that panicked.
                Poll::Ready(end) => {
                    let end = match end {
                        Some(Streamed::Job(JobEvent::Ended(end))) => Some(end),
                        _ => None,
                    };
                    this.ended = true;
                    ImageStreamEvent::Error(ErrorEvent { r#type: "error", error: Refusal::of(end, this.route).error })
                }
            };
            return Poll::Ready(Some(Ok(Self::event(&event))));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DEFAULT_IMAGE_SIZES;

    fn limits() -> ImageGroup {
        ImageGroup::default()
    }

    fn request(fields: serde_json::Value) -> Result<ImageRequest, Response> {
        checked(serde_json::from_value(fields).unwrap(), &limits())
    }

    async fn refusal(result: Result<ImageRequest, Response>) -> (StatusCode, serde_json::Value) {
        let response = result.expect_err("refused");
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    /// AC 4: the field rules.
    #[tokio::test]
    async fn every_field_rule_refuses_with_a_400_naming_the_field() {
        let ok = request(serde_json::json!({ "prompt": "a fox" })).unwrap();
        assert_eq!((ok.n, ok.size, ok.background, ok.stream), (1, None, Background::Auto, false));
        let full = request(serde_json::json!({
            "prompt": "", "n": 10, "size": "1536x2752", "response_format": "b64_json", "output_format": "png",
            "background": "opaque", "seed": 9, "stream": true, "quality": "high", "user": "u"
        }))
        .unwrap();
        assert_eq!((full.n, full.size, full.background, full.seed), (10, Some(ImageSize::new(1536, 2752)), Background::Opaque, Some(9)));
        for (fields, param, words) in [
            (serde_json::json!({}), "prompt", "required"),
            (serde_json::json!({ "prompt": "x", "n": 0 }), "n", "between 1 and 10"),
            (serde_json::json!({ "prompt": "x", "n": 11 }), "n", "image.max_n"),
            (serde_json::json!({ "prompt": "x", "size": "512x512" }), "size", "2048x2048, 2400x1792"),
            (serde_json::json!({ "prompt": "x", "size": "big" }), "size", "1024x1024 (image.sizes)"),
            (serde_json::json!({ "prompt": "x", "response_format": "url" }), "response_format", "b64_json"),
            (serde_json::json!({ "prompt": "x", "response_format": "xml" }), "response_format", "xml"),
            (serde_json::json!({ "prompt": "x", "output_format": "webp" }), "output_format", "png only"),
            (serde_json::json!({ "prompt": "x", "background": "green" }), "background", "green"),
            (serde_json::json!({ "prompt": "x", "partial_images": 2 }), "partial_images", "comes whole"),
        ] {
            let (status, body) = refusal(request(fields.clone())).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{fields}");
            assert_eq!(body["error"]["param"], param, "{fields}: {body}");
            assert!(body["error"]["message"].as_str().unwrap().contains(words), "{fields}: {body}");
        }
    }

    /// AC 4: `auto` is 2048x2048 for a generation and, for an edit, the
    /// preset nearest the last condition image's aspect ratio.
    #[test]
    fn auto_is_the_square_or_the_preset_nearest_the_last_images_ratio() {
        let sizes = DEFAULT_IMAGE_SIZES;
        assert_eq!(auto_size(&sizes, 1.0), Some(ImageSize::new(2048, 2048)), "the larger square on the tie");
        assert_eq!(auto_size(&sizes, 16.0 / 9.0), Some(ImageSize::new(2752, 1536)));
        assert_eq!(auto_size(&sizes, 9.0 / 16.0), Some(ImageSize::new(1536, 2752)));
        assert_eq!(auto_size(&sizes, 4.0 / 3.0), Some(ImageSize::new(2400, 1792)));
        assert_eq!(auto_size(&sizes, 3.0 / 2.0), Some(ImageSize::new(2528, 1696)));
        assert_eq!(auto_size(&sizes, 1.0 / 1.4), Some(ImageSize::new(1792, 2400)));
        assert_eq!(auto_size(&[ImageSize::new(1024, 1024)], 3.0), Some(ImageSize::new(1024, 1024)));
        assert_eq!(auto_size(&[], 1.0), None);
    }

    /// Review-322cf L1: each condition image is resized as it is decoded, to
    /// the size it is encoded at unless the budget scales it, and keeps its
    /// size as sent; one that is not an image is named by its place.
    #[test]
    fn condition_images_are_resized_as_they_are_decoded() {
        let image = |width: u32, height: u32| RgbaImage { size: ImageSize::new(width, height), pixels: vec![120; (width * height * 4) as usize] };
        let pngs = [job::encode_png(&image(64, 32)), job::encode_png(&image(1024, 1024))];
        let decoded = decode_conditions(&pngs, 1 << 20).unwrap();
        assert_eq!(decoded.sources, [ImageSize::new(64, 32), ImageSize::new(1024, 1024)]);
        let sizes: Vec<ImageSize> = decoded.images.iter().map(|image| image.size).collect();
        assert_eq!(sizes, [ImageSize::new(1440, 736), ImageSize::new(1024, 1024)], "diffusers' size at 1024² of area");
        assert_eq!(decoded.images[1], image(1024, 1024), "an image at its size already is kept as it is");
        let refused = decode_conditions(&[pngs[0].clone(), b"not an image".to_vec()], 1 << 20).unwrap_err();
        assert!(refused.starts_with("condition image 1: "), "{refused}");
    }

    /// An edit's text parts parse as the JSON fields do, and a malformed
    /// number is a 400 naming its part.
    #[tokio::test]
    async fn an_edits_text_parts_parse_as_the_json_fields() {
        let mut fields = GenerationRequest::default();
        for (name, text) in [("prompt", "a cat"), ("n", "3"), ("seed", "42"), ("stream", "true"), ("prompt_enhancement", "false"), ("quality", "high")] {
            set_field(&mut fields, name, text.to_owned()).unwrap();
        }
        assert_eq!((fields.prompt.as_deref(), fields.n, fields.seed, fields.stream), (Some("a cat"), Some(3), Some(42), true));
        assert_eq!(fields.prompt_enhancement, Some(false));
        let refused = set_field(&mut fields, "n", "three".to_owned()).unwrap_err();
        let (status, body) = refusal(Err(refused)).await;
        assert_eq!((status, body["error"]["param"].as_str()), (StatusCode::BAD_REQUEST, Some("n")));
        // A part the routes do not read is the JSON value it spells, for the
        // field table: `quality` passes, `partial_images: 1` is refused.
        assert_eq!(fields.other["quality"], "high");
        set_field(&mut fields, "partial_images", "1".to_owned()).unwrap();
        assert_eq!(fields.other["partial_images"], 1);
        let (status, body) = refusal(checked(fields, &limits())).await;
        assert_eq!((status, body["error"]["param"].as_str()), (StatusCode::BAD_REQUEST, Some("partial_images")), "{body}");
    }
}
