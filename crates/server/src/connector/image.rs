//! `generate_image` (spec #656, tickets #661 and #663): the server tool
//! whose call is more than its route's. It is declared on `POST
//! /v1/images/generations` (`crate::images::routes::create_image`), and its
//! call goes, in order:
//!
//! 1. [`ImageCalls`]: the turn. The connector's image calls run one at a
//!    time, so a call sent while another one's switch runs waits for it to
//!    end rather than meet its `503 model_switching`, and finds the image
//!    model loaded: several calls in one turn cost one switch.
//! 2. [`agent_sees`]: whether the agent's text model has vision, the one
//!    loaded or, with the image model loaded, the one the server switched
//!    away from. Judged once the call has its turn, so never by the enhancer
//!    a call before it visits (`image.prompt_enhancement`). Without, the call
//!    answers naming the skill's script ([`unseen`]), and nothing switches
//!    for an image nobody sees.
//! 3. [`image_model`]: the catalog's image model on disk, read through
//!    `GET /v1/models/catalog` with the caller's key. With none, the tool's
//!    error names the download that fetches it.
//! 4. [`made`]: the route, naming that model, with the agent's `prompt` and
//!    `size` and nothing else, so one image. Naming it is what switches the
//!    server to it (`switch.allow_implicit`); nothing switches back: the
//!    harness's next turn names its own model, and the implicit switch
//!    returns to it. A call carrying a progress token reads the route's
//!    events (`stream: true`), each phase a progress notification
//!    ([`streamed`]). The turn ends once the image is read off the route.
//! 5. [`shown`]: the image as a PNG of at most [`PREVIEW_EDGE`] px on its
//!    longest edge, first, then one line of text.
//!
//! Each refusal on the way is the route's own answer, through
//! [`super::call::answer`]; a full image queue among them. A call its
//! harness gives up, by closing the request as its timeout does, leaves the
//! line, or drops its route's answer, which cancels its job, and keeps the
//! turn until the switch it began, which runs on, has ended.

use std::sync::atomic::{AtomicUsize, Ordering};

use axum::body::{Body, Bytes};
use axum::http::{header, HeaderValue, Request, StatusCode};
use axum::response::Response;
use axum::Router;
use http_body_util::BodyExt;
use ignis_core::compute::ModelKind;
use ignis_core::image::ImageSize;
use rmcp::model::{CallToolResult, ContentBlock, JsonObject, ProgressNotificationParam, ProgressToken};
use rmcp::service::RequestContext;
use rmcp::RoleServer;
use serde_json::value::RawValue;
use serde_json::Value;

use super::call;
use super::tools::ServerTool;
use super::Shared;
use crate::Server;

/// The tool's name, as its route's extension declares it.
pub(crate) const TOOL: &str = "generate_image";

/// The tool that starts a catalog entry's download (ticket #660), listed
/// only with `connector.allow_changes` on.
const DOWNLOAD_TOOL: &str = "download_model";

/// The longest edge of the preview the agent sees, in pixels (spec #656):
/// enough to check the image is the one asked for, a few hundred vision
/// tokens of the request's budget (`vision.max_tokens`).
pub(crate) const PREVIEW_EDGE: u32 = 512;

/// What `instructions` say of the tool: an image's cost.
pub(crate) const INSTRUCTIONS: &str = "generate_image costs about 20 s of model switching, out and back on your next turn; other clients get 503 meanwhile, and you see the image only with vision on.";

/// The connector's `generate_image` calls (spec #656), one at a time: a
/// call holds the turn from its vision check until its image is read off
/// the route, so that a call sent meanwhile never meets the first one's
/// switch, which would answer it `503 model_switching`, and finds the image
/// model loaded once its turn comes. Not a retry: behind it, the image
/// route's own queue serves every client. One per server, shared by every
/// router it builds ([`crate::Server::image_calls`]).
#[derive(Debug, Default)]
pub struct ImageCalls {
    /// The turn, handed on in the order the calls asked for it.
    turn: tokio::sync::Mutex<()>,
    /// The calls waiting for their turn now.
    waiting: AtomicUsize,
}

impl ImageCalls {
    /// How many calls wait for their turn now.
    pub fn waiting(&self) -> usize {
        self.waiting.load(Ordering::SeqCst)
    }

    /// The turn: at once when no call holds it, else, `waits` run first,
    /// once every call that asked before has ended; `None` when `given_up`
    /// comes first, and the call leaves the line. No bound of its own: the
    /// call ahead ends by its route's own deadlines, and a harness's timeout
    /// gives a waiting call up.
    async fn turn(
        &self,
        waits: impl std::future::Future<Output = ()>,
        given_up: impl std::future::Future<Output = ()>,
    ) -> Option<tokio::sync::MutexGuard<'_, ()>> {
        if let Ok(turn) = self.turn.try_lock() {
            return Some(turn);
        }
        let _counted = Waiting::count(&self.waiting);
        let turn = async {
            waits.await;
            self.turn.lock().await
        };
        tokio::select! {
            turn = turn => Some(turn),
            () = given_up => None,
        }
    }
}

/// A call counted in [`ImageCalls::waiting`] until dropped.
struct Waiting<'a>(&'a AtomicUsize);

impl<'a> Waiting<'a> {
    fn count(waiting: &'a AtomicUsize) -> Self {
        waiting.fetch_add(1, Ordering::SeqCst);
        Self(waiting)
    }
}

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Call `generate_image` with `arguments`, through the connector's routes,
/// presenting `authorization`, the harness's own header; `context` is the
/// MCP request's.
pub(super) async fn generate(
    connector: &Shared,
    tool: &ServerTool,
    arguments: Option<JsonObject>,
    authorization: Option<HeaderValue>,
    context: &RequestContext<RoleServer>,
) -> CallToolResult {
    let (routes, tools) = (&connector.routes, &connector.tools);
    let fields = match fields(arguments) {
        Ok(fields) => fields,
        Err(refused) => return call::answer(refused).await,
    };
    let mut progress = Progress::of(context);
    let waits = progress.tell("waiting for the image call before this one".to_owned());
    let Some(turn) = connector.server.image_calls.turn(waits, context.ct.cancelled()).await else {
        return given_up();
    };
    // Judged once the call has its turn: the call before it has ended, and
    // with it a visit to the enhancer, which serves meanwhile.
    if !agent_sees(&connector.server) {
        return unseen();
    }
    let model = match image_model(routes, authorization.clone()).await {
        Ok(Pick::OnDisk(model)) => model,
        Ok(Pick::Absent { model, unavailable }) => {
            return call::answer(absent(&model, unavailable.as_deref(), tools.named(DOWNLOAD_TOOL).is_some())).await;
        }
        Err(refused) => return refused,
    };
    let made = tokio::select! {
        made = made(routes, tool, fields, model, authorization, &mut progress) => Some(made),
        () = context.ct.cancelled() => None,
    };
    let Some(made) = made else {
        // Dropped with the route's answer, the job is cancelled; a switch the
        // call began runs on without it. The turn is kept until that switch
        // ends, so the next call finds the image model loaded rather than
        // meet its `503 model_switching`.
        until_no_switch_runs(&connector.server).await;
        drop(turn);
        return given_up();
    };
    drop(turn);
    match made {
        Ok(encoded) => shown(encoded).await,
        Err(refused) => refused,
    }
}

/// Until nothing holds the server's switch slot: no switch, reload, load,
/// visit or unload runs. Each ends by its own deadline; the serve stopping
/// ends the wait too.
async fn until_no_switch_runs(server: &Server) {
    let Some(switcher) = &server.switcher else {
        return;
    };
    loop {
        // Taken before the read, so a slot freed in between is not missed:
        // freeing it wakes this.
        let mut changed = server.status.subscribe();
        if switcher.running().is_none() {
            return;
        }
        if !matches!(switcher.unless_stopping(changed.changed()).await, Some(Ok(()))) {
            return;
        }
    }
}

/// The result of a call its harness gave up (a timeout, a disconnect)
/// before its image was made: nobody reads it.
fn given_up() -> CallToolResult {
    CallToolResult::error(vec![ContentBlock::text("the call was given up before its image was made")])
}

/// Whether the agent's text model sees images (spec #656): the loaded
/// model's vision when it is a text model; with the image model loaded, the
/// vision of the text model the server switched away from, which the
/// harness's next turn returns to ([`crate::model_switch::Switcher::text_model_left`]).
/// A server that knows none, started on the image model, is taken to see:
/// nothing says the agent cannot, and no switch is spent on the image.
fn agent_sees(server: &Server) -> bool {
    let loaded = server.active();
    if loaded.kind() == ModelKind::Text {
        return loaded.media.is_some();
    }
    server.switcher.as_ref().and_then(|switcher| switcher.text_model_left()).is_none_or(|left| left.vision)
}

/// What a call answers when the agent's model has vision off: no image, no
/// switch, and the skill's script that makes one as a file, with the
/// skill's install. Not an error: the agent asked nothing wrong, and the
/// way to an image is the script.
fn unseen() -> CallToolResult {
    CallToolResult::success(vec![ContentBlock::text(format!("{UNSEEN}\n{}", super::INSTALL))])
}

/// [`unseen`]'s text, the skill's install after it.
const UNSEEN: &str = "No image was made: the model you run on has vision off, so it could not see one, and making it would switch models for nothing. The ignis skill's generate-image script makes the image as a file: in the skill's folder, ~/.agents/skills/ignis, run node scripts/generate-image.mjs \"<prompt>\" <out.png> [--size WxH]. Without the skill, install it first, with the line for your shell:";

/// The arguments the route is sent: `prompt` and `size`, as given. Any
/// other is refused, naming it: the route's `n`, `model` and `stream` are
/// the connector's, so that a call makes one image.
fn fields(arguments: Option<JsonObject>) -> Result<JsonObject, Response> {
    let mut fields = JsonObject::new();
    for (name, value) in arguments.unwrap_or_default() {
        if !matches!(name.as_str(), "prompt" | "size") {
            let message = format!("{TOOL} takes `prompt` and `size` only, not `{name}`");
            return Err(crate::api::bad_request_param(&message, &name));
        }
        fields.insert(name, value);
    }
    Ok(fields)
}

/// What the catalog has of an image model.
#[derive(Debug, PartialEq)]
enum Pick {
    /// The one to name: the loaded one, else the first on disk.
    OnDisk(String),
    /// None is on disk: the entry a download would fetch, one this card can
    /// load first, and why this card cannot load it when it cannot.
    Absent { model: String, unavailable: Option<String> },
}

/// The catalog's image model, read as the agent would read the catalog:
/// its refusal (a warm-up's `503`, the key's `401`) is the tool's error.
async fn image_model(routes: &Router, authorization: Option<HeaderValue>) -> Result<Pick, CallToolResult> {
    let mut request = Request::get("/v1/models/catalog");
    if let Some(authorization) = authorization {
        request = request.header(header::AUTHORIZATION, authorization);
    }
    let request = request.body(Body::empty()).expect("a GET of a fixed path");
    let response = call::send(routes, request).await;
    if !response.status().is_success() {
        return Err(call::answer(response).await);
    }
    let catalog = read(response).await.and_then(|body| serde_json::from_slice(&body).map_err(|err| err.to_string()));
    match catalog {
        Ok(catalog) => Ok(pick(&catalog)),
        Err(why) => Err(unreadable("the catalog", &why)),
    }
}

/// [`Pick`] from `GET /v1/models/catalog`'s body.
fn pick(catalog: &Value) -> Pick {
    let rows: Vec<&Value> = catalog["data"].as_array().map_or_else(Vec::new, |rows| rows.iter().filter(|row| row["kind"] == "image").collect());
    let id = |row: &Value| row["id"].as_str().unwrap_or_default().to_owned();
    let on_disk = rows.iter().find(|row| row["loaded"] == true).or_else(|| rows.iter().find(|row| row["on_disk"] == "complete"));
    if let Some(row) = on_disk {
        return Pick::OnDisk(id(row));
    }
    match rows.iter().find(|row| row["available"] != false).or_else(|| rows.first()) {
        Some(row) => Pick::Absent { model: id(row), unavailable: row["reason"].as_str().map(str::to_owned) },
        // A server built with no catalog lists none: the built-in one has it.
        None => Pick::Absent { model: built_in_image_model(), unavailable: None },
    }
}

/// The built-in catalog's image model, which its own tests pin.
fn built_in_image_model() -> String {
    let entries = crate::download::Catalog::built_in().entries();
    let image = entries.iter().find(|entry| entry.model_kind() == Some(ModelKind::Image));
    image.expect("the built-in catalog lists the image model").id.clone()
}

/// The tool's error when no image model is on disk: `404
/// model_not_found`, as the route answers a model it cannot switch to,
/// naming the download that fetches `model` (the command, run where the
/// server runs, and `download_model` when that tool is listed), or why this
/// card cannot load it.
fn absent(model: &str, unavailable: Option<&str>, download_listed: bool) -> Response {
    let message = match unavailable {
        Some(reason) => format!("no image model is on disk on this server, and this card cannot load the catalog's: {reason}"),
        None => {
            let mut message =
                format!("no image model is on disk on this server: `ignis-server model download {model}`, run where the server runs, fetches it");
            if download_listed {
                message.push_str(&format!(", and so does the tool {DOWNLOAD_TOOL} with `model` {model}"));
            }
            message
        }
    };
    crate::api::error_response(StatusCode::NOT_FOUND, "model_not_found", "model_not_found", message)
}

/// The image the route makes of `fields`, naming `model`: its PNG in
/// base64, read off the route's whole answer, or the tool's result when the
/// route refuses. A call that carries a progress token asks for the route's
/// events (`stream: true`) and tells `progress` each of its phases as it
/// starts ([`streamed`]).
async fn made(
    routes: &Router,
    tool: &ServerTool,
    mut fields: JsonObject,
    model: String,
    authorization: Option<HeaderValue>,
    progress: &mut Progress<'_>,
) -> Result<String, CallToolResult> {
    fields.insert("model".to_owned(), Value::String(model));
    if progress.heard() {
        fields.insert("stream".to_owned(), Value::Bool(true));
    }
    let request = call::request(tool, Some(fields), authorization)
        .map_err(|refused| CallToolResult::error(vec![ContentBlock::text(refused)]))?;
    let response = call::send(routes, request).await;
    if !response.status().is_success() {
        return Err(call::answer(response).await);
    }
    if progress.heard() {
        return streamed(response, progress).await;
    }
    let body = read(response).await.map_err(|why| unreadable("the image", &why))?;
    let answer: Value = serde_json::from_slice(&body).map_err(|err| unreadable("the image", &err.to_string()))?;
    match answer["data"][0]["b64_json"].as_str() {
        Some(encoded) => Ok(encoded.to_owned()),
        None => Err(unreadable("the image", "the answer holds no image")),
    }
}

/// The route's `stream: true` answer, read to its end: each progress event
/// told to `progress`, the image off its completed event; or the tool's
/// error from an `error` event, the server's error JSON as a client reads
/// it from a whole answer (a stream answered `200` before its refusal has
/// no `Retry-After` to pass on).
async fn streamed(response: Response, progress: &mut Progress<'_>) -> Result<String, CallToolResult> {
    let mut body = response.into_body();
    let mut unread = Vec::new();
    let mut image = None;
    loop {
        while let Some(end) = unread.windows(2).position(|pair| pair == b"\n\n") {
            let event: Vec<u8> = unread.drain(..end + 2).collect();
            let Some(data) = event_data(&event) else {
                continue;
            };
            let event: Value = serde_json::from_str(&data).map_err(|err| unreadable("the image's events", &err.to_string()))?;
            match event["type"].as_str().unwrap_or_default() {
                "error" => return Err(stream_refused(&data)),
                kind if kind.ends_with(".progress") => progress.tell(phase(&event)).await,
                kind if kind.ends_with(".completed") => image = event["b64_json"].as_str().map(str::to_owned),
                _ => {}
            }
        }
        match body.frame().await {
            None => break,
            Some(Ok(frame)) => {
                if let Ok(data) = frame.into_data() {
                    unread.extend_from_slice(&data);
                }
            }
            Some(Err(err)) => return Err(unreadable("the image's events", &err.to_string())),
        }
    }
    image.ok_or_else(|| unreadable("the image's events", "they ended with no image"))
}

/// The `data:` lines of one Server-Sent Event, joined; `None` for an event
/// that has none.
fn event_data(event: &[u8]) -> Option<String> {
    let event = String::from_utf8_lossy(event);
    let lines: Vec<&str> = event
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(|data| data.strip_prefix(' ').unwrap_or(data))
        .collect();
    (!lines.is_empty()).then(|| lines.join("\n"))
}

/// The tool's error for the route's `error` event `data`: its `error`
/// object, as the route writes it, in the body a whole answer would have.
fn stream_refused(data: &str) -> CallToolResult {
    #[derive(serde::Deserialize)]
    struct Refused<'a> {
        #[serde(borrow)]
        error: &'a RawValue,
    }
    let text = match serde_json::from_str::<Refused<'_>>(data) {
        Ok(refused) => format!("{{\"error\":{}}}", refused.error.get()),
        Err(_) => data.to_owned(),
    };
    CallToolResult::error(vec![ContentBlock::text(text)])
}

/// A route's progress event as a notification's message: the phase first
/// (`switching`, `loading`, `enhancing`, `encoding`, `denoising`,
/// `decoding`), then, for a phase of several steps, which one.
fn phase(event: &Value) -> String {
    let phase = event["phase"].as_str().unwrap_or("working");
    match (event["step"].as_u64(), event["total"].as_u64()) {
        (Some(step), Some(total)) if total > 1 => format!("{phase} {step} of {total}"),
        _ => phase.to_owned(),
    }
}

/// Where a call's progress goes (spec #656): MCP progress notifications on
/// the harness's progress token, when its call carries one, each one's
/// `progress` one past the last; nowhere otherwise.
struct Progress<'a> {
    context: &'a RequestContext<RoleServer>,
    token: Option<ProgressToken>,
    told: u32,
}

impl<'a> Progress<'a> {
    fn of(context: &'a RequestContext<RoleServer>) -> Self {
        Self { context, token: context.meta.get_progress_token(), told: 0 }
    }

    /// Whether the call carries a progress token.
    fn heard(&self) -> bool {
        self.token.is_some()
    }

    /// `message` as the next notification. One that cannot be sent is
    /// dropped: a harness gone gives the call up by its own token.
    async fn tell(&mut self, message: String) {
        let Some(token) = &self.token else {
            return;
        };
        self.told += 1;
        let note = ProgressNotificationParam::new(token.clone(), f64::from(self.told)).with_message(message);
        let _ = self.context.peer.notify_progress(note).await;
    }
}

/// The image, `encoded`, as the harness reads it: the preview first, then
/// one line of text. Decoding, resizing and encoding a picture of a few
/// megapixels is CPU work: off the async workers.
async fn shown(encoded: String) -> CallToolResult {
    match tokio::task::spawn_blocking(move || preview(&encoded)).await {
        Ok(Ok(preview)) => {
            let image = ContentBlock::image(crate::images::job::base64(&preview.png), "image/png");
            CallToolResult::success(vec![image, ContentBlock::text(preview.line())])
        }
        Ok(Err(why)) => unreadable("the image", &why),
        Err(failed) => unreadable("the image", &failed.to_string()),
    }
}

/// A preview of the image a call made.
struct Preview {
    /// The PNG.
    png: Vec<u8>,
    /// The image's size as made.
    made: ImageSize,
    /// The preview's.
    shown: ImageSize,
}

impl Preview {
    /// The line beside it: the size made, that this is a preview, and how
    /// to keep an image at full size.
    fn line(&self) -> String {
        format!(
            "Made one {} PNG; shown here as a {} preview, and not kept. The ignis skill's generate-image script makes one at full size and saves it to a file.",
            self.made, self.shown
        )
    }
}

/// The route's image, `encoded`, as a [`Preview`]: as it is when within
/// [`PREVIEW_EDGE`], else resized to fit it, aspect kept.
fn preview(encoded: &str) -> Result<Preview, String> {
    let png = crate::media::base64_bytes(encoded).ok_or("the answer's image is not base64")?;
    // The route's own PNG: no bound of a client's to hold it to.
    let image = crate::media::decode_condition_image(&png, u64::MAX)?;
    let shown = preview_size(image.size);
    if shown == image.size {
        return Ok(Preview { png, made: image.size, shown });
    }
    let resized = crate::media::resize_rgba(&image, shown);
    Ok(Preview { png: crate::images::job::encode_png(&resized), made: image.size, shown })
}

/// `size` scaled so its longest edge is at most [`PREVIEW_EDGE`], each side
/// rounded and at least 1 px; never enlarged.
fn preview_size(size: ImageSize) -> ImageSize {
    let longest = u64::from(size.width.max(size.height));
    if longest <= u64::from(PREVIEW_EDGE) {
        return size;
    }
    let scaled = |side: u32| ((u64::from(side) * u64::from(PREVIEW_EDGE) + longest / 2) / longest).max(1) as u32;
    ImageSize::new(scaled(size.width), scaled(size.height))
}

async fn read(response: Response) -> Result<Bytes, String> {
    axum::body::to_bytes(response.into_body(), usize::MAX).await.map_err(|err| err.to_string())
}

/// The tool's error when a route's successful answer cannot be read.
fn unreadable(what: &str, why: &str) -> CallToolResult {
    CallToolResult::error(vec![ContentBlock::text(format!("{what} the server answered could not be read: {why}"))])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn row(id: &str, on_disk: &str, loaded: bool) -> Value {
        json!({ "id": id, "kind": "image", "on_disk": on_disk, "loaded": loaded, "available": true })
    }

    /// The loaded image model wins, else the first on disk; a text model is
    /// never one; with none on disk, the entry a download would fetch, one
    /// this card can load first, else the first with the reason it cannot.
    #[test]
    fn the_pick_is_the_loaded_image_model_else_one_on_disk_else_the_download() {
        let text = json!({ "id": "text-on-disk", "kind": "text", "on_disk": "complete", "loaded": false });
        let catalog = |rows: Vec<Value>| json!({ "object": "list", "data": rows });
        let two = catalog(vec![text.clone(), row("img-a", "complete", false), row("img-b", "complete", true)]);
        assert_eq!(pick(&two), Pick::OnDisk("img-b".into()));
        let one = catalog(vec![text.clone(), row("img-a", "partial", false), row("img-b", "complete", false)]);
        assert_eq!(pick(&one), Pick::OnDisk("img-b".into()));
        let mut big = row("img-big", "absent", false);
        big["available"] = json!(false);
        big["reason"] = json!("img-big needs a 32 GB card or larger");
        let none = catalog(vec![text.clone(), big.clone(), row("img-small", "partial", false)]);
        assert_eq!(pick(&none), Pick::Absent { model: "img-small".into(), unavailable: None });
        let too_big = catalog(vec![big]);
        assert_eq!(pick(&too_big), Pick::Absent { model: "img-big".into(), unavailable: Some("img-big needs a 32 GB card or larger".into()) });
        // No image row at all: the built-in catalog's image model.
        assert_eq!(pick(&catalog(vec![text])), Pick::Absent { model: "ignis-image-2.1-turbo".into(), unavailable: None });
    }

    async fn error_of(refused: Response) -> Value {
        assert_eq!(refused.status(), StatusCode::NOT_FOUND);
        serde_json::from_slice(&read(refused).await.unwrap()).unwrap()
    }

    /// With no image model on disk the error names the command that fetches
    /// it and, only when it is listed, `download_model`; on a card that
    /// cannot load it, why instead.
    #[tokio::test]
    async fn the_absent_model_names_its_download_and_download_model_only_when_listed() {
        let unlisted = error_of(absent("img-a", None, false)).await;
        assert_eq!(unlisted["error"]["code"], "model_not_found");
        let message = unlisted["error"]["message"].as_str().unwrap();
        assert!(message.contains("`ignis-server model download img-a`"), "{message}");
        assert!(!message.contains(DOWNLOAD_TOOL), "{message}");
        let listed = error_of(absent("img-a", None, true)).await;
        let message = listed["error"]["message"].as_str().unwrap();
        assert!(message.contains("`ignis-server model download img-a`") && message.contains("download_model"), "{message}");
        let refused = error_of(absent("img-a", Some("img-a needs a 32 GB card or larger"), true)).await;
        let message = refused["error"]["message"].as_str().unwrap();
        assert!(message.ends_with("img-a needs a 32 GB card or larger") && !message.contains("model download"), "{message}");
    }

    /// The longest edge comes down to 512 px, aspect kept; an image within
    /// it is not enlarged.
    #[test]
    fn the_preview_fits_512_px_and_is_never_enlarged() {
        for ((width, height), expected) in [
            ((2048, 2048), (512, 512)),
            ((1024, 640), (512, 320)),
            ((1792, 2400), (382, 512)),
            ((2752, 1536), (512, 286)),
            ((512, 128), (512, 128)),
            ((300, 200), (300, 200)),
            ((4000, 2), (512, 1)),
        ] {
            let shown = preview_size(ImageSize::new(width, height));
            assert_eq!((shown.width, shown.height), expected, "{width}x{height}");
        }
    }

    /// Only `prompt` and `size` reach the route: anything else is refused,
    /// naming it.
    #[tokio::test]
    async fn an_argument_past_prompt_and_size_is_refused_naming_it() {
        let arguments = json!({ "prompt": "a fox", "size": "1024x1024" }).as_object().cloned();
        assert_eq!(Value::Object(fields(arguments).unwrap()), json!({ "prompt": "a fox", "size": "1024x1024" }));
        let refused = fields(json!({ "prompt": "a fox", "n": 3 }).as_object().cloned()).err().expect("refused");
        assert_eq!(refused.status(), StatusCode::BAD_REQUEST);
        let body: Value = serde_json::from_slice(&read(refused).await.unwrap()).unwrap();
        assert_eq!(body["error"]["param"], "n", "{body}");
    }

    /// A refusal that comes once the route's events have begun, an `error`
    /// event (a full queue after a switch), is the tool's error: the body a
    /// whole answer would have had, its keys in the route's order; and a
    /// progress event's message is its phase, then its step when it has
    /// several.
    #[test]
    fn an_error_event_is_the_tools_error_and_a_progress_event_names_its_phase() {
        let event = "event: error\ndata: {\"type\":\"error\",\"error\":{\"message\":\"the image queue is full\",\"type\":\"engine_full\",\"code\":\"engine_full\"}}\n\n";
        let result = serde_json::to_value(stream_refused(&event_data(event.as_bytes()).unwrap())).unwrap();
        assert_eq!(result["isError"], true, "{result}");
        assert_eq!(
            result["content"][0]["text"],
            r#"{"error":{"message":"the image queue is full","type":"engine_full","code":"engine_full"}}"#
        );
        assert_eq!(event_data(b": a comment\n\n"), None);
        let progress = |phase: &str, step: u32, total: u32| json!({ "phase": phase, "image": 0, "step": step, "total": total });
        assert_eq!(phase(&progress("switching", 1, 1)), "switching");
        assert_eq!(phase(&progress("switching", 2, 2)), "switching 2 of 2");
        assert_eq!(phase(&progress("denoising", 3, 8)), "denoising 3 of 8");
    }

    /// A call waits for the turn another holds, counted while it waits; given
    /// up meanwhile (its harness's timeout), it leaves the line without the
    /// turn; once the holder ends, the next call has it at once.
    #[tokio::test]
    async fn a_call_given_up_while_it_waits_for_its_turn_leaves_the_line() {
        let calls = ImageCalls::default();
        let first = calls.turn(async {}, std::future::pending()).await.expect("no call holds the turn");
        let (give_up, given_up) = tokio::sync::oneshot::channel::<()>();
        let waiter = calls.turn(async {}, async {
            let _ = given_up.await;
        });
        tokio::pin!(waiter);
        assert!(futures_util::poll!(&mut waiter).is_pending());
        assert_eq!(calls.waiting(), 1);
        give_up.send(()).unwrap();
        assert!(waiter.await.is_none(), "given up, it has no turn");
        assert_eq!(calls.waiting(), 0);
        drop(first);
        assert!(calls.turn(async {}, std::future::pending()).await.is_some());
    }

    /// Spec #656: the image's line of `instructions` keeps within its share
    /// of the 1 KB, the line break before it included.
    #[test]
    fn the_image_instructions_keep_within_their_share() {
        assert!(INSTRUCTIONS.len() < 180, "{} bytes and the line break", INSTRUCTIONS.len());
        assert!(!INSTRUCTIONS.contains('\n'));
    }
}
