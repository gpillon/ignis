//! `generate_image` (spec #656, ticket #661): the server tool whose call is
//! more than its route's. It is declared on `POST /v1/images/generations`
//! (`crate::images::routes::create_image`), and its call goes, in order:
//!
//! 1. [`image_model`]: the catalog's image model on disk, read through
//!    `GET /v1/models/catalog` with the caller's key. With none, the tool's
//!    error names the download that fetches it.
//! 2. [`made`]: the route, naming that model, with the agent's `prompt` and
//!    `size` and nothing else, so one image. Naming it is what switches the
//!    server to it (`switch.allow_implicit`); nothing switches back: the
//!    harness's next turn names its own model, and the implicit switch
//!    returns to it.
//! 3. [`shown`]: the image as a PNG of at most [`PREVIEW_EDGE`] px on its
//!    longest edge, first, then one line of text.
//!
//! Each refusal on the way is the route's own answer, through
//! [`super::call::answer`]. What the spec puts around these steps next
//! (ticket #663) goes here too: the vision check on the text model switched
//! away from, before step 1; the hold that runs the connector's image calls
//! one at a time, around step 2; and the route's `stream: true` phases as
//! progress notifications, in step 2.

use axum::body::{Body, Bytes};
use axum::http::{header, HeaderValue, Request, StatusCode};
use axum::response::Response;
use axum::Router;
use ignis_core::compute::ModelKind;
use ignis_core::image::ImageSize;
use rmcp::model::{CallToolResult, ContentBlock, JsonObject};
use serde_json::Value;

use super::call;
use super::tools::{ServerTool, Table};

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

/// Call `generate_image` with `arguments`, through `routes`, presenting
/// `authorization`, the harness's own header; `tools` is what the
/// connector lists.
pub(crate) async fn generate(
    routes: &Router,
    tools: &Table,
    tool: &ServerTool,
    arguments: Option<JsonObject>,
    authorization: Option<HeaderValue>,
) -> CallToolResult {
    let fields = match fields(arguments) {
        Ok(fields) => fields,
        Err(refused) => return call::answer(refused).await,
    };
    let model = match image_model(routes, authorization.clone()).await {
        Ok(Pick::OnDisk(model)) => model,
        Ok(Pick::Absent { model, unavailable }) => {
            return call::answer(absent(&model, unavailable.as_deref(), tools.named(DOWNLOAD_TOOL).is_some())).await;
        }
        Err(refused) => return refused,
    };
    let response = match made(routes, tool, fields, model, authorization).await {
        Ok(response) => response,
        Err(refused) => return refused,
    };
    if !response.status().is_success() {
        return call::answer(response).await;
    }
    shown(response).await
}

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

/// The route's answer to `fields` naming `model`: one image, or its refusal.
async fn made(
    routes: &Router,
    tool: &ServerTool,
    mut fields: JsonObject,
    model: String,
    authorization: Option<HeaderValue>,
) -> Result<Response, CallToolResult> {
    fields.insert("model".to_owned(), Value::String(model));
    match call::request(tool, Some(fields), authorization) {
        Ok(request) => Ok(call::send(routes, request).await),
        Err(refused) => Err(CallToolResult::error(vec![ContentBlock::text(refused)])),
    }
}

/// The route's image as the harness reads it: the preview first, then one
/// line of text. Decoding, resizing and encoding a picture of a few
/// megapixels is CPU work: off the async workers.
async fn shown(response: Response) -> CallToolResult {
    let body = match read(response).await {
        Ok(body) => body,
        Err(why) => return unreadable("the image", &why),
    };
    match tokio::task::spawn_blocking(move || preview(&body)).await {
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

/// The route's JSON answer as a [`Preview`]: its one image as it is when
/// within [`PREVIEW_EDGE`], else resized to fit it, aspect kept.
fn preview(answer: &[u8]) -> Result<Preview, String> {
    let answer: Value = serde_json::from_slice(answer).map_err(|err| err.to_string())?;
    let encoded = answer["data"][0]["b64_json"].as_str().ok_or("the answer holds no image")?;
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

    /// Spec #656: the image's line of `instructions` keeps within its share
    /// of the 1 KB, the line break before it included.
    #[test]
    fn the_image_instructions_keep_within_their_share() {
        assert!(INSTRUCTIONS.len() < 180, "{} bytes and the line break", INSTRUCTIONS.len());
        assert!(!INSTRUCTIONS.contains('\n'));
    }
}
