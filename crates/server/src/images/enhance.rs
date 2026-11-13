//! Prompt enhancement's rewrite (spec image-model/05 §Prompt enhancement):
//! a text model rewrites an image request's prompt with Qwen's own system
//! prompts ([`super::prompts`]) before the image model reads it — a
//! generation's in English, or in Chinese when the prompt holds a CJK
//! ideograph, an edit's with its condition images when the text model can
//! see them. Reasoning is off. The switches that bring the text model in are
//! the routes' ([`super::routes`]); this is the rewrite on it.

use std::collections::BTreeMap;
use std::path::PathBuf;

use ignis_core::compute::ModelKind;
use ignis_core::image::RgbaImage;
use ignis_core::{FinishReason, RequestClass};

use super::prompts;
use crate::template::{ChatMessage, ContentPart, MessageContent};
use crate::thinking::ThinkingOptions;
use crate::Server;

/// What a rewrite is asked to do, which picks its system prompt and its
/// user turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    /// A generation's prompt, rewritten in English.
    GenerationEn,
    /// A generation's prompt holding a CJK ideograph, rewritten in Chinese.
    GenerationZh,
    /// An edit's instruction.
    Edit,
}

impl Task {
    /// The task for `prompt`, an edit's when it has condition images. A
    /// generation's language is upstream's `get_caption_language`: Chinese
    /// when the prompt holds a CJK Unified Ideograph (U+4E00 to U+9FFF),
    /// English otherwise.
    pub fn of(prompt: &str, edit: bool) -> Self {
        if edit {
            Self::Edit
        } else if prompt.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c)) {
            Self::GenerationZh
        } else {
            Self::GenerationEn
        }
    }

    /// Its system prompt.
    pub fn system_prompt(self) -> &'static str {
        match self {
            Self::GenerationEn => prompts::GENERATION_EN,
            Self::GenerationZh => prompts::GENERATION_ZH,
            Self::Edit => prompts::EDIT,
        }
    }

    /// The user turn upstream sends after its system prompt, the prompt
    /// trimmed as upstream trims a generation's.
    fn user_turn(self, prompt: &str) -> String {
        let prompt = prompt.trim();
        match self {
            Self::GenerationEn => format!("User Input: {prompt}\n\n Rewritten Prompt:"),
            Self::GenerationZh => format!("用户输入：{prompt}\n改写输出："),
            Self::Edit => format!("User Input: {prompt}\n\nRewritten Prompt:"),
        }
    }

    /// The rewrite's two turns: the system prompt, and the user turn — after
    /// `images` as `image_url` parts, the condition images an edit's
    /// enhancer can see, in order.
    pub fn messages(self, prompt: &str, images: &[RgbaImage]) -> Vec<ChatMessage> {
        let user_turn = self.user_turn(prompt);
        let user = if images.is_empty() {
            ChatMessage::text("user", user_turn)
        } else {
            let part = |kind: &str, text: Option<String>, url: Option<String>| ContentPart {
                kind: Some(kind.to_owned()),
                text,
                url,
                cache_control: None,
            };
            let mut parts: Vec<ContentPart> = images
                .iter()
                .map(|image| part("image_url", None, Some(format!("data:image/png;base64,{}", super::job::base64(&super::job::encode_png(image))))))
                .collect();
            parts.push(part("text", Some(user_turn), None));
            ChatMessage { content: MessageContent::Parts(parts), ..ChatMessage::text("user", "") }
        };
        vec![ChatMessage::text("system", self.system_prompt()), user]
    }

    /// The rewritten prompt in the model's answer `content`, as upstream
    /// reads it: an edit's from the `{"Rewritten": ...}` object it is asked
    /// for (fenced as JSON or not; an answer that is not one is taken as the
    /// prompt itself), each trimmed and its lines joined with spaces. `None`
    /// when nothing is left.
    pub fn read(self, content: &str) -> Option<String> {
        let mut text = content.trim().to_owned();
        if self == Self::Edit {
            let unfenced = text.replace("```json", "").replace("```", "");
            if let Some(rewritten) = serde_json::from_str::<serde_json::Value>(unfenced.trim())
                .ok()
                .and_then(|answer| answer.get("Rewritten").and_then(serde_json::Value::as_str).map(str::to_owned))
            {
                text = rewritten;
            }
        }
        let text = text.trim().replace('\n', " ");
        (!text.is_empty()).then_some(text)
    }
}

/// Why `enhancer` (`image.prompt_enhancer`) names no text model a rewrite
/// could run on, where the server knows its text models — it can switch:
/// neither the loaded text model nor one an explicit switch by id may name
/// ([`crate::model_switch::Switcher::text_model`]), `listing` (the known
/// models a config change is about to set) listed too. `None` when it names
/// one, or when the server cannot switch: a request then falls back, with a
/// warning, as it does for any enhancer it cannot reach. The enhancer is the
/// one this card serves ([`enhancer_on_this_card`]).
pub fn unknown_enhancer(server: &Server, enhancer: &str, listing: &BTreeMap<String, PathBuf>) -> Option<String> {
    let switcher = server.switcher.as_ref()?;
    let loaded = server.active();
    let served = enhancer_on_this_card(server, enhancer);
    if loaded.kind() == ModelKind::Text && loaded.model_id() == served {
        return None;
    }
    switcher.text_model_among(&served, listing).is_none().then(|| {
        format!(
            "image.prompt_enhancer `{enhancer}` names no text model this server can switch to: list it in switch.known_models, start on it, or download it from the catalog"
        )
    })
}

/// The text model `enhancer` (`image.prompt_enhancer`) names, as this card
/// serves it — through the same step a request's `model` takes
/// ([`crate::api::model_on_this_card`], ADR 0064 §2): on a card below the
/// 27B's default entry, the default enhancer `qwen3.8-27b` is the entry that
/// answers to it, so the rewrite runs on that one, loaded or visited, and is
/// never a visit to the model already loaded. As configured when no entry
/// answers in its place — then an enhancer above the card's class is one no
/// visit reaches ([`crate::model_switch::Switcher::text_model`]).
pub fn enhancer_on_this_card(server: &Server, enhancer: &str) -> String {
    crate::api::model_on_this_card(server, Some(enhancer.to_owned())).ok().flatten().unwrap_or_else(|| enhancer.to_owned())
}

/// A prompt the enhancer rewrote, and what the response says beside it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rewritten {
    pub prompt: String,
    pub warnings: Vec<String>,
}

/// Rewrite `prompt` — an edit's when `images` holds its condition images —
/// on the text model `server` is pinned to: a request of its own on that
/// engine, reasoning off, the model card's instruct sampling seeded with
/// `seed` (the same seed, the same rewrite, the same image), at most
/// `image.prompt_enhancer_max_tokens` tokens within `server.request_timeout`.
/// An edit's images go to an enhancer with a vision tower; one without
/// rewrites from the text alone, and says so. `Err` says why there is no
/// rewrite.
pub async fn rewrite(server: &Server, prompt: &str, images: &[RgbaImage], seed: u64) -> Result<Rewritten, String> {
    let loaded = server.active();
    let enhancer = loaded.model_id();
    let engine = loaded.engine().ok_or_else(|| format!("{enhancer} is not a text model"))?;
    let task = Task::of(prompt, !images.is_empty());
    let sees = !images.is_empty() && loaded.media.is_some();
    let mut warnings = Vec::new();
    if !images.is_empty() && !sees {
        warnings.push(format!(
            "prompt enhancement: {enhancer} has no vision tower loaded, so it rewrote the edit's prompt without seeing its condition images"
        ));
    }
    // An edit's images are PNG-encoded off the async workers.
    let messages = if sees {
        let (prompt, images) = (prompt.to_owned(), images.to_vec());
        tokio::task::spawn_blocking(move || task.messages(&prompt, &images))
            .await
            .map_err(|panicked| format!("encoding the condition images for {enhancer} panicked: {panicked}"))?
    } else {
        task.messages(prompt, &[])
    };
    let thinking = ThinkingOptions { enable_thinking: false, ..ThinkingOptions::default() };
    let limits = server.live();
    let params = crate::api::SamplingRequestFields::default().resolve(Some(limits.image.prompt_enhancer_max_tokens), false, false, Some(seed))?;
    let (input, _, _, media) = match crate::api::prepare_request(server, None, &messages, params, &thinking, &[], None).await {
        Ok(prepared) => prepared,
        Err(refused) => return Err(format!("{enhancer} refused the rewrite: {}", error_message(refused).await)),
    };
    let notes = crate::engine::RequestNotes { media, ..crate::engine::RequestNotes::default() };
    let (request, mut events) = engine
        .submit_with_notes(input, RequestClass::Interactive, notes)
        .await
        .map_err(|refused| format!("{enhancer} did not take the rewrite: {refused:?}"))?;
    // A rewrite whose request goes away mid-way is cancelled on the engine.
    let mut cancel = crate::api::CancelOnDrop::new(engine.clone(), request);
    let completion = crate::engine::collect_completion(&mut events, limits.request_timeout).await.map_err(|error| match error {
        crate::engine::CollectError::DeadlinePassed(_) => {
            cancel.timed_out();
            format!("{enhancer} did not finish the rewrite within server.request_timeout")
        }
        crate::engine::CollectError::EngineGaveUp => {
            cancel.completed();
            format!("{enhancer}'s engine dropped the rewrite")
        }
    })?;
    cancel.completed();
    if completion.reason == FinishReason::Error {
        return Err(format!("{enhancer}'s engine failed the rewrite"));
    }
    let (_, content) = crate::api::split_reasoning(loaded.template.as_ref(), &completion.tokens, &thinking);
    let prompt = task.read(&content).ok_or_else(|| format!("{enhancer} answered no prompt"))?;
    Ok(Rewritten { prompt, warnings })
}

/// The message of `refused`, an OpenAI error answer.
async fn error_message(refused: axum::response::Response) -> String {
    let status = refused.status();
    let bytes = axum::body::to_bytes(refused.into_body(), usize::MAX).await.unwrap_or_default();
    serde_json::from_slice::<serde_json::Value>(&bytes)
        .ok()
        .and_then(|body| body["error"]["message"].as_str().map(str::to_owned))
        .unwrap_or_else(|| status.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ignis_core::image::ImageSize;

    /// AC 9: a prompt holding a CJK ideograph picks the Chinese system
    /// prompt, any other the English one — upstream's range, U+4E00 to
    /// U+9FFF, and nothing past it (kana, hangul, the extensions) — and an
    /// edit the edit prompt, whatever its language.
    #[test]
    fn a_cjk_prompt_picks_the_chinese_system_prompt() {
        for (prompt, task) in [
            ("a red fox in snow", Task::GenerationEn),
            ("一只雪地里的红狐狸", Task::GenerationZh),
            ("a poster reading 福 in gold", Task::GenerationZh),
            ("\u{4e00}", Task::GenerationZh),
            ("\u{9fff}", Task::GenerationZh),
            ("\u{4dff}\u{a000}", Task::GenerationEn),
            ("ひらがな カタカナ 한글", Task::GenerationEn),
        ] {
            assert_eq!(Task::of(prompt, false), task, "{prompt}");
        }
        assert_eq!(Task::of("一只狐狸", true), Task::Edit);
        let zh = Task::GenerationZh.messages("一只狐狸", &[]);
        assert_eq!((zh[0].role.as_str(), zh[0].content.text()), ("system", prompts::GENERATION_ZH.to_owned()));
        assert!(prompts::GENERATION_ZH.starts_with("# 图像 Prompt 改写专家"));
        assert_eq!(zh[1].content.text(), "用户输入：一只狐狸\n改写输出：");
        let en = Task::GenerationEn.messages("  a fox ", &[]);
        assert!(en[0].content.text().starts_with("# Image Prompt Rewriting Expert"));
        assert_eq!(en[1].content.text(), "User Input: a fox\n\n Rewritten Prompt:");
        assert!(Task::Edit.system_prompt().starts_with("# Edit Prompt Enhancer"));
    }

    /// An edit's enhancer that sees gets each condition image as a PNG
    /// `image_url` part, in order, before the user turn's text.
    #[test]
    fn an_edit_shows_its_images_before_the_instruction() {
        let image = RgbaImage { size: ImageSize::new(2, 1), pixels: vec![255, 0, 0, 128, 0, 255, 0, 255] };
        let messages = Task::Edit.messages("make it blue", &[image.clone(), image]);
        let MessageContent::Parts(parts) = &messages[1].content else { panic!("parts: {:?}", messages[1]) };
        assert_eq!(parts.len(), 3);
        for part in &parts[..2] {
            assert_eq!(part.kind.as_deref(), Some("image_url"));
            assert!(part.url.as_deref().is_some_and(|url| url.starts_with("data:image/png;base64,iVBOR")), "{part:?}");
        }
        assert_eq!((parts[2].kind.as_deref(), parts[2].text.as_deref()), (Some("text"), Some("User Input: make it blue\n\nRewritten Prompt:")));
    }

    /// The answer read as upstream reads it: lines joined, an edit's JSON
    /// object unwrapped, fenced or not, and an answer with nothing in it no
    /// prompt.
    #[test]
    fn the_answer_is_read_as_upstream_reads_it() {
        assert_eq!(Task::GenerationEn.read("  A fox.\nIn snow.  "), Some("A fox. In snow.".to_owned()));
        assert_eq!(Task::Edit.read("```json\n{\n  \"Rewritten\": \"Make the car blue.\\nKeep the rest.\"\n}\n```"), Some("Make the car blue. Keep the rest.".to_owned()));
        assert_eq!(Task::Edit.read("{\"Rewritten\": \"Add a cat\"}"), Some("Add a cat".to_owned()));
        assert_eq!(Task::Edit.read("Add a light-gray cat"), Some("Add a light-gray cat".to_owned()), "not the object: the answer itself");
        assert_eq!(Task::GenerationZh.read(" \n "), None);
        assert_eq!(Task::Edit.read("{\"Rewritten\": \"  \"}"), None);
    }
}
