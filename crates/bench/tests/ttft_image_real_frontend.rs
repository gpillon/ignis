//! The multimodal TTFT cell's prompts over the real artifact's frontend
//! (GitHub #181): the length a cell claims is the chat template's text plus
//! the committed screenshot's vision tokens, and every sample's copy of the
//! screenshot — one pixel changed — expands to that same length.
//!
//! CPU only (the tokenizer, the template and the vision processor; no GPU,
//! ADR 0006). Skips when the artifact is not at its machine-local path, the
//! convention `crates/artifact/tests/real_frontend.rs` sets.

use std::path::{Path, PathBuf};

use ignis_artifact::vision::IMAGE_PAD_ID;
use ignis_artifact::{FrontendSet, Reader};
use ignis_bench::ttft::{self, PromptTemplate};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";

/// The screenshot `tools/vision-ttft/screenshot.py` draws: 1280x800, on the
/// processor's 32-pixel grid, so 40x25 merged blocks.
const VISION_TOKENS: usize = 40 * 25;

fn screenshot() -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vision_ttft/screenshot.png");
    std::fs::read(path).expect("the committed screenshot")
}

#[test]
fn the_screenshot_cell_claims_its_text_plus_a_thousand_vision_tokens() {
    let path = Path::new(ARTIFACT);
    if !path.exists() {
        eprintln!("skip: {ARTIFACT} does not exist");
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {ARTIFACT}: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let png = screenshot();
    let question = ttft::DEFAULT_IMAGE_QUESTION;

    let set = ttft::generate_image_cell_prompts(&frontend, &png, question, 6).expect("the cell's prompts");
    let with_image = frontend.encode_user_message_with_image(&set.prompts[0], &set.images[0]).expect("encode");
    let text_only = frontend.encode_user_message(&set.prompts[0]).expect("encode");
    // The image part renders as its vision-start/end markers around the
    // placeholder run; the run is the image's merged blocks.
    let pads = with_image.iter().filter(|&&id| id == IMAGE_PAD_ID).count();
    eprintln!(
        "screenshot cell: {} tokens ({} text-only, {pads} image pads)",
        set.prompt_tokens,
        text_only.len()
    );
    assert_eq!(set.prompt_tokens as usize, with_image.len());
    assert_eq!(pads, VISION_TOKENS, "1280x800 is 40x25 merged blocks");
    assert!(with_image.len() > text_only.len() + VISION_TOKENS, "the text is still there");
}
