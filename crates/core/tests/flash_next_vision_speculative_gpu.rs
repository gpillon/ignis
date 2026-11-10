//! After an image, Flash-Next's verify round keeps the text path's state and
//! its routes drift as they do on text (GitHub #564, spec #555, ADR 0087): the
//! diagnosis of `crates/server/tests/flash_next_vision_mtp_gpu.rs`'s split.
//!
//! That test generated the four vision canaries' prompts greedily on a load
//! with the MTP head and on one without it, and judged each split by the
//! prefill route of the load without it, after the agreed text: the two picks
//! within one BF16 ulp, or the prefill siding with the head's run. One split
//! failed (circles, detail: the prefill route 0.75 logits, six ulps, on the
//! headless run's side), and another passed only because the prefill sided
//! with the head's run by the same 0.75 -- there the plain decode round, with
//! no speculation in it at all, was the one six ulps off the prefill. Either
//! (a) something an image leaves in a sequence's state is not kept by a
//! verify round as a plain round keeps it, or (b) after an image the three
//! routes -- the plain round, the verify round's k + 1 columns, the prefill's
//! chunk -- differ by summation order as on text, and the image prompts'
//! long descriptions meet more near-ties for that drift to split at.
//!
//! One test, two loads with vision (the tower in host memory, BF16 KV, one
//! lane, 128-token chunks, MTP-3 -- the failing test's settings), and the
//! canary prompts beside text controls of the same length, each image's
//! placeholder run (with its vision start and end) replaced by a sentence
//! describing it, repeated to the run's length: the same positions, no image
//! (and beside themselves with a text's rows for the image's, the arms
//! below). 64 greedy tokens a prompt. What it holds (exact, BF16 KV):
//!
//! - **A rejected draft after an image is a draft never made.** On the MTP
//!   load, from the same standing prompt: rounds without drafts (`none`), a
//!   drafter whose every draft is rejected, one whose every draft is
//!   accepted (none's own text), and the head's own drafts commit none's
//!   text, and leave the state none leaves -- the next token prefilled at the
//!   frontier, its logits bit for bit (KV, GDN states, conv taps, the n-gram
//!   conv; the indexer's tail and axis records reach a logit only past its
//!   dense range, which these prompts of about a hundred tokens never leave,
//!   as the failing test's never did). Red: a verify round's commit or
//!   restore after an image is wrong -- (a).
//! - **The head's load computes the headless load's trunk after an image.**
//!   The plain rounds on the MTP load commit the load without the head's
//!   tokens with its logits, every round bit for bit, and the prompt's last
//!   prefill row too -- held for the image prompts and the placed ones
//!   whenever the text controls hold it. Red: the head's entries or sections
//!   reach the trunk after an image -- (a). Where the text controls differ
//!   across the loads too, the two loads compute differently whatever the
//!   prompt (no text test compares across them): printed, not held, and not
//!   the image's doing -- though the failing test's split then carries that
//!   difference beside the routes'.
//!
//! What it reports, per kind (image prompts, placed text rows, text
//! controls: the arms below), along each route's own text with the prefill
//! of that text as the reference: the
//! largest logit drift a row (BF16 ulps at the row's largest magnitude, as
//! `support/logit_drift.rs` measures it) for the plain round against the
//! prefill, the verify round against the prefill, and the verify round
//! against the plain round teacher-forced -- the plain text offered as the
//! drafts, ADR 0056's measure, which it put at up to 40.2 ulps on text at
//! this round's 8 rows; the picks each moved, at what gap in the
//! reference row; the near-ties along the plain text; and every split of the
//! verify round from the plain round on this one load, judged by the failing
//! test's rule (the prefill's gap within one ulp at the two picks' magnitude,
//! or on the verify round's side) and by ADR 0056's moved-pick rule (the
//! plain row's gap within `PICK_GAP_ULPS`).
//!
//! Reading it: a red assertion is (a), and names where. All green, with the
//! image prompts' drift on a route pair well past the text controls' (twice
//! their median or p99, say), is (a) too: an image-dependent numeric defect
//! in that route (the rope delta's rotation, attention over the image's keys),
//! the pair naming which. All green, the image drift within the text
//! controls', and the splits at gaps the plain row calls near-ties, is (b):
//! the failing rule asks the prefill route to referee two routes it sits as
//! far from as they sit from each other. The rule is the owner's to change.
//!
//! **Whether the image's rows or the multimodal path doubles the drift.** The
//! first run was all green, every route pair about twice as far on the image
//! prompts as on their text controls; the owner's call (2026-11-09): one
//! decisive experiment, then the fix or the rule. So each image prompt runs a
//! second time with its columns placed, not encoded
//! (`FlashNextEngine::place_media`, the forward test's placed columns): its
//! text control's own embedding rows at the placeholders' places. All else
//! is the image prompt's -- its ids, its three-axis positions and rope delta,
//! the pool and the scatter of its columns over the placeholders, its n-gram
//! rows hashing the placeholders, its vision start and end. Three arms:
//!
//! - A, the image prompts: the multimodal path, the tower's rows.
//! - B, the placed text rows: the multimodal path, a text's rows.
//! - C, the text controls: a text's rows and no multimodal path. Its rows are
//!   B's but at the run's two ends (the description where B has the vision
//!   start and end), and its n-gram rows hash the description where B's hash
//!   the placeholders.
//!
//! Beside the drifts, each arm's columns as the pool holds them, read back:
//! their per-row L2 norm (median, p90, max) and largest |x|, the tower's
//! rows against a text's; and each pair's drift in logits beside the ulp it
//! is counted in (the reference row's largest magnitude's, which doubles at
//! each power of two).
//!
//! The reading, fixed before any run, on each route pair's median drift, B
//! over C. Within 1.3x on all three pairs: the multimodal path drifts as text
//! does, and A's extra drift is its rows' statistics (the norms beside) -- no
//! fix, and the MTP test's split takes ADR 0056's rule. Past 1.6x on any
//! pair: the path itself drifts, a fix, test first, where the pair points --
//! plain round against prefill, the decode's rotation at its position plus
//! the delta against the prefill's three axes; a verify pair, the verify
//! round's columns over the image's keys; any pair, the scatter or the media
//! rows' rounding into the four streams. Otherwise inconclusive, and reported
//! so. Whether B sits nearer C or A on a pair is printed beside.
//!
//! **Which piece of the path (GitHub #565).** That run read past 1.6x on
//! every pair (B over C 1.87x, 2.00x, 1.80x). The reference's multimodal
//! forward differs from its text forward only in the columns it scatters over
//! the placeholders and the three axes they rotate at: its n-gram rows hash
//! the placeholder ids as any text (`ple_input_ids` is the input ids, taken
//! before the scatter), and the scattered rows enter every stream. So B
//! shares with A, and not with C, three pieces: the three-axis positions with
//! the rope delta, the n-gram rows hashing the placeholders, and the scatter
//! of its columns out of the pool -- beside the vision start and end, whose
//! two rows every arm but C keeps. Three more arms, each B with one piece
//! taken back to C's, measured as B (the drafter checks, held on A, B and C,
//! left out):
//!
//! - D, B at text positions: its three axes its index on all three, as C's,
//!   and no rope delta, so its rounds rotate at their positions;
//! - E, B with the text's n-gram rows: they hash the text control's ids
//!   (`FlashNextEngine::span`'s `ngram_tokens`), B's columns and positions;
//! - F, B through the lookup: its placeholders' ids replaced by the text ids
//!   its columns are the rows of, embedded by the lookup, no pool and no
//!   scatter; its n-gram rows still hash the placeholders, its positions B's.
//!
//! The placed columns are the lookup's own decode of the table, so where the
//! scatter writes what the lookup writes F computes B bit for bit: held
//! prompt by prompt, the plain rounds' text and rows and the prompt's last
//! prefill row. Red: the scatter is the defect.
//!
//! The reading, fixed before any run, on each arm's median drift over C on
//! each route pair: within 1.3x on all three pairs the arm is back, and the
//! piece it takes away carries the drift; past 1.6x on any pair it is not.
//! One arm back names the piece; two or more, the pieces carry it only
//! together; none, no one piece does -- the drift is the multimodal span's as
//! a whole (or its vision start and end's), the issue closes on that, and the
//! MTP test keeps ADR 0056's rule.
//! A piece named is a defect only where a route computes it otherwise than
//! the reference: every route rotates through one `rope_pair` at the
//! reference's positions and the n-gram ids are the reference's hashing, so
//! D or E back names a property of the input; F apart from B is a defect of
//! the scatter.
//!
//! **The vision start and end alone (GitHub #565).** That run (log
//! `.scratch/555/gpu/s_arms.log` of the integration worktree) read every arm
//! but C at about twice C's drift, none of D, E, F back within the cut. What
//! every arm but C keeps is the vision start and end (ids 248053 and 248054)
//! around the image's run; C replaced the whole run by a description. One
//! more arm, measured as C:
//!
//! - G, C with only the vision start and end put where the image's run sits
//!   in A: the run's content stays the text control's tokens (so G's ids are
//!   F's), the text positions, the text's n-gram rows, the lookup -- no media,
//!   no three-axis positions. Only the measure runs on it (no drafter checks).
//!
//! The reading, fixed before any run, on G's median drift over C on each route
//! pair: within 1.3x on all three pairs, the two special tokens do not carry
//! the drift (and the doubling belongs to the multimodal span as a whole);
//! 1.6x or more on any pair, the two special tokens carry it (a property of
//! the model's inputs, not a defect); between the cuts, inconclusive. One
//! reading line is printed, and beside it the L2 norm of the two special
//! tokens' embedding rows against the text rows' median.
//!
//! `IGNIS_VISION_ARMS=g` runs only C and G (16 prompts, the drafter checks
//! only on C: about 5 minutes instead of about 15); the report then leaves
//! out the readings that need the other arms. Unset, every arm runs.
//!
//! ```text
//! cargo test -p ignis-core --features cuda --test flash_next_vision_speculative_gpu -- --ignored --nocapture --test-threads=1
//! IGNIS_VISION_ARMS=g cargo test -p ignis-core --features cuda --test flash_next_vision_speculative_gpu -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Machine-local: `F:/ai/models/Qwen3.8-Flash-Next-ignis/` (or
//! `IGNIS_FLASH_NEXT_DIR`) with its MTP and vision companions; the canary
//! images are the server's fixture (`crates/server/tests/fixtures/vision_canary/`).
//! Explicit GPU profile (ADR 0006): outside `IGNIS_GPU_PROFILE=1` a missing
//! artifact, companion or GPU is a skip, under it a failure. Two loads one
//! after the other, each pinning the ~38 GB expert pool; 48 prompts, the
//! piece arms' and G's without the drafter checks: about 15 minutes on the RTX 5090.

#![cfg(feature = "cuda")]

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use ignis_artifact::packer::ARTIFACT_FILE_NAME;
use ignis_artifact::vision::VisionProcessor;
use ignis_artifact::{ChatMessage, ChatRenderOptions, ContentPart, FrontendSet, MessageContent, Reader, Role};
use ignis_core::compute::ModelConfig;
use ignis_core::flash_next::{EngineOptions, FlashNextEngine, LaneRound, RoundLogits};
use ignis_core::kv_format::KvFormat;
use ignis_core::ngram::NgramContext;
use ignis_core::seq::Seq;
use ignis_core::speculation::{FlashNextSpeculation, SpeculativeBackend};
use ignis_core::step::{MediaEmbedding, MultimodalPrefill, SpanMediaColumns};
use ignis_core::vision::Multimodal;
use ignis_core::{gpu_profile, Vision, VisionResidency};

#[path = "support/logit_drift.rs"]
mod logit_drift;

use logit_drift::{bf16, bf16_ulp, row_drift, RowDrift, PICK_GAP_ULPS};

#[path = "support/embedding_rows.rs"]
mod embedding_rows;

use embedding_rows::embedding_rows;

const MODEL_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
const MAX_CONTEXT: u32 = 8192;
/// Narrower than a canary image's placeholder run, as the failing test's.
const PREFILL_CHUNK: u32 = 128;
/// The head's draft tokens, the failing test's.
const DRAFT_TOKENS: u32 = 3;
/// Tokens each prompt generates.
const TOKENS: usize = 64;
/// A draft no greedy step picks: the vocabulary's last id, a reserved token.
const REJECTED: u32 = 248_319;
/// The failing test's second question per canary.
const DETAIL_QUESTION: &str = "Describe everything you can see in this image, in detail.";
/// What each canary image shows, for its text control.
const DESCRIPTIONS: [(&str, &str); 4] = [
    ("number", "A white picture with the number 47 printed in bold black digits in its centre."),
    ("colour", "A plain light picture with one solid red square in its centre."),
    ("circles", "A white picture with three solid blue circles of the same size side by side."),
    ("text", "A white picture with one line of black text that reads: error: config.yaml not found."),
];

fn model_dir() -> PathBuf {
    std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(MODEL_DIR), PathBuf::from)
}

/// The vision both loads take: the context's envelope, the tower in host
/// memory (the RTX 5090 profile's residency).
fn vision() -> Vision {
    Vision::new(MAX_CONTEXT).expect("an envelope").with_residency(VisionResidency::Host)
}

/// A vision load with the MTP head (`mtp`) or without speculation, or `None`
/// through `skip_or_fail`.
fn engine(mtp: bool) -> Option<FlashNextEngine> {
    let dir = model_dir();
    let speculation = mtp.then(|| FlashNextSpeculation::new(SpeculativeBackend::Mtp, DRAFT_TOKENS, 0).expect("mtp-3"));
    let options = EngineOptions {
        prefill_chunk_tokens: PREFILL_CHUNK,
        max_context_tokens: MAX_CONTEXT,
        kv_format: KvFormat::Bf16,
        decode_lanes: 1,
        speculation,
        vision: Some(vision()),
        ..EngineOptions::default()
    };
    match FlashNextEngine::load(&dir, options) {
        Ok(engine) => Some(engine),
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("load Flash-Next with vision (MTP {mtp}): {e}"));
            None
        }
    }
}

/// A load's pinned expert pool may come back to the host after its drop:
/// wait, bounded, until the host has back what it had before the load
/// (`flash_next_vision_mtp_gpu.rs`'s wait).
fn wait_for_host_memory(before: u64) {
    use ignis_core::residency::available_physical_bytes;
    for _ in 0..60 {
        if available_physical_bytes().unwrap_or(0) + (1 << 30) >= before {
            return;
        }
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
}

// ── the prompts ─────────────────────────────────────────────────────────────

/// Which arm of the drift experiment a prompt is (the module's doc).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    /// A: a canary image, its columns the tower's.
    Image,
    /// B: the same prompt, its columns its text control's embedding rows.
    Placed,
    /// C: the text control.
    Text,
    /// D (GitHub #565): B at its index on all three axes, no rope delta.
    TextPositions,
    /// E: B with its n-gram rows hashing the text control's ids.
    TextNgram,
    /// F: B with its placeholders' text ids embedded by the lookup, no
    /// pool and no scatter.
    Lookup,
    /// G (GitHub #565): C with only the vision start and end where the
    /// image's run sits in A.
    Specials,
}

impl Arm {
    const ALL: [Arm; 7] =
        [Arm::Image, Arm::Placed, Arm::Text, Arm::TextPositions, Arm::TextNgram, Arm::Lookup, Arm::Specials];
    /// The arms that each take one piece of B's path back to C's.
    const PIECES: [Arm; 3] = [Arm::TextPositions, Arm::TextNgram, Arm::Lookup];

    fn label(self) -> &'static str {
        match self {
            Arm::Image => "image prompts",
            Arm::Placed => "placed text rows",
            Arm::Text => "text controls",
            Arm::TextPositions => "D, placed at text positions",
            Arm::TextNgram => "E, placed with the text's n-gram rows",
            Arm::Lookup => "F, placed through the lookup",
            Arm::Specials => "G, text with the vision start and end",
        }
    }

    /// The piece of B's path the arm takes back to C's.
    fn piece(self) -> &'static str {
        match self {
            Arm::TextPositions => "the three-axis positions and the rope delta (a property of the input)",
            Arm::TextNgram => "the n-gram rows hashing the placeholders (a property of the input)",
            Arm::Lookup => "the scatter out of the pool (a defect: it writes what the lookup writes)",
            Arm::Image | Arm::Placed | Arm::Text | Arm::Specials => "no piece",
        }
    }

    /// Whether the prompt's run holds the drafter checks (A, B, C: the
    /// module's doc), which the piece arms leave out.
    fn checks_drafters(self) -> bool {
        matches!(self, Arm::Image | Arm::Placed | Arm::Text)
    }
}

/// What an image prompt's placeholders take (the module's doc).
enum Placeholders {
    /// The tower's columns, encoded into the pool (A).
    Tower,
    /// Columns placed in the pool, BF16 bits `[placeholders][hidden]` (B, D, E).
    Placed(Vec<u16>),
    /// Their ids' rows through the embedding lookup, no pool and no scatter
    /// (F; and a text prompt's every id).
    Lookup,
}

/// A prompt: its ids, the ids its n-gram rows hash where they are not its
/// own, and an image's positions, rope delta and media item, with what its
/// placeholders take.
struct Prompt {
    name: String,
    arm: Arm,
    tokens: Vec<u32>,
    /// The ids the n-gram rows hash in `tokens`' place (E, F).
    hashed: Option<Vec<u32>>,
    multimodal: Option<Multimodal>,
    placeholders: Placeholders,
}

/// Whether `IGNIS_VISION_ARMS=g` asks for C and G alone.
fn arms_only_g() -> bool {
    std::env::var("IGNIS_VISION_ARMS").is_ok_and(|v| v.eq_ignore_ascii_case("g"))
}

/// `ignis_server::media::load_processor`'s processor at this load's context.
fn processor(frontend: &FrontendSet) -> VisionProcessor {
    let mut options = frontend.vision_processor().expect("the vision processor").options().clone();
    options.max_vision_tokens = u64::from(vision().envelope_tokens(MAX_CONTEXT));
    options.max_raw_patches = options.max_vision_tokens * 4;
    VisionProcessor::new(frontend.tokenizer(), options).expect("the bounded processor")
}

/// The one id `text` tokenizes to.
fn single_id(frontend: &FrontendSet, text: &str) -> u32 {
    let ids = frontend.tokenizer().encode(text).unwrap_or_else(|e| panic!("tokenize {text}: {e}"));
    assert_eq!(ids.len(), 1, "{text} is one token: {ids:?}");
    ids[0]
}

/// The canaries (`id`, image bytes, own question), off the server's fixture.
fn canaries() -> Vec<(String, Vec<u8>, String)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../server/tests/fixtures/vision_canary");
    let text = std::fs::read_to_string(dir.join("fixture.json")).expect("read the vision canary fixture");
    let fixture: serde_json::Value = serde_json::from_str(&text).expect("fixture json");
    fixture["canaries"]
        .as_array()
        .expect("canaries")
        .iter()
        .map(|canary| {
            let field = |name: &str| canary[name].as_str().unwrap_or_else(|| panic!("{name}")).to_owned();
            (field("id"), std::fs::read(dir.join(field("image"))).expect("a canary image"), field("question"))
        })
        .collect()
}

/// Each canary with its own question and the long one, thinking off and the
/// image first (`flash_next_vision_mtp_gpu.rs`'s `prepare`); the same prompts
/// with their text controls' embedding rows placed over the placeholders;
/// then the text controls: each prompt with the image's run -- its vision
/// start, its placeholders, its vision end -- replaced by its description's
/// ids, repeated to the run's length; and the piece arms D, E, F. The piece
/// arms first, so a span they ask for and a load refuses shows a minute into
/// the run, then A, B, C; each arm's prompts in the canaries' order.
fn prompts(dir: &Path) -> Vec<Prompt> {
    let reader = Reader::open(&dir.join(ARTIFACT_FILE_NAME)).expect("open the artifact");
    let frontend = FrontendSet::from_reader(&reader).expect("frontend");
    let processor = processor(&frontend);
    let (start, end) = (single_id(&frontend, "<|vision_start|>"), single_id(&frontend, "<|vision_end|>"));
    let mut all = Vec::new();
    for (id, image, own) in canaries() {
        let description = DESCRIPTIONS
            .iter()
            .find(|(canary, _)| *canary == id)
            .unwrap_or_else(|| panic!("no description of canary {id}"))
            .1;
        let described = frontend.tokenizer().encode(&format!(" {description}")).expect("tokenize the description");
        for (kind, question) in [("own", own.as_str()), ("detail", DETAIL_QUESTION)] {
            let name = format!("{id} ({kind})");
            let messages = [ChatMessage {
                role: Role::User,
                content: MessageContent::Parts(vec![ContentPart::Image { url: None }, ContentPart::Text(question.to_owned())]),
                tool_calls: Vec::new(),
                reasoning_content: None,
            }];
            let options = ChatRenderOptions { enable_thinking: false, ..Default::default() };
            let prepared = frontend
                .prepare_prompt(&processor, &messages, &[image.as_slice()], options, None)
                .unwrap_or_else(|e| panic!("{name}: prepare the prompt: {e}"));
            let (tokens, multimodal) = Multimodal::from_prepared(prepared);
            assert_eq!(multimodal.media.len(), 1, "{name}: one image");
            assert_ne!(multimodal.rope_delta, 0, "{name}: an image's rope delta");
            let span = multimodal.media[0].token_span;
            let (first, last) = (span.begin - 1, span.begin + span.count);
            assert_eq!((tokens[first], tokens[last]), (start, end), "{name}: the image's run is wrapped in its vision start and end");
            let run: Vec<u32> = described.iter().cycle().take(last + 1 - first).copied().collect();
            let text = [&tokens[..first], &run[..], &tokens[last + 1..]].concat();
            assert_eq!(text.len(), tokens.len(), "{name}: the text control's length");
            let placeholders = span.begin..span.begin + span.count;
            let rows = embedding_rows(&reader, &text[placeholders.clone()]);
            // D: the text control's positions, its index on all three axes, and no rope delta.
            let linear = Multimodal {
                positions: (0..3).flat_map(|_| 0..tokens.len() as i32).collect(),
                rope_delta: 0,
                media: multimodal.media.clone(),
            };
            // F: the ids whose rows B places, where B has its placeholders.
            let looked_up = [&tokens[..span.begin], &text[placeholders.clone()], &tokens[placeholders.end..]].concat();
            let prompt = |suffix: &str, arm, tokens: &[u32], hashed: Option<&[u32]>, multimodal: &Multimodal, placeholders| Prompt {
                name: format!("{name} {suffix}"),
                arm,
                tokens: tokens.to_vec(),
                hashed: hashed.map(<[u32]>::to_vec),
                multimodal: Some(multimodal.clone()),
                placeholders,
            };
            all.push(prompt("placed", Arm::Placed, &tokens, None, &multimodal, Placeholders::Placed(rows.clone())));
            all.push(prompt("at text positions", Arm::TextPositions, &tokens, None, &linear, Placeholders::Placed(rows.clone())));
            all.push(prompt("with text n-gram rows", Arm::TextNgram, &tokens, Some(text.as_slice()), &multimodal, Placeholders::Placed(rows)));
            all.push(prompt("through the lookup", Arm::Lookup, &looked_up, Some(tokens.as_slice()), &multimodal, Placeholders::Lookup));
            all.push(Prompt {
                name: format!("{name} as text"),
                arm: Arm::Text,
                tokens: text,
                hashed: None,
                multimodal: None,
                placeholders: Placeholders::Lookup,
            });
            // G: the text control with only the run's two ends the vision start and end: F's
            // ids (`looked_up`), hashed as themselves, no media and no three-axis positions.
            all.push(Prompt {
                name: format!("{name} as text with the vision start and end"),
                arm: Arm::Specials,
                tokens: looked_up,
                hashed: None,
                multimodal: None,
                placeholders: Placeholders::Lookup,
            });
            all.push(Prompt { name, arm: Arm::Image, tokens, hashed: None, multimodal: Some(multimodal), placeholders: Placeholders::Tower });
        }
    }
    // IGNIS_VISION_ARMS=g: only C and G (the module's doc).
    if arms_only_g() {
        all.retain(|prompt| matches!(prompt.arm, Arm::Text | Arm::Specials));
    }
    // Stable: each arm's prompts stay in the canaries' order.
    all.sort_by_key(|prompt| (prompt.arm.checks_drafters(), prompt.arm as usize));
    all
}

/// The columns `prompt`'s placeholders take on `engine`: its image encoded by
/// the tower, or the rows placed for it; `None` for a text prompt and for
/// placeholders taking the lookup's rows.
fn embedding<'e>(engine: &'e FlashNextEngine, prompt: &Prompt) -> Option<MediaEmbedding<'e>> {
    let multimodal = prompt.multimodal.as_ref()?;
    match &prompt.placeholders {
        Placeholders::Placed(rows) => {
            Some(engine.place_media(rows).unwrap_or_else(|e| panic!("{}: place the rows: {e}", prompt.name)))
        }
        Placeholders::Tower => Some(
            engine
                .encode_media(&multimodal.media[0])
                .unwrap_or_else(|e| panic!("{}: encode the image: {e}", prompt.name)),
        ),
        Placeholders::Lookup => None,
    }
}

/// A media embedding's columns, as its pool holds them: each one's L2 norm,
/// and the largest |x| of any.
struct ColumnNorms {
    l2: Vec<f32>,
    max_abs: f32,
}

impl ColumnNorms {
    fn of(media: &MediaEmbedding<'_>) -> Self {
        let hidden = ModelConfig::qwen38_flash_next().hidden;
        let bits = media.read_bits(hidden).expect("read the columns back");
        let l2 = bits
            .chunks_exact(hidden as usize)
            .map(|column| column.iter().map(|&b| f64::from(bf16(b)).powi(2)).sum::<f64>().sqrt() as f32)
            .collect();
        Self { l2, max_abs: bits.iter().map(|&b| bf16(b).abs()).fold(0f32, f32::max) }
    }
}

/// `prompt` and `appended` (text after it) prefilled from position 0 as one
/// span -- at the image's three axes with its columns over its placeholders,
/// or as text -- its n-gram rows hashing `prompt.hashed` then `appended`
/// where it names other ids: the standing sequence, its context, and every
/// row's logits.
fn standing<'e>(
    engine: &'e FlashNextEngine,
    prompt: &Prompt,
    media: Option<&'e MediaEmbedding<'e>>,
    appended: &[u32],
) -> (Seq<'e>, NgramContext, Vec<u16>) {
    let mut tokens = prompt.tokens.clone();
    tokens.extend_from_slice(appended);
    let hashed = prompt.hashed.as_ref().map(|ids| [ids.as_slice(), appended].concat());
    let result = match &prompt.multimodal {
        None => engine.span(&tokens, hashed.as_deref(), None),
        Some(multimodal) => {
            let mut multimodal = multimodal.clone();
            assert!(multimodal.append_text(appended.len()), "{}: the text continues the prompt's positions", prompt.name);
            let span = multimodal.media[0].token_span;
            let scatter: Vec<i32> = (span.begin as i32..(span.begin + span.count) as i32).collect();
            // None where the placeholders take the lookup's rows (F).
            let columns = media.map(|embedding| SpanMediaColumns { embedding, first_column: 0, scatter_indices: &scatter });
            let span = MultimodalPrefill {
                positions: &multimodal.positions,
                rope_delta: multimodal.rope_delta,
                media: columns,
            };
            engine.span(&tokens, hashed.as_deref(), Some(span))
        }
    };
    result.unwrap_or_else(|e| panic!("{}: the span: {e}", prompt.name))
}

// ── the runs ────────────────────────────────────────────────────────────────

/// Collects every round's first column -- the row that picks the token after
/// the round's anchor -- or, `all`, every column (one lane).
struct Columns {
    vocab: usize,
    all: bool,
    rows: Vec<Vec<u16>>,
}

impl Columns {
    fn new(vocab: usize, all: bool) -> Self {
        Self { vocab, all, rows: Vec::new() }
    }

    fn take(&mut self, round: RoundLogits<'_>) {
        assert_eq!(round.lanes.len(), 1, "one lane");
        let width = if self.all { round.columns } else { 1 };
        self.rows.push(round.rows[..width * self.vocab].to_vec());
    }
}

/// One speculative run from the standing prompt: its tokens, its rounds, the
/// first column (or every column) of each, the head's acceptance (rounds,
/// drafted, accepted), and the probe after it.
struct SpecRun {
    tokens: Vec<u32>,
    rounds: Vec<LaneRound>,
    rows: Vec<Vec<u16>>,
    acceptance: (u32, u32, u32),
    probe: Vec<f32>,
}

fn speculative(
    engine: &FlashNextEngine,
    prompt: &Prompt,
    media: Option<&MediaEmbedding<'_>>,
    all: bool,
    drafter: Option<&mut dyn FnMut(usize, &[u32], u32) -> Vec<u32>>,
) -> SpecRun {
    let (seq, context, _) = standing(engine, prompt, media, &[]);
    let mut columns = Columns::new(engine.vocab(), all);
    let mut run = engine
        .generate_speculative_standing(
            vec![seq],
            vec![context],
            TOKENS,
            drafter,
            Some(&mut |round: RoundLogits<'_>| columns.take(round)),
        )
        .unwrap_or_else(|e| panic!("{}: speculative rounds: {e}", prompt.name));
    let probe = {
        let (seq, context) = (&mut run.sequences[0], &mut run.contexts[0]);
        engine.probe_logits(seq, context).unwrap_or_else(|e| panic!("{}: the probe: {e}", prompt.name))
    };
    let rounds: Vec<LaneRound> = run.rounds.iter().flatten().copied().collect();
    let acceptance = rounds.iter().fold((0, 0, 0), |(r, d, a), lane| (r + 1, d + lane.extent, a + lane.committed - 1));
    SpecRun { tokens: run.tokens.remove(0), rounds, rows: columns.rows, acceptance, probe }
}

/// ADR 0056's teacher-forced comparison: `forced` verified `plain`'s own
/// text as its drafts, so while its committed text is `plain`'s, each
/// drafted column holds `plain`'s token and its row reads what the plain
/// round at the same index read. Every such row against that plain row, up
/// to `len`; a run whose pick moved leaves after that round.
fn teacher_forced(forced: &SpecRun, plain: &[u32], plain_rows: &[Vec<u16>], len: usize, vocab: usize) -> Vec<RowDrift> {
    let mut drifts = Vec::new();
    let mut anchor = 0usize;
    for (round, rows) in forced.rounds.iter().zip(&forced.rows) {
        if anchor >= len || forced.tokens[..=anchor] != plain[..=anchor] {
            break;
        }
        let columns = (rows.len() / vocab).min(round.extent as usize + 1);
        for j in 0..columns {
            let r = anchor + j;
            if r < len {
                drifts.push(row_drift(&plain_rows[r], &rows[j * vocab..(j + 1) * vocab]));
            }
        }
        anchor += round.committed as usize;
    }
    drifts
}

/// `count` plain rounds from the standing prompt: tokens and every row.
fn plain(engine: &FlashNextEngine, prompt: &Prompt, media: Option<&MediaEmbedding<'_>>) -> (Vec<u32>, Vec<Vec<u16>>, u64) {
    let (seq, context, rows) = standing(engine, prompt, media, &[]);
    let last = hash(&rows[rows.len() - engine.vocab()..]);
    drop(rows);
    let mut columns = Columns::new(engine.vocab(), false);
    let mut tokens = engine
        .generate_standing(vec![seq], vec![context], TOKENS, Some(&mut |round: RoundLogits<'_>| columns.take(round)))
        .unwrap_or_else(|e| panic!("{}: plain rounds: {e}", prompt.name));
    (tokens.remove(0), columns.rows, last)
}

/// The prefill route along `text`: the rows that pick `text[1..]` and the
/// one after it -- `rows[r]` reads what round `r` read.
fn prefill_route(engine: &FlashNextEngine, prompt: &Prompt, media: Option<&MediaEmbedding<'_>>, text: &[u32]) -> Vec<Vec<u16>> {
    let (seq, _, rows) = standing(engine, prompt, media, text);
    drop(seq);
    let vocab = engine.vocab();
    let first = prompt.tokens.len();
    (0..text.len()).map(|r| rows[(first + r) * vocab..(first + r + 1) * vocab].to_vec()).collect()
}

fn hash(row: &[u16]) -> u64 {
    let mut hasher = DefaultHasher::new();
    row.hash(&mut hasher);
    hasher.finish()
}

// ── the measurements ────────────────────────────────────────────────────────

/// The verify round's first split from the plain round on one load.
struct Split {
    at: usize,
    plain: u32,
    verify: u32,
    /// The prefill's `logit(plain) - logit(verify)` and the failing test's
    /// ulp (at the larger of the two picks' magnitudes).
    prefill_gap: f32,
    ulp: f32,
    /// ADR 0056's measure: the plain row's gap between the picks, in ulps at
    /// its largest magnitude.
    plain_row_gap_ulps: f32,
}

/// What one prompt measured on the MTP load.
struct Measured {
    name: String,
    arm: Arm,
    /// The columns the image's placeholders took (`None` on text).
    norms: Option<ColumnNorms>,
    plain: Vec<u32>,
    plain_hashes: Vec<u64>,
    prompt_row: u64,
    plain_vs_prefill: Vec<RowDrift>,
    verify_vs_prefill: Vec<RowDrift>,
    verify_vs_plain: Vec<RowDrift>,
    /// The prefill's top-two margin along the plain text, in ulps at the
    /// row's largest magnitude.
    margins: Vec<f32>,
    split: Option<Split>,
    /// (drafter, commits none's text, leaves none's state).
    drafters: Vec<(&'static str, bool, bool)>,
    head: (u32, u32, u32),
}

/// The rows of `text` up to its first `eos`, inclusive (a run past the stop
/// it would have ended at is not the failing test's).
fn until_eos(text: &[u32], eos: u32) -> usize {
    text.iter().position(|&t| t == eos).map_or(text.len(), |at| at + 1)
}

fn top_two_margin_ulps(row: &[u16]) -> f32 {
    let (mut first, mut second) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    let mut scale = 0f32;
    for &bits in row {
        let v = bf16(bits);
        scale = scale.max(v.abs());
        if v > first {
            second = first;
            first = v;
        } else if v > second {
            second = v;
        }
    }
    (first - second) / bf16_ulp(scale)
}

fn measure(engine: &FlashNextEngine, prompt: &Prompt, eos: u32) -> Measured {
    let media = embedding(engine, prompt);
    let norms = media.as_ref().map(ColumnNorms::of);
    let media = media.as_ref();
    let (plain_text, plain_rows, prompt_row) = plain(engine, prompt, media);
    let none = speculative(engine, prompt, media, false, Some(&mut |_, _, _| Vec::new()));
    assert!(none.acceptance.1 == 0 && none.acceptance.0 > 0, "{}: none verified drafts", prompt.name);
    let reference = none.tokens.clone();
    // The drafter checks, where the arm holds them (the module's doc).
    let checked = prompt.arm.checks_drafters().then(|| {
        let oracle = speculative(
            engine,
            prompt,
            media,
            false,
            Some(&mut |_, emitted: &[u32], w| {
                let next = emitted.len() + 1;
                reference[next.min(TOKENS)..(next + w as usize).min(TOKENS)].to_vec()
            }),
        );
        let reject = speculative(engine, prompt, media, false, Some(&mut |_, _, w| vec![REJECTED; w as usize]));
        let head = speculative(engine, prompt, media, false, None);
        [("oracle", oracle), ("reject", reject), ("head", head)]
    });
    // ADR 0056's teacher-forced run: the plain rounds' own text as drafts.
    let forced = speculative(
        engine,
        prompt,
        media,
        true,
        Some(&mut |_, emitted: &[u32], w| {
            let next = emitted.len() + 1;
            plain_text[next.min(TOKENS)..(next + w as usize).min(TOKENS)].to_vec()
        }),
    );
    let same = |run: &SpecRun| {
        (run.tokens == none.tokens, run.probe.iter().zip(&none.probe).all(|(a, b)| a.to_bits() == b.to_bits()))
    };
    let drafters = checked
        .iter()
        .flatten()
        .map(|(name, run)| {
            let (text, state) = same(run);
            (*name, text, state)
        })
        .collect();
    let head_acceptance = checked.as_ref().map_or((0, 0, 0), |runs| runs[2].1.acceptance);
    drop(checked);

    let along_plain = prefill_route(engine, prompt, media, &plain_text);
    let along_none = prefill_route(engine, prompt, media, &none.tokens);
    let plain_len = until_eos(&plain_text, eos);
    let none_len = until_eos(&none.tokens, eos);
    let plain_vs_prefill: Vec<RowDrift> =
        (0..plain_len).map(|r| row_drift(&along_plain[r], &plain_rows[r])).collect();
    let verify_vs_prefill: Vec<RowDrift> = (0..none_len).map(|r| row_drift(&along_none[r], &none.rows[r])).collect();
    let margins = (0..plain_len).map(|r| top_two_margin_ulps(&along_plain[r])).collect();
    let at = plain_text.iter().zip(&none.tokens).position(|(a, b)| a != b);
    // Both runs' first token is the prompt's one prefill draw.
    assert_ne!(at, Some(0), "{}: one load's prefill drew two tokens", prompt.name);
    let verify_vs_plain = teacher_forced(&forced, &plain_text, &plain_rows, plain_len, engine.vocab());
    drop(forced);
    // A split past either run's stop is not one the failing test met.
    let split = at.filter(|&at| at < plain_len.min(none_len)).map(|at| {
        let (p, v) = (plain_text[at], none.tokens[at]);
        let row = &along_plain[at - 1];
        let (a, b) = (bf16(row[p as usize]), bf16(row[v as usize]));
        Split {
            at,
            plain: p,
            verify: v,
            prefill_gap: a - b,
            ulp: bf16_ulp(a.abs().max(b.abs())),
            plain_row_gap_ulps: row_drift(&plain_rows[at - 1], &none.rows[at - 1]).gap_ulps,
        }
    });
    Measured {
        name: prompt.name.clone(),
        arm: prompt.arm,
        norms,
        plain_hashes: plain_rows.iter().map(|row| hash(row)).collect(),
        plain: plain_text,
        prompt_row,
        plain_vs_prefill,
        verify_vs_prefill,
        verify_vs_plain,
        margins,
        split,
        drafters,
        head: head_acceptance,
    }
}

// ── the report ──────────────────────────────────────────────────────────────

/// `values`' quantile `q`, nearest rank.
fn quantile(values: &mut [f32], q: f64) -> f32 {
    if values.is_empty() {
        return f32::NAN;
    }
    values.sort_by(f32::total_cmp);
    values[((values.len() - 1) as f64 * q).round() as usize]
}

/// A route pair's drift over `rows`: median, p90, p99, max ulps, and its
/// moved picks with their gaps in the reference row; then the same drift in
/// logits (median, p99) and the ulp it was counted in (median).
fn drift_line(what: &str, rows: &[&RowDrift]) -> (String, f32, f32) {
    let mut ulps: Vec<f32> = rows.iter().map(|d| d.ulps).collect();
    let (median, p90, p99, max) =
        (quantile(&mut ulps, 0.5), quantile(&mut ulps, 0.9), quantile(&mut ulps, 0.99), quantile(&mut ulps, 1.0));
    let moved: Vec<f32> = rows.iter().filter(|d| d.pick_off != d.pick_on).map(|d| d.gap_ulps).collect();
    let mut logits: Vec<f32> = rows.iter().map(|d| d.max_abs).collect();
    // A row that did not move says nothing of its ulp.
    let mut ulp: Vec<f32> = rows.iter().filter(|d| d.ulps > 0.0).map(|d| d.max_abs / d.ulps).collect();
    let line = format!(
        "{what:30} {:4} rows  drift ulps median {median:6.2} p90 {p90:6.2} p99 {p99:6.2} max {max:6.2}  picks moved {} at gaps {moved:?}\n\
         {:32}in logits median {:.4} p99 {:.4}, at an ulp of median {}",
        rows.len(),
        moved.len(),
        "",
        quantile(&mut logits, 0.5),
        quantile(&mut logits, 0.99),
        quantile(&mut ulp, 0.5)
    );
    (line, median, p99)
}

/// The route pairs a prompt's drifts are reported for.
const PAIRS: [&str; 3] = ["plain round vs prefill", "verify round vs prefill", "verify vs plain, teacher-forced"];

fn pair(m: &Measured, index: usize) -> &[RowDrift] {
    match index {
        0 => &m.plain_vs_prefill,
        1 => &m.verify_vs_prefill,
        _ => &m.verify_vs_plain,
    }
}

/// The reading's cuts on B over C, each route pair's median drift (the
/// module's doc): within the first on every pair, the path drifts as text
/// does; past the second on any, the path drifts.
const PATH_AS_TEXT: f32 = 1.3;
const PATH_DRIFTS: f32 = 1.6;

/// The reading of B over C's median ratio on each route pair.
fn reading(placed_over_text: &[f32]) -> &'static str {
    if placed_over_text.iter().all(|&r| r <= PATH_AS_TEXT) {
        "the multimodal path drifts as text does: the image's extra drift is its rows' statistics -- \
         no fix, the MTP test's split takes ADR 0056's rule"
    } else if placed_over_text.iter().any(|&r| r >= PATH_DRIFTS) {
        "the multimodal path itself drifts, on the pairs past the cut: look for the cause and fix it, test first"
    } else {
        "inconclusive: a pair between the cuts, none past them"
    }
}

fn report(measured: &[Measured]) {
    // Per arm, each pair's (median, p99) drift ulps.
    let mut stats = Vec::new();
    for arm in Arm::ALL {
        let of: Vec<&Measured> = measured.iter().filter(|m| m.arm == arm).collect();
        if of.is_empty() {
            // An arm left out (IGNIS_VISION_ARMS).
            stats.push(Vec::new());
            continue;
        }
        println!("{}:", arm.label());
        let mut pairs = Vec::new();
        for (index, what) in PAIRS.iter().enumerate() {
            let all: Vec<&RowDrift> = of.iter().flat_map(|m| pair(m, index)).collect();
            let (line, median, p99) = drift_line(what, &all);
            println!("  {line}");
            pairs.push((median, p99));
        }
        let norms: Vec<&ColumnNorms> = of.iter().filter_map(|m| m.norms.as_ref()).collect();
        if !norms.is_empty() {
            let mut l2: Vec<f32> = norms.iter().flat_map(|n| n.l2.iter().copied()).collect();
            let max_abs = norms.iter().map(|n| n.max_abs).fold(0f32, f32::max);
            println!(
                "  the columns: {} (with repeats), L2 median {:.3} p90 {:.3} max {:.3}; largest |x| {max_abs}",
                l2.len(),
                quantile(&mut l2, 0.5),
                quantile(&mut l2, 0.9),
                quantile(&mut l2, 1.0)
            );
        }
        let margins: Vec<f32> = of.iter().flat_map(|m| m.margins.iter().copied()).collect();
        println!(
            "  near-ties along the plain text: {} of {} rows within 1 ulp, {} within 8",
            margins.iter().filter(|&&m| m <= 1.0).count(),
            margins.len(),
            margins.iter().filter(|&&m| m <= 8.0).count()
        );
        let splits: Vec<(&str, &Split)> = of.iter().filter_map(|m| m.split.as_ref().map(|s| (m.name.as_str(), s))).collect();
        println!("  {} of {} prompts split (verify round from plain round, one load):", splits.len(), of.len());
        for (name, s) in splits {
            println!(
                "    {name}: at token {}: plain {} verify {}; prefill gap plain - verify = {} (ulp {}): the failing rule {}; plain row's gap {:.2} ulps: ADR 0056's rule ({PICK_GAP_ULPS}) {}",
                s.at,
                s.plain,
                s.verify,
                s.prefill_gap,
                s.ulp,
                if s.prefill_gap <= s.ulp { "passes" } else { "FAILS" },
                s.plain_row_gap_ulps,
                if s.plain_row_gap_ulps <= PICK_GAP_ULPS { "passes" } else { "FAILS" },
            );
        }
        if arm.checks_drafters() {
            let head = of.iter().fold((0, 0, 0), |t, m| (t.0 + m.head.0, t.1 + m.head.1, t.2 + m.head.2));
            println!("  the head's drafts: {} rounds, {} drafted, {} accepted", head.0, head.1, head.2);
        }
        stats.push(pairs);
    }
    let (image, placed, text) = (&stats[0], &stats[1], &stats[2]);
    let ratio = |a: (f32, f32), b: (f32, f32)| (a.0 / b.0.max(f32::MIN_POSITIVE), a.1 / b.1.max(f32::MIN_POSITIVE));
    if !text.is_empty() && !stats[Arm::Specials as usize].is_empty() {
        let specials = &stats[Arm::Specials as usize];
        let over_text: Vec<f32> = (0..PAIRS.len()).map(|i| ratio(specials[i], text[i]).0).collect();
        println!(
            "{} / text drift, medians (plain/prefill, verify/prefill, verify/plain) {over_text:.2?}",
            Arm::Specials.label()
        );
        println!(
            "the vision start and end's reading (cuts {PATH_AS_TEXT}x and {PATH_DRIFTS}x on G over the text controls): {}",
            specials_reading(&over_text)
        );
    }
    if image.is_empty() || placed.is_empty() || text.is_empty() {
        return;
    }
    let mut placed_over_text = Vec::new();
    for (i, name) in PAIRS.iter().enumerate() {
        let (ac, bc, ab) = (ratio(image[i], text[i]), ratio(placed[i], text[i]), ratio(image[i], placed[i]));
        println!("image / text drift, {name}: median {:.2}x, p99 {:.2}x", ac.0, ac.1);
        println!(
            "placed / text drift, {name}: median {:.2}x, p99 {:.2}x; image / placed median {:.2}x, p99 {:.2}x: the placed rows sit nearer {}",
            bc.0,
            bc.1,
            ab.0,
            ab.1,
            if bc.0.ln().abs() <= ab.0.ln().abs() { "the text controls" } else { "the image prompts" }
        );
        placed_over_text.push(bc.0);
    }
    println!(
        "the reading (placed / text medians {placed_over_text:.2?}, cuts {PATH_AS_TEXT}x and {PATH_DRIFTS}x): {}",
        reading(&placed_over_text)
    );
    // GitHub #565: each piece arm over C, and over B beside.
    let (mut back, mut between) = (Vec::new(), Vec::new());
    for arm in Arm::PIECES {
        let of = &stats[arm as usize];
        if of.is_empty() {
            continue;
        }
        let over_text: Vec<f32> = (0..PAIRS.len()).map(|i| ratio(of[i], text[i]).0).collect();
        let over_placed: Vec<f32> = (0..PAIRS.len()).map(|i| ratio(of[i], placed[i]).0).collect();
        let verdict = piece_reading(&over_text);
        println!(
            "{} / text drift, medians (plain/prefill, verify/prefill, verify/plain) {over_text:.2?}, over the placed text rows {over_placed:.2?}: {}",
            arm.label(),
            match verdict {
                Some(true) => "back",
                Some(false) => "not back",
                None => "between the cuts",
            }
        );
        match verdict {
            Some(true) => back.push(arm),
            None => between.push(arm),
            Some(false) => {}
        }
    }
    println!(
        "the pieces' reading (cuts {PATH_AS_TEXT}x and {PATH_DRIFTS}x on each arm over the text controls): {}",
        pieces_reading(&back, &between)
    );
}

/// G's reading on its median over C, each route pair (the module's doc).
fn specials_reading(over_text: &[f32]) -> &'static str {
    if over_text.iter().all(|&r| r <= PATH_AS_TEXT) {
        "the vision start and end do not carry the drift: the doubling belongs to the multimodal span as a whole"
    } else if over_text.iter().any(|&r| r >= PATH_DRIFTS) {
        "the two special tokens carry the drift, on the pairs past the cut: a property of the model's inputs, not a defect"
    } else {
        "inconclusive: a pair between the cuts, none past them"
    }
}

/// A piece arm's reading on its median over C, each route pair (the module's
/// doc): back within the first cut on every pair (`Some(true)`), not back past
/// the second on any (`Some(false)`), else between the cuts.
fn piece_reading(over_text: &[f32]) -> Option<bool> {
    if over_text.iter().all(|&r| r <= PATH_AS_TEXT) {
        Some(true)
    } else if over_text.iter().any(|&r| r >= PATH_DRIFTS) {
        Some(false)
    } else {
        None
    }
}

/// What the piece arms back, and between the cuts, read as (the module's doc).
fn pieces_reading(back: &[Arm], between: &[Arm]) -> String {
    let pieces = |arms: &[Arm]| arms.iter().map(|arm| arm.piece()).collect::<Vec<_>>().join("; ");
    let reading = match back {
        [] if between.is_empty() => "no one piece carries the drift: it is the multimodal span's as a whole \
                                     (or its vision start and end's) -- #565 closes on that, the MTP test keeps \
                                     ADR 0056's rule"
            .to_owned(),
        [] => "inconclusive: no arm back".to_owned(),
        [arm] => format!("one piece carries the drift: {}", arm.piece()),
        _ => format!("the pieces carry it only together: {}", pieces(back)),
    };
    if between.is_empty() {
        reading
    } else {
        format!("{reading}; between the cuts: {}", pieces(between))
    }
}

/// F against B, prompt by prompt (the canaries' order in both): whether the
/// lookup's prompt computes the placed prompt's plain rounds -- their text and
/// every row -- and its prompt's last prefill row, bit for bit.
fn lookup_against_placed(measured: &[Measured]) -> Vec<(String, bool)> {
    let of = move |arm| measured.iter().filter(move |m: &&Measured| m.arm == arm);
    of(Arm::Lookup)
        .zip(of(Arm::Placed))
        .map(|(f, b)| {
            (f.name.clone(), f.plain == b.plain && f.plain_hashes == b.plain_hashes && f.prompt_row == b.prompt_row)
        })
        .collect()
}

// ── the test ────────────────────────────────────────────────────────────────

#[test]
#[ignore = "GPU profile only: Flash-Next with its MTP and vision companions on the RTX 5090"]
fn after_an_image_a_verify_round_keeps_the_plain_round_s_state_and_its_routes_drift_as_on_text() {
    let dir = model_dir();
    let path = dir.join(ARTIFACT_FILE_NAME);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("no Flash-Next artifact at {}", path.display())) {
        return;
    }
    for (what, companion) in [
        ("MTP", ignis_core::flash_next_mtp::companion_path(&dir)),
        ("vision", ignis_core::flash_next_vision::companion_path(&dir)),
    ] {
        if !companion.exists() && gpu_profile::skip_or_fail(&format!("no {what} companion at {}", companion.display())) {
            return;
        }
    }
    let prompts = prompts(&dir);
    let eos = {
        let reader = Reader::open(&path).expect("open the artifact");
        FrontendSet::from_reader(&reader).expect("frontend").eos_token_id().expect("eos")
    };
    let all: Vec<&Prompt> = prompts.iter().collect();
    for prompt in &all {
        println!("{}: {} prompt tokens", prompt.name, prompt.tokens.len());
    }
    let before = ignis_core::residency::available_physical_bytes().unwrap_or(0);

    let measured: Vec<Measured> = {
        let Some(on) = engine(true) else { return };
        println!(
            "the head's load: round graphs {:#b}, verify graphs {:?} ({:?})",
            on.graphs_ready(),
            on.verify_graphs_ready(),
            on.graph_error()
        );
        all.iter().map(|prompt| measure(&on, prompt, eos)).collect()
    };
    wait_for_host_memory(before);
    // The load without the head: its plain rounds against the head's load's.
    let mut cross = Vec::new();
    {
        let Some(off) = engine(false) else { return };
        for (prompt, m) in all.iter().zip(&measured) {
            let media = embedding(&off, prompt);
            let (tokens, rows, prompt_row) = plain(&off, prompt, media.as_ref());
            let differing = rows.iter().zip(&m.plain_hashes).filter(|(row, h)| hash(row) != **h).count();
            cross.push((m.name.clone(), m.arm, tokens == m.plain && differing == 0 && prompt_row == m.prompt_row, differing));
        }
    }

    for m in &measured {
        let medians: Vec<String> = (0..PAIRS.len())
            .map(|index| {
                let mut ulps: Vec<f32> = pair(m, index).iter().map(|d| d.ulps).collect();
                format!("{:.2}", quantile(&mut ulps, 0.5))
            })
            .collect();
        let norms = m.norms.as_ref().map(|n| {
            let mut l2 = n.l2.clone();
            format!("; its columns' L2 median {:.3}, largest |x| {}", quantile(&mut l2, 0.5), n.max_abs)
        });
        println!(
            "{}: plain {:?}... split {:?}; median drift ulps (plain/prefill, verify/prefill, verify/plain) {medians:?}{}",
            m.name,
            &m.plain[..8.min(m.plain.len())],
            m.split.as_ref().map(|s| (s.at, s.plain, s.verify, s.prefill_gap)),
            norms.unwrap_or_default()
        );
    }
    report(&measured);
    if all.iter().any(|prompt| prompt.arm == Arm::Specials) {
        let reader = Reader::open(&path).expect("open the artifact");
        let frontend = FrontendSet::from_reader(&reader).expect("frontend");
        let specials = [single_id(&frontend, "<|vision_start|>"), single_id(&frontend, "<|vision_end|>")];
        let mut ids: Vec<u32> = all
            .iter()
            .filter(|prompt| prompt.arm == Arm::Specials)
            .flat_map(|prompt| prompt.tokens.iter().copied())
            .filter(|id| !specials.contains(id))
            .collect();
        ids.sort_unstable();
        ids.dedup();
        let hidden = ModelConfig::qwen38_flash_next().hidden as usize;
        let l2 = |rows: &[u16]| -> Vec<f32> {
            rows.chunks_exact(hidden)
                .map(|row| row.iter().map(|&b| f64::from(bf16(b)).powi(2)).sum::<f64>().sqrt() as f32)
                .collect()
        };
        let mut text_l2 = l2(&embedding_rows(&reader, &ids));
        println!(
            "embedding rows' L2: vision start {:?}, vision end {:?}; the text rows' median {:.3} over {} distinct ids",
            l2(&embedding_rows(&reader, &specials[..1])),
            l2(&embedding_rows(&reader, &specials[1..])),
            quantile(&mut text_l2, 0.5),
            ids.len()
        );
    }
    let lookup = lookup_against_placed(&measured);
    println!("F against B, the plain rounds' text and rows and the prompt's last row bit for bit (the scatter writes what the lookup writes):");
    for (name, equal) in &lookup {
        println!("  {name}: {equal}");
    }
    println!("the headless load's plain rounds against the head's load's (all equal, rows differing):");
    for (name, _, equal, differing) in &cross {
        println!("  {name}: {equal} {differing}");
    }
    let text_equal = cross.iter().filter(|(_, arm, _, _)| *arm == Arm::Text).all(|(_, _, equal, _)| *equal);
    if !text_equal {
        println!("the loads differ on the text controls too: a difference of the loads, not of the image");
    }

    for m in &measured {
        for (drafter, text, state) in &m.drafters {
            assert!(text, "{}: the {drafter} drafter committed other text than rounds without drafts", m.name);
            assert!(state, "{}: the {drafter} drafter's rounds left another state than rounds without drafts", m.name);
        }
    }
    // G is a text prompt with two special tokens: printed above, not held.
    for (name, _, equal, differing) in
        cross.iter().filter(|(_, arm, _, _)| !matches!(arm, Arm::Text | Arm::Specials) && text_equal)
    {
        assert!(
            equal,
            "{name}: after the image the load without the head computes another trunk ({differing} rows differ), where the text controls agree"
        );
    }
    for (name, equal) in &lookup {
        assert!(equal, "{name}: the lookup's rows compute other logits than the same rows scattered out of the pool");
    }
}
