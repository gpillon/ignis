//! How many answer slots does this tokenizer actually offer?
//!
//! A typed-decision readout (`classify_readout_gpu.rs`) names each option by a
//! label and reads that label's token logit at one position. The whole
//! technique rests on the label being **exactly one token**: at two tokens the
//! readout would see only the first, and two labels sharing a first token
//! ("aa", "ab") would be indistinguishable without a second forward pass —
//! which is the one thing a zero-decode endpoint must not need.
//!
//! TypeSafe's Jev allows up to 255 options per question. SemIf's uppercase
//! alphabet reaches 16. This test counts what is actually available here, so
//! the endpoint's ceiling is a measured number rather than a guess.
//!
//! Machine-local: skips when the artifact is absent (`docs/agents/testing.md`
//! — the tokenizer is CPU-only and nowhere near the forward pass, so a skip
//! here is green).

use std::collections::BTreeSet;
use std::path::Path;

use ignis_artifact::{FrontendSet, Reader, Tokenizer};
use ignis_core::decision::AnswerAlphabet;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";

/// Whether `label` is one token that decodes back to itself.
fn is_clean_slot(tokenizer: &Tokenizer, label: &str) -> Option<u32> {
    let encoded = tokenizer.encode(label).ok()?;
    if encoded.len() != 1 {
        return None;
    }
    (tokenizer.decode(&encoded).ok()? == label).then_some(encoded[0])
}

/// Every clean single-token label in `candidates`, and the set of token ids
/// they resolve to — a label that collides with another's id is no use as a
/// slot even if it is clean on its own.
fn clean_slots<'a>(
    tokenizer: &Tokenizer,
    candidates: impl Iterator<Item = &'a str>,
) -> (Vec<(&'a str, u32)>, usize) {
    let mut slots = Vec::new();
    let mut ids = BTreeSet::new();
    let mut collisions = 0;
    for label in candidates {
        let Some(id) = is_clean_slot(tokenizer, label) else {
            continue;
        };
        if !ids.insert(id) {
            collisions += 1;
            continue;
        }
        slots.push((label, id));
    }
    (slots, collisions)
}

/// Does this model carry the grounding vocabulary Qwen-VL uses for pointing
/// at a region of an image? If the special tokens are there, the model was
/// trained to emit boxes and points, and coordinates should be asked for in
/// its own format rather than through an invented grid.
#[test]
fn report_the_grounding_vocabulary() {
    let path = Path::new(ARTIFACT);
    if !path.exists() {
        eprintln!("skip: {ARTIFACT} does not exist");
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {ARTIFACT}: {e}"));
    let set = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let tokenizer = set.tokenizer();
    for marker in [
        "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>", "<|point_start|>", "<|point_end|>",
        "<|vision_start|>", "<|vision_end|>", "<|image_pad|>",
    ] {
        match tokenizer.encode(marker) {
            Ok(ids) if ids.len() == 1 => eprintln!("ignis grounding: {marker} = single token {}", ids[0]),
            Ok(ids) => eprintln!("ignis grounding: {marker} = {} tokens (not a special token)", ids.len()),
            Err(e) => eprintln!("ignis grounding: {marker} = encode failed: {e}"),
        }
    }
}

#[test]
fn count_the_single_token_answer_slots() {
    let path = Path::new(ARTIFACT);
    if !path.exists() {
        eprintln!("skip: {ARTIFACT} does not exist");
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {ARTIFACT}: {e}"));
    let set = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let tokenizer = set.tokenizer();

    let upper: Vec<String> = ('A'..='Z').map(|c| c.to_string()).collect();
    let lower: Vec<String> = ('a'..='z').map(|c| c.to_string()).collect();
    let digits: Vec<String> = ('0'..='9').map(|c| c.to_string()).collect();
    let upper_bigrams: Vec<String> = ('A'..='Z')
        .flat_map(|a| ('A'..='Z').map(move |b| format!("{a}{b}")))
        .collect();
    let lower_bigrams: Vec<String> = ('a'..='z')
        .flat_map(|a| ('a'..='z').map(move |b| format!("{a}{b}")))
        .collect();
    let numbers: Vec<String> = (0..=255).map(|n: u32| n.to_string()).collect();

    for (name, labels) in [
        ("A-Z", &upper),
        ("a-z", &lower),
        ("0-9", &digits),
        ("AA-ZZ", &upper_bigrams),
        ("aa-zz", &lower_bigrams),
        ("0-255", &numbers),
    ] {
        let (slots, collisions) = clean_slots(tokenizer, labels.iter().map(String::as_str));
        eprintln!(
            "ignis slots: {name:<6} {:>3}/{:<3} clean single tokens, {collisions} id collisions",
            slots.len(),
            labels.len()
        );
        // The first few misses are the interesting part: which labels the
        // tokenizer splits, and into what.
        let missing: Vec<&str> = labels
            .iter()
            .map(String::as_str)
            .filter(|label| is_clean_slot(tokenizer, label).is_none())
            .take(6)
            .collect();
        if !missing.is_empty() {
            let shown: Vec<String> = missing
                .iter()
                .map(|label| {
                    let pieces = tokenizer.encode(label).map(|ids| ids.len()).unwrap_or(0);
                    format!("{label:?}({pieces} tokens)")
                })
                .collect();
            eprintln!("ignis slots: {name:<6} first misses: {}", shown.join(" "));
        }
    }

    // The ceiling the endpoint can actually offer, if it draws from the
    // alphabets in this order and stops when it has enough.
    let mut pool: Vec<String> = Vec::new();
    pool.extend(upper.iter().cloned());
    pool.extend(lower.iter().cloned());
    pool.extend(digits.iter().cloned());
    pool.extend(upper_bigrams.iter().cloned());
    pool.extend(lower_bigrams.iter().cloned());
    let (slots, collisions) = clean_slots(tokenizer, pool.iter().map(String::as_str));
    eprintln!(
        "ignis slots: pooled A-Z + a-z + 0-9 + bigrams = {} distinct clean slots ({collisions} collisions dropped); Jev's ceiling is 255",
        slots.len()
    );
    let sample: Vec<&str> = slots.iter().skip(62).take(12).map(|&(l, _)| l).collect();
    eprintln!("ignis slots: first bigram slots past the 62 singles: {sample:?}");
}

/// The **answer alphabet** this artifact's tokenizer actually yields
/// (GitHub #237, ADR 0034), against the same tokenizer the counts above are
/// measured on.
///
/// The counts are pinned rather than merely reported because the rule they
/// come from is a correctness rule: `BQ` is two tokens here, and an
/// alphabet that admitted it would read the logit of `B` at `BQ`'s slot —
/// `B` being option 2's own answer token. Two options reading one logit
/// cannot be told apart without a second forward pass, which is the one
/// thing a zero-decode decision must not need.
#[test]
fn the_answer_alphabet_rejects_the_labels_this_tokenizer_splits() {
    let path = Path::new(ARTIFACT);
    if !path.exists() {
        eprintln!("skip: {ARTIFACT} does not exist");
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {ARTIFACT}: {e}"));
    let set = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let tokenizer = set.tokenizer();
    let alphabet = AnswerAlphabet::from_tokenizer(tokenizer);

    // The named case. `BQ` splits, and its first piece is exactly `B`.
    assert!(
        alphabet.position("BQ").is_none(),
        "`BQ` is two tokens in this tokenizer and must not be an answer token"
    );
    let split = tokenizer.encode("BQ").expect("encode BQ");
    assert_eq!(split.len(), 2, "the premise: `BQ` is two tokens here");
    let b = alphabet.position("B").map(|i| alphabet.tokens()[i].id).expect("`B` is one token");
    assert_eq!(
        split[0], b,
        "admitting `BQ` would have read option `B`'s own logit at `BQ`'s slot"
    );

    // The singles are all clean, and the bigrams are not: 562 of 676.
    for label in ('A'..='Z').chain('a'..='z').chain('0'..='9').map(|c| c.to_string()) {
        assert!(alphabet.position(&label).is_some(), "the single {label:?} is one token");
    }
    let bigrams = alphabet.len() - 62;
    assert_eq!(
        bigrams, 562,
        "114 of the 676 uppercase bigrams are two tokens in this tokenizer"
    );
    assert_eq!(alphabet.len(), 624, "the alphabet the endpoint can offer");

    // No two slots name one token — the property the whole readout rests on.
    let ids: BTreeSet<u32> = alphabet.tokens().iter().map(|token| token.id).collect();
    assert_eq!(ids.len(), alphabet.len(), "no two answer tokens share an id");

    eprintln!(
        "ignis alphabet: {} answer tokens (62 singles + {bigrams} uppercase bigrams); Jev's ceiling is 255",
        alphabet.len()
    );
}
