//! The readout seam (GitHub #237, ADR 0034): the logits of named **answer
//! tokens** at a prompt's last position, carried back through `Compute`
//! without the full-vocabulary buffer ever crossing it.
//!
//! CPU-only, deliberately: the whole point of ADR 0006 is that the
//! scheduler's paths stay testable without a card, and a readout is a path
//! the scheduler will grow a request kind for (GitHub #238). A seam only
//! `CudaLeaf` could answer would stop being covered the moment it mattered.

use std::sync::Arc;

use ignis_core::decision::{AnswerAlphabet, LabelTokenizer, Readout};
use ignis_core::scheduler::{Compute, PrefillJob};
use ignis_core::{DecodeParams, MockCompute, TokenId};

fn job(request: u64, readout: Option<Vec<TokenId>>) -> PrefillJob {
    PrefillJob {
        request,
        tokens: vec![1, 2, 3],
        context_tokens: 16,
        start_position: 0,
        params: DecodeParams::default(),
        shared_prefix: None,
        publish_prefix: None,
        checkpoint: None,
        capture_checkpoint: None,
        multimodal: None,
        readout: readout.map(Arc::from),
        permitted: None,
    }
}

// ── the seam (acceptance 1, 4) ───────────────────────────────────────────

#[test]
fn a_requested_readout_comes_back_through_the_compute_seam() {
    let compute = MockCompute::new();
    let answers: Vec<TokenId> = vec![32, 33, 34, 35];
    let outcomes = compute
        .prefill_step(&[job(7, Some(answers.clone()))])
        .expect("the mock never fails an unarmed prefill");

    let readout = outcomes[0]
        .readout
        .as_ref()
        .expect("a job that asked for a readout gets one");
    assert_eq!(
        readout.logits.len(),
        answers.len(),
        "one logit per answer token, in the order asked for"
    );
    assert!(
        readout.logits.iter().all(|v| v.is_finite()),
        "a readout's logits are finite: {:?}",
        readout.logits
    );
    let mass = readout.answer_mass();
    assert!(
        (0.0..=1.0).contains(&mass),
        "answer mass is a probability, got {mass}"
    );
    assert!(
        mass > 0.0,
        "the mock's answer tokens hold some of its distribution"
    );
    assert!(
        mass < 1.0,
        "and never all of it: a mock whose mass were exactly 1 would let a caller          that forgot to check the mass pass on CPU and fail on the card, got {mass}"
    );
    let winner = readout.winner().expect("a non-empty readout names a winning slot");
    assert_eq!(
        readout.full_argmax, answers[winner],
        "the mock's unrestricted argmax is a declared answer, as the served          model's is on every row the finding scored"
    );
}

#[test]
fn a_job_that_asks_for_no_readout_gets_none() {
    let compute = MockCompute::new();
    let outcomes = compute
        .prefill_step(&[job(7, None)])
        .expect("the mock never fails an unarmed prefill");
    assert!(
        outcomes[0].readout.is_none(),
        "no readout was asked for, so none comes back"
    );
}

#[test]
fn the_mock_readout_is_deterministic() {
    let answers: Vec<TokenId> = vec![32, 33, 34];
    let first = MockCompute::new()
        .prefill_step(&[job(11, Some(answers.clone()))])
        .expect("prefill")[0]
        .readout
        .clone()
        .expect("a readout");
    let second = MockCompute::new()
        .prefill_step(&[job(11, Some(answers))])
        .expect("prefill")[0]
        .readout
        .clone()
        .expect("a readout");
    assert_eq!(first, second, "the same job reads out the same way twice");
}

#[test]
fn the_mock_reads_each_request_out_differently() {
    let answers: Vec<TokenId> = vec![32, 33, 34];
    let compute = MockCompute::new();
    let outcomes = compute
        .prefill_step(&[job(1, Some(answers.clone())), job(2, Some(answers))])
        .expect("prefill");
    assert_ne!(
        outcomes[0].readout.as_ref().expect("a readout").logits,
        outcomes[1].readout.as_ref().expect("a readout").logits,
        "two requests in one batch do not read out identically"
    );
}

// ── the gather itself ────────────────────────────────────────────────────

#[test]
fn the_gather_picks_the_named_tokens_and_summarizes_the_rest() {
    // A tiny vocabulary standing in for the 248,320-wide one: only the
    // gathered slice and the two summaries ever leave the adapter.
    let mut logits = vec![0.0f32; 8];
    logits[2] = 3.0;
    logits[5] = 1.0;
    logits[6] = 9.0;
    let readout = Readout::gather(&logits, &[2, 5]);

    assert_eq!(readout.logits, vec![3.0, 1.0]);
    assert_eq!(
        readout.full_argmax, 6,
        "the unrestricted argmax is reported even when it is not an answer"
    );
    assert_eq!(readout.winner(), Some(0), "the higher answer logit wins");
    let mass = readout.answer_mass();
    assert!(
        mass > 0.0 && mass < 0.01,
        "almost all the mass sits outside the answers, got {mass}"
    );
    let probabilities = readout.probabilities();
    assert!(
        (probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-9,
        "the restricted softmax sums to one: {probabilities:?}"
    );
}

#[test]
fn a_readout_whose_answers_hold_everything_has_mass_one() {
    // Every non-answer logit far below the answers: the restriction throws
    // nothing away, and the mass must not exceed one on the way back.
    let logits = vec![-60.0, 2.0, -60.0, 1.0];
    let readout = Readout::gather(&logits, &[1, 3]);
    let mass = readout.answer_mass();
    assert!(
        (0.0..=1.0).contains(&mass),
        "answer mass never leaves [0, 1], got {mass}"
    );
    assert!(mass > 0.999, "the answers hold the distribution, got {mass}");
}

#[test]
fn an_answer_token_past_the_vocabulary_reads_negative_infinity() {
    // Defensive rather than expected: a caller that names an id the model
    // has no column for gets a logit that loses every comparison, never a
    // panic on the prefill path and never another token's value.
    let readout = Readout::gather(&[1.0, 2.0], &[1, 99]);
    assert_eq!(readout.logits[0], 2.0);
    assert!(readout.logits[1].is_infinite() && readout.logits[1] < 0.0);
    assert_eq!(readout.winner(), Some(0));
}

// ── the answer alphabet (acceptance 2, 3) ────────────────────────────────

/// A tokenizer that is exactly a table: every entry is one token, anything
/// else splits into its characters. Enough to decide what the alphabet
/// admits, which is all the alphabet ever asks a tokenizer.
struct TableTokenizer {
    entries: Vec<String>,
}

impl TableTokenizer {
    /// Every single character and every uppercase bigram *except* the ones
    /// named — those split, the way `BQ` splits in the 27B's tokenizer.
    fn without(split: &[&str]) -> Self {
        let mut entries: Vec<String> = Vec::new();
        for c in ('A'..='Z').chain('a'..='z').chain('0'..='9') {
            entries.push(c.to_string());
        }
        for a in 'A'..='Z' {
            for b in 'A'..='Z' {
                let bigram = format!("{a}{b}");
                if !split.contains(&bigram.as_str()) {
                    entries.push(bigram);
                }
            }
        }
        Self { entries }
    }
}

impl LabelTokenizer for TableTokenizer {
    fn encode(&self, text: &str) -> Option<Vec<TokenId>> {
        if let Some(index) = self.entries.iter().position(|entry| entry == text) {
            return Some(vec![index as TokenId]);
        }
        text.chars()
            .map(|c| {
                self.entries
                    .iter()
                    .position(|entry| entry == &c.to_string())
                    .map(|index| index as TokenId)
            })
            .collect()
    }

    fn decode(&self, ids: &[TokenId]) -> Option<String> {
        ids.iter()
            .map(|&id| self.entries.get(id as usize).cloned())
            .collect::<Option<Vec<String>>>()
            .map(|pieces| pieces.concat())
    }
}

#[test]
fn the_alphabet_is_built_from_the_tokenizer_it_is_given() {
    let alphabet = AnswerAlphabet::from_tokenizer(&TableTokenizer::without(&["BQ", "CJ"]));
    assert!(
        alphabet.position("BQ").is_none(),
        "a label this tokenizer splits is not an answer token"
    );
    assert!(alphabet.position("CJ").is_none());
    assert!(
        alphabet.position("BR").is_some(),
        "its neighbours are unaffected"
    );
    assert_eq!(
        alphabet.len(),
        26 + 26 + 10 + 676 - 2,
        "everything else this tokenizer holds whole is admitted"
    );
}

#[test]
fn a_two_token_label_would_have_read_another_labels_logit() {
    // The reason acceptance 2 exists: `BQ` splits into `B` and `Q`, so a
    // readout that admitted it would read `B`'s logit at its slot — and `B`
    // is a declared option of its own. The alphabet must never let two slots
    // name one token.
    let tokenizer = TableTokenizer::without(&["BQ"]);
    let split = tokenizer.encode("BQ").expect("the table splits it");
    assert_eq!(split.len(), 2, "the premise: `BQ` is two tokens here");
    let alphabet = AnswerAlphabet::from_tokenizer(&tokenizer);
    let b = alphabet
        .position("B")
        .map(|index| alphabet.tokens()[index].id)
        .expect("`B` itself is one token");
    assert_eq!(
        split[0], b,
        "the first of the two is exactly the token `B` stands for"
    );
    let ids: Vec<TokenId> = alphabet.tokens().iter().map(|token| token.id).collect();
    let distinct: std::collections::BTreeSet<TokenId> = ids.iter().copied().collect();
    assert_eq!(distinct.len(), ids.len(), "no two answer tokens share an id");
}

#[test]
fn the_alphabet_orders_single_characters_before_bigrams() {
    let alphabet = AnswerAlphabet::from_tokenizer(&TableTokenizer::without(&[]));
    let labels: Vec<&str> = alphabet.tokens().iter().map(|t| t.label.as_str()).collect();
    assert_eq!(&labels[..4], &["A", "B", "C", "D"]);
    assert_eq!(labels[26], "a");
    assert_eq!(labels[52], "0");
    assert_eq!(labels[62], "AA", "the bigrams begin where the singles end");
}

#[test]
fn the_alphabet_hands_out_exactly_the_slots_a_decision_asks_for() {
    let alphabet = AnswerAlphabet::from_tokenizer(&TableTokenizer::without(&[]));
    let three = alphabet.take(3).expect("three of many");
    assert_eq!(
        three.iter().map(|t| t.label.as_str()).collect::<Vec<_>>(),
        vec!["A", "B", "C"]
    );
    assert!(
        alphabet.take(alphabet.len() + 1).is_none(),
        "a decision with more options than the tokenizer can name is refused, not truncated"
    );
}

#[test]
fn a_tokenizer_that_names_nothing_yields_an_empty_alphabet() {
    struct Nothing;
    impl LabelTokenizer for Nothing {
        fn encode(&self, _text: &str) -> Option<Vec<TokenId>> {
            None
        }
        fn decode(&self, _ids: &[TokenId]) -> Option<String> {
            None
        }
    }
    let alphabet = AnswerAlphabet::from_tokenizer(&Nothing);
    assert!(alphabet.is_empty());
    assert!(alphabet.take(1).is_none());
}
