//! Does a decision's rendered prompt actually end where its answer is read?
//!
//! The whole readout rests on one unstated assumption: that the position
//! whose logits are gathered — the last one of the rendered prompt — is the
//! position the model would put its answer letter at. Every CPU test in
//! `decide_http.rs` runs against `SimpleTemplateProvider`, whose "template"
//! is a word hash, so none of them can see this. The real chat template can,
//! and it needs no GPU to say so.
//!
//! Two things are checked here, both against the loaded artifact's own
//! frontend:
//!
//! 1. The prompt's **last** position is the one the model would write its
//!    answer at. With thinking off this template does *not* stop at the
//!    generation opener: it appends a closed, empty think block, so the
//!    prompt runs four tokens past the opener and ends after `</think>\n\n`.
//!    That is still the position the answer goes at — the readout is right —
//!    but "a decision's prompt ends at its opener" is false, and code that
//!    assumed it would be wrong about where a checkpoint may be captured.
//! 2. The **answer boundary** holds: appending a label to the rendered text
//!    extends its tokenization by exactly that label's token, rather than
//!    re-tokenizing the tail. This is the check `classify_readout_gpu.rs`
//!    does per row and `AnswerAlphabet` deliberately does not
//!    (GitHub #237) — it is a property of the rendered prompt, not of the
//!    tokenizer alone.
//! 3. The **evidence is inside the system block**, and how much of it
//!    survives the page floor (GitHub #240). A fan-out of N questions over
//!    one `state` shares that state only through a retained prefix, and a
//!    retained prefix is cut at `floor(system_block_tokens, KV_PAGE_TOKENS)`
//!    — so a state shorter than what is left of a page is shared in full on
//!    paper and not at all in practice. Only the real tokenizer can say
//!    which, and this is the only test that has one.
//!
//! Machine-local: skips when the artifact is absent (`docs/agents/testing.md`
//! — this is CPU-only and nowhere near the forward pass, so a skip is green).

use std::path::Path;

use ignis_artifact::{FrontendSet, Reader};
use ignis_core::decision::AnswerAlphabet;
use ignis_server::artifact_template::ArtifactTemplateProvider;
use ignis_server::decide::{Evidence, OrderedValue, messages_for, prepare};
use serde_json::json;
use ignis_server::template::TemplateProvider;
use ignis_server::thinking::ThinkingOptions;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";

const BODY: &str = r#"{
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Payments, invoicing, refunds",
        "technical": "Bugs, outages, integrations",
        "sales": "Pricing, upgrades, new accounts"
      }
    }
  }
}"#;

#[test]
fn a_decisions_prompt_ends_where_its_answer_is_read() {
    let path = Path::new(ARTIFACT);
    if !path.exists() {
        eprintln!("skip: {ARTIFACT} does not exist");
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {ARTIFACT}: {e}"));
    // Two sets: the provider takes one, and the boundary check needs the
    // tokenizer beside it.
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let tokenizer_set =
        FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let tokenizer = tokenizer_set.tokenizer();
    let provider = ArtifactTemplateProvider::new(frontend);

    let alphabet = AnswerAlphabet::from_tokenizer(tokenizer);
    // Parsed from the raw text, never through a `serde_json::Value`, which
    // would have sorted the options away from their declared order.
    let request: ignis_server::decide::DecideRequest =
        serde_json::from_str(BODY).expect("the request parses");
    let encode = |text: &str| provider.encode_literal(text);
    let prepared =
        prepare(&request.questions, &alphabet, &encode, None).expect("a valid decision");

    let messages = messages_for(&Evidence::read(&request.state), &prepared[0]);
    // Exactly what the endpoint asks for.
    let thinking = ThinkingOptions {
        enable_thinking: false,
        ..ThinkingOptions::default()
    };
    let rendered = provider
        .apply_chat_template(&messages, &thinking, &[])
        .unwrap_or_else(|e| panic!("template: {}", e.message));

    let text = tokenizer
        .decode(&rendered.tokens)
        .unwrap_or_else(|e| panic!("decode the rendered prompt: {e}"));
    eprintln!("ignis decide: prompt tail = {:?}", tail(&text, 80));
    eprintln!(
        "ignis decide: {} tokens, opener at {:?}",
        rendered.tokens.len(),
        rendered.opener_tokens
    );

    // 1. The prompt ends where generation begins — after a closed think
    //    block, not at the opener. What must hold is that nothing is left
    //    *open*: an unterminated `<think>` would mean the next token is
    //    reasoning, and the readout would be gathering a reasoning trace's
    //    first token and calling it an answer.
    assert!(
        text.ends_with("</think>\n\n") || text.ends_with("<|im_start|>assistant\n"),
        "the prompt must end ready for the answer, with no reasoning block \
         left open. Tail: {:?}",
        tail(&text, 80)
    );
    let opener = rendered.opener_tokens.expect("this template reports an opener");
    assert!(
        opener <= rendered.tokens.len() as u32,
        "the opener cannot be past the prompt"
    );
    // Recorded rather than asserted equal: this is the fact that makes
    // `Request::checkpoint_point`'s length guard *not* refuse a decision's
    // capture on its own, which is why GitHub #238 refuses it explicitly
    // instead of leaning on that arithmetic.
    eprintln!(
        "ignis decide: {} tokens after the opener ({} of {})",
        rendered.tokens.len() as u32 - opener,
        opener,
        rendered.tokens.len()
    );

    // 2. The answer boundary: appending a label adds exactly its token.
    for answer in &prepared[0].answers {
        let with_label = tokenizer
            .encode(&format!("{text}{}", answer.label))
            .unwrap_or_else(|e| panic!("encode the prompt with {:?}: {e}", answer.label));
        assert_eq!(
            with_label.len(),
            rendered.tokens.len() + 1,
            "appending {:?} must extend the tokenization by exactly one token, \
             not re-tokenize the prompt's tail",
            answer.label
        );
        assert_eq!(
            with_label[..rendered.tokens.len()],
            rendered.tokens[..],
            "and must leave every earlier token alone ({:?})",
            answer.label
        );
        assert_eq!(
            with_label[rendered.tokens.len()],
            answer.id,
            "and the token it adds is the one whose logit the readout reads ({:?})",
            answer.label
        );
    }
    eprintln!(
        "ignis decide: the answer boundary holds for all {} labels",
        prepared[0].answers.len()
    );

    // 3. The evidence is in the system block — the only place a sibling
    //    question can claim it from.
    let block = rendered
        .system_block_tokens
        .expect("a decision's prompt opens with a system block");
    let head = tokenizer
        .decode(&rendered.tokens[..block as usize])
        .unwrap_or_else(|e| panic!("decode the block: {e}"));
    assert!(
        head.contains("payouts have been failing"),
        "the state is inside the system block, not merely first: {head:?}"
    );
    assert!(
        !head.contains("Which team should handle this?"),
        "and the question is not, or two questions would share nothing: {head:?}"
    );

    // How much of it a retained prefix actually keeps. Reported, not
    // asserted: the floor is a property of the state's length, and Jev's own
    // documented example is a single sentence.
    let floor = block / ignis_core::KV_PAGE_TOKENS * ignis_core::KV_PAGE_TOKENS;
    eprintln!(
        "ignis decide: system block {block} tokens, retained prefix {floor} (page {})",
        ignis_core::KV_PAGE_TOKENS
    );

    // The same decision over a state worth sharing. The sentence above is
    // Jev's documentation, not its workload — a fan-out is asked for when
    // the state is a ticket, a transcript or a document.
    let long = json!(format!(
        "{} ",
        "The customer writes at length about a payout failure that has now lasted three days, with invoice numbers, timestamps and an increasingly short temper. "
            .repeat(12)
    ));
    let messages = messages_for(&Evidence::read(&OrderedValue::from_json(&long)), &prepared[0]);
    let rendered = provider
        .apply_chat_template(&messages, &thinking, &[])
        .unwrap_or_else(|e| panic!("template: {}", e.message));
    let block = rendered
        .system_block_tokens
        .expect("a decision's prompt opens with a system block");
    let floor = block / ignis_core::KV_PAGE_TOKENS * ignis_core::KV_PAGE_TOKENS;
    eprintln!(
        "ignis decide: a {}-token prompt over a real state retains {floor} of a {block}-token block",
        rendered.tokens.len()
    );
    assert!(
        floor > 0,
        "a state of any size worth fanning out over reaches the first page boundary: block {block}, floor {floor}"
    );
    assert!(
        (rendered.tokens.len() as u32) - floor < block,
        "and what each sibling question re-prefills is its own tail, not the state"
    );
}

fn tail(text: &str, bytes: usize) -> &str {
    let start = text.len().saturating_sub(bytes);
    let start = (start..text.len())
        .find(|&i| text.is_char_boundary(i))
        .unwrap_or(text.len());
    &text[start..]
}
