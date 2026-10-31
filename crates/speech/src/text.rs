//! Text: the Qwen3 vocabulary the decoder GGUF carries, built into a BPE
//! tokenizer with the `tokenizers` crate (Q41); the official prompt (Q15);
//! and the model output's parser (Q51).
//!
//! The tokenizer is the checkpoint's: NFC, the Qwen2 split regex, byte-level
//! BPE, byte-level decoding, and the added tokens as whole tokens. The
//! prompt's special tokens are placed by id, never parsed out of text, and
//! special tokens inside a context are encoded as plain text.

use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::{Vocab, BPE};
use tokenizers::normalizers::unicode::NFC;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::{AddedToken, SplitDelimiterBehavior, Tokenizer};

use crate::gguf::{Gguf, Value};

/// The Qwen2 pre-tokenizer's split (`tokenizer.ggml.pre = qwen2`).
const QWEN2_SPLIT: &str =
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// GGUF token types: a control token is special (skipped when decoding), a user-defined one is
/// an added token that decodes as text (`<asr_text>`).
const TOKEN_CONTROL: i64 = 3;
const TOKEN_USER_DEFINED: i64 = 4;

pub struct Text {
    tok: Tokenizer,
    pub im_start: u32,
    pub im_end: u32,
    pub audio_start: u32,
    pub audio_end: u32,
    /// The audio placeholder of the official prompt; the executor puts audio embeddings there.
    pub audio_pad: u32,
    pub asr_text: u32,
    /// Decoding stops at any of these (151643 and 151645 for Qwen3-ASR).
    pub eos: Vec<u32>,
}

fn arr<'a>(g: &'a Gguf, key: &str) -> Result<&'a [Value], String> {
    g.get(key).and_then(Value::as_array).ok_or_else(|| format!("{}: no {key}", g.path().display()))
}

impl Text {
    pub fn from_gguf(g: &Gguf) -> Result<Text, String> {
        let model = g.get("tokenizer.ggml.model").and_then(Value::as_str);
        let pre = g.get("tokenizer.ggml.pre").and_then(Value::as_str);
        if model != Some("gpt2") || pre != Some("qwen2") {
            return Err(format!("{}: tokenizer {model:?}/{pre:?}, not gpt2/qwen2", g.path().display()));
        }
        let tokens: Vec<&str> =
            arr(g, "tokenizer.ggml.tokens")?.iter().map(|v| v.as_str().ok_or("a token is not a string")).collect::<Result<_, _>>()?;
        let types: Vec<i64> = match g.get("tokenizer.ggml.token_type").and_then(Value::as_array) {
            Some(a) => a.iter().map(|v| v.as_f64().unwrap_or(1.0) as i64).collect(),
            None => vec![1; tokens.len()],
        };
        let merges: Vec<(String, String)> = arr(g, "tokenizer.ggml.merges")?
            .iter()
            .map(|v| {
                let m = v.as_str().ok_or("a merge is not a string")?;
                let (a, b) = m.split_once(' ').ok_or("a merge without a space")?;
                Ok((a.to_string(), b.to_string()))
            })
            .collect::<Result<_, &str>>()?;
        // Added tokens are matched whole by the tokenizer, not by the BPE model.
        let added: Vec<usize> = (0..tokens.len()).filter(|&i| matches!(types.get(i), Some(&TOKEN_CONTROL) | Some(&TOKEN_USER_DEFINED))).collect();
        let vocab: Vocab = tokens.iter().enumerate().map(|(i, t)| (t.to_string(), i as u32)).collect();
        let bpe = BPE::builder().vocab_and_merges(vocab, merges).build().map_err(|e| format!("BPE: {e}"))?;
        let mut tok = Tokenizer::new(bpe);
        let split = Split::new(SplitPattern::Regex(QWEN2_SPLIT.into()), SplitDelimiterBehavior::Isolated, false).map_err(|e| format!("split: {e}"))?;
        tok.with_normalizer(Some(NFC));
        tok.with_pre_tokenizer(Some(Sequence::new(vec![
            PreTokenizerWrapper::Split(split),
            PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, true, false)),
        ])));
        tok.with_decoder(Some(ByteLevelDecoder::default()));
        let special: Vec<AddedToken> =
            added.iter().filter(|&&i| types[i] == TOKEN_CONTROL).map(|&i| AddedToken::from(tokens[i].to_string(), true).normalized(false)).collect();
        let user: Vec<AddedToken> =
            added.iter().filter(|&&i| types[i] == TOKEN_USER_DEFINED).map(|&i| AddedToken::from(tokens[i].to_string(), false).normalized(false)).collect();
        tok.add_special_tokens(&special);
        tok.add_tokens(&user);
        tok.set_encode_special_tokens(true);
        let id = |s: &str| tokens.iter().position(|t| *t == s).map(|i| i as u32).ok_or_else(|| format!("no {s} in the vocabulary"));
        let mut eos = vec![id("<|im_end|>")?];
        eos.extend(id("<|endoftext|>"));
        if let Some(e) = g.get("tokenizer.ggml.eos_token_id").and_then(Value::as_u64) {
            if !eos.contains(&(e as u32)) {
                eos.push(e as u32);
            }
        }
        Ok(Text {
            tok,
            im_start: id("<|im_start|>")?,
            im_end: id("<|im_end|>")?,
            audio_start: id("<|audio_start|>")?,
            audio_end: id("<|audio_end|>")?,
            audio_pad: id("<|audio_pad|>")?,
            asr_text: id("<asr_text>")?,
            eos,
        })
    }

    /// Token ids of plain text: special tokens in it are encoded as text.
    pub fn encode(&self, s: &str) -> Vec<u32> {
        match self.tok.encode(s, false) {
            Ok(e) => e.get_ids().to_vec(),
            Err(e) => panic!("encoding text: {e}"),
        }
    }

    /// Text of token ids, control tokens skipped (`<asr_text>` is kept: it is the output's marker).
    pub fn decode(&self, ids: &[u32]) -> String {
        self.tok.decode(ids, true).unwrap_or_default()
    }
}

/// The official prompt around the audio (Q15):
/// `<|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n<|audio_start|>` (the head,
/// the prefix cache's entry), the audio tokens, then
/// `<|audio_end|><|im_end|>\n<|im_start|>assistant\n` (the tail), with a forced language's
/// official prefill `language <Name><asr_text>` after it (Q6).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prompt {
    pub head: Vec<u32>,
    pub tail: Vec<u32>,
    /// Context tokens in the head (after the cap).
    pub context_tokens: usize,
    /// The context as the model reads it: its last `max_context` tokens.
    pub context: String,
}

impl Prompt {
    /// The context's last `max_context` tokens form the system turn (spec 02: the oldest are dropped).
    pub fn new(text: &Text, context: &str, max_context: usize, language: Option<&str>) -> Prompt {
        let ctx = text.encode(context);
        // The rendered template tokenizes "system\n{context}" as one span; over the cap the kept
        // tail of the context follows "system\n" on its own.
        let (system, context_tokens, read) = if ctx.len() <= max_context {
            let s = text.encode(&format!("system\n{context}"));
            (s, ctx.len(), context.to_string())
        } else {
            let mut s = text.encode("system\n");
            s.extend_from_slice(&ctx[ctx.len() - max_context..]);
            (s, max_context, text.decode(&ctx[ctx.len() - max_context..]))
        };
        let nl = text.encode("\n");
        let mut head = vec![text.im_start];
        head.extend(system);
        head.push(text.im_end);
        head.extend(&nl);
        head.push(text.im_start);
        head.extend(text.encode("user\n"));
        head.push(text.audio_start);
        let mut tail = vec![text.audio_end, text.im_end];
        tail.extend(&nl);
        tail.push(text.im_start);
        match language {
            Some(l) => {
                tail.extend(text.encode(&format!("assistant\nlanguage {l}")));
                tail.push(text.asr_text);
            }
            None => tail.extend(text.encode("assistant\n")),
        }
        Prompt { head, tail, context_tokens, context: read }
    }

    /// Text tokens around the audio.
    pub fn text_tokens(&self) -> usize {
        self.head.len() + self.tail.len()
    }
}

/// A transcript as the model gave it (Q51).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Parsed {
    /// The model's label, `None` when it said `language None` or gave none.
    pub language: Option<String>,
    pub text: String,
}

/// Splits `language X<asr_text>text` into X and the text, as the checkpoint's own parser does:
/// `language None` followed by text is speech (the text is kept), `language None` alone is no
/// speech (an empty transcript), and output without the marker is all text. A forced language's
/// prefill is not part of the output, so its `forced` name is the label. A marker the model
/// writes again after the first is no text (on noise it can write `<asr_text>` twice, #362).
pub fn parse(output: &str, forced: Option<&str>) -> Parsed {
    const MARKER: &str = "<asr_text>";
    let s = output.trim();
    if let Some(l) = forced {
        let t = s.split_once(MARKER).map_or(s, |(_, t)| t);
        return Parsed { language: Some(l.to_string()), text: t.replace(MARKER, "").trim().to_string() };
    }
    let Some((prefix, text)) = s.split_once(MARKER) else {
        return Parsed { language: None, text: s.to_string() };
    };
    let prefix = prefix.trim();
    let text = text.replace(MARKER, "").trim().to_string();
    if prefix.eq_ignore_ascii_case("language none") {
        return Parsed { language: None, text };
    }
    let first = prefix.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("");
    let language = match first.get(..9) {
        Some(p) if p.eq_ignore_ascii_case("language ") => Some(first[9..].trim().to_string()),
        _ => Some(first.to_string()),
    }
    .filter(|l| !l.is_empty());
    Parsed { language, text }
}

/// The context the speech model is told (spec speech/02 §`keywords[]`): the prompt, then the
/// keywords as one line, joined by commas, the pieces [`copies_context`] reads a near copy off.
pub fn context_of(prompt: Option<&str>, keywords: &[String]) -> String {
    let prompt = prompt.map(str::trim).filter(|prompt| !prompt.is_empty());
    let line = keywords.iter().map(|keyword| keyword.trim()).filter(|keyword| !keyword.is_empty()).collect::<Vec<_>>().join(", ");
    match (prompt, line.is_empty()) {
        (Some(prompt), false) => format!("{prompt}\n{line}"),
        (Some(prompt), true) => prompt.to_owned(),
        (None, false) => line,
        (None, true) => String::new(),
    }
}

/// What [`copies_context`] takes for a copy of the context: `run`, the words in a row of one line
/// a copy repeats (`speech.context_copy_words`; 0 turns the guard off), and `edits`, the words a
/// near copy may change, add or drop from a whole line or piece (`speech.context_copy_edits`; 0
/// keeps the exact rule). One type, so the two are never passed in each other's place.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContextCopy {
    pub run: usize,
    pub edits: usize,
}

impl ContextCopy {
    /// The server's defaults, which its config fields take.
    pub const DEFAULT: ContextCopy = ContextCopy { run: 3, edits: 2 };
}

/// Whether a transcript only repeats its context (#362). Given little or no speech, a VAD's cut
/// of room noise or a backchannel, the model can write a line of its system turn instead of what
/// it heard: the topic, the keyword line. A transcript is taken for such a copy when its words
/// (letters and digits, lower-cased) are a whole line of the context of two words or more, or
/// `run` or more of one line's words in a row (`speech.context_copy_words`, 3; 0 turns the guard
/// off); a line's leading one-word `Label:` is read with and without it, never a colon further
/// on (`10:30`). A single word never is: a name the context lists is said alone too.
///
/// The model also writes a line with a word or two changed (`migrate` for `migrazione`, the
/// topic turned into an imperative, #442): a transcript at most `edits` words (changed, added
/// or dropped) from a whole line, or from a whole piece of one between commas or sentence marks
/// (a keyword), is a copy too when the words it keeps of it are `run` or more, those edits taken
/// off the shorter of the two (`speech.context_copy_edits`; 0 keeps the exact rule). That line is
/// read without its label only, which is no word a transcript may change.
pub fn copies_context(context: &str, transcript: &str, copy: ContextCopy) -> bool {
    let ContextCopy { run, edits } = copy;
    let said = words(transcript);
    if run == 0 || said.len() < 2 {
        return false;
    }
    let unlabelled = |line: &str| line.split_once(':').filter(|(label, _)| label.chars().all(char::is_alphabetic) && !label.is_empty()).map(|(_, rest)| rest.to_string());
    let near = |line: &str| {
        edits > 0
            && said.len() >= run
            && std::iter::once(line).chain(line.split([',', ';', '.', '!', '?'])).any(|piece| {
                let piece = words(piece);
                let changed = word_edits(&said, &piece, edits);
                changed <= edits && said.len().min(piece.len()).saturating_sub(changed) >= run
            })
    };
    context.lines().any(|line| {
        let exact = std::iter::once(line.to_string()).chain(unlabelled(line)).any(|line| {
            let whole = words(&line);
            said == whole || (said.len() >= run && whole.windows(said.len()).any(|run| run == said))
        });
        // A label is no word a near copy may change: `Topic` would buy a short topic an edit.
        exact || near(&unlabelled(line).unwrap_or_else(|| line.to_string()))
    })
}

fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(str::to_lowercase).collect()
}

/// The words changed, added or dropped from `a` to `b` (Levenshtein over words), or more than
/// `bound` when their lengths alone differ by more.
fn word_edits(a: &[String], b: &[String], bound: usize) -> usize {
    if a.len().abs_diff(b.len()) > bound {
        return bound + 1;
    }
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.iter().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, y) in b.iter().enumerate() {
            let next = (row[j + 1] + 1).min(row[j] + 1).min(diagonal + usize::from(x != y));
            diagonal = row[j + 1];
            row[j + 1] = next;
        }
    }
    row[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The guard at the run of three words, a near copy changing up to `edits` of them.
    fn edits(edits: usize) -> ContextCopy {
        ContextCopy { run: 3, edits }
    }

    #[test]
    fn the_parser_keeps_the_label_and_the_text() {
        let p = parse("language Italian<asr_text>Ieri Martelli ha giurato.", None);
        assert_eq!(p, Parsed { language: Some("Italian".into()), text: "Ieri Martelli ha giurato.".into() });
        // `None` followed by text is speech: the text is kept, with no label.
        assert_eq!(parse("language None<asr_text>uh, yes", None), Parsed { language: None, text: "uh, yes".into() });
        // `None` alone is no speech: an empty transcript.
        assert_eq!(parse("language None<asr_text>", None), Parsed { language: None, text: String::new() });
        assert_eq!(parse("  ", None), Parsed { language: None, text: String::new() });
        // No marker: all of it is text.
        assert_eq!(parse("just words", None), Parsed { language: None, text: "just words".into() });
        // A forced language: the output has no header, and the forced name is the label.
        assert_eq!(parse("Buongiorno a tutti.", Some("Italian")), Parsed { language: Some("Italian".into()), text: "Buongiorno a tutti.".into() });
        assert_eq!(parse("language English<asr_text>", None), Parsed { language: Some("English".into()), text: String::new() });
        // The marker written twice, as on noise: no text, never the marker's spelling.
        assert_eq!(parse("language None<asr_text><asr_text>", None), Parsed { language: None, text: String::new() });
        assert_eq!(parse("<asr_text>", Some("Italian")), Parsed { language: Some("Italian".into()), text: String::new() });
    }

    #[test]
    fn the_context_is_the_prompt_then_the_keywords_as_one_line() {
        let keywords = vec!["ignis".to_owned(), " Qwen3 ".to_owned(), String::new()];
        assert_eq!(context_of(Some("A talk about GPUs."), &keywords), "A talk about GPUs.\nignis, Qwen3");
        assert_eq!(context_of(None, &keywords), "ignis, Qwen3");
        assert_eq!(context_of(Some("  just a prompt "), &[]), "just a prompt");
        assert_eq!(context_of(None, &[]), "");
        assert_eq!(context_of(Some(" "), &[]), "");
    }

    /// The Playground's context, shaped as in #362's probe: the language hint, the topic, the keyword line.
    const CONTEXT: &str = "Languages: Italian.\nTopic: migrazione della pila software verso GitOps.\n\
        pila software, Argo CD, Forgejo Enterprise, GitOps, Forgejo Apps, Tekton, pull mode, OKD, Fedora, OCM, Jenkins, Helm Chart, Harbor, Gen Two, Marta, Paolo";

    /// #442's `it-topic4` context: a topic of four words and sixteen terms, half of them phrases.
    const TOPIC4: &str = "Languages: Italian.\nTopic: revisione settimanale del rilascio\n\
        pila software, Argo CD, rilascio in produzione, revisione del codice, chiusura del ticket aperto, \
        aggiornamento della documentazione tecnica, ambiente di collaudo, Helm Chart, pipeline di integrazione continua, \
        Forgejo, Jenkins, pull request, Marta, Paolo, Gen Two, OKD";

    #[test]
    fn a_line_said_with_a_word_in_front_or_inflected_is_one_edit_off_it() {
        // Italian as people say it (#442's review): a chair opening the item, a term after its
        // preposition, the topic with its article, a verb conjugated. Each is one edit off a line
        // or piece of four words and keeps three: the exact rule keeps it, the near rule takes it
        // for a copy from one edit.
        let topic_verb = "Languages: Italian.\nTopic: aggiornare la documentazione tecnica\npila software, Argo CD, Marta";
        for (context, said) in [
            (TOPIC4, "Allora, revisione settimanale del rilascio."),
            (TOPIC4, "Nella pipeline di integrazione continua."),
            (TOPIC4, "la revisione settimanale del rilascio"),
            (topic_verb, "Aggiorniamo la documentazione tecnica."),
        ] {
            assert!(!copies_context(context, said, edits(0)), "{said}");
            assert!(copies_context(context, said, edits(1)), "{said}");
        }
        // The copy the model wrote of that topic on overlapped speech (AMI IS1009a, 358.3 s) is
        // the same words: no rule on the text tells the two apart.
        assert!(copies_context(TOPIC4, "La revisione settimanale del rilascio", edits(1)));
    }

    #[test]
    fn a_transcript_that_only_repeats_a_line_of_the_context_is_a_copy() {
        // The keyword line whole, as the model wrote it on a pause.
        let line = CONTEXT.lines().last().unwrap();
        assert!(copies_context(CONTEXT, line, edits(2)));
        // The topic, with or without its label, its period or not, and its head.
        assert!(copies_context(CONTEXT, "migrazione della pila software verso GitOps.", edits(2)));
        assert!(copies_context(CONTEXT, "Topic: migrazione della pila software verso GitOps", edits(2)));
        assert!(copies_context(CONTEXT, "Migrazione della pila", edits(2)));
        // Three of the keyword line's words in a row.
        assert!(copies_context(CONTEXT, "Jenkins, Helm Chart", edits(2)));
        // A context cut by its token cap: the lines the model read.
        assert!(copies_context("ra della pila software verso GitOps.\npila software, Argo CD", "pila software, Argo CD", edits(2)));
    }

    #[test]
    fn a_line_of_the_context_with_one_word_changed_is_a_copy_too() {
        // #442: on a backchannel two speakers overlapped (AMI ES2004a, 603.6 s), the model wrote
        // the topic with its first word changed; under two other contexts the same audio was
        // "Right. Yeah.".
        assert!(copies_context(CONTEXT, "migrate della pila software verso GitOps", edits(1)));
        // A word dropped or added.
        assert!(copies_context(CONTEXT, "migrazione pila software verso GitOps", edits(1)));
        assert!(copies_context(CONTEXT, "la migrazione della pila software verso GitOps", edits(1)));
        // A keyword of four words with one inflected, or turned into an imperative.
        let context = "Topic: il rilascio.\nArgo CD, chiusura del ticket aperto, aggiornare la documentazione tecnica, Marta";
        assert!(copies_context(context, "Chiusura dei ticket aperto.", edits(1)));
        assert!(copies_context(context, "Aggiorna la documentazione tecnica.", edits(1)));
    }

    #[test]
    fn the_words_a_copy_may_change_are_the_servers_and_0_keeps_the_exact_rule() {
        assert!(!copies_context(CONTEXT, "migrate della pila software verso GitOps", edits(0)));
        // Two words changed are a copy only when two may be.
        assert!(!copies_context(CONTEXT, "migrate della pila software verso Git", edits(1)));
        assert!(copies_context(CONTEXT, "migrate della pila software verso Git", edits(2)));
    }

    #[test]
    fn a_near_copy_keeps_run_words_of_its_line_whatever_the_words_it_may_change() {
        // The topic turned into an imperative: two words changed, four of six kept (AMI ES2004a,
        // 603.6 s, the same backchannel under a context of forty terms).
        assert!(copies_context(CONTEXT, "migrate la pila software verso GitOps.", edits(2)));
        assert!(!copies_context(CONTEXT, "migrate la pila software verso GitOps.", edits(1)));
        // A keyword of four one word off is a copy whatever the words that may change...
        let context = "Topic: il rilascio.\nArgo CD, chiusura del ticket aperto, Marta";
        assert!(copies_context(context, "Chiusura dei ticket aperto.", edits(2)));
        // ...but two words changed of four keep two: no copy.
        assert!(!copies_context(context, "chiusura dei ticket aperti", edits(2)));
    }

    #[test]
    fn a_short_keyword_with_a_word_more_or_changed_is_no_copy() {
        // A near copy keeps `run` of its words, so a line or keyword of `run` words or fewer is
        // matched only exactly, and one of `run` + k words takes at most k edits, whatever
        // `edits` allows: "la revisione del codice" is said for real.
        let context = "Topic: il rilascio.\nArgo CD, revisione del codice, Marta";
        assert!(!copies_context(context, "la revisione del codice", edits(1)));
        assert!(!copies_context(context, "revisione del progetto", edits(1)));
        assert!(copies_context(context, "revisione del codice", edits(1)));
        // A transcript of fewer words than that is no near copy either.
        assert!(!copies_context(CONTEXT, "migrazione pila software", edits(1)));
    }

    #[test]
    fn a_lines_label_is_no_word_a_near_copy_may_change() {
        // Read with its label, a three-word topic is four words, and "la revisione del codice",
        // said for real, one word off it: near copies are read off the line without its label.
        for n in 0..=3 {
            assert!(!copies_context("Topic: revisione del codice", "la revisione del codice", edits(n)), "{n} edits");
        }
        // The exact rule still reads the line with its label and without.
        assert!(copies_context("Topic: revisione del codice", "Topic: revisione del codice", edits(0)));
        assert!(copies_context("Topic: revisione del codice", "revisione del codice", edits(0)));
    }

    #[test]
    fn the_run_a_copy_takes_is_the_servers_and_0_turns_the_guard_off() {
        assert!(!copies_context(CONTEXT, "Jenkins, Helm Chart", ContextCopy { run: 4, edits: 2 }));
        assert!(copies_context(CONTEXT, "migrazione della", ContextCopy { run: 2, edits: 2 }));
        assert!(!copies_context(CONTEXT, CONTEXT.lines().last().unwrap(), ContextCopy { run: 0, edits: 2 }));
        // One word is never a copy, whatever the run.
        assert!(!copies_context(CONTEXT, "Marta", ContextCopy { run: 1, edits: 2 }));
    }

    #[test]
    fn only_a_leading_one_word_label_is_read_off_a_line() {
        // The rest of a line after a colon further on is no line of its own.
        let context = "La riunione è alle 10:30 sharp";
        assert!(!copies_context(context, "30 sharp", edits(2)));
        assert!(copies_context(context, "riunione è alle", edits(2)));
        assert!(copies_context("Topic: rilascio settimanale", "rilascio settimanale", edits(2)));
    }

    #[test]
    fn a_transcript_that_says_something_of_its_own_is_no_copy() {
        // Two words of a longer line, and one word, are said for real too.
        assert!(!copies_context(CONTEXT, "migrazione della", edits(2)));
        assert!(!copies_context(CONTEXT, "Marta?", edits(2)));
        assert!(!copies_context(CONTEXT, "Italian.", edits(2)));
        // A sentence that names what the context lists.
        assert!(!copies_context(CONTEXT, "Marta ha aggiornato la Helm Chart ieri.", edits(2)));
        assert!(!copies_context(CONTEXT, "la migrazione della pila software parte lunedì", edits(2)));
        assert!(!copies_context("", "migrazione della pila software", edits(2)));
    }
}
