//! ignis-grammar: structured output's **grammar** (GitHub #287, ADR 0089).
//!
//! The model samples freely; the host **checks** each sampled token against
//! the request's grammar before the model consumes it; only a **miss**
//! triggers a **repair**, which samples that position again restricted to the
//! grammar's **permitted set** over the whole vocabulary (a mask). This crate
//! is the grammar: a pushdown automaton over bytes for `json_object` and for
//! JSON schemas (OpenAI's strict subset, enforced exactly), with
//! - [`Vocab`]: the vocabulary as bytes and a byte trie over it, built once
//!   per loaded tokenizer;
//! - [`Grammar`]: compiled from `json_object` or a schema, or a [`Refusal`]
//!   that names its keyword or limit; [`GrammarCache`] keeps one per
//!   distinct schema, in bounded memory;
//! - [`Matcher`]: one request's state. [`Matcher::check`] and
//!   [`Matcher::accept`] take one token (the common path),
//!   [`Matcher::accept_run`] a run up to its first miss (a verify round's
//!   cut), and [`Matcher::mask`] gives the permitted set where it stands (the
//!   repair). The permitted set holds every token that keeps the answer
//!   completable: no state is a dead end.
//!
//! It is our own Rust (no grammar engine), CPU-only; `regex-automata` builds
//! the DFAs of `pattern` and `format`.

mod automaton;
mod bitset;
mod cache;
pub mod json;
mod mask;
mod number;
mod regex;
mod schema;
mod string;
mod vocab;

use std::sync::Arc;

pub use bitset::TokenSet;
pub use cache::{GrammarCache, DEFAULT_CACHE_BYTES};
pub use mask::{MASK_CACHE_ENTRIES, STR_TABLE_ENTRIES};
pub use schema::{
    Grammar, Inert, Refusal, ANNOTATIONS, DEFAULT_MAX_WS_RUN, FORMATS, MAX_ENUM_VALUES, MAX_NESTING, MAX_PROPERTIES,
    MAX_REF_CHAIN, MAX_SCHEMA_CHARS, MAX_STACK, STRICT_SUBSET,
};
pub use vocab::{Vocab, VocabEntry};

use automaton::{Config, Frame, State};

/// One request's position in its grammar.
pub struct Matcher {
    g: Arc<Grammar>,
    state: State,
    scratch: State,
    forks: Vec<Config>,
    walker: mask::Walker,
    finished: bool,
}

/// A snapshot of the request's position (its scratch is not copied).
impl Clone for Matcher {
    fn clone(&self) -> Self {
        Self { state: self.state.clone(), finished: self.finished, ..Matcher::new(self.g.clone()) }
    }
}

/// What kind of position a state is, for reporting where misses fall.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Position {
    /// Writing or about to write an object key.
    Key,
    /// Between values: braces, brackets, commas, colons.
    Structural,
    /// A value is due and has not started (its first byte decides its type).
    ValueStart,
    StringValue,
    Number,
    /// Inside `true`/`false`/`null` or an `enum`/`const`.
    Literal,
    /// After the root value: only whitespace or a stop.
    End,
}

impl Matcher {
    pub fn new(g: Arc<Grammar>) -> Self {
        let state = State::initial(g.core.root);
        Self { g, state, scratch: State::default(), forks: Vec::new(), walker: Default::default(), finished: false }
    }

    pub fn grammar(&self) -> &Arc<Grammar> {
        &self.g
    }

    /// Whether `token` is valid here. Leaves the state unchanged.
    pub fn check(&mut self, token: u32) -> bool {
        if self.finished {
            return false;
        }
        if self.g.vocab.is_stop(token) {
            return self.state.accepting();
        }
        let Some(bytes) = self.g.vocab.token_bytes(token) else { return false };
        self.scratch.clone_from(&self.state);
        for &b in bytes {
            if !self.scratch.step(&self.g.core, b, &mut self.forks) {
                return false;
            }
        }
        true
    }

    /// Check `token` and, when valid, consume it. A stop token finishes the
    /// matcher.
    pub fn accept(&mut self, token: u32) -> bool {
        if !self.check(token) {
            return false;
        }
        if self.g.vocab.is_stop(token) {
            self.finished = true;
        } else {
            std::mem::swap(&mut self.state, &mut self.scratch);
        }
        true
    }

    /// Accept `tokens` in order up to the first miss, and return how many
    /// were accepted: a verify round's licensed run is cut there, and the
    /// miss's position repaired from the state the matcher is left in.
    pub fn accept_run(&mut self, tokens: &[u32]) -> usize {
        tokens.iter().take_while(|&&t| self.accept(t)).count()
    }

    /// Whether the answer has started: anything but leading whitespace has
    /// been accepted. Until then a caller may admit another opener beside
    /// the grammar's (#592: the tool-call opener, when tools ride beside
    /// `response_format`) by checking a token the matcher refuses against it,
    /// and adding it to the mask.
    pub fn answer_started(&self) -> bool {
        self.finished || !self.state.configs.iter().all(|c| c.frames.len() == 2 && matches!(c.frames[1], Frame::Value { .. }))
    }

    /// Feed raw bytes (not a token): a test and generation helper.
    pub fn accept_bytes(&mut self, bytes: &[u8]) -> bool {
        self.scratch.clone_from(&self.state);
        for &b in bytes {
            if !self.scratch.step(&self.g.core, b, &mut self.forks) {
                return false;
            }
        }
        std::mem::swap(&mut self.state, &mut self.scratch);
        true
    }

    /// The root value is complete: a stop token is valid here.
    pub fn is_accepting(&self) -> bool {
        self.state.accepting()
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// The repair's permitted set. Returns `true` when the mask came whole
    /// from the grammar's cache.
    pub fn mask(&mut self, out: &mut TokenSet) -> bool {
        if self.finished {
            out.clear();
            return false;
        }
        self.g.mask(&self.state, out, &mut self.walker)
    }

    /// The reference the mask is tested against: [`Matcher::check`] on every
    /// id of the vocabulary.
    pub fn mask_reference(&mut self, out: &mut TokenSet) {
        out.clear();
        for id in 0..self.g.vocab.len() as u32 {
            if self.check(id) {
                out.insert(id);
            }
        }
    }

    /// Where in the document the current state sits.
    pub fn position(&self) -> Position {
        let Some(c) = self.state.configs.first() else { return Position::Structural };
        if c.accepting() {
            return Position::End;
        }
        match c.frames.last() {
            Some(Frame::Str { .. }) => match c.frames.get(c.frames.len().wrapping_sub(2)) {
                Some(Frame::FreeObj { phase, .. }) if *phase == automaton::AFTER_KEY => Position::Key,
                _ => Position::StringValue,
            },
            Some(Frame::Num { .. }) => Position::Number,
            Some(Frame::Value { .. }) => Position::ValueStart,
            Some(Frame::Lit { .. }) => match c.frames.get(c.frames.len().wrapping_sub(2)) {
                Some(Frame::Obj { phase, .. }) if *phase == automaton::AFTER_KEY => Position::Key,
                _ => Position::Literal,
            },
            Some(Frame::Obj { phase, .. } | Frame::FreeObj { phase, .. })
                if *phase == automaton::OPEN || *phase == automaton::BEFORE =>
            {
                Position::Key
            }
            _ => Position::Structural,
        }
    }

    /// How many configurations an `anyOf` keeps alive (almost always one).
    pub fn alternatives(&self) -> usize {
        self.state.configs.len()
    }

    /// Whether the last bytes consumed were whitespace (a run in progress).
    pub fn in_whitespace_run(&self) -> bool {
        self.state.configs.iter().any(|c| c.ws > 0)
    }

    /// The bytes valid next (a generation helper).
    pub fn allowed_bytes(&mut self) -> Vec<u8> {
        (0..=255u8)
            .filter(|&b| {
                self.scratch.clone_from(&self.state);
                self.scratch.step(&self.g.core, b, &mut self.forks)
            })
            .collect()
    }
}

/// A small deterministic PRNG for the generators and benches.
pub struct XorShift(pub u64);

impl XorShift {
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A random document of the grammar's language, by a random walk over the
/// bytes the automaton allows, steered toward closing once it passes `soft`
/// bytes. `Err` is a dead end: a state with no valid byte that is not
/// complete, which the automaton must never reach. `Ok(None)`: gave up at
/// `hard` bytes. A test and bench helper.
pub fn random_document(g: &Arc<Grammar>, seed: u64, soft: usize, hard: usize) -> Result<Option<Vec<u8>>, String> {
    random_walk(Matcher::new(g.clone()), seed, soft, hard)
}

/// A valid completion of `m`'s state: a random walk steered toward closing
/// from its first byte, with the same verdicts as [`random_document`]. A
/// test and bench helper.
pub fn random_completion(m: &Matcher, seed: u64, hard: usize) -> Result<Option<Vec<u8>>, String> {
    random_walk(m.clone(), seed, 0, hard)
}

fn random_walk(mut m: Matcher, seed: u64, soft: usize, hard: usize) -> Result<Option<Vec<u8>>, String> {
    let mut rng = XorShift(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut out = Vec::new();
    loop {
        if m.is_accepting() && (out.len() >= soft || rng.unit() < 0.3) {
            return Ok(Some(out));
        }
        if out.len() >= hard {
            return Ok(None);
        }
        let allowed = m.allowed_bytes();
        if allowed.is_empty() {
            return if m.is_accepting() { Ok(Some(out)) } else { Err(format!("dead end after {:?}", String::from_utf8_lossy(&out))) };
        }
        let late = out.len() >= soft;
        let weight = |b: u8| -> f64 {
            match b {
                b'"' | b'}' | b']' if late => 200.0,
                b',' if late => 0.01,
                b' ' | b'\n' | b'\t' | b'\r' => 0.05,
                0x80..=0xFF => 0.03,
                0x00..=0x1F => 0.0,
                b'\\' => 0.05,
                _ => 1.0,
            }
        };
        let total: f64 = allowed.iter().map(|&b| weight(b)).sum();
        let pick = if total <= 0.0 {
            allowed[(rng.next_u64() % allowed.len() as u64) as usize]
        } else {
            let mut r = rng.unit() * total;
            let mut chosen = allowed[allowed.len() - 1];
            for &b in &allowed {
                r -= weight(b);
                if r <= 0.0 {
                    chosen = b;
                    break;
                }
            }
            chosen
        };
        assert!(m.accept_bytes(&[pick]));
        out.push(pick);
    }
}

#[cfg(test)]
mod tests;
