//! A request's tool calls held to its tools (GitHub #611, spec #582, ADR
//! 0094): the **grammar** of the content channel of a request that has a
//! **held tool**, read as bytes.
//!
//! The model writes its calls in its own dialect ([`crate::call`]) and in
//! its own order; the grammar checks each sampled token and the scheduler
//! repairs a miss, as for structured output (ADR 0089). Where it holds:
//! - **Outside a call**, any text. `<tool_call>` opens a call, whether the
//!   model writes the added token or its bytes: the text the call parser
//!   reads is what arms the grammar. A stop passes here.
//! - **The call's head**: whitespace, `<function=`, the name of a tool the
//!   request declares, `>`.
//! - **A tool that is not held** keeps free arguments: its call runs to the
//!   first `</tool_call>`, the parser's first-occurrence rule, and a stop
//!   passes there too (the call is cut, as today).
//! - **A held tool's call** is held to its parameters: each value ends where
//!   [`call::line`] says the call continues -- the one rule the parser reads
//!   by, called here on the bytes since each undecided line start -- so a
//!   parameter is opened at most once, in the model's order, and the
//!   function closes only once every required parameter is written. What
//!   precedes the first parameter is whitespace. A stop is a miss.
//! - **The confirmed close** returns to free text, where a further call
//!   opens the same way.
//!
//! A held value is the dialect's raw text, one leading and one trailing
//! newline aside (the parser trims them), read into the arguments as its
//! [`call::Reading`] says (#612):
//! - a free string is any UTF-8 text; with a required parameter still
//!   missing, a line of it may not start `</function`, and a `</parameter>`
//!   line may not be followed by `</` (the two miss points the spike found on
//!   the 27B's tokenizer: the close it is steered away from);
//! - a string with an `enum`, a `const`, a `pattern` or a `format` is held
//!   to it, written on one line that does not start with one of the
//!   dialect's own tags (`<parameter=`, `</parameter>`, `</function>`), so
//!   where it ends is never in doubt; a value outside that is refused when
//!   the request arrives, by name (`no_writable_value`);
//! - a string that may be `null` is either, and its text `null` is `null`:
//!   such a value cannot be the string `null`;
//! - a value of another type (an integer, a number, a boolean, an object, an
//!   array), or of a union no raw text holds (a string beside another type),
//!   is JSON held to its subschema by the schema's own automaton, from the
//!   value's first byte (no whitespace before it but the trimmed newline);
//!   whitespace may follow it, then the call's next step at a line's start.
//!
//! A nullable parameter is not required: a call may leave it out, and its
//! arguments then carry `null` for it ([`call::Parameters::reading`]).
//!
//! Whitespace between a step's tags, before `<function=` and before the
//! first parameter is bounded ([`crate::GrammarLimits::max_whitespace`]):
//! the byte past the bound is a miss. Added tokens are read as the text the
//! server's decoder gives them ([`crate::GrammarCache::with_added_texts`]);
//! a column with no text passes outside a held call and is a miss in one.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::automaton::{accepts_value, Config, Frame, State};
use crate::call::{self, Line, Parameters, Reading, Step, CALL_CLOSE, CALL_OPEN, FUNCTION_CLOSE, FUNCTION_OPEN, PARAMETER_CLOSE, PARAMETER_OPEN};
use crate::json::Json;
use crate::regex::{ByteDfa, DEAD, FREE};
use crate::schema::{Grammar, LitTrie, Node, Refusal, NO_RE};
use crate::vocab::{Trie, Vocab};
use crate::TokenSet;

pub(crate) mod normalize;

/// A function tool the request declares, as the grammar is given it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeclaredTool {
    pub name: String,
    pub hold: ToolHold,
}

/// Whether a declared tool is held, and to what. A schema is the tool's
/// `parameters` as sent: their key order is the order its arguments are
/// delivered in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolHold {
    /// A tool that is not held, whose arguments stay free.
    Free,
    /// A **strict tool**'s schema, refused outside OpenAI's strict subset
    /// and where the dialect cannot hold it.
    Strict(String),
    /// A tool sent without `strict` where the surface serves OpenAI's
    /// default, the Responses API's (GitHub #614): its schema normalized into
    /// strict form, every object closed and its optional parameters left out
    /// when the call leaves them out ([`normalize::strict_form`]), and held.
    /// A schema that does not normalize, or a tool the dialect cannot hold,
    /// leaves the tool free: never refused.
    Normalized(String),
}

/// A held tool's compiled `parameters` and how the call may leave them out.
pub(crate) struct HeldSchema {
    pub grammar: Arc<Grammar>,
    /// The parameters the client left optional, which a normalized tool's
    /// call may leave out and whose arguments then omit them.
    pub optional: Vec<String>,
    /// A normalized tool: what the dialect cannot hold leaves it free.
    pub normalized: bool,
}

/// Which of a tool's fields a refusal is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolField {
    /// `function.name`.
    Name,
    /// `function.parameters`.
    Parameters,
}

/// Why a request's tools cannot be held: the tool (its place in the
/// request's `tools`), the field, and the reason by name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolRefusal {
    pub tool: usize,
    pub field: ToolField,
    pub refusal: Refusal,
}

/// The texts of a tokenizer's added tokens and the columns with none, read
/// once per vocabulary.
pub(crate) struct Controls {
    /// An added token's id, and the text the decoder gives it.
    texts: HashMap<u32, Vec<u8>>,
    /// The ids with no bytes at all: the stops, and columns no token fills.
    opaque: Vec<u32>,
    /// The tokens whose bytes hold the opener's with more after it: the only
    /// draws free text can miss on.
    opener_inside: Vec<u32>,
}

impl Controls {
    pub(crate) fn new(vocab: &Vocab, added: &[(u32, Vec<u8>)]) -> Self {
        let texts: HashMap<u32, Vec<u8>> = added
            .iter()
            .filter(|(id, text)| (*id as usize) < vocab.len() && vocab.token_bytes(*id).is_none() && !vocab.is_stop(*id) && !text.is_empty())
            .cloned()
            .collect();
        let opener = CALL_OPEN.as_bytes();
        let mut opaque = Vec::new();
        let mut opener_inside = Vec::new();
        for id in 0..vocab.len() as u32 {
            let bytes = vocab.token_bytes(id).or_else(|| texts.get(&id).map(Vec::as_slice));
            match bytes {
                None => opaque.push(id),
                Some(bytes) => {
                    if bytes.windows(opener.len()).position(|w| w == opener).is_some_and(|at| at + opener.len() < bytes.len()) {
                        opener_inside.push(id);
                    }
                }
            }
        }
        Self { texts, opaque, opener_inside }
    }

    fn bytes<'a>(&'a self, vocab: &'a Vocab, id: u32) -> Option<&'a [u8]> {
        vocab.token_bytes(id).or_else(|| self.texts.get(&id).map(Vec::as_slice))
    }

}

/// What a held tool's declared parameter's value is held to.
enum Slot {
    /// What precedes the first parameter: whitespace.
    Junk,
    /// A string with no `enum`, `pattern` or `format`: any UTF-8 text.
    Free,
    /// A string held to a `pattern` or a `format`: its DFA (`re`, in the
    /// tool's own grammar), which states can still reach a match on one
    /// line, and the same for each prefix of a dialect tag the value may
    /// start with; with `null`, the text `null` too.
    Pattern { re: u32, coacc: Vec<bool>, prefixes: Vec<(u8, bool)>, null: bool },
    /// A string held to an `enum` or a `const`: the values that can be
    /// written, raw (`null` among them where the value may be null).
    Enum(LitTrie),
    /// JSON held to `node`, the parameter's subschema in the tool's own
    /// grammar (GitHub #612).
    Json { node: u32 },
}

/// A held tool: what the confirmed close knows of it, a slot per declared
/// parameter, and its compiled schema (the DFAs).
struct Held {
    parameters: Parameters,
    slots: Vec<Slot>,
    grammar: Arc<Grammar>,
}

/// The value [`Value::slot`] holds before the first parameter.
const JUNK: u32 = u32::MAX;
static JUNK_SLOT: Slot = Slot::Junk;

/// The grammar of one request's tool calls: one per distinct set of
/// declared tools, shared by every request that sends it
/// ([`crate::GrammarCache::tool_calls`]).
pub struct ToolCallGrammar {
    vocab: Arc<Vocab>,
    controls: Arc<Controls>,
    /// The callable names (none holding `>`), each leading to its tool.
    names: LitTrie,
    name_tools: HashMap<u32, u32>,
    callable: usize,
    /// Every declared tool, by its place in the request; `None` where not held.
    held: Vec<Option<Held>>,
    tool_names: Vec<String>,
    max_ws: u8,
    memo: Mutex<Memo>,
    memo_budget: usize,
    tables_bytes: usize,
    first_draw_misses: bool,
}

impl std::fmt::Debug for ToolCallGrammar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolCallGrammar").field("tools", &self.tool_names).field("callable", &self.callable).finish_non_exhaustive()
    }
}

#[derive(Default)]
struct Memo {
    masks: HashMap<CallState, Arc<TokenSet>>,
    bytes: usize,
}

/// Where the reading stands, between two bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum At {
    /// Free text; `kmp` bytes of the opener matched.
    Free { kmp: u8 },
    /// After the opener: `ws` bytes of whitespace.
    Open { ws: u8 },
    /// `at` bytes of `<function=` read.
    FunctionTag { at: u8 },
    /// Inside the name: a node of the name trie.
    Name { node: u32 },
    /// A call to a tool that is not held: `kmp` bytes of `</tool_call>`
    /// matched.
    Body { kmp: u8 },
    /// A held call, reading a value.
    Value(Value),
}

impl Default for At {
    fn default() -> Self {
        At::Free { kmp: 0 }
    }
}

/// A held call's value being read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Value {
    tool: u32,
    /// The declared parameter's place, or [`JUNK`].
    slot: u32,
    /// Nothing read yet.
    lead: bool,
    /// UTF-8 continuation bytes owed (`string.rs`'s encoding).
    utf8: u8,
    /// A held string's DFA state or enum trie node ([`DEAD`] once its text
    /// can no longer be the value's).
    cs: u32,
    /// How much of a dialect tag the value's text has started with.
    lit: u8,
    /// A held string's line is over: the next line must be a step.
    terminated: bool,
    /// The two miss points' progress (see `free_misses`).
    r1: u8,
    r2: u8,
    /// Whitespace before the first parameter, or after a typed value's JSON.
    ws: u8,
    /// How much of `null` a held string that may be null has read, or
    /// [`OFF`].
    nul: u8,
}

/// A matcher's position in a [`ToolCallGrammar`].
#[derive(Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct CallState {
    at: At,
    /// A held call's parameters written, by their place.
    written: Vec<bool>,
    /// A held value's bytes since its earliest undecided line start.
    pending: Vec<u8>,
    /// The undecided line starts in `pending`, in order: the offset, shifted
    /// past the flags [`END_OK`] and [`MUST_STEP`].
    starts: Vec<u32>,
    /// A typed value's JSON as the schema's automaton has read it: no
    /// configuration once the text can no longer be JSON of its subschema,
    /// and none outside a typed value.
    json: State,
}

impl Clone for CallState {
    fn clone(&self) -> Self {
        Self { at: self.at, written: self.written.clone(), pending: self.pending.clone(), starts: self.starts.clone(), json: self.json.clone() }
    }
    fn clone_from(&mut self, o: &Self) {
        self.at = o.at;
        self.written.clone_from(&o.written);
        self.pending.clone_from(&o.pending);
        self.starts.clone_from(&o.starts);
        self.json.clone_from(&o.json);
    }
}

/// The value may end at this line start.
const END_OK: u32 = 1;
/// This line must be a step: a held string's line is over.
const MUST_STEP: u32 = 2;
const FLAGS: u32 = 2;

/// A tracker that no longer matches.
const OFF: u8 = u8::MAX;
/// `lit`: the value's text is no tag's prefix.
const ESCAPED: u8 = u8::MAX;
/// `lit`: the value's text starts with a whole tag: it cannot be the value.
const TAG: u8 = u8::MAX - 1;
/// `lit` before the first byte: every tag still possible.
const LIT_START: u8 = 0b111;
/// The tags a held string may not start with: a line starting so would be
/// read as the call's next step.
const TAGS: [&str; 3] = [PARAMETER_OPEN, PARAMETER_CLOSE, FUNCTION_CLOSE];
const FUNCTION_LINE: &[u8] = b"</function";
const NULL: &[u8] = b"null";

/// The next `nul` after `b`.
#[inline]
fn null_step(nul: u8, b: u8) -> u8 {
    match NULL.get(nul as usize) {
        Some(&c) if c == b => nul + 1,
        _ => OFF,
    }
}

/// Whether a typed value's JSON has read a byte other than whitespace: its
/// root is no longer the value expected.
fn json_begun(json: &State) -> bool {
    !matches!(json.configs.as_slice(), [c] if c.frames.len() == 2 && matches!(c.frames[1], Frame::Value { .. }))
}

/// A typed value's JSON once whole and followed by whitespace: one
/// configuration, done, whose whitespace the call counts from here.
fn json_done(json: &mut State) {
    json.configs.retain(Config::accepting);
    json.configs.truncate(1);
    json.configs[0].ws = 0;
}

#[inline]
fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\r' | b'\n')
}

/// The next `lit` after `b`.
fn lit_step(lit: u8, b: u8) -> u8 {
    if lit == ESCAPED || lit == TAG {
        return lit;
    }
    let (len, which) = ((lit >> 3) as usize, lit & 0b111);
    let mut next = 0;
    let mut whole = false;
    for (i, tag) in TAGS.iter().enumerate() {
        let tag = tag.as_bytes();
        if which & (1 << i) != 0 && tag.get(len) == Some(&b) {
            next |= 1 << i;
            whole |= tag.len() == len + 1;
        }
    }
    match (next, whole) {
        (0, _) => ESCAPED,
        (_, true) => TAG,
        _ => (((len + 1) as u8) << 3) | next,
    }
}

/// `tag` matched to `k` bytes, then `b`: tags whose first byte occurs only
/// there.
#[inline]
fn kmp(tag: &[u8], k: u8, b: u8) -> u8 {
    if tag[k as usize] == b {
        k + 1
    } else if tag[0] == b {
        1
    } else {
        0
    }
}

/// UTF-8 after `b`, with `debt` continuation bytes owed (`string.rs`'s
/// encoding), or `None` where `b` breaks it.
#[inline]
pub(crate) fn utf8_next(debt: u8, b: u8) -> Option<u8> {
    if debt != 0 {
        let ok = match debt {
            3 => (0xA0..=0xBF).contains(&b),
            4 => (0x80..=0x9F).contains(&b),
            6 => (0x90..=0xBF).contains(&b),
            7 => (0x80..=0x8F).contains(&b),
            _ => (0x80..=0xBF).contains(&b),
        };
        return ok.then_some(match debt {
            1 => 0,
            2..=4 => 1,
            _ => 2,
        });
    }
    match b {
        0x00..=0x7F => Some(0),
        0xC2..=0xDF => Some(1),
        0xE0 => Some(3),
        0xE1..=0xEC | 0xEE..=0xEF => Some(2),
        0xED => Some(4),
        0xF0 => Some(6),
        0xF1..=0xF3 => Some(5),
        0xF4 => Some(7),
        _ => None,
    }
}

fn refusal(code: &'static str, keyword: &str, path: &str, detail: impl Into<String>) -> Refusal {
    Refusal { code, keyword: keyword.to_string(), path: path.to_string(), detail: detail.into() }
}

impl Slot {
    fn empty_ok(&self, g: &Held) -> bool {
        match self {
            Slot::Junk | Slot::Free => true,
            Slot::Pattern { re, .. } => {
                let dfa = &g.grammar.core.dfas[*re as usize];
                pattern_accepts(dfa, dfa.start)
            }
            Slot::Enum(trie) => trie.terminal(0),
            Slot::Json { .. } => false,
        }
    }

    fn start(&self, g: &Held) -> u32 {
        match self {
            Slot::Pattern { re, .. } => g.grammar.core.dfas[*re as usize].start,
            _ => 0,
        }
    }

    fn next(&self, g: &Held, cs: u32, b: u8) -> u32 {
        if cs == DEAD {
            return DEAD;
        }
        match self {
            Slot::Pattern { re, .. } => g.grammar.core.dfas[*re as usize].next(cs, b),
            Slot::Enum(trie) => trie.child(cs, b).unwrap_or(DEAD),
            _ => cs,
        }
    }

    /// Whether a held string's text, read whole, is a value: one of its own,
    /// or, where the value may be null, `null`.
    fn accepts(&self, g: &Held, v: &Value) -> bool {
        let own = v.cs != DEAD
            && v.lit != TAG
            && match self {
                Slot::Pattern { re, .. } => pattern_accepts(&g.grammar.core.dfas[*re as usize], v.cs),
                Slot::Enum(trie) => trie.terminal(v.cs),
                _ => true,
            };
        own || v.nul as usize == NULL.len()
    }

    /// Whether a held string's text, read so far, can still become a value
    /// on its one line.
    fn text_alive(&self, v: &Value) -> bool {
        let own = v.cs != DEAD
            && v.lit != TAG
            && match self {
                Slot::Pattern { coacc, .. } if v.lit == ESCAPED => coacc[v.cs as usize],
                Slot::Pattern { prefixes, .. } => prefixes.iter().any(|&(lit, alive)| lit == v.lit && alive),
                _ => true,
            };
        own || v.nul != OFF
    }
}

impl Slot {
    /// The slot's own tables (a pattern's DFA is its schema grammar's).
    fn heap_bytes(&self) -> usize {
        64 + match self {
            Slot::Pattern { coacc, prefixes, .. } => coacc.len() + prefixes.len() * 2,
            Slot::Enum(trie) => trie.strings().iter().map(|s| s.len() * 16).sum(),
            Slot::Junk | Slot::Free | Slot::Json { .. } => 0,
        }
    }
}

fn pattern_accepts(dfa: &ByteDfa, s: u32) -> bool {
    s == FREE || dfa.accepts_at_end(s)
}

/// A pattern's slot: which DFA states reach a match over bytes other than a
/// newline, and for each prefix of a dialect tag the value may start with,
/// whether a value starting so can still be written; with `null`, the text
/// `null` is a value too. `None` when no value can be.
fn pattern_slot(dfa: &ByteDfa, re: u32, null: bool) -> Option<Slot> {
    let n = dfa.state_count();
    let mut back: Vec<Vec<u32>> = vec![Vec::new(); n];
    for s in 0..n as u32 {
        let mut seen: Vec<u32> = Vec::new();
        for b in 0..=255u8 {
            let t = dfa.next(s, b);
            if b != b'\n' && t != DEAD && !seen.contains(&t) {
                seen.push(t);
                back[t as usize].push(s);
            }
        }
    }
    let mut coacc = vec![false; n];
    let mut queue: Vec<u32> = (0..n as u32).filter(|&s| pattern_accepts(dfa, s)).collect();
    for &s in &queue {
        coacc[s as usize] = true;
    }
    while let Some(t) = queue.pop() {
        for &s in &back[t as usize] {
            if !coacc[s as usize] {
                coacc[s as usize] = true;
                queue.push(s);
            }
        }
    }
    // The tag prefixes, with the DFA state each leaves the value in.
    fn alive(dfa: &ByteDfa, coacc: &[bool], lit: u8, s: u32, out: &mut Vec<(u8, bool)>) -> bool {
        let mut any = pattern_accepts(dfa, s);
        for b in 0..=255u8 {
            let t = dfa.next(s, b);
            if b == b'\n' || t == DEAD {
                continue;
            }
            let next = lit_step(lit, b);
            let ok = match next {
                ESCAPED => coacc[t as usize],
                TAG => false,
                _ => alive(dfa, coacc, next, t, out),
            };
            any |= ok;
        }
        out.push((lit, any));
        any
    }
    let mut prefixes = Vec::new();
    let any = alive(dfa, &coacc, LIT_START, dfa.start, &mut prefixes);
    (any || null).then_some(Slot::Pattern { re, coacc, prefixes, null })
}

impl ToolCallGrammar {
    /// Compile a request's tools. `schemas[i]` is tool `i`'s compiled
    /// `parameters` when it is held; a normalized tool the dialect cannot
    /// hold is left free here, where a strict one is refused. With `only`,
    /// the function a named `tool_choice` names, a call names that tool
    /// alone, when the dialect can write its name, and every declared tool
    /// otherwise.
    pub(crate) fn compile(
        vocab: Arc<Vocab>,
        controls: Arc<Controls>,
        max_ws: u8,
        memo_budget: usize,
        tools: &[DeclaredTool],
        schemas: Vec<Option<HeldSchema>>,
        only: Option<&str>,
    ) -> Result<Self, ToolRefusal> {
        let only = only.filter(|name| tools.iter().any(|tool| tool.name == *name) && writable_name(name));
        let mut names = LitTrie::new();
        let mut name_tools = HashMap::new();
        let mut callable = 0;
        let mut held = Vec::new();
        for (i, (tool, mut schema)) in tools.iter().zip(schemas).enumerate() {
            let name_refusal = |code, detail: String| ToolRefusal { tool: i, field: ToolField::Name, refusal: refusal(code, "name", "#", detail) };
            let writable = writable_name(&tool.name);
            let duplicate = tools.iter().enumerate().any(|(j, other)| j != i && other.name == tool.name);
            if schema.as_ref().is_some_and(|s| s.normalized) && (!writable || duplicate) {
                schema = None;
            }
            if schema.is_some() {
                if !writable {
                    return Err(name_refusal(
                        "tool_name_unwritable",
                        format!("the model's tool-call dialect cannot write the name \"{}\": a name ends at its first `>`", tool.name),
                    ));
                }
                if duplicate {
                    return Err(name_refusal("duplicate_tool_name", format!("another tool is also named \"{}\", so a call to it names neither", tool.name)));
                }
            }
            if writable && !tool.name.is_empty() && only.is_none_or(|only| only == tool.name) {
                let before = name_tools.len();
                names.add(tool.name.as_bytes());
                let node = name_node(&names, tool.name.as_bytes());
                name_tools.entry(node).or_insert(i as u32);
                callable += (name_tools.len() > before) as usize;
            }
            held.push(match schema {
                Some(schema) => match held_tool(schema.grammar, &schema.optional) {
                    Ok(tool) => Some(tool),
                    Err(_) if schema.normalized => None,
                    Err(refusal) => return Err(ToolRefusal { tool: i, field: ToolField::Parameters, refusal }),
                },
                None => None,
            });
        }
        let mut g = Self {
            vocab,
            controls,
            names,
            name_tools,
            callable,
            held,
            tool_names: tools.iter().map(|t| t.name.clone()).collect(),
            max_ws,
            memo: Mutex::new(Memo::default()),
            memo_budget,
            tables_bytes: 0,
            first_draw_misses: false,
        };
        g.tables_bytes = g.held.iter().flatten().flat_map(|h| &h.slots).map(Slot::heap_bytes).sum::<usize>() + g.tool_names.iter().map(|n| n.len() * 3 + 64).sum::<usize>();
        let start = CallState::default();
        g.first_draw_misses = g.callable == 0
            || g.controls.opener_inside.iter().any(|&id| {
                let mut s = start.clone();
                !g.controls.bytes(&g.vocab, id).is_some_and(|bytes| bytes.iter().all(|&b| g.step(&mut s, b)))
            });
        Ok(g)
    }

    /// Each held tool's name and what the confirmed close knows of it, its
    /// parameters declared in the schema's order: what the call parser must
    /// read the same calls by.
    pub fn held(&self) -> impl Iterator<Item = (&str, &Parameters)> + '_ {
        self.tool_names.iter().zip(&self.held).filter_map(|(name, held)| held.as_ref().map(|h| (name.as_str(), &h.parameters)))
    }

    /// Whether the request's `tool`th tool is held: a strict tool always, a
    /// normalized one when its schema normalized and the dialect holds it.
    pub fn holds(&self, tool: usize) -> bool {
        self.held.get(tool).is_some_and(Option::is_some)
    }

    pub fn vocab(&self) -> &Arc<Vocab> {
        &self.vocab
    }

    /// Whether the first token the model draws can be a miss (so its scores
    /// must be kept): only when a token holds the opener with more after it,
    /// or no tool can be called.
    pub fn first_draw_may_miss(&self) -> bool {
        self.first_draw_misses
    }

    /// Memory held by the compiled tables and the cached masks.
    pub fn heap_bytes(&self) -> usize {
        self.tables_bytes + self.memo.lock().unwrap().bytes
    }

    fn tool(&self, tool: u32) -> &Held {
        self.held[tool as usize].as_ref().expect("a held call's tool is held")
    }

    /// Advance `s` by one byte; `false` when the byte is a miss (and `s` is
    /// then garbage).
    fn step(&self, s: &mut CallState, b: u8) -> bool {
        match s.at {
            At::Free { kmp: k } => {
                let k = kmp(CALL_OPEN.as_bytes(), k, b);
                if k as usize == CALL_OPEN.len() {
                    if self.callable == 0 {
                        return false;
                    }
                    s.at = At::Open { ws: 0 };
                } else {
                    s.at = At::Free { kmp: k };
                }
                true
            }
            At::Open { ws } => {
                if is_ws(b) && ws < self.max_ws {
                    s.at = At::Open { ws: ws + 1 };
                    true
                } else if b == FUNCTION_OPEN.as_bytes()[0] {
                    s.at = At::FunctionTag { at: 1 };
                    true
                } else {
                    false
                }
            }
            At::FunctionTag { at } => {
                if FUNCTION_OPEN.as_bytes()[at as usize] != b {
                    return false;
                }
                s.at = match at as usize + 1 == FUNCTION_OPEN.len() {
                    true => At::Name { node: 0 },
                    false => At::FunctionTag { at: at + 1 },
                };
                true
            }
            At::Name { node } => {
                if b == b'>' {
                    let Some(&tool) = self.name_tools.get(&node) else { return false };
                    match &self.held[tool as usize] {
                        Some(held) => {
                            s.written.clear();
                            s.written.resize(held.parameters.declared().len(), false);
                            s.pending.clear();
                            s.starts.clear();
                            self.enter(s, tool, JUNK);
                        }
                        None => s.at = At::Body { kmp: 0 },
                    }
                    return true;
                }
                match self.names.child(node, b) {
                    Some(next) => {
                        s.at = At::Name { node: next };
                        true
                    }
                    None => false,
                }
            }
            At::Body { kmp: k } => {
                let k = kmp(CALL_CLOSE.as_bytes(), k, b);
                s.at = match k as usize == CALL_CLOSE.len() {
                    true => At::Free { kmp: 0 },
                    false => At::Body { kmp: k },
                };
                true
            }
            At::Value(v) => self.value_byte(s, v, b),
        }
    }

    /// A value of `slot` (a declared parameter's place, or [`JUNK`]) about
    /// to be read.
    fn value(&self, tool: u32, slot: u32) -> Value {
        let held = self.tool(tool);
        let kind = if slot == JUNK { &JUNK_SLOT } else { &held.slots[slot as usize] };
        let nul = if matches!(kind, Slot::Pattern { null: true, .. }) { 0 } else { OFF };
        Value { tool, slot, lead: true, utf8: 0, cs: kind.start(held), lit: LIT_START, terminated: false, r1: OFF, r2: OFF, ws: 0, nul }
    }

    /// Start reading that value in `s`: a typed value's JSON from its
    /// subschema's root.
    fn enter(&self, s: &mut CallState, tool: u32, slot: u32) {
        s.at = At::Value(self.value(tool, slot));
        s.json.configs.clear();
        if slot != JUNK
            && let Slot::Json { node } = self.tool(tool).slots[slot as usize]
        {
            s.json.configs.push(Config::initial(node));
        }
    }

    /// A typed value's byte as its JSON's (GitHub #612): `false` on a miss.
    /// No whitespace before the JSON (the trimmed newline aside); once it is
    /// whole, whitespace, bounded so that a line break is always left, and
    /// any other byte ends it: that byte is a step's, or a miss.
    fn json_byte(&self, s: &mut CallState, v: &mut Value, held: &Held, b: u8) -> bool {
        if s.json.configs.is_empty() {
            return true;
        }
        if is_ws(b) && !json_begun(&s.json) {
            return false;
        }
        // Whitespace after the whole JSON, the byte that ends a number
        // included.
        let trailing = is_ws(b) && s.json.accepting();
        if !trailing {
            let mut forks = Vec::new();
            if !s.json.step(&held.grammar.core, b, &mut forks) {
                s.json.configs.clear();
                return true;
            }
            if !(is_ws(b) && s.json.accepting()) {
                return true;
            }
        }
        json_done(&mut s.json);
        v.ws = v.ws.saturating_add(1);
        self.whitespace_fits(v.ws, b)
    }

    /// Whether a run of whitespace whose `ws`th byte is `b` fits the bound,
    /// a line break always left for the step that must start a line.
    fn whitespace_fits(&self, ws: u8, b: u8) -> bool {
        ws < self.max_ws || (b == b'\n' && ws == self.max_ws)
    }

    fn value_byte(&self, s: &mut CallState, mut v: Value, b: u8) -> bool {
        let held = self.tool(v.tool);
        let slot = if v.slot == JUNK { &JUNK_SLOT } else { &held.slots[v.slot as usize] };
        let tracks = matches!(slot, Slot::Free) && !held.parameters.may_close(&s.written);
        if is_ws(b) && v.slot == JUNK {
            v.ws = v.ws.saturating_add(1);
            if !self.whitespace_fits(v.ws, b) {
                return false;
            }
        }
        // The value's first byte starts its first line; a newline there is
        // the one the parser trims, and the line starts after it.
        if v.lead {
            v.lead = false;
            s.starts.push(slot.empty_ok(held) as u32 * END_OK);
            if tracks {
                (v.r1, v.r2) = (0, 0);
            }
            if b == b'\n' {
                s.at = At::Value(v);
                return true;
            }
        }
        let pure = s.starts.is_empty();
        let was_terminated = v.terminated;
        // The byte as the value's text.
        match slot {
            Slot::Junk => {
                if pure && !is_ws(b) {
                    return false;
                }
            }
            Slot::Free => {
                let Some(utf8) = utf8_next(v.utf8, b) else { return false };
                v.utf8 = utf8;
                if tracks && !self.free_misses(&mut v, b) {
                    return false;
                }
            }
            Slot::Json { .. } => {
                if !self.json_byte(s, &mut v, held, b) {
                    return false;
                }
            }
            Slot::Pattern { .. } | Slot::Enum(_) if !v.terminated => {
                if b == b'\n' {
                    if !(v.utf8 == 0 && slot.accepts(held, &v)) {
                        return false;
                    }
                    v.terminated = true;
                } else {
                    match utf8_next(v.utf8, b) {
                        Some(utf8) => {
                            v.utf8 = utf8;
                            v.cs = slot.next(held, v.cs, b);
                        }
                        None => v.cs = DEAD,
                    }
                    v.lit = lit_step(v.lit, b);
                    v.nul = null_step(v.nul, b);
                }
            }
            Slot::Pattern { .. } | Slot::Enum(_) => {}
        }
        // The byte joins every undecided line; a newline starts one. A line
        // whose first byte is whitespace is never a step: one that is not
        // the earliest is dropped as that byte arrives.
        if !s.starts.is_empty() {
            s.pending.push(b);
            let newest = |&last: &u32| (last >> FLAGS) as usize + 1 == s.pending.len() && last & MUST_STEP == 0;
            if is_ws(b) && s.starts.len() > 1 && s.starts.last().is_some_and(newest) {
                s.starts.pop();
            }
        }
        if b == b'\n' {
            let at = (s.pending.len() as u32) << FLAGS;
            match slot {
                Slot::Junk | Slot::Free => s.starts.push(at | END_OK),
                // A typed value's line ends where its JSON is whole.
                Slot::Json { .. } => {
                    if s.json.accepting() {
                        s.starts.push(at | END_OK);
                    }
                }
                _ if !was_terminated => s.starts.push(at | END_OK | MUST_STEP),
                _ => {}
            }
        }
        // The earliest undecided line decides, as `call::value_end` reads.
        while let Some(&first) = s.starts.first() {
            let off = (first >> FLAGS) as usize;
            match call::line(&s.pending[off..], &held.parameters, &s.written) {
                Line::Pending => break,
                Line::Step(step) => {
                    if first & END_OK == 0 {
                        return false;
                    }
                    self.end_value(s, v.tool, step);
                    return true;
                }
                Line::Text => {
                    if first & MUST_STEP != 0 {
                        return false;
                    }
                    s.starts.remove(0);
                    let next = s.starts.first().map_or(s.pending.len(), |&n| (n >> FLAGS) as usize);
                    if matches!(slot, Slot::Junk) && s.pending[off..next].iter().any(|&c| !is_ws(c)) {
                        return false;
                    }
                    s.pending.drain(..next);
                    for start in &mut s.starts {
                        *start -= (next as u32) << FLAGS;
                    }
                }
            }
        }
        // Still completable: the earliest undecided line may end the value,
        // or the text read so far may still be the value's.
        let can_step = s.starts.first().is_some_and(|&first| first & END_OK != 0);
        let text = match slot {
            Slot::Free => true,
            Slot::Junk => s.pending.iter().all(|&c| is_ws(c)),
            Slot::Json { .. } => !s.json.configs.is_empty(),
            _ => !v.terminated && slot.text_alive(&v),
        };
        if !(can_step || text) {
            return false;
        }
        // Whitespace between a step's tags is bounded.
        if is_ws(b) && !s.starts.is_empty() && s.pending.iter().rev().take_while(|&&c| is_ws(c)).count() > self.max_ws as usize {
            return false;
        }
        s.at = At::Value(v);
        true
    }

    /// The two miss points of a free value while a required parameter is
    /// still missing (spike S1 of spec #582, the 27B's): a line may not start
    /// `</function` (the token after a line-start `</` that names the
    /// function's closer), and a line that starts `</parameter>` may not go
    /// on, after its whitespace, with `</` (the token that starts a closing
    /// tag after a parameter's closer and its newline). `false` on either.
    fn free_misses(&self, v: &mut Value, b: u8) -> bool {
        if (v.r2 as usize) < FUNCTION_LINE.len() {
            v.r2 = match FUNCTION_LINE[v.r2 as usize] == b {
                true if v.r2 as usize + 1 == FUNCTION_LINE.len() => return false,
                true => v.r2 + 1,
                false => OFF,
            };
        }
        let closer = PARAMETER_CLOSE.as_bytes();
        let after = closer.len() as u8;
        v.r1 = match v.r1 {
            OFF => OFF,
            n if n < after => match closer[n as usize] == b {
                true => n + 1,
                false => OFF,
            },
            n if n == after && is_ws(b) => after,
            n if n == after && b == b'<' => after + 1,
            n if n == after + 1 && b == b'/' => return false,
            _ => OFF,
        };
        if b == b'\n' {
            v.r2 = 0;
            if v.r1 == OFF {
                v.r1 = 0;
            }
        }
        true
    }

    fn end_value(&self, s: &mut CallState, tool: u32, step: Step) {
        s.pending.clear();
        s.starts.clear();
        match step {
            Step::Parameter { index, .. } => {
                s.written[index] = true;
                self.enter(s, tool, index as u32);
            }
            Step::Close { .. } => {
                s.written.clear();
                s.json.configs.clear();
                s.at = At::Free { kmp: 0 };
            }
        }
    }

    /// Whether `s` is free text or a free call's body: where a stop, or a
    /// column with no text, passes.
    fn outside_held(s: &CallState) -> bool {
        matches!(s.at, At::Free { .. } | At::Body { .. })
    }

    /// Mark every token of `trie` whose bytes `start` accepts.
    fn walk(&self, trie: &Trie, start: &CallState, out: &mut TokenSet, levels: &mut Vec<CallState>) {
        if levels.len() < trie.max_depth + 1 {
            levels.resize_with(trie.max_depth + 1, CallState::default);
        }
        levels[0].clone_from(start);
        let mut i = 0;
        while i < trie.nodes.len() {
            let node = trie.nodes[i];
            let d = node.depth as usize;
            let (lo, hi) = levels.split_at_mut(d);
            let cur = &mut hi[0];
            cur.clone_from(&lo[d - 1]);
            if !self.step(cur, node.byte) {
                i = node.end as usize;
                continue;
            }
            for &t in trie.tokens(i) {
                out.insert(t);
            }
            i += 1;
        }
    }

    fn mask(&self, state: &CallState, out: &mut TokenSet, levels: &mut Vec<CallState>) -> bool {
        if let Some(m) = self.memo.lock().unwrap().masks.get(state) {
            out.copy_from(m);
            return true;
        }
        out.clear();
        let fast = match state.at {
            At::Value(v) => v.slot != JUNK
                && matches!(self.tool(v.tool).slots[v.slot as usize], Slot::Free)
                && state.starts.is_empty()
                && !v.lead
                && v.r1 == OFF
                && v.r2 == OFF,
            _ => false,
        };
        let walk_one = |id: u32, bytes: &[u8], scratch: &mut CallState| {
            scratch.clone_from(state);
            bytes.iter().all(|&b| self.step(scratch, b)).then_some(id)
        };
        let mut scratch = CallState::default();
        if fast {
            // Mid-line in a free value: every token with no newline that
            // keeps the text UTF-8 passes; the rest are walked one by one.
            let At::Value(v) = state.at else { unreachable!() };
            let raw = self.vocab.raw_tables();
            out.copy_from(&raw.plain[v.utf8 as usize]);
            for &id in &raw.newline {
                if let Some(id) = walk_one(id, self.vocab.token_bytes(id).unwrap(), &mut scratch) {
                    out.insert(id);
                }
            }
        } else {
            self.walk(&self.vocab.trie, state, out, levels);
        }
        for (&id, text) in &self.controls.texts {
            if let Some(id) = walk_one(id, text, &mut scratch) {
                out.insert(id);
            }
        }
        if Self::outside_held(state) {
            for &id in &self.controls.opaque {
                out.insert(id);
            }
        }
        let mut memo = self.memo.lock().unwrap();
        if !memo.masks.contains_key(state) && memo.bytes.saturating_add(out.heap_bytes() + 64) <= self.memo_budget {
            memo.bytes += out.heap_bytes() + 64;
            memo.masks.insert(state.clone(), Arc::new(out.clone()));
        }
        false
    }
}

/// Whether the dialect can write a function's name: it ends at its first
/// `>`, and `</tool_call` in it would read as the call's end.
fn writable_name(name: &str) -> bool {
    !name.contains('>') && !name.contains(CALL_CLOSE.trim_end_matches('>'))
}

/// The node `bytes` reach in `trie`.
fn name_node(trie: &LitTrie, bytes: &[u8]) -> u32 {
    bytes.iter().fold(0, |node, &b| trie.child(node, b).expect("a name just added"))
}

/// A held tool from its compiled strict `parameters`: the declared
/// parameters in the schema's order, each required but the nullable ones (a
/// strict schema requires them all, and one that may be null may be left
/// out, delivered `null`) and the `optional` ones (a normalized tool's that
/// the client left optional, left out of the arguments when the call leaves
/// them out, #614), how each value reads, and a slot per parameter.
fn held_tool(grammar: Arc<Grammar>, optional: &[String]) -> Result<Held, Refusal> {
    let core = &grammar.core;
    let object = core.alts[core.root as usize].iter().find_map(|&a| match &core.nodes[a as usize] {
        Node::Object { keys, values } => Some((keys, values)),
        _ => None,
    });
    let (keys, values) = object.expect("a strict schema compiles to an object");
    let decode = |set: u32| -> Option<String> {
        let spelled = core.lits[set as usize].strings().into_iter().next()?;
        match Json::parse(std::str::from_utf8(&spelled).ok()?) {
            Ok(Json::Str(s)) => Some(s),
            _ => None,
        }
    };
    let mut names = Vec::new();
    let mut slots = Vec::new();
    let mut readings = Vec::new();
    let mut required = Vec::new();
    for (&key, &value) in keys.iter().zip(values.iter()) {
        let name = decode(key).expect("a key is a string");
        let path = format!("#/properties/{}", name.replace('~', "~0").replace('/', "~1"));
        if name.contains('>') {
            return Err(refusal(
                "parameter_name_unwritable",
                "properties",
                &path,
                format!("the model's tool-call dialect cannot write the parameter name \"{name}\": a name ends at its first `>`"),
            ));
        }
        let nullable = accepts_value(core, value, NULL);
        let (slot, reading) = slot_of(core, value, nullable, &path)?;
        let optional = optional.contains(&name);
        slots.push(slot);
        readings.push((reading, nullable && !optional));
        if !nullable && !optional {
            required.push(name.clone());
        }
        names.push(name);
    }
    Ok(Held { parameters: Parameters::new(names, required).reading(readings), slots, grammar })
}

/// What a value of `node` (which may be `null` when `nullable`) is held to,
/// and how its text reads (GitHub #612): the dialect's raw text where every
/// alternative is a string or `null` and one slot holds them all (a free
/// string, a single pattern or format, or enum strings), else JSON held to
/// the node, a string beside another type written quoted.
fn slot_of(core: &crate::schema::Core, node: u32, nullable: bool, path: &str) -> Result<(Slot, Reading), Refusal> {
    let unwritable = |keyword: &str| {
        refusal(
            "no_writable_value",
            keyword,
            path,
            format!("no value of this `{keyword}` can be written on one line of the model's tool-call dialect that does not start with one of its tags"),
        )
    };
    let json = Ok((Slot::Json { node }, Reading::Json));
    let (mut free, mut patterns, mut strings) = (false, Vec::new(), Vec::new());
    for &alt in &core.alts[node as usize] {
        match &core.nodes[alt as usize] {
            Node::String { re } if *re == NO_RE => free = true,
            Node::String { re } => patterns.push(*re),
            Node::Literal { set } => {
                for spelled in core.lits[*set as usize].strings() {
                    match Json::parse(std::str::from_utf8(&spelled).unwrap_or("")) {
                        Ok(Json::Str(value)) => strings.push(value),
                        Ok(Json::Null) => {}
                        _ => return json,
                    }
                }
            }
            _ => return json,
        }
    }
    let reading = if nullable { Reading::TextOrNull } else { Reading::Text };
    if free {
        return Ok((Slot::Free, reading));
    }
    match (patterns.as_slice(), strings.is_empty()) {
        ([re], true) => pattern_slot(&core.dfas[*re as usize], *re, nullable).map(|slot| (slot, reading)).ok_or_else(|| unwritable("pattern")),
        ([], false) => {
            let mut trie = LitTrie::new();
            let mut kept = 0;
            for value in strings.iter().map(String::as_str).chain(nullable.then_some("null")) {
                let starts_with_tag = value.bytes().try_fold(LIT_START, |lit, b| Some(lit_step(lit, b)).filter(|&l| l != TAG)).is_none();
                if !value.contains('\n') && !starts_with_tag {
                    trie.add(value.as_bytes());
                    kept += 1;
                }
            }
            match kept {
                0 => Err(unwritable("enum")),
                _ => Ok((Slot::Enum(trie), reading)),
            }
        }
        // `null` alone, several patterns, or a pattern beside enum strings.
        _ => json,
    }
}

/// One request's position in its tool calls' grammar.
pub struct ToolCallMatcher {
    grammar: Arc<ToolCallGrammar>,
    state: CallState,
    scratch: CallState,
    levels: Vec<CallState>,
}

impl Clone for ToolCallMatcher {
    fn clone(&self) -> Self {
        Self { state: self.state.clone(), ..Self::new(self.grammar.clone()) }
    }
}

impl ToolCallMatcher {
    pub fn new(grammar: Arc<ToolCallGrammar>) -> Self {
        Self { grammar, state: CallState::default(), scratch: CallState::default(), levels: Vec::new() }
    }

    pub fn grammar(&self) -> &Arc<ToolCallGrammar> {
        &self.grammar
    }

    /// Whether `token` is valid here. Leaves the state unchanged.
    pub fn check(&mut self, token: u32) -> bool {
        let g = &*self.grammar;
        if g.vocab.is_stop(token) {
            return self.is_accepting();
        }
        let Some(bytes) = g.controls.bytes(&g.vocab, token) else {
            return ToolCallGrammar::outside_held(&self.state);
        };
        self.scratch.clone_from(&self.state);
        bytes.iter().all(|&b| g.step(&mut self.scratch, b))
    }

    /// Check `token` and, when valid, take it. A stop passes where it is
    /// valid and changes nothing: the generation ends there unless the
    /// lane decodes past its EOS (`ignore_eos`), and then the content goes on
    /// as free text does without a grammar.
    pub fn accept(&mut self, token: u32) -> bool {
        if !self.check(token) {
            return false;
        }
        let g = &*self.grammar;
        if !g.vocab.is_stop(token) && g.controls.bytes(&g.vocab, token).is_some() {
            std::mem::swap(&mut self.state, &mut self.scratch);
        }
        true
    }

    /// Feed raw bytes (not a token): a test and generation helper.
    pub fn accept_bytes(&mut self, bytes: &[u8]) -> bool {
        self.scratch.clone_from(&self.state);
        if !bytes.iter().all(|&b| self.grammar.step(&mut self.scratch, b)) {
            return false;
        }
        std::mem::swap(&mut self.state, &mut self.scratch);
        true
    }

    /// The bytes valid next (a generation helper).
    pub fn allowed_bytes(&mut self) -> Vec<u8> {
        (0..=255u8)
            .filter(|&b| {
                self.scratch.clone_from(&self.state);
                self.grammar.step(&mut self.scratch, b)
            })
            .collect()
    }

    /// Whether a stop may end the generation here: in free text, or in the
    /// body of a call to a tool that is not held.
    pub fn is_accepting(&self) -> bool {
        ToolCallGrammar::outside_held(&self.state)
    }

    /// Never: the content's text never completes, a stop passing in it.
    pub fn is_finished(&self) -> bool {
        false
    }

    /// Whether a call is open: its opener read, its close not yet.
    pub fn in_call(&self) -> bool {
        !matches!(self.state.at, At::Free { .. })
    }

    /// The repair's permitted set. Returns `true` when it came whole from
    /// the grammar's cache.
    pub fn mask(&mut self, out: &mut TokenSet) -> bool {
        self.grammar.mask(&self.state, out, &mut self.levels)
    }

    /// The reference the mask is tested against: [`ToolCallMatcher::check`]
    /// on every id of the vocabulary.
    pub fn mask_reference(&mut self, out: &mut TokenSet) {
        out.clear();
        for id in 0..self.grammar.vocab.len() as u32 {
            if self.check(id) {
                out.insert(id);
            }
        }
    }
}

#[cfg(test)]
mod tests;
