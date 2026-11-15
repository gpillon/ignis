//! Compiled grammars kept per loaded tokenizer: one per distinct schema, so a
//! request that sends a schema already seen pays nothing for it, in bounded
//! memory.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::json::Json;
use crate::schema::{Grammar, GrammarLimits, Refusal};
use crate::tool_calls::normalize::strict_form;
use crate::tool_calls::{Controls, DeclaredTool, HeldSchema, ToolCallGrammar, ToolHold, ToolRefusal};
use crate::vocab::Vocab;

/// The memory a [`GrammarCache`] keeps its grammars in by default: room for
/// hundreds of SDK-sized schemas with their masks.
pub const DEFAULT_CACHE_BYTES: usize = 256 << 20;

/// The grammars compiled against one tokenizer's [`Vocab`].
///
/// A schema is the same schema when its JSON is, whitespace aside, with the
/// same `strict` flag; its refusal is kept too. A request's tool calls
/// ([`GrammarCache::tool_calls`], #611) are the same when its declared tools
/// are, names, strict and normalized schemas in order. Past the capacity
/// ([`GrammarLimits::cache_bytes`]), the least recently used entries are
/// dropped; a request still holding one keeps it alive. A grammar's own
/// caches grow while requests use it, never past the capacity, and the bound
/// is applied again at every lookup.
pub struct GrammarCache {
    vocab: Arc<Vocab>,
    limits: GrammarLimits,
    json_object: OnceLock<Arc<Grammar>>,
    /// The tokenizer's added tokens and the text the decoder gives each
    /// ([`GrammarCache::with_added_texts`]), read by tool calls' grammars.
    added: Vec<(u32, Vec<u8>)>,
    controls: OnceLock<Arc<Controls>>,
    inner: Mutex<Inner>,
    compiles: AtomicU64,
}

#[derive(Default)]
struct Inner {
    entries: HashMap<(Vec<u8>, Kind), Entry>,
    clock: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Kind {
    Schema { strict: bool },
    ToolCalls,
}

#[derive(Clone)]
enum Compiled {
    Schema(Result<Arc<Grammar>, Refusal>),
    ToolCalls(Result<Arc<ToolCallGrammar>, ToolRefusal>),
}

struct Entry {
    result: Compiled,
    used: u64,
}

impl Entry {
    fn heap_bytes(&self, key: &[u8]) -> usize {
        let refused = |r: &Refusal| r.detail.len() + r.path.len() + r.keyword.len();
        key.len()
            + match &self.result {
                Compiled::Schema(Ok(g)) => g.heap_bytes(),
                Compiled::Schema(Err(r)) => refused(r),
                Compiled::ToolCalls(Ok(g)) => g.heap_bytes(),
                Compiled::ToolCalls(Err(r)) => refused(&r.refusal),
            }
    }
}

impl GrammarCache {
    /// A cache over `vocab` keeping at most about `capacity_bytes` of
    /// compiled grammars, each compiled within the default limits.
    pub fn new(vocab: Arc<Vocab>, capacity_bytes: usize) -> Self {
        Self::within(vocab, GrammarLimits { cache_bytes: capacity_bytes, ..GrammarLimits::default() })
    }

    /// A cache over `vocab` whose grammars are compiled within `limits`, and
    /// hold at most about its `cache_bytes`.
    pub fn within(vocab: Arc<Vocab>, limits: GrammarLimits) -> Self {
        Self {
            vocab,
            limits,
            json_object: OnceLock::new(),
            added: Vec::new(),
            controls: OnceLock::new(),
            inner: Mutex::default(),
            compiles: AtomicU64::new(0),
        }
    }

    /// The same cache, whose tool calls' grammars read each of the
    /// tokenizer's added tokens -- control tokens to a schema's grammar,
    /// with no bytes -- as `texts` gives it: the text the server's decoder
    /// writes for it into the content a tool call is read from (#611).
    pub fn with_added_texts(self, texts: Vec<(u32, Vec<u8>)>) -> Self {
        Self { added: texts, ..self }
    }

    /// The limits its grammars are compiled within.
    pub fn limits(&self) -> &GrammarLimits {
        &self.limits
    }

    pub fn vocab(&self) -> &Arc<Vocab> {
        &self.vocab
    }

    /// `json_object`'s grammar, compiled on first use and kept.
    pub fn json_object(&self) -> Arc<Grammar> {
        self.json_object
            .get_or_init(|| {
                self.compiles.fetch_add(1, Ordering::Relaxed);
                Grammar::json_object_within(self.vocab.clone(), &self.limits)
            })
            .clone()
    }

    /// The grammar of a `json_schema` request's schema, compiled once per
    /// distinct schema and `strict` flag.
    pub fn json_schema(&self, schema: &str, strict: bool) -> Result<Arc<Grammar>, Refusal> {
        // Whitespace and number spelling aside, key order kept: the same
        // schema is the same key.
        let key = (canonical(schema), Kind::Schema { strict });
        if let Some(Compiled::Schema(result)) = self.lookup(&key) {
            return result;
        }
        // Compiled outside the lock: a slow schema does not hold up the
        // lookups of others.
        self.compiles.fetch_add(1, Ordering::Relaxed);
        let result = Grammar::json_schema_within(self.vocab.clone(), schema, strict, &self.limits);
        self.keep(key, Compiled::Schema(result.clone()));
        result
    }

    /// The grammar of a request's tool calls (GitHub #611, ADR 0094): the
    /// function tools it declares, in order, each strict one's `parameters`
    /// compiled under `strict` (and cached as [`GrammarCache::json_schema`]
    /// keeps it), and each normalized one's in strict form (#614), compiled
    /// once per distinct set. A strict schema outside OpenAI's strict subset,
    /// and a strict tool the model's dialect cannot hold, are refused, naming
    /// the tool and its field; a normalized tool whose schema does not
    /// normalize, or which the dialect cannot hold, is left free
    /// ([`ToolCallGrammar::holds`]).
    pub fn tool_calls(&self, tools: &[DeclaredTool]) -> Result<Arc<ToolCallGrammar>, ToolRefusal> {
        let mut key = Vec::new();
        for tool in tools {
            key.extend_from_slice(&(tool.name.len() as u32).to_le_bytes());
            key.extend_from_slice(tool.name.as_bytes());
            let (kind, schema) = match &tool.hold {
                ToolHold::Free => (0, None),
                ToolHold::Strict(schema) => (1, Some(schema)),
                ToolHold::Normalized(schema) => (2, Some(schema)),
            };
            key.push(kind);
            if let Some(schema) = schema {
                let schema = canonical(schema);
                key.extend_from_slice(&(schema.len() as u32).to_le_bytes());
                key.extend_from_slice(&schema);
            }
        }
        let key = (key, Kind::ToolCalls);
        if let Some(Compiled::ToolCalls(result)) = self.lookup(&key) {
            return result;
        }
        let mut schemas = Vec::with_capacity(tools.len());
        for (tool, declared) in tools.iter().enumerate() {
            schemas.push(match &declared.hold {
                ToolHold::Free => None,
                ToolHold::Strict(schema) => Some(HeldSchema {
                    grammar: self.json_schema(schema, true).map_err(|refusal| ToolRefusal {
                        tool,
                        field: crate::tool_calls::ToolField::Parameters,
                        refusal,
                    })?,
                    optional: Vec::new(),
                    normalized: false,
                }),
                ToolHold::Normalized(schema) => Json::parse(schema)
                    .ok()
                    .and_then(|parameters| strict_form(&parameters))
                    .and_then(|form| Some(HeldSchema { grammar: self.json_schema(&form.schema, true).ok()?, optional: form.optional, normalized: true })),
            });
        }
        let controls = self.controls.get_or_init(|| Arc::new(Controls::new(&self.vocab, &self.added))).clone();
        self.compiles.fetch_add(1, Ordering::Relaxed);
        let result = ToolCallGrammar::compile(self.vocab.clone(), controls, self.limits.max_whitespace, self.limits.cache_bytes, tools, schemas)
            .map(Arc::new);
        self.keep(key, Compiled::ToolCalls(result.clone()));
        result
    }

    fn lookup(&self, key: &(Vec<u8>, Kind)) -> Option<Compiled> {
        let mut inner = self.inner.lock().unwrap();
        inner.clock += 1;
        let now = inner.clock;
        let e = inner.entries.get_mut(key)?;
        e.used = now;
        let result = e.result.clone();
        self.evict(&mut inner);
        Some(result)
    }

    fn keep(&self, key: (Vec<u8>, Kind), result: Compiled) {
        let mut inner = self.inner.lock().unwrap();
        inner.clock += 1;
        let used = inner.clock;
        inner.entries.insert(key, Entry { result, used });
        self.evict(&mut inner);
    }

    /// Drop the least recently used schemas until the cache fits, keeping
    /// the most recent one. Each grammar counts its own memory as it grows,
    /// so the walk reads one number per schema.
    fn evict(&self, inner: &mut Inner) {
        let mut total: usize = inner.entries.iter().map(|((k, _), e)| e.heap_bytes(k)).sum();
        while inner.entries.len() > 1 && total > self.limits.cache_bytes {
            let oldest = inner.entries.iter().min_by_key(|(_, e)| e.used).map(|(k, _)| k.clone()).unwrap();
            let e = inner.entries.remove(&oldest).unwrap();
            total -= e.heap_bytes(&oldest.0);
        }
    }

    /// How many grammars this cache has compiled (`json_object` and tool
    /// calls' included): a schema sent again is not compiled again.
    pub fn compiles(&self) -> u64 {
        self.compiles.load(Ordering::Relaxed)
    }

    /// How many schemas (and tool sets) the cache holds.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Memory held by the schemas' grammars and their caches.
    pub fn heap_bytes(&self) -> usize {
        self.inner.lock().unwrap().entries.iter().map(|((k, _), e)| e.heap_bytes(k)).sum()
    }
}

/// A schema's key: its JSON written canonically (key order kept), or its
/// text when it is not JSON.
fn canonical(schema: &str) -> Vec<u8> {
    match Json::parse(schema) {
        Ok(j) => {
            let mut b = Vec::new();
            j.write_canonical(&mut b);
            b
        }
        Err(_) => schema.as_bytes().to_vec(),
    }
}
