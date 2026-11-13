//! A request's JSON schema compiled to the grammar's nodes, or refused by
//! name.
//!
//! - **Strict** (`strict: true`, OpenAI's strict mode): the schema must lie
//!   in OpenAI's strict subset ([`STRICT_SUBSET`]), and all of it is
//!   enforced. A keyword outside the subset, a broken strict rule (a root
//!   that is not an object, `additionalProperties` not false, a property not
//!   required), a non-regular pattern, a non-local reference or a bound the
//!   grammar cannot hold exactly is refused, naming the keyword.
//! - **Not strict**: what the subset can enforce is enforced, and the rest is
//!   **inert**: not enforced, and listed by [`Grammar::inert`] so the server
//!   and the guide can say so. An untyped schema is given the types its
//!   keywords imply, an open object writes only the keys it names, and every
//!   property is written: each choice keeps every answer valid.
//! - **Both**: a malformed schema, a schema no finite value satisfies and
//!   OpenAI's four size limits are refused.
//!
//! What the grammar generates is a subset of the valid documents: keys in
//! schema order, literals in canonical spelling, integers as digits only,
//! bounded numbers in plain decimal, whitespace in short runs, nesting
//! bounded by [`GrammarLimits::max_depth`].

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use crate::json::{write_string, Json};
use crate::number::{parse_scaled, NumError, NumSpec, SCALE};
use crate::regex::{format_pattern, translate_ecma, ByteDfa, DfaError, DEFAULT_MAX_PATTERN_STATES};
use crate::vocab::Vocab;

pub(crate) type NodeId = u32;
pub(crate) const NO_RE: u32 = u32::MAX;
/// A node's need when it has no finite value.
const UNPRODUCTIVE: u32 = u32::MAX;

/// OpenAI's limits on a strict schema, held in both modes.
pub const MAX_PROPERTIES: usize = 5000;
pub const MAX_NESTING: usize = 10;
pub const MAX_ENUM_VALUES: usize = 1000;
pub const MAX_SCHEMA_CHARS: usize = 120_000;
/// [`GrammarLimits::max_ref_chain`]'s default: far beyond any real schema.
pub const DEFAULT_MAX_REF_CHAIN: usize = 256;
/// [`GrammarLimits::max_whitespace`]'s default: enough for pretty-printed
/// JSON at depth ~15, few enough that an endless whitespace stream is
/// impossible.
pub const DEFAULT_MAX_WS_RUN: u8 = 48;
/// [`GrammarLimits::max_depth`]'s default.
pub const DEFAULT_MAX_DEPTH: usize = 128;

/// The grammar's own limits: what a schema may compile to, what the model may
/// write, and the memory compiled grammars keep (CODING_STANDARDS: limits are
/// config fields; the server sets them from `model.grammar_*`). The defaults
/// are the measured ones. OpenAI's four limits ([`MAX_PROPERTIES`],
/// [`MAX_NESTING`], [`MAX_ENUM_VALUES`], [`MAX_SCHEMA_CHARS`]) are the
/// contract's and stay constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GrammarLimits {
    /// The most states a `pattern` or a `format` may compile to: a larger one
    /// is refused under `strict` (`pattern_too_large`) and inert without it.
    pub max_pattern_states: usize,
    /// The deepest the automaton's stack may grow, about one frame per level
    /// of nesting. A value is entered only where the stack has room to finish
    /// it, so the bound never strands the model, and a schema whose shortest
    /// answer nests deeper is refused (`exceeds_max_depth`).
    pub max_depth: usize,
    /// How deep the compiler may recurse through a schema's subschemas and
    /// `$ref`s: deeper is refused (`references_too_deep`).
    pub max_ref_chain: usize,
    /// The longest whitespace run the model may write between two
    /// structural bytes.
    pub max_whitespace: u8,
    /// The memory a cache's compiled grammars may hold
    /// ([`crate::GrammarCache`]), their masks and string tables included:
    /// one grammar's own caches stop growing at this many bytes.
    pub cache_bytes: usize,
}

impl Default for GrammarLimits {
    fn default() -> Self {
        Self {
            max_pattern_states: DEFAULT_MAX_PATTERN_STATES,
            max_depth: DEFAULT_MAX_DEPTH,
            max_ref_chain: DEFAULT_MAX_REF_CHAIN,
            max_whitespace: DEFAULT_MAX_WS_RUN,
            cache_bytes: crate::DEFAULT_CACHE_BYTES,
        }
    }
}

/// The keywords of OpenAI's strict subset: the ones the grammar enforces.
pub const STRICT_SUBSET: &[&str] = &[
    "type", "properties", "required", "additionalProperties", "items", "minItems", "maxItems", "enum", "const",
    "anyOf", "$defs", "definitions", "$ref", "pattern", "format", "minimum", "maximum", "exclusiveMinimum",
    "exclusiveMaximum", "multipleOf",
];
/// Annotations: accepted anywhere, never enforced and never reported.
pub const ANNOTATIONS: &[&str] =
    &["title", "description", "$comment", "examples", "default", "deprecated", "readOnly", "writeOnly", "$schema", "$id"];
/// The string formats the grammar enforces, OpenAI's nine.
pub const FORMATS: &[&str] = &["date-time", "time", "date", "duration", "email", "hostname", "ipv4", "ipv6", "uuid"];

const TYPES: &[&str] = &["null", "boolean", "object", "array", "number", "integer", "string"];

/// Why a schema is refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
    /// A stable code for the reason (`unsupported_keyword`,
    /// `too_many_properties`, ...).
    pub code: &'static str,
    /// What the refusal names: the schema keyword it is about (`allOf`,
    /// `pattern`, `$ref`, ...), or, for one of OpenAI's limits, the limit
    /// (`properties`, `nesting`, `enum values`, `characters`).
    pub keyword: String,
    /// Where in the schema, as a JSON pointer fragment (`#` is the root).
    pub path: String,
    /// The reason in a sentence.
    pub detail: String,
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} (at {})", self.detail, self.path)
    }
}

impl std::error::Error for Refusal {}

/// A keyword a schema compiled without `strict` carries and the grammar does
/// not enforce.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Inert {
    pub keyword: String,
    /// Where in the schema, as a JSON pointer fragment (`#` is the root).
    pub path: String,
}

fn refuse(code: &'static str, keyword: &str, path: &str, detail: impl Into<String>) -> Refusal {
    Refusal { code, keyword: keyword.to_string(), path: path.to_string(), detail: detail.into() }
}

pub(crate) enum Node {
    /// An alias or `anyOf`/type union; flattened into `alts`.
    Union(Vec<NodeId>),
    /// An object with any keys, each holding a value of `value`.
    FreeObject { value: NodeId },
    FreeArray,
    Object { keys: Vec<u32>, values: Vec<NodeId> },
    Array { item: NodeId, min: u32, max: u32 },
    String { re: u32 },
    Number { spec: u32 },
    Literal { set: u32 },
}

/// A set of literal byte strings (`enum`, `const`, keys) as a trie.
#[derive(Default)]
pub(crate) struct LitTrie {
    edges: Vec<Vec<(u8, u32)>>,
    terminal: Vec<bool>,
}

impl LitTrie {
    fn new() -> Self {
        Self { edges: vec![Vec::new()], terminal: vec![false] }
    }
    fn add(&mut self, s: &[u8]) {
        let mut cur = 0usize;
        for &b in s {
            cur = match self.edges[cur].iter().find(|(c, _)| *c == b) {
                Some(&(_, n)) => n as usize,
                None => {
                    let n = self.edges.len();
                    self.edges.push(Vec::new());
                    self.terminal.push(false);
                    self.edges[cur].push((b, n as u32));
                    n
                }
            };
        }
        self.terminal[cur] = true;
    }
    #[inline]
    pub fn child(&self, pos: u32, b: u8) -> Option<u32> {
        self.edges[pos as usize].iter().find(|(c, _)| *c == b).map(|&(_, n)| n)
    }
    #[inline]
    pub fn terminal(&self, pos: u32) -> bool {
        self.terminal[pos as usize]
    }
    #[inline]
    pub fn leaf(&self, pos: u32) -> bool {
        self.edges[pos as usize].is_empty()
    }
}

pub(crate) const LIT_BOOL: u32 = 0;
pub(crate) const LIT_NULL: u32 = 1;
pub(crate) const NUM_FREE: u32 = 0;
pub(crate) const NUM_FREE_INT: u32 = 1;

/// The tables the automaton reads.
pub(crate) struct Core {
    pub nodes: Vec<Node>,
    /// Every node's concrete alternatives, the ones with a finite value.
    pub alts: Vec<Vec<NodeId>>,
    /// The fewest frames above its own a value of the node needs to finish
    /// (the cheapest alternative's for a union).
    pub need: Vec<u32>,
    pub root: NodeId,
    /// The free value of `json_object` (unused by a schema grammar).
    pub any: NodeId,
    pub lits: Vec<LitTrie>,
    pub dfas: Vec<ByteDfa>,
    pub nums: Vec<NumSpec>,
    pub max_ws: u8,
    /// [`GrammarLimits::max_depth`].
    pub max_stack: usize,
}

impl Core {
    /// Whether `node`'s value can start with byte `b`.
    #[inline]
    pub(crate) fn can_start(&self, node: NodeId, b: u8) -> bool {
        match &self.nodes[node as usize] {
            Node::FreeObject { .. } | Node::Object { .. } => b == b'{',
            Node::FreeArray | Node::Array { .. } => b == b'[',
            Node::String { .. } => b == b'"',
            Node::Number { .. } => b == b'-' || b.is_ascii_digit(),
            Node::Literal { set } => self.lits[*set as usize].child(0, b).is_some(),
            Node::Union(_) => false,
        }
    }

    /// The concrete (non-union) alternatives of every node.
    fn flat_alts(&self) -> Vec<Vec<NodeId>> {
        let mut stamp = vec![u32::MAX; self.nodes.len()];
        (0..self.nodes.len())
            .map(|i| {
                let mut out = Vec::new();
                self.collect(i as NodeId, i as u32, &mut out, &mut stamp);
                out
            })
            .collect()
    }

    fn collect(&self, n: NodeId, mark: u32, out: &mut Vec<NodeId>, stamp: &mut [u32]) {
        if stamp[n as usize] == mark {
            return;
        }
        stamp[n as usize] = mark;
        match &self.nodes[n as usize] {
            Node::Union(xs) => {
                for &x in xs {
                    self.collect(x, mark, out, stamp);
                }
            }
            _ => out.push(n),
        }
    }

    /// Flatten the unions, find every node's need, and drop the
    /// alternatives no finite value can finish (a recursion through required
    /// keys, a `false` schema): the automaton never enters one.
    fn analyse(&mut self) {
        self.alts = self.flat_alts();
        let n = self.nodes.len();
        let mut need = vec![UNPRODUCTIVE; n];
        loop {
            let mut changed = false;
            for i in 0..n {
                let of = |x: NodeId| self.alts[x as usize].iter().map(|&a| need[a as usize]).min().unwrap_or(UNPRODUCTIVE);
                let v = match &self.nodes[i] {
                    Node::Union(_) => of(i as NodeId),
                    Node::Literal { .. } | Node::String { .. } | Node::Number { .. } | Node::FreeObject { .. } | Node::FreeArray => 0,
                    // An item sits one frame above its array, a key and a
                    // value one above their object.
                    Node::Array { item, min, .. } => if *min == 0 { 0 } else { of(*item).saturating_add(1) },
                    Node::Object { values, .. } => values.iter().map(|&v| of(v)).max().map_or(0, |m| m.saturating_add(1)),
                };
                if v < need[i] {
                    need[i] = v;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        for a in &mut self.alts {
            a.retain(|&x| need[x as usize] != UNPRODUCTIVE);
        }
        self.need = need;
    }
}

/// A compiled grammar: one per distinct schema (and one for `json_object`),
/// shared by every request that sends it.
pub struct Grammar {
    pub(crate) core: Core,
    pub(crate) vocab: Arc<Vocab>,
    pub(crate) memo: Mutex<crate::mask::Memo>,
    /// The most bytes `memo` keeps ([`GrammarLimits::cache_bytes`]).
    pub(crate) memo_budget: usize,
    /// The compiled tables' own memory, `memo` aside: fixed once compiled.
    tables_bytes: usize,
    inert: Vec<Inert>,
}

impl fmt::Debug for Grammar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Grammar").field("nodes", &self.core.nodes.len()).field("inert", &self.inert).finish_non_exhaustive()
    }
}

impl Grammar {
    /// `response_format: {"type": "json_object"}`: any JSON object, values
    /// of any type, whitespace in bounded runs.
    pub fn json_object(vocab: Arc<Vocab>) -> Arc<Grammar> {
        Self::json_object_within(vocab, &GrammarLimits::default())
    }

    /// [`Grammar::json_object`] within `limits`.
    pub fn json_object_within(vocab: Arc<Vocab>, limits: &GrammarLimits) -> Arc<Grammar> {
        let mut c = Compiler::new(vocab, false, limits);
        let any = c.any_value();
        let root = c.push(Node::FreeObject { value: any });
        c.core.any = any;
        Arc::new(c.finish(root))
    }

    /// `response_format: {"type": "json_schema", ...}`: the schema text (its
    /// `schema` member) and its `strict` flag.
    pub fn json_schema(vocab: Arc<Vocab>, schema: &str, strict: bool) -> Result<Arc<Grammar>, Refusal> {
        Self::json_schema_within(vocab, schema, strict, &GrammarLimits::default())
    }

    /// [`Grammar::json_schema`] within `limits`.
    pub fn json_schema_within(vocab: Arc<Vocab>, schema: &str, strict: bool, limits: &GrammarLimits) -> Result<Arc<Grammar>, Refusal> {
        let doc = Json::parse(schema).map_err(|e| refuse("invalid_json", "schema", "#", format!("the schema is not JSON: {e}")))?;
        let mut c = Compiler::new(vocab, strict, limits);
        if strict {
            if doc.get("anyOf").is_some() {
                return Err(refuse("root_anyof", "anyOf", "#", "the root schema must be an object, not `anyOf`"));
            }
            if doc.get("type") != Some(&Json::Str("object".into())) {
                return Err(refuse("root_not_object", "type", "#", "the root schema must have `type` \"object\""));
            }
        }
        c.check_limits(&doc)?;
        c.doc = Some(doc);
        let root = c.compile("#")?;
        c.resolve_literals()?;
        let refs = c.refs;
        let g = c.finish(root);
        if g.core.alts[root as usize].is_empty() {
            return Err(refuse(
                "no_finite_value",
                if refs { "$ref" } else { "schema" },
                "#",
                "no finite document satisfies the schema (a recursion through required keys, or a `false` schema)",
            ));
        }
        // The root's own two frames, then what its shortest answer nests.
        if g.core.need[root as usize] as usize + 2 > g.core.max_stack {
            return Err(refuse(
                "exceeds_max_depth",
                "depth",
                "#",
                format!("the schema's shortest answer nests deeper than the grammar's {} levels", g.core.max_stack),
            ));
        }
        Ok(Arc::new(g))
    }

    /// The keywords this grammar carries and does not enforce (only a schema
    /// compiled without `strict` has any), in schema-path order.
    pub fn inert(&self) -> &[Inert] {
        &self.inert
    }

    pub fn node_count(&self) -> usize {
        self.core.nodes.len()
    }

    pub fn dfa_states(&self) -> usize {
        self.core.dfas.iter().map(|d| d.state_count()).sum()
    }

    pub fn vocab(&self) -> &Arc<Vocab> {
        &self.vocab
    }

    /// Memory held by the compiled grammar (DFAs, literal tries) and its
    /// caches.
    pub fn heap_bytes(&self) -> usize {
        self.tables_bytes + self.memo.lock().unwrap().heap_bytes()
    }

    /// Memory held by the caches alone (string tables, whole masks).
    pub fn cache_bytes(&self) -> usize {
        self.memo.lock().unwrap().heap_bytes()
    }
}

struct Compiler {
    vocab: Arc<Vocab>,
    strict: bool,
    limits: GrammarLimits,
    doc: Option<Json>,
    core: Core,
    dfa_ids: HashMap<String, u32>,
    /// The node compiled for each schema path.
    compiled: HashMap<String, NodeId>,
    any_value: Option<NodeId>,
    inert: Vec<Inert>,
    /// A `$ref` was followed.
    refs: bool,
    /// The `enum`/`const` sets still to fill.
    literals: Vec<Literals>,
    /// `compile` calls in progress (a `$ref` chain).
    depth: usize,
}

/// An `enum`/`const` set and what its values are held to.
struct Literals {
    set: u32,
    /// The schema's typed node, when it names a type.
    probe: Option<NodeId>,
    /// The values, in canonical spelling.
    values: Vec<Vec<u8>>,
    keyword: &'static str,
    path: String,
}

/// The keywords that imply a type, for an untyped schema compiled without
/// `strict`.
const IMPLIED: &[(&str, &[&str])] = &[
    ("object", &["properties", "required", "additionalProperties"]),
    ("array", &["items", "minItems", "maxItems"]),
    ("string", &["pattern", "format"]),
    ("number", &["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"]),
];

/// The keyword a subschema at `path` sits under (`properties`, `items`,
/// `anyOf`, `$defs`, ...), for a refusal of the subschema itself.
fn holder(path: &str) -> &str {
    let segs: Vec<&str> = path.split('/').skip(1).collect();
    let (mut i, mut last) = (0, "schema");
    while i < segs.len() {
        last = segs[i];
        // These keywords are followed by a name or an index.
        i += if matches!(last, "properties" | "$defs" | "definitions" | "anyOf") { 2 } else { 1 };
    }
    last
}

impl Compiler {
    fn new(vocab: Arc<Vocab>, strict: bool, limits: &GrammarLimits) -> Self {
        let mut bool_set = LitTrie::new();
        bool_set.add(b"true");
        bool_set.add(b"false");
        let mut null_set = LitTrie::new();
        null_set.add(b"null");
        Self {
            vocab,
            strict,
            limits: *limits,
            doc: None,
            core: Core {
                nodes: Vec::new(),
                alts: Vec::new(),
                need: Vec::new(),
                root: 0,
                any: 0,
                lits: vec![bool_set, null_set],
                dfas: Vec::new(),
                nums: vec![NumSpec::unconstrained(false), NumSpec::unconstrained(true)],
                max_ws: limits.max_whitespace,
                max_stack: limits.max_depth,
            },
            dfa_ids: HashMap::new(),
            compiled: HashMap::new(),
            any_value: None,
            inert: Vec::new(),
            refs: false,
            literals: Vec::new(),
            depth: 0,
        }
    }

    fn push(&mut self, n: Node) -> NodeId {
        self.core.nodes.push(n);
        (self.core.nodes.len() - 1) as NodeId
    }

    fn finish(mut self, root: NodeId) -> Grammar {
        self.core.analyse();
        self.core.root = root;
        self.inert.sort();
        let dfas: usize = self.core.dfas.iter().map(|d| d.heap_bytes()).sum();
        let lits: usize = self.core.lits.iter().map(|l| l.edges.iter().map(|e| e.len() * 8 + 24).sum::<usize>()).sum();
        let tables_bytes = dfas + lits + self.core.nodes.len() * 48;
        Grammar {
            core: self.core,
            vocab: self.vocab,
            memo: Mutex::new(crate::mask::Memo::default()),
            memo_budget: self.limits.cache_bytes,
            tables_bytes,
            inert: self.inert,
        }
    }

    /// What strict refuses and the rest leaves inert: `Err` under strict,
    /// otherwise the keyword is recorded as inert and compiling goes on
    /// without it.
    fn give_up(&mut self, code: &'static str, keyword: &str, path: &str, detail: impl Into<String>) -> Result<(), Refusal> {
        if self.strict {
            return Err(refuse(code, keyword, path, detail));
        }
        self.inert.push(Inert { keyword: keyword.to_string(), path: path.to_string() });
        Ok(())
    }

    /// The node every free value (`json_object`) uses.
    fn any_value(&mut self) -> NodeId {
        if let Some(n) = self.any_value {
            return n;
        }
        let u = self.push(Node::Union(Vec::new()));
        self.any_value = Some(u);
        let parts = vec![
            self.push(Node::FreeObject { value: u }),
            self.push(Node::FreeArray),
            self.push(Node::String { re: NO_RE }),
            self.push(Node::Number { spec: NUM_FREE }),
            self.push(Node::Literal { set: LIT_BOOL }),
            self.push(Node::Literal { set: LIT_NULL }),
        ];
        self.core.nodes[u as usize] = Node::Union(parts);
        self.core.any = u;
        u
    }

    fn resolve(&self, pointer: &str) -> Result<Json, Refusal> {
        let doc = self.doc.as_ref().unwrap();
        let rest = pointer.strip_prefix('#').unwrap_or(pointer);
        let mut cur = doc;
        for raw in rest.split('/').skip(1) {
            let key = raw.replace("~1", "/").replace("~0", "~");
            cur = match cur {
                Json::Obj(_) => cur.get(&key),
                Json::Arr(a) => key.parse::<usize>().ok().and_then(|i| a.get(i)),
                _ => None,
            }
            .ok_or_else(|| refuse("unresolved_ref", "$ref", pointer, format!("`$ref` \"{pointer}\" names nothing in the schema")))?;
        }
        Ok(cur.clone())
    }

    fn compile(&mut self, path: &str) -> Result<NodeId, Refusal> {
        if let Some(&n) = self.compiled.get(path) {
            return Ok(n);
        }
        if self.depth >= self.limits.max_ref_chain {
            let most = self.limits.max_ref_chain;
            return Err(refuse("references_too_deep", "$ref", path, format!("references chain deeper than {most}")));
        }
        let id = self.push(Node::Union(Vec::new()));
        self.compiled.insert(path.to_string(), id);
        let j = self.resolve(path)?;
        self.depth += 1;
        let node = self.compile_value(&j, path);
        self.depth -= 1;
        self.core.nodes[id as usize] = Node::Union(node?);
        Ok(id)
    }

    /// Compile the schema at `path` into its alternatives.
    fn compile_value(&mut self, j: &Json, path: &str) -> Result<Vec<NodeId>, Refusal> {
        let entries = match j {
            Json::Obj(entries) => entries,
            // `true` admits any value, `false` none (an alternative never
            // entered, or a refusal when nothing else is left).
            Json::Bool(true) if self.strict => {
                return Err(refuse("missing_type", "type", path, "a strict schema names its `type`; `true` names none"));
            }
            Json::Bool(true) => return Ok(vec![self.any_value()]),
            Json::Bool(false) => return Ok(Vec::new()),
            _ => return Err(refuse("not_a_schema", holder(path), path, "a schema must be an object")),
        };
        for (k, _) in entries {
            if !ANNOTATIONS.contains(&k.as_str()) && !STRICT_SUBSET.contains(&k.as_str()) {
                self.give_up("unsupported_keyword", k, path, format!("`{k}` is outside the strict subset"))?;
            }
        }
        let siblings = |except: &[&str]| -> Vec<String> {
            entries
                .iter()
                .map(|(k, _)| k.clone())
                .filter(|k| STRICT_SUBSET.contains(&k.as_str()) && !except.contains(&k.as_str()) && !matches!(k.as_str(), "$defs" | "definitions"))
                .collect()
        };
        if let Some(r) = j.get("$ref") {
            let Some(target) = r.as_str() else {
                return Err(refuse("invalid_ref", "$ref", path, "`$ref` must be a string"));
            };
            for k in siblings(&["$ref"]) {
                self.give_up("ref_with_siblings", &k, path, format!("`{k}` beside `$ref` is not supported"))?;
            }
            // A local JSON pointer: `#`, or `#/` and a path.
            let local = target.strip_prefix('#').is_some_and(|rest| rest.is_empty() || rest.starts_with('/'));
            if !local {
                self.give_up("unsupported_ref", "$ref", path, format!("`$ref` \"{target}\" is not a local reference (#/...)"))?;
                return Ok(vec![self.any_value()]);
            }
            self.refs = true;
            return Ok(vec![self.compile(target)?]);
        }
        if let Some(any) = j.get("anyOf") {
            let items = match any {
                Json::Arr(items) if !items.is_empty() => items,
                _ => return Err(refuse("invalid_anyof", "anyOf", path, "`anyOf` must be a non-empty array")),
            };
            for k in siblings(&["anyOf"]) {
                self.give_up("anyof_with_siblings", &k, path, format!("`{k}` beside `anyOf` is not supported"))?;
            }
            let mut out = Vec::new();
            for i in 0..items.len() {
                out.push(self.compile(&format!("{path}/anyOf/{i}"))?);
            }
            return Ok(out);
        }
        let mut types: Vec<String> = match j.get("type") {
            Some(Json::Str(t)) => vec![t.clone()],
            Some(Json::Arr(ts)) if !ts.is_empty() => ts
                .iter()
                .map(|t| t.as_str().map(str::to_string).ok_or_else(|| refuse("invalid_type", "type", path, "`type` entries must be strings")))
                .collect::<Result<_, _>>()?,
            Some(_) => return Err(refuse("invalid_type", "type", path, "`type` must be a string or a non-empty array")),
            None => Vec::new(),
        };
        if let Some(t) = types.iter().find(|t| !TYPES.contains(&t.as_str())) {
            return Err(refuse("invalid_type", "type", path, format!("unknown `type` \"{t}\"")));
        }
        let literals = match (j.get("enum"), j.get("const")) {
            (Some(Json::Arr(vals)), None) => Some(vals.clone()),
            // Both: the enum values equal to the constant.
            (Some(Json::Arr(vals)), Some(c)) => {
                let canon = |v: &Json| {
                    let mut b = Vec::new();
                    v.write_canonical(&mut b);
                    b
                };
                Some(vals.iter().filter(|v| canon(v) == canon(c)).cloned().collect())
            }
            (Some(_), _) => return Err(refuse("invalid_enum", "enum", path, "`enum` must be an array")),
            (None, Some(v)) => Some(vec![v.clone()]),
            (None, None) => None,
        };
        if types.is_empty() && literals.is_none() {
            if self.strict {
                return Err(refuse("missing_type", "type", path, "a strict schema names its `type`"));
            }
            // The types the schema's keywords imply; any value when none do.
            types = IMPLIED
                .iter()
                .filter(|(_, keys)| keys.iter().any(|k| j.get(k).is_some()))
                .map(|(t, _)| t.to_string())
                .collect();
            if types.is_empty() {
                return Ok(vec![self.any_value()]);
            }
        }
        let mut typed = Vec::new();
        for t in &types {
            typed.push(self.compile_type(t, j, path)?);
        }
        let Some(values) = literals else {
            return Ok(typed);
        };
        // enum/const: the literals that are also valid under the typed
        // node, kept once the whole schema is compiled (`resolve_literals`).
        let probe = (!types.is_empty()).then(|| self.push(Node::Union(typed)));
        let values = values
            .iter()
            .map(|v| {
                let mut bytes = Vec::new();
                v.write_canonical(&mut bytes);
                bytes
            })
            .collect();
        self.core.lits.push(LitTrie::new());
        let set = (self.core.lits.len() - 1) as u32;
        let keyword = if j.get("enum").is_some() { "enum" } else { "const" };
        self.literals.push(Literals { set, probe, values, keyword, path: path.to_string() });
        Ok(vec![self.push(Node::Literal { set })])
    }

    /// Fill every `enum`/`const` set with its values the schema's type
    /// admits, each held to its typed node once, over the finished nodes.
    fn resolve_literals(&mut self) -> Result<(), Refusal> {
        self.core.analyse();
        for l in std::mem::take(&mut self.literals) {
            let mut set = LitTrie::new();
            let mut kept = 0;
            for bytes in &l.values {
                if l.probe.is_none_or(|p| crate::automaton::accepts_value(&self.core, p, bytes)) {
                    set.add(bytes);
                    kept += 1;
                }
            }
            if kept == 0 {
                let keyword = l.keyword;
                return Err(refuse("empty_enum", keyword, &l.path, format!("no `{keyword}` value fits the schema")));
            }
            self.core.lits[l.set as usize] = set;
        }
        Ok(())
    }

    /// The node of one `type` of the schema at `path`.
    fn compile_type(&mut self, t: &str, j: &Json, path: &str) -> Result<NodeId, Refusal> {
        let node = match t {
            "null" => Node::Literal { set: LIT_NULL },
            "boolean" => Node::Literal { set: LIT_BOOL },
            "string" => Node::String { re: self.string_re(j, path)? },
            "integer" | "number" => Node::Number { spec: self.number_spec(t == "integer", j, path)? },
            "array" => return self.compile_array(j, path),
            "object" => return self.compile_object(j, path),
            _ => unreachable!("types are checked against TYPES"),
        };
        Ok(self.push(node))
    }

    /// The DFA a string's `pattern` or `format` holds it to, or `NO_RE`.
    fn string_re(&mut self, j: &Json, path: &str) -> Result<u32, Refusal> {
        let pattern = match j.get("pattern") {
            None => None,
            Some(Json::Str(p)) => Some(p.clone()),
            Some(_) => return Err(refuse("invalid_pattern", "pattern", path, "`pattern` must be a string")),
        };
        let format = match j.get("format") {
            None => None,
            Some(Json::Str(f)) => Some(f.clone()),
            Some(_) => return Err(refuse("invalid_format", "format", path, "`format` must be a string")),
        };
        if let Some(p) = pattern {
            if format.is_some() {
                self.give_up("pattern_with_format", "format", path, "`format` beside `pattern` is not supported")?;
            }
            let translated = match translate_ecma(&p) {
                Ok(t) => t,
                Err(e) => {
                    self.give_up("pattern_not_regular", "pattern", path, format!("`pattern` cannot be enforced: {e}"))?;
                    return Ok(NO_RE);
                }
            };
            return match self.dfa(&translated) {
                Ok(id) => Ok(id),
                Err(DfaError::TooLarge(most)) => {
                    self.give_up("pattern_too_large", "pattern", path, format!("`pattern` compiles to more than {most} states"))?;
                    Ok(NO_RE)
                }
                Err(DfaError::Invalid(e)) => {
                    self.give_up("pattern_invalid", "pattern", path, format!("`pattern` cannot be enforced: {e}"))?;
                    Ok(NO_RE)
                }
            };
        }
        let Some(f) = format else { return Ok(NO_RE) };
        let Some(p) = format_pattern(&f) else {
            self.give_up("unsupported_format", "format", path, format!("`format` \"{f}\" is not supported"))?;
            return Ok(NO_RE);
        };
        // The nine formats' own patterns compile (regex::tests builds every
        // one), though not always within a lowered state bound.
        match self.dfa(&p) {
            Ok(id) => Ok(id),
            Err(DfaError::TooLarge(most)) => {
                self.give_up("pattern_too_large", "format", path, format!("`format` \"{f}\" compiles to more than {most} states"))?;
                Ok(NO_RE)
            }
            Err(DfaError::Invalid(e)) => unreachable!("a format's pattern compiles: {e}"),
        }
    }

    /// A number's spec: its bounds and `multipleOf`, held exactly.
    fn number_spec(&mut self, integer: bool, j: &Json, path: &str) -> Result<u32, Refusal> {
        let bound = |c: &mut Self, k: &str| -> Result<Option<i128>, Refusal> {
            match j.get(k) {
                None => Ok(None),
                Some(Json::Bool(_)) if k.starts_with("exclusive") => {
                    c.give_up("boolean_exclusive_bound", k, path, format!("`{k}` as a boolean (draft 4) is not supported"))?;
                    Ok(None)
                }
                Some(Json::Num(t)) => match parse_scaled(t) {
                    Some(x) => Ok(Some(x)),
                    None => {
                        c.give_up("number_not_representable", k, path, format!("`{k}` must be a decimal with at most 18 fraction digits, below 1e19"))?;
                        Ok(None)
                    }
                },
                Some(_) => Err(refuse("invalid_bound", k, path, format!("`{k}` must be a number"))),
            }
        };
        let tighter_lo = |a: Option<(i128, bool)>, b: Option<(i128, bool)>| match (a, b) {
            (Some(x), Some(y)) => Some(if y.0 > x.0 || (y.0 == x.0 && y.1) { y } else { x }),
            (x, None) => x,
            (None, y) => y,
        };
        let tighter_hi = |a: Option<(i128, bool)>, b: Option<(i128, bool)>| match (a, b) {
            (Some(x), Some(y)) => Some(if y.0 < x.0 || (y.0 == x.0 && y.1) { y } else { x }),
            (x, None) => x,
            (None, y) => y,
        };
        let lo = tighter_lo(bound(self, "minimum")?.map(|v| (v, false)), bound(self, "exclusiveMinimum")?.map(|v| (v, true)));
        let hi = tighter_hi(bound(self, "maximum")?.map(|v| (v, false)), bound(self, "exclusiveMaximum")?.map(|v| (v, true)));
        let mut mult = bound(self, "multipleOf")?;
        if mult.is_some_and(|m| m <= 0) {
            return Err(refuse("invalid_multiple_of", "multipleOf", path, "`multipleOf` must be positive"));
        }
        let spec = match NumSpec::new(integer, lo, hi, mult) {
            Err(NumError::StepTooLarge) => {
                self.give_up("number_not_representable", "multipleOf", path, "`multipleOf` is too large for an integer step")?;
                mult = None;
                NumSpec::new(integer, lo, hi, mult)
            }
            other => other,
        };
        if lo.is_none() && hi.is_none() && mult.is_none() {
            return Ok(if integer { NUM_FREE_INT } else { NUM_FREE });
        }
        match spec {
            Ok(s) => {
                self.core.nums.push(s);
                Ok((self.core.nums.len() - 1) as u32)
            }
            Err(_) => {
                // The bounds alone, or the step between them.
                let keyword = if NumSpec::new(integer, lo, hi, None).is_ok() {
                    "multipleOf"
                } else if j.get("minimum").is_some() {
                    "minimum"
                } else if j.get("exclusiveMinimum").is_some() {
                    "exclusiveMinimum"
                } else {
                    "maximum"
                };
                Err(refuse("empty_number_range", keyword, path, "no number satisfies the bounds"))
            }
        }
    }

    fn compile_array(&mut self, j: &Json, path: &str) -> Result<NodeId, Refusal> {
        let count = |k: &str| -> Result<Option<u32>, Refusal> {
            match j.get(k) {
                None => Ok(None),
                Some(Json::Num(t)) if parse_scaled(t).is_some_and(|v| v >= 0 && v % SCALE == 0 && v / SCALE <= u32::MAX as i128) => {
                    Ok(Some((parse_scaled(t).unwrap() / SCALE) as u32))
                }
                Some(_) => Err(refuse("invalid_item_count", k, path, format!("`{k}` must be a non-negative integer"))),
            }
        };
        let min = count("minItems")?.unwrap_or(0);
        let max = count("maxItems")?.unwrap_or(u32::MAX);
        if min > max {
            return Err(refuse("empty_item_range", "minItems", path, "`minItems` exceeds `maxItems`"));
        }
        let id = self.push(Node::Union(Vec::new()));
        let item = match j.get("items") {
            Some(Json::Obj(_) | Json::Bool(_)) => self.compile(&format!("{path}/items"))?,
            Some(_) => return Err(refuse("invalid_items", "items", path, "`items` must be a schema")),
            None => {
                self.give_up("array_without_items", "items", path, "a strict array names its `items`")?;
                self.any_value()
            }
        };
        self.core.nodes[id as usize] = Node::Array { item, min, max };
        Ok(id)
    }

    fn compile_object(&mut self, j: &Json, path: &str) -> Result<NodeId, Refusal> {
        // The schema the keys beyond `properties` hold: none when closed.
        let extra: Option<Option<String>> = match j.get("additionalProperties") {
            Some(Json::Bool(false)) => None,
            Some(Json::Bool(true)) | None | Some(Json::Obj(_)) if self.strict => {
                return Err(refuse("additional_properties_not_false", "additionalProperties", path, "a strict object sets `additionalProperties` to false"));
            }
            Some(Json::Bool(true)) | None => Some(None),
            Some(Json::Obj(_)) => Some(Some(format!("{path}/additionalProperties"))),
            Some(_) => return Err(refuse("invalid_additional_properties", "additionalProperties", path, "`additionalProperties` must be a boolean or a schema")),
        };
        let props: Vec<String> = match j.get("properties") {
            Some(Json::Obj(p)) => p.iter().map(|(k, _)| k.clone()).collect(),
            None => Vec::new(),
            Some(_) => return Err(refuse("invalid_properties", "properties", path, "`properties` must be an object")),
        };
        let required: Vec<String> = match j.get("required") {
            Some(Json::Arr(r)) => r
                .iter()
                .map(|x| x.as_str().map(str::to_string).ok_or_else(|| refuse("invalid_required", "required", path, "`required` must list strings")))
                .collect::<Result<_, _>>()?,
            None => Vec::new(),
            Some(_) => return Err(refuse("invalid_required", "required", path, "`required` must be an array")),
        };
        if self.strict {
            let mut a = props.clone();
            let mut b = required.clone();
            a.sort();
            b.sort();
            b.dedup();
            if a != b {
                return Err(refuse("not_all_required", "required", path, "a strict object lists every property in `required`"));
            }
        }
        // Every listed property is written (a valid answer either way), and
        // a required key `properties` does not list after them, holding what
        // `additionalProperties` allows.
        let mut unlisted: Vec<String> = Vec::new();
        for k in &required {
            if !props.contains(k) && !unlisted.contains(k) {
                unlisted.push(k.clone());
            }
        }
        if !unlisted.is_empty() && extra.is_none() {
            return Err(refuse("no_finite_value", "required", path, format!("`required` names \"{}\", which a closed object cannot hold", unlisted[0])));
        }
        let id = self.push(Node::Union(Vec::new()));
        if props.is_empty()
            && unlisted.is_empty()
            && let Some(extra) = extra
        {
            let value = match extra {
                Some(p) => self.compile(&p)?,
                None => self.any_value(),
            };
            self.core.nodes[id as usize] = Node::FreeObject { value };
            return Ok(id);
        }
        let mut keys = Vec::new();
        let mut values = Vec::new();
        let extra_value = match &extra {
            Some(Some(p)) if !unlisted.is_empty() => Some(self.compile(p)?),
            Some(None) if !unlisted.is_empty() => Some(self.any_value()),
            _ => None,
        };
        for k in props.iter().chain(&unlisted) {
            let mut set = LitTrie::new();
            let mut bytes = Vec::new();
            write_string(k, &mut bytes);
            set.add(&bytes);
            self.core.lits.push(set);
            keys.push((self.core.lits.len() - 1) as u32);
            if props.contains(k) {
                let esc = k.replace('~', "~0").replace('/', "~1");
                values.push(self.compile(&format!("{path}/properties/{esc}"))?);
            } else {
                values.push(extra_value.expect("an open object"));
            }
        }
        self.core.nodes[id as usize] = Node::Object { keys, values };
        Ok(id)
    }

    fn dfa(&mut self, pattern: &str) -> Result<u32, DfaError> {
        if let Some(&id) = self.dfa_ids.get(pattern) {
            return Ok(id);
        }
        let d = ByteDfa::build(pattern, self.limits.max_pattern_states)?;
        self.core.dfas.push(d);
        let id = (self.core.dfas.len() - 1) as u32;
        self.dfa_ids.insert(pattern.to_string(), id);
        Ok(id)
    }

    /// OpenAI's limits: properties, nesting, enum values, characters.
    fn check_limits(&self, doc: &Json) -> Result<(), Refusal> {
        #[derive(Default)]
        struct Tally {
            props: usize,
            enums: usize,
            chars: usize,
            depth: usize,
        }
        fn walk(j: &Json, depth: usize, t: &mut Tally) {
            let Json::Obj(entries) = j else { return };
            let is_container = matches!(j.get("type"), Some(Json::Str(s)) if s == "object" || s == "array")
                || matches!(j.get("type"), Some(Json::Arr(ts)) if ts.iter().any(|x| matches!(x.as_str(), Some("object" | "array"))));
            let depth = if is_container { depth + 1 } else { depth };
            t.depth = t.depth.max(depth);
            for (k, v) in entries {
                match (k.as_str(), v) {
                    ("properties", Json::Obj(p)) => {
                        t.props += p.len();
                        for (name, sub) in p {
                            t.chars += name.chars().count();
                            walk(sub, depth, t);
                        }
                    }
                    ("$defs" | "definitions", Json::Obj(d)) => {
                        for (name, sub) in d {
                            t.chars += name.chars().count();
                            walk(sub, 0, t);
                        }
                    }
                    ("items" | "additionalProperties", sub) => walk(sub, depth, t),
                    ("anyOf", Json::Arr(xs)) => xs.iter().for_each(|x| walk(x, depth, t)),
                    ("enum", Json::Arr(xs)) => {
                        t.enums += xs.len();
                        t.chars += xs.iter().filter_map(|x| x.as_str()).map(|s| s.chars().count()).sum::<usize>();
                    }
                    ("const", Json::Str(s)) => t.chars += s.chars().count(),
                    _ => {}
                }
            }
        }
        let mut t = Tally::default();
        walk(doc, 0, &mut t);
        if t.props > MAX_PROPERTIES {
            return Err(refuse("too_many_properties", "properties", "#", format!("the schema has {} properties; the limit is {MAX_PROPERTIES}", t.props)));
        }
        if t.depth > MAX_NESTING {
            return Err(refuse("too_deeply_nested", "nesting", "#", format!("the schema nests {} levels; the limit is {MAX_NESTING}", t.depth)));
        }
        if t.enums > MAX_ENUM_VALUES {
            return Err(refuse("too_many_enum_values", "enum values", "#", format!("the schema has {} enum values; the limit is {MAX_ENUM_VALUES}", t.enums)));
        }
        if t.chars > MAX_SCHEMA_CHARS {
            return Err(refuse(
                "too_many_characters",
                "characters",
                "#",
                format!("the schema's names, enum values and constants have {} characters; the limit is {MAX_SCHEMA_CHARS}", t.chars),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::VocabEntry;
    use crate::Matcher;

    fn vocab() -> Arc<Vocab> {
        Vocab::new((0..=255u8).map(|b| VocabEntry::Text(vec![b])).chain([VocabEntry::Control]).collect(), &[256])
    }

    fn compile(schema: &str, strict: bool) -> Result<Arc<Grammar>, Refusal> {
        Grammar::json_schema(vocab(), schema, strict)
    }

    /// The refusal's code and the keyword or limit it names, or "ok".
    fn named(schema: &str, strict: bool) -> (&'static str, String) {
        compile(schema, strict).err().map_or(("ok", String::new()), |r| (r.code, r.keyword))
    }

    fn obj(inner: &str) -> String {
        format!(r#"{{"type":"object","properties":{{"a":{inner}}},"required":["a"],"additionalProperties":false}}"#)
    }

    fn accepts(g: &Arc<Grammar>, text: &str) -> bool {
        let mut m = Matcher::new(g.clone());
        m.accept_bytes(text.as_bytes()) && m.accept(256)
    }

    /// Every refusal this module can give, with a schema that gives it and
    /// the keyword or limit it must name; `strict` is the mode it is given
    /// in. `every_refusal_code_is_pinned` holds this list to the source.
    fn refusals() -> Vec<(String, bool, &'static str, &'static str)> {
        let deep_refs = {
            let defs: Vec<String> = (0..300).map(|i| format!(r##""d{i}":{{"$ref":"#/$defs/d{}"}}"##, i + 1)).collect();
            format!(r##"{{"type":"object","properties":{{"a":{{"$ref":"#/$defs/d0"}}}},"required":["a"],"additionalProperties":false,"$defs":{{{},"d300":{{"type":"null"}}}}}}"##, defs.join(","))
        };
        let props: Vec<String> = (0..5001).map(|i| format!(r#""p{i}":{{"type":"null"}}"#)).collect();
        let names: Vec<String> = (0..5001).map(|i| format!(r#""p{i}""#)).collect();
        let many_props = format!(r#"{{"type":"object","properties":{{{}}},"required":[{}],"additionalProperties":false}}"#, props.join(","), names.join(","));
        let mut deep = r#"{"type":"string"}"#.to_string();
        for _ in 0..11 {
            deep = obj(&deep);
        }
        let vals: Vec<String> = (0..1001).map(|i| format!(r#""v{i}""#)).collect();
        let many_enums = obj(&format!(r#"{{"type":"string","enum":[{}]}}"#, vals.join(",")));
        let long = obj(&format!(r#"{{"type":"string","enum":["{}"]}}"#, "x".repeat(MAX_SCHEMA_CHARS)));
        let endless = r##"{"type":"object","properties":{"child":{"$ref":"#"}},"required":["child"],"additionalProperties":false}"##;
        // A chain of required objects through `$defs`: OpenAI's nesting
        // limit reads the schema's text, the grammar's depth its answer.
        let chain = |levels: usize| {
            let defs: Vec<String> = (0..levels)
                .map(|i| format!(r##""d{i}":{{"type":"object","properties":{{"x":{{"$ref":"#/$defs/d{}"}}}},"required":["x"],"additionalProperties":false}}"##, i + 1))
                .collect();
            format!(r##"{{"type":"object","properties":{{"a":{{"$ref":"#/$defs/d0"}}}},"required":["a"],"additionalProperties":false,"$defs":{{{},"d{levels}":{{"type":"null"}}}}}}"##, defs.join(","))
        };
        // Exponential in its DFA: 2^17 states for a match 16 bytes back.
        let wide = obj(r#"{"type":"string","pattern":"a[ab]{16}$"}"#);
        let s = |x: &str| x.to_string();
        vec![
            (s("{"), true, "invalid_json", "schema"),
            (s(r#"{"anyOf":[{"type":"object"}]}"#), true, "root_anyof", "anyOf"),
            (s(r#"{"type":"string"}"#), true, "root_not_object", "type"),
            (s(endless), true, "no_finite_value", "$ref"),
            (s(r#"{"type":"object","required":["x"],"additionalProperties":false}"#), false, "no_finite_value", "required"),
            (deep_refs, true, "references_too_deep", "$ref"),
            // 126 levels: the root's two frames past the default 128, before
            // the compiler's recursion reaches its own default bound.
            (chain(DEFAULT_MAX_DEPTH - 2), true, "exceeds_max_depth", "depth"),
            (obj(r##"{"$ref":"#/$defs/missing"}"##), true, "unresolved_ref", "$ref"),
            (obj(r#"{"description":"no type"}"#), true, "missing_type", "type"),
            (obj("true"), true, "missing_type", "type"),
            (obj("3"), true, "not_a_schema", "properties"),
            (obj(r#"{"type":"string","allOf":[]}"#), true, "unsupported_keyword", "allOf"),
            (obj(r#"{"type":"string","not":{}}"#), true, "unsupported_keyword", "not"),
            (obj(r#"{"type":"string","if":{}}"#), true, "unsupported_keyword", "if"),
            (obj(r#"{"type":"string","then":{}}"#), true, "unsupported_keyword", "then"),
            (obj(r#"{"type":"string","else":{}}"#), true, "unsupported_keyword", "else"),
            (obj(r#"{"type":"object","properties":{},"additionalProperties":false,"dependentRequired":{}}"#), true, "unsupported_keyword", "dependentRequired"),
            (obj(r#"{"type":"object","properties":{},"additionalProperties":false,"dependentSchemas":{}}"#), true, "unsupported_keyword", "dependentSchemas"),
            (obj(r#"{"oneOf":[{"type":"string"}]}"#), true, "unsupported_keyword", "oneOf"),
            (obj(r#"{"type":"string","minLength":3}"#), true, "unsupported_keyword", "minLength"),
            (obj(r#"{"$ref":3}"#), true, "invalid_ref", "$ref"),
            (obj(r##"{"$ref":"#/$defs/x","type":"string"}"##), true, "ref_with_siblings", "type"),
            (obj(r#"{"$ref":"https://example.com/s.json"}"#), true, "unsupported_ref", "$ref"),
            (obj(r##"{"$ref":"#anchor"}"##), true, "unsupported_ref", "$ref"),
            (obj(r#"{"anyOf":[]}"#), true, "invalid_anyof", "anyOf"),
            (obj(r#"{"anyOf":[{"type":"string"}],"type":"string"}"#), true, "anyof_with_siblings", "type"),
            (obj(r#"{"type":5}"#), true, "invalid_type", "type"),
            (obj(r#"{"type":"text"}"#), true, "invalid_type", "type"),
            (obj(r#"{"enum":"a"}"#), true, "invalid_enum", "enum"),
            (obj(r#"{"type":"string","enum":[1,2]}"#), true, "empty_enum", "enum"),
            (obj(r#"{"type":"string","const":1}"#), true, "empty_enum", "const"),
            (obj(r#"{"type":"string","pattern":5}"#), true, "invalid_pattern", "pattern"),
            (obj(r#"{"type":"string","format":5}"#), true, "invalid_format", "format"),
            (obj(r#"{"type":"string","pattern":"^a","format":"email"}"#), true, "pattern_with_format", "format"),
            (obj(r#"{"type":"string","pattern":"(a)\\1"}"#), true, "pattern_not_regular", "pattern"),
            (obj(r#"{"type":"string","pattern":"(a"}"#), true, "pattern_invalid", "pattern"),
            (wide, true, "pattern_too_large", "pattern"),
            (obj(r#"{"type":"string","format":"uri"}"#), true, "unsupported_format", "format"),
            (obj(r#"{"type":"number","exclusiveMinimum":true}"#), true, "boolean_exclusive_bound", "exclusiveMinimum"),
            (obj(r#"{"type":"number","minimum":1e300}"#), true, "number_not_representable", "minimum"),
            (obj(r#"{"type":"integer","multipleOf":999999999999999999.999999999999999999}"#), true, "number_not_representable", "multipleOf"),
            (obj(r#"{"type":"number","minimum":"1"}"#), true, "invalid_bound", "minimum"),
            (obj(r#"{"type":"number","multipleOf":0}"#), true, "invalid_multiple_of", "multipleOf"),
            (obj(r#"{"type":"integer","minimum":5,"maximum":4}"#), true, "empty_number_range", "minimum"),
            (obj(r#"{"type":"integer","minimum":1,"maximum":3,"multipleOf":5}"#), true, "empty_number_range", "multipleOf"),
            (obj(r#"{"type":"array","items":{"type":"null"},"minItems":-1}"#), true, "invalid_item_count", "minItems"),
            (obj(r#"{"type":"array","items":{"type":"null"},"minItems":3,"maxItems":2}"#), true, "empty_item_range", "minItems"),
            (obj(r#"{"type":"array","items":[{"type":"null"}]}"#), true, "invalid_items", "items"),
            (obj(r#"{"type":"array"}"#), true, "array_without_items", "items"),
            (s(r#"{"type":"object","properties":{}}"#), true, "additional_properties_not_false", "additionalProperties"),
            (s(r#"{"type":"object","properties":{},"additionalProperties":3}"#), false, "invalid_additional_properties", "additionalProperties"),
            (s(r#"{"type":"object","properties":[],"additionalProperties":false}"#), true, "invalid_properties", "properties"),
            (s(r#"{"type":"object","properties":{},"required":{},"additionalProperties":false}"#), true, "invalid_required", "required"),
            (s(r#"{"type":"object","properties":{"a":{"type":"string"}},"required":[],"additionalProperties":false}"#), true, "not_all_required", "required"),
            (many_props, true, "too_many_properties", "properties"),
            (deep, true, "too_deeply_nested", "nesting"),
            (many_enums, true, "too_many_enum_values", "enum values"),
            (long, true, "too_many_characters", "characters"),
        ]
    }

    #[test]
    fn every_refusal_names_its_keyword_or_limit() {
        for (schema, strict, code, keyword) in refusals() {
            let got = named(&schema, strict);
            assert_eq!(got, (code, keyword.to_string()), "{}", &schema[..schema.len().min(160)]);
        }
    }

    #[test]
    fn every_refusal_code_is_pinned() {
        let pinned: Vec<&str> = refusals().iter().map(|r| r.2).collect();
        let source = include_str!("schema.rs");
        let tests_start = source.find("#[cfg(test)]").unwrap();
        let mut codes = Vec::new();
        for call in ["refuse(\"", "give_up(\""] {
            for (at, _) in source[..tests_start].match_indices(call) {
                let rest = &source[at + call.len()..];
                codes.push(&rest[..rest.find('"').unwrap()]);
            }
        }
        assert!(codes.len() > 40, "{codes:?}");
        for code in codes {
            assert!(pinned.contains(&code), "refusal `{code}` has no test in refusals()");
        }
    }

    #[test]
    fn json_object_and_every_keyword_of_the_strict_subset_compile() {
        assert!(accepts(&Grammar::json_object(vocab()), r#"{"a":[1,{"b":null}]}"#));
        let formats: Vec<String> = FORMATS.iter().map(|f| format!(r#""{f}":{{"type":"string","format":"{f}"}}"#)).collect();
        let names: Vec<String> = FORMATS.iter().map(|f| format!(r#""{f}""#)).collect();
        let schema = format!(
            r##"{{"$schema":"https://json-schema.org/draft/2020-12/schema","title":"All","description":"every keyword","type":"object",
            "properties":{{
              "name":{{"type":"string","pattern":"^[a-z]+$","description":"d"}},
              "kind":{{"type":"string","enum":["a","b"]}},
              "fixed":{{"const":"x"}},
              "count":{{"type":"integer","minimum":1,"maximum":10}},
              "ratio":{{"type":"number","exclusiveMinimum":0,"exclusiveMaximum":1}},
              "price":{{"type":"number","multipleOf":0.01}},
              "note":{{"type":["string","null"]}},
              "tags":{{"type":"array","items":{{"type":"string"}},"minItems":1,"maxItems":3}},
              "shape":{{"anyOf":[{{"$ref":"#/$defs/circle"}},{{"$ref":"#/definitions/square"}}]}},
              "next":{{"anyOf":[{{"$ref":"#"}},{{"type":"null"}}]}},
              "flag":{{"type":"boolean"}},
              "nothing":{{"type":"null"}},
              "formats":{{"type":"object","properties":{{{}}},"required":[{}],"additionalProperties":false}}
            }},
            "required":["name","kind","fixed","count","ratio","price","note","tags","shape","next","flag","nothing","formats"],
            "additionalProperties":false,
            "$defs":{{"circle":{{"type":"object","properties":{{"r":{{"type":"number"}}}},"required":["r"],"additionalProperties":false}}}},
            "definitions":{{"square":{{"type":"object","properties":{{"side":{{"type":"number"}}}},"required":["side"],"additionalProperties":false}}}}}}"##,
            formats.join(","),
            names.join(",")
        );
        let g = compile(&schema, true).unwrap_or_else(|e| panic!("{e}"));
        assert!(g.inert().is_empty());
        let doc = r#"{"name":"ab","kind":"b","fixed":"x","count":10,"ratio":0.5,"price":3.07,"note":null,"tags":["t"],"shape":{"side":2},"next":null,"flag":true,"nothing":null,"formats":{"date-time":"2026-11-12T10:00:00Z","time":"10:00:00+01:00","date":"2026-02-28","duration":"P1DT2H","email":"a@b.example","hostname":"b.example","ipv4":"10.0.0.1","ipv6":"::1","uuid":"123e4567-e89b-12d3-a456-426614174000"}}"#;
        assert!(accepts(&g, doc));
        for bad in [r#""count":11"#, r#""ratio":1,"#, r#""price":3.071"#, r#""kind":"c""#, r#""tags":[]"#, r#""ipv4":"10.0.0.256""#] {
            let (k, _) = bad.split_once(':').unwrap();
            let start = doc.find(k).unwrap();
            let end = start + doc[start..].find([',', '}']).unwrap();
            let end = if bad.ends_with(',') { end + 1 } else { end };
            let mutated = format!("{}{}{}", &doc[..start], bad, &doc[end..]);
            assert!(!accepts(&g, &mutated), "{mutated}");
        }
    }

    #[test]
    fn without_strict_what_cannot_be_enforced_is_inert_and_named() {
        let cases: &[(&str, &[&str])] = &[
            (r#"{"type":"string","minLength":3}"#, &["minLength"]),
            (r#"{"type":"string","pattern":"(a)\\1"}"#, &["pattern"]),
            (r#"{"type":"string","format":"uri"}"#, &["format"]),
            (r#"{"type":"string","pattern":"^a","format":"email"}"#, &["format"]),
            (r#"{"type":"number","exclusiveMinimum":true,"minimum":1e300}"#, &["exclusiveMinimum", "minimum"]),
            (r#"{"oneOf":[{"type":"string"}],"allOf":[]}"#, &["allOf", "oneOf"]),
            (r#"{"$ref":"https://example.com/s.json"}"#, &["$ref"]),
            (r#"{"anyOf":[{"type":"string"}],"type":"string"}"#, &["type"]),
        ];
        for (inner, inert) in cases {
            let schema = obj(inner);
            assert!(named(&schema, true).0 != "ok", "strict refuses {inner}");
            let g = compile(&schema, false).unwrap_or_else(|e| panic!("{inner}: {e}"));
            let got: Vec<&str> = g.inert().iter().map(|i| i.keyword.as_str()).collect();
            assert_eq!(&got, inert, "{inner}");
            assert!(g.inert().iter().all(|i| i.path == "#/properties/a"));
        }
        // An inert `minLength` leaves the string free; the rest still holds.
        let g = compile(&obj(r#"{"type":"string","minLength":3,"pattern":"^x"}"#), false).unwrap();
        assert!(accepts(&g, r#"{"a":"x"}"#) && !accepts(&g, r#"{"a":"y"}"#));
    }

    #[test]
    fn without_strict_a_looser_schema_is_served_with_valid_answers() {
        // A root of any type, an untyped schema given the types its keywords
        // imply, an open object writing the keys it names, a map.
        let g = compile(r#"{"type":"array","items":{"type":"integer"}}"#, false).unwrap();
        assert!(accepts(&g, "[1, 2]") && !accepts(&g, "{}"));
        let g = compile(r#"{"anyOf":[{"type":"string"},{"type":"null"}]}"#, false).unwrap();
        assert!(accepts(&g, r#""x""#) && accepts(&g, "null"));
        let g = compile(r#"{"properties":{"a":{"maximum":3}}}"#, false).unwrap();
        assert!(accepts(&g, r#"{"a":3}"#) && !accepts(&g, r#"{"a":4}"#) && !accepts(&g, r#"{"a":"x"}"#));
        assert!(!accepts(&g, r#"{}"#), "every listed property is written");
        assert!(g.inert().is_empty());
        let g = compile(r#"{"type":"object","properties":{"a":{"type":"string"}},"required":["a","b"]}"#, false).unwrap();
        assert!(accepts(&g, r#"{"a":"x","b":[true]}"#) && !accepts(&g, r#"{"a":"x"}"#));
        let g = compile(r#"{"type":"object","additionalProperties":{"type":"integer"}}"#, false).unwrap();
        assert!(accepts(&g, r#"{"x":1,"y":2}"#) && accepts(&g, "{}") && !accepts(&g, r#"{"x":"1"}"#));
        let g = compile(r#"{"type":"object"}"#, false).unwrap();
        assert!(accepts(&g, r#"{"x":{"y":[1,"z"]}}"#));
        let g = compile(r#"{"type":"object","properties":{"a":{"type":"string","enum":["p","q"],"const":"q"}},"required":["a"],"additionalProperties":false}"#, true).unwrap();
        assert!(accepts(&g, r#"{"a":"q"}"#) && !accepts(&g, r#"{"a":"p"}"#), "enum beside const is their intersection");
    }
}
