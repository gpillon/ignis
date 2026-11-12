//! Compiled grammars kept per loaded tokenizer: one per distinct schema, so a
//! request that sends a schema already seen pays nothing for it, in bounded
//! memory.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::json::Json;
use crate::schema::{Grammar, Refusal};
use crate::vocab::Vocab;

/// The memory a [`GrammarCache`] keeps its grammars in by default: room for
/// hundreds of SDK-sized schemas with their masks.
pub const DEFAULT_CACHE_BYTES: usize = 256 << 20;

/// The most schemas a cache keeps, whatever their size (each lookup walks
/// them to account their memory).
const MAX_ENTRIES: usize = 4096;

/// The grammars compiled against one tokenizer's [`Vocab`].
///
/// A schema is the same schema when its JSON is, whitespace aside, with the
/// same `strict` flag; its refusal is kept too. Past the capacity, the least
/// recently used schemas are dropped; a request still holding one keeps it
/// alive. A grammar's own caches grow while requests use it (each bounded in
/// entries), and the bound is applied again at every lookup.
pub struct GrammarCache {
    vocab: Arc<Vocab>,
    capacity: usize,
    json_object: OnceLock<Arc<Grammar>>,
    inner: Mutex<Inner>,
    compiles: AtomicU64,
}

#[derive(Default)]
struct Inner {
    entries: HashMap<(Vec<u8>, bool), Entry>,
    clock: u64,
}

struct Entry {
    result: Result<Arc<Grammar>, Refusal>,
    used: u64,
}

impl Entry {
    fn heap_bytes(&self, key: &[u8]) -> usize {
        key.len()
            + match &self.result {
                Ok(g) => g.heap_bytes(),
                Err(r) => r.detail.len() + r.path.len() + r.keyword.len(),
            }
    }
}

impl GrammarCache {
    /// A cache over `vocab` keeping at most about `capacity_bytes` of
    /// compiled grammars.
    pub fn new(vocab: Arc<Vocab>, capacity_bytes: usize) -> Self {
        Self { vocab, capacity: capacity_bytes, json_object: OnceLock::new(), inner: Mutex::default(), compiles: AtomicU64::new(0) }
    }

    pub fn vocab(&self) -> &Arc<Vocab> {
        &self.vocab
    }

    /// `json_object`'s grammar, compiled on first use and kept.
    pub fn json_object(&self) -> Arc<Grammar> {
        self.json_object
            .get_or_init(|| {
                self.compiles.fetch_add(1, Ordering::Relaxed);
                Grammar::json_object(self.vocab.clone())
            })
            .clone()
    }

    /// The grammar of a `json_schema` request's schema, compiled once per
    /// distinct schema and `strict` flag.
    pub fn json_schema(&self, schema: &str, strict: bool) -> Result<Arc<Grammar>, Refusal> {
        // Whitespace and number spelling aside, key order kept: the same
        // schema is the same key.
        let key = match Json::parse(schema) {
            Ok(j) => {
                let mut b = Vec::new();
                j.write_canonical(&mut b);
                b
            }
            Err(_) => schema.as_bytes().to_vec(),
        };
        let key = (key, strict);
        {
            let mut inner = self.inner.lock().unwrap();
            inner.clock += 1;
            let now = inner.clock;
            if let Some(e) = inner.entries.get_mut(&key) {
                e.used = now;
                let result = e.result.clone();
                self.evict(&mut inner);
                return result;
            }
        }
        // Compiled outside the lock: a slow schema does not hold up the
        // lookups of others.
        self.compiles.fetch_add(1, Ordering::Relaxed);
        let result = Grammar::json_schema(self.vocab.clone(), schema, strict);
        let mut inner = self.inner.lock().unwrap();
        inner.clock += 1;
        let used = inner.clock;
        inner.entries.insert(key, Entry { result: result.clone(), used });
        self.evict(&mut inner);
        result
    }

    /// Drop the least recently used schemas until the cache fits, keeping
    /// the most recent one.
    fn evict(&self, inner: &mut Inner) {
        let mut total: usize = inner.entries.iter().map(|((k, _), e)| e.heap_bytes(k)).sum();
        while inner.entries.len() > 1 && (total > self.capacity || inner.entries.len() > MAX_ENTRIES) {
            let oldest = inner.entries.iter().min_by_key(|(_, e)| e.used).map(|(k, _)| k.clone()).unwrap();
            let e = inner.entries.remove(&oldest).unwrap();
            total -= e.heap_bytes(&oldest.0);
        }
    }

    /// How many grammars this cache has compiled (`json_object` included):
    /// a schema sent again is not compiled again.
    pub fn compiles(&self) -> u64 {
        self.compiles.load(Ordering::Relaxed)
    }

    /// How many schemas the cache holds.
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
