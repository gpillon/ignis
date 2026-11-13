//! The repair's permitted set: every token valid in the current state, as a
//! mask over the whole vocabulary.
//!
//! A string state (the widest: almost the whole vocabulary fits inside a free
//! string) is answered from a lexer-only table per string state: the tokens
//! that stay inside the string, plus, for the tokens that close it, a trie of
//! what follows the closing quote, walked from the configuration below the
//! string. Free strings' tables are vocabulary-wide and built at load; a
//! pattern's are built on a state's first visit and kept in the grammar.
//! Every other state walks the vocabulary's byte trie from its root, pruned
//! at the first byte the automaton refuses. Whole masks are cached by state.

use std::collections::HashMap;
use std::sync::Arc;

use crate::automaton::{Config, Frame, State};
use crate::regex::ByteDfa;
use crate::schema::{Core, Grammar, NO_RE};
use crate::string::{self, StrState, StrStep};
use crate::vocab::{StrTable, Trie, Vocab};
use crate::TokenSet;

/// What a grammar remembers between requests: pattern-string tables (one
/// per DFA state and UTF-8 debt reached) and whole masks (31 KB each at
/// 248,320), together in at most the grammar's
/// [`GrammarLimits::cache_bytes`](crate::GrammarLimits::cache_bytes): past
/// it, what is built is used and not kept.
#[derive(Default)]
pub(crate) struct Memo {
    pub str_tables: HashMap<(u32, StrState), Arc<StrTable>>,
    pub masks: HashMap<Vec<Config>, Arc<TokenSet>>,
    /// What the two hold, counted as they are filled.
    bytes: usize,
}

impl Memo {
    pub fn heap_bytes(&self) -> usize {
        self.bytes
    }

    /// Whether `bytes` more fit in `budget`; counted when they do.
    fn keeps(&mut self, bytes: usize, budget: usize) -> bool {
        let kept = self.bytes.saturating_add(bytes) <= budget;
        if kept {
            self.bytes += bytes;
        }
        kept
    }
}

/// Per-matcher scratch for the trie walks.
#[derive(Default)]
pub(crate) struct Walker {
    levels: Vec<State>,
    forks: Vec<Config>,
}

pub(crate) fn build_str_table(v: &Vocab, dfa: Option<&ByteDfa>, st0: StrState) -> StrTable {
    let trie = &v.trie;
    let mut inside = TokenSet::new(v.len());
    let mut close_exact = Vec::new();
    let mut rest: Vec<(&[u8], u32)> = Vec::new();
    let mut states = vec![st0; trie.max_depth + 1];
    let mut i = 0;
    while i < trie.nodes.len() {
        let node = trie.nodes[i];
        let d = node.depth as usize;
        match string::step(dfa, states[d - 1], node.byte) {
            StrStep::Inside(s) => {
                states[d] = s;
                for &t in trie.tokens(i) {
                    inside.insert(t);
                }
                i += 1;
            }
            StrStep::Close => {
                for j in i..node.end as usize {
                    for &t in trie.tokens(j) {
                        let bytes = v.token_bytes(t).unwrap();
                        if bytes.len() == d {
                            close_exact.push(t);
                        } else {
                            rest.push((&bytes[d..], t));
                        }
                    }
                }
                i = node.end as usize;
            }
            StrStep::Reject => i = node.end as usize,
        }
    }
    StrTable { inside, close_exact, rest: Trie::build(rest) }
}

/// Mark every token of `trie` whose bytes `start` accepts.
pub(crate) fn walk(core: &Core, trie: &Trie, start: &Config, out: &mut TokenSet, w: &mut Walker) {
    if w.levels.len() < trie.max_depth + 1 {
        w.levels.resize_with(trie.max_depth + 1, State::default);
    }
    w.levels[0].configs.clear();
    w.levels[0].configs.push(start.clone());
    let mut i = 0;
    while i < trie.nodes.len() {
        let node = trie.nodes[i];
        let d = node.depth as usize;
        let (lo, hi) = w.levels.split_at_mut(d);
        let cur = &mut hi[0];
        cur.clone_from(&lo[d - 1]);
        if !cur.step(core, node.byte, &mut w.forks) {
            i = node.end as usize;
            continue;
        }
        for &t in trie.tokens(i) {
            out.insert(t);
        }
        i += 1;
    }
}

impl Grammar {
    /// The lexer-only table of a string state, or `None` when the state has
    /// none and the trie is walked instead. A free string's come from the
    /// vocabulary (built at load, never grown); a pattern's are built on the
    /// state's first visit and kept while the grammar's caches have room.
    pub(crate) fn str_table(&self, re: u32, st: StrState) -> Option<Arc<StrTable>> {
        if re == NO_RE {
            return self.vocab.free_tables.get(&st).cloned();
        }
        if let Some(t) = self.memo.lock().unwrap().str_tables.get(&(re, st)) {
            return Some(t.clone());
        }
        let t = Arc::new(build_str_table(&self.vocab, Some(&self.core.dfas[re as usize]), st));
        let mut memo = self.memo.lock().unwrap();
        if !memo.str_tables.contains_key(&(re, st)) && memo.keeps(t.heap_bytes(), self.memo_budget) {
            memo.str_tables.insert((re, st), t.clone());
        }
        Some(t)
    }

    fn config_mask(&self, c: &Config, out: &mut TokenSet, w: &mut Walker) {
        if let Some(&Frame::Str { re, st }) = c.frames.last()
            && let Some(table) = self.str_table(re, st)
        {
            out.union_with(&table.inside);
            for &t in &table.close_exact {
                out.insert(t);
            }
            let mut closed = c.clone();
            closed.frames.pop();
            closed.ws = 0;
            walk(&self.core, &table.rest, &closed, out, w);
            return;
        }
        walk(&self.core, &self.vocab.trie, c, out, w);
    }

    /// The mask for `state`; `true` when it came from the cache.
    pub(crate) fn mask(&self, state: &State, out: &mut TokenSet, w: &mut Walker) -> bool {
        if let Some(m) = self.memo.lock().unwrap().masks.get(&state.configs) {
            out.copy_from(m);
            return true;
        }
        out.clear();
        for c in &state.configs {
            self.config_mask(c, out, w);
        }
        if state.accepting() {
            for &s in self.vocab.stops() {
                out.insert(s);
            }
        }
        let mut memo = self.memo.lock().unwrap();
        if !memo.masks.contains_key(&state.configs) && memo.keeps(out.heap_bytes() + 64, self.memo_budget) {
            memo.masks.insert(state.configs.clone(), Arc::new(out.clone()));
        }
        false
    }
}
