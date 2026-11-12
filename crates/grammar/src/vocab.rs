//! The vocabulary as bytes: each token's exact bytes, decoded once at load
//! from the tokenizer, and a byte trie over them that a mask walks.

use std::collections::HashMap;
use std::sync::Arc;

use crate::string::StrState;

/// What a vocabulary id is to the grammar.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VocabEntry {
    /// An ordinary token: its exact bytes.
    Text(Vec<u8>),
    /// An added or special token (`<|im_end|>`, `<think>`, ...): never part
    /// of a structured answer's text; a stop id ends it.
    Control,
    /// A logits column the tokenizer has no token for.
    Absent,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct TrieNode {
    pub byte: u8,
    /// Bytes from the root to this node, inclusive (the root's children: 1).
    pub depth: u16,
    /// One past the last node of this node's subtree, in preorder.
    pub end: u32,
    pub tok_start: u32,
    pub tok_len: u32,
}

/// A byte trie in preorder: a node's subtree is `[i, nodes[i].end)`.
#[derive(Default)]
pub(crate) struct Trie {
    pub nodes: Vec<TrieNode>,
    pub toks: Vec<u32>,
    pub max_depth: usize,
}

impl Trie {
    /// Build from `(bytes, id)` pairs; empty byte strings are skipped.
    pub fn build(mut entries: Vec<(&[u8], u32)>) -> Trie {
        entries.retain(|(b, _)| !b.is_empty());
        entries.sort_unstable();
        let mut t = Trie::default();
        let mut path: Vec<usize> = Vec::new(); // node index per depth-1
        let mut prev: &[u8] = &[];
        for (bytes, id) in entries {
            let common = prev.iter().zip(bytes).take_while(|(a, b)| a == b).count();
            while path.len() > common {
                let i = path.pop().unwrap();
                t.nodes[i].end = t.nodes.len() as u32;
            }
            for (k, &b) in bytes.iter().enumerate().skip(common) {
                path.push(t.nodes.len());
                t.nodes.push(TrieNode { byte: b, depth: (k + 1) as u16, end: 0, tok_start: 0, tok_len: 0 });
            }
            let last = *path.last().unwrap();
            if t.nodes[last].tok_len == 0 {
                t.nodes[last].tok_start = t.toks.len() as u32;
            }
            t.nodes[last].tok_len += 1;
            t.toks.push(id);
            t.max_depth = t.max_depth.max(bytes.len());
            prev = bytes;
        }
        while let Some(i) = path.pop() {
            t.nodes[i].end = t.nodes.len() as u32;
        }
        t
    }

    #[inline]
    pub fn tokens(&self, i: usize) -> &[u32] {
        let n = &self.nodes[i];
        &self.toks[n.tok_start as usize..(n.tok_start + n.tok_len) as usize]
    }

    pub fn heap_bytes(&self) -> usize {
        self.nodes.len() * std::mem::size_of::<TrieNode>() + self.toks.len() * 4
    }
}

/// The lexer-only result for a string state: which tokens stay inside the
/// string, and, for those that close it, what follows the closing quote.
pub(crate) struct StrTable {
    pub inside: crate::TokenSet,
    /// Tokens whose closing quote is their last byte.
    pub close_exact: Vec<u32>,
    /// Tokens that close the string and go on: their bytes after the quote.
    pub rest: Trie,
}

impl StrTable {
    pub fn heap_bytes(&self) -> usize {
        self.inside.heap_bytes() + self.close_exact.len() * 4 + self.rest.heap_bytes()
    }
}

pub struct Vocab {
    n: usize,
    bytes: Vec<u8>,
    offs: Vec<u32>,
    text: Vec<bool>,
    stops: Vec<u32>,
    pub(crate) trie: Trie,
    /// Free strings' tables, one per lexer state a free string sits in at a
    /// token boundary almost always, built here once: vocabulary-wide,
    /// shared by every grammar, never grown afterwards. The rarer states
    /// (inside a `\u` escape, thousands of them) walk the trie instead.
    pub(crate) free_tables: HashMap<StrState, Arc<StrTable>>,
}

impl Vocab {
    /// `entries[id]` for every logits column `0..entries.len()`; `stops` are
    /// the ids that end a generation (allowed only where the answer is
    /// complete).
    pub fn new(entries: Vec<VocabEntry>, stops: &[u32]) -> Arc<Vocab> {
        let n = entries.len();
        let mut bytes = Vec::new();
        let mut offs = Vec::with_capacity(n + 1);
        let mut text = Vec::with_capacity(n);
        offs.push(0);
        for e in &entries {
            if let VocabEntry::Text(b) = e {
                bytes.extend_from_slice(b);
                text.push(!b.is_empty());
            } else {
                text.push(false);
            }
            offs.push(bytes.len() as u32);
        }
        let trie = {
            let pairs: Vec<(&[u8], u32)> = (0..n)
                .filter(|&i| text[i])
                .map(|i| (&bytes[offs[i] as usize..offs[i + 1] as usize], i as u32))
                .collect();
            Trie::build(pairs)
        };
        let mut v = Vocab {
            n,
            bytes,
            offs,
            text,
            stops: stops.iter().copied().filter(|&s| (s as usize) < n).collect(),
            trie,
            free_tables: HashMap::new(),
        };
        // The states a free string sits in at a token boundary almost
        // always: no escape pending, any UTF-8 debt, and after a backslash.
        // Each is one walk of the whole trie, paid here once per tokenizer.
        let mut eager = Vec::new();
        for utf8 in 0..=7u8 {
            eager.push(StrState { esc: 0, utf8, hex: 0, dfa: 0 });
        }
        eager.push(StrState { esc: 1, utf8: 0, hex: 0, dfa: 0 });
        let tables = eager.into_iter().map(|s| (s, Arc::new(crate::mask::build_str_table(&v, None, s)))).collect();
        v.free_tables = tables;
        Arc::new(v)
    }

    /// The logits row's width.
    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    #[inline]
    pub fn token_bytes(&self, id: u32) -> Option<&[u8]> {
        let i = id as usize;
        if i < self.n && self.text[i] {
            Some(&self.bytes[self.offs[i] as usize..self.offs[i + 1] as usize])
        } else {
            None
        }
    }

    pub fn is_stop(&self, id: u32) -> bool {
        self.stops.contains(&id)
    }

    pub fn stops(&self) -> &[u32] {
        &self.stops
    }

    pub fn text_token_count(&self) -> usize {
        self.text.iter().filter(|t| **t).count()
    }

    pub fn trie_node_count(&self) -> usize {
        self.trie.nodes.len()
    }

    /// Memory held: byte table, trie and the free-string tables.
    pub fn heap_bytes(&self) -> usize {
        let tables: usize = self.free_tables.values().map(|t| t.heap_bytes()).sum();
        self.bytes.len() + self.offs.len() * 4 + self.text.len() + self.trie.heap_bytes() + tables
    }

    pub fn free_table_count(&self) -> usize {
        self.free_tables.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trie_preorder_subtrees() {
        let t = Trie::build(vec![(b"ab", 0), (b"a", 1), (b"b", 2), (b"abc", 3), (b"ab", 4)]);
        let shape: Vec<(u8, u16, u32)> = t.nodes.iter().map(|n| (n.byte, n.depth, n.end)).collect();
        assert_eq!(shape, vec![(b'a', 1, 3), (b'b', 2, 3), (b'c', 3, 3), (b'b', 1, 4)]);
        assert_eq!(t.tokens(0), &[1]);
        assert_eq!(t.tokens(1), &[0, 4]);
        assert_eq!(t.tokens(2), &[3]);
    }
}
