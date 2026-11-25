//! The radix index: a page-granular prefix tree over the match key, with
//! **sparse recurrent-state checkpoints** hung on its nodes.
//!
//! This is ignis's answer to SGLang's RadixAttention on a hybrid model
//! (Qwen3.8-27B: 16 GQA layers + 48 GDN layers). The attention half is the
//! classic radix cache: one node per KV page, found by the prompt's cumulative
//! [`MatchKey`] at each page boundary, so any request sharing a head with any
//! earlier one finds it in one forward walk. The GDN half is what a plain
//! radix cache cannot do: a recurrent state exists only where it was captured
//! (core-02, ADR 0029), and one costs about as much memory as ~16K tokens of
//! KV. So a node carries a state **or nothing** (a *tombstone*), and the tree
//! decides where states are kept — the part ignis today predicts by hand
//! (system block, generation opener, `/v1/decide` fan-outs).
//!
//! What the tree adds over the flat [`crate::prefix::PrefixCache`]:
//!
//! - **Fork discovery.** A lookup whose KV match runs deeper than its state
//!   match has found a head that two requests share and nobody checkpointed.
//!   [`Lookup::fork_tokens`] names it, so the request can publish there and
//!   the next one resumes exactly (Marconi; SGLang's `branching_seqlen`).
//! - **Marginal-saving eviction.** A state is worth the tokens it saves over
//!   the nearest state below it on its path, times how often its prefix is
//!   walked, decayed by age ([`RadixIndex::state_victim`]). Old turns' openers
//!   fall out on their own; a hot fork survives.
//! - **One structure for every tier's key space.** Nothing here touches the
//!   device: a node holds an opaque [`StateTag`] the scheduler maps to a
//!   retained slot, a KV-RAM blob or a disk file.
//!
//! A state at a token that is **not** a page boundary (the generation opener
//! is never one) is an *anchor*: it hangs on the node of the page below it and
//! carries its own key, exactly as the pages-only link of ADR 0029's #306
//! amendment lends the whole pages and the checkpoint owns the tail.
//!
//! The index is pure bookkeeping, CPU-only (ADR 0006), and free of the
//! scheduler: `crates/core/tests/radix_sim.rs` drives it against the current
//! policy on synthetic agent traces.

use std::collections::HashMap;

use crate::identity::{MatchKey, PromptContent};

/// A node's handle. `ROOT` is the empty prefix.
pub type NodeId = u32;

/// The empty prefix: no tokens, no state, never evicted.
pub const ROOT: NodeId = 0;

/// What a state is, to the scheduler: a retained slot, a KV-RAM blob or a disk
/// file. Opaque here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StateTag(pub u64);

/// Where a state hangs: on a page node, or on a non-page-aligned anchor under
/// one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StateSite {
    /// The state at the end of this page node.
    Node(NodeId),
    /// The state at `key`'s length, anchored under `node`, the page below it.
    Anchor { node: NodeId, key: MatchKey },
}

/// A state a lookup can resume from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateHit {
    pub tokens: u32,
    pub site: StateSite,
    pub tag: StateTag,
}

/// What [`RadixIndex::lookup`] found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lookup {
    /// Tokens whose KV the tree holds as whole pages (page-aligned).
    pub kv_tokens: u32,
    /// The deepest page node matched ([`ROOT`] for none).
    pub kv_node: NodeId,
    /// The longest state that is a prefix of the prompt, if any. Reuse resumes
    /// from here: everything past `state.tokens` is prefilled, even where KV
    /// pages exist (the recurrent state is the binding constraint).
    pub state: Option<StateHit>,
    /// Where the prompt should publish a new state, when the KV match runs
    /// deeper than the state match: a head shared by two requests with no
    /// checkpoint on it. `None` when there is no such head.
    pub fork_tokens: Option<u32>,
}

impl Lookup {
    /// Tokens a request skips by taking this lookup's state.
    pub fn reused_tokens(&self) -> u32 {
        self.state.map_or(0, |s| s.tokens)
    }
}

/// Why an operation on the index was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadixError {
    /// A state at token 0 is the empty prefix: nothing to resume.
    EmptyPrefix,
    /// The state's pages are not in the tree: [`RadixIndex::insert`] first.
    PagesMissing,
    /// The state would sit inside a media item's placeholders (ADR 0029, #193).
    InsideMedia,
}

/// A state and what is known of its use. Eviction reads this, never the walks
/// through the node: a state that is an ancestor of a newer one is walked
/// constantly and resumed from never.
#[derive(Debug, Clone, Copy)]
struct Meta {
    tag: StateTag,
    born: u64,
    last_used: u64,
    /// Lookups that resumed from this state.
    uses: u32,
    /// Published on a guess (a discovered fork) and not because a request is
    /// about to need it: it earns no probation, since it pays only if a third
    /// request arrives.
    speculative: bool,
}

impl Meta {
    fn new(tag: StateTag, tick: u64, speculative: bool) -> Self {
        Self { tag, born: tick, last_used: tick, uses: 0, speculative }
    }
}

#[derive(Debug, Clone)]
struct Anchor {
    tokens: u32,
    key: MatchKey,
    meta: Meta,
}

/// How eviction ranks states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvictionPolicy {
    /// A state's worth halves every this many ticks it goes unused.
    pub half_life: u64,
    /// The share of the state budget, in percent, held for the newest states
    /// never yet resumed from: a conversation's latest opener has not had the
    /// chance to be needed. The newest of them are spared; an unproven state
    /// older than that competes on its worth like the rest, so an abandoned
    /// branch cannot hold a slot for long. (Segmented LRU's probation, sized
    /// by the budget and not by a clock.)
    pub probation_pct: u32,
}

impl Default for EvictionPolicy {
    fn default() -> Self {
        Self { half_life: 64, probation_pct: 50 }
    }
}

#[derive(Debug, Clone)]
struct Node {
    parent: NodeId,
    key: MatchKey,
    /// Tokens from the root to the end of this page.
    tokens: u32,
    children: u32,
    state: Option<Meta>,
    anchors: Vec<Anchor>,
    /// Lookups whose KV walk went through this node (KV popularity; not a
    /// state's worth).
    passes: u32,
    last_used: u64,
    /// Live claimants holding this node's path.
    pins: u32,
    live: bool,
}

impl Node {
    fn dead() -> Self {
        Self {
            parent: ROOT,
            key: MatchKey::empty(),
            tokens: 0,
            children: 0,
            state: None,
            anchors: Vec::new(),
            passes: 0,
            last_used: 0,
            pins: 0,
            live: false,
        }
    }

    /// The deepest state this node itself carries, in tokens.
    fn best_state_tokens(&self) -> u32 {
        let own = if self.state.is_some() { self.tokens } else { 0 };
        self.anchors.iter().map(|a| a.tokens).fold(own, u32::max)
    }
}

/// The tree.
#[derive(Debug, Clone)]
pub struct RadixIndex {
    page: u32,
    nodes: Vec<Node>,
    free: Vec<NodeId>,
    by_key: HashMap<MatchKey, NodeId>,
    states: usize,
}

impl RadixIndex {
    /// An empty tree over pages of `page_tokens` tokens (`KV_PAGE_TOKENS`, 64).
    pub fn new(page_tokens: u32) -> Self {
        assert!(page_tokens > 0, "a page holds at least one token");
        let mut root = Node::dead();
        root.live = true;
        Self {
            page: page_tokens,
            nodes: vec![root],
            free: Vec::new(),
            by_key: HashMap::new(),
            states: 0,
        }
    }

    pub fn page_tokens(&self) -> u32 {
        self.page
    }

    /// Live page nodes, root excluded.
    pub fn pages(&self) -> usize {
        self.nodes.len() - 1 - self.free.len()
    }

    /// States held, on nodes and anchors.
    pub fn states(&self) -> usize {
        self.states
    }

    fn page_keys(&self, content: &PromptContent<'_>) -> crate::identity::PromptKeys {
        let pages = content.tokens() / self.page;
        content.keys_for((1..=pages).map(|p| p * self.page))
    }

    /// Walk `content`'s whole pages down the tree: the node reached and the
    /// path from the root.
    fn walk(&self, content: &PromptContent<'_>) -> (NodeId, Vec<NodeId>) {
        let keys = self.page_keys(content);
        let pages = content.tokens() / self.page;
        let mut node = ROOT;
        let mut path = vec![ROOT];
        for p in 1..=pages {
            let Some(key) = keys.key_at(p * self.page) else { break };
            match self.by_key.get(&key) {
                // The parent check is belt and braces: the key is cumulative.
                Some(&id) if self.nodes[id as usize].parent == node => {
                    node = id;
                    path.push(id);
                }
                _ => break,
            }
        }
        (node, path)
    }

    /// What the tree knows about `content`, without changing it.
    pub fn lookup(&self, content: PromptContent<'_>) -> Lookup {
        let (kv_node, path) = self.walk(&content);
        let kv_tokens = self.nodes[kv_node as usize].tokens;

        // The longest state on the path: page states from the deep end, then
        // the anchors hanging on any node of it.
        let mut best: Option<StateHit> = None;
        for &id in path.iter().rev() {
            let n = &self.nodes[id as usize];
            if id != ROOT && let Some(m) = n.state && !content.splits_media(n.tokens) {
                best = Some(StateHit { tokens: n.tokens, site: StateSite::Node(id), tag: m.tag });
                break;
            }
        }
        let lens: Vec<u32> = path
            .iter()
            .flat_map(|&id| self.nodes[id as usize].anchors.iter().map(|a| a.tokens))
            .collect();
        if !lens.is_empty() {
            let keys = content.keys_for(lens);
            for &id in &path {
                for a in &self.nodes[id as usize].anchors {
                    if keys.matches(a.tokens, a.key)
                        && !content.splits_media(a.tokens)
                        && best.is_none_or(|b| a.tokens > b.tokens)
                    {
                        best = Some(StateHit {
                            tokens: a.tokens,
                            site: StateSite::Anchor { node: id, key: a.key },
                            tag: a.meta.tag,
                        });
                    }
                }
            }
        }

        let state_tokens = best.map_or(0, |s| s.tokens);
        // The fork: where this prompt leaves a path other prompts continue
        // down, with no state there. The deepest matched page is a fork only
        // if the tree goes on below it (this prompt makes a second branch);
        // a prompt that merely extends a leaf has found no shared head, and a
        // page whose state was evicted is not a new discovery either.
        let at = self.nodes[kv_node as usize].tokens;
        let fork_tokens = (kv_node != ROOT
            && self.nodes[kv_node as usize].children > 0
            && at > state_tokens
            && !content.splits_media(at))
        .then_some(at);

        Lookup {
            kv_tokens: kv_tokens.max(state_tokens),
            kv_node,
            state: best,
            fork_tokens,
        }
    }

    /// How many pages hang directly below `node`.
    pub fn children_of(&self, node: NodeId) -> u32 {
        self.nodes[node as usize].children
    }

    /// Count a lookup's use: every node of its KV path was walked, and the
    /// state it resumed from was taken.
    pub fn touch(&mut self, lookup: &Lookup, tick: u64) {
        let mut id = lookup.kv_node;
        loop {
            let n = &mut self.nodes[id as usize];
            n.passes = n.passes.saturating_add(1);
            n.last_used = tick;
            if id == ROOT {
                break;
            }
            id = n.parent;
        }
        let used = |m: &mut Meta| {
            m.uses = m.uses.saturating_add(1);
            m.last_used = tick;
        };
        match lookup.state.map(|s| s.site) {
            Some(StateSite::Node(id)) => {
                if let Some(m) = self.nodes[id as usize].state.as_mut() {
                    used(m);
                }
            }
            Some(StateSite::Anchor { node, key }) => {
                if let Some(a) = self.nodes[node as usize].anchors.iter_mut().find(|a| a.key == key) {
                    used(&mut a.meta);
                }
            }
            None => {}
        }
    }

    /// Make sure every whole page of `content` is in the tree; the deepest
    /// node is returned. A request calls this when its prefill lands.
    pub fn insert(&mut self, content: PromptContent<'_>, tick: u64) -> NodeId {
        let keys = self.page_keys(&content);
        let pages = content.tokens() / self.page;
        let mut node = ROOT;
        for p in 1..=pages {
            let Some(key) = keys.key_at(p * self.page) else { break };
            node = match self.by_key.get(&key) {
                Some(&id) if self.nodes[id as usize].parent == node => id,
                _ => self.add(node, key, p * self.page, tick),
            };
        }
        node
    }

    fn add(&mut self, parent: NodeId, key: MatchKey, tokens: u32, tick: u64) -> NodeId {
        let n = Node { parent, key, tokens, last_used: tick, live: true, ..Node::dead() };
        let id = if let Some(id) = self.free.pop() {
            self.nodes[id as usize] = n;
            id
        } else {
            self.nodes.push(n);
            (self.nodes.len() - 1) as NodeId
        };
        self.nodes[parent as usize].children += 1;
        self.by_key.insert(key, id);
        id
    }

    /// Hang a state at the first `tokens` of `content`. The pages below it
    /// must be in the tree. Returns the tag it replaced, if the site was
    /// taken.
    pub fn attach_state(
        &mut self,
        content: PromptContent<'_>,
        tokens: u32,
        tag: StateTag,
        tick: u64,
    ) -> Result<Option<StateTag>, RadixError> {
        self.attach(content, tokens, tag, tick, false)
    }

    /// [`RadixIndex::attach_state`] for a state published at a discovered
    /// fork ([`Lookup::fork_tokens`]): a bet that earns no probation.
    pub fn attach_fork(
        &mut self,
        content: PromptContent<'_>,
        tokens: u32,
        tag: StateTag,
        tick: u64,
    ) -> Result<Option<StateTag>, RadixError> {
        self.attach(content, tokens, tag, tick, true)
    }

    fn attach(
        &mut self,
        content: PromptContent<'_>,
        tokens: u32,
        tag: StateTag,
        tick: u64,
        speculative: bool,
    ) -> Result<Option<StateTag>, RadixError> {
        if tokens == 0 {
            return Err(RadixError::EmptyPrefix);
        }
        if content.splits_media(tokens) {
            return Err(RadixError::InsideMedia);
        }
        let floor = tokens / self.page * self.page;
        let keys = content.keys_for([floor, tokens]);
        let floor_node = if floor == 0 {
            ROOT
        } else {
            let k = keys.key_at(floor).ok_or(RadixError::PagesMissing)?;
            *self.by_key.get(&k).ok_or(RadixError::PagesMissing)?
        };
        if tokens == floor {
            let n = &mut self.nodes[floor_node as usize];
            n.last_used = tick;
            // Re-publishing a site keeps what is known of it: a refreshed copy
            // of a proven state is still a proven state.
            let old = n.state.replace(match n.state {
                Some(m) => Meta { tag, last_used: tick, ..m },
                None => Meta::new(tag, tick, speculative),
            });
            if old.is_none() {
                self.states += 1;
            }
            return Ok(old.map(|m| m.tag));
        }
        let key = keys.key_at(tokens).ok_or(RadixError::PagesMissing)?;
        let n = &mut self.nodes[floor_node as usize];
        n.last_used = tick;
        if let Some(a) = n.anchors.iter_mut().find(|a| a.key == key) {
            let kept = Meta { tag, last_used: tick, ..a.meta };
            return Ok(Some(std::mem::replace(&mut a.meta, kept).tag));
        }
        n.anchors.push(Anchor { tokens, key, meta: Meta::new(tag, tick, speculative) });
        self.states += 1;
        Ok(None)
    }

    /// Drop the state at `site`: the node stays as a tombstone, its KV intact.
    pub fn drop_state(&mut self, site: StateSite) -> Option<StateTag> {
        let dropped = match site {
            StateSite::Node(id) => self.nodes[id as usize].state.take().map(|m| m.tag),
            StateSite::Anchor { node, key } => {
                let anchors = &mut self.nodes[node as usize].anchors;
                anchors.iter().position(|a| a.key == key).map(|i| anchors.swap_remove(i).meta.tag)
            }
        };
        if dropped.is_some() {
            self.states -= 1;
        }
        dropped
    }

    /// Protect `node`'s path from eviction while a claimant holds it.
    pub fn pin(&mut self, node: NodeId) {
        self.walk_up(node, |n| n.pins += 1);
    }

    /// The inverse of [`RadixIndex::pin`].
    pub fn unpin(&mut self, node: NodeId) {
        self.walk_up(node, |n| n.pins = n.pins.saturating_sub(1));
    }

    fn walk_up(&mut self, mut id: NodeId, mut f: impl FnMut(&mut Node)) {
        loop {
            f(&mut self.nodes[id as usize]);
            if id == ROOT {
                break;
            }
            id = self.nodes[id as usize].parent;
        }
    }

    /// The deepest state strictly below `id` on its path, in tokens: what a
    /// request falls back to if the state at `id` goes.
    fn base_below(&self, id: NodeId) -> u32 {
        let mut best = 0;
        let mut cur = id;
        while cur != ROOT {
            cur = self.nodes[cur as usize].parent;
            best = best.max(self.nodes[cur as usize].best_state_tokens());
        }
        best
    }

    /// The state that costs the least to lose, or `None` when every state is
    /// pinned. `spared` is how many of the newest never-resumed states are
    /// exempt ([`EvictionPolicy::probation_pct`] of the budget).
    ///
    /// A state's worth is the tokens it saves over the nearest state below it
    /// on its path, times one plus the times it was resumed from, halved for
    /// every `half_life` ticks it has gone unused.
    pub fn state_victim(&self, now: u64, policy: EvictionPolicy, spared: usize) -> Option<(StateSite, StateTag)> {
        struct Cand {
            meta: Meta,
            saved: u32,
            site: StateSite,
        }
        let mut cands: Vec<Cand> = Vec::with_capacity(self.states);
        for (i, n) in self.nodes.iter().enumerate().skip(1) {
            if !n.live || n.pins > 0 {
                continue;
            }
            let id = i as NodeId;
            let base = self.base_below(id);
            if let Some(m) = n.state {
                cands.push(Cand { meta: m, saved: n.tokens - base.min(n.tokens), site: StateSite::Node(id) });
            }
            for a in &n.anchors {
                let below = base.max(if n.state.is_some() { n.tokens } else { 0 });
                cands.push(Cand {
                    meta: a.meta,
                    saved: a.tokens - below.min(a.tokens),
                    site: StateSite::Anchor { node: id, key: a.key },
                });
            }
        }
        // The newest `spared` unproven states are on probation.
        let mut unproven: Vec<(u64, usize)> =
            cands.iter().enumerate().filter(|(_, c)| c.meta.uses == 0 && !c.meta.speculative).map(|(i, c)| (c.meta.born, i)).collect();
        unproven.sort_unstable_by(|a, b| b.cmp(a));
        let shielded: std::collections::HashSet<usize> = unproven.iter().take(spared).map(|&(_, i)| i).collect();
        let worth = |c: &Cand| {
            let decay = 0.5f64.powf(now.saturating_sub(c.meta.last_used) as f64 / policy.half_life.max(1) as f64);
            c.saved as f64 * (1 + c.meta.uses) as f64 * decay
        };
        cands
            .iter()
            .enumerate()
            .min_by(|(i, a), (j, b)| {
                (shielded.contains(i), worth(a))
                    .partial_cmp(&(shielded.contains(j), worth(b)))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(_, c)| (c.site, c.meta.tag))
    }

    /// Evict states until at most `max` remain; the tags freed, for the
    /// scheduler to release. Stops early if everything left is pinned.
    pub fn evict_states_to(&mut self, max: usize, now: u64, policy: EvictionPolicy) -> Vec<StateTag> {
        let spared = (max * policy.probation_pct as usize / 100).max(1);
        let mut freed = Vec::new();
        while self.states > max {
            let Some((site, tag)) = self.state_victim(now, policy, spared) else { break };
            self.drop_state(site);
            freed.push(tag);
        }
        freed
    }

    /// Evict least-recently-used unpinned leaves until at most `max_pages`
    /// page nodes remain, returning the tags of the states they took with
    /// them. A leaf is a node nothing is built on: an internal page cannot go
    /// while a longer prefix still stands on it.
    pub fn evict_leaves_to(&mut self, max_pages: usize) -> Vec<StateTag> {
        let mut freed = Vec::new();
        while self.pages() > max_pages {
            let victim = self
                .nodes
                .iter()
                .enumerate()
                .skip(1)
                .filter(|(_, n)| n.live && n.children == 0 && n.pins == 0)
                .min_by_key(|(_, n)| n.last_used)
                .map(|(i, _)| i as NodeId);
            let Some(id) = victim else { break };
            let n = std::mem::replace(&mut self.nodes[id as usize], Node::dead());
            if let Some(m) = n.state {
                freed.push(m.tag);
                self.states -= 1;
            }
            for a in n.anchors {
                freed.push(a.meta.tag);
                self.states -= 1;
            }
            self.by_key.remove(&n.key);
            self.nodes[n.parent as usize].children -= 1;
            self.free.push(id);
        }
        freed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::MediaKey;
    use crate::types::TokenId;

    const PAGE: u32 = 4;

    fn toks(range: std::ops::Range<u32>) -> Vec<TokenId> {
        range.collect()
    }

    fn lookup(t: &RadixIndex, v: &[TokenId]) -> Lookup {
        t.lookup(PromptContent::text(v))
    }

    fn put(t: &mut RadixIndex, v: &[TokenId], tick: u64) -> NodeId {
        t.insert(PromptContent::text(v), tick)
    }

    #[test]
    fn an_empty_tree_matches_nothing() {
        let t = RadixIndex::new(PAGE);
        let l = lookup(&t, &toks(0..20));
        assert_eq!((l.kv_tokens, l.state, l.fork_tokens), (0, None, None));
    }

    #[test]
    fn pages_are_shared_across_requests_with_a_common_head() {
        let mut t = RadixIndex::new(PAGE);
        put(&mut t, &toks(0..16), 1);
        assert_eq!(t.pages(), 4);
        // Same first 8 tokens, then a different tail: two new pages, not four.
        let mut b = toks(0..8);
        b.extend(100..108);
        put(&mut t, &b, 2);
        assert_eq!(t.pages(), 6);
        assert_eq!(lookup(&t, &b).kv_tokens, 16);
        assert_eq!(lookup(&t, &toks(0..16)).kv_tokens, 16);
        // A prompt diverging inside a page shares only the whole pages below.
        let mut c = toks(0..6);
        c.push(999);
        assert_eq!(lookup(&t, &c).kv_tokens, 4);
    }

    #[test]
    fn a_state_resumes_only_where_it_was_captured() {
        let mut t = RadixIndex::new(PAGE);
        let p = toks(0..16);
        put(&mut t, &p, 1);
        t.attach_state(PromptContent::text(&p), 8, StateTag(7), 1).unwrap();
        let l = lookup(&t, &p);
        // KV runs to 16, the state to 8: resume from 8, and 16 is a fork.
        assert_eq!(l.kv_tokens, 16);
        assert_eq!(l.state.map(|s| (s.tokens, s.tag)), Some((8, StateTag(7))));
        // Resume from 8. The tree goes nowhere past 16, so 16 is no fork.
        assert_eq!(l.fork_tokens, None);
        assert_eq!(l.reused_tokens(), 8);
    }

    #[test]
    fn a_fork_is_named_where_the_kv_match_outruns_the_state() {
        let mut t = RadixIndex::new(PAGE);
        let a = toks(0..24);
        put(&mut t, &a, 1);
        t.attach_state(PromptContent::text(&a), 24, StateTag(1), 1).unwrap();
        // B shares 12 tokens with A, then diverges: KV hit 12, no state at 12.
        let mut b = toks(0..12);
        b.extend(500..512);
        let l = lookup(&t, &b);
        assert_eq!((l.kv_tokens, l.reused_tokens(), l.fork_tokens), (12, 0, Some(12)));
        // B publishes at the fork; C, diverging again after it, resumes there.
        put(&mut t, &b, 2);
        t.attach_state(PromptContent::text(&b), 12, StateTag(2), 2).unwrap();
        let mut c = toks(0..12);
        c.extend(700..712);
        let l = lookup(&t, &c);
        assert_eq!((l.reused_tokens(), l.fork_tokens), (12, None));
    }

    #[test]
    fn a_state_off_the_page_grid_is_an_anchor_that_longer_prompts_still_find() {
        let mut t = RadixIndex::new(PAGE);
        let p = toks(0..11); // opener at 11: pages 0..8 + a 3-token tail
        put(&mut t, &p, 1);
        t.attach_state(PromptContent::text(&p), 11, StateTag(3), 1).unwrap();
        // The next turn extends the prompt: it matches the anchor at 11 even
        // though only two whole pages of it exist in the tree.
        let next = toks(0..30);
        put(&mut t, &next, 2);
        let l = lookup(&t, &next);
        assert_eq!(l.state.map(|s| s.tokens), Some(11));
        assert!(matches!(l.state.unwrap().site, StateSite::Anchor { .. }));
        // A prompt that differs inside the tail does not match it.
        let mut other = toks(0..10);
        other.push(9999);
        other.extend(0..8);
        assert_eq!(lookup(&t, &other).reused_tokens(), 0);
    }

    #[test]
    fn a_state_needs_its_pages_and_a_legal_point() {
        let mut t = RadixIndex::new(PAGE);
        let p = toks(0..16);
        let c = PromptContent::text(&p);
        assert_eq!(t.attach_state(c, 8, StateTag(1), 0), Err(RadixError::PagesMissing));
        assert_eq!(t.attach_state(c, 0, StateTag(1), 0), Err(RadixError::EmptyPrefix));
        let media = [MediaKey { begin: 4, count: 8, digest: [1; 32], grid: [1, 2, 4] }];
        let mc = PromptContent::new(&p, &media);
        t.insert(mc, 0);
        assert_eq!(t.attach_state(mc, 8, StateTag(1), 0), Err(RadixError::InsideMedia));
        assert!(t.attach_state(mc, 12, StateTag(1), 0).is_ok());
    }

    #[test]
    fn a_media_item_blocks_a_state_inside_it_in_lookup_too() {
        let mut t = RadixIndex::new(PAGE);
        let p = toks(0..16);
        let plain = PromptContent::text(&p);
        t.insert(plain, 0);
        t.attach_state(plain, 8, StateTag(1), 0).unwrap();
        // The same tokens with an image spanning 8: the key differs (the item
        // is absorbed), so nothing matches at all.
        let media = [MediaKey { begin: 6, count: 6, digest: [9; 32], grid: [1, 2, 3] }];
        let l = t.lookup(PromptContent::new(&p, &media));
        assert_eq!(l.reused_tokens(), 0);
    }

    #[test]
    fn pinned_paths_survive_eviction_and_unpinned_leaves_go_lru_first() {
        let mut t = RadixIndex::new(PAGE);
        let a = toks(0..8);
        let b = toks(100..108);
        let na = put(&mut t, &a, 1);
        put(&mut t, &b, 2);
        t.attach_state(PromptContent::text(&a), 8, StateTag(1), 1).unwrap();
        t.attach_state(PromptContent::text(&b), 8, StateTag(2), 2).unwrap();
        t.pin(na);
        let freed = t.evict_leaves_to(2);
        assert_eq!(freed, vec![StateTag(2)]);
        assert_eq!(lookup(&t, &a).kv_tokens, 8);
        assert_eq!(lookup(&t, &b).kv_tokens, 0);
        t.unpin(na);
        assert_eq!(t.evict_leaves_to(0).len(), 1);
        assert_eq!((t.pages(), t.states()), (0, 0));
    }

    #[test]
    fn an_internal_page_is_never_evicted_under_a_longer_prefix() {
        let mut t = RadixIndex::new(PAGE);
        put(&mut t, &toks(0..12), 1);
        t.evict_leaves_to(2);
        // Leaves go from the deep end: the head stays, still matchable.
        assert_eq!(lookup(&t, &toks(0..12)).kv_tokens, 8);
        assert_eq!(t.pages(), 2);
    }

    #[test]
    fn eviction_takes_the_state_that_saves_the_least() {
        let mut t = RadixIndex::new(PAGE);
        // A 40-token path with states at 8 (shallow) and 40 (deep).
        let p = toks(0..40);
        put(&mut t, &p, 1);
        t.attach_state(PromptContent::text(&p), 8, StateTag(1), 1).unwrap();
        t.attach_state(PromptContent::text(&p), 40, StateTag(2), 1).unwrap();
        // Marginal saving: 8 tokens for the shallow, 32 for the deep.
        let (_, tag) = t.state_victim(1, EvictionPolicy { half_life: 100, probation_pct: 0 }, 0).unwrap();
        assert_eq!(tag, StateTag(1));
        // Resume from the shallow one a lot (a hot shared head): it now
        // outranks the deep one, which nobody has resumed from.
        let p8 = toks(0..8);
        for i in 0..10 {
            let l = lookup(&t, &p8);
            assert_eq!(l.reused_tokens(), 8);
            t.touch(&l, 2 + i);
        }
        let (_, tag) = t.state_victim(12, EvictionPolicy { half_life: 1000, probation_pct: 0 }, 0).unwrap();
        assert_eq!(tag, StateTag(2));
    }

    #[test]
    fn a_state_nobody_has_resumed_from_is_spared_while_on_probation() {
        let mut t = RadixIndex::new(PAGE);
        // An old, used-once state with a big saving, and a brand-new one with
        // a small saving: the newest must not be the first thing evicted.
        let old = toks(0..40);
        put(&mut t, &old, 1);
        t.attach_state(PromptContent::text(&old), 8, StateTag(1), 1).unwrap();
        t.attach_state(PromptContent::text(&old), 40, StateTag(2), 1).unwrap();
        let l = lookup(&t, &old);
        t.touch(&l, 2); // resumed once from 40
        for i in 0..3 {
            let l = lookup(&t, &old[..8]);
            t.touch(&l, 3 + i); // and the shallow head is popular
        }
        let mut newer = toks(0..40);
        newer.extend(1000..1008);
        put(&mut t, &newer, 30);
        t.attach_state(PromptContent::text(&newer), 48, StateTag(3), 30).unwrap();
        let policy = EvictionPolicy { half_life: 64, probation_pct: 50 };
        // One state is spared: the newest unproven one, state 3.
        let (_, victim) = t.state_victim(31, policy, 1).unwrap();
        assert_ne!(victim, StateTag(3), "the unproven newest state is on probation");
        // With nothing spared it competes on worth: small saving, never used.
        let (_, victim) = t.state_victim(31, policy, 0).unwrap();
        assert_eq!(victim, StateTag(3));
    }

    #[test]
    fn eviction_to_a_budget_frees_exactly_the_excess_and_honours_pins() {
        let mut t = RadixIndex::new(PAGE);
        let mut leaves = Vec::new();
        for i in 0..5u32 {
            let p = toks(i * 1000..i * 1000 + 8);
            let n = put(&mut t, &p, i as u64);
            t.attach_state(PromptContent::text(&p), 8, StateTag(i as u64), i as u64).unwrap();
            leaves.push(n);
        }
        t.pin(leaves[0]);
        let freed = t.evict_states_to(2, 10, EvictionPolicy { half_life: 100, probation_pct: 0 });
        assert_eq!(freed.len(), 3);
        assert_eq!(t.states(), 2);
        assert!(!freed.contains(&StateTag(0)), "the pinned state stays");
    }

    #[test]
    fn dropping_a_state_leaves_a_tombstone_with_its_kv() {
        let mut t = RadixIndex::new(PAGE);
        let p = toks(0..8);
        let n = put(&mut t, &p, 1);
        t.attach_state(PromptContent::text(&p), 8, StateTag(1), 1).unwrap();
        assert_eq!(t.drop_state(StateSite::Node(n)), Some(StateTag(1)));
        let l = lookup(&t, &p);
        assert_eq!((l.kv_tokens, l.reused_tokens(), l.fork_tokens), (8, 0, None));
        assert_eq!(t.drop_state(StateSite::Node(n)), None);
    }

    #[test]
    fn freed_slots_are_reused() {
        let mut t = RadixIndex::new(PAGE);
        put(&mut t, &toks(0..8), 1);
        t.evict_leaves_to(0);
        put(&mut t, &toks(50..58), 2);
        assert_eq!(t.pages(), 2);
        assert_eq!(lookup(&t, &toks(50..58)).kv_tokens, 8);
    }
}
