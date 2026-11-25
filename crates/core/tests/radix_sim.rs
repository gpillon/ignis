//! Policy simulation: today's reuse policy against the radix policy, on
//! synthetic agent traces, at the **same number of retained state slots**.
//!
//! This is a model, not ignis: it counts prompt tokens a policy would have to
//! prefill, given where each policy keeps recurrent state. It models the
//! *policy* of ADR 0029 as built (the system block's page floor as a retained
//! prefix, one checkpoint at each request's generation opener, at most two per
//! conversation, checkpoints given up before prefixes, LRU within a kind) and
//! the radix policy of `radix.rs` (the same two boundaries, plus forks the
//! tree discovers, under marginal-saving eviction with no per-conversation
//! cap). It does not model KV page pressure, concurrency in one prefill batch,
//! or the ~19 ms an extra prefill cut costs (counted, not priced).
//!
//! The traces are assumptions about agent behaviour, stated in each
//! generator, and none of them is a recorded session. A real trace replay (`docs/specs/bench/01-trace-replay.md`) is
//! the check on them; what this settles is whether the radix policy can lose.

use ignis_core::identity::PromptContent;
use ignis_core::radix::{EvictionPolicy, RadixIndex, StateTag};
use ignis_core::types::TokenId;

const PAGE: u32 = 64;
const PREFILL_TOK_S: f64 = 11_300.0; // README: cold prefill, 46K prompt

/// A request: its conversation, its whole prompt, and where the system block
/// ends (the one boundary today's policy predicts).
struct Req {
    conv: u32,
    tokens: Vec<TokenId>,
    sys_len: u32,
}

/// Deterministic pseudo-randomness, so a number printed here reproduces.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn chance(&mut self, pct: u64) -> bool {
        self.below(100) < pct
    }
}

/// Fresh, never-reused token ids: every span of "text" is unique unless a
/// generator copies it, so a match is always a deliberate share.
struct Tokens(TokenId);
impl Tokens {
    fn fresh(&mut self, n: usize) -> Vec<TokenId> {
        let v: Vec<TokenId> = (self.0..self.0 + n as TokenId).collect();
        self.0 += n as TokenId;
        v
    }
}

/// Today's policy as built (ADR 0029), at `slots` retained slots.
struct Baseline {
    slots: usize,
    entries: Vec<Entry>,
    tick: u64,
}
struct Entry {
    tokens: Vec<TokenId>,
    checkpoint_of: Option<u32>,
    last: u64,
}
impl Baseline {
    fn new(slots: usize) -> Self {
        Self { slots, entries: Vec::new(), tick: 0 }
    }
    /// Returns (tokens reused, extra cuts).
    fn serve(&mut self, r: &Req) -> (u32, u32) {
        self.tick += 1;
        let hit = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| r.tokens.starts_with(&e.tokens))
            .max_by_key(|(_, e)| e.tokens.len())
            .map(|(i, e)| (i, e.tokens.len() as u32));
        let reused = hit.map_or(0, |(i, n)| {
            self.entries[i].last = self.tick;
            n
        });
        // Publish the system block's page floor, once.
        let floor = (r.sys_len / PAGE * PAGE) as usize;
        if floor > 0 && !self.entries.iter().any(|e| e.checkpoint_of.is_none() && e.tokens == r.tokens[..floor]) {
            self.entries.push(Entry { tokens: r.tokens[..floor].to_vec(), checkpoint_of: None, last: self.tick });
        }
        // The checkpoint at the opener (the end of the prompt); a conversation
        // keeps its latest two.
        self.entries.retain(|e| e.checkpoint_of != Some(r.conv) || e.tokens != r.tokens);
        self.entries.push(Entry { tokens: r.tokens.clone(), checkpoint_of: Some(r.conv), last: self.tick });
        let mine: Vec<usize> = (0..self.entries.len()).filter(|&i| self.entries[i].checkpoint_of == Some(r.conv)).collect();
        if mine.len() > 2 {
            let oldest = *mine.iter().min_by_key(|&&i| self.entries[i].last).unwrap();
            self.entries.remove(oldest);
        }
        // Over budget: checkpoints go before prefixes, least recently used.
        while self.entries.len() > self.slots {
            let pick = |kind: bool| {
                self.entries
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| e.checkpoint_of.is_some() == kind)
                    .min_by_key(|(_, e)| e.last)
                    .map(|(i, _)| i)
            };
            let v = pick(true).or_else(|| pick(false)).unwrap();
            self.entries.remove(v);
        }
        (reused, 0)
    }
}

/// The radix policy at `slots` state slots.
struct Radix {
    slots: usize,
    tree: RadixIndex,
    tick: u64,
    next_tag: u64,
    policy: EvictionPolicy,
    fork_discovery: bool,
}
impl Radix {
    fn new(slots: usize, fork_discovery: bool) -> Self {
        Self { slots, tree: RadixIndex::new(PAGE), tick: 0, next_tag: 0, policy: EvictionPolicy::default(), fork_discovery }
    }
    fn serve(&mut self, r: &Req) -> (u32, u32) {
        self.tick += 1;
        let c = PromptContent::text(&r.tokens);
        let look = self.tree.lookup(c);
        let reused = look.reused_tokens();
        self.tree.touch(&look, self.tick);
        self.tree.insert(c, self.tick);
        let mut cuts = 0;
        let floor = r.sys_len / PAGE * PAGE;
        let mut at = vec![(r.tokens.len() as u32, false)];
        if floor > 0 {
            at.push((floor, false));
        }
        if self.fork_discovery
            && let Some(f) = look.fork_tokens
            && f > reused
            && !at.iter().any(|&(t, _)| t == f)
        {
            at.push((f, true));
            cuts += 1;
        }
        for (t, fork) in at {
            // A state already held at this point is not stored again.
            if look.state.is_some_and(|s| s.tokens == t) {
                continue;
            }
            self.next_tag += 1;
            let tag = StateTag(self.next_tag);
            if fork {
                self.tree.attach_fork(c, t, tag, self.tick).unwrap();
            } else {
                self.tree.attach_state(c, t, tag, self.tick).unwrap();
            }
        }
        self.tree.evict_states_to(self.slots, self.tick, self.policy);
        (reused, cuts)
    }
}

struct Outcome {
    prompt: u64,
    prefilled: u64,
    cuts: u32,
}
impl Outcome {
    fn pct(&self) -> f64 {
        100.0 * (self.prompt - self.prefilled) as f64 / self.prompt as f64
    }
    fn secs(&self) -> f64 {
        self.prefilled as f64 / PREFILL_TOK_S
    }
}

fn run(trace: &[Req], mut serve: impl FnMut(&Req) -> (u32, u32)) -> Outcome {
    let mut o = Outcome { prompt: 0, prefilled: 0, cuts: 0 };
    for r in trace {
        let (reused, cuts) = serve(r);
        assert!(reused as usize <= r.tokens.len());
        o.prompt += r.tokens.len() as u64;
        o.prefilled += (r.tokens.len() as u32 - reused) as u64;
        o.cuts += cuts;
    }
    o
}

/// Interleave several request queues round-robin, as concurrent agents do.
fn interleave(queues: Vec<Vec<Req>>) -> Vec<Req> {
    let mut queues: Vec<std::collections::VecDeque<Req>> = queues.into_iter().map(Into::into).collect();
    let mut out = Vec::new();
    while queues.iter().any(|q| !q.is_empty()) {
        for q in queues.iter_mut() {
            if let Some(r) = q.pop_front() {
                out.push(r);
            }
        }
    }
    out
}

const SYS: usize = 13_000; // scripts/agent-swarm.py: one shared ~13K head
const TOOL: usize = 1_500; // the swarm's "short" load
const REPLY: usize = 200;

/// A. Swarm (the benchmark the repo already uses): `agents` conversations over
/// one system head, each turn appending a reply and a tool result. Forks: none
/// beyond the system block.
fn swarm(t: &mut Tokens, agents: u32, turns: usize, sys: &[TokenId]) -> Vec<Req> {
    let qs = (0..agents)
        .map(|c| {
            let mut p = sys.to_vec();
            p.extend(t.fresh(60));
            (0..turns)
                .map(|_| {
                    let r = Req { conv: c, tokens: p.clone(), sys_len: sys.len() as u32 };
                    p.extend(t.fresh(REPLY));
                    p.extend(t.fresh(TOOL));
                    r
                })
                .collect()
        })
        .collect();
    interleave(qs)
}

/// B. Context fork: a main agent works `turns` turns; after turn 4, with 50%
/// chance per turn it spawns two subagents whose context is the main's
/// history *as of three turns ago* plus a task each (an agent handing a
/// subagent "the conversation so far, minus the last bit"). The shared head
/// ends at no system block and at neither of the main's last two openers.
fn context_fork(t: &mut Tokens, conv: u32, turns: usize, sys: &[TokenId], rng: &mut Rng) -> Vec<Req> {
    let mut out = Vec::new();
    let mut history: Vec<Vec<TokenId>> = Vec::new();
    let mut p = sys.to_vec();
    p.extend(t.fresh(60));
    let mut sub = conv + 100;
    for turn in 0..turns {
        history.push(p.clone());
        out.push(Req { conv, tokens: p.clone(), sys_len: sys.len() as u32 });
        if turn >= 4 && rng.chance(50) {
            for _ in 0..2 {
                let mut s = history[turn - 3].clone();
                s.extend(t.fresh(300));
                out.push(Req { conv: sub, tokens: s, sys_len: sys.len() as u32 });
                sub += 1;
            }
        }
        p.extend(t.fresh(REPLY));
        p.extend(t.fresh(TOOL));
    }
    out
}

/// C. Rewind: a conversation that, from turn 5, with 30% chance returns to an
/// earlier turn's prompt and takes a different path (undo/edit/regenerate from
/// a checkpoint). The abandoned branch stays in the tree.
fn rewind(t: &mut Tokens, conv: u32, turns: usize, sys: &[TokenId], rng: &mut Rng) -> Vec<Req> {
    let mut out = Vec::new();
    let mut history: Vec<Vec<TokenId>> = Vec::new();
    let mut p = sys.to_vec();
    p.extend(t.fresh(60));
    for turn in 0..turns {
        history.push(p.clone());
        out.push(Req { conv, tokens: p.clone(), sys_len: sys.len() as u32 });
        let base = if turn >= 5 && rng.chance(30) {
            let j = 1 + rng.below(history.len() as u64 - 1) as usize;
            history[j].clone()
        } else {
            p.clone()
        };
        p = base;
        p.extend(t.fresh(REPLY));
        p.extend(t.fresh(TOOL));
    }
    out
}

fn report(name: &str, slots: usize, trace: &[Req]) -> (Outcome, Outcome, Outcome) {
    let base = run(trace, {
        let mut b = Baseline::new(slots);
        move |r| b.serve(r)
    });
    let radix_nofork = run(trace, {
        let mut x = Radix::new(slots, false);
        move |r| x.serve(r)
    });
    let radix = run(trace, {
        let mut x = Radix::new(slots, true);
        move |r| x.serve(r)
    });
    println!(
        "{name:<26} slots={slots:<3} reqs={:<4} prompt={:>8}K | today {:>5.1}% reused ({:>6.1}s prefill) | radix, eviction only {:>5.1}% ({:>6.1}s) | radix + forks {:>5.1}% ({:>6.1}s, +{} cuts) | prefill saved vs today: {:>5.1}%",
        trace.len(),
        base.prompt / 1000,
        base.pct(),
        base.secs(),
        radix_nofork.pct(),
        radix_nofork.secs(),
        radix.pct(),
        radix.secs(),
        radix.cuts,
        100.0 * (base.prefilled as f64 - radix.prefilled as f64) / base.prefilled as f64,
    );
    (base, radix_nofork, radix)
}

fn traces() -> Vec<(&'static str, Vec<Req>)> {
    let mut t = Tokens(1);
    let sys = t.fresh(SYS);
    let mut rng = Rng(42);
    let a = swarm(&mut t, 8, 8, &sys);
    let b = context_fork(&mut t, 1, 14, &sys, &mut rng);
    let c = interleave((0..4).map(|i| rewind(&mut t, 10 + i, 14, &sys, &mut rng)).collect());
    let mixed = interleave(vec![
        swarm(&mut t, 6, 8, &sys),
        context_fork(&mut t, 50, 14, &sys, &mut rng),
        rewind(&mut t, 60, 14, &sys, &mut rng),
        rewind(&mut t, 61, 14, &sys, &mut rng),
    ]);
    vec![("A swarm 8x8", a), ("B context fork", b), ("C rewind x4 convs", c), ("D mixed", mixed)]
}

/// The invariant the radix policy is held to: at the same slot budget, its
/// retention and eviction never prefill more than today's. Fork discovery is
/// reported beside it and **not** asserted: on these traces every fork sits on
/// an opener that is already checkpointed, so it buys nothing and costs a cut
/// per fork (`+N cuts`). It stays off until a real trace shows a fork no
/// boundary predicts.
#[test]
fn radix_retention_never_prefills_more_than_todays_at_equal_slots() {
    println!("\nmodel of ADR 0029's policy vs the radix policy; page={PAGE}; prefill at {PREFILL_TOK_S} tok/s");
    for slots in [4usize, 8, 16] {
        for (name, trace) in traces() {
            let (base, retention, _forks) = report(name, slots, &trace);
            assert!(
                retention.prefilled <= base.prefilled,
                "{name} at {slots} slots: radix retention prefilled {} > today {}",
                retention.prefilled,
                base.prefilled
            );
        }
        println!();
    }
}

#[test]
fn a_swarm_with_only_a_system_head_gains_nothing_and_loses_nothing() {
    // The honest control: every boundary the swarm has is one today predicts.
    let mut t = Tokens(1);
    let sys = t.fresh(SYS);
    let trace = swarm(&mut t, 8, 8, &sys);
    let (base, radix, _) = report("control: swarm", 16, &trace);
    let delta = (base.prefilled as i64 - radix.prefilled as i64).unsigned_abs() as f64 / base.prefilled as f64;
    assert!(delta < 0.02, "swarm moved by {:.1}%", delta * 100.0);
}
