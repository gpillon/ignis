//! The pushdown automaton: a configuration is a stack of frames, the top one
//! a lexer (string, number, literal) or a structural position; a state is
//! the set of configurations an `anyOf` has not yet told apart (almost always
//! one). Bytes go in one at a time.

use crate::number::{NumState, NumStep};
use crate::schema::{Core, Node, NodeId, MAX_STACK, NO_RE};
use crate::string::{self, StrState, StrStep};

pub(crate) const OPEN: u8 = 0;
pub(crate) const BEFORE: u8 = 1; // after a comma: a key (object) or a value (array)
pub(crate) const AFTER_KEY: u8 = 2;
pub(crate) const AFTER_VALUE: u8 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum Frame {
    /// The root value is complete: trailing whitespace, then a stop token.
    Done,
    /// A value of `node` is expected (leading whitespace allowed).
    Value { node: NodeId },
    Obj { node: NodeId, idx: u32, phase: u8 },
    /// An object of free keys, each holding a value of `value`.
    FreeObj { value: NodeId, phase: u8 },
    Arr { node: NodeId, count: u32, phase: u8 },
    FreeArr { phase: u8 },
    Str { re: u32, st: StrState },
    Num { spec: u32, st: NumState },
    Lit { set: u32, pos: u32 },
}

#[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct Config {
    pub frames: Vec<Frame>,
    /// Whitespace bytes in the current run.
    pub ws: u8,
}

impl Clone for Config {
    fn clone(&self) -> Self {
        Self { frames: self.frames.clone(), ws: self.ws }
    }
    fn clone_from(&mut self, o: &Self) {
        self.frames.clone_from(&o.frames);
        self.ws = o.ws;
    }
}

impl Config {
    pub fn initial(root: NodeId) -> Self {
        Self { frames: vec![Frame::Done, Frame::Value { node: root }], ws: 0 }
    }
    pub fn accepting(&self) -> bool {
        self.frames.len() == 1
    }
}

#[derive(Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct State {
    pub configs: Vec<Config>,
}

impl Clone for State {
    fn clone(&self) -> Self {
        Self { configs: self.configs.clone() }
    }
    fn clone_from(&mut self, o: &Self) {
        self.configs.clone_from(&o.configs);
    }
}

impl State {
    pub fn initial(root: NodeId) -> Self {
        Self { configs: vec![Config::initial(root)] }
    }

    pub fn accepting(&self) -> bool {
        self.configs.iter().any(Config::accepting)
    }

    /// Advance every configuration by `b`. On `false` the state is dead and
    /// its contents are garbage (kept only for their allocations).
    #[inline]
    pub fn step(&mut self, core: &Core, b: u8, forks: &mut Vec<Config>) -> bool {
        forks.clear();
        if self.configs.len() == 1 {
            let ok = advance(core, &mut self.configs[0], b, forks);
            if forks.is_empty() {
                return ok;
            }
            if !ok {
                self.configs.clear();
            }
        } else {
            let mut i = 0;
            while i < self.configs.len() {
                if advance(core, &mut self.configs[i], b, forks) {
                    i += 1;
                } else {
                    self.configs.swap_remove(i);
                }
            }
        }
        self.configs.append(forks);
        if self.configs.len() > 1 {
            self.configs.sort_unstable();
            self.configs.dedup();
        }
        !self.configs.is_empty()
    }
}

#[inline]
fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

#[inline]
fn set_top(c: &mut Config, f: Frame) {
    *c.frames.last_mut().unwrap() = f;
}

/// Whether a value of `node`, entered `above` frames over the top one,
/// leaves room on the stack to finish.
#[inline]
fn fits(core: &Core, c: &Config, above: usize, node: NodeId) -> bool {
    c.frames.len() + above + core.need[node as usize] as usize <= MAX_STACK
}

#[inline]
fn push(c: &mut Config, f: Frame) -> bool {
    if c.frames.len() >= MAX_STACK {
        return false;
    }
    c.frames.push(f);
    true
}

/// Advance one configuration by one byte, in place. Configurations an
/// `anyOf` forks off (already advanced by `b`) go to `forks`.
pub(crate) fn advance(core: &Core, c: &mut Config, b: u8, forks: &mut Vec<Config>) -> bool {
    let ws = is_ws(b);
    macro_rules! bump {
        () => {{
            c.ws += 1;
            return c.ws <= core.max_ws;
        }};
    }
    macro_rules! consume {
        () => {{
            c.ws = 0;
            return true;
        }};
    }
    loop {
        let Some(&top) = c.frames.last() else { return false };
        match top {
            Frame::Done => {
                if ws {
                    bump!();
                }
                return false;
            }
            Frame::Value { node } => {
                if ws {
                    bump!();
                }
                let mut first = None;
                for &a in &core.alts[node as usize] {
                    // The value replaces this frame: it needs its own room.
                    if core.can_start(a, b) && fits(core, c, 0, a) {
                        if first.is_none() {
                            first = Some(a);
                        } else {
                            let mut c2 = c.clone();
                            set_top(&mut c2, Frame::Value { node: a });
                            if advance(core, &mut c2, b, forks) {
                                forks.push(c2);
                            }
                        }
                    }
                }
                let Some(a) = first else { return false };
                match &core.nodes[a as usize] {
                    Node::FreeObject { value } => {
                        set_top(c, Frame::FreeObj { value: *value, phase: OPEN });
                        consume!();
                    }
                    Node::Object { .. } => {
                        set_top(c, Frame::Obj { node: a, idx: 0, phase: OPEN });
                        consume!();
                    }
                    Node::FreeArray => {
                        set_top(c, Frame::FreeArr { phase: OPEN });
                        consume!();
                    }
                    Node::Array { .. } => {
                        set_top(c, Frame::Arr { node: a, count: 0, phase: OPEN });
                        consume!();
                    }
                    Node::String { re } => {
                        let dfa = (*re != NO_RE).then(|| &core.dfas[*re as usize]);
                        set_top(c, Frame::Str { re: *re, st: StrState::start(dfa) });
                        consume!();
                    }
                    Node::Number { spec } => set_top(c, Frame::Num { spec: *spec, st: NumState::start() }),
                    Node::Literal { set } => set_top(c, Frame::Lit { set: *set, pos: 0 }),
                    Node::Union(_) => return false,
                }
            }
            Frame::Obj { node, idx, phase } => {
                if ws {
                    bump!();
                }
                let Node::Object { keys, values } = &core.nodes[node as usize] else { return false };
                let n = keys.len() as u32;
                match phase {
                    OPEN | BEFORE => {
                        if phase == OPEN && n == 0 {
                            if b == b'}' {
                                c.frames.pop();
                                consume!();
                            }
                            return false;
                        }
                        if b != b'"' {
                            return false;
                        }
                        set_top(c, Frame::Obj { node, idx, phase: AFTER_KEY });
                        if !push(c, Frame::Lit { set: keys[idx as usize], pos: 0 }) {
                            return false;
                        }
                    }
                    AFTER_KEY => {
                        if b != b':' {
                            return false;
                        }
                        set_top(c, Frame::Obj { node, idx, phase: AFTER_VALUE });
                        if !push(c, Frame::Value { node: values[idx as usize] }) {
                            return false;
                        }
                        consume!();
                    }
                    _ => {
                        if b == b',' && idx + 1 < n {
                            set_top(c, Frame::Obj { node, idx: idx + 1, phase: BEFORE });
                            consume!();
                        }
                        if b == b'}' && idx + 1 == n {
                            c.frames.pop();
                            consume!();
                        }
                        return false;
                    }
                }
            }
            Frame::FreeObj { value, phase } => {
                if ws {
                    bump!();
                }
                match phase {
                    OPEN | BEFORE => {
                        // A key, then its value: one frame above this one.
                        if b == b'"' && fits(core, c, 1, value) {
                            set_top(c, Frame::FreeObj { value, phase: AFTER_KEY });
                            if !push(c, Frame::Str { re: NO_RE, st: StrState::start(None) }) {
                                return false;
                            }
                            consume!();
                        }
                        if b == b'}' && phase == OPEN {
                            c.frames.pop();
                            consume!();
                        }
                        return false;
                    }
                    AFTER_KEY => {
                        if b != b':' {
                            return false;
                        }
                        set_top(c, Frame::FreeObj { value, phase: AFTER_VALUE });
                        if !push(c, Frame::Value { node: value }) {
                            return false;
                        }
                        consume!();
                    }
                    _ => {
                        if b == b',' {
                            set_top(c, Frame::FreeObj { value, phase: BEFORE });
                            consume!();
                        }
                        if b == b'}' {
                            c.frames.pop();
                            consume!();
                        }
                        return false;
                    }
                }
            }
            Frame::Arr { node, count, phase } => {
                if ws {
                    bump!();
                }
                let Node::Array { item, min, max } = core.nodes[node as usize] else { return false };
                match phase {
                    OPEN | BEFORE => {
                        if b == b']' && phase == OPEN {
                            if min == 0 {
                                c.frames.pop();
                                consume!();
                            }
                            return false;
                        }
                        if count >= max {
                            return false;
                        }
                        set_top(c, Frame::Arr { node, count: count + 1, phase: AFTER_VALUE });
                        if !push(c, Frame::Value { node: item }) {
                            return false;
                        }
                    }
                    _ => {
                        // Another item only where one fits.
                        if b == b',' && count < max && fits(core, c, 1, item) {
                            set_top(c, Frame::Arr { node, count, phase: BEFORE });
                            consume!();
                        }
                        if b == b']' && count >= min {
                            c.frames.pop();
                            consume!();
                        }
                        return false;
                    }
                }
            }
            Frame::FreeArr { phase } => {
                if ws {
                    bump!();
                }
                match phase {
                    OPEN | BEFORE => {
                        if b == b']' && phase == OPEN {
                            c.frames.pop();
                            consume!();
                        }
                        set_top(c, Frame::FreeArr { phase: AFTER_VALUE });
                        if !push(c, Frame::Value { node: core.any }) {
                            return false;
                        }
                    }
                    _ => {
                        if b == b',' {
                            set_top(c, Frame::FreeArr { phase: BEFORE });
                            consume!();
                        }
                        if b == b']' {
                            c.frames.pop();
                            consume!();
                        }
                        return false;
                    }
                }
            }
            Frame::Str { re, st } => {
                let dfa = (re != NO_RE).then(|| &core.dfas[re as usize]);
                match string::step(dfa, st, b) {
                    StrStep::Inside(s) => {
                        set_top(c, Frame::Str { re, st: s });
                        consume!();
                    }
                    StrStep::Close => {
                        c.frames.pop();
                        consume!();
                    }
                    StrStep::Reject => return false,
                }
            }
            Frame::Num { spec, st } => match core.nums[spec as usize].step(st, b) {
                NumStep::Inside(s) => {
                    set_top(c, Frame::Num { spec, st: s });
                    consume!();
                }
                NumStep::End => {
                    c.frames.pop();
                }
                NumStep::Reject => return false,
            },
            Frame::Lit { set, pos } => {
                let t = &core.lits[set as usize];
                if let Some(next) = t.child(pos, b) {
                    if t.terminal(next) && t.leaf(next) {
                        c.frames.pop();
                    } else {
                        set_top(c, Frame::Lit { set, pos: next });
                    }
                    consume!();
                }
                if t.terminal(pos) {
                    c.frames.pop();
                } else {
                    return false;
                }
            }
        }
    }
}

/// Whether `bytes` are exactly one value of `node` (used while compiling, to
/// keep only the `enum` values the schema's type admits).
pub(crate) fn accepts_value(core: &Core, node: NodeId, bytes: &[u8]) -> bool {
    let mut s = State { configs: vec![Config { frames: vec![Frame::Done, Frame::Value { node }], ws: 0 }] };
    let mut forks = Vec::new();
    for &b in bytes {
        if !s.step(core, b, &mut forks) {
            return false;
        }
    }
    // A number or a literal ends on the byte after it: give it a space.
    s.step(core, b' ', &mut forks) && s.accepting()
}
