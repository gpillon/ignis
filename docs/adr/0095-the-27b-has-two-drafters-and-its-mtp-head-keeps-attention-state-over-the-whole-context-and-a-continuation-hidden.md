---
status: proposed
amends: [ADR-0024, ADR-0046]
---

# The 27B has two drafters, and its MTP head keeps attention state over the whole context and a continuation hidden

The 27B drafted with DFlash2 alone, a second small model: 1.08 GB of weights, a 40 MiB drafter window in every
lane and retained slot, a workspace for its round and scratch in every prefill chunk. Every 27B artifact ignis
ships also carries the checkpoint's own draft head, its MTP layer: 12 `mtp/` objects, five W8G32 matrices and
seven BF16 gains, 451,267,584 bytes, the same bytes on the NVFP4 and the 16 GB artifacts (the head is never
trellis). Spec #618 makes that head the 27B's second drafter, the cheap one the small cards need. DFlash2 stays,
and stays the default; MTP is chosen at load.

**The head is the checkpoint's, and its program is ignis's own.** A row of the head at position `p` takes the
target's post-final-norm hidden `h_p` and the token `x_{p+1}` after it: `x = W_fc [RMSNorm(embed(x_{p+1}));
RMSNorm(h_p)]`, then one gated full-attention layer (24 query and 4 KV heads of 256, q/k norm, the target's
rotary table at `p`, a sigmoid output gate) whose attention reads the head's own rows `0..p`, a dense SwiGLU MLP
and the head's final norm; the target's output head scores that hidden. The head reads the target's token
embedding and output head; it has neither of its own. The binding, the stem, the layer's composition, the
state, the readout and the round are ignis's (`kernel/src/mtp_head.h`); nothing of the reference engine's MTP
is ported, and its source is a design reference. The head's matrices run on the op layer every 27B matrix runs
on: the vendored W8 `ops::linear`, whose registry already carries the head's five geometries (they were vendored
with the linear op). The fused projections (`attn_input_proj`, `linear_add`, `linear_swiglu`) are not
registered for W8 at the 27B's shapes, so the layer composes `linear`, a split of the fused projection's rows
(query, key, gate, value), `residual_add` and `silu_mul`. Those W8 routes are BF16 tensor-core MMA with FP32
accumulation, which the full-rate rule refuses on RTX 30 and 40 cards; the head's routes there are #627's.

**The head's attention state spans the whole context, in the KV pool, as one more attention layer.** The head
was trained attending over every row before it; a head that sees only the last rows would save the state's bytes
and was never trained so (spec #618, out of scope). A pool built for the head (`speculative_backend` 3) stores
it as the last plane run of every page, 17 attention layers in place of the 27B's 16, in the load's KV format:
the pool's bytes per token grow by one layer in sixteen (69,632 under BF16, 9,792 under hq-e8-2b), and the
pages are shared, cloned and freed as the target's are. The head's frontier is not one of the sequence's
`gqa_positions`, whose 16 entries and whose progress image stay the target's: it follows the sequence's own
(`ignis_mtp_head_frontier`). A pool and a blob without the head are what they were, byte for byte: 16 layers,
the same sections, snapshot format 5. Under hq-e8-2b the head's layer takes one more layer's residual planes and
ring words of its own, after the target's, because its appends do not land where the target's do (below) and a
chain's rows will run past the frontier (#621); the target's words keep saying which of the target's ring rows
are exact.

**The entry at `p` is stored at `p` and written when `x_{p+1}` is consumed.** A step that consumes positions
`[a, b)` writes the head's rows `[a - 1, b - 1)`: the first from the continuation hidden, the target's hidden at
`a - 1` that the step before it left, the rest from the step's own hiddens; a prompt's first chunk has no row
at -1. So after any completed step the head holds every position but the sequence's last, whose entry waits for
the token the sequence consumes next. That is what makes the head right when the host replaces a pending token
(a grammar's repair, a forced close, the loop guard): the entry is built from the token consumed, never from
one drawn and then replaced, and nothing has to be rewritten. Storing each entry one slot later (`p + 1`), so
that a step writes only the slots of the tokens it consumes, was the other layout. The attention kernels attend
every key from 0 to the query's position, so that layout leaves slot 0 empty in every query's keys; excluding it
needs a first key in every attention route the head runs (the vendored BF16 and hq kernels, ignis's own hq
prompt and decode routes), which is out of proportion to what it buys.

What that costs is at a shared prefix. A prefix is whole pages, and the entry at its last position `P - 1` is
written after the publish, by the publisher's next step and by every claimant's first, each from its own next
token; the claimant's entry must be its own (spec #618). So #622 gives each holder its own copy of the prefix's
last page, in which it writes that entry from its next token and the continuation hidden the prefix carries,
and shares the pages below it. The cost is one page a holder.

**A sequence keeps a continuation hidden**: the target's post-final-norm hidden at its last consumed position,
one BF16 vector of 5,120 per pool slot. It is the half of the entry at the last position that the step which
consumed that position had: without it, the next step would need the target's hidden of a position it does not
compute. A restored or claimed sequence that carries it builds its first head row, and drafts, in its first
round. The target keeps its pre-norm residual (`verify.hidden`), and a prefill normalized only its last column;
on an MTP load a prefill normalizes every column the head reads, in calls of at most 128 rows, and leaves the
chunk's last column as the continuation hidden; a plain round writes each lane's row before its position from
the continuation hidden and the token it consumes, and leaves the round's post-norm hidden.

**The head drafts at the start of a verify round, and the round's commit rewrites the head's rows from the
committed run alone** (#621). At the round's proposal seam, inside its pass and the graph captured per lane count,
each lane's head chains its drafts: step 0 builds the row at `P - 1` (`P` the lane's frontier, its anchor the
pending token) from the continuation hidden and the anchor, step `s` the row at `P - 1 + s` from the head's own
hidden at step `s - 1` and the draft that step chose; step `s` chooses draft `s + 1`, the argmax of the target's
output head over the head's hidden, at every temperature (the fixed draft, taken by the accept the fixed draft uses
today). A lane needs steps `0 .. e - 1` for its extent `e`; a step past them is the attention's masked empty row and
writes nothing, so a lane at extent 0 writes no row and none writes past its budget or its context. The chain's rows
are scratch: built from the head's hidden, not the target's, they are not entries. After the cut the commit writes
the rows the committed run of `c` tokens licenses, `[P - 1, P - 1 + c)`, again, from what the sequence consumed: the
continuation hidden and the anchor, then the target's post-final-norm hidden of each committed column (the pass's
own normalization of the residual it kept) and the run's next token. One head call over the round's `lanes x (k +
1)` columns, its attention in the masked form, `c` valid columns a lane, and only its stem and projection: a
committed row's key and value are all the state takes from it. The continuation hidden then moves to the run's last
column, the round's last device step, so a round that fails before it leaves the lane's frontier and its hidden
together. The chain's rows past the commit, `[P - 1 + c, P - 1 + e)`, stand past the head's frontier and are never
read before a later step writes them; under hq-e8-2b their ring bits come off in the head's own words, as the
target's rejected columns' do in the target's, and a round that fails after its pass clears the head's words over
`[P - 1, P + e)`. The commit is eager, beside the GDN fold and DFlash2's append: it reads the cut, which the host
decides.

What "the head's state is the committed run's" is held to (spec #618 asked for bit for bit against a sequence that
committed the same tokens by plain rounds). That cannot hold, for a reason outside the head: a committed row is
built from the target's hidden at its position, and the target's hidden is the round's that computed it. A plain
round's is the one-column decode's, a verify round's the verify tile's (ADR 0056's drift), and a verify round's at
another extent already parts (measured: the same load committing the same canaries one token a round, at extent 0,
leaves the drafted text at a near-tie at one window and two lanes). The head's own call also runs at the round's
width, where the W8 small-T route splits its K by the width. So the claim is held in each round instead, under BF16
KV, drafts accepted and rejected: the continuation hidden is the round's last committed column's, bit for bit; the
head's rows at the run's positions are the head's in double precision over what the run consumed (the continuation
hidden before the round, the round's own committed columns, the run's tokens) within the BF16 bounds #620 measured;
and a row built as the chain builds its own (the head's hidden in place of the target's) lands far outside them
(measured: 0.0038 keys and 0.0034 values against bounds of 0.0096 and 0.0087; a chain-built row 0.35). Against
plain rounds held to the same tokens the drafted rows stand within twice the distance a prefill of the same text
stands from them, two correct routes (measured 0.087 and 0.103 relative L2), and a chain-built row past both.

The head leaves the target's computation as it was: the verify substrate alone, given each round the drafts it
committed and one the target rejects, commits the same runs at the same extents, bit for bit, at every window from 1
to 7 and every lane count from one to eight. The greedy text is the same load's spec-off text up to the near-tie
rule, with one exception the rule does not know and that names a new threshold: where both prefill routes put
spec-off's pick ahead but the verify substrate, given the same drafts, makes spec-on's, the two picks may stand at
most 16 BF16 ulps apart at the routes' top logit (ADR 0056's moved-pick bound, measured moved picks 0 to 8 ulps on
Flash-Next's verify rows): the verify tile's drift from a one-column decode, which any drafter's accepted drafts
expose, not the head's. Measured once on the canaries, at four lanes: 3 ulps.

**The plan states the head's lines, and nothing of DFlash2.** `weights` holds the head's 451,267,584 bytes;
`workspace` adds one head call of `min(chunk, 128)` rows (its inputs, its buffers; the call's attention takes
the workspace the target's GQA layers already hold at the same widths); `decode_graph` adds one call over the
widest round's `lanes × (k + 1)` columns; `lane_state` and `retained_slots` add 10,240 bytes a slot;
`hq_residual_window` adds the head's layer, 2,228,288 bytes a slot; the KV pool's page is 17/16 of the 27B's.
No drafter weights, window, round or tap scratch. The verify round (#621) adds no line: its chain runs in the
round's scratch in a scope of its own, released before the pass bumps its buffers, its scores in the round's own
logits, which the pass writes whole after it; its commit is the one head call over the widest round's columns the
`decode_graph` line already holds. `ignis_core::mtp_head` states each line on the CPU and a GPU
test holds the leaf's plan, and the load, to it to the byte.

**The head's state is held to its weights in double precision.** A test-only readout beside the program
(`kernel/include/ignis_mtp_readout.h`, the prefill's `out_span_hidden`) reads the head's state, its first draft
at the frontier and its scores; `ignis_artifact::f64_reference::evaluate_mtp_head` recomputes them from the
stored `mtp/` weights from the exact BF16 inputs the device read, under BF16 KV, within the relative L2 that the
BF16 roundings on each path allow. On the 16 GB artifact, whose output head is trellis and decoded by no host
code, the scores are the device's own output head over the f64 hidden; the head's every weight is already in the
state and hidden it is held to.

This amends ADR 0024. A pool built for the head holds two pieces of state that are sections to be, the head's
ring words and the continuation hidden, beside the head's layer in its pages; they are not in the section table
yet, so until #622 lists them (and earns their snapshot point) such a pool refuses every snapshot, restore,
retained slot, prefix and checkpoint by name rather than move a sequence without its head. A blob names its
drafter in its identity already (ADR 0029: "mtp window 7"). This amends ADR 0046 too: its example gives MTP the
Flash-Next family alone, and MTP is the 27B's as well once the server accepts the option (#623); until then a
27B load built for the head is reached through the leaf, the core load API and the runtime's compute seam only.
Until #624, an MTP load takes no vision; it drafts under the fixed draft only.
