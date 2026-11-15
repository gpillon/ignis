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
chain's rows run past the frontier (#621); the target's words keep saying which of the target's ring rows
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
token; the claimant's entry must be its own (spec #618). So each holder has its own copy of the prefix's last
page, in which it writes that entry from its next token and the continuation hidden the prefix carries, and
shares the pages below it (#622, below). The cost is one page a holder.

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
to 7 and every lane count from one to eight, and on the 16 GB artifact. The greedy text is the same load's spec-off text up to the near-tie
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

**The replay's instruments read the head's chain and force its rounds as DFlash2's do** (#626). The proposal
readout under MTP is the round's own chain (`ignis::mtp::propose`), run once, eagerly, at width 1, for one sequence
staged as a round's lane at the extent asked: the same steps on the same inputs as the chain in the sequence's next
width-1 round, so the same drafts, at one chain's cost and without the round. Only where each step's scores land
moves: a block of the readout's own scratch, not the round's logits, which a host cut's repair may still read.
#620's frontier readout stays as it was for the double-precision checks: its row is built on the prefill's
attention route, not the round's. Each draft's probability is its share of the softmax
at temperature 1 over the head's BF16 scores, the whole vocabulary, taken on the host in double precision (a score
more than 40 below the step's best is left out: under 1e-12 of the sum); the leading candidates are the step's 16
best ids, the lower id first on a tie, the first the device's argmax, which the host checks. What the readout leaves
is what a round's chain leaves: the head's rows at and past its frontier, `[P - 1, P - 1 + e)`, which no committed
state holds and every later step writes before it reads. Under hq-e8-2b those appends also land on ring rows older
keys inside the window still name, and set their bits; clearing the bits, as a failed round's guard does, would be
safe but leave the next round reading the codec where a sequence never read reads exact rows. So the readout saves
the slot's head residual planes (2 x 1,114,112 bytes) and head ring words before the chain and puts them back after
it, and a sequence read before every round generates bit for bit what one never read generates, under BF16 and
under hq-e8-2b past the ring's first lap. A teacher-forced round on an MTP load needs nothing of its own: the
verdict is accept-side and reads the chain's drafts back as it reads DFlash2's, and since the head drafts under the
fixed draft only, only greedy lanes are forced. Fed the text its own rounds committed, it is those rounds bit for
bit at one and two lanes. The replay of recorded turns under MTP is its own example and record
(`crates/runtime/examples/mtp_replay.rs`, 1,016 bytes a round: the drafts, the head's probability of each, its
candidates, the accepted count, the committed run, the window in the header), with its own reader
(`tools/mtp-replay/records.py`); DFlash2's replay and its 9,520-byte record are unchanged, and a directory holds one
replay (`replay.json`). The run is its own check: it exits 1 where a round's forced run is not the read drafts'
agreement with the recording.

**The head's state travels with the sequence, and a blob belongs to one drafter** (#622). This amends ADR 0024.
The head's attention state rides the sections a sequence already has: its layer is the last plane run of every KV
page (shareable, like the target's), and under hq-e8-2b the last layer of the residual window, whose section
carries the head's ring words after the target's. The continuation hidden is a section of its own
(`IGNIS_SEQ_SECTION_MTP_CONTINUATION`, cloned), listed only by a pool built for the head. So every move of a
sequence carries them: a snapshot and its restore, whole or a window at a time, KV-RAM and KV disk (which move
those blobs), a retained slot on the device or on the host, a prefix's image and a checkpoint's.

The blob's format stays 5. A pool without the head writes the bytes it wrote before: the same sections, and the
records' room before the first payload, now eleven, still rounds to 512 bytes. A pool built for the head refused
every move until this table listed its state, so no blob of one exists to be reread; and a blob of either kind of
pool restored into the other is refused by name before its geometry is read ("the MTP head's state", the leaf's
named check on the blob's records, as DFlash2's window lane size names the other drafter). Bumping the version for
the head's pools alone would have made the blob identity name the layout version rather than the drafter, the
field that differs: an MTP load refuses a DFlash2 blob and one with no drafter as `drafter is mtp window 7 here,
dflash2 window 7 in the blob` (or `none`), the others refuse an MTP blob the same way, a KV-disk file names its
drafter as `mtp:7:full` and is refused by it, and the refused prompt is prefilled again. The format document is the
leaf's section header (`kernel/include/ignis_seq_sections.h`), whose version history says why the number holds.

The snapshot point is earned, not inherited: at a completed chunk boundary the chunk's head call has written every
row below the head's frontier and left the chunk's last column as the hidden, inside the chunk the one synchronize
confirms; after a committed round the commit has rewritten the run's rows and moved the hidden as its last device
step, and under hq-e8-2b cleared the head's ring bits of every chain row past the commit. The head's frontier is
derived from the sequence's, so nothing of its own can disagree with the progress image. Measured on the card under
BF16 and hq-e8-2b: a sequence restored at a chunk boundary, after a round with rejected drafts and after a round at
extent 0 drafts in its first round and every round after it is the uninterrupted run's, extent and committed run.

**The page holding a frontier's last position is every holder's own.** With the head, a sequence continuing from a
frontier writes the head's entry at the position before it (`ignis_seq_pool::holder_lag`, 1). A prefix of `P`
tokens shares its first `P/64 - 1` pages and keeps a copy of its last (`ignis_seq_prefix::last`) while its handle
lives, the image a claim copies; the publisher takes a copy as its first own page at the publish, each claimant at
its claim, and each writes the entry at `P - 1` from its own next token: a claimant's is bit for bit a sequence's
that prefilled the same tokens cold, never its publisher's. A chained prefix's first page is its publisher's copy,
which holds that publisher's entry, part of the chain's history. A checkpoint copies the page of its opener's last
position, so an opener on a page boundary now keeps a tail page too, and lends one page fewer to its link. A
spilled prefix's blob ends with the copy, as a checkpoint's ends with its tail. Flash-Next's head, which stores its
entries the same way, keeps its rule: a claimant there reads its publisher's last entry.

What that costs the scheduler's ledger, which charges a prefix its whole pages and a claimant the pages past them
as for any other pool, is bounded rather than tracked: at most one page more a sequence (its copy; the prefix's
last page as charged covers one holder's) and one a retained slot (a prefix's or a checkpoint's copy, each held
with the image a retained slot holds and gone with it). `ignis_core::seq::holder_page_reserve` states it, `slot_count
+ retained_slots` on the 27B's MTP pool and 0 on every other, and the 27B's load keeps it back from admission's
pages (#623 serves the backend). The one place the ledger reads the leaf's count is a capture's loan: a capture by a
sequence holding a copy lends that copy with the pages the ledger moves (the ledger charges it to the prefix), so
the scheduler takes a loan of one page more than it moves on such a load (`SchedulerConfig::kv_holder_lag`, 1 there)
and releases a capture that lent more, as before.

**The server takes the head as the 27B's opt-in drafter** (#623). This amends ADR 0046 too: its example gives MTP
the Flash-Next family alone, and `--spec-backend mtp` is now both models' value. A model lists the backends it
drafts with, its default first (`ModelFamily::drafters`: the 27B's DFlash2 and MTP, Flash-Next's MTP), so a model
switch to the 27B keeps an MTP setting instead of dropping it, and every profile and card class still gives the 27B
DFlash2. The draft window is the load's, 0 to 7 on the 27B's head; named nowhere it is 7, a provisional default the
measurement (#628) sets. A window a profile gives beside its own DFlash2 (every built-in profile's 7) is that
drafter's, so `mtp` named over a profile takes the head's default, and a profile that chose `mtp` keeps its own. A
window of 0 is a load with no speculation: the configuration resolves it to what `off` resolves to, the same engine
shape field for field, so nothing of the head is bound, no pool carries its section and the plan is the off load's
to the byte. Flash-Next's head drafts 1 to 7, so 0 named for both models is refused when Flash-Next loads, at start
and on a switch alike (a bound, as a context past the 27B's envelope is, not a drop), naming the 27B's own flag;
DFlash2 refuses 0 everywhere. A rule only the loading model decides runs where the configuration is fitted to it,
never where every model's values are validated, or a value for one would refuse a start of the other.

DFlash2's own settings are refused by name on the head's load, and only where the operator named them, never because
a default is in force: the sampled draft has been the 27B's default since #578, so the head's load drafts the fixed
draft, its only rule, without anyone having to say so. "Named" is new here, because the repository's rule that
naming a default says nothing would let `--spec-draft-rule sampled` pass silently and leave the shortlist head the
only setting the head could refuse: a flag or an env var names its value even at the default; a config file and a
live patch name a value only away from the default, because a file `config generate` wrote holds every field at its
default and must start either drafter (`Resolution::named_by_operator`). So `--spec-draft-rule sampled`, and the
drafter's temperature factor and top-p at any value, are refused by a flag or an env var; a file's `fixed` names the
head's own rule and is taken; the shortlist head is refused as it was, by either model's head. The effective
configuration document of a head's load reads back to the same speculation.

**The head drafts after images** (#624). Until then the head's load refused vision on, at start, on a switch and on
a live change, by one predicate in the configuration's fit, because every built-in profile turns the 27B's vision on
and a load quietly without it would fail at the first image a client sends; that predicate and the leaf's refusal are
gone, and the head binds beside the tower under either residency, as DFlash2 does. Three things make the head's
state right after an image.

A row rotates where the target's hidden at its position did, and its key lands at the position itself, the KV
index. Inside a multimodal span that is the position's three MRoPE axes (pair `i` at axis `i mod 3`, the text
layers' rule); everywhere else, a text span, a plain round, a verify round, it is the position plus the sequence's
rope delta, as the target's decode and verify columns rotate. The chain needs no buffer of its own for it: step `s`
of a lane rotates at the verify round's own rotation of the lane's first column (`base + delta`, staged per round
at the address its graphs read) less one, plus `s`; the commit's rows, built on the host, add the lane's delta; a
plain round's row is its rotation position less one, as its KV row is its position less one.

A row whose next token is an image placeholder embeds the image's column the target's embedding took there, not the
placeholder id's row: Flash-Next's rule (#564), the input the target itself took at that position, which every
placeholder id shares and so cannot tell one image from another. Measured on the canaries, the head so built
commits 4.67 tokens a full-window round on the text after an image, and the first round after number.png drafted
its whole window and committed all of it. Because the entry at `p` is written by the step that consumes `x_{p+1}`, that column is always
in the chunk that writes the row, so nothing like Flash-Next's continuation past a span ending inside an image is
needed: the head's call copies the chunk's media columns into its embedding, from wherever the residency put the
pool.

The row before a span that continues a sequence (a scheduler chunk after the first, a claimant of a prefix that
ends inside an image, a later turn) is built from the span's first token, at a position no column of the span
holds, and the sequence cannot say where that position rotated: its rope delta is the whole prompt's, set after
each span, and a prefix's image carries none. Keeping that position's axes as sequence state would make them a
section that every move carries; the caller already holds the whole prompt's positions. So the span names them:
`ignis_prefill_options::mrope_previous` (the options grow from 240 to 248 bytes), the prompt's column before the
span (`Multimodal::axes_before`), refused NULL on a multimodal span that continues a sequence on the head's load and
unread everywhere else. A prompt cut into spans a chunk each leaves the head's rows of the same prompt as one span,
bit for bit.

What vision adds to the head's plan is those rotations: a multimodal chunk's head call takes three I32 axes a row in
place of one (1,024 bytes at its 128 rows, inside the arena the encoder usually sizes), and the round scratch one
I32 a column of the widest round. A text-only head load's lines are what they were, to the byte. Held on the card
(BF16, number.png, rope delta -182): the prompt's rows at and around the image within the f64 bounds (keys 0.0037,
values 0.0031), the placeholder ids' embeddings and the KV positions' rotation far outside them (values 0.33, keys
0.26); the frontier's hidden 0.0056 and the chain's first scores 0.0050; the first verify round after the image
drafted its whole window and committed all of it, and its commit's rows and the plain round's row after it stand
within the bounds where the bare positions' rotation lands at 0.49 and 0.50.

**What admission keeps back is held by the plan's floor.** The scheduler bounds the holders' copies rather than
tracking them (#622, above): the ledger charges a prefix its whole pages and each holder the pages past them; a live
sequence holds one copy at most (the one frontier it continues from), a lane one live sequence, and a retained slot
one image with one copy at most, gone with it; the prefix's own last page as charged covers one holder's. So the leaf
holds at most the ledger plus a page a lane and a page a retained slot, and admission, kept within the pool less
those, never asks the leaf for a page it lacks. What #622 left was the floor: the plan held one context and a page
per retained slot, so a pool at that floor left admission a page a lane short of one context, and a request
reserving the whole context the server advertises would have been refused as larger than the pool. The plan's floor
now holds those pages too (`VramRequest::holder_pages`, from the same `seq::holder_page_reserve` admission subtracts,
0 on every other load): a named pool below it is refused naming them, and a pool at it leaves admission exactly the
context and the retained slots' pages a pool without the head leaves at its own floor. The plan event says how many
of its pages admission keeps back (`kv_holder_pages`); `kv_pool_pages` and `kv_pool_tokens` stay the leaf's pool.

The load's log names the drafter and the window (`ignis.model.loaded`'s `speculation`: `mtp draft_tokens=7`, and
`off (mtp draft_tokens=0)` for the window of 0), and nothing of DFlash2's settings; Flash-Next's line is what it was.
The request log's `spec.*` fields and the `ignis_speculative_*` counters keep their meaning: a round's extent, its
accepted drafts, the per-position profile. The head's load drafts under the fixed draft only.
