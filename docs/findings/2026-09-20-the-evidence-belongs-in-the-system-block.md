# The evidence belongs in the system block, not merely first

- Kind: defect
- Status: current
- Observed: 2026-09-20
- Last verified: 2026-09-20
- Scope: serving / decisions, the `/v1/decide` prompt, cross-request state reuse, spec 04's fan-out
- Related: `docs/findings/2026-09-20-evidence-first-needs-explicit-key-order.md`, `docs/findings/2026-09-19-typed-option-logit-readout.md`, ADR 0029, ADR 0034, `crates/server/src/decide.rs`, `crates/server/tests/classify_readout_gpu.rs`, `crates/server/tests/decide_prompt_tail.rs`, GitHub #238, #239, #240
- Superseded by: none

## Question

Spec 04 buys its fan-out with one sentence: put the evidence first, and
twenty questions over one `state` prefill that state once instead of twenty
times. The previous finding fixed the *bytes* — the payload really does begin
with the evidence now — and left the reuse claim standing.

While implementing #240 the obvious question was asked of the engine rather
than of the prompt: **which reuse tier is a decision's sibling actually
claiming?**

None.

## Evidence

A claimant can stand on one of three things (ADR 0029), and a decision is
disqualified from two of them by what makes it cheap:

- a **live shared prefix** needs its publisher to still be running. A
  decision terminates in the same tick its prefill does (#238), so by the
  time a sibling is submitted there is no publisher left;
- a **prompt checkpoint** is refused outright for a decision
  (`ConcreteScheduler`'s capture point, #238);
- a **retained prefix** outlives its publisher — the one tier that fits — and
  is cut at `Request::retained_prefix_point`, which is
  `floor(system_block_tokens, KV_PAGE_TOKENS)` and `0` for a prompt that
  reports no system block.

Nothing outside the system block is ever part of a retained prefix. Evidence
at the head of the user payload is therefore first, shared, and re-prefilled
by every sibling — the saving was never in reach, whatever the byte order.

`decide_prompt_tail.rs` measures it on the real tokenizer and the real chat
template, with no GPU:

| `state` | system block | retained prefix | prompt | each sibling re-prefills |
| --- | --- | --- | --- | --- |
| Jev's documented example (one sentence) | 51 | **0** | 119 | 119 |
| a realistic ticket (12 sentences) | 388 | 384 | 456 | 72 |

The fix is to render the instruction *and the evidence* as the system
message, leaving the question in the user turn. That is a different prompt
from the one every accuracy number was measured on, so both were swept over
SemIf's 144 authored rows on the served 27B, on one load, in one run
(`classify_readout_gpu.rs`, two layouts over the same rows):

| | measured layout | shipped layout |
| --- | --- | --- |
| accuracy | 0.938 | **0.965** |
| balanced accuracy | 0.934 | **0.963** |
| argmax in slots | 144/144 | 144/144 |
| allowed mass, median | 0.9983 | 0.9988 |
| allowed mass, min | 0.9288 | 0.9596 |
| prompt tokens, median | 132 | 135 |

The two layouts give the same answer on 137 of 144 rows; of the seven they
disagree on, the shipped layout is right on four more than it loses.

## Implications

**The re-measurement the previous finding asked for is taken**, and it is of
the prompt `/v1/decide` actually sends: the sweep builds the shipped layout
by calling the endpoint's own `prepare` and `messages_for` rather than a copy
of them, so the two cannot drift apart. 0.963 balanced accuracy supersedes
0.934 as the number describing the served prompt.

**Four rows on 144 is not a quality claim** and none is made. What the sweep
resolves is the only question that could have blocked the change — whether
moving the evidence between turns broke the readout — and it does not: the
mass stays at 0.999 and the unrestricted winner is a declared letter on every
row, as before.

**A page floor is not a rounding error.** A retained prefix that starts below
one page keeps nothing at all, so a fan-out over a short `state` shares
nothing however the prompt is built — Jev's own documented example included.
That is a property of the state's length rather than a defect, and it is
reported by the test rather than left to be rediscovered. A fan-out is worth
asking for when the state is a ticket, a transcript or a document, which is
also the only case where the N x 16K prefill it prevents would have hurt.

**Spec 04's acceptance 2 cannot be met as written.** `check_content_parts`
refuses media in a system or developer message (#175), so an image `state`
cannot go where the retained prefix is cut: a fan-out over an image
re-encodes it per question. The refusal is pinned by a test
(`decide_wire.rs::an_image_state_stays_in_the_user_turn`) so the gap is
visible rather than assumed away. Lifting that refusal, or #235's
one-request-N-suffix shape, are the two ways out and both are somebody's
decision, not an implementation detail.

Worth carrying elsewhere: **"shared" and "reusable" are different
properties.** Two prompts having a common prefix says nothing about whether
the engine can claim it — that depends on which tier the publisher qualifies
for and where that tier is cut. A reuse argument that reasons about text
rather than about the cache is an argument about a saving nobody gets.
