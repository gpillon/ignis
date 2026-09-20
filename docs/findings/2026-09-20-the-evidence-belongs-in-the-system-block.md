# The evidence belongs in the system block, not merely first

- Kind: discovery
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

## Finding

**Observed.** A decision's sibling claims no reuse tier at all when the
evidence sits in the user turn: the two facts above — no live publisher, no
checkpoint — are `ConcreteScheduler`'s, and `retained_prefix_point` is
`floor(system_block_tokens, KV_PAGE_TOKENS)` by inspection. The token counts
in the table are measured by `decide_prompt_tail.rs` on the loaded artifact's
own tokenizer and chat template. The A/B numbers are measured on the served
27B over all 144 rows.

**Inferred.** That the shipped layout is *better* rather than merely not
worse is an inference from four net rows out of 144 and is not claimed as
one. What the sweep establishes is the negative: moving the evidence between
turns did not break the readout.

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

**Spec 04's acceptance 2 is unmet, and not for the reason first given.** An
image `state` stays in the user turn, so a fan-out over an image re-encodes
it per question. The first version of this document said
`check_content_parts` made that impossible; it does refuse media in a system
message (#175) but it is called only from the chat and responses routes and
never on the decide path, so it does not bind here. What actually stands in
the way is two things and a choice:

- an image in a system message would reverse, on this one route, a policy
  the server enforces on every other, against a chat template nobody has
  checked renders it at all;
- and inside the block it would usually be excluded anyway —
  `prefix_floor` walks the page floor back out of any media item it lands
  inside (#193), so a retained prefix keeps an image only when a whole page
  of something else follows it within the block. Image first, then the
  instruction, is a third prompt layout, with its own measurement.

So there is a route, and it is a design fork rather than an impossibility:
this finding's earlier "cannot" was wrong. Taking it, or #235's
one-request-N-suffix shape, is the owner's call.

Worth carrying elsewhere: **"shared" and "reusable" are different
properties.** Two prompts having a common prefix says nothing about whether
the engine can claim it — that depends on which tier the publisher qualifies
for and where that tier is cut. A reuse argument that reasons about text
rather than about the cache is an argument about a saving nobody gets.

## Limits and unknowns

- **144 rows, one model, one sweep.** No confidence interval is computed and
  the 0.934 -> 0.963 difference is 4 rows; a second sweep could move it back.
  The mass and in-slot figures are the ones this rests on, and those are
  flat.
- **The rows are authored, English, and short** (median 135 prompt tokens).
  Nothing here says how either layout behaves on the long states that are
  the whole reason a fan-out exists — which is also the regime where the
  reuse pays.
- **The reuse itself is measured on `MockCompute`, not on the card.** That a
  follower prefills 72 tokens instead of 456 is a scheduler-level fact
  (`decide_http.rs::twenty_questions_over_one_state_prefill_it_once`); no GPU
  run has yet timed a fan-out end to end, so the wall-clock saving is
  inferred from the token count rather than observed.
- **Nothing was measured for an image `state`**, which does not use this
  layout.
- **The fan-out's saving is conditional on the engine's configuration.** It
  needs `--prompt-reuse` on and a free retained slot (`--retained-slots`
  above 0); with either off the questions still answer and silently pay full
  price. No test covers that degradation.
- **A block boundary the tokenizer disagrees about disables it silently.**
  `ArtifactTemplateProvider::system_block_tokens` returns `None` unless the
  block's text tokenizes to an exact prefix of the whole prompt — correct,
  and it means an unusual `state` could restore the N-fold prefill with
  nothing reported. `decide_prompt_tail.rs` checks one state on the real
  tokenizer; the CPU tests use a word-hash template that cannot disagree
  with itself.

## Follow-ups

- Whether the image fan-out is worth either exit — lifting #175's refusal of
  media in a system message, or GitHub #235's one-request-N-suffix shape — is
  a design fork and the owner's call. Unfiled on purpose.
- A GPU run that times a real N-question fan-out would turn the token-count
  saving into a latency one. GitHub #241 owns the metrics that would report
  it.
