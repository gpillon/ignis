# 16 - reuse boundaries: where a decision's state can be resumed

GitHub: #270

A `state` sent as content parts (text + `image_url`) reuses nothing today:
not a long static text head across requests, not the whole state across the
questions of a fan-out. A JSON `state` is reused because it is rendered into
the system block, and the system block is the only place this engine ever
keeps state for a later request. This spec generalizes that one place into a
list of **reuse boundaries** and adds three ways of predicting them. The
prompt the model reads does not change by a byte.

Prior art: `docs/findings/2026-09-26-prefix-reuse-prior-art-for-decisions.md`.
ADR: 0029, amended 2026-09-26 by this spec.

## Problem Statement

Measured on `main` 2591900, 2026-09-26, with the issue's script (5 identical
sequential requests per case, median of requests 2-5, served 27B, hq-e8-2b,
`--vision --spec dflash2`):

| case | state | median | reuse |
|---|---|---|---|
| A | JSON `{"mission": LONG}` (~1.5K tokens) | 70 ms | prefix +13,760 tokens |
| B | parts `[LONG]` | 324 ms | none |
| C | parts `[SHORT, image]` | 56 ms | none |
| D | parts `[LONG, image]` | 339 ms | none |
| E | D with 4 questions | 1,388 ms (= 4 x D) | none |

The cause is structural, not a bug in a publish path:

- **A hybrid resumes only where it snapshotted.** The 48 GDN layers carry
  recurrent state that exists at a position only if the prefill captured it
  there (ADR 0029). Each capture is a **retained slot** (~188 MiB; 8 by
  default, shared with chat checkpoints and agents' prefixes) plus a chunk
  split. So the server must predict, before prefill, where a later request
  will resume.
- **It predicts one place.** A retained prefix is cut at the page floor of
  `system_block_tokens`. `decide::messages_for` keeps a parts state in the
  user turn (images cannot enter a system message, and moving text there
  would change measured prompt bytes), so nothing of the state is ever inside
  the cut. A fan-out's first question is sequenced (spec 04) but leaves
  nothing its siblings can claim.

The real workload is a game agent at ~1 decision/s: `[mission 2.7K][NOW +
situation][label][image]x3` with two same-kind questions. The mission
(~220 ms of prefill) is identical in every request; the situation and the
images change every step.

## Solution

**A request carries an ordered list of reuse boundaries**, each published as
a shared prefix at a chunk boundary during prefill, as the system block is
today. Four sources fill the list:

1. **Block boundary** — the end of the system block. Unchanged.
2. **Fan-out head** — the longest common prefix of a fan-out's rendered
   prompts. Computed, not detected: `/v1/decide` renders every question
   before it submits any. Published by the sequenced first question and
   given up when the fan-out ends.
3. **Observed fork** — for a parts `state`, the end of the longest run of
   leading parts that a recent request also began with. The second request
   that repeats a run publishes it; the third and later claim it. This is how
   vLLM, SGLang and DeepSeek place recurrent checkpoints without a hint,
   snapped here to part boundaries.
4. **Reuse marker** — `"cache_control": {"type": "ephemeral"}` on a `state`
   content part: a boundary at the end of that part, claimed from the second
   request on. The shape Alibaba Model Studio (this model family's vendor)
   and Anthropic use, and that OpenRouter translates to OpenAI's
   `prompt_cache_breakpoint`. A request carrying a marker is **explicit-only**:
   it gets no observed-fork boundary.

What each buys on the measured cases: B and D resume after their whole state
(a repeat is its own longest run), E's four questions prefill the state once,
and the game agent resumes after its mission and — within one step — its
second question resumes after the whole state, images included.

Matching does not change: a request still claims the longest retained state
whose **match key** is a prefix of its prompt (ADR 0029). The sources above
only decide *where state is kept*; none of them names a session or makes
reuse depend on anything but content.

## User Stories

1. As a caller sending the same long instructions before a changing image,
   I want that text prefilled once, so each decision pays for the image and
   its question only.
2. As a caller asking N same-kind questions about one image state, I want the
   state — image included — prefilled once, so N questions cost one state and
   N tails.
3. As a caller who knows exactly which part of my state is static, I want to
   say so, so reuse starts from my second request and nothing else of my
   state is kept.
4. As a caller who says nothing, I want repeated leading parts found for me,
   so an unmodified client still gets reuse.
5. As an operator, I want a fan-out's shared state to leave the device when
   the fan-out ends, so eight retained slots are not spent on states nobody
   will send again.
6. As the owner, I want the model to read exactly the prompt it reads today,
   so no accuracy number measured on `/v1/decide` needs re-measuring.

## Implementation Decisions

### The list

- A **reuse boundary** is a prompt position, in tokens, where the prefill is
  cut and the shared prefix ending there is published. The request input
  carries them **ascending**, each with its **lifetime**: *retained* (kept
  with no claimant until the device needs the room — today's retained prefix)
  or *fan-out* (kept until the fan-out that owns it ends).
- Every source produces a **byte offset in the rendered prompt**. The
  renderer turns it into a token count the way it does the system block
  today, through the placeholder expansion, and **fails closed**: an offset
  that does not tokenize to an exact token prefix of the prompt yields no
  boundary. Then each is floored to a whole KV page, walked back out of any
  media item it lands inside (#193), and capped at the request's publish
  reach (a decision holds back its prefill tail, #238). Boundaries that
  collapse onto the same position merge, keeping the longer lifetime; a
  boundary at 0 is dropped.
- Publishing reuses what exists: the first boundary is a plain publish, the
  rest are **chained prefixes** (#187), each taking one retained slot. When
  no slot is free the publish is skipped as today (`publish_skipped_no_slot`).
  Every extra boundary costs one chunk split (~19 ms fixed,
  `docs/findings/2026-09-18-prompt-reuse-tax-on-short-ttft.md`); #272 removes
  that cost later.
- `--prompt-reuse off` publishes none of them, as it publishes no block
  boundary today.

### Fan-out head

- Computed over the N rendered prompts of a fan-out with **N >= 2**, on match
  keys (token ids *and* media identity), before the first question is
  submitted. It is published by the first question only; the followers carry
  it as a boundary they claim, not one they publish.
- Same-kind questions share their instruction and the whole state, so the
  head runs through the images. Questions of different kinds do not share
  their system text (`DIRECT_SYSTEM`, `point_system`, ...), so their common
  prefix is a few template tokens and floors to nothing: **no head, no
  slot**. Sharing across kinds is a layout question (#271).
- **Lifetime.** When the fan-out ends — every question answered or failed,
  or the handler's future dropped (the unit spec 04 already cancels as one) —
  the head's retention is given up and its slot returns. A head that
  coincides with a longer-lived boundary (the same position as a marker or a
  fork) keeps the longer lifetime.
- A JSON-state fan-out gets a head too. It lands at or one page past the
  block boundary, which changes nothing measurable; it is not special-cased.

### Observed fork

- Only for a `state` given as content parts. A JSON state's evidence already
  sits inside the block boundary.
- The server keeps a bounded, in-memory **fork history**: the match keys of
  recent parts states at each **run end** — the token position where
  `parts[0..=i]` ends, for every `i` — and no state at all. Keying by the
  match key at that position means everything before the run (the chat
  template head, the question kind's system text) is part of the key, so a
  run seen under another instruction does not count.
- A request whose longest run end is already in the history gets a
  *retained* boundary there; then all of its run ends enter the history.
  The history is bounded (a fixed count of keys, least recently seen dropped
  first); its size is an implementation constant, not a flag.
- A run end at or below the block boundary adds nothing.
- Explicit-only: a request with at least one reuse marker skips the fork
  step entirely, reading and writing nothing.

### Reuse marker

- Wire: a `state` content part may carry `"cache_control": {"type":
  "ephemeral"}`. The boundary is at the end of that part (after its image,
  for an `image_url` part). Lifetime *retained*.
- Refused with 422, before any prefill, like every other malformed decision
  request:
  - `malformed_reuse_marker` — `cache_control` is not exactly
    `{"type": "ephemeral"}` (retention here is by eviction, not by time, so a
    `ttl` would be a promise not kept);
  - `too_many_reuse_markers` — more than **4** markers in one state (the
    hosted APIs' limit; each marker costs a slot and a split).
- The chat and Responses routes are unchanged: they neither honour nor
  refuse the field.

### What does not change

- `decide::messages_for` and every rendered byte of every decision prompt.
- The block boundary, prompt checkpoints (still refused for decisions), the
  retained-state eviction order (ADR 0029, #188) and KV-RAM spill.
- A multimodal claim still leaves at least one prompt token to prefill
  (#193, #201).

## Testing Decisions

- **Rendering is pinned.** A parts state with and without markers renders to
  the same bytes, and to the bytes it rendered before this spec.
- **Core, on the mock backend** (beside `crates/core/tests/retained_prefix.rs`):
  a request with two retained boundaries publishes both, as a chain; a later
  request claims the longer one it matches; a *fan-out* boundary is released
  when its owner says the fan-out ended and not before; a fan-out boundary at
  the same position as a retained one stays retained.
- **Server, over the mock engine:** fork history (1st request nothing, 2nd
  publishes, 3rd claims; a different first part publishes nothing; a varying
  image after a repeated text part does not move the boundary; a request
  with a marker neither reads nor writes the history); marker refusals; the
  fan-out head of same-kind and mixed-kind fan-outs; the OpenAPI document.
- **GPU, once at the end** (after checking the card is free, AGENTS.md): the
  issue's script against a live server, extended with a marker case and kept
  in the repository so the acceptance can be rerun.

## Acceptance

1. **Bytes.** Every decision prompt renders to the same bytes as before this
   spec, with and without markers.
2. **The list.** A request publishes every reuse boundary it carries, in
   order, one retained slot each, and a later request claims the longest one
   its prompt matches. Asserted in core tests.
3. **Fan-out head.** Four same-kind questions over a parts state with an
   image prefill the state once: every follower's prefilled tokens are at
   most its own tail plus one page. Asserted on prefill token counts, not on
   wall time. This meets the prefill half of spec 04's acceptance 2.
   A fan-out of two kinds (`choice` + `point`) publishes no head.
4. **Fan-out lifetime.** After a fan-out ends — answered, one question
   failed, or the client gone — `ignis_retained_slots{state="in_use"}` is
   back to its value before the fan-out, unless the head coincides with a
   marker or fork boundary.
5. **Observed fork.** Over `[static text >= 2 pages][varying text][varying
   image]`, the first request publishes no fork boundary, the second
   publishes one at the end of the static part, and the third and later
   claim it (reused prefix tokens >= the static part's page floor). A
   request whose first part differs publishes none.
6. **Marker.** A marker on a part puts a boundary at that part's end, claimed
   from the second request; a request with a marker gets no fork boundary;
   the two refusals answer 422 with their codes.
7. **Live** (the committed script, the issue's server flags, 5 sequential
   requests per case, median of requests 2-5):
   - B <= 1.3 x A;
   - D <= C + 30 ms;
   - E <= 2 x D, and E's first request <= 1.5 x D's first request (the
     fan-out head alone, before any cross-request reuse);
   - with a marker on D's text part, D's second request is already
     <= C + 30 ms;
   - `ignis_retained_reused_tokens_total{kind="prefix"}` grows in B, D and E.

   Recorded as a finding with a README row.
8. **Docs.** ADR 0029's amendment matches what was built, `CONTEXT.md`'s
   reuse terms match it, and the OpenAPI document shows the marker and both
   refusals.
9. `cargo test` passes workspace-wide.

## Out of Scope

- **Sharing across question kinds** — moving the per-kind instruction after
  the state: #271 (a layout study with an accuracy re-measure).
- **Capturing a boundary without a chunk split**: #272.
- Markers or observed forks on the chat, Responses or a future
  `/v1/messages` route.
- Prefilling a fan-out's suffixes in one batched pass (#235).
- A retention policy smarter than today's LRU (#199), and any per-part
  "never retain this" flag.

## Further Notes

- **Why not move the leading text into the system block** (the issue's
  option 1): it changes measured prompt bytes, never shares an image, and
  fails the game agent's own shape — its situation text sits before the
  first image, so the block would change every step.
- **Why not a "first text part" rule**: nobody publishes one, vLLM tried
  "semantic" template boundaries and closed it with no measured gain (#49574),
  and it spends a slot every time the first part is the one that changes.
- **The closest external design** is vLLM RFC #55697 (client marker +
  same-step producer/consumer scheduling + two-phase GDN prefill for 1-to-N
  multimodal scoring on Qwen3.5). Its scheduling and kernel halves are what
  its reviewer called "lots of complexity"; this spec takes only the marker.
  Same-step pairing would make a decision a live publisher, which #238 rules
  out.
- **ADR 0029 listed this** under considered options: "A learned
  (observed-LCP) retained-prefix boundary. To revisit only if measurement
  shows shared heads extending past the system block." #270's measurement is
  that.
- Expected for the game agent, not measured: stage 1 from ~660 ms to
  ~140 ms (its first question resumes after the mission, its second after
  the whole state).
