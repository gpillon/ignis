# Testing `/v1/decide`

## `locate` over a text state (GitHub #275, ADR 0041)

`crates/server/tests/attention_readout_text_gpu.rs` (`cuda`, `attn-tap`)
holds the leaf's rows of the served vote's 32 heads to the attention tap over
text spans of up to 4,526 keys, under BF16 and hq-e8-2b, for a question and
for its content-free twin claiming the question's retained state.
`kernel/tests/test_attention_readout.cu` has the rows arm,
`test_model_load_vision_options.cpp` the text room, and
`crates/runtime/tests/vram_plan_gpu.rs` states the room in the plan
(26,409,216 B since GitHub #278: 32 rows over the shortlist's 200,000-key
window). CPU-side: `crates/core/tests/locate_reading.rs` holds the vote to
`score.py`'s golden cases, `locate_segments.rs` the segmentation,
`decide_http.rs` the endpoint over the mock, and `decide_locate_prompt.rs`
(artifact present, else a skip) the served prompt to set D's recorded
renders. The acceptance runs through a live server:
`tools/locate-sets/served.py`.

The shortlist (GitHub #278, spec 22, ADR 0042): `tools/locate-sets/golden22.py`
writes the golden cases the host functions are held to — the template fold
(`crates/core/tests/locate_fold.rs`), the end and sum readings, windows, merges
and shortlists (`locate_readings.rs`), the labelled renders and the `found`
rules (`locate_renders.rs`), `auto` and spaced JSON
(`crates/server/tests/locate_auto.rs`) — from the Python the research measured
with. `decide_http.rs` holds the route over the mock (defaults, every
combination, refusals, windows with a `window_keys` set small on the load's
calibration, a fold's order of steps, the last request, sharing, fan-out,
metrics), `locate_request_log.rs` its log line, and the vote's tests ask for
`method: "vote"`, `compression: "none"` by name.
`crates/server/tests/locate_shortlist_gpu.rs` answers the synthetic fixtures'
questions on the card through `/v1/decide`, one of them past a window.

## The entailment fixture

`crates/server/tests/fixtures/entailment/` is a contributed set of 18
synthetic "does the cited evidence support the claim?" cases for `noul`
(provenance, ignis's corrections and the tiers in its `NOTICE.md`).
A case is a `gate` when it is right by at least one logit as the minimum
over five measured arms (the harness under each GDN prefill, the live server
under each and under the int8 prompt-attention band), and the gate holds a
case of each label; the fixture records the rule, the date, the build and
every case's logit in every arm. `decide_entailment_gpu.rs` asserts the label
of the `gate` cases only, on the card's default route, and prints every
case's p(yes) and logit; `watch` and `known_failure` cases are printed, never
asserted. `decide_entailment_fixture.rs` checks the set's shape on CPU: every
tier is the one the rule gives from the case's logits, every label has a gate
case, and every cited name is a public one. A change of the numerics that
moves a watch case is not a fault; re-measure the five arms and the tiers
follow.
