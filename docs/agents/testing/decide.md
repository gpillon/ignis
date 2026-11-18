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

`crates/server/tests/fixtures/entailment/` is a set of 20 synthetic "does
the cited evidence support the claim?" cases for `noul`: 18 contributed,
and 19 and 20 written by ignis (provenance, ignis's corrections, its own
cases with every wording tried, and the tiers in its `NOTICE.md`). Each
case's tier follows from its logits over the fixture's arms by one rule, the
fixture's `measurement.rule`, which `tier_by_rule` in
`crates/server/tests/support/entailment.rs` applies; the fixture also
records the date, the build and every case's logit in every arm.

The arms (GitHub #675: `measurement.arms`, `support/decide_arms.rs`) name
what moves these reads: the exception weights, the context, the rope table,
the GDN prefill route, the prompt attention band and prompt reuse. None
reads a config file. A `harness` arm is the shape the test builds by hand;
a `served` arm is a start under the `rtx5090` profile with each of those
values named as a flag, resolved with no file and no environment and loaded
through the server's load path. The fourteen arms cover BF16 and FP8 on the
harness and at each one's served shape on the RTX 5090, and under each
every switch that moves a read: the rope table, the vendored GDN prefill,
the int8 band, prompt reuse off.

`decide_entailment_gpu.rs` asserts the label of the `gate` cases only, on
every arm it loads, and prints every case's p(yes) and logit; `watch` and
`known_failure` cases are printed, never asserted. `IGNIS_DECIDE_ARMS`
picks the arms, one load each: a comma list of names, or `all`; unset,
`h-bf16` and `h-fp8`, the harness under each exception weights. Each arm
ends with a `logits <arm> {...}` line in the fixture's form, and
`IGNIS_DECIDE_DRAFTS=1` also reads the recorded drafts (a `drafts <arm>
{...}` line). `decide_scalar_gpu.rs` runs on the same arms, with the same
default; its balance case (−0.75) is a printed watch line, never asserted,
its sign being a coin under both exception weights (#670, #671).
`decide_entailment_fixture.rs` checks the set's shape on CPU: 20 cases, 19
and 20 ignis's and "yes" gates, every tier the one the rule gives from the
case's logits, every label a gate case, every arm loading the shape it
names, and every cited name a public one. A change of the numerics that
moves a watch case is not a fault: re-measure every arm
(`IGNIS_DECIDE_ARMS=all IGNIS_DECIDE_DRAFTS=1`, under 5 minutes on the RTX
5090) and the tiers follow.
