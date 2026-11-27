# Running the gate

ADR 0082's gate admits an artifact, or a change to a served route's numerics,
by three legs measured on ignis alone, in the load the artifact serves. The
decision, what passes each leg, the limits and who approves one are the
ADR's. This page says how each leg is run. The release's sm_86 gate is in
[testing.md](../testing.md#the-sm_86-gate).

The sets the legs read are not in the repo (ADR 0082), so the gate runs where
they are: check `make gpu-status` first, as for any GPU run.

## KLD against BF16

- The engine's `span_logits` dump of set A, at the serving configuration
  (hq-e8-2b KV, the served chunk), is scored per position against the model's
  own BF16 checkpoint, run layer by layer: KL(BF16 ‖ engine) over the full
  vocabulary, the mean over every position, with top-1 and a window-bootstrap
  95% CI beside it. The tools, set A's windows and the commands are in
  [`tools/kld-gate/README.md`](../../../tools/kld-gate/README.md) (`ref_kld.py`,
  `gate1.py`).
- A fine-tune is scored against its own BF16, with `gate1.py`'s two cross
  terms.
- Where the card cannot run the checkpoint, the reference is the BF16 model's
  64 most probable tokens per position, built once on the RTX 5090
  (`ref_top64.py`, `score_top64.py`).

## DFlash2 on against off

- Nine decode cells, the ones #340 measured: 8,192, 65,536 and 253,440 tokens
  of context, each at one lane cold, at two lanes, and at one lane with reuse.
  Decode tok/s, the drafter on against off.
- Reported beside it, not binding: #267's teacher-forced replay of 200 recorded
  agent turns, tokens per round against the counterpart's, with the 0.90 floor
  that #340 and #422 held as binding.

## Loop rate

- #340's loop set, four prompts at seeds 101 to 108, 32 requests, runs on the
  artifact and on its counterpart: one build, the same prompts and seeds.
- A request loops if the guard rewound it, if its reasoning holds a 6-gram at
  least 25 times, or if its worst 3,000-character zlib window compresses below
  0.20.
- The leg fails if the artifact loops more, at a one-sided Fisher exact p
  below 0.05 on the pooled counts.
- The loop guard stays at its default for this leg alone (ADR 0082).
