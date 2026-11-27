---
status: accepted
---

# Rope scaling is `auto` by default: the YaRN factor follows the context

The checkpoints are trained over 262,144 positions, the trained position envelope, and a context past it needs the
YaRN table at the factor `max_context / 262,144`: YaRN's own `s = L'/L`, and what Qwen3-Next's card uses (4 over
262,144 original positions for about 1M, 2 for 524,288). A profile that raised `max_context` had to spell the
matching `yarn:F`, per family since the 27B and Flash-Next sit at different contexts on the same card, and a
mismatch is silent: a lower factor leaves positions extrapolated, a higher one degrades short contexts for nothing.
So `model.rope_scaling` accepts `auto`, the default for both models (Flash-Next applies the table as ADR 0059
says): the table is resolved at load, per model, from that model's effective `model.max_context`.

- **Within the envelope `auto` is no scaling**: `RopeScaling::NONE`, the table the engine has always used, not
  "YaRN at 1".
- **Past it `auto` is YaRN at the exact ratio**, with the default temperature and ramp: the owner's 2, 3 and 4 at
  524,288, 786,432 and 1,048,576 tokens.
- **Past the widest factor a load takes** (`MAX_YARN_FACTOR` times the envelope) `auto` has no table, and the start
  is refused naming the context.
- **`none` and `yarn:F[,...]` still override it** from any source, a model's own section included.
- **The built-in profiles spell no factor**, so each model runs the factor of its own context.
- **The setting stays as set, and identity follows the table.** `GET /v1/config` and `config print` show the
  configured value. The table every identity keys on (KV-disk, retained pages, the switch's reload check) is the
  resolved one, so `auto` at 1,048,576 tokens and `yarn:4` are the same load, and a `PATCH` between them applies
  with no reload. A switch, or a `PATCH` of a model's `max_context`, resolves `auto` again against that model's
  context.

## Considered Options

- **Keep the profiles' explicit `yarn:4`, with `none` still the default**: rejected. A factor per profile that each
  model's `max_context` has to match by hand is the silent mismatch above, and the owner asked for the context to
  decide it.
- **`auto` rounds the ratio up to a whole factor** (655,360 tokens would be 3, not 2.5): rejected. The owner's three
  points are exact ratios, and a rounded-up factor scales past what the context needs, which degrades short contexts
  for nothing. No measurement favours whole factors.

## Consequences

- A context within the envelope loads the table it always did. A start that raises a model's `max_context` past the
  envelope gets YaRN without a second flag.
- Each factor is the smallest that covers the context, which is what the owner asked for, and none is a measured
  optimum: the 27B's factor on the `rtx5090` profile is unmeasured against the `yarn:4` it replaced, and
  Flash-Next's factors are unmeasured at long context.
