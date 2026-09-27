# ADR 0042 — a locate copies its answer, over a folded state by default

## Status

Proposed (2026-09-28 — spec `docs/specs/decide/22-locate-by-copy-over-a-folded-state.md`,
GitHub #278). Accepted when spec 22's acceptance holds, with its numbers
written here. **Extends ADR 0041**, whose head vote becomes one of two
methods, unchanged, and **ADR 0034**, whose constrained decode gains a second
shape.

Sources: `docs/findings/2026-09-27-locate-at-length.md`,
`docs/findings/2026-09-27-the-instance-is-read-while-copying.md` and
`docs/findings/2026-09-28-locating-a-line-in-real-logs.md` (specs 20 and 21).

## Context

ADR 0041 serves `locate` as a vote of 32 attention heads at the copy
scaffold `{"quote":"`, one prefill, measured to 4,554 keys. The logs a
`locate` is for are 16K to 200K tokens, and on them the vote fails for a
reason length does not explain alone: it finds the right *kind* of line and
loses the instance among its near-duplicates — 28 of 50 on set R, 20 of 58
(34.5%) on the fresh set R2, 81% misses on targets with six or more
siblings. The instance is resolved while the line is **copied**: the heads
settle on the target where its written prefix stops matching every other
line (a median 26 tokens; rho 0.59 on R2). Generating the line on the whole
log reads 86.2% of R2 but pays that log's prefill (median 47 s, 139 s at
200K tokens) and matches free text back to a line.

The owner's idea closes the cost: fold near-duplicate lines into templates,
find the template, then the instance among its values, and unfold. On R2
that route, copying at each level, read 55 of 58 (94.8%) in a median 1.3 s;
folded + vote read 28.

## Decision

- **`locate` has a `method` and a `compression`, and both are enums**:
  `method` is `copy` or `vote`, `compression` is `template_fold` or `none`.
  All four combinations are served. A new way to read or to compress is a
  new value with its own measurement; an unknown value is refused naming the
  accepted ones, never read as the default. `compression` is refused on
  every other primitive; `method` keeps each primitive's own values.
- **The defaults are `copy` and `template_fold`**, the measured-better route
  on real logs. The answer names the method and the compression that
  produced it, as a `point` names its method.
- **The vote is kept**, unchanged under `none` and read at each level under
  `template_fold`, on the loads calibrated for it. **`LOCATE_MAX_KEYS`
  limits the text the vote reads** — the target, or each level's text — and
  stays 4,554 keys on the served 27B. A `copy` is bounded by the context
  alone and is served on every load.
- **A copy is a constrained decode whose permitted sets follow its draws.**
  The candidates' rendered text, tokenized alone, forms a prefix tree; each
  draw is restricted to the children of the node reached, and the run stops
  when the written prefix belongs to one text — which is the answer, with no
  string matched. Where several segments share that text, the first is the
  answer.
- **The leaf reports the draw a call makes, in that call** (an appended
  output on the prefill and decode options, ADR 0016). What a round commits
  is unchanged — the previous call's draw — so ADR 0034's one-round lag
  stays; the host learns the draw one call earlier, which is what a set
  that depends on it needs.
- **A copy's permitted set may be as wide as the vocabulary**: a per-lane
  vocabulary bitmask beside the 32-id row, honoured at any width and never
  refused or truncated for it, its staging reserved at load (ADR 0030). The
  32-id path is unchanged bit for bit, and a call with no copy lane
  allocates and launches nothing new.
- **`template_fold` is the first compression**: `tools/locate-sets/compress.py`'s
  fold at the settings R2 judged, ported as a pure host function and held to
  golden cases the reference writes. It is reversible through its map
  (template, row) → original segments, runs before any prefill, once per
  target per request, and changes nothing in the caller's state: each level
  is asked over the state with its target replaced by that level's text.

## Considered options

**Booleans** (`fold: true`, `copy: true`). Two flags whose four combinations
are an accident of their spelling, and a third method or compression would
be a third flag contradicting them. Rejected.

**Keep the vote as the default and lift its ceiling.** Real logs put its
ceiling where set D did (spec 20), and folding does not rescue it: folded +
vote read 28 of R2's 58.

**Generate freely and match the quote to a line** — the route R2 measured.
It works (86.2% on the whole log, 94.8% folded), but the match is a parser
of free text (exact, contained, then word overlap), and the model writes the
whole line (median 90 tokens) where 26 identify it. The constrained copy is
the same mechanism without the parser and stops at the prefix that decides.

**Score every candidate by teacher forcing.** Exact, but one forced pass per
candidate — thousands of lines, or hundreds of templates.

**A readout at each branch of the tree**: one decision per branch, each a
request that claims the previous one's state. No leaf change, but a
retained-state claim per branch, where a decode round per token costs a few
milliseconds.

**Walk the tree on the device.** The lag would not matter, but the tree, its
walk and its stop rule would move into the kernel. Rejected while the host
can compute the next set from a reported draw.

**Raise the 32-id cap instead of a mask.** A copy's root can hold hundreds
of distinct first tokens, the mask kernel's linear scan grows with the set,
and any cap is one a real question can reach and be refused at.

## Consequences

- A caller who sent a `locate` with no `method` got the vote and now gets a
  copy over a folded state; the answer's `method` and `compression` say so,
  and the vote remains one field away.
- A copy holds a decode lane for its rounds (a median of tens), which a vote
  never did; a folded question is two internal requests in sequence.
- A load with no `locate` calibration now answers the default `locate`, and
  refuses only `vote`.
- A `template_fold` prompt holds no state, so it shares no prefix with the
  other primitives over that state; `template_fold` questions over one
  target share their level-1 prompt.
- Folding removes the order of lines and every time from level 1: a question
  that needs context across lines, or names a line by its time alone, is
  better asked with `none`. Lines identical but for their time answer with
  the first of them.
- `ignis_locates_total{method, compression}` joins the metrics contract
  (ADR 0017); `ignis_decisions_total{type="locate"}` still counts one per
  question.
- `CONTEXT.md`'s *Permitted set* ("at most 32") and *Draw* change with this
  ADR.
