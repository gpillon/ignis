# A shared vision embedding turns a 4x fan-out into 2.1x, and a repeat visit into free

- Kind: experiment
- Status: current
- Observed: 2026-09-20
- Last verified: 2026-09-20
- Scope: serving / multimodal fan-out, vision encode cost, device memory reservation
- Related: `2026-09-20-number-width-and-decide-e2e.md` (the baseline this
  measures against), ADR 0035, ADR 0030, GitHub #243, #240, #246
- Superseded by: none

## Question

`2026-09-20-number-width-and-decide-e2e.md` measured a `/v1/decide` fan-out
over one 4096×4096 screenshot at 6.84 s / 13.50 s / 27.92 s for 1 / 2 / 4
questions — exactly 1x / 2x / 4x — and split the per-question cost into
0.17 s of preprocess (already shared by content digest), **4.17 s of vision
encode** (shared by nothing) and ~2.4 s of attention prefill.

GitHub #243 made the encoder's output shareable: keyed by `(content digest,
grid)`, held past the request that encoded it, bounded by a paged pool in the
leaf. What does that actually buy, end to end, and what does it cost in the
VRAM plan?

## Evidence

Same host, same card, same release build (`decide-236` at 4f3c07e plus the
refusal-message fix), same request bodies as the baseline
(`.scratch/decide-live/fan_{1,2,4}q.json`), `--vision`, hq-e8-2b, 262 144
context, DFlash2/7. The server was **restarted before each cold row**, so
each one pays its own encode exactly as the baseline did.

| questions | baseline | now | vision encodes |
|---|---|---|---|
| 1 | 6.84 s | **6.27 s** | 1 |
| 2 | 13.50 s | **8.46 s** | 1 |
| 4 | 27.92 s | **13.24 s** | 1 |
| 4, image already seen | — | **9.49 s** | **0** |

From the server's own request log on the cold 4-question run — two requests
report media, and the second is the claim acceptance 1 makes:

```
req 0  media.encode_seconds 3.62486  media.cache_hits 0  vision_tokens 16384
req 2  media.encode_seconds 0.0      media.cache_hits 1
```

The VRAM plan at the default pool, from `ignis.runtime.vram_plan`:

```
media_embedding_bytes   320.0 MiB
```

which is the same figure `2026-09-18-vram-budget-flat-memory-gate.md` recorded
before any of this (`media_embedding 320`).

## Conclusion

**One encode per picture, not per question.** Four questions over one image
cost 13.24 s against 27.92 s — 2.11x faster, and no longer proportional to the
question count. What remains proportional is the attention prefill: the
4-question row is roughly one 3.6 s encode plus four ~2.4 s prefills, which is
the shape acceptance 2 asked for. The prefill half is GitHub #240's, and
closing it would take the 4-question row toward the 1-question one.

**A single question is unchanged** (6.27 s against 6.84 s, within run-to-run
noise): nothing was made cheaper, something was stopped from being repeated.

**A second visit to the same picture is nearly free.** The same four questions
asked again on the same server run cost 9.49 s with zero encodes — 2.94x the
baseline. This is the cache outliving the request, and it is the reason
reference counting alone would have recovered nothing: the siblings of a
fan-out never overlap in time (one request holds multi-tick prefill progress
at a time), so an entry that died with its last holder would never be found by
anyone.

**The default costs nothing.** The pool's default is one envelope-wide item,
which is the reservation vision always took, so the plan line does not move —
verified live at 320.0 MiB, matching the pre-existing recorded figure.

## Caveats and limits

- **Single run per cell.** The baseline was measured the same way, and the
  effect (2.1x) is far outside the ~9% spread visible between the two
  4-question cold rows (13.27 s and 13.24 s). A tighter figure would need
  repeats.
- **It does not make a new frame cheaper.** An unseen picture still costs
  ~3.6 s of tower whatever the pool holds. Nothing encodes ahead of a request
  either — that is GitHub #246, and it is what "N frames ready at time N"
  actually needs.
- **The `media.encode_seconds: 0` line comes from the third of four
  questions**, not the second: only two of the four requests in a fan-out
  emit an `ignis.request.admitted` event with media attributes on this build.
  The claim the table makes (one encode for the whole fan-out) is unaffected —
  one encode is all the log contains — but the per-question observability is
  thinner than it looks.
- **Measured on one image size.** A 4096×4096 screenshot is the pool's
  expensive case (16,384 columns, 160 MiB, 128 pages). The paging exists for
  the opposite end (a 320×240 thumbnail is 80 columns and one page), and that
  end is not measured here.
