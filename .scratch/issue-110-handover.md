# Issue #110 handover

## Scope and repository state

- Repository: `gpillon/ignis`.
- Worktree: `F:\ai\opencode\.inference-qwen-worktrees\issue-110`.
- Branch: `issue-110-itl-measurement`.
- Starting commit: `bfe55270785a2fd79cd71996bd002d1acb513bc9`.
- `.scratch/experiment-issue-110.md` contains a rejected comment diff and was
  deliberately not treated as a solution.

## Finding

The original live ITL fixture could not produce a verdict against the NInfer
reference. Four decode lanes capped at 512 tokens ended after roughly 15-19 s,
while ten sequential 32,768-token prefillers occupied roughly 62-73 s. The
stricter window validation added for #110 correctly exposed the mismatch; the
older pooling behavior had hidden it.

Raising the cap alone is insufficient because lanes can emit EOS before the
last prefiller. A live reproduction with a 3,072-token cap still ended the four
reference lanes at 350, 545, 1,035, and 1,195 generated tokens with
`finish=stop_token`.

## Implemented design

- The decode cap is a 3,072-token safety reservation. Four lanes plus one
  prefiller reserve at most 61,504 tokens, below the specified 65,536-token
  pool.
- Decode streams run only until the final prefiller window ends, then the bench
  cancels them instead of requiring them to exhaust the safety cap. On the
  reference leg that intent was not achieved: see "How each lane actually
  ended" in `.scratch/g3-gate-110/README.md`.
- Dropping an Ignis HTTP stream propagates cancellation into the engine and
  scheduler, releasing its slot and KV reservation.
- Measurement lanes set Ignis `ignore_eos`; the normal serving default remains
  `false`.
- The bench sends all artifact EOS ids as `logit_bias` exclusions to compatible
  reference endpoints.
- As a further guard, decode prompts end with: `Produce at least 3072 tokens. Do
  not stop, conclude, or emit EOS earlier.` Corpus landing shortens the source
  window so the post-template prompt remains exactly 4,096 tokens.

## Tests added

- Dropping a streaming HTTP response cancels the request and releases a
  `max_in_flight=1` slot.
- Runtime measurement lanes can continue past EOS when `ignore_eos` is set.
- G3 decode lanes are stopped after the final prefiller window.
- Both endpoint request shapes carry the measurement-only EOS guard.
- Corpus prompts with the shared instruction suffix still land at their exact
  claimed post-template length and retain first-content-token divergence.

Validation completed:

- `cargo test --workspace`: pass.
- `cargo build -p ignis-bench --release`: pass.
- `cargo build -p ignis-server --release --features cuda`: pass.

The repository is not `rustfmt`-clean under the installed stable rustfmt and
carries no `rustfmt.toml`, so this branch does **not** reformat anything: the
diff was rebuilt to contain semantic changes only, matching each file's
existing wrapping style. `git diff --check` is clean.

## Live gate

Canonical profile:

- Artifact: `F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer`.
- Corpus: `F:\ai\q38\ninfer\bench\fixtures\bench_corpus.ids`.
- KV: `hq-e8-2b`; prefill chunk: 1,024; max context: 40,960.
- Reference KV capacity: auto, resolved to 327,680 tokens.
- Reference max concurrency: 8; greedy; thinking disabled.
- Final live/live session: `g3-110-window-cancel-v3`.

Both records are cold and complete:

| cell | Ignis | reference | ratio | result |
|---|---:|---:|---:|---|
| C=1 | 68.3 tok/s | 62.0 tok/s | 1.101 | PASS |
| C=4 | 19.7 tok/s | 13.0 tok/s | 1.514 | PASS |
| ITL p95 | 222.46 ms | 201.11 ms | 1.106 | **FAIL** |

The ITL fixture itself is now valid: the reference produced 1,845 pooled
intervals and Ignis produced 1,352, with ten cold prefillers and four complete
decode lanes on both sides. The formal verdict is still FAIL because 1.106 is
above ADR 0015's tolerated-with-warning ceiling of 1.10. Do not round this down
or claim #110 is resolved.

1.106 is **not** an improvement on the previously recorded 1.130. The two
numbers measure different things -- different cap, EOS suppression,
window-restricted pooling, cancellation -- and the harness itself refuses to
compare records across that change. 1.130 is void; 1.106 is the first valid
measurement, and it happens to fail too.

Artifacts:

- `.scratch/g3-gate-110/reference-final.json`.
- `.scratch/g3-gate-110/ignis-final.json`.
- `.scratch/g3-gate-110/verdict-final.json`.
- Server logs remain in the same directory (ignored by Git).

Formal comparison command:

```powershell
.\target\x86_64-pc-windows-msvc\release\ignis-bench.exe g3-gate `
  --ours .scratch\g3-gate-110\ignis-final.json `
  --ref .scratch\g3-gate-110\reference-final.json `
  --out .scratch\g3-gate-110\verdict-final.json
```

Each live record used `ignis-bench g3`, the artifact and corpus above,
`--session g3-110-window-cancel-v3`, and its respective endpoint (`:8000` for
Ignis, `:8080` for NInfer). Never compare records with different session ids.

The first reference restart failed during CUDA Graph preparation (114,163,712
bytes consumed versus a 100,663,296-byte allowance). An identical retry
started successfully; its startup log reported `graphs=0.00 MiB/96.00 MiB`.
This did not invalidate coldness or completeness, but should be retained as a
profile caveat if the narrow 1.106 miss is reproduced.

## Prior evidence

- `.scratch/g3-gate-110/reference.json`: first 3,072-cap attempt, only one EOS
  id suppressed; no ITL verdict.
- `.scratch/g3-gate-110/reference-2.json`: all artifact EOS ids sent as
  `logit_bias`; no ITL verdict because NInfer still returned `finish=stop_token`.
- Corresponding `reference*.stderr.log` files contain per-request token counts
  and finish reasons.

## Root cause of the 1.106 ITL p95

Separating the pooled intervals into those blocked behind a prefill chunk
(>= 40 ms) and those that are a decode round alone (< 40 ms):

| | Ignis | reference |
|---|---:|---:|
| blocked intervals, mean | 180.8 ms | 155.4 ms |
| intervals under 40 ms | 0 of 1,352 | 530 of 1,845 |
| decode rounds per lane per window | 33.80 | 46.12 |
| prefill throughput per prefiller | 5,375 tok/s | 6,164 tok/s |
| time per 1,024-token chunk | 190.5 ms | 166.1 ms |

During a prefill window the ITL floor is one chunk plus one decode round, so
the p95 is a second measurement of prefill throughput. The blocked-interval
ratio is 1.163 and the p95 ratio is 1.106. Restricting the reference to its
blocked intervals alone moves its p95 from 201.1 ms to 204.0 ms, so its cheap
decode rounds are not what makes it win the p95.

**The p95 gap is prefill throughput.** That much the records show. The
leading explanation is that the two engines are not running the same KV
precision -- Ignis is `BF16 KV`, the reference `hq-e8-2b KV` -- since every
chunk's attention rereads the whole prior KV at 32,768 tokens, so Ignis moves
roughly twice the bytes on a bandwidth-bound operation. ADR 0015 chose this
inequality deliberately and records it next to the verdict rather than
correcting for it, and the v1 design schedules hq-e8-2b for phase 4 (G4).

Treat the KV explanation as a hypothesis. Nothing here isolates the KV format
from the rest of what differs between the two engines, and ADR 0015 forecloses
the matched-KV control that would test it. It becomes testable when Ignis has
hq-e8-2b of its own.

**The p50 gap is a different, real defect.** `ConcreteScheduler::advance`
(`crates/core/src/concrete.rs:977`) runs exactly one prefill chunk and then
exactly one batched decode round, so a decode lane can never emit two tokens
between chunk boundaries. That is why Ignis has no interval under 40 ms at
all while the reference has 28.7%, and why the lanes generated 3,612 tokens
against the reference's 12,072 in the same window.

The earlier claim that Ignis decodes its resident lanes sequentially is
wrong. `crates/core/src/step.rs:555` batches every lane into one
`ignis_program_decode` call, with a CUDA graph captured per exact batch width
1..=8.

## Remaining work

1. Review the scoped diff, commit, and push the branch.
2. Add the evidence-backed result to GitHub issue #110 without closing it.
3. If another run is requested, reproduce both live/live legs under one fresh
   session; do not reuse one side of `g3-110-window-cancel-v3` with a new run.
4. Propose resolving #110 as deferred, with the two halves split by metric:
   the p95 to G4's hq-e8-2b KV, the p50 and decode throughput to the phase-6
   overlap work. Both need their own issue.

   Note this is a **substitution** of #110's own acceptance criterion, which
   offers only "deferred to the phase-6 overlap work" as the escape hatch.
   The evidence points the p95 half at phase 4 instead. That reassignment is
   the repo owner's call, not the agent's, and #110 stays open until made.

5. Raise the ITL decode cap for the next run, or record each lane's finish
   reason in the record so a cap-terminated leg is visible in the data rather
   than only by inspecting token counts.
