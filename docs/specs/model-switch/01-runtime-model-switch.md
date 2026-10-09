# 01 — runtime model switch between the 27B and Flash-Next (full reload, no VRAM for the switch)

GitHub: #305 (master #298). Supersedes the notes in
`docs/specs/flash-next/phase2-model-switch-notes.md` (owner, 2026-10-04),
which stays as the historical record of the rules this spec locks in.

## Problem Statement

Ignis is wired to exactly one model for the whole life of the process.
`Engine.model_id` is documented as "immutable for the server's life, so it is
captured once here instead of crossing the command channel"
(`crates/server/src/engine.rs:143`), `Server::new` (`crates/server/src/lib.rs:160`)
captures `engine`/`template` together with no setter that ever replaces
either, and `GET /v1/models`' own doc says it plainly: "A server serves one
model" (`crates/server/src/api.rs:1314`). `main.rs` never joins the model
thread — the process exits with it still running — because nothing in
production ever needs to.

The owner's workflow needs a second mode: ignis serves the 27B most of the
time, and when a task needs more (longer context, a harder problem) it
switches to Qwen3.8-Flash-Next and back, without restarting the process or
losing the other requests already in flight.

The rules for *how* a switch behaves were already decided
(`phase2-model-switch-notes.md`, 2026-10-04) and are restated as fact here,
not re-opened:

- A switch is a **full reload** in both directions. Everything belonging to
  the old model — VRAM, pinned host memory, KV-RAM, retained slots — is
  dropped, and the new model gets the whole machine.
- The two models are **never resident together**.
- **Never VRAM for the switch.** A little host RAM is fine only if it buys a
  lot of speed.

What was not decided — the trigger, in-flight handling, client visibility —
is decided below (§Implementation Decisions), closing the "questions for when
the phase opens" the notes left open.

### Correcting the 2026-10-04 blocker survey

The notes listed three blockers. Re-checked against the current code, only
one is real:

1. **"The pinned host pool refuses a second create until the first is
   destroyed."** True, but it is not a bug to fix — `HostPinnedPool` is
   already RAII (`crates/core/src/seq.rs:1485-1529`): `Drop` calls
   `ignis_host_pinned_pool_destroy` unless the tier was disabled, and the
   guard is already owned by `CudaLeaf` (`crates/runtime/src/cuda_leaf.rs:161,197`),
   which the model thread drops when its `Scheduler` is dropped. The global
   is a **sequencing contract** ("destroy before the next create"), and the
   code that satisfies it already exists — it has just never run in
   production, because production never tears the model down. No C++/CUDA
   change is needed; the fix is making the server actually run that existing
   teardown chain before the next load's create call, which is this spec's
   job.
2. **"The attention tap holds a global sized by the GQA layer count."**
   `attn_tap.rs`'s own doc says it: "Gated behind the non-default `attn-tap`
   feature... no production build turns it on" (`crates/core/src/attn_tap.rs:12-14`,
   confirmed off by default in `crates/core/Cargo.toml:52` /
   `crates/server/Cargo.toml:63`). It cannot block a production switch and is
   out of scope.
3. **"`Server::new` captures the engine and template together;
   `Engine.model_id` is immutable."** This one is real and is what
   §Implementation Decisions fixes.

So this spec makes **no kernel/CUDA changes at all** — it is an ownership and
orchestration change confined to `crates/server`.

## Solution

Add one explicit trigger (`POST /v1/models/switch`) that tears the current
model down and loads a new one, in order, on the same process, reusing the
model thread's already-documented clean-shutdown property
(`crates/server/src/engine.rs:191-201`, server/05's prerequisite for exactly
this). While a switch is in flight, new `/v1` requests are refused with `503
model_switching` (the same gate shape as today's `503 server_not_ready`
warm-up gate) and `GET /v1/models` reports the transition; requests already
admitted on the old model get a bounded grace window to finish before being
cancelled, so the switch has a finite upper bound regardless of load. A
failed load leaves the previous model serving — a switch never leaves the
server in a half-torn-down state.

## User Stories

1. As the owner, I want to call one endpoint naming the artifact and model id
   I want served, so that I can move from the 27B to Flash-Next (or back)
   without restarting ignis.
2. As the owner, I want the switch to be a full reload — the old model's
   VRAM, pinned host memory, KV-RAM and retained slots all freed before the
   new model allocates anything — so that the active model always gets the
   whole machine, exactly as decided in `phase2-model-switch-notes.md`.
3. As the owner, I want the two models never resident together, so that a
   switch never costs more VRAM than either model needs on its own.
4. As the owner, I want the switch machinery itself to spend no VRAM, so that
   occasionally using Flash-Next never taxes the 27B's resources or vice
   versa.
5. As an API client, I want a request sent while a switch is in progress
   refused with `503 model_switching` and a `Retry-After` header, so that I
   back off instead of hanging on a model that is being replaced.
6. As an API client, I want `GET /v1/models` to show a `status` of
   `"serving"`, `"switching"`, or `"failed"` beside the model entry, so that
   a client (the Playground included) can show the transition instead of
   guessing from refused requests.
7. As an API client with a request already admitted on the old model when a
   switch starts, I want a bounded grace window to finish before I am
   cancelled, so that a switch does not silently drop work that was almost
   done.
8. As the owner, I want a switch that cannot finish its grace window (a
   request that will not finish) to still complete — the lingering request
   cancelled with a clear error — so that one slow client can never block a
   switch indefinitely.
9. As the owner, I want a switch whose new load fails (bad artifact path,
   out-of-memory, a kernel load error) to leave the previous model serving,
   with `GET /v1/models` reporting `"failed"` only transiently before falling
   back to `"serving"` the old model, so that a bad switch request degrades
   to "nothing happened" rather than "the server is down."
10. As a maintainer, I want the per-model state (`Engine`, the template
    provider, `ModelFamily`, pointing/locate calibration, the decide
    alphabet, the vision media acquirer) grouped into one bundle that the
    switch replaces atomically, so that there is one seam to swap instead of
    N fields each needing their own synchronization.
11. As a maintainer, I want reads of the active bundle to be wait-free
    (`ArcSwap`, matching the pattern already used for the telemetry counters
    snapshot, `crates/server/src/engine.rs:32,154`), so that ordinary request
    handling never takes a lock the switch orchestrator also touches.
12. As a maintainer, I want the drain step to reuse the engine's existing
    wait-free `waiting`/`running` counters snapshot rather than inventing a
    second in-flight tracker, so that the switch and `/metrics` agree on what
    "in flight" means.
13. As a maintainer, I want the model thread's `JoinHandle` (today discarded
    in production — `Engine::with_clock` throws it away, only
    `with_clock_and_driver` keeps it, and only test harnesses call that) kept
    reachable from `Server`, so that a switch can actually join the old
    thread instead of leaking it.
14. As a maintainer, I want this change to touch no vendored kernel file and
    no CUDA source, so that it carries none of ADR 0010/0037's verbatim-port
    obligations — confirmed in §Problem Statement, the real blocker is pure
    Rust wiring.
15. As a maintainer, I want the switch mechanism proven on `MockCompute` with
    two distinct mock model ids and no GPU, so that `cargo test` covers the
    orchestration (gating, draining, teardown ordering, failure fallback)
    without needing the card.
16. As a release engineer, I want one GPU-profile test (`#[ignore]`d,
    `IGNIS_GPU_PROFILE=1`) that switches the real 27B and Flash-Next
    artifacts back and forth and records the wall time of each direction, so
    that `phase2-model-switch-notes.md`'s unmeasured 14-15s/8-10s estimates
    are replaced with a measured number the way spec server/05 and the
    expert-pool-read finding replaced estimates elsewhere.
17. As a client (any OpenAI-compatible one, e.g. opencode), I want naming a
    different, known model in my request's `model` field to switch the
    server to it and then answer my request on it, so that I do not need my
    own retry loop around `503` just to use the model I asked for.
18. As the owner, I want implicit switching gated by one flag
    (`--allow-model-switch`, default on) and a named list of switchable
    models and their artifacts (`--known-model <id>=<path>`), so that a
    server I did not mean to make switchable cannot be moved by an
    arbitrary `model` string, and so a model with no known artifact cannot
    be requested into existence.
19. As the owner, I want a request naming an unknown model, or naming a
    different model while implicit switching is off, refused by name
    (`404 model_not_found`) rather than silently served by whatever is
    active, so that a typo'd or stale `model` field is never mistaken for
    success.
20. As the owner, I want an implicit switch to close the gate to every
    request — old model or new, admitted or not — the instant the mismatch
    is seen, so that "exhaust the queue, then accept nothing else" holds for
    an implicit switch exactly as it does for an explicit one.
21. As a client whose request triggered an implicit switch, I want my own
    request held until the switch finishes and then served on the new
    model, so that I get an answer rather than a `202` I have to poll for a
    switch I did not explicitly ask to track.

## Implementation Decisions

### The `ActiveModel` bundle

A new type groups every field on `Server` that is a property of the loaded
artifact rather than of the server process: `engine: Engine`,
`template: Arc<dyn TemplateProvider>`, `family: ModelFamily`,
`calibration: Option<Calibration>`, `locate: Option<LocateCalibration>`,
`alphabet: Arc<AnswerAlphabet>`, `media: Option<Arc<MediaAcquirer>>`, and the
model thread's `driver: std::thread::JoinHandle<()>` (today returned by
`Engine::with_clock_and_driver` and discarded by every production call site —
production switches to calling that constructor and keeping the handle).

Everything else already on `Server` — `request_timeout`,
`default_enable_thinking`/`default_reasoning_effort`/`default_thinking_budget`,
`playground`, `metrics`, `api_key`, `instruction_policy`, `fork_history`,
`next_fan_out`, `responses`, `wall_clock`, `seedless_seed` — is server-level
configuration or shared infrastructure, not model-derived, and stays exactly
where it is. `fork_history` is the one exception that needs an action, not a
move: it names recent `/v1/decide` parts states by match key against the old
model's KV, so a switch clears it (a fresh `ForkHistory::default()`) rather
than carrying stale keys into a model that cannot resolve them.

`Server.active: Arc<arc_swap::ArcSwap<ActiveModel>>` replaces those fields.
Every existing read site (`server.engine`, `server.template`, `server.family`,
`server.calibration`, `server.locate`, `server.alphabet`, `server.media`) in
`api.rs`, `responses/`, `decide.rs`, `locate.rs`, `media.rs` becomes
`server.active.load().engine` etc. — a mechanical rename, not a logic change,
since `ArcSwap::load()` is wait-free and the returned guard derefs like the
value it replaces.

### `ModelStatus`: generalizing `ready`

`Server.ready: Arc<AtomicBool>` (`crates/server/src/lib.rs:144`) already does
exactly the gate this spec needs, one state too few. It becomes
`Server.status: Arc<ArcSwap<ModelStatus>>` with

```rust
enum ModelStatus {
    WarmingUp,                  // today's `ready == false`
    Serving,                    // today's `ready == true`
    Switching { from: String, to: String },
    Failed { reason: String },  // transient: one tick, then back to Serving
}
```

`require_ready` (`crates/server/src/api.rs:238`) becomes `require_serving`: it
passes `Serving`, and refuses every other state — `503 server_not_ready` for
`WarmingUp` (unchanged wire contract), `503 model_switching` for `Switching`
(new), `503 server_not_ready` for `Failed` (the old model is what is actually
being served again by the time a client could see this, so no new error code
is needed). The `OPTIONS` passthrough and the `Retry-After` header stay as
they are today.

### The trigger: `POST /v1/models/switch`

```
POST /v1/models/switch
{ "artifact": "<path>", "model": "<served id>" }
→ 202 { "status": "switching", "from": "<old id>", "to": "<new id>" }
```

Mirrors the CLI's own `--artifact`/`--model` knobs (`config.rs`), so a switch
asks for exactly what a restart with different flags would have asked for.
Returns immediately; the switch itself runs on a spawned task
(`model_switch::switch`, new module) so the HTTP response is never the thing
waiting on a model load. A second switch request while one is already running
is refused `409` naming the in-progress switch — this spec serializes
switches, it does not queue them.

### Draining, then teardown, then load — in order

1. Set `status = Switching { from, to }`. `require_serving` now refuses new
   `/v1` traffic with `503 model_switching`.
2. **Drain**: poll `server.active.load().engine`'s existing wait-free
   counters snapshot (`crates/server/src/engine.rs:154`, the same one
   `/metrics` reads) until `waiting == 0 && running == 0`, or a bounded
   timeout (`--switch-drain-timeout`, default 30s) elapses. This is the only
   wait in the whole switch that depends on other requests; everything else
   is this task's own work.
3. **Teardown**: swap `server.active` to a placeholder/empty value so no new
   reader can reach the old `Engine`, take the old `ActiveModel` out (its
   `Arc` strong count is now 1 — every handler that held an `ArcSwap::load()`
   guard across the drain window already finished, by construction of step
   2), drop its `Engine` (disconnects the command channel) and join its
   `driver: JoinHandle<()>` — the exact recipe `crates/server/tests/support/live_server.rs:45-52`
   already uses, lifted into a reachable production path. Joining blocks, so
   this step runs inside `spawn_blocking`.
4. **Load**: run the same artifact-loading path `main.rs` runs at startup
   (`loader::load_artifact` → `artifact_family` → `cuda_scheduler` /
   `flash_next_scheduler_with_ngram_cache` → `Engine::with_clock_and_driver`)
   against the new artifact, build the new `ActiveModel`, including a fresh
   `ForkHistory` and a fresh warm-up traversal (reusing `Server::with_warm_up`'s
   existing logic).
5. *(Superseded by §Implementation notes: teardown always precedes the
   load; a GPU-side failure reloads the old model.)*
   On success: `server.active.store(new)`, `status = Serving`.
   On failure at any point in step 4: the **old** `ActiveModel` was already
   torn down in step 3, so there is nothing to fall back to in-process —
   this spec's failure contract is therefore: step 3 and step 4 run inside
   one task, and the old model's teardown in step 3 is **not** committed
   (the old `Engine`/`driver` are held, not dropped) until step 4's load
   reports success. Only then does step 3's actual drop/join run, followed
   immediately by `server.active.store(new)`. A failed step 4 leaves the old
   `ActiveModel` untouched, `server.active` unchanged, `status = Failed {
   reason }` for one tick and then `Serving` again (AC 9). This reordering —
   build the new load fully before releasing the old one's resources — costs
   one instant of (old VRAM + new VRAM) only on the failure path's probe, not
   on a successful switch, because the new load's own allocation already
   fails fast (ADR 0030: "nothing fits" refuses before any GPU allocation
   happens) in the common failure case (bad path, bad checksum, VRAM budget
   that cannot fit) — only a failure *after* the new model's device
   allocations already landed (a kernel load error, not a budget refusal)
   genuinely holds both for an instant. That residual risk is accepted and
   named here rather than engineered away, because the alternative (tear
   down old before even attempting new) turns every failed switch into a
   stopped server, which AC 9 explicitly rules out.
6. **Cancel stragglers**: if step 2's drain timed out, the requests still
   running on the old engine when step 3 begins are cancelled (their stream
   ends with a `model_switching` error chunk) rather than waited on forever —
   `Engine`'s existing per-request channel teardown-on-drop already produces
   this when the command channel disconnects; no new cancellation code is
   needed, only not waiting past the deadline.

### What does not change

- The `Scheduler` trait, `crates/core`'s scheduling behavior, the per-request
  `SchedEvent` → `ChunkStream` delivery mechanism — none of this is touched.
- No kernel/CUDA file changes (§Problem Statement).
- The warm-up traversal's own logic (`Server::with_warm_up`) — reused as-is
  for the new model's post-load warm-up.
- `ignis_host_pinned_pool_create`/`destroy` and every other RAII guard in
  `crates/core`/`crates/runtime` — unchanged; this spec is what finally
  exercises their existing `Drop` ordering in production.

## Testing Decisions

Per ADR 0006, no test depends on wall-clock timing or the GPU to prove the
orchestration; a GPU-profile test separately proves the real numbers.

### The orchestration seam — a new `model_switch` test module

Two `MockCompute`-backed engines with distinct model ids (`"mock-a"`,
`"mock-b"`), driven directly (not through HTTP): start a switch, assert
`status` reads `Switching` immediately and `require_serving`'s gate (tested at
the function level) refuses with `model_switching`; use the same
synchronization-primitive pattern as spec server/05 to hold one request
mid-decode on `"mock-a"` and prove the switch's drain step actually waits for
it, then completes once it is released. A second case holds a request past
`--switch-drain-timeout` (test-injected, short) and asserts it is cancelled
and the switch still completes. A third drives a load that is made to fail
(an artifact path that does not exist) and asserts `server.active` and
`status` end back at `Serving` on the original model — AC 9's test.

### The HTTP seam — `crates/server/tests/model_switch_http.rs`

Drives `POST /v1/models/switch` against the mock-backed router: `202` with
the envelope, `GET /v1/models` reporting `switching` then `serving` on the
new id, a concurrent `/v1/chat/completions` sent mid-switch getting `503
model_switching` with `Retry-After`, and a second `POST /v1/models/switch`
sent while one is already running getting `409`.

### GPU end-to-end (new, `#[ignore]`d)

`crates/server/tests/model_switch_gpu.rs`: against the real 27B and
Flash-Next artifacts, switch 27B → Flash-Next → 27B through the live HTTP
surface (`live_server.rs`'s pattern, extended with a `switch_to` helper), and
record each direction's wall time (request to `202` → `GET /v1/models`
reporting `serving` on the new id) as the measured replacement for
`phase2-model-switch-notes.md`'s estimates. Also asserts, via
`crates/core::seq::host_pool_stats()`, that the host pinned pool's capacity
reported immediately after the switch matches the *new* model's own
`--kv-host-pool-bytes` plan line, not the old model's — the direct proof that
teardown actually ran before load, not just that the HTTP surface looks
right. This is a measurement run, not a gate: it prints its numbers rather
than asserting a threshold, exactly like the G2/G4 instruments.

### The gate

`cargo test` stays green, CPU-only, deterministic, workspace-wide; the GPU
case runs under `scripts/gpu-profile.ps1` only, on the owner's own schedule —
it needs both artifacts present and the card exclusively, so this spec does
not run it itself.

## Out of Scope

- **A router that picks the model per task.** Nothing here infers which
  model a task needs. §Implicit switch (below, added 2026-10-10) lets a
  client *name* the model it wants and have the server get there; a policy
  that chooses the model *for* the client is a separate, later decision
  (`phase2-model-switch-notes.md`'s first open question, still open).
- **The KV-disk tier (ADR 0045 Tier 2) surviving a switch.** Interesting
  (phase2 notes), unmeasured, and a separate ticket — this spec's full
  reload drops retained state on both sides, full stop.
- **Progressive / layer-by-layer load during a switch.** The phase2 notes'
  follow-up idea, and genuinely compatible with this spec's `model_switch::switch`
  seam later, but today's `fill_expert_pool` parallel-read path
  (GitHub #306, 2026-10-09) already halved Flash-Next's load time without
  it; adding it here would be optimizing a mechanism before it has shipped.
- **Any CUDA/kernel change.** None is needed (§Problem Statement).
- **Removing the attention tap global.** Not a production concern; out of
  scope by construction (non-default feature).
- **Queuing multiple switch requests.** A switch in progress refuses a second
  one; a queue is unneeded complexity for "2-3 agents, occasional switch."

## Further Notes

The phase2 notes' load-time estimates (27B→FN ~14-15s, FN→27B ~8-10s) predate
the 2026-10-09 expert-pool parallel-read change, which took Flash-Next's own
start-to-ready from ~46.4s to ~31.9s on its own
(`docs/findings/2026-10-09-flash-next-expert-pool-parallel-read.md`) — closer
to this spec's own switch overhead (drain + join + teardown, expected low
single-digit seconds) than to the dominant cost, which remains the load
itself. The GPU-profile test above replaces the estimate with a real number
once this lands; nothing in this spec should be tuned against the old
estimate.

The failure-ordering decision in §Draining (hold the old model until the new
one's load reports success, rather than tearing down first) is the one place
this spec knowingly trades a small, failure-path-only departure from "never
resident together" for "a switch can never stop the server." Worth flagging
to the owner explicitly if a future spec tightens the VRAM budget to the
point where even that momentary double allocation cannot fit — at that point
"tear down first, accept that a failed switch is a cold-start" becomes the
only option, and AC 9 would need revisiting.

## Implementation notes (2026-10-10, branch `model-switch-305`)

Where the implementation departs from the sections above, and why.

- **Teardown before load, always.** §Draining step 5's "load the new model
  before releasing the old" cannot work on the card even on a *successful*
  switch, in either direction: every load sizes its VRAM plan from the
  memory NVML reports free at its start (`crates/server/src/runtime.rs`),
  which the old model still holds — the new plan would refuse or be built
  around the old weights, and the two would be resident together, which
  this spec rules out. On top of that the 27B pins its KV-RAM arena as a
  process-wide singleton whose create refuses while one exists
  (`kernel/src/seq.cu`, `ignis_host_pinned_pool_create`: "destroy it before
  creating another"); Flash-Next's arena is its own instance's
  (`HostArena`, spec flash-next/05's no-singleton rule), so the singleton
  alone blocks only a 27B load over a 27B. So the order is steps 1-4 as
  written (gate, drain, teardown, load), and
  AC 9 is kept another way: everything that can refuse a target without the
  GPU — the path, the sidecar, the checksum, the artifact's model against
  the start flags, the thinking defaults, the vision processor — runs
  *before* the drain, while the old model still serves (a refusal there
  touches nothing); a load that fails on the GPU after the teardown reloads
  the old model from its artifact (`ActiveModel::source`). Only a reload that
  fails too leaves `failed` standing, with the process up and a further
  switch accepted. The Further Notes' "tear down first" option, taken.
- **Stragglers are cut by an explicit shutdown, and end with
  `engine_error`.** Step 6's "the command channel disconnects" never happens
  while a streaming response's `CancelOnDrop` holds an `Engine` clone, so the
  engine gained a `Shutdown` command: each request still on the old model is
  sent a `Done` with `FinishReason::Error`, which every handler already
  reports as the `engine_error` chunk — not a new `model_switching` chunk.
  A request that reached its handler before the gate closed and submits
  after the teardown is answered `503 engine_full` ("retry").
- **The drain's counts are published as requests come and go,** not only at
  a step's end: a request submitted during a long first step was otherwise
  invisible to the drain.
- **Flags only the other model takes are dropped for the load, not
  refused** (`config::fit_to_family`, logged as
  `ignis.model.switch_flags_dropped`): `--vision` on Flash-Next, the other
  model's `--spec` backend, and the Flash-Next-only knobs on the 27B.
  Refusing them, as a restart does, would make the switch impossible on the
  flags the owner starts the 27B with. `--max-context` past the 27B's
  envelope is still refused.
- **Wire details.** Both `artifact` and `model` are required. `GET
  /v1/models` carries `status` beside `data`, plus `switching: {from, to}`
  while switching and `reason` while failed; it and `POST
  /v1/models/switch` stay open while a switch runs or has failed (the gate
  holds every other route), and both stay held during the warm-up as
  before. A server built without a loader answers the switch `501
  switch_unavailable`.
- **Known limit.** The request body cap is chosen when the router is built,
  from the start model: a server started on Flash-Next (never `--vision`)
  keeps the text-only cap after switching to the 27B. Moot today — a switch
  only enables vision when the start flags named it, and Flash-Next refuses
  to start with them.

### The GPU profile, run (2026-10-10)

`model_switch_gpu.rs` ran on the owner's machine (RTX 5090, 64 GB RAM, ~45
GB free at run time) against the real artifacts, replacing AC 16's
estimate:

| | wall time | breakdown |
|---|---|---|
| 27B cold start | 9.22s | — |
| 27B → Flash-Next | 33.41s | teardown 246ms, load 31,918ms, warm-up 76ms |
| Flash-Next → 27B | 12.94s | drain 0ms, teardown 3,743ms, load 8,459ms, warm-up 179ms |

All three host-pool-arena assertions passed: pinned (2,147,483,648 bytes)
while the 27B served, `0` immediately after the switch to Flash-Next, pinned
again after the switch back — the direct proof that the teardown ran before
the next load's create, in both directions. Load dominates both directions,
as expected; the switch's own overhead (drain + teardown + warm-up) is low
hundreds of ms on the 27B side and under 350ms on the Flash-Next side.

**The first attempt failed for a reason worth recording.** Flash-Next's
host plan (`crates/core/src/residency/plan.rs`) needs the expert pool
(~35.2 GiB, not a knob) plus n-gram hot rows (1 GiB default), prompt reuse's
retained host slots (~1 GiB default) and its KV-RAM arena (2 GiB), plus the
6 GiB paging margin — around 48 GiB total — checked against physical memory
free at that instant. With ~37-45 GiB free (this machine runs other
sessions/applications too), the first two attempts refused with
`HostPlanError::BelowMargin` naming `expert_pool` as the crossing line, and
the switch correctly rolled back to the 27B each time (AC 9, proven on real
hardware, not just the mock). The fix was not a code change: the test's
`options()` now sends `--retained-host 0`, an existing flag, saving ~1 GiB
without touching anything the test asserts (the arena stays at its default
for the 27B-side assertions; the expert pool cannot be shrunk). No change
to `HOST_MARGIN_BYTES` or the check itself — the margin did its job both
times, naming exactly what to shrink or what to free, and the owner's own
flags were enough once there was a little more free RAM at run time besides.

## Implicit switch: a request's `model` field (added 2026-10-10)

Trying this end to end against a real client (opencode) exposed a gap this
spec didn't cover: the `model` field on `/v1/chat/completions` (and every
other endpoint that carries one) was never checked against the loaded
model at all — only its `@<lane>` class suffix is read
(`split_model_lane`, `crates/server/src/api.rs:570`). Naming the other
model does nothing; the active model answers regardless, silently. The
owner's ask: let naming the other model *do* something — switch to it —
behind a flag, and only once every request already on the current model is
drained, accepting nothing else meanwhile.

### The rule

A request whose `model` (after stripping the lane suffix) names a model
other than the one currently `Serving`:

- `--allow-model-switch` (default **true**, env `IGNIS_ALLOW_MODEL_SWITCH`)
  off, or the named model unknown to this server: refused, not silently
  served by the wrong model — `404 model_not_found` naming the id and (if
  the flag is off) that implicit switching is disabled. This closes a real
  gap: today every model name is accepted and ignored, which looks like the
  field is honoured when it never was (the #284 field table's own
  "honoured or refused" rule, applied here for the first time to `model`
  itself rather than its suffix).
- Flag on and the model is known: this request **triggers** a switch to it,
  reusing `model_switch::switch` exactly as `POST /v1/models/switch` does —
  same gate, same drain (`--switch-drain-timeout`), same teardown-then-load
  ordering, same rollback on failure. Nothing about the switch mechanism
  itself is new; only who may start one is wider.

### What "known to this server" means: `--known-model <id>=<path>`

An implicit switch has no request body field for the target artifact's
path — a chat-completions request carries a model *name*, not a path. A
new repeatable flag, `--known-model <id>=<path>` (env `IGNIS_KNOWN_MODELS`,
`;`-separated `id=path` pairs — not `,`, since a Windows path never
contains `;` but can contain `,`), names every model this server is
willing to switch to and where its artifact lives. The model and artifact
the server actually starts on is added to this table automatically, so
switching back to it never needs its own flag. Naming a model with no
entry here is the `404 model_not_found` case above, flag or no flag — an
unlisted model literally cannot be loaded, since nothing names its
artifact.

### Ordering: close first, drain, then switch — the triggering request included

"Must drain everything queued, and must not accept anything else once a
switch is wanted" (the owner's wording) is exactly §Draining's existing
gate-then-drain order, with one new wrinkle: the request that *discovered*
the need to switch is not yet an admitted request when this happens
(the check runs where `resolve_model_and_class` already runs, before
admission), so it is not one of the things the drain waits for — it waits
*with* the client instead.

1. The gate closes (`Switching{from, to}`) the instant the mismatch is
   seen, before anything else about this request is evaluated. Every other
   request — on the old model or the new one, admitted or not — now gets
   `503 model_switching`, identically to an explicit switch. This is the
   "accepts nothing else" half.
2. The triggering request's own HTTP call blocks (does not return a
   response yet) while `model_switch::switch` drains what was already
   running on the old model, tears it down, loads the named model, and
   warms it up — the "exhaust the queue first" half, unchanged from
   §Draining.
3. On success, the triggering request is admitted and served **on the new
   model**, as if it had been sent after the switch finished — the client
   that asked for Flash-Next gets Flash-Next, not a `202` it has to poll.
4. On failure, the triggering request is refused with the switch's own
   reason (the same string `POST /v1/models/switch` would have reported via
   `GET /v1/models`'s `reason` field), not silently served by the
   (reloaded) old model — this client asked for a specific model and
   either gets it or a reason why not, never a substitution.
5. A second request naming a model while step 2 is already running for a
   *different* target gets `503 model_switching` like any other request
   during a switch — it does not queue a second switch, and does not
   retarget the one in progress. Named the same target already in flight:
   still `503 model_switching` for this slice; joining an in-progress
   switch instead of waiting it out as a plain 503 is a possible
   refinement, not required here.

Consequence for a real client (what prompted this): opencode's chat
completion call simply takes as long as the switch takes the first time it
names a different model (seconds, per the GPU numbers above), then answers
normally — no client-side retry loop needed, which a bare `503` would have
demanded of every OpenAI-compatible client pointed at this server.

### Out of scope, still

A policy that infers the right model for a task without being told its
name — still a router, still not this. Multiple servers, or serving two
models from two processes behind a single endpoint, is a different
architecture this spec never considered.
