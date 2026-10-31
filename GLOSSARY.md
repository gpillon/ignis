# ignis

ignis is the Rust inference engine for Qwen models on a single SM120a card:
text models and an image model, one loaded at a time, and a speech model
loaded beside whichever it is. It is deliberately specialized, to few model
families and one GPU class.

## Language

### Engine & process

**Kernel leaf**:
The C++/CUDA compute library linked into ignis behind a C ABI. The forward
pass of the text and image models and all GPU compute live here; Rust owns
everything above the **step** — scheduling, KV accounting, serving (ADR 0009).

**Step ABI**:
The C ABI between Rust and the leaf, named after its granularity: one call per
*step* (model load, sequence lifecycle, a prefill span, a decode round), not
one per operator. Device-resident — no host activation pointer crosses it (ADR
0009).

**Program**:
The leaf-side layer that owns the forward pass and the sequence state, on top
of the vendored ops. Ours, not vendored (ADR 0009).

**Vendored op**:
An operator copied *verbatim* from a named upstream (the reference, or a third
party under MIT, BSD or Apache-2.0: ADR 0051) and tracked by a manifest (pinned
commit, content hashes, recorded patches). Anything hand-written is not one,
and carries no port claim (ADR 0010).

**Adapted op**:
Our own operator that contains parts copied from a named upstream under MIT,
BSD or Apache-2.0, with its repo, commit, path and licence in the file header
and in `kernel/NOTICE`. Ours to change; no parity claim (ADR 0051).
_Avoid_: port, vendored (for an adapted op).

**Port claim**:
The claim that a kernel file is its named upstream's code, which must be
diffable: only a **vendored op**, verbatim or under a recorded patch, makes one
(ADR 0010, ADR 0031), and our own ops and **adapted ops** make none (ADR 0051).
It says where kernel code came from, not what a seam is: `codebase-design`'s
**port** is the interface at a seam and its **Adapter** the thing that fills
one.
_Avoid_: port, adapter (for kernel provenance: say port claim, vendored op,
adapted op); port claim, adapted op (for a seam role: say port, Adapter).

**Reference parity**:
A behaviour ignis computes exactly as the **reference** does. The default for
every vendored op, and not a claim of correctness: the reference can be wrong
(ADR 0037).
_Avoid_: faithful, reference behaviour (for a behaviour ignis has changed).

**Ignis patched**:
A behaviour ignis deliberately computes differently from the **reference**,
because the reference is wrong there; each is a recorded patch with a test that
tells the two apart. A difference from the reference it causes is a declared
departure, not a regression (ADR 0037).
_Avoid_: fork, divergence.

**Lane**:
A concurrent decode slot. Requests hold a lane while decoding.

**Lane tag**:
The request's own statement of its class, carried as an ignis extension field
on the request and echoed in the request log. Two classes, `Interactive` and
`Agent`; absent or unrecognized means `Interactive`, except on a **decision**
request, where it means `Agent`.

**Prefill lane**:
The single global prefill slot: exactly one request at a time holds
device-resident **prefill progress** and consumes chunks, and every other
prefill queues behind it. It is a slot, not a promise about decode: whether
decoding continues while it runs is decided by **prefill/decode
interleaving**, not by this term (ADR 0018).

**Admission state machine**:
The fairness machinery (protection, backfill class, temporal credit, frontier
distance) deciding which request gets which lane. Pages are a dimension of
their own: see **KV reservation**.

**Generation cap**:
The most a request may generate, reasoning included: its `max_tokens`
(`max_completion_tokens`, `max_output_tokens`), or else the server's
`--model-default-max-tokens`, clamped to what the prompt leaves of
`--model-max-context`. Reaching it ends the request with `finish_reason:
"length"`.

**Deadline**:
How long a request may run before the server cuts it (ADR 0058):
`server.request_timeout` for one answered whole, `server.stream_timeout` for
one that streams, from its admission (an **image job**'s from its start, never
its queue); `0` is none. A passed deadline cancels the request and frees its
**lane**.

**Partial reply**:
What a request past its **deadline** is answered under `server.timeout_reply:
partial`, the default: what it generated, ended as the **generation cap** ends
one (`finish_reason: "length"`, a response `incomplete` with
`max_output_tokens`). Not a cap: the header `ignis-stopped-by` and the request
log's `cause: timeout` say which it was.

**Reasoning redirect**:
An EOS a request draws while its reasoning block is still open becomes the
block's `</think>`, and the model goes on to answer (ADR 0048, GitHub #315).

**Loop guard**:
The server task between a reasoning request's engine event stream and its
consumer that watches the reasoning for a loop (ADR 0052, config
`model.loop_guard*`, `rewind` by default).

**Rewind**:
The loop guard's action: the engine request is cancelled and a
**continuation** submitted. The client keeps what it received and gets the tag
`[LOOP DETECTED, REWINDING n TOKENS / m CHARS · h]` in the reasoning, `h` a
short hash of the struck text; the Playground strikes the `m` characters before
it through.

**Continuation**:
The request a rewind submits: the same input, its prompt extended by the tokens
kept before the cut and a **nudge**, with what is left of the generation cap
and of the thinking budget.

**Nudge**:
The sentence, in the model's own voice, a continuation ends its prompt with
("Wait, I'm starting to repeat myself. …"): `loop_guard_nudge` for the first
rewind, `loop_guard_nudge_again` for every later one. After
`loop_guard_retries` nudges, the thinking close takes its place and the
reasoning ends.

**History repair**:
On every incoming request, guard on or off, the `m` characters before each
rewind tag and the tag itself are removed from assistant reasoning when they
hash to the tag's `h`, so the model is given back the text it actually saw (ADR
0052). Never applied to any other text.

**Holdback**:
`loop_guard_holdback` characters of reasoning held back from the client (0,
off, by default), so a rewind whose cut falls in the unsent part reaches the
client with no tag at all.

**KV reservation**:
The KV pages the pool has given a request. Its *bound* is prompt plus
**generation cap**, never past `--model-max-context`.
_Avoid_: allocation (the leaf's word for the mapped pages).

**Parked lane**:
On the growth branch, a decode lane held out of rounds because its room ahead
is below one round's largest append and no page can be had yet. It keeps its
pages and resumes at the first advance with room.

**VRAM budget**:
The device memory the whole ignis process may hold, weights included: the
number Task Manager shows. Either explicit (the operator names it) or derived
(the memory free at start minus the **VRAM headroom**).

**VRAM headroom**:
The device memory a derived **VRAM budget** leaves to everything else on the
card: the desktop and other processes.

**VRAM oversubscription**:
Holding more device memory than the card has free. Windows serves it by paging
device allocations to system RAM (Task Manager's "Shared GPU memory") instead
of failing, which slows every step; other systems fail the allocation.
_Avoid_: shared memory, spill (a spill is KV-RAM's).

**Host-mapped**:
Kept in pinned host RAM and read by the GPU where it sits, over PCIe, never
copied to the device: the 27B's token embedding (ADR 0062). Every read crosses
the link, so it suits data a step reads once.
_Avoid_: on the host, offloaded (an **offloaded** load is a KV pool policy).

**Staged**:
Kept in pinned host RAM and copied to the device for each use, into room the
load already reserved, then read there at device speed. Nothing stays on the
device between two uses, unlike an expert in the **expert cache**. Suits data a
step reads many times.
_Avoid_: on the host, offloaded.

**KV pool policy**:
How a load sizes its KV pool, chosen by whether every weight is on the device
(ADR 0045). A **resident** load (the 27B, and Flash-Next on a card that holds
every expert) gives the pool the rest of the **VRAM budget**; an **offloaded**
one (Flash-Next with its experts streaming, as on the 5090) reserves the pool
first and gives the **expert cache** the rest.

**Trained position envelope**:
The 262,144 rotary positions the checkpoint was trained over. Not a limit the
engine enforces: a sequence past it is served (the attention envelope reaches
1,048,576 keys under hq-e8-2b), it is simply rotated at angles the model never
saw.

**RoPE scaling**:
The text rotary frequency table a load runs on, chosen at load and frozen for
its life. Either *none* — the linear table, correct through the **trained
position envelope** — or **YaRN** at a factor, which rescales that envelope:
the high-frequency pairs keep extrapolating, the low ones interpolate at
`1/factor`, a ramp blends the band between, and an attention factor `m` rides
along.

**Performance-first**:
The organizing principle: correctness (a self-check of *sane* output) is a
non-negotiable floor; above it, performance is the #1 objective and
tie-breaker for every scope, feature, and kernel decision.

**North-star**:
The holistic objective: "the best local coding engine" — max performance
**and** agent parallelism that saturates the GPU in prefill *and* decode.

**Self-bootstrapping**:
The dev/test loop where the very runner used to build ignis (ninfer, running
Qwen 3.8-27B) is running while we test; two engines share the 5090, so GPU
testing is exclusive (ADR 0006).

**Reference**:
The ninfer stack: the source of the artifact, the vendored ops, the oracle
recordings and the speed numbers. A reference, not a target to match
token-for-token (ADR 0005 / 0007).

### Model

**Layer kinds**:
The model is 64 layers: **16 GQA** (full attention) and **48 GDN** (linear
attention), hidden 5120, intermediate 17408, vocab 248320.

**GQA layer**:
A grouped-query attention layer: 24 query heads / 4 KV heads of 256, over
paged KV, rotated by RoPE at the *token's* position.

**GDN layer**:
A gated-delta-net (linear attention) layer: **16 key heads and 48 value heads
of 128**, whose recurrence carries the sequence's GDN slot instead of a KV
cache.

**Fused plane**:
A single stored tensor holding several projections stacked by row (e.g.
`attention/query_key_gate_value`, `mlp/gate_up`). Its row order is part of the
artifact contract, not a guess.

### Flash-Next

**Flash-Next**:
Qwen3.8-Flash-Next, the second model: 125B mixture of experts, 6B active, 48
layers in 12 blocks of 3 GDN + 1 QSA, hidden 2560. Served from its own
artifact, loaded at start or by a switch, and never beside the 27B.

**Expert**:
One of a layer's 512 routed SwiGLU MLPs (2560 → 640 → 2560); the **router**
picks 10 per token. The **shared expert** runs for every token, gated by a
sigmoid.

**Expert projection**:
An expert's fused gate/up plane or its down plane: the unit the artifact stores
contiguously and residency moves.

**K**:
A matrix's bit width in the **trellis** format, chosen by **allocation** on
calibration distortion: per expert projection on Flash-Next (one of 2, 2.5, 3,
4; mean 2.5 bits per weight), per projection on a 27B artifact whose **body
format** is trellis (2 to 6).

**K class**:
Expert projections of one shape and one K, hence one byte size: eight classes,
each with its own slot pool in the expert cache.

**Expert pool**:
Every expert projection, in pinned host RAM for the life of the load. Only an
offloaded load has one; a **device-only** load has none.

**Device-only residency**:
A resident load's expert residency (ADR 0054): the expert cache holds every
projection, so there is no expert pool. At load each record is **admitted**
from the artifact straight to its slot, a layer at a time, and no step misses.

**Expert cache**:
The fixed-size VRAM slot pools that hold the projections in use, replaced
least-recently-used; a plan line at load. It takes what the **KV pool policy**
leaves.

**Pinned hot set**:
The calibration-hottest experts an offloaded load keeps in its **expert cache**
for its whole life (`--vram-pinned-expert-bytes`: 0 by default, 2 GiB under
the `rtx5090` profile; ADR 0055). Each projection is a **pinned expert**: a
slot past its K class's LRU slots, a hit whenever selected, never evicted or
copied, and no copy in the **expert pool**, which is smaller by their bytes.
_Avoid_: "pinned" alone for it: the expert pool is pinned (page-locked) host
memory. Nor "pinned" for the slots a step selects: a step **holds** them only
until it ends (the kernel's `s_held`, the policy model's `held`), held in the
sense a request holds a **media embedding** — in use until released — where a
pinned expert is never released.

**Router lookahead**:
The next layer's router applied to this layer's MoE input to prefetch its
likely experts into the cache.

**Demand copy**:
A step's copy of the projections it selected that the expert cache did not
hold (its misses), on the layer's stream: the expert op waits for it. A
**prefetch** copy runs beside the op instead, for the next layer.

**Stall**:
The device time the expert ops waited on their steps' demand copies, per phase
(ADR 0017). The wait for a previous step's unfinished prefetch is not in it.
_Avoid_: residency wait (for the stall alone).

**QSA layer**:
Flash-Next's attention layer: GQA 24 query / 2 KV heads of 256 with an
**indexer** that, above 2051 tokens, selects the key blocks the attention reads
(sparse); dense below.

**Hyper-connection streams**:
Flash-Next's residual: four streams per token, each sublayer reading a learned
mix and writing back through a gated residual.

**N-gram table**:
Flash-Next's hashed n-gram embedding table (320M rows, INT4), a
**host-streamed** object: read row by row from NVMe through a RAM hot-row cache
(or held whole in RAM when the hot-row budget covers it), never materialized on
the device.

**MTP head**:
The checkpoint's own one-layer draft head (spec `flash-next/07`): it combines
the trunk's pre-mixer streams with the next token's embedding and drafts the
token after it.

**Companion container**:
A small `.ninfer` beside the main one that holds only the MTP head
(flash-next-layout.md §13), with its own identity; bound only when MTP is on.

**Pair pin**:
The companion's record of the one main container it belongs to (`model_id`,
`weights_id`, content hash), checked at load; a mismatched pair is refused by
name.

### Image model

**Image model**:
Qwen-Image-2.1-Turbo, served from its own artifact: a Qwen3-VL text encoder, a
7B diffusion transformer (the **DiT**) and a VAE. Reached by a switch, never
loaded beside a text model.

**Image job**:
One image request from admission to the delivery of its last image.

**Target image**:
The image an image job is generating.

**Condition image**:
An input image an image job is conditioned on (an edit's source or one of its
references).
_Avoid_: reference image (**Reference** is the ninfer stack).

**Image session**:
The Playground's line of work on pictures (spec playground/06): an **image
pool** and its steps, each step a prompt, the **image inputs** it was sent with
and the pictures it made. Kept in the page's image store.

**Image pool**:
Every picture of an image session — uploaded, made by a step, marked, or
brought from the chat — each named `#N` for good.

**Image input**:
A picture of the image pool chosen for the next step; the prompt calls the
inputs image 1, image 2… in order. One input is one condition image, or two
when it is marked as a mask (the picture, then its mask).
_Avoid_: reference, reference image.

**Condition prefix**:
The prompt and the condition images as the DiT sees them before the target
image; computed once per image job and reused by every denoising step of every
image the job makes.
_Avoid_: prefix, shared prefix (those are a text model's).

**K/V run**:
A span of keys and values stored in 8 bits for the DiT's attention and
quantized as one unit: keys smoothed by one mean, values scaled per channel
over the whole span. The condition prefix is one run, the target image another.

**Condition budget**:
How many condition-prefix tokens the image model can hold, set at load from
the VRAM the rest of the plan leaves. Condition images beyond it are scaled
down, never refused.

**Denoising step**:
One evaluation of the DiT over the target image at one noise level, and the
sampler update after it. Always qualified.
_Avoid_: step (bare: that is the Step ABI's unit).

**Prompt enhancement**:
An image job's optional rewrite of its prompt by a text model before the image
model sees it.
_Avoid_: polish, rewrite.

**Revised prompt**:
The prompt that prompt enhancement produced, returned to the client beside the
image.

**Enhancer model**:
The text model that performs prompt enhancement, named by its served id in the
config.

**Enhancement visit**:
A switch to the enhancer model and back inside an image job, made when the
image model is loaded. It is not a switch for **Last switch**.
_Avoid_: borrow.

### Speech model

**Speech model**:
A model that takes audio in and gives back its transcript; it never converses.
The third **model kind**, beside the text model and the image model:
Qwen3-ASR.
_Avoid_: voice model, ASR model, audio model

**Transcription**:
One server job turning audio into text, from admission to the delivery of its
last text: a whole clip, or live from a stream of audio.
_Avoid_: recognition (bare)

**Dictation**:
The Playground act: the user speaks into a composer and the transcript lands at
the cursor, editable, never sent by itself.
_Avoid_: voice input

**Diarizer**:
A model that tells the voices in audio apart and gives back who spoke when; it
never transcribes. The fourth **model kind**: Nemotron 3 Diarization, loaded
beside the **speech model** and never resident in VRAM (ADR 0069).
_Avoid_: speaker model, voice model, diarization model

**Transcript**:
The Playground's record of one conversation heard in its Transcribe view: its
**utterances** in order, each with its times, its **speaker** and its text, and
never its audio. Not a **dictation**: nothing lands in a composer.
_Avoid_: conversation (the chat's), recording, meeting

**Utterance**:
One stretch of a **transcript** spoken by one **speaker**, from where speech is
heard to the silence, the speaker change or the length limit that ends it. A
pause can split one speaker's turn into two.
_Avoid_: turn, segment (a **locate**'s), item (the Realtime protocol's)

**Speaker**:
One voice the **diarizer** told apart in a **transcript**, numbered in the
order voices first spoke, and renamed by the user; on a call, the microphone's
voice is "Me".
_Avoid_: channel (the model's output slot), participant

**Notes**:
What the text model writes about a **transcript**, kept with it, in sections
its kind of conversation asks for, in two forms: **running notes** and a
**recap**. Not a **memory note**.
_Avoid_: summary (one of the sections), minutes

**Running notes**:
The **notes** kept up while a conversation goes on, in sections for talking:
what is on the table now, who is who, what was said that adds up. They change
a line at a time.
_Avoid_: live notes, minutes

**Recap**:
The **notes** written once from a whole conversation, in sections for what
comes after it: what it was about, the decisions, the action items, the open
questions.
_Avoid_: summary (one of its sections), final notes, minutes

**Context image**:
A picture added to a **transcript** at one of its moments, cropped or blacked
out by the user, that tells what the conversation was looking at from then on.
_Avoid_: slide, screenshot, attachment (each is one case of it)

### Weights & artifact

**Artifact**:
The `.ninfer` container a model is served from: its weights in the artifact's
**body format**, its endpoints (token embedding, output head), an optional
grafted DFlash2 module, and frontend objects (tokenizer / chat template).
Several artifacts share one **model family**.

**Body format**:
The weight format of an artifact's transformer layers: **NVFP4**, or
**trellis**. Read from the artifact, never configured.
_Avoid_: "3-bit model", "small model" (the family is the same).

**Trellis**:
The EXL3 weight code (16×16 tiles, the `mul1` codebook, Hadamard rotation with
per-channel scales on both sides), bit-compatible with exllamav3's records, at
a **K** per matrix.

**Object**:
The unit inside an artifact. The binder must consume every object at bind time;
an unconsumed object is a load failure.

**Device view**:
The artifact crate's export for a bound tensor: device pointer, storage layout,
format and shape, so the program binds planes directly with no host dequant.

**NVFP4**:
The weight quantization format of most of the model: E2M1 values, an E4M3
group scale per 16 stored in a separate **scale plane**
(`blockscale-k16-m128x4-v1` layout), and a per-tensor **weight divisor**.

**hq-e8-2b**:
The reference's HyperQuant KV cache format, and the profile the owner actually
runs: a randomized Hadamard rotation, E8-lattice quantization and Rice coding
packed into a fixed per-row budget of 64 code bytes plus 8 metadata bytes per
(token, KV head). Fixed bytes per token, so page addressing and CUDA-graph
address stability are unchanged.

**Prompt planes**:
The BF16 copies of a GQA layer's visible keys and values that the vendored
hq-e8-2b prompt route decodes the cache into before a prefill chunk attends:
prefill-only scratch, 4,096 bytes per context token up to 262,144 keys (one
band). Gone from serving since #335 (ADR 0061): every hq prefill chunk takes
the **prompt attention** instead, and the vendored route survives only as the
oracle the route agreement test holds it against.

**Prompt attention**:
ignis's own hq-e8-2b prefill attention (an adapted op, ADR 0051/0061): two
passes sharing one online softmax, the **residual window** in BF16 and every
other key through the **band** in 8 bits (or FP16 with FP16 accumulate, the
fallback; the load option `--model-prefill-attention` picks one, `fp8` naming
the 8-bit band). Its workspace stops growing with the context at one band.

**Band**:
The fixed buffer the prompt attention decodes the codec's keys into, once per
chunk and layer: 32,768 keys of INT8 K and E4M3 V^T (64 MiB) with a scale per
32-key tile, walked band after band over a longer history. Not the prompt
planes' "band", which was their 262,144-key ceiling.

**Residual window**:
The keys hq-e8-2b attention reads exact instead of decoding: a sequence's first
32 keys (the **sinks**), a 512-slot **ring** of recent keys, and the prefill
chunk being attended. Per slot, not per page, so it is a **state section** a
clone or a restore carries; a ring bit names no position, so appended columns a
sequence does not keep (a verify round's rejected drafts) must have theirs
cleared.

**KV format**:
Which of the two KV cache formats a load runs: **hq-e8-2b**, the serving
default from G4, or **BF16**, retained as the format every correctness oracle
runs against. A model-load option, fixed for the life of the load.

### Models, catalog & download

**Model family**:
The architecture an artifact runs as (`Qwen3.8-27B`, `Qwen3.8-Flash-Next`,
`Qwen-Image-2.1`), read from the artifact's own header: a catalog entry may
state it, and the header must agree (ADR 0074). It decides the forward
program, the drafter and the family scope.
_Avoid_: model (for the architecture)

**Model kind**:
What a model takes in and gives back: a **text model** takes text and images
and gives back tokens, and converses (the 27B, Flash-Next); an **image model**
takes a prompt and gives back pixels (the image model, ADR 0050); a **speech
model** takes audio and gives back its transcript, and never converses (ADR
0067); a **diarizer** takes audio and gives back who spoke when, and never
transcribes (ADR 0069). Above the model family: every family is of one kind.
_Avoid_: modality (the 27B already takes images in), pipeline

**Served id**:
The name a client puts in a request's `model` and `GET /v1/models` reports:
the operator's `model.id`, or the family's default id when none is named. It
tells artifacts of one family apart; the family never does.
_Avoid_: model name

**Alias**:
A served id one catalog entry also answers to on a **card class** below that
id's own entry: its family's default id, or another entry's served id of the
same family (ADR 0064, ADR 0074). It resolves only within
one model family — the entry that answers to it holds an artifact whose header
names that family — and only to an entry the card can load.

**Uncensored variant**:
A catalog entry of a model family whose weights carry an abliteration of the
refusal direction: the same family, told apart by its served id, and it does
not refuse.
_Avoid_: abliterated model (the upstream checkpoint), graft (the DFlash2
module's)

**Catalog**:
The models the server knows how to fetch: the **built-in catalog**, shipped
inside the binary, plus the **operator catalog**, a separate file the config
names.
_Avoid_: registry (that word is the remote server, as in "a private
registry"), model list

**Catalog entry**:
One published model: its served id, repo, revision, the artifact to load, every
file it needs with its byte count and SHA-256 pinned, the smallest **card
class** it loads on (`card_class`, optional: an entry stating none is never
refused by class; every built-in text and image model states it, while the
speech models and the diarizer, which run on the CPU, state none), the
**aliases** it may answer to (ADR 0064), and its **model family** when
neither its id nor its aliases name one (`family`, ADR 0074). An operator
entry never reuses a built-in id.

**Endpoint**:
The server a catalog entry's files are fetched from: Hugging Face by default,
or a mirror answering the same repo/revision/file paths. Chosen per machine,
not per entry.
_Avoid_: registry, hub

**Download**:
The server fetching a catalog entry's pinned files from the endpoint into its
models folder, resumed where it stopped and verified against the pins. It is
started at start-up, by the `model download` command, or from the Server
view, one at a time, and never by a switch.
_Avoid_: pull, install

**Removal**:
Deleting a downloaded catalog entry's files, finished or partial, from the
models folder. It never touches the model loaded, loading or being downloaded,
nor a file another entry on disk also lists.
_Avoid_: uninstall, delete (for a model)

**Switch by id**:
An explicit `POST /v1/models/switch` naming only the served id: resolved as a
known model (`switch.known_models`, the start model), then a catalog entry on
disk. `switch.allow_implicit` does not gate it; it gates only the **implicit
switch** a request triggers by naming a model.

**Servable model**:
A model a request may name and be served by now, with no download: the active
model, plus every id the **implicit switch** resolves to an artifact that is a
file (a catalog entry on disk, a known model whose artifact is there). What
`GET /v1/models` lists, the active one first (`active: true`), computed at each
request; with `switch.allow_implicit` off, the active model alone.
_Avoid_: available model, installed model

**Last switch**:
How the most recent switch or reload ended (`switched`, `refused`, `restored`,
`failed`, with the reason), kept on `GET /v1/models` after the status itself
has moved on. An **idle unload** and the load after it are not switches and
leave it as it was.

**Idle unload**:
Tearing the loaded model down after `model.idle_unload` seconds (default 300, 0
= never) with no request using it and the engine empty, the way a switch tears
one down, while the process and the API stay up (ADR 0049). `GET /v1/models`
then reports `status: unloaded` under the same id; the next authenticated
request that needs the model loads it again (`status: loading`, requests
arriving meanwhile wait) and is served.
_Avoid_: sleep, suspend, eviction (that word is a sequence's, not a model's)

### Prompt rendering

**Instruction message**:
A chat message addressed to the model rather than said in the conversation:
role `system`, or role `developer`. The model was trained on neither more than
one of them nor on `developer` at all, so every instruction message the engine
accepts reaches the model as a system block.

**System prompt**:
The instruction the rendered prompt opens with, where the tools block and the
reasoning instructions attach and a **retained prefix** is cut: the leading run
of `system` messages joined into one, or a leading `developer` message when no
`system` precedes it.

**System message policy**:
What the engine does with a `system` message that is not the first message. By
default a run of them at the start joins the **system prompt** and a later one
is rendered in place as a block of its own; strict refuses the request instead,
as the model's template would.

**Developer message policy**:
What the engine does with a `developer` message: render it in place (the
default), join it into the **system prompt**, gather every one into a single
block after it, accept only one right after it, or refuse it. Joining and
gathering re-render earlier history when a new one arrives mid-conversation, so
they trade **prefix reuse** for fewer system blocks; the others never move a
message.

**Render refusal**:
A request whose messages cannot be rendered into a prompt. It is refused before
admission, never served as an empty prompt.

### Sequence state

**Sequence handle**:
The leaf-owned, opaque per-sequence object created by the step ABI: its KV
pages, its GDN slot, its conv taps, its position, its last token, and (P3-03)
its penalty-count row. It is what a snapshot captures and a restore rebuilds.

**KV page**:
A fixed-size device page of the paged KV cache, addressed through a
per-sequence block table. Page geometry is reported by the runtime and is what
the scheduler's KV pool counts.

**GDN slot**:
One sequence's linear-attention state: per GDN layer, **48 value heads × 128 ×
128 fp32** (all 48 GDN layers ≈ 144 MiB per sequence), drawn from a slot pool
sized by the concurrency.

**Conv taps**:
The per-sequence causal-conv1d history of a GDN layer.

**Penalty-count row**:
A sequence's per-vocab-entry occurrence count (one int32 per vocab entry),
drawn from a slot-indexed device buffer the same shape as the KV/GDN pools (one
row per slot). Device-side sampling's presence/frequency penalties read and
atomically update it; zeroed at `ignis_seq_alloc` like every other slot
section.

**State section**:
One part of what a sequence is made of: its KV pages, its GDN slot, its conv
taps, its position and last token, its penalty-count row, on a pool built with
DFlash2 its **drafter window**, and on an hq-e8-2b pool its **residual
window**. Each is either shareable read-only (KV pages) or must be cloned per
sequence (everything mutable).

**Snapshot blob**:
The opaque, versioned byte image of a whole sequence that crosses the ABI.
Callers ask the leaf for its size and a format version and never see the
section layout; a restore validates the header and refuses a stale or foreign
one.

**Blob identity**:
What state was produced under, carried in the **snapshot blob**'s header and in
the retained entry alike: the artifact's content hash, the **KV format**, the
blob layout version the leaf writes, and the drafter's presence and draft
window. It answers whether these bytes may be written into a sequence at all,
and a mismatch is refused before one of them moves.

**Residency tier**:
Where sequence state lives, in three levels: **Tier 0 — device** (VRAM), **Tier
1 — KV-RAM**, **Tier 2 — KV-disk**. State moves down a tier when the one above
needs the room, and back up to be served.
_Avoid_: host tier (say KV-RAM).

**KV-RAM**:
Tier 1: the host-RAM tier that holds **snapshot blobs** of evicted live
sequences (so they resume instead of re-prefilling) and of retained state
(**prompt checkpoints**, **retained prefixes**) that left the device. Bounded
by a byte budget held whole from the start rather than grown blob by blob — one
page-locked arena, blobs placed first-fit inside it, so room for a blob is a
free *span* and not a count of free bytes — two-tier (probation → protected),
and written only at a **chunk boundary**.

**KV-disk**:
Tier 2: the disk tier below KV-RAM (ADR 0045). It takes what KV-RAM gives up,
instead of discarding it, and a device victim KV-RAM cannot take.

**Live move**:
A sequence still being served, moved down a **residency tier** for the room an
admission ranked above it needs, and brought back later to resume bit-exact
where it stopped (ADR 0045). It is a transfer, not a call: the sequence is in
no round from its start until its last window lands, keeping its pages until it
has left.

**Prompt checkpoint**:
The whole-sequence state of a request at its generation opener, the point
where the rendered prompt hands over to the model, retained after the request
ends so a later request whose prompt extends it (the next turn of a chat or of
an agent loop) resumes there instead of re-prefilling. Matched by prompt
content, never by a session identifier.

**Retained prefix**:
A **shared prefix** kept alive with no claimant, so a later, non-concurrent
request (the next subagent of a burst) claims it as if it were a sibling. It
ends at a **reuse boundary** — for a chat, the end of the system and tools
block, floored to whole KV pages — never at the publishing request's whole
prompt head.

**Reuse boundary**:
A prompt position where a request's state is kept for a later request to
resume from, predicted before prefill because a hybrid can only resume where it
captured (ADR 0029). Sources: the end of the system block, a **fan-out head**,
an **observed fork**, a **reuse marker**.
_Avoid_: breakpoint, cache point (a **prompt checkpoint** is something
else).

**Fan-out head**:
The **reuse boundary** at the longest common prefix of a decision fan-out's
rendered questions, computed before the first is submitted. Kept only until
that fan-out ends.

**Observed fork**:
A **reuse boundary** at the end of the longest run of leading content parts
that a recent decision `state` also began with, keyed by the **match key**
there. The second request that repeats a run keeps it; the third and later
resume from it.

**Reuse marker**:
`cache_control: {"type": "ephemeral"}` on a decision `state`'s content part:
the caller's **reuse boundary** at the end of that part. It chooses where state
is kept, never what a request may claim — matching stays by content.
_Avoid_: cache breakpoint.

**Lineage**:
One conversation's chain of **prompt checkpoints**, and the only name a
conversation has. The clients send no session id, so what ties turn N+1's
checkpoint to turn N's is the link that does exist: turn N+1 claimed turn N's
entry.

**Chained prefix**:
A **shared prefix** published on top of another. It publishes the pages it
warmed itself *over* that entry, taking over the reference it was holding
rather than adding one, so every page is still charged to the pool exactly
once and a claimant of the chain shares all of it.

**Pages-only link**:
A **chained prefix** with no image of mutable state: the whole pages between
what a request shares and its generation opener's page floor, handed over by
the **prompt checkpoint**'s capture so that something outlives the request to
hold them. No request stands at its end, so it is never claimed on its own; it
takes no **retained slot**, and it goes when nothing stands on it any more.
_Avoid_: imageless prefix.

**Retained state**:
Prompt checkpoints and retained prefixes: state no live request needs. It
never costs a live request anything — on the device it is always the first
thing to go — and in KV-RAM it is discarded before any evicted live sequence.

**Retained slot**:
A place reserved at load for one mutable-state image of a **shared prefix**
(retained, chained or still claimed) or of a **prompt checkpoint**, the same
size as a **lane**'s own state (under hq-e8-2b that includes its **residual
window**, which the VRAM plan counts on a line of its own). Two kinds: a
**device** retained slot sits in the device state arenas beside the lanes and
is copied device to device; a **host** retained slot sits in one pinned host
block reserved at load and costs a PCIe copy per capture and per claim, but no
VRAM.
_Avoid_: retained pool (a ledger of bytes, not a place).

**Match key**:
What a piece of **retained state**, and every **shared prefix**, is addressed
by: a hash chain over the prompt tokens it covers, with each media item's
identity (its content digest and grid) mixed in at that item's **first**
placeholder, so a prefix covering one placeholder has committed to the whole
image. Token ids alone are not enough: two same-size images expand to
identical placeholder ids.

**Prefix reuse**:
Requests sharing a prefix skip the redundant prefill — concurrent siblings
through a **shared prefix**, later requests through **retained state**: the
shared **KV pages** are refcounted and shared in place on the device, and the
mutable **state sections** are cloned device-to-device. It never round-trips
through host RAM, and sharing pages alone would save nothing, since prefill
must traverse every layer to produce the mutable state (ADR 0024).

**Shared prefix**:
The leaf-owned object prefix reuse is built on: the physical KV pages of a
prompt head plus a device image of the mutable state at its end, held by a
refcount. Every holder's block-table row addresses the same pages, the pool is
charged for them once, and they return when the last holder releases, unless
it becomes a **retained prefix**.

**Publish point**:
The chunk boundary a prefix is published at, and 0 for a request that publishes
nothing. A shared or retained prefix's is always a whole number of KV pages in
— a **chained prefix**'s is measured from the start of the prompt, not from
what the publisher already shares; a **prompt checkpoint**'s is the generation
opener wherever it falls, and whoever reuses it copies the partial page rather
than sharing it.

**Eviction priority**:
The one ordering that decides what loses residency, on the device, in KV-RAM
and on KV-disk (ADR 0023, ADR 0045). A need moves only what ranks below its
requester: retained state first, then `Agent` before `Interactive`, then what
is not in the decode round, then the latest-submitted.

**Chunked prefill**:
Prefilling a prompt span through the span+position prefill call in **prefill
chunks** rather than one token at a time. The leaf knows how to loop over a
span of any length; it is no longer the only one that does — under
interleaving the scheduler hands it one chunk per call and keeps the loop
itself.

**Prefill chunk**:
The number of tokens one traversal of the model processes during chunked
prefill: a model-load option, default 1024, a multiple of 128. The unit the
**prefill scratch** is sized for.

**Prefill scratch**:
The load's one device region for a step's transient buffers, sized at load for
the larger of what a **prefill chunk** needs and what a **media encode** of one
**item bound** needs (the VRAM plan's `workspace` line). The two never run at
once, and nothing in it survives from one step to the next.
_Avoid_: workspace (a user's files), arena (KV-RAM's).

**Per-token prefill route**:
The G1 prefill path that runs the program one token at a time (recurrent GDN,
small-T attention). Retained after G2 as a test-only, per-call route: the
self-oracle chunked prefill is checked against.

**Compute policy**:
The per-call activation-precision policy handed to every NVFP4 projection
(`A16Only` or `AllowA4`). The engine's policy is the reference's: `AllowA4` on
every NVFP4 text projection, prefill and decode alike, with the vendored
per-projection token thresholds deciding the actual route.

**Decode round**:
One step for every decode-ready lane, run as a single batch-wide traversal of
the model: the lanes are the batch's rows at one token each, so eight lanes
stream the weights once rather than eight times. It is the unit a decode CUDA
graph is captured over, per batch width.

**Prefill batch**:
The group of queued requests handed to the backend in one prefill call. A call
shape, not a traversal: the runtime walks the group and runs one model
traversal per request.

**Packed prefill**:
Several requests' prefill tokens in *one* traversal of the model (varlen
attention, per-sequence GDN state side by side). Not built; its phase is
decided after the per-chunk synchronization investigation reports.
_Avoid_: prefill batch (for this: naming a group of requests does not make it
one forward pass).

**Prefill/decode interleaving**:
Running a decode round between two prefill chunks of the same sequence of
work, on the one model stream. Nothing is concurrent on the GPU: the two take
turns, so a long prefill costs the decode lanes one chunk of latency instead of
the whole span.

**Decode share**:
The part of the model's time decoding lanes keep while a prompt prefills
(`--model-decode-share`, in percent). It bounds the lanes' rate during a
prefill, not the gap a single chunk leaves: only the chunk width moves that.

**True prefill/decode overlap**:
Prefill and decode resident on the GPU at the same time, on separate streams,
contending for SMs. Distinct from **prefill/decode interleaving** and
deliberately not built: the north-star item (roadmap phase 6), excluded from G3
by ADR 0018.

**Chunk boundary**:
The position between two prefill chunks: the sequence's KV pages, GDN
recurrent slot, conv taps and position have all advanced past the same token,
and the chunk's synchronization has returned. A property of the serving loop,
and what **prefill progress** resumes from — resuming is nothing more than
scheduling the next chunk.

**Snapshot point**:
A position from which the *whole* sequence state may be captured to the host
tier and later restored. Every **chunk boundary** is one for the state that
exists today; the two terms coincide without being the same property, and a
new state section (G4's exact-key side store) must earn the permission again.

**Prefill progress**:
How far into its prompt a request has been prefilled. It makes `Prefilling` a
state a request *lives in* for tens of rounds rather than a moment between two,
so the admission machinery can see a half-prefilled request.

**Decode graph**:
The CUDA graph replayed for a decode round, captured per **exact** batch width
1..8. Widths are never padded up to a captured one: the GDN slot traffic is per
sequence (144 MiB each), so padding a width-1 round to width 8 moves 1.15 GB
instead of 144 MB and spends several times the C=1 gate's whole margin.

**Per-lane sampling**:
Sampling parameters and RNG state carried per sequence, not per decode round:
lanes in one round hold different temperatures, seeds and penalty histories.
Sampling happens device-side in the leaf, which returns token ids: the
**decode** path never ships logits to the host.

**N-lane concurrency**:
The resident decode lanes a load runs at once: 8 (N=8) on a 32 GB **card
class**, fewer on a smaller one, with overflow to the host KV-RAM tier; sized
for a ~10-subagent concurrent coding workload. Real at long context only under
hq-e8-2b: eight lanes at a 40,960-token context are ~20 GiB of BF16 KV against
~3.02 GB of hq (**KV format**).

**DFlash2**:
The 5-layer sliding-window (2048) speculative-decoding drafter (hidden 5120,
draft tokens 1..7, native acceptance 3.4–3.7 tokens/round). Selected at load
(`--spec-backend dflash2 --spec-draft-tokens N`); without it none of the 66
`dflash2/*` objects is bound.

**Draft window**:
The number of tokens the drafter proposes per round (`--spec-draft-tokens`,
1..7), fixed for the life of a load. Not to be confused with the drafter's own
2048-token sliding **drafter window**.

**Drafter window**:
A sequence's DFlash2 context: every drafter layer's BF16 K and V for the last
2048 positions (40 MiB). Per-sequence state in the sequence pool, one lane per
slot, and one **state section** of a pool built with the drafter.

**Feature taps**:
The target's hidden states at layers 5, 19, 33, 47 and 61, concatenated and
projected into the drafter's context. Taken by a prefill chunk and by the
verify traversal; scoped to that chunk or round: never persisted, never a state
section.

**Verify round**:
The speculative form of the **decode round** (P5-04): each lane contributes
`k+1` columns — its anchor (the pending successor) followed by the drafts, the
tail padded with the anchor — and the 64 layers traverse the batch's columns
once, the GQA layers appending every valid column and the GDN layers recording
instead of advancing. Selected per call through the decode options at the
load's draft window, never padded to it; window 0 is today's round.

**Extent (verify round)**:
The draft columns a lane actually verifies in a round: `min(k, proposed,
remaining budget − 1, remaining context − 1)`, where under DFlash2 the drafter
always proposes `k`. Extent 0 is a fallback step inside the same round, one
committed token.

**Committed run**:
The tokens a verify round emits for a lane: the anchor plus the accepted
drafts, cut at the first stop id inclusive (1..k+1). The KV frontier, the GDN
slot and the pending token move by exactly the run, so a sequence never stands
past the text it emitted.

**Prefill outcome**:
What one prefill chunk cost beyond warming its KV (`PrefillOutcome`,
GitHub #192): today the microseconds the leaf spent encoding a media item,
which is 0 on a text chunk and on one that reuses an embedding an earlier chunk
of the same item encoded. One per job, in order, the way a committed run crosses the
seam per lane; the request log sums them into `media.encode_seconds`.

**ReplaySSM record and fold**:
How the GDN layers speculate without advancing: the verify traversal records
each column's conv input, key, value and gates (the **records**), and after
accept the **fold** replays the committed prefix of each lane's records into
its recurrent slot and conv taps in one vendored call. A rejected draft's
transition is recorded and never folded — that is the whole rollback.

**Drafter**:
Whatever fills a verify round's proposal seam: **DFlash2** inside the leaf
(P5-05), proposing each lane's drafts from its **drafter window** in the same
pass that verifies them, or a fake drafter in the substrate's own tests
(`verify-only` load: the verify substrate with nothing bound, the drafts passed
per call).

**Teacher-forced round**:
A **verify round** whose accept verdict is a recorded text's instead of the
target's draw (GitHub #267; test-only, the decode options' `forced_tokens`,
which no serving path sets). The pass is any round's — under **DFlash2** the
drafter proposes from the lane's **drafter window** — but the lane accepts the
leading drafts the recording agrees with, at most its **extent**, and takes the
recorded token after them as its correction, so the **committed run** is the
recording's.

**Proposal readout**:
The drafter's proposals at a sequence's frontier, read without running a round
(GitHub #267, test-only): the drafts, each draft column's 16 candidates, their
proposal-head logits and the selector's transition scores, the lattice the
replay walks other selection rules over. Named after the **readout**'s
precedent — a measurement beside the program, never a second way to generate —
but not one: a **readout** reads named tokens' target logits at one position
and costs a prefill; this reads the drafter's lattice and costs one eager
drafter forward.

**MTP**:
The model's native multi-token-prediction head, the alternative drafter (draft
window 1..7 chosen at load, verification width chosen per round). Flash-Next
drafts with it, its **MTP head** (`--spec-backend mtp`); the 27B drafts with
**DFlash2** and refuses it.

**Vision**:
Multimodal input: images today, video refused until its own spec. A load
option (`--vision-enabled`): without it nothing of the tower is bound or
reserved and text serving is unchanged.

**Vision residency**:
Where a vision load keeps the **vision encoder**'s weights and the **embedding
pool**: **host** (the default: the weights **staged** for each **media
encode**, the pool **host-mapped**), so that vision costs no context while no
image is in flight; or **device** (both in VRAM for the life of the load), for
a card with VRAM to spare on a host short of RAM.

**Media acquisition**:
Turning a request's image parts into prepared **media items** before it is
admitted: decoding a `data:` URI or fetching a URL under the media policy
(private addresses refused unless `--media-allow-private-network`, byte cap,
deadline), then preprocessing, memoized by content digest. A refused image
refuses its request before admission; nothing reaches the scheduler.

**Media encode**:
The prefill step that runs the **vision encoder** over one **media item** and
leaves its **media embedding** in the **embedding pool**, in the first chunk
that reaches the item — unless the pool already holds that item's, in which
case the chunk does no encoding at all. Its wall time is reported on the
request, apart from the prefill's.

**Media item**:
One image in a prompt: its patch grid, the run of placeholder tokens it expands
to, its BF16 patch rows and the digest of the bytes it came from. The unit the
processor prepares, the encoder encodes and a prefill chunk carries at most one
of.

**Vision tokens**:
A media item's *merged* tokens, one per 2x2 block of patches: the placeholder
run's length, and what the per-request envelope (`--vision-max-tokens`) is
counted in.

**Item bound**:
The most **vision tokens** one **media item** may take: the smaller of the
request's envelope and the artifact's pixel bound (16,384 for the 27B), so an
envelope of 32,768 still bounds each image at 16,384. An image over it is
downscaled onto a smaller grid, not refused (#248).
_Avoid_: envelope (the request's total).

**Media embedding**:
A media item's encoder output, `[hidden, vision tokens]`, leaf-owned. **Held** by a request from its encode until the item's last
placeholder is prefilled, and **resident** in the **embedding pool** past that,
until something needs its room: a fan-out of N questions over one image
encodes it once (ADR 0035).

**Embedding pool**:
The load's one reservation for **media embeddings**, where the **vision
residency** puts it (`--vision-embedding-pool-mib`, defaulting to one
envelope-wide item), carved
into fixed-width **column pages** so an item takes the pages its own columns
need rather than a slot sized for the envelope. An embedding is contiguous in
column space and not in memory: the encoder writes it a page at a time and a
prefill chunk scatters it back a page run at a time, with the same ops either
did over one flat buffer.

**Vision encoder**:
The 27-block tower plus the 2x2 merger that turns patch rows into a media
embedding. Run once per item by the **media encode** step, never per chunk.

**Rope delta**:
`max_position + 1 - prompt_length` for a multimodal prompt, whose three-axis
positions advance more slowly than its tokens. Every decode round after such a
prompt rotates at `position + rope_delta`; the position itself stays the KV
index and the sampler's key.

### Decisions

**Decision**:
One question put to the model about supplied evidence, whose answer is a
distribution over a declared set of options rather than generated text. It is
the unit a decision request is made of: one evidence, many decisions.

**Readout**:
Reading the logits of named tokens at one position instead of sampling a token
there. A readout costs one prefill and generates nothing.

**Answer token**:
The single token that stands for one option of a **decision**, and whose logit
the **readout** reads. Nothing to do with a decode **lane**, a **GDN slot** or a
**retained slot**, which are all per-sequence device resources; an answer token
is a vocabulary entry.

**Answer alphabet**:
The ordered pool answer tokens are drawn from: single characters first
(`A`-`Z`, `a`-`z`, `0`-`9`), then uppercase bigrams. It is **computed from the
loaded tokenizer, never compiled in**: a label is admitted only if it encodes to
exactly one token that decodes back to itself, which rejects 114 of the 676
uppercase bigrams in the 27B's tokenizer.

**Constrained decode**:
A decode whose sampling is restricted, per lane, to a declared set of tokens.
What a **readout** is to one position, this is to a run of them: the answer's
shape is the caller's, so no parser has to survive a shape the model chose.

**Permitted set**:
The vocabulary entries one step of a **constrained decode** may draw from, at
most 32 of them. The leaf drives every other column of the logits row to
`-1e30` before the vendored sampler runs, so the constraint composes with
temperature, top-k, seeds and penalties instead of replacing them, and what
comes back is a token id like any other.

**Schedule**:
One **permitted set** per token a **constrained decode** may emit, in order. It
is that request's whole generation budget, and its whole stopping condition
**unless it names a terminator**: the run ends when the schedule is spent, with
`stop`, and never on EOS, because a token drawn from a set of digits cannot be
one.
_Avoid_: scheduler (for a schedule: it schedules alphabets, not requests).

**Terminator**:
A token a **schedule** names, permitted by its steps like any other, that ends
the run when it is drawn. Not EOS by another name, and the difference is whose
decision it is: EOS is the model saying it has nothing more to say, while a
terminator closes the *shape the caller asked for*.

**Draw**:
One token a **constrained decode** emitted, and its probability *within that
step's **permitted set*** — a restricted softmax over at most 32 logits,
computed on the device and temperature-free on purpose: it is the model's
confidence in the token, not the chance the sampler happened to pick it. A
run's draws, in order, are its **trace**.

**Forced literal**:
Tokens a request is made to generate, one **permitted set** of one per round,
after which it is **released** and generates freely: not a **schedule**, which
ends its run when spent. `tool_choice: "required"` or a named function forces
the opening of a tool call this way (GitHub #286), from the first token with
thinking off and right after the reasoning block with it on; the thinking
budget's forced close is the same shape.

**Answer mass**:
How much of the whole next-token distribution the answer tokens hold,
`exp(logsumexp(answer logits) - logsumexp(all logits))`. It is what says
whether a restricted softmax is reading the model or renormalizing noise: a
**decision** whose answer mass is near zero has an argmax that means nothing.

**Attention readout**:
Reading attention heads' pre-softmax scores (`q · k / sqrt(head_dim)`) at a
prefill's last position over one span of its keys — an image's placeholders, or
a text state's tokens — instead of the logits of answer tokens
(GitHub #260, #263, #264 and #275; ADR 0038, ADR 0039, ADR 0040 and ADR 0041).
A `point`, a head `box` or a **locate** answered this way is a **decision**:
one prefill, no decode round, no lane.

**Pointing head**:
The attention head that names *which* object a `point` or a head `box` is
about: L39.h10 (GQA ordinal 9, query head 10) on the served 27B, read at the
position after the forced `{"x":`. A **calibrated constant keyed to the
artifact's content hash**: chosen on labelled scenes by cross-validation, never
computed at load, and never read on an artifact nobody calibrated it for — such
a load answers `point` with the digit chain.

**Head set**:
The attention heads read together beside the **pointing head**, one image cell
each: where that head looks hardest, the **fallback cells** skipped
(GitHub #263, ADR 0039). Each reads a different part of the object the question asked
for — its boundary, mostly — so the cells near the pointing head's point
outline the object.

**Fallback cells**:
The image cells attention heads peak on when nothing in the image matches the
question: the first and the last image cell on every grid, and on a grid the
calibration measured, the cells it found there (on 32x32 also (0,1) and
(6,31)). The **head set**'s argmax skips them; the **pointing head**'s map
keeps them, and a map flat enough to fall back to one (`region.share` about
0.03) is nothing found.

**Sub-cell peak**:
Where a **head set**'s head looks hardest *inside* its image cell, not just
which cell (GitHub #264, ADR 0040): parabolic interpolation over its score at
its peak and the two in-line neighbours on each axis, in cells, zero when a
neighbour is off the grid or the triple is not a strict peak. It is what lets a
box frame an object smaller than the cell grid can express — a 1024 px button
is 1.44 cells tall, and a box on cell geometry cannot clear the digit chain at
that size.

**Allowance**:
What the **extent** grows the kept heads' span by on each side: `0.05 + 4.5 /
n` cells, `n` the distinct cells they peak on (never under three, the fewest
measured). The span is a *lower bound* on the object — it reaches only the
parts the heads hit — and what one part leaves unobserved falls as more parts
are seen: on synthetic scenes the set tiles the object (17-56 distinct cells),
on a photograph it piles onto one distinctive part (6-13).

**Locate**:
The `/v1/decide` primitive that names the **segment** of a text `state` an
instruction asks for (GitHub #275, #278; specs 18 and 22; ADR 0041, ADR 0042),
with nothing written into the state and nothing generated. Two methods: the
**shortlist** (the default) and the **head vote**.

**Shortlist**:
A **locate**'s default method (GitHub #278, ADR 0042): the calibrated heads
narrow the text to a few candidates — 5 templates at a fold's **level 1**, 16
rows at its **level 2**, 16 lines, sentences or records otherwise — and a
labelled `choice` over them decides, with no token generated. The heads narrow
a very long text; the `choice` resolves near-duplicates, which it does well
among a handful and badly among hundreds.

**End reading**:
A **shortlist**'s reading of logs and records: a segment scored by where it
closes — its last key, its separator and the next segment's first key — each
head's lift over the **content-free baseline** standardized over the window's
segments and summed over the **end heads**. At length a line is marked at its
end, not its start.

**Sum reading**:
A **shortlist**'s reading of prose: a segment scored by all its keys, as the
**head vote** does, standardized and summed over the **sum heads** rather than
voted.

**End heads**:
One of the two head sets of a `locate` calibration (spec 22): 32 heads of GQA
layers 35 to 55 with the best single-head top-1 at a line's closing keys on
development sets (spec 23). Keyed to the artifact's content hash with
`LOCATE_WINDOW_KEYS`, as the **sum heads** are.

**Sum heads**:
The other head set of a `locate` calibration (spec 22): the **head vote**'s 32
heads.

**Template fold**:
A `locate`'s `template_fold` compression (spec 22): a log's lines grouped
Drain-style into templates — same source label, same token count, tokens
agreeing on half the positions, times removed and obvious variables masked — so
a long log is read as a short text first. Reversible: every **level-2** row
maps back to the lines it stands for.

**Level 1**:
A **template fold**'s first text: one line per template, its variable slots
showing their distinct values while they fit 600 characters, and `(xN)`.

**Level 2**:
One template's lines as their values only, values first and the line's time
last, exact repeats one row with `(xN)`. The **shortlist** ranks rows by their
values and shows the last `choice` each row's first original line, because
whether a line answers cannot be read from its values alone.

**Window**:
A stretch of at most `LOCATE_WINDOW_KEYS` (200,000 on the served 27B) keys of a
text the heads read, cut at segment boundaries — at an empty segment where one
is in reach (a paragraph break), else before the segment that would not fit —
and read as its own prefill with its own **content-free baseline**; windows'
scores are standardized and merged. A text's length is then bounded by the
request, not the context.

**Pointer**:
Every candidate of a `locate`'s last `choice` at a share of 0.05 or more, the
pick always first: one answer, or the several segments a two-part answer needs.

**Records array**:
A JSON array of two or more elements, every one an object, and not the
content-parts shape (every element with a string `type`): what `auto` reads as
`records`, without a statistic.

**Found**:
Whether a `locate`'s text answers at all (spec 22 § Not found): the last
`choice` asked again with a "none" option, and for a folded log a yes/no; `1 -
p(none)`, averaged with `p(yes)` for a log. Below 0.5 the answer names no
segment and keeps its ranking.

**Segment**:
One place a **locate** can answer with: a line of a string state, split on `\n`
exactly (a `\r` stays part of its line), or an element of an array, `within` an
optional JSON Pointer. Numbered the way the caller would number them — the
0-based index of `split("\n")`, the array index.

**Head vote**:
How a **locate** is read (spec 19's track L, GitHub #276): each of 32 heads of
GQA layers 35 to 63 names the **segment** whose share of its softmax over the
span the question raised most above the **content-free baseline**'s, and the
most-voted segment wins, a tie to the segment the best-ranked head named. A
**calibrated constant keyed to the artifact's content hash**, like the
**pointing head**: the heads with the most training hits on development sets,
never computed at load.

**Content-free baseline**:
The same prompt with the instruction replaced by `N/A` (ICR's, QRHead's and
contextual calibration's "N/A"): what the heads do with the state and the
scaffold when nothing is asked — sinks, first lines, position — which the
**head vote** subtracts before a head votes. One per **locate** target in a
request, shared by every `locate` over it; it claims the state's prefix like
any sibling, and costs input tokens but is no **decision** of its own.

**Extent (head box)**:
The box the kept heads of a **head set** span: the heads within three times the
median distance of the **pointing head**'s point, at their **sub-cell peaks**,
their 15%-85% quantiles on each axis grown by the **allowance**, clamped to the
image. A head `point` is its centre and carries it as `extent`; a head `box` is
it.

### The Responses API

**Stream**:
One of the independent request sequences a Responses WebSocket multiplexes
(OpenAI's `stream_id`; the connection's **default stream** when it names
none). Its requests run in order and never overlap; different streams run side
by side.
_Avoid_: lane (for a stream: a lane is a decode slot, and the stream is the
standard's word, nothing in the engine).

**Continuation cache**:
What one Responses WebSocket remembers so a request can name
`previous_response_id` and send only new items: per **stream**, its latest
finished response and the whole conversation that response saw. It chooses
nothing about retained state: a continuation re-renders that conversation, and
reuse stays a content match on the **prompt checkpoint** and **retained
prefix** (ADR 0029).

**Warm-up request**:
A request that ends where its prefill ends and generates nothing, like a
**decision**, but reads nothing out and keeps what an ordinary request of the
same prompt keeps: the **retained prefix** at its system block and the **prompt
checkpoint** at its generation opener, for the turn that follows. The Responses
WebSocket's `generate: false`.
_Avoid_: prewarm (for the request kind; it is the client's word for the act).

**Socket admission queue**:
The one server-wide first-in, first-out queue a Responses WebSocket request
waits in when the engine is full, told so with `response.queued`, instead of
being refused with a 503 as an HTTP request is. Only its head retries, each
time a request leaves the scheduler.

**Idle socket**:
A Responses WebSocket with nothing running, queued or waiting on it and no text
or binary frame from its client since its last response ended; a ping does not
make it busy. One idle for `server.socket_idle_close` is closed by the server
(GitHub #329), since an open socket holds a Knative pod.

### Configuration

**Field**:
One configurable setting, declared once (`config::schema`): its flag
`--<group>-<field>`, env var `IGNIS_<GROUP>_<FIELD>`, config-file key
`<group>.<field>`, default, rule and help text all come from that one
declaration. Fields are grouped by concern (`server`, `model`, `vram`, `reuse`,
`switch`, `spec`, `vision`, `media`, `download`, `ngram`, `kv_disk`).

**Family scope**:
A field both model families take but want sized differently has a variant per
family (`qwen38`, `qwen38flashnext`, from `ModelFamily`, never a served id):
`--qwen38flashnext-reuse-kv-host-pool-bytes`. A tiebreaker *within* each
source, not a source of its own: the command line wins whatever its scope.

**Applicability**:
The families a field means anything for (`Only(Flash-Next)` for the expert
cache's floor, `Only(27B)` for vision). An explicit value for a family that
cannot take it is refused at start and dropped, with a log line, on a switch;
a value equal to the field's default is neither.

**Card class**:
The device memory a profile is shaped for: 32 GB (RTX 5090), 96 GB (RTX PRO
6000), 16 GB (RTX 5060 Ti / 5070 Ti / 5080). It decides which artifact, how
many lanes and what context the profile defaults to.
_Avoid_: GPU tier.

**Profile**:
A named bundle of hardware-shaped field defaults (`--profile`, built-in
`rtx50-16gb`, `rtx5090` or `rtxpro6000`, or a config file's `profiles:`):
below the config file, above the hardcoded defaults — a default, never an
override. `auto`, the default, picks the built-in profile of the card's **card
class** from NVML's total memory (ADR 0063); a named profile always wins.

**Adaptation**:
A 27B load lowering the values its profile or the hardcoded defaults gave, in a
fixed order, until its VRAM plan fits (ADR 0063): the drafter's `shortlist`
head to `full`, then the context down its **ladder**, then one decode lane.
Only a plan refused below its minimum adapts; one that fits as configured
changes nothing.

**Ladder**:
The contexts an adaptation plans, from the configured one down every multiple
of the context step (`model.adapt_context_step`, 65,536) to one step, keeping
the largest whose KV pool holds one full sequence and a spare reservation
(`vram.adapt_spare_tokens`, 65,536) at more than one lane.
_Avoid_: step (bare) for one of its rungs: an adaptation taken is an
adaptation step.

**Live change**:
A `PATCH /v1/config` that touches only server-level values (timeouts, message
policies, thinking defaults, switch knobs) and no model.

**Reload**:
A `PATCH /v1/config` that runs the whole patch through the model switch,
reloading the same model with the patched configuration. One patch is a live
change or a reload, never both.

### Observability

**Canonical event**:
The one structured representation of a log occurrence (severity, event name,
body, attributes, trace context). Formatters are presentation, never the
source of truth (ADR 0011).

**Event name**:
A stable, class-level identifier (`ignis.<subsystem>.<event>`, e.g.
`ignis.model.loaded`), never occurrence-specific data.

**`request.id`**:
Reused as the OTel `trace_id` for the request's span tree (root span at HTTP
ingress, children per admission/prefill/**decode round**/MTP
verify/completion); one identifier for the same causal unit across fairness,
logs, and spans (ADR 0012).

**Hot-path logging constraint**:
Normal INFO operation must not emit one record per
token/layer/kernel/allocation on the prefill/decode path; any logging or
tracing change touching that path must pass the G4 performance gate (ADR
0007) before merge, not just design review.

**Hotpath lint**:
The grep-based, hand-maintained scan over a fixed list of
prefill/decode/CUDA-graph-adjacent files that fails `cargo test` on any
INFO-or-above `tracing` call or raw `println!`/`eprintln!`/`print!` found
there.

**Monitor**:
The Playground's live dashboard over the Prometheus exposition (GitHub #165):
the browser
scrapes `/ui/metrics`, keeps a rolling in-memory history, and derives rates,
latency quantiles and a plain-words verdict from ADR 0017's contract metrics
only. Offered only when metrics are on; not a replacement for Prometheus, and
gone on reload.

### Playground

**Playground**:
The opt-in browser page ignis serves for working with its models by hand —
chat, pictures, speech and **workspaces** — with per-request figures the
browser measures itself, and the **Monitor**. The user's workbench over the
engine: not an operator surface (that is Prometheus), and never a source of
facts the engine does not already expose.

**Workspace**:
A named set of **workspace files** the user and the model work on together,
kept by the page and never by ignis. Born with its first file, it outlives the
chat that made it: a chat works in at most one at a time, its own or one the
user picked.
_Avoid_: project, library, artifact (the `.ninfer` container)

**Workspace file**:
One named item of a **workspace** — text, a picture or audio — that the model
and the user both read and change; "file" alone when the workspace is clear.
_Avoid_: artifact, document

**Memory note**:
A note the Playground keeps across every chat, a title and a body: the model
sees the titles and reads a body when one looks relevant. Never part of a
**workspace**.
_Avoid_: memory (bare: that is the device's), workspace file

**Turn notice**:
The **instruction message** the page adds after the history at a turn: at a
chat's first turn the titles of the **memory notes** and the files of its
**workspace**, later only what someone else changed since, and the moment when
it is live. Kept with its turn once sent, so the **system prompt** never
changes.
_Avoid_: context note

### Acceptance

**Gate**:
A milestone acceptance measured **on the GPU**, never on CPU tests. G1
correctness floor, G2 prefill, G3 decode + serving loop, G4 the reference
feature floor, G5 speculative decoding (`docs/ROADMAP.md`).

**GPU profile**:
The explicit test profile for GPU work: it requires the 5090 free and
**fails**, never skips, when the GPU is busy or a kernel errors.

**Generated token**:
A token the engine decoded, on **either** channel: the answer
(`delta.content`) or the thinking trace (`delta.reasoning_content`, on by
default). Every throughput and latency cell counts generated tokens, because
both channels are the same decode work; a correctness check reads the answer
channel alone, because scratch work is not an answer.

**Canary suite**:
The fixed set of short, high-signal prompts used to detect divergences.
Sanity is judged on everything the turn generated: a turn whose budget ran out
mid-thought produced sane tokens, not empty output.

**Canary oracle**:
The recorded fixture the canary suite is checked against: the reference
engine's greedy (exact-argmax) completions on those prompts for the same
artifact, tokenized with the artifact's tokenizer.

**Teacher-forced agreement**:
How the canary oracle is scored, and the G1 correctness floor: at each of the
first 32 positions the engine is fed the *oracle's own* token prefix, and its
greedy next-token argmax is compared with the oracle's recorded token. The
floor is ≥ 95% overall (ADR 0014).

**Free-running agreement**:
The older scoring, where the engine generates its own continuation and the two
token streams are diffed. **Diagnostic only** (ADR 0014): after the first
divergence the engines no longer share a prefix, so it measures continuation
similarity, not forward-pass health.

**TTFT cell**:
One fixed prompt length (8K, 32K) at which time to first token is measured
through the bench, the same way against either engine: streaming, greedy,
thinking off, a small output budget, deterministic synthetic prompts of
exactly that many tokens. Every sample is a **cold prefix**: each sample (the
warmup included) gets its own prompt, distinct from the first content token,
so no prefix cache or host KV tier on either engine can serve it; the engine's
own computed-prefill-token count must equal the prompt length or the sample is
void.

**ITL measurement window**:
The stretch of a G3 ITL cell in which *every* decode lane was alive: it opens
at the last lane's first token and closes at the first lane's end, and only
the prefill windows closing inside it are pooled against. A lane the harness
cancelled ends at the cancellation, since it was generating until then; a lane
the engine ended — its own EOS, the safety cap — ends at its last token.

**Live/live gate**:
How G2 is judged: ignis and the reference measured on the same TTFT cells in
the same session on the same machine, and the ratio ignis/reference must be
≤ 1.5. A committed reference record exists for regression and sanity only; it
never decides the gate.

**Measurement session**:
The identifier both engines' TTFT records carry when they were measured back
to back in one sitting. It is what makes live/live checkable rather than
asserted: `ignis-bench g2` refuses a verdict when the two records do not share
one, which is also what stops a committed fixture from ever being the live
side.

**TTFT record**:
One engine's measured cells plus the identity a later reader needs to audit
them: every sample and its computed-prefill count, the median, the endpoint,
the engine, the artifact, the profile, the date and the measurement session
(`ignis-bench ttft --out`).

**f64 layer reference**:
A CPU fp64 computation of one GQA layer and one GDN layer on the real weights,
the tolerance target for the leaf's per-layer output at G1.

**Performance gate (99%)**:
The acceptance criterion at G4: ≥ 99% of the reference's performance
(throughput / latency) on the trace-replay load, **not** token-agreement;
correctness is self-checked (sane output, same model, greedy, fixed seed). It
is the first of a ladder of performance gates: G5 asks ≥ 99% of the
reference's committed tok/s under DFlash2, and the phase beyond the reference
gates each item on its own (`docs/ROADMAP.md`).

**Trace replay**:
Re-sending a recorded "1 main agent + N subagents" load trace against both
engines to compare scheduler behavior, scored per **lane tag**. The trace is
the load, not the baseline: the reference's own run is measured live in the
same session (**live/live gate**), over at least two launches per engine (ADR
0021), and the verdict is the pooled cell statistic.

**Load trace**:
The recorded agent session a **trace replay** sends. It holds real working
content, so it is never committed: its SHA-256 and its shape (request count,
class split, arrival span, prompt-length distribution) ship with the gate
artifacts, and both run records carry the hash so a reader can prove the two
engines replayed the same load.
