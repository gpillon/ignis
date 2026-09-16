# 01 — Vision: image input on `/v1/chat/completions`

GitHub: #174

## Problem Statement

ignis is text-only. A coding agent that wants to show the model a
screenshot, a rendered UI, a diagram or a failing chart — the everyday
multimodal moves of an agentic session — has no way to do it: the wire
`ChatMessage.content` is a plain string, an OpenAI content-part array is
rejected at deserialization, and nothing below the HTTP layer knows what an
image is. The artifact already carries the whole Qwen 3.8 vision tower
(`vision/*` objects, Q4/Q5/Q6 row-split + BF16) and the frontend already
ships `preprocessor_config.json`, the `<|image_pad|>` / `<|vision_start|>`
tokens and a chat template whose vision branches match content parts — but
the text-scope binder leaves every vision object unbound and the reference
is the only engine on the owner's card that can answer a question about an
image. Switching the agents' backend to ignis today means losing vision.

## Solution

Image input, end to end, the reference's way, behind a load option:

- `ignis serve --vision` binds and materializes the vision tower and
  reserves its fixed device workspace at load. Without it nothing vision-
  related is bound or allocated, text serving is byte-for-byte today's, and
  a request carrying an image is a 400 `vision_disabled`.
- `POST /v1/chat/completions` accepts OpenAI content parts on `user`,
  `assistant` and `tool` messages: `{"type":"text"}` and
  `{"type":"image_url","image_url":{"url": ...}}`, where `url` is a base64
  `data:` URI or an HTTP(S) URL. Plain-string content keeps working
  unchanged.
- The server acquires the bytes, the frontend decodes and preprocesses each
  image exactly as the reference processor does (smart resize to the
  32-pixel merge factor, antialiased bicubic, `v/127.5 − 1`, BF16 patch rows
  of width 1536), expands each placeholder into its merged-grid run of
  `<|image_pad|>` tokens, and assigns three-axis (temporal/height/width)
  positions plus the prompt's `rope_delta`.
- During prefill the leaf runs the 27-layer vision encoder and 2×2 merger
  once per image, scatters its `[5120, V]` output over the placeholder
  columns, and rotates the GQA layers with MRoPE. Decode continues with
  scalar positions offset by the sequence's `rope_delta`. Vision state is
  never retained after the item's last placeholder is prefilled.
- Everything that already makes ignis an agentic engine keeps working on a
  multimodal request: chunked prefill with prefill/decode interleaving,
  N-lane decode, DFlash2 speculative decoding, prefix reuse, KV-RAM, tool
  calling, thinking.

The client sees the reference's contract: same wire shape, same error
codes, the image's tokens counted in `usage.prompt_tokens`.

## User Stories

1. As a coding agent, I want to send a screenshot as an `image_url` content part, so that the model can read an error dialog I cannot copy as text.
2. As a coding agent, I want to send an image as a base64 `data:` URI, so that I never have to host a local file on a web server.
3. As a coding agent, I want to send an image by HTTP(S) URL, so that I can point the model at a published asset without downloading it myself.
4. As a coding agent, I want several images in one message, so that I can show a before/after pair.
5. As a coding agent, I want images in earlier turns of the history to stay part of the conversation, so that I can ask follow-up questions about a screenshot I sent three turns ago.
6. As a coding agent, I want a `tool` message to carry an image (e.g. a browser tool's screenshot result), so that a tool's visual output reaches the model the same way its text does.
7. As a coding agent, I want text and image parts interleaved in any order inside one message, so that the prompt reads the way I wrote it.
8. As a coding agent, I want plain-string `content` to keep working exactly as today, so that none of my text-only requests change.
9. As a coding agent, I want the answer to an image question to stream with the same reasoning/content split and tool-call parsing as a text request, so that my client needs no special case.
10. As a coding agent, I want `usage.prompt_tokens` to include the image's expanded tokens, so that my context accounting stays honest.
11. As a coding agent, I want a clear 400 when the expanded prompt does not fit the context, so that I can downscale or drop an image instead of guessing.
12. As a coding agent, I want a clear 400 `vision_disabled` when the server was started without `--vision`, so that I know it is a server configuration, not a malformed request.
13. As a coding agent, I want a clear 400 when an image cannot be decoded (corrupt bytes, unsupported format, extreme aspect ratio), naming the offending part, so that one bad attachment is diagnosable.
14. As a coding agent, I want a `video_url` part to be refused with a clear 400, so that I am never silently answered as if the video were not there.
15. As a coding agent, I want an image in a `system` message to be refused with a 400, so that I learn the template does not render it there rather than having it dropped.
16. As a coding agent, I want a URL fetch that times out or returns an HTTP error to be a 400 naming the fetch failure, so that I can tell a network problem from a model problem.
17. As a subagent sharing a system prompt with my siblings, I want prefix reuse to keep working when the shared head contains an image, so that a burst of siblings does not re-encode and re-prefill the same screenshot N times.
18. As a subagent, I want two different images of the same size never to be served each other's prefix, so that I am never answered about a picture I did not send.
19. As an agent resuming an evicted sequence from KV-RAM, I want the restored sequence to continue decoding at the right MRoPE positions, so that a multimodal conversation survives eviction.
20. As an agent on a DFlash2 load, I want speculative decoding to keep its acceptance on the text that follows an image, so that vision does not cost me the drafter.
21. As an interactive user decoding in another lane, I want an image encode to cost my lane at most one bounded stall, so that someone else's screenshot does not freeze my stream.
22. As an agent re-sending the whole history every turn, I want an image the server has already preprocessed to skip decode and resize, so that a long multimodal session does not pay the CPU preprocessing again on every turn.
23. As the owner, I want `--vision` to be a load option fixed for the life of the load, so that the device layout, KV byte budget and graphs are decided once, as every other load option is.
24. As the owner, I want a text-only load (no `--vision`) to allocate nothing for vision and to bind no vision object, so that the text serving profile I gate against is untouched.
25. As the owner, I want the startup capacity line to show the vision reservation and the KV budget left after it, so that I can see what vision costs me in lanes and context.
26. As the owner, I want the HTTP(S) fetch to refuse private, loopback and link-local addresses by default, so that an `--expose`d server cannot be used to probe my LAN.
27. As the owner, I want an opt-in to allow private-network image URLs, so that a local agent can reference an image served on my own machine.
28. As the owner, I want per-request byte, decoded-pixel and vision-token budgets enforced before any device work, so that one oversized image cannot exhaust host memory or the vision workspace.
29. As the owner, I want the request log to record the number of images, their vision tokens, preprocessing time and encode time, so that a slow multimodal TTFT is attributable.
30. As the owner, I want a cancelled request to stop preprocessing and release any encoded vision output, so that a disconnected client does not keep burning CPU or VRAM.
31. As the owner, I want the vendored vision ops tracked by the manifest like every other vendored op, so that their provenance and patches stay auditable (ADR 0010).
32. As the owner, I want the processor's output (token ids, positions, `rope_delta`, patch bytes) to be exactly the reference processor's on the same image, so that a divergence in vision answers is never a preprocessing bug.
33. As the owner, I want a vision canary with a teacher-forced floor against the reference's greedy recording, so that a wrong layout, wrong merge order or wrong MRoPE axis is caught as a gross error on the GPU.
34. As the owner, I want one live/live multimodal TTFT measurement against the reference, so that a large vision performance error is detected (an error detector, not a gate).
35. As the owner, I want the snapshot blob to carry the sequence's `rope_delta`, versioned, so that a stale blob from a text-only build is refused rather than restored at wrong positions.
36. As a maintainer, I want vision vocabulary (media item, vision tokens, rope delta, vision encoder) in `CONTEXT.md`, so that later tickets speak one language.

## Implementation Decisions

### Scope and load option

- Images only. `video_url` parts are refused with 400 `video_unsupported`
  (the reference's video path needs an FFmpeg demuxer and frame sampler;
  deferred, see Out of Scope). The processor's data model still carries a
  temporal grid axis so video is an additive change later.
- `--vision` is a model-load option (CLI flag and config key), fixed for the
  life of the load, like the KV format and the draft window. It changes the
  binder's scope: the `vision/*` objects are bound and materialized with
  the text scope; without it they remain unconsumed under `Binder::plan`
  exactly as today.
- The vision reservation (weights + encoder workspace sized for the vision
  envelope + the per-item output transient) is taken before the KV pool's
  byte budget is derived, so enabling vision reduces derived KV token
  capacity rather than risking a later OOM. The startup capacity event
  reports it.
- Vision envelope defaults to the reference's: 32,768 merged vision tokens
  (131,072 raw patches) per request, and the effective limit is
  `min(max_context, envelope)`. It is a load option (`--vision-max-tokens`)
  because the workspace it sizes is VRAM taken from the KV pool; the GPU
  acceptance step records the reservation at the default and the owner may
  lower it.

### Wire contract (server)

- The wire `ChatMessage.content` becomes string-or-parts. Parts:
  `text` and `image_url` (object with `url`; the OpenAI `detail` field is
  accepted and ignored). `video_url` → 400 `video_unsupported`. Any other
  part type → 400 naming it. `image_url` on a `system` message → 400
  (reference posture: the template does not render media there).
- Media sources: `data:<mime>;base64,<payload>` and `http`/`https` URLs.
  Credential-bearing URLs are refused. Fetch policy mirrors the reference:
  resolve first, refuse private/loopback/link-local/multicast/CGNAT
  addresses unless `--media-allow-private-network`, pin the connection to
  the resolved address (no DNS rebinding window), bounded redirects each
  re-checked, connect and total timeouts bounded by the request, byte cap
  enforced while streaming, no proxy.
- Error codes follow the reference's names, all HTTP 400:
  `vision_disabled`, `video_unsupported`, `invalid_media` (undecodable,
  malformed data URI, bad aspect ratio > 200), `media_budget_exceeded`
  (bytes, decoded pixels, raw patches, vision tokens), `media_fetch_failed`,
  `media_fetch_timeout`, and the existing `ContextExceeded` 400 (#166) for a
  prompt that no longer fits after expansion. Errors are raised before the
  request is admitted to a lane.
- Acquisition, decode and preprocessing run on a bounded host worker pool
  off the async runtime and off the model thread (ADR / spec 05 model-thread
  isolation stays intact), honour cancellation and the request deadline,
  and drop the source bytes once patches are built.

### Processor (artifact frontend)

- A new frontend entry point turns messages + media bytes into a
  **prepared prompt**: token ids, per-token modality type, three-axis
  positions (axis-major `[3, T]`), `rope_delta`, and one media item per
  image (grid `t,h,w`, token span, patch rows, SHA-256 content digest of the
  acquired bytes). The existing render-then-encode path is the special case
  with no media (positions equal on all axes, `rope_delta = 0`).
- The algorithm is the reference processor's, step for step, so the output
  is exactly comparable:
  - decode to RGB8 (PNG, JPEG, WebP, GIF first frame; EXIF orientation
    applied); aspect ratio ≤ 200;
  - `smart_resize` with factor 32, round-half-even to the factor, then
    rescale into `[min_pixels, max_pixels]` read from the artifact's
    `preprocessor_config.json` (no hard-coded pixel limits);
  - separable antialiased bicubic (Keys, a = −0.5, torchvision/Pillow
    support rule), u8 intermediate with round-half-even clamping;
  - normalization LUT `v/127.5 − 1` stored as BF16 with round-to-nearest-even;
  - patches channel-major `3 × 2 × 16 × 16` (the image frame repeated to
    form the temporal pair), emitted in 2×2 merge-block order
    (block_y, block_x, merge_y, merge_x) — the order the merger's
    `[4608, V]` view assumes.
- Placeholder expansion happens on the rendered text, before tokenization:
  the i-th `<|image_pad|>` becomes `(gh/2)·(gw/2)` copies; order mismatch or
  a leftover placeholder is an error. Byte offsets the frontend already
  tracks (thinking/rewrite boundaries, shared-prefix boundary) are shifted
  across the expansion, and a boundary landing inside a placeholder is a
  logic error. The image and video pad token ids are validated against the
  tokenizer at load (248056 / 248057), failing the load on a mismatch.
- Positions: text runs advance all three axes together; an image run takes
  `(current, current+y, current+x)` over its merged grid and advances
  `current` by `max(gh/2, gw/2)`; `rope_delta = max_position + 1 − T`.
- A host **media cache**: prepared patch payloads keyed by
  (content digest, modality), LRU under `--media-cache-mib` (default 1024,
  0 disables), with single-flight for concurrent identical misses. A cache
  hit skips decode/resize/pack but never skips the digest.

### Compute seam (core ↔ runtime ↔ leaf)

- The request the scheduler holds gains an optional multimodal part: the
  three-axis positions, `rope_delta` and the media items. Text-only
  requests carry none and take today's path unchanged.
- Chunk planning: a prefill chunk contains at most one media item's
  columns; a media item may span several chunks (the encoded output is
  kept until its last column is prefilled). A **publish point** never
  splits a media item's token span. Interleaving stays one chunk per call.
- **Step ABI** (ADR 0009 / 0016 — options grow, entry points do not
  multiply without cause):
  - a new step, *media encode*: takes one item's host BF16 patch rows plus
    its grid (host inputs, like token ids — no host activation crosses the
    ABI) and returns an opaque, leaf-owned, device-resident **media
    embedding** handle `[5120, V]`; released explicitly;
  - `ignis_prefill_options` grows the multimodal fields: the span's
    three-axis positions, and an optional (media embedding handle, first
    column, scatter indices) triple for the span's placeholder columns;
  - the sequence handle gains its `rope_delta`, set at alloc/prefill,
    applied by every decode and verify round as `rope_position =
    position + rope_delta`. The sequence's *position* (KV index, RNG key)
    is unchanged.
- The leaf **program** gains the vision encoder, built on vendored ops:
  patch projection + bias, bilinear 48×48 position-table add, 27 blocks of
  LayerNorm → QKV + bias → 2-D vision RoPE → segmented attention
  (`cu_seqlens` per image) → projection + bias + residual → LayerNorm →
  fc1 + bias → GELU-tanh → fc2 + bias + residual, then merger LayerNorm →
  2×2 view → fc1 + bias → GELU-exact → fc2 + bias. Its per-item control
  (patch position ids, `cu_seqlens`, bilinear table indices/weights,
  scatter indices) is computed host-side in Rust and passed with the
  encode call.
- GQA layers accept MRoPE positions (`[T, 3]`, pair i uses axis i mod 3)
  through the already-vendored `qk_norm_rope`/`rope` text MRoPE mode; a
  text-only span keeps the 1-D route so the text hot path does not change.
  Decode graphs stay address-stable: `rope_delta` is read from per-row
  device staging like the position itself (ADR 0019/0020).
- Vision weights bind natively as device views in their stored row-split
  Q4G64/Q5G64/Q6G64 (F16 scale) and BF16 formats, the reference's storage
  and kernels — not the host bf16 dequant `normalize` offers today
  (performance-first, ADR 0005). Q6 row-split storage and every missing
  vision op (`vision_attention`, `vision_pos_embed`, `layer_norm`, `gelu`,
  `add_bias`, `scatter`, vision RoPE frequencies, the Q4/Q5/Q6 row-split
  linear kernels) are brought in through the vendor manifest at the pinned
  reference commit, with their reference tests (ADR 0010).
- **DFlash2**: the feature taps on a multimodal prefill span come from the
  target's hidden states as usual; the drafter consumes the same positions
  the reference passes its context append on a multimodal span (the
  implementer checks the reference's DFlash2 multimodal prefill sink). No
  vision-specific drafter state.
  - *Checked, #195.* The reference's prefill sink captures the span's
    **cache** positions (`tap.capture_positions(positions, ...)`,
    `impl/runtime/text_context_impl.h`), not its rope positions, so the
    drafter needs nothing vision-specific — as above. The verify round is a
    **deliberate departure**: the reference's DFlash2 round passes its
    proposal positions for both the cache and the rotation
    (`impl/runtime/dflash2_impl.h`'s `TargetVerifyFrameView`), and its
    `DFlashDecodeIngress` carries no rope delta at all, while its ordinary
    decode batch adds `sequence.rope_delta` and its MTP round carries one
    per lane. ignis applies the delta on every round, as the sequence-handle
    bullet above requires; that consistency is also what makes greedy
    spec-on equal spec-off on the text after an image.

### Prefix reuse, KV-RAM, snapshots

- Prefix identity becomes token ids **plus** the ordered media items fully
  inside the prefix (content digest, grid). A claim matches only if both
  agree; a prefix cannot end inside a media item's span. Two same-size
  images therefore never share a prefix.
- The snapshot blob gains `rope_delta` in the position section; the blob
  format version is bumped so a pre-vision blob is refused (ADR 0024: the
  changed section re-earns the snapshot-point permission).
- An evicted/restored multimodal sequence needs no vision state: after its
  last media item is prefilled, nothing vision-related is live.

### Observability

- `ignis.request.*` events gain attributes: media item count, vision tokens,
  media bytes, preprocess seconds, cache hits/misses, encode seconds. No new
  Prometheus metric in this spec (ADR 0017's contract changes need owner
  sign-off; a follow-up can propose them).
- The encode step is a DEBUG-level span at item granularity, never per
  layer (hot-path logging constraint; the hotpath lint's file list gains the
  new leaf-facing Rust files).

## Testing Decisions

Good tests here check what a client or the next layer observes — the
prepared prompt, the HTTP response, the generated tokens — never the
internal buffers of the encoder or the order of CUDA calls.

**Seam A — the frontend prepared prompt (CPU, highest non-GPU seam).**
Given an artifact's frontend and a message list with image bytes, the
prepared prompt must equal a recorded reference fixture *exactly*: token
ids, token types, three-axis positions, `rope_delta`, grid, token spans and
the SHA-256 of the packed BF16 patch rows. Fixtures are recorded once from
the reference processor on a small committed image set chosen to hit the
edges: an image already on the 32-px grid, one needing upscaling below
`min_pixels`, one needing downscaling above `max_pixels`, an odd aspect
ratio, a 1×N strip near the 200:1 limit, a PNG with alpha, a JPEG with EXIF
rotation, two images in one message, an image inside a tool message, and a
thinking-on render whose boundary offsets cross an expansion. Plus
behaviour tests on this seam for every processor error (bad aspect, corrupt
bytes, budget exceeded, placeholder/order mismatch). Prior art:
`crates/artifact/tests/real_frontend.rs` (real-artifact frontend),
`frontend.rs` unit tests (template renders content parts).

**Seam B — `POST /v1/chat/completions` (the existing HTTP seam).**
- CPU, mock engine: wire parsing of string-or-parts, every 400 code above
  with its message, `vision_disabled` without `--vision`, data-URI decoding,
  URL policy (private address refused, redirect to a private address
  refused, byte cap, timeout — against a local test server, with
  `--media-allow-private-network` for the positive path), `usage.prompt_tokens`
  counting expanded tokens, and cancellation during preprocessing. Prior
  art: `openai_http.rs`, `openai_http_tool_definitions.rs`,
  `api_key_http.rs`.
- GPU profile (fails, never skips, when the card is busy; check
  `make gpu-status` first):
  - **vision canary**: a handful of fixed images with short questions
    whose answers are unambiguous (a rendered number, a colour, a count of
    shapes, a line of text from a screenshot). Scored by teacher-forced
    agreement against the reference's greedy recording with `--vision` on
    the same artifact, first 32 positions, floor ≥ 95% (ADR 0014's floor,
    applied to multimodal prompts), read on the answer channel with
    thinking off. Prior art: `oracle_teacher_forced_gpu.rs`,
    `hq_canary_gpu.rs`.
  - **mixed load**: one multimodal prefill interleaved with text decode
    lanes produces sane output on every lane; the same with DFlash2
    loaded; two same-size different images sent as siblings get different
    answers (prefix identity); a multimodal sequence evicted to KV-RAM and
    restored continues sanely. Prior art: `openai_http_gpu.rs`,
    `decode_runs_http.rs`, `chunked_prefill_self_oracle_gpu.rs`.
  - **text unchanged**: with `--vision` loaded, a text-only chunked prefill
    still matches the per-token self-oracle (the MRoPE route must not leak
    into text spans).
- Vendored vision ops bring their reference op tests through the manifest
  (`kernel/vendor/tests`), run the way the existing vendored op tests run.
- Snapshot blob: the existing snapshot round-trip GPU test covers a
  sequence with non-zero `rope_delta`, and a pre-bump blob is refused.

**Performance (error detector, not a gate).** One live/live measurement,
same session, of TTFT on a fixed image prompt (one ~1-megapixel
screenshot + short question) against the reference with `--vision`, and
the text TTFT cells with `--vision` loaded vs not. Recorded in the ticket;
a large regression is a finding to file, not a merge block (owner
decision 2026-09-13: the 99% gate runs once at phase end).

`cargo test` passes workspace-wide.

## Out of Scope

- Video input (`video_url`): FFmpeg decode, frame sampling, timestamps and
  temporal grids. Refused with 400 until a follow-up spec.
- Local file paths as media sources.
- Image input on `/v1/completions` (no message structure to carry it).
- Playground support for attaching images (the composer's attachments are
  text today); a follow-up ticket once the server side lands.
- New Prometheus metrics for vision (needs an ADR 0017 amendment).
- MTP-with-vision specifics (MTP is deferred behind DFlash2).
- Packed prefill of several requests' vision encodes in one traversal; one
  item per encode call, like the reference.
- Encoding vision on a separate CUDA stream concurrently with decode (true
  prefill/decode overlap, ADR 0018 exclusion).
- The 35B-A3B target and the reference's DFlash (non-2) backend.

## Further Notes

- Reference sources read for this spec (ninfer `gpillon/coding`):
  `targets/qwen3_6/impl/frontend/processor.cpp` (preprocessing, expansion,
  positions), `impl/runtime/vision_context_impl.h` (encoder, workspace
  layout, chunk capping), `impl/runtime/text_context_impl.h` (scatter +
  MRoPE prefill, `rope_delta` in decode), `impl/runtime/prefix_identity.cpp`
  (media-aware prefix identity), `impl/vision/bindings.cpp` (object names
  and formats), `product/media_acquire/acquire.cpp` (data URI, URL fetch,
  SSRF policy), `serve/generation_service.cpp` (error codes),
  `docs/serving.md` §Multimodal, `docs/maintainer/qwen3.6-27b-model.md`
  §9–11.
- The reference runs the whole preprocessing on the host in BF16 directly,
  with no FP32 staging; keeping that property is why patches cross the ABI
  as BF16 rows.
- Likely a multi-ticket implementation; a natural split is processor +
  wire (Seam A + CPU half of B), leaf encoder + vendoring + step ABI,
  scheduler/prefix/snapshot integration, then the GPU acceptance. Split at
  pickup if the implementer prefers; this spec is the whole feature.
