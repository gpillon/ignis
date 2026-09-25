# Who places the reuse boundary: prior art for decision prefixes on a hybrid model

- Kind: research
- Status: current
- Observed: 2026-09-26
- Last verified: 2026-09-26
- Scope: serving / cross-request state reuse, `/v1/decide` fan-out, recurrent-state checkpoint placement, multimodal prefix keys, decision prompt layout
- Related: [GitHub #270](https://github.com/gpillon/ignis/issues/270), [#235](https://github.com/gpillon/ignis/issues/235), [#240](https://github.com/gpillon/ignis/issues/240), [#243](https://github.com/gpillon/ignis/issues/243), `docs/adr/0029-cross-request-state-reuse.md`, `docs/adr/0034-the-leaf-answers-without-generating.md`, [`2026-09-20-the-evidence-belongs-in-the-system-block.md`](2026-09-20-the-evidence-belongs-in-the-system-block.md), [`2026-09-18-prompt-reuse-tax-on-short-ttft.md`](2026-09-18-prompt-reuse-tax-on-short-ttft.md), [`2026-09-20-shared-vision-embedding-fan-out.md`](2026-09-20-shared-vision-embedding-fan-out.md), `.scratch/sota-research-2026-09-24/02-kv-longctx.md` §4 (re-checked and extended here)
- Superseded by: none

## Question

GitHub #270: on `POST /v1/decide`, a `state` given as content parts (text +
`image_url`) reuses nothing. A long, byte-identical text head is re-prefilled
on every request (~220-280 ms), and the whole state is re-prefilled for each
of the N questions of a fan-out. On this hybrid (48 GDN + 16 GQA layers), a
resume at position P needs the recurrent state **at P**. That state has to be
snapshotted during prefill (~188 MiB a retained slot; 8 slots, shared with
everything else). So the server must **predict** reuse boundaries before it
prefills.

Today it predicts one boundary, P1: the page-floored end of the system block.
The options on the table are:

- **option 1:** move the leading text parts into the system block;
- **option 3:** a list of predicted boundaries:
  - P2, the exact common prefix of a fan-out's N prompts;
  - P3, a static head shared across requests, found by (a) a "first text
    part" convention, (b) an explicit client marker such as Anthropic's
    `cache_control`, or (c) server-side history of recent prompts
    (Marconi-like).

What have others done? The survey covers decisions over a shared context,
who places a cache boundary, images inside a cached prefix, and above all
where to put **recurrent-state checkpoints**. Did anyone solve this exact
problem?

## Short answer

- **Nobody publishes the whole combination.** No system does all of these on
  a hybrid recurrent model: reuse a static text head across requests, vary
  the images after it, and share the state exactly across a fan-out. Every
  piece exists somewhere.
- **The closest piece is an open vLLM RFC, not a result.**
  [vLLM #55697](https://github.com/vllm-project/vllm/issues/55697)
  (2026-09-07; PRs #55873/#55875/#55876 open) targets our shape:
  - a 1-to-9 *multimodal* matching fan-out, read from logits, on Qwen3.5 (a
    GDN hybrid);
  - a **client-placed input marker**, `<|mamba_checkpoint|>`;
  - a producer/consumer scheduler.

  Its argument: "from an engine-internal perspective, it is virtually
  impossible to heuristically predict where that single, high-value
  checkpoint boundary should be placed". Its numbers are the author's own.
  On 2026-09-24 the only vLLM maintainer to review it put the
  implementation on hold: "I don't really think we want this feature, it
  introduces lots of complexity."
- **Besides a block grid and the prompt tail, engines that predict without a
  hint use two kinds of predictor: structure and observed forks.**
  - *Structure.* llama.cpp checkpoints at every user-message start, plus twice
    near the prompt end. OpenAI (GPT-5.6+) breaks at the end of the latest
    eligible message. A vLLM draft PR stops after the final image. ignis stops
    at the end of the system block.
  - *Observed forks* (Marconi). vLLM (merged 2026-06-10) and SGLang both
    **detect a fork from the dense attention-KV match**. Where the attention
    KV hits and the recurrent state misses, there is a shared prefix nobody
    checkpointed, so the next request that reaches it checkpoints it.
    DeepSeek's API does the same after its move to sliding-window attention.
    A fork pays off from its **third** occurrence and is exact. In vLLM it is
    floored to the recurrent block (528-2096 tokens for Qwen3.5).
- **Hosted APIs make the client place the write.**
  - Anthropic and Alibaba (Qwen's vendor): up to 4 `cache_control` markers.
    Writes happen only at a marker, and reads look back 20 blocks for *prior
    writes*. Anthropic: "The lookback does not find stable content behind your
    breakpoint and cache it".
  - OpenAI GPT-5.6+: an implicit breakpoint at the latest message end, plus
    up to four explicit `prompt_cache_breakpoint` writes on content blocks.
  - Anthropic and Alibaba both tell concurrent callers to wait for the first
    request to write before sending the rest. ignis's fan-out already
    sequences its first question that way.
- **Jev-like systems solve the fan-out within one request, for text only.**
  - Jev "ingests the `state` once and evaluates every question against it in
    parallel".
  - SemIf prefills the state once and scores N suffixes in one batched pass
    **on a GDN hybrid** (Qwen3.5-4B), 8.6x faster than fresh scoring.
  - SGLang's `select` and `fork` send the prefix alone first.

  None of them takes an image.
- **Images are keyed by content everywhere, so text before an image stays
  reusable** — for attention KV. On a hybrid that only helps if a recurrent
  checkpoint exists at or before the image. That checkpoint is exactly where
  vLLM (#43587) and llama.cpp (#26994) have open bugs.
- **Sharing one state across *different* instructions placed before it is
  not a prefix problem.** For hybrids, the only published answers are lossy
  position-independent caching: HYPIC has a 1.71-point quality gap, and
  LinearKV recovers up to 92% of full quality. The exact answer in the
  literature is layout: shared context first, per-question text last.
  - That is Jev's API shape.
  - It is Alibaba's advice for asking several questions about one image.
  - Most measurements favour it for accuracy.

## Evidence

### 0. arXiv 2609.22753, the owner's pointer

**What it is.** "Replacing Large Language Models with Jev Decision Models for
Low-Latency Edge Service Orchestration", by Li, Wang, Gong, Lang and Yu
(University of Technology Sydney), arXiv v1 19 Sep 2026
([html](https://arxiv.org/html/2609.22753v1)). An edge admission pipeline
interprets natural-language service requests with an LLM; the paper swaps
that LLM for Jev and measures decision latency and end-to-end completion.

**Mechanism.** A request is interpreted into four bounded fields (service,
locality, quality floor, urgency: 108 tuples). Appendix A-A: "Jev uses
OpenRouter's Decisions endpoint with identifier typesafe/jev-1.13; responses
identify jev-1.13-20260917 … **Four native Choice questions return the shared
intent fields in one request**." The baselines:

- deepseek-v4.1-flash (JSON output, temperature 0, 128-token cap);
- Qwen2.5-7B-Instruct, self-hosted on one L40;
- a regex parser.

The studies ran on 18 Sep 2026.

**Numbers (measured, from the paper's own tables).**

| Table II (216 requests per block) | block 1 | block 2 | block 3 |
|---|---|---|---|
| Jev median | 319.5 ms | 314.7 ms | 320.7 ms |
| DeepSeek median | 434.7 ms | 425.9 ms | 381.4 ms |
| Jev correct / 216 | 214 | 213 | 212 |
| DeepSeek correct / 216 | 216 | 215 | 216 |

That is a median 15.9-26.5% lower for Jev, and p95 9.1-36.7% lower.

Study B (real OCR, two nodes), correct on-time OCR out of 288:

| backend | correct |
|---|---|
| Jev | 168 |
| DeepSeek | 166 |
| Qwen2.5-7B | 105 |

Jev's API fees per correct completion were 68.97-70.61% lower.

**Its cache is not a prefix cache.**

- §III-B: "each backend reuses its own validated interpretation for identical
  normalized request text under the same extraction policy".
- A-B: the cache is "keyed by normalized full request text and extraction
  policy … concurrent misses are not coalesced".
- §V-D: with repeated text and the cache on, medians fall to 110.9 ms (Jev)
  and 112.4 ms (DeepSeek), because no interpreter is called at all. "Changing
  text conditions have no exact cache hits."

The paper says nothing about prefill, KV, or prefix reuse, and it does not
describe how Jev works inside. Jev reads text only; images go to the OCR
worker, never to the interpreter.

**Its related work** (§II-A..C) covers three areas:

- intent-based networking;
- small or bounded models (Sentence-BERT, SetFit, GLiNER/GLiNER2);
- serving and cost (Clipper, InferLine, PagedAttention, SGLang,
  JSONSchemaBench, FrugalML/FrugalGPT, RouteLLM, GPTCache).

None of its references concerns prefix or state reuse for decisions. Its two
serving systems are covered below from their own sources.

**How it relates to #270 (inference).** It confirms the workload shape:
several typed questions over one input, in one request. The only reuse its
authors needed was an exact-match cache of *responses*. That cache never
fires when the input changes every step, which is the game agent's regime. So
the paper explains why Jev-shaped endpoints matter at the edge; it is not
prior art for the mechanism.

### 1. Jev-like systems: N questions over one state

**Jev itself** (vendor docs, fetched 2026-09-26):

- "Jev ingests the `state` once and evaluates every question against it in
  parallel. The 64k budget covers the `state` plus all questions combined;
  the 32k budget applies to the `state` plus the single longest question"
  ([models](https://docs.typesafe.ai/models.md)).
- "Adding questions barely changes the response time and costs only the
  tokens for the extra questions"
  ([primitives](https://docs.typesafe.ai/primitives.md)).
- The
  [parallel-questions cookbook](https://docs.typesafe.ai/cookbooks/parallel_questions.md)
  asks 13 questions over the ~54,000-character GDPR article:

  | | cost | time |
  |---|---|---|
  | one call, 13 questions | $0.000497 | 0.27 s |
  | 13 calls, summed | $0.006090 | 2.71 s |

  The cookbook says "12.2x cheaper, 10.0x faster". The primitives page
  quotes 11.5x / 9.6x for the same cookbook.
- `usage` reports only `input_tokens` and `output_tokens`; there is no
  cached-token field ([API](https://docs.typesafe.ai/api.md)).
- "Jev accepts text only … Images, audio, and video are not supported (yet)"
  ([state](https://docs.typesafe.ai/concepts/state.md)).

*Inference:* a budget of "state plus the longest question" per question is
the signature of one state prefix with N per-question continuations. How Jev
actually implements it is not published.

**SemIf** ([TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf) @ `23cf1f3`),
the MIT project ignis's decision fixture comes from. Its `score_shared`
(`src/semif_phase1/shared.py:14-128`) works in four steps:

1. It builds the shared prefix **by construction**: the chat template's head
   plus `{"evidence": <state>`, minus its last token ("Appending JSON
   punctuation can merge with the final boundary token").
2. It checks that every full prompt starts with that prefix.
3. It prefills the prefix once under Transformers and replicates the cache
   with `cache.reorder_cache(torch.zeros(N))`.
4. It scores all N padded suffixes in one batched forward, with
   `logits_to_keep` at each suffix end.

The llama.cpp backend instead calls `prefill`, `save_state`, then
`restore_state` once per branch.

The README measures this on a frozen **Qwen3.5-4B** on one RTX 3090 (37
states × 21 criteria = 777 decisions):

| execution path | decisions/s | 777 decisions |
|---|---|---|
| fresh scoring | 2.33 | 333.1 s |
| serial prefix reuse | 10.75 | 72.3 s |
| parallel suffixes | 20.03 | 38.8 s |

"BF16 execution changed 5–6 of 777 argmaxes relative to fresh scoring."
Qwen3.5-4B is a GDN hybrid: its config lists `linear_attention` three times,
then `full_attention`, with `full_attention_interval: 4`
([config](https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/config.json)). So
one prefill with N suffixes **works on a GDN hybrid**, provided the
framework's cache object replicates the recurrent state along with the KV.
Text only.

**SGLang's frontend.** The paper ([arXiv 2312.07104](https://arxiv.org/html/2312.07104v2)
§3): "During the execution of the fork primitive, the frontend sends the
prefix first as a hint, ensuring the prefix is correctly inserted into the
tree. It then sends the remaining prompts." A "No Frontend Hint" ablation is
slower (§6.3). `select` does the same in code
(`python/sglang/lang/backend/runtime_endpoint.py:246-272` @ `c421d16`):

1. a request with the prompt alone and `max_new_tokens: 0` ("# Cache common
   prefix");
2. one batch of `prompt + choice` texts, with `return_logprob` and
   `logprob_start_len = prompt_len - 2` (token healing).

The runtime's own `n>1` does the same thing ("# Cache the common prefix for
parallel sampling", `tokenizer_manager.py:2018-2035`). The client knows the
boundary; the runtime only has to be told first.

**vLLM.**

- `n>1` fans out as N separate engine requests ("# Fan out child requests
  (for n>1)", `vllm/v1/engine/async_llm.py:507-525` @ `4ccfe12`), which share
  the prompt only through automatic prefix caching.
- A classifier that reads logprobs through `prompt_logprobs` loses prefix
  caching. `get_computed_blocks` skips the lookup when "the request is marked
  as skipping kv cache read (which happens when the request requires prompt
  logprobs or calls a pooling model with all pooling)"
  (`vllm/v1/core/kv_cache_manager.py:281-286`). The flag is set whenever
  `prompt_logprobs is not None` (`vllm/sampling_params.py:549-553`), and the
  docs say so: "the engine will ignore the prefix cache and recompute the
  prefill of full prompt" (`docs/usage/v1_guide.md:59-62`). Only reads are
  skipped, and logprobs on a *generated* token keep the cache.
- `LLM.score` replicates `data_1` N times in the 1-to-N case and again relies
  on prefix caching (`vllm/entrypoints/pooling/offline.py:285-306`).

**Constrained-generation frameworks do not score options over a shared
prefix.**

- Guidance's `select` and Outlines' `Choice` are grammar constraints on one
  decode.
- DSPy enforces a `Literal` with prompt text plus parsing.
- LMQL's `distribution` clause scores `[prompt_seq] * len(value_ids)`. The
  source does not show whether the prompt's KV is shared across values; this
  is unverified.

The sources, read at pinned commits:

- guidance `guidance/_grammar.py:61-69` @ `21b1d90`;
- outlines `src/outlines/types/dsl.py` @ `c52af84`;
- lmql `src/lmql/runtime/postprocessing/conditional_prob.py:13-58` @ `fc8edd4`;
- dspy `dspy/adapters/utils.py:143-147` @ `8de96ea`.

One line each is all they merit.

**Batch prompting** ([arXiv 2301.08721](https://arxiv.org/abs/2301.08721))
packs b questions into one prompt instead of sharing a prefix. It measures up
to ~5x less cost at b = 6, and accuracy moves by −4.0 to +3.0 points across
ten datasets (Codex, Table 1). "Performance typically decreases as b
increases." It is a different trade: the questions see each other.

### 2. Cross-request static prefixes: who places the boundary

**Anthropic** ([prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching),
fetched 2026-09-26; the page carries no date).

- Up to 4 breakpoints. Top-level "automatic caching" puts one breakpoint on
  the last cacheable block and uses one of the 4 slots.
- Three rules, verbatim:
  - "**Cache writes happen only at your breakpoint.**"
  - "Cache reads look backward for entries that prior requests wrote … It is
    looking for prior writes, not for stable content."
  - "The lookback window is 20 blocks."
- On layout: "For a prompt with a varying suffix (timestamps, per-request
  context, the incoming message), place the breakpoint at the end of the
  static prefix, not on the varying block."
- Minimum cacheable length: 512 tokens (Fable 5.1, Opus 5.5, …), 1,024
  (Sonnet 5, Opus 4.8, …), 4,096 (Haiku 4.5).
- TTL: 5 min (writes cost 1.25x) or 1 h (2x).
- Invalidation follows the hierarchy tools → system → messages. "Adding/removing
  images anywhere in the prompt affects message blocks"; the system and tools
  caches survive.
- On concurrency: "For concurrent requests, note that a cache entry only
  becomes available after the first response begins. If you need cache hits
  for parallel requests, wait for the first response before sending
  subsequent requests."

**Alibaba Model Studio** ([context cache](https://www.alibabacloud.com/help/en/model-studio/context-cache)),
the vendor of this model family.

- Explicit cache uses an Anthropic-shaped `cache_control`: "examines up to 20
  preceding content blocks", "up to four cache markers", 1,024 tokens
  minimum, 5-minute validity.
- "The system creates the cache after the model generates a response."
- For vision models: "To ask multiple questions about the same image or
  video, place the image or video before the text. To ask the same question
  about different images or videos, place the text before the image or
  video."

**OpenAI** ([prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching.md),
fetched 2026-09-26). The minimum is 1,024 tokens on GPT-5.6+, which has two
modes (explicit controls added 2026-07-09 per the changelog):

- *Implicit mode*: "a breakpoint at the end of the latest eligible message".
  Eligible messages are user messages, the last tool response of a group, and
  the last developer message of the initial group. Lookups try "the first 2
  and latest 50 explicit breakpoints, the implicit breakpoint, up to 20
  earlier eligible message endings, and the endpoint of the initial
  consecutive block of developer messages".
- *Explicit mode*: `prompt_cache_breakpoint: { "mode": "explicit" }` goes "to
  a supported content block inside an input message … Each request can create
  up to four cache writes". Explicit-only mode avoids writing the changing
  suffix.

Older models used a fixed grid instead:

- GPT-5.5 placed implicit breakpoints "at regular 2,048-token intervals".
- Earlier models used "model-dependent intervals", with `cached_tokens`
  rounded down to 128.

Images, documents and tools are all part of the cached rendered context. The
layout advice: "Put stable developer instructions and shared reference
material first", with dynamic content at the end. `prompt_cache_key` steers
routing; it matters before GPT-5.6 and only splits accounting after it.

*Observed trend:* OpenAI moved from a fixed token grid to structural message
boundaries plus client-declared breakpoints.

**Brokers and others.**

- OpenRouter translates between the two markers: "a text block marked with
  Anthropic-style `cache_control` gets a `prompt_cache_breakpoint` when routed
  to a supporting OpenAI model", and it places `prompt_cache_breakpoint` on
  text blocks only
  ([docs](https://openrouter.ai/docs/guides/best-practices/prompt-caching.md)).
- Gemini: implicit caching is on by default, with a minimum of 4,096 tokens on
  Gemini 3.x and 2,048 on 2.5. "Try putting large and common contents at the
  beginning of your prompt"
  ([caching](https://ai.google.dev/gemini-api/docs/caching), updated
  2026-09-02). Explicit caches are prefixes too, with a default TTL of 1 h.
- **DeepSeek** ([KV cache](https://api-docs.deepseek.com/guides/kv_cache)) is
  a hosted engine with a layer that cannot be rolled back, like a recurrent
  one: "Due to the Sliding Window Attention mechanism … Each cached prefix is
  an independent, complete unit." It persists a unit in three cases:
  - "at the end position of the user input and the end position of the model
    output";
  - by "Common prefix detection persistence";
  - "at fixed token intervals".

  In its long-document example (same report, different question), "A + B"
  then "A + C" then "A + D" hits only on the third request. This is Marconi
  in production.

**Engines, for attention models.** vLLM hashes every full block (chained over
the parent hash, plus extra keys: multimodal identifiers, LoRA, cache salt).
SGLang keeps a radix tree over token ids. Neither needs to predict a boundary
for an attention-only model, because any block of KV is reusable on its own.
The prediction problem appears only with state that cannot be rolled back
(§3).

### 3. Hybrid checkpoint placement: how each system chooses the boundary

**Marconi** ([arXiv 2411.19379](https://arxiv.org/html/2411.19379v3), MLSys
2025; v3 10 Apr 2025).

- §4.1 splits reused prefixes into two kinds. "Purely input" prefixes are
  system prompts, few-shot examples, long-document QA and self-consistency.
  "Input and output" prefixes are conversations and agent trajectories.
- For the first kind: "prior to prefilling each sequence, Marconi employs a
  speculative insertion of the input tokens to see if new intermediate nodes
  will be created … If so, Marconi caches the prefix's states during
  prefill".
- For the second kind it "checkpoints the state after the last decoding
  step".
- At most two SSM states per sequence are admitted. "'Purely input' prefixes
  only benefit from reusing starting from the third occurrence of the
  prefix."
- With chunked state passing, it checkpoints at the chunk boundary before the
  target ("if we need to cache the state at token 80, we can checkpoint the
  state at token 64"); otherwise it runs a two-pass prefill.
- Eviction weighs recency against **FLOP efficiency** (compute saved per byte
  of state).

Measured on a 7B hybrid (4/24/28 attention/SSM/MLP layers) and on
Jamba-1.5-Mini on vLLM, 4×A100-40GB. The baselines are the authors' own
hybrid extensions: of vLLM (a state every 32-token block) and of SGLang.

- Token hit rate +4.5×/7.3×/34.4× on LMSys, ShareGPT and SWE-Bench.
- P95 TTFT up to 71.1% (617 ms) lower.
- "prefix reusing is exact".

**vLLM** (main @ `4ccfe12`, 2026-09-25).

- With prefix caching on, `mamba_cache_mode` defaults to "align": "only cache
  the mamba state of the last token of each scheduler step and when the token
  is at position i * block_size". Mode "all" caches every block
  (`vllm/config/cache.py:190-198`).
- Mamba prefix caching has been on by default since
  [#50991](https://github.com/vllm-project/vllm/pull/50991) (2026-08-04).

vLLM chooses a boundary in three ways:

- **The block grid and the prompt tail.** The chunk is clipped to block
  boundaries and to the last cacheable one.
- **A Marconi junction detected from the attention cache.**
  [#37898](https://github.com/vllm-project/vllm/pull/37898) (merged
  2026-06-10): "existing regular attention KVCache is used for detecting
  presence of a shared prefix (i.e. a non-cached shared prefix exists if
  standard attention KVCache has hits, but SSM attention doesn't …)". The
  state is cached "at the last block-aligned position". Its synthetic test:
  "Main is worse by up to 40% (`Qwen/Qwen3.5-0.8B`) or 66%
  (`Qwen/Qwen3.5-35B-A3B`)". In code:
  - `shared_prefix_boundary = num_new_computed_tokens + num_uncached`
    (`kv_cache_manager.py:312-321`);
  - `_mamba_block_aligned_split` stops a chunk there, block-floored ("a
    sub-block junction's state is not separately cacheable";
    `sched/scheduler.py:409-522`).
- **Retention.** [#52216](https://github.com/vllm-project/vllm/pull/52216)
  (merged 2026-08-17) sets `prefix_cache_retention_interval` to 0 by default,
  which "retains only semantic checkpoints, including the latest replay
  boundary and shared-prefix junctions". Separately, `prefix_match_unit` lets
  a hit land on a finer grid than the physical block
  ([#45702](https://github.com/vllm-project/vllm/issues/45702)).

What fails, from vLLM's own tracker:

- With only the tail checkpoint, reuse "silently drops to 0%" when the last
  block boundary falls in the unique suffix
  ([#45238](https://github.com/vllm-project/vllm/issues/45238), open). On
  Qwen3.5-4B:

  | shared / unique tokens | hits | TTFT |
  |---|---|---|
  | 1600 / 600 | 52/64 | 433 ms |
  | 1500 / 700 | **0/64** | 905 ms |
- The attention block is forced to 528 tokens for Qwen3.5-4B, so prompts
  under 528 tokens get 0%
  ([#40696](https://github.com/vllm-project/vllm/issues/40696), open).
- The Marconi path gives hits "on 3rd appearance of shared prefix"
  ([#55403](https://github.com/vllm-project/vllm/pull/55403), open). That PR
  re-admits masked blocks for a synthetic 35% prefill cut on Qwen3.5-9B.
- A metric for "shared-prefix tokens lost to a missing sparse-retention
  checkpoint" is still an open PR
  ([#52527](https://github.com/vllm-project/vllm/pull/52527)).
- A *proactive* alternative — checkpoints at chat-template "semantic"
  boundaries (`input_end`, `reasoning_end`, `response_end`) — was tried in
  [#49574](https://github.com/vllm-project/vllm/pull/49574) and closed
  unmerged on 2026-09-11. Its one preliminary result: "An `input_end`-only
  semantic-boundary experiment produced the same prefix-hit token count as
  the sparse baseline on the existing canonical user-input and tool-call
  workloads."

**vLLM, automatic media boundary.**
[#56818](https://github.com/vllm-project/vllm/pull/56818) (open draft,
2026-09-14) "stops align-mode Mamba prefill once at that cache-aligned
boundary" after the **final multimodal item**, "deliberately limited to the
final MM item, so it adds at most one prefill split per multimodal request".
It is related to [#43587](https://github.com/vllm-project/vllm/issues/43587)
("Prefix caching fails for incremental multimodal requests on Mamba-Attention
hybrid models (Qwen3.5)", open). Its author reports that #43587's 4 → 5 image
repro already hits on v0.29.0 (`cached=1632`). The PR targets the remaining
miss, where "full attention can identify the shared prefix, but align-mode
Mamba has no materialized state at that boundary".

**vLLM RFC #55697: the client places it**
([issue](https://github.com/vllm-project/vllm/issues/55697), open,
2026-09-07).

- An input-side special token, `<|mamba_checkpoint|>`, marks the boundary.
- The frontend records its position **after multimodal placeholder
  expansion**, aligns it to `prefix_match_unit`, and strips the token before
  hashing.
- The producer's chunk is clipped at that position and its checkpoint hash is
  marked UNREADY. Consumers that share the prefix are deferred, then woken
  and batched.

The workload is "1-to-9 Multimodal Pairwise Matching": a shared ~624-token
prefix (text attributes + one image), then nine 350-650-token suffixes, each
with its own image. The output is "a sequence classification or scoring
probability (logits)". Claimed on Qwen3.5-35B / L40S: 5.9 → 12.4 QPS and
TTFT 482 → 248 ms (later 13.8 QPS on MRv2).

The implementation PRs (read 2026-09-26) add two things the RFC text only
sketches:

- **Same-step pairing**
  ([#55875](https://github.com/vllm-project/vllm/pull/55875)). "When a
  Producer and Consumer arrive in the same scheduling pass, the Consumer
  inherits the Producer's prefix blocks directly … enabling them to execute
  in the same engine step."
- **A two-phase GDN prefill**
  ([#55876](https://github.com/vllm-project/vllm/pull/55876)). "In Phase 1,
  all Producers run in a single packed kernel invocation and populate shared
  checkpoint state rows. In Phase 2, Consumers read the shared checkpoint
  rows (`index_select`), copy them to private destination slots
  (`index_copy_`), and compute suffixes concurrently in a second packed
  kernel invocation." The author's GDN prefill microbenchmark on an L40S
  (one layer's kernel, not end to end):

  | producers + consumers | serial | two-phase |
  |---|---|---|
  | 1 + 1 | 2.15 ms | 1.55 ms |
  | 1 + 16 | 12.70 ms | 1.67 ms (7.58x) |

The author reports accuracy as parity, not bit-exactness:

- gsm8k on Qwen3.5-9B: 0.868/0.880 at baseline against 0.880/0.884 with the
  feature;
- "99.97% prediction parity and >0.9988 Pearson correlation" on 3,428
  internal multimodal pairs.

Status on 2026-09-26: all three PRs are open and in merge conflict. #55873
has no maintainer review. The RFC issue has no maintainer reply. The only
maintainer who reviewed the other two requested changes:

- on #55875 (2026-09-23): "we don't want to introduce more changes to MRv1
  now, as it will be deprecated soon";
- on #55876 (2026-09-24): "After some consideration I temporaly hold this PR,
  please have another maintainer approve your RFC first. I don't really
  think we want this feature, it introduces lots of complexity."

A companion RFC, [#52959](https://github.com/vllm-project/vllm/issues/52959)
(open), captures a checkpoint *inside* one forward instead of splitting the
prefill: "forward [0, 8000) -> checkpoint state@7680 -> running state@8000".

**SGLang** (main @ `c421d16`, 2026-09-25). LMSYS on Qwen3.8
([2026-08-12](https://www.lmsys.org/blog/2026-08-12-qwen3-8-day0-support/)):
"SGLang creates new checkpoints at prefill chunk boundaries and regular decode
intervals", with copy-on-write before a forward mutates a shared checkpoint.
In code:

- The buffered strategy tracks **one state per extend**, at the last
  checkpoint-grid-aligned position. It reads that state from the chunked
  kernel's intermediate `h` inside the same forward (`_force_track_h`,
  `managers/schedule_batch.py:3001-3087`).
- The radix match can report a `mamba_branching_seqlen`: "the longest
  page-aligned position that could've been cache hit if there exists a mamba
  state" (`mem_cache/base_prefix_cache.py:244-246`). The match computes it as
  "the last Mamba-cache-chunk-aligned position within the Full-KV hit that
  lies beyond the current Mamba boundary"
  (`mem_cache/unified_cache/components/mamba.py:166-183`). When that position
  falls inside the current extend, SGLang tracks it instead. This is the same
  Marconi-style detector as vLLM's. The first sibling to diverge pays, and
  later siblings hit.
- The same comment warns that under HiCache, "incremental persistence of a
  new branching state is currently write-through only".
- The grid is the larger of the FLA chunk size and the page size
  (`arg_groups/overrides.py:1787-1790`).
- The hybrid VLMs `Qwen3_5ForConditionalGeneration` and
  `Qwen3_5MoeForConditionalGeneration` are routed through this cache
  (`arg_groups/overrides.py:345-374`). No SGLang issue or test for image hits
  on a hybrid VLM was found.
- A logprob request caps its own match: "the matched length is at most 1
  less than the input length to enable logprob computation", and at most
  `logprob_start_len` (`managers/schedule_batch.py:1686-1696`). `select`
  therefore reuses its warmed prefix only up to `prompt_len - 2`.
- `mamba_track_interval` (default 256) spaces the decode checkpoints.
- `mamba_max_states_per_path` preserves "Tail, fork, and locked nodes".
- `--enable-int8-mamba-checkpoint` (off by default) stores cached states in
  int8 for ~2x capacity. It is lossy (`arg_groups/fields/exec_.py:370-414`).

**llama.cpp** (master @ `4b1a27f`, 2026-09-25). Context checkpoints exist for
memories that cannot be partially truncated (recurrent) and for SWA. The
default is 32 per slot, with `--checkpoint-min-step` 8192
(`common/common.h:630-632`). They are placed in three kinds of position:

- **at the start of user messages**, found from the chat template's message
  spans ([#24176](https://github.com/ggml-org/llama.cpp/pull/24176), merged
  2026-06-23: "create checkpoints at every user message");
- **twice near the prompt end**, 4 + n_ubatch and 4 tokens before it
  ([#20288](https://github.com/ggml-org/llama.cpp/pull/20288));
- never in a batch that processed a media chunk ("do not checkpoint after
  mtmd chunks", `tools/server/server-context.cpp:3460-3634`).

The last rule is what [#26994](https://github.com/ggml-org/llama.cpp/issues/26994)
(open, 2026-08-13) hits: "Image prompt cache never hits for hybrid SSM models
(Qwen3.6) when only the text suffix changes". That is `[image][question_i]`,
our fan-out's shape.

**Sparse Prefix Caching** ([arXiv 2605.05219](https://arxiv.org/abs/2605.05219),
17 Apr 2026) places K exact checkpoints inside one entry. An exact O(NM)
dynamic program works over the *distribution* of overlap depths ("asking
different questions about a single long document"). It complements Marconi's
admission policy. The limits of its evidence:

- the wall-clock numbers are a prototype on one layer group (1
  full-attention + 3 GDN layers of Qwen3.5-0.8B);
- checkpoints were captured "in separate untimed runs, because the current
  Flash Linear Attention library does not expose a Python API for non-last
  checkpoint extraction".

**No checkpoint at all (lossy).**

- Tail-Replay ([arXiv 2608.30310](https://arxiv.org/abs/2608.30310), 31 Aug
  2026) keeps the attention KV exact and rebuilds GDN state by replaying a
  5-10% recent suffix: "the reuse boundary is determined by the shared tokens
  rather than by recurrent-state checkpoints". It keeps 92.8-99.9% of
  full-prefill quality and cuts TTFT 9.1-14.3x at 32K.
- DASC ([arXiv 2608.30386](https://arxiv.org/abs/2608.30386)) compresses
  checkpoints (KDA 2.63x, −42.6% mean TTFT), zero-filling or refreshing the
  units it omits.

Both give up ADR 0029's bit-exactness to remove or shrink the checkpoint.

**A correctness trap others hit.** A logit readout needs the last position
computed, so a recurrent checkpoint must stop short of the prompt's end.

- vLLM recomputes at least the last token on a full hit "to obtain logits"
  (`kv_cache_manager.py:288-294`).
- llama.cpp guards a checkpoint so that "at least one token" is processed
  (`[TAG_PROMPT_LOGITS]`).
- A GLM-5.3-Flash + LMCache study found a hybrid that "restored state for the
  full prompt while the scheduler credited one fewer token"
  ([arXiv 2609.15030](https://arxiv.org/abs/2609.15030)).

### 4. Non-prefix (modular) reuse, and why it barely reaches a recurrent layer

Six attention-only systems, read from the papers' own text:

- **Prompt Cache** ([arXiv 2311.04934](https://arxiv.org/abs/2311.04934),
  MLSys 2024):
  - *Segments:* the **client** declares reusable "prompt modules" in a
    Prompt Markup Language schema. Each module is encoded once, with position
    ids reserved by "the absolute location of the prompt module within the
    schema".
  - *Accuracy:* approximate — "the approximation made by Prompt Cache is to
    limit the attention window to each prompt module" — and exact only with
    "scaffolds".
  - *Measured:* TTFT 1.5-10× on GPU.
  - *Recurrent layers:* attention-only by construction.
- **CacheBlend** ([arXiv 2405.16444](https://arxiv.org/abs/2405.16444),
  EuroSys '25):
  - *Segments:* chunks from the retriever. It concatenates precomputed chunk
    KV and recomputes <15% of tokens.
  - *Accuracy:* lossy.
  - *Measured:* TTFT 2.2-3.3×.
  - *Recurrent layers:* "currently only applies to language models with
    transformer structures … such as Mamba and Griffin for future work".
- **EPIC** ([arXiv 2410.15332](https://arxiv.org/abs/2410.15332), ICML 2025):
  - *Segments:* the client submits immutable chunks.
  - *Accuracy:* LegoLink recomputes each chunk's first k tokens; "accuracy
    losses limited to within 7%".
  - *Measured:* up to 8× TTFT.
  - *Recurrent layers:* attention-only.
- **Block-Attention** ([arXiv 2409.15355](https://arxiv.org/abs/2409.15355),
  ICLR 2025):
  - *Segments:* blocks set by the application.
  - *Accuracy:* **needs fine-tuning** ("dropping significantly from 67.9% to
    48.0%" without it).
  - *Recurrent layers:* attention-only.
- **RAGCache** ([arXiv 2404.12457](https://arxiv.org/abs/2404.12457)):
  - *Segments:* a prefix tree over ordered document ids.
  - *Accuracy:* exact, because it is a prefix method.
  - *Recurrent layers:* not mentioned by the paper.
- **TurboRAG** ([arXiv 2410.07590](https://arxiv.org/abs/2410.07590)):
  - *Segments:* offline chunks.
  - *Accuracy:* **needs fine-tuning**.
  - *Recurrent layers:* attention-only.

Two app-declared systems handle *prefix* structure rather than non-prefix
reuse:

- **Parrot** ([arXiv 2405.19888](https://arxiv.org/abs/2405.19888), OSDI
  2024) hashes prefixes only "at positions after each Semantic Variable",
  that is, at boundaries the application's template declares.
- **BatchLLM** ([arXiv 2412.03594](https://arxiv.org/abs/2412.03594))
  identifies a batch's common prefixes ahead of time with a prefix tree.

**On recurrent state, every composition method is approximate.**

- PICASO ([arXiv 2502.17605](https://arxiv.org/abs/2502.17605), ICLR 2025)
  composes Mamba-2 states. It is exact only "in a single-layer model"; for
  "multi-layer SSMs … Proposition 1 does not hold exactly". It is "limited to
  architectures based on SSM layers" (no hybrids).
- HYPIC ([arXiv 2607.01299](https://arxiv.org/abs/2607.01299), v2 12 Jul
  2026) caches, per segment, "the segment-cumulative transition operator …
  alongside each segment's zero-start end-state". It repairs attention across
  segments with a "seam window", and segments come from a client
  `PIC_SEPARATOR` marker. It reports TTFT 3.25x lower than prefix caching
  "with a 1.71-point gap from Full Recompute".
- LinearKV ([arXiv 2608.11231](https://arxiv.org/html/2608.11231), 31 Jul
  2026) tests on Qwen3.6-27B, "16 FA/48 GDN of 64 layers" — this model's
  layout. On GDN, a single cached state and exact composition "tie … up to
  92%" of full quality, at a TTFT of 0.46x full prefill. It names the cause:
  "independently prefilled deep operators … built from context-mismatched
  hidden inputs".

*Inference:* the GDN recurrence is affine in its state. But at every layer
past the first, a segment's inputs depend on what preceded the segment. So
even "exact" composition is only exact for the wrong inputs.

### 5. Images inside a cached prefix

All three engines key an image by a content hash, applied at its placeholder
position.

- **vLLM** (`vllm/v1/core/kv_cache_utils.py:500-565` @ `4ccfe12`): "the extra
  keys are (mm_hash, start_offset) that indicate a mm input contained in the
  block". They are added only to blocks that overlap the item's placeholders.
  The hash is blake3, by default, over the encoded file bytes plus
  `model_id` and processor kwargs (`vllm/multimodal/hasher.py`). A block
  wholly before an image carries no image key, so text before an image is
  reusable whatever the image is. The block holding the image's first
  placeholder, and every block after it, differs.
- **SGLang** replaces an image's placeholder ids with a `pad_value` derived
  from the item's hash: "Each item has its own hash and pad_value, enabling
  per-image RadixAttention caching"
  (`managers/schedule_batch.py:369-443` @ `c421d16`).
- **llama.cpp** compares media chunks by id and token count when it computes
  the common prefix (`tools/server/server-common.cpp:697-745` @ `4b1a27f`).
  The id is a SHA-256 of the file bytes: "use sha256 to prevent cache
  poisoning" (`tools/mtmd/mtmd-helper.cpp:384-387`). With media,
  `--cache-reuse` is switched off: "cache_reuse is not supported by
  multimodal, it will be disabled" (`server-context.cpp:1179-1182`). On a
  recurrent model with no usable checkpoint, the log reads "forcing full
  prompt re-processing due to lack of cache data (likely due to SWA or
  hybrid/recurrent memory …)" (`server-context.cpp:3380`).

vLLM's align mode is built to cut inside an image. It requires chunked
multimodal input "because we need the flexibility to schedule a multiple of
block_size tokens even if they are in the middle of a mm input"
(`vllm/config/vllm.py:3373-3378`).

The hosted APIs:

- **Anthropic**: images are cacheable in user turns. Adding or removing one
  "anywhere in the prompt" invalidates the messages cache, but not the system
  cache.
- **OpenAI**: images are part of the rendered prefix, and the documented
  explicit breakpoints sit on text blocks.
- **MPIC** ([arXiv 2502.01960](https://arxiv.org/abs/2502.01960)) reuses image
  KV at any position with partial recompute. It is approximate and
  attention-only (LLaVA-1.6-7B).

On a hybrid, the hash key is the easy half of the problem.

- RFC #55697: "The engine cannot know through geometry alone whether an image
  belongs to `Item 1` or `Candidate 2`".
- A recurrent block grid of 528-2096 tokens does not land on an image
  boundary.
- llama.cpp refuses to checkpoint right after an image (#26994).
- vLLM's only automatic media predictor (#56818, draft) sits after the
  *final* image.
- ignis's own floor walks back out of a media item (#193).

### 6. Layout: shared context first, question last

Vendor guidance conflicts and is not backed by published data:

- Anthropic: "Put longform data at the top … Queries at the end can improve
  response quality by up to 30 percent in tests"
  ([prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices.md)).
  No published test matches that figure.
- Gemini: "put your query / question at the end of the prompt"
  ([long context](https://ai.google.dev/gemini-api/docs/long-context)).
- OpenAI's GPT-4.1 guide (2025-04-14) is the one contrary voice: "If you'd
  prefer to only have your instructions once, then above the provided
  context works better than below"
  ([cookbook](https://cookbook.openai.com/examples/gpt4-1_prompting_guide)).
  It gives no numbers.

Measured:

- *Lost in the Prompt Order* ([arXiv 2601.14152](https://arxiv.org/abs/2601.14152),
  21 models 0.5-9B): context before question and options beats the reverse
  "by over 14%p". The causal mask hides the context from the option tokens.
- LongReason ([arXiv 2501.15089](https://arxiv.org/abs/2501.15089), Table 3),
  accuracy %, question after / question before the context:

  | model | 8K | 64K | 128K |
  |---|---|---|---|
  | GPT-4o | 77.30 / 75.41 | 74.02 / 65.32 | 73.39 / 65.95 |
  | Claude-3.5 Sonnet | 73.01 / 68.60 | 68.22 / 64.69 | 65.95 / 66.20 |

  At 128K, Claude-3.5 is a tie.
- For vision models, image first with the question after is measured better
  on Qwen3-VL-8B:
  - NaturalBench group accuracy 0.351 against 0.270
    ([arXiv 2607.15565](https://arxiv.org/abs/2607.15565), Table 1);
  - MMStar 63.9 against 44.9
    ([arXiv 2607.20351](https://arxiv.org/abs/2607.20351)).

  Anthropic's vision docs say the same: "Claude works best when images come
  before text."
- System vs user role: the measured effects are small and depend on model and
  task. [arXiv 2608.08254](https://arxiv.org/html/2608.08254) (48 items):
  definitions in the system role help GPT-4.1/5.4 and hurt Haiku 4.5.

**This repository measured the same direction.** The shipped JSON layout puts
the evidence in the system block and the question after it. It scored 0.963
balanced accuracy, against 0.934 for criterion-first, on SemIf's 144 rows on
this model
([`2026-09-20-the-evidence-belongs-in-the-system-block.md`](2026-09-20-the-evidence-belongs-in-the-system-block.md)).
Role and order changed together in that sweep.

## Finding

**Observed facts.**

1. Every engine that reuses a hybrid's prefix can resume only where it
   snapshotted, and each chooses snapshot positions by one or more of four
   rules:
   - *structure*: a message, turn, system or media boundary;
   - *grid or tail*: fixed intervals, or the last block;
   - *an observed fork*: the second occurrence reveals it, the third pays;
   - *a client hint*.

   None predicts a static head inside a user turn on its first occurrence
   without a hint.
2. The observed-fork detectors in production code (vLLM #37898, SGLang
   `mamba_branching_seqlen`) use **the attention layers' dense KV index as the
   history**: they need no separate radix of past prompts. The KV already
   records where requests diverged; only the recurrent state is missing.
3. Client hints have converged: ≤4 writes, a 20-block backward lookback,
   writes only at the hint, and a marker on a content block (Anthropic,
   Alibaba, OpenAI GPT-5.6, OpenRouter translating between them). The one
   hybrid engine proposal (vLLM RFC #55697) uses an in-band token instead of
   an API field. **No merged public engine turns `cache_control` into a
   recurrent checkpoint.** The earlier internal survey's B4 idea is unclaimed
   ground, not a copy.
4. Fan-out is served in one of two ways:
   - *sequencing a publisher first*: SGLang's frontend hint and its runtime
     `n>1`; Anthropic's and Alibaba's "wait for the first response"; vLLM
     RFC #55697's producer/consumer;
   - *one batched pass over N suffixes*: SemIf and, by its docs, Jev.

   The batched pass is demonstrated exact-in-kind on a GDN hybrid (SemIf), up
   to BF16 batch-shape drift (5-6 of 777 argmaxes).
5. Every engine keys images by content, so text before an image is reusable
   for attention. For hybrids, the missing checkpoint around an image is an
   open bug in vLLM (#43587) and in llama.cpp (#26994).
6. The published ways to share a segment that does *not* sit at the prefix
   on a hybrid are all lossy: HYPIC, LinearKV, PICASO, Tail-Replay.
7. Most measurements put shared context first and the question last, for
   text and for images. Vendor advice disagrees (OpenAI GPT-4.1 against
   Anthropic and Gemini).

**Inferences (not established by any source).**

- P3(a), "first text part", is a structural predictor of the same family as
  llama.cpp's user-message starts and vLLM #56818's final-media boundary. It
  mirrors #56818: the boundary before the first media item rather than after
  the last one. Nobody has published it.
- P3(c) is what vLLM, SGLang and DeepSeek already run. Its cost profile is
  known: nothing on the 2nd request, a hit from the 3rd. For a game agent at
  ~1 decision/s that is ~2 s of warm-up. The real constraint here is the 8
  shared retained slots, where Marconi's FLOP-efficiency admission is the only
  published policy.
- P2 needs no detection at all. ignis renders all N prompts of a fan-out
  itself, so the exact common prefix is known before prefill, as it is for
  SemIf. The first question is already sequenced like SGLang's frontend hint.
  For same-kind questions that prefix runs through the images.
- Option 1 has partial external support. Anthropic's invalidation rules keep
  a system-block cache alive across image changes, and this repo measured
  context-in-system as more accurate for JSON states. But it is a different
  measured prompt (ADR 0034's rule), and it does not share images.
- Mixed-kind fan-outs can be shared exactly only by moving the per-kind text
  *after* the state. That is Jev's API shape, Alibaba's "same image, several
  questions" advice, and the direction most layout studies favour. Whether it
  holds for this model's decision accuracy on parts is unmeasured.
- Each extra predicted boundary costs ignis one more traversal split today
  (~19 ms fixed each,
  [`2026-09-18-prompt-reuse-tax-on-short-ttft.md`](2026-09-18-prompt-reuse-tax-on-short-ttft.md)).
  Two designs avoid the split by reading a checkpoint from the chunked GDN
  kernel's intermediate states within one forward: SGLang's `_force_track_h`
  and vLLM RFC #52959. That pairing decides whether many boundaries are
  cheap.

| approach | who predicts the boundary | exact on a hybrid? | images | relation to #270 |
|---|---|---|---|---|
| ignis today (ADR 0029) | server, structure: system-block end, page floor | yes (bit-exact at the same split) | parts stay in the user turn; the floor walks out of media | P1 |
| llama.cpp context checkpoints | server, structure: every user-message start + 2 near the end | yes | none after a media batch; #26994 open | P1-like, message granularity |
| OpenAI GPT-5.6+ implicit | server, structure: latest eligible message end (+20 earlier ends) | n/a (hosted) | in the cached prefix | message-level P1 analogue |
| Anthropic / Alibaba `cache_control`, OpenAI explicit | client, ≤4 writes; server looks back 20 blocks | n/a (hosted) | cacheable; Anthropic: adding/removing images invalidates messages, not system | P3(b); first-response sequencing = today's fan-out |
| vLLM RFC #55697 `<\|mamba_checkpoint\|>` | client, in-band token | author-claimed parity (99.97% of predictions), not bit-exact; unmerged, put on hold by the one reviewing maintainer | yes, position after placeholder expansion | P3(b) + P2 producer/consumer + same-step two-phase GDN; closest to our shape |
| Marconi | server, observed fork (speculative insertion into a history radix) + last decoded token | yes | not addressed | P3(c), pays from the 3rd occurrence |
| vLLM #37898 junction | server, observed: attention-KV hit beyond the recurrent hit, block-floored | yes | mm key only on image blocks; hybrid+MM bugs open | P3(c); also P2 for later siblings |
| vLLM align tail / "all" | server, block grid + tail | yes | #40709 deadlock | fails when the tail is in the unique part (#45238) |
| vLLM #56818 (draft) | server, structure: end of the final media item | yes | yes | media-boundary predictor; the mirror of P3(a) |
| SGLang Unified Radix Cache | server, chunk grid + observed `mamba_branching_seqlen` | yes (int8 option lossy) | pad value from the hash | P3(c) |
| DeepSeek API | server: end of input, end of output, detected common prefix, fixed intervals | n/a (SWA) | — | P3(c) at production scale |
| Sparse Prefix Caching | server, DP on the overlap-depth distribution | yes | no | P3(c) refinement inside one entry |
| SemIf `score_shared` | application, by construction | yes on Qwen3.5-4B (5-6/777 BF16 drift) | text only | P2 in one batched pass |
| Jev | server-internal, undisclosed | unknown | text only | P2 |
| SGLang `select` / `fork` | client sends the prefix first | via the radix tree | — | P2 by sequencing (ignis already does it) |
| Prompt Cache (PML) | client schema | attention-only | — | P3(b) for non-prefix modules |
| CacheBlend, EPIC, Block-Attention, TurboRAG, MPIC | retriever / client | attention-only, lossy; two need fine-tuning | MPIC yes | no exact analogue |
| HYPIC, LinearKV, PICASO | client separator / chunking | lossy (1.71 pt; ≤92%) | no | the only answers to mixed-kind (non-prefix) sharing |
| Tail-Replay, DASC | none / server | lossy | no | removes or shrinks the checkpoint, not exact |

## Implications

These are relevance notes for #270, not choices.

- **P1 + P2 + P3(a/b/c) is not a novel architecture.** It is the union of
  what llama.cpp (structure), vLLM/SGLang (observed forks + tail), the hosted
  APIs (client hint) and SemIf/SGLang (fan-out by construction or
  sequencing) each do separately. What would be new is combining them under
  one 8-slot budget, and using a content-part boundary as a structural
  predictor.
- **If P3(c) is built, the attention KV is the natural detector.** vLLM and
  SGLang use it, and ignis matches prompts by token and media content already
  (ADR 0029). ADR 0029's rejection of an "observed-LCP" boundary said
  "revisit only if measurement shows shared heads extending past the system
  block". #270's repro is that measurement, and vLLM #45238 is the same
  failure seen elsewhere.
- **If P3(b) is built, the ecosystem has converged on a shape:** a
  block-level marker, ≤4 per request, writes only at markers, and a short
  backward lookback. Anthropic and Alibaba also require the first writer to
  finish before readers arrive. A decision `state` given as content parts
  would carry the marker on a text part. OpenRouter already translates
  `cache_control` ↔ `prompt_cache_breakpoint` on text blocks.
- **P2 can be exact without new scheduling.** ignis already sequences the
  first question, and SemIf's by-construction prefix (minus one token, to
  dodge a BPE merge at the joint) is the precedent for computing it rather
  than detecting it. #235's batched-suffix question stays separate: SemIf
  measured 20.03 against 10.75 decisions/s for batched against serial reuse
  on a 4B. vLLM #55875/#55876 are the only published attempt to run the
  producer and its consumers in the *same* step on a hybrid, with two GDN
  prefill phases. That would make a decision a live publisher, which ignis
  rules out today (#238). The only maintainer to review them judged the
  complexity not worth having.
- **Mixed-kind fan-outs are a layout question, not a caching one.** Any exact
  sharing needs the kind-specific instruction after the shared state. A
  re-measure of decision accuracy on this model would be the price (ADR
  0034). The external evidence leans towards that order being neutral or
  better.
- **The logit readout constrains placement.** A checkpoint can never sit at
  the last prompt position, and vLLM, llama.cpp and a GLM study all
  special-case this.

## Limits and unknowns

- **Vendor pages are undated contracts, not measurements.** The Anthropic,
  OpenAI, Alibaba, Gemini and DeepSeek pages describe hosted systems whose
  architectures are not public, so their boundary rules are API design
  evidence, not evidence about recurrent checkpoints. Anthropic's "up to 30
  percent" traces to no published test.
- **vLLM RFC #55697's numbers are the author's**, on unmerged PRs. Its
  7.58x is a one-kernel microbenchmark, and its accuracy claim is 99.97%
  prediction parity, not bit-exactness. The only maintainer to review the
  PRs has put the implementation on hold, and the RFC has no maintainer
  reply. PR #56818 is a draft. Marconi is evaluated against its authors' own extensions of
  vLLM and SGLang. Sparse Prefix Caching's wall-clock is one layer group.
- **"Exact" varies across sources.** SemIf's parallel suffixes move 5-6 of 777
  argmaxes in BF16. ADR 0029's own claim is bit-exactness against a cold
  prefill split at the same boundary, not against an unsplit one.
- **Jev's serving is undocumented** beyond "ingests the state once". The P2
  reading of its 32k limit is an inference.
- **The layout studies are mostly on ≤9B or 2024-2025 models**, and none on
  Qwen3.8. The repo's own 0.963/0.934 sweep changed role and order together
  and used JSON states, not content parts.
- **Only some sources were re-read line by line.** A delegated reader read
  the modular-reuse papers (§4: Prompt Cache, CacheBlend, EPIC,
  Block-Attention, RAGCache, TurboRAG, Parrot, BatchLLM, PICASO) from their
  PDFs, and the constrained-generation frameworks (§1) from their code. Five
  of the §4 quotes were spot-checked against the text; the rest were not.
  Every vLLM, SGLang and llama.cpp line reference cited here was re-read at
  the pinned commit. In §6, the LongReason, 2607.15565, 2607.20351 and
  2608.08254 numbers were re-checked against the arXiv HTML. Batch
  prompting's accuracy range was not.
- **What nobody has solved**, as far as these sources show:
  - exact sharing of one state across *different* instructions placed before
    it, on a hybrid;
  - a first-occurrence prediction of a non-structural static head, without a
    hint;
  - a merged engine that maps `cache_control` to a recurrent checkpoint;
  - a measured checkpoint placed right *before* an image on a hybrid;
  - an admission policy for a single-GPU budget of a few large slots shared
    by chat checkpoints and decision prefixes.

## Follow-ups

- [#270](https://github.com/gpillon/ignis/issues/270) owns the design choice.
  This finding is its prior-art input.
- [#235](https://github.com/gpillon/ignis/issues/235) (batched suffixes)
  gains a hybrid precedent (SemIf on Qwen3.5-4B). It also gains a same-step,
  two-phase GDN design (vLLM #55876), which that engine's maintainer held
  back as too complex.
- Worth watching: vLLM #55697 / #55873-#55876 (client marker),
  [#52959](https://github.com/vllm-project/vllm/issues/52959) (checkpoint
  inside one forward), #56818 (media boundary), and llama.cpp #26994.
