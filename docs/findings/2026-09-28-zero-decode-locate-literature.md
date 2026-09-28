# Zero-decode locate: the literature finds the region, and resolves an instance only by shortening, re-rendering or copying

- Kind: research
- Status: current
- Observed: 2026-09-28
- Last verified: 2026-09-28
- Scope: `/v1/decide` `locate` over long text states (real logs, 16K-200K tokens), read in one prefill with no token generated: attention-based retrieval and re-ranking, KV-selection scores, separators and landmarks, readouts and query placement, probes and labelled one-pass routes; log parsing and log QA for the method paper
- Related: [Locating a line in real logs](2026-09-28-locating-a-line-in-real-logs.md) (the failure this pass reads against); [The instance is read while copying](2026-09-27-the-instance-is-read-while-copying.md); [Locate at length](2026-09-27-locate-at-length.md); [A span from attention: the literature](2026-09-27-span-from-attention-literature.md) (the previous pass: its verified numbers are reused here, marked *(verified 09-27)*); [spec 21](../specs/decide/21-locate-in-real-logs.md)
- Superseded by: none

## Question

One prefill of the served 27B hybrid (16 GQA layers of 24 query heads
among Gated DeltaNet layers) and a plurality vote of 32 heads, read at the
copy scaffold `{"quote":"` less a content-free twin, names the line an
instruction asks for 91-94% of the time on short states (< 4.5K tokens) and
34.5% on real logs at 16K-200K. What is known of why:

- the heads find the right *kind* of line and lose the *instance* among
  near-duplicates (misses 61% with no sibling, 81% with 6+), more so with
  length;
- about half of the attention row stays on the state at every length, so
  each line's share falls roughly as 1/length;
- some heads peak on the target's last keys or the separator and first keys
  after it; no head marks the start (≤ 4%);
- teacher-forcing the target, the heads settle where its written prefix
  becomes unique (rho 0.59); folding near-duplicates into templates and
  *generating* reads 94.8%, folding and voting 28/58.

Which published methods read relevance without decoding, what signal and
which heads and positions they use, what they report, and which of them
could give a better zero-decode `locate` in this engine — which today reads
raw `q·k` for up to 32 heads over any key range at the **last** prompt
position only, and has offline dumps of per-segment softmax shares for all
384 heads on ~120 questions and full score rows for 32 heads? And, for the
method paper: log parsing and log question answering.

## Evidence

### How the sources were read

- Primary sources only: arXiv abstracts (fetched verbatim through the arXiv
  API), arXiv HTML and PDF full texts, the ACL/ICLR/ICML venue as the arXiv
  comment states it; for Drain, its dblp record (found by search) and the
  logparser documentation. No blog or third-party summary.
- **Checked against the raw text** (downloaded, converted to text, the
  sentence or table row found by search): SnapKV, TOVA, Quest (Table 1),
  DuoAttention, RazorAttention, InfiniRetri, AttentionRAG, InfLLM,
  LongHeads, SepLLM, Label Words (numbers), BlockRank (Table 3), Prompt
  Repetition, Lu et al., BAP, LOVA, Lost in the Middle §4.2, the LogPAI
  benchmark (Table IV), Loghub-2.0, LogQA (dataset sizes), RULER (the
  distractor sentence, re-fetched verbatim).
- The rest was read through a summarising fetcher; *(extraction only)* marks
  details read that way once. Abstract-level claims are quoted from the
  verbatim abstract.
- *(verified 09-27)* marks numbers checked against raw text in the previous
  pass and not re-read here.
- Venues are given only where the arXiv comment or the proceedings state
  them.
- Downloads were not committed (the session scratchpad, and the untracked
  `.scratch/lit-zero-decode/`).

### 1. Retrieval and re-ranking by attention, without decoding

- **ICR**, Chen, Jiménez Gutiérrez, Su, [2410.02642](https://arxiv.org/abs/2410.02642),
  ICLR 2025. Signal: the change in attention the query causes, read at
  **every query token** (averaged), **all layers and heads summed**, a
  content-free `N/A` pass subtracted per token. "ICR only requires two
  (O(1)) forward passes to re-rank N documents"; "cutting the latency by
  more than 60%" against RankGPT. Its ablation *(verified 09-27)*, Llama-3.1-8B
  single-/multi-hop: full 57.1/64.3; last query token only 46.3/47.2; last
  token uncalibrated 40.5/45.7. → Ours: our reading is ICR's weakest
  ablation cell (last position); ICR is the model for reading at more
  positions.
- **QRHead**, Zhang, Yin, Yen, Chen, Ye, [2506.09944](https://arxiv.org/abs/2506.09944),
  EMNLP 2025. Heads ranked by query-token attention mass on gold documents
  over "a handful of examples from real-world tasks"; 16 heads (32 at 70B);
  scores are the heads' accumulated attention mass. BEIR (Llama-3.1-8B)
  50.0 vs ICR 48.5; head choice random 37.5, all 42.8, Wu's retrieval heads
  43.4, QR 47.5 *(verified 09-27)*; "over 10% performance gains over full
  context" on LongMemEval and CLIPPER by selecting parts (abstract).
  → Ours: a head-selection rule scored on the failing distribution.
- **Retrieval heads**, Wu, Wang, Xiao, Peng, Fu, [2404.15574](https://arxiv.org/abs/2404.15574).
  Defined **while decoding** the needle (copy-paste); "less than 5%" of
  heads; in Llama-2 7B "12 retrieval heads always attend to the required
  information"; pruning them makes retrieval fail. → Ours: a prior on
  heads, not a prompt-position reading.
- **InfiniRetri**, Ye, Wang, Wang, [2502.12962](https://arxiv.org/abs/2502.12962).
  Signal: attention from the **question's tokens** in the **last layer**,
  heads summed, smoothed by "a 1D convolution using a kernel … filled with
  ones" (a phrase window); the top-K tokens keep their **whole sentences**
  in a cache; the text is read in chunks (defaults 15 / 300 / 1,024 tokens,
  *extraction only*). NIH "100% accuracy … over 1M tokens using a 0.5B parameter model";
  HotpotQA with Qwen2-7B-Instruct 14.8 → 57.52 (+288%) while keeping 795 of
  9,152 tokens. → Ours: select by attention, expand to the unit, **re-read
  short**; the answer still comes from generation.
- **Attention sorting**, Peysakhovich & Lerer, [2310.01427](https://arxiv.org/abs/2310.01427).
  "Even when models fail to use the information from a relevant document …
  they still pay preferential attention to that document"; one decoding
  step, sort documents by attention, repeat. One decoded token, not zero.
- **AttentionRAG**, Fang, Sun, Shi, Gu, [2503.10720](https://arxiv.org/abs/2503.10720).
  The query becomes an answer hint prefix ("Daniel is in the"), one anchor
  token is **generated**, its attention **summed over all layers** scores
  the context; sentences holding the top-k tokens are kept. "Up to 6.3×
  context compression while outperforming LLMLingua methods by around 10%"
  (abstract). → Ours: our scaffold is the same idea without the generated
  token.
- **AT2**, Cohen-Wang, Chuang, Madry, [2504.13752](https://arxiv.org/abs/2504.13752).
  Heads' attention weights as features, one learned coefficient per head
  fitted to ablation effects; figures only *(verified 09-27)*.
- **Found in the Middle**, Hsieh et al., [2406.16008](https://arxiv.org/abs/2406.16008),
  ACL Findings 2024. "A U-shaped attention bias where the tokens at the
  beginning and at the end of its input receive higher attention,
  regardless of their relevance"; calibration subtracts a dummy document's
  attention at the same position; NQ recall@3, gold in the middle, 20
  documents: 0.2052 → 0.6832 *(verified 09-27)*; up to 15 points in RAG
  (abstract). → Ours: the content-free twin already subtracts per key at
  the same position.
- **BlockRank**, Gupta, You, Bhojanapalli, Kumar, Dhillon, Yu,
  [2510.05396](https://arxiv.org/abs/2510.05396), NeurIPS 2025 (arXiv
  journal-ref). **Fine-tuned** Mistral-7B. "Last and some specific query
  tokens like ':' … develop strong attention weights towards relevant
  document tokens, particularly in the model's middle layers"; the prompt is
  `{Inst}. {d_1 … d_N} {q}` where Inst "can also include the query", because
  "including the query in Inst allows the model to condition each
  document's representation on the specific information need from the
  outset" (their hypothesis). MSMarco, N = 50, P@1 decode / attention
  (Table 3): full fine-tune 28.7 / 27.6; BlockRank 28.7 / 29.1. 4.7× faster
  at 100 documents; 500 documents (~100K tokens) in 1.15 s. → Ours: in a
  model fine-tuned for ranking, reading attention at signal tokens equals
  or beats decoding; the query is placed **before** the documents too.
- **Atlas of ICL**, Kahardipraja et al., [2505.15807](https://arxiv.org/abs/2505.15807),
  NeurIPS 2025. The last prompt position, retrieval heads weighted by a
  learned `w_h` times each head's logit-lens score: top-1 localisation
  97 / 96 / 84% (Llama-3.1-8B, Mistral-7B, Gemma-2-9B); plain averaging
  about 10 points lower *(verified 09-27)*.

### 2. KV-cache selection: relevance from an observation window

| method | score read | pooling, heads | reported |
|---|---|---|---|
| **H2O**, Zhang et al., [2306.14048](https://arxiv.org/abs/2306.14048) | attention each key has **accumulated** over the queries seen, per layer (Alg. 1) *(extraction only)* | keep heavy hitters + recent tokens | heavy hitters "strongly correlate with the frequent co-occurrence of tokens"; 20% heavy hitters, throughput up to 29× (abstract) |
| **TOVA**, Oren, Hassid, Yarden, Adi, Schwartz, [2401.06104](https://arxiv.org/abs/2401.06104) | drop "the token with the lowest attention score" at each step | "averaging the attention scores across the heads of a given layer is superior to considering each head individually" (preliminary) | "in some cases only 1/8 of the original cache size" nearly on par with the full model, 4.8× throughput; "the first token is kept until the end"; "punctuation and other special symbols tend to be kept"; "only 73-76% of the tokens are recent" |
| **SnapKV**, Li et al., [2404.14469](https://arxiv.org/abs/2404.14469) | `C = Σ_{i ∈ window} W_obs[:, i, :]`: attention from the prompt's **last 16-64 tokens**, summed, **per head** | 1D **max-pool** ("clustering"), kernel 5-13; top-k per head | pooling "significantly enhances retrieval accuracy" (figure); "hit rates are consistently high regardless of whether instructions are positioned before or after extensive supplementary contexts"; 3.6× decoding speed, 8.2× memory at 16K (abstract) |
| **Quest**, Tang et al., [2406.10774](https://arxiv.org/abs/2406.10774), ICML 2024 | per 16-token page, channel-wise min `m` and max `M` of the keys; `Σ_i max(q_i m_i, q_i M_i)`, "always greater than any product of Q_i with the Key value K_i for all tokens in this page regardless of the sign of Q_i" | top-K pages, per head | 100K passkey (Yarn-Llama-2-7b-128k), 1,024-token budget: Quest 96%, H2O 2%, TOVA 2%, StreamingLLM 1% — eviction "incorrectly discard[s] the KV cache of the answer before receiving the question" (Table 1) |
| **PyramidKV**, Cai et al., [2406.02069](https://arxiv.org/abs/2406.02069) | SnapKV-style scores | budget per layer: attention "scattering widely in lower layers … ultimately focusing on critical tokens in higher layers" | 12% of the KV cache matches full LongBench (abstract, *extraction only*) |
| **DuoAttention**, Xiao et al., [2410.10819](https://arxiv.org/abs/2410.10819) | a trainable gate per KV head blending full and streaming attention, optimised on synthetic passkey data (distillation + L1, *extraction only*) | retrieval heads keep full KV | 25% retrieval heads for Llama-2-7B (MHA), 50% for Llama-3-8B (GQA); the optimisation beats attention profiling at identifying them (Fig. 13) |
| **RazorAttention**, Tang et al., [2407.15891](https://arxiv.org/abs/2407.15891) | **echo** heads (attend to a previous token identical to the current one) and **induction** heads (attend to the token after it), scored on 2,500 random tokens repeated 4× *(extraction only)* | keep ~14% induction + 1% echo heads full; others local + one compensation token = the mean of the dropped keys and values *(extraction only)* | "only about 15% of the heads … are capable of effectively utilizing long-range information"; > 70% KV reduction; adding 1% echo heads "significantly enhances the retrieving performance" (Fig. 5) |
| **HeadKV**, Fu et al., [2410.19258](https://arxiv.org/abs/2410.19258), ICLR 2025 | per-head importance from a retrieval-and-reasoning estimate | head-level budgets | 1.5% of the KV cache, 97% of full on contextual QA (abstract) |
| **InfLLM**, Xiao et al., [2402.04617](https://arxiv.org/abs/2402.04617) | 128-token memory units; each unit's **representative tokens** are those with the highest mean score from the local window's queries, `r_m = (1/l_L) Σ_j q_{m+j}·k_m` | unit relevance `Σ q·k` over its representatives; top units loaded | "the computation of representative scores requires no additional parameters"; 1,024K passkey 100%; ∞-Bench 57.7 vs 21.5 for StreamingLLM on Mistral-7B (formulas and numbers *extraction only*) |
| **LongHeads**, Lu et al., [2402.10685](https://arxiv.org/abs/2402.10685) | 256-token chunks; a chunk's key is attention-pooled with the chunk's mean output as query *(extraction only)* | each head picks its top chunks plus the first and last | mean pooling of keys "demonstrated suboptimal performance in preliminary experiments, particularly in selecting the correct chunks"; "100% accuracy at the 128k length" on passkey with a 2K window, no training |

→ Ours: the observation-window methods are the zero-decode relevance
signal written as engineering: a window of query positions, per-head
scores, a local max-pool, a per-head top-k, query-aware beating
query-agnostic by a wide margin (Quest). None is evaluated on picking one
instance among near-duplicates: they keep hundreds to thousands of tokens.

### 3. Separators, anchors, landmarks, span ends

- **SepLLM**, Chen et al., [2412.12094](https://arxiv.org/abs/2412.12094),
  ICML 2025. Separators (". , ? ! ; : space \t \n") "contribute massive
  attentions"; "segment information is compressed and embedded into these
  separator tokens". Training-free, keeping only initial, separator and
  neighbouring tokens: GSM8K-CoT (Llama-3-8B) 77.18 vs 77.79 with 47.36% of
  the KV.
- **Label words are anchors**, Wang et al., [2305.14160](https://arxiv.org/abs/2305.14160),
  EMNLP 2023 (GPT2-XL, GPT-J). "Semantic information aggregates into label
  word representations during the shallow computation layers' processing";
  deep layers read them. Blocking the flow into label words in shallow
  layers impairs the model, in deep layers it is inconsequential
  *(extraction only)*. Anchor re-weighting "a 16.7% average accuracy boost";
  compression "1.8× speedup".
- **Gather-and-Aggregate**, Bick, Xing, Gu, [2504.18574](https://arxiv.org/abs/2504.18574):
  "the newline token (\n) at the end of each answer choice acts as a summary
  token"; aggregate heads read it *(verified 09-27)*.
- **LLM-Microscope**, Razzhigaev et al., [2502.15007](https://arxiv.org/abs/2502.15007),
  NAACL 2025: determiners and punctuation "carry surprisingly high context";
  removing stopwords, articles and commas lowers MMLU and BABILong-4k
  *(extraction only)*.
- **Punctuation and Predicates**, Chauhan et al., [2508.14067](https://arxiv.org/abs/2508.14067):
  "for GPT-2, punctuation is both necessary and sufficient in multiple
  layers, while this holds far less in DeepSeek and not at all in Gemma" —
  the summary role is model-specific.
- **Extracting Paragraphs**, Pochinkov et al., [2409.06328](https://arxiv.org/abs/2409.06328):
  patching the `"\n\n"` activation "can transfer significant information
  about the context of the following paragraph" — a separator can also
  look *ahead*.
- **Token erasure**, Feucht, Atkinson, Wallace, Bau, [2406.20086](https://arxiv.org/abs/2406.20086):
  "last token representations of named entities and multi-token words
  exhibit a pronounced 'erasure' effect" — the unit is represented at its
  last token (Llama-2-7b, Llama-3-8B).
- **Span ends and offsets** *(verified 09-27)*: induction heads land one
  token after the match (Olsson et al., [2209.11895](https://arxiv.org/abs/2209.11895));
  "concept induction heads learn to attend to the ends of multi-token
  words" (Dual-Route, [2504.03022](https://arxiv.org/abs/2504.03022), COLM
  2025); copy suppression attends to an earlier copy and writes it down
  ([2310.04625](https://arxiv.org/abs/2310.04625)); retrieval heads sit on
  the span's first token at the first copy step. Razor's echo (offset 0)
  and induction (+1) heads are the same pair of offsets.
- **Hybrids** *(verified 09-27)*: in GDN-attention hybrids the local
  "carrying" that prepares an attention layer's keys "concentrates on lag
  one: the token immediately before the historical value" (Cheng et al.,
  [2609.15545](https://arxiv.org/abs/2609.15545), incl. Qwen3.5-4B).
- **Trained summaries**: Landmark Attention (Mohtashami & Jaggi,
  [2305.16300](https://arxiv.org/abs/2305.16300), NeurIPS 2023) inserts a
  landmark after each block with a grouped softmax and needs fine-tuning
  (LLaMA 7B, 15,000 steps; 98% passkey at 32K, *extraction only*);
  Anchor-based LLMs ([2402.07616](https://arxiv.org/abs/2402.07616), ACL
  2024) train sequences into an anchor token ("up to 99% keys/values cache
  reduction"). Registers ([2309.16588](https://arxiv.org/abs/2309.16588))
  are the vision analogue: ViTs repurpose "low-informative background"
  tokens for internal computation unless given spare tokens. Sinks are
  positional (StreamingLLM, [2309.17453](https://arxiv.org/abs/2309.17453),
  ICLR 2024) and output gating shrinks them (46.7% → 4.8% first-token share,
  Qiu et al.) *(verified 09-27)*.
- **Not found**: a head defined by attending to the end or boundary of a
  multi-token *line* to be copied, read at a prompt position; any analysis
  of separators or line summaries in a Qwen3.x model.

### 4. Readouts: values, flow, gradients, positions, placement

- **Value- and flow-aware** *(verified 09-27)*: Kobayashi et al.
  ([2004.10102](https://arxiv.org/abs/2004.10102)) weight by `‖α f(x)‖`,
  `f(x) = (xW^V + b^V)W^O` — high-attention [SEP], [CLS] and punctuation
  contribute little; rollout / flow (Abnar & Zuidema,
  [2005.00928](https://arxiv.org/abs/2005.00928)) raise Spearman with
  blank-out from 0.29 to 0.71 on a small encoder; no published rollout
  through Gated DeltaNet layers. **AlignedWVA** ([2602.01572](https://arxiv.org/abs/2602.01572)):
  "the attention scores of the last token function as the weights, while
  the output projection matrix (W_O) aligns these weighted value vectors",
  the best training-free LLM sentence embedding there (abstract).
- **Gradient hybrids**: AttnLRP ([2402.05602](https://arxiv.org/abs/2402.05602))
  and Chefer et al. ([2012.09838](https://arxiv.org/abs/2012.09838))
  propagate relevance through attention and skip connections, at about one
  backward pass. The engine has no backward pass.
- **Which query positions**: all query tokens (ICR, QRHead), the last 16-64
  prompt tokens (SnapKV), the question's tokens in the last layer
  (InfiniRetri), signal tokens `:` and `[` in middle layers (BlockRank), one
  generated token (AttentionRAG, attention sorting), the last prompt
  position (Atlas, and ours).
- **Query before the data.** Lost in the Middle (Liu et al.,
  [2307.03172](https://arxiv.org/abs/2307.03172), §4.2, raw text):
  "placing the query before and after the data, enabling query-aware
  contextualization … dramatically improves performance on the key-value
  retrieval task — all models achieve near-perfect performance on the 75,
  140, and 300 key-value pair settings"; without it "the worst-case
  performance is 45.6%"; but it "minimally affects performance trends in
  the multi-document question answering task".
- **Prompt repetition**, Leviathan, Kalman, Matias, [2512.14982](https://arxiv.org/abs/2512.14982):
  `<QUERY><QUERY>` so that "each prompt token [can] attend to every other
  prompt token"; "wins 47 out of 70 … with 0 losses"; on NameIndex (the
  25th of 50 names) Gemini 2.0 Flash-Lite 21.33% → 97.33%; "smaller
  improvements for the multiple-choice benchmarks with question-first, and
  larger improvements with options-first"; prefill-only cost (latency grew
  only for Claude on very long requests).
- **Echo embeddings** ([2402.15449](https://arxiv.org/abs/2402.15449), ICLR
  2025): embeddings read from a repeated copy, "over 5%" better zero-shot.
  **Re2** ([2309.06275](https://arxiv.org/abs/2309.06275), EMNLP 2024):
  re-reading the question gives a "bidirectional" encoding. **PartRep**
  ([2607.01792](https://arxiv.org/abs/2607.01792)): repeating only
  high-NLL tokens keeps "most of the gains of full repetition while using
  only 59.4% of its KV cache and 79.0% of its prefill FLOPs".
- **TimeStampEval**, McCammon, [2511.11594](https://arxiv.org/abs/2511.11594)
  — the nearest published *locate a line* task (sentence-timestamped
  transcripts, 120K tokens): "placing the query before the transcript and
  using compact formatting improved accuracy by 3-20 points while reducing
  token count by 30-40%"; "off-by-one errors form a distinct category";
  a fuzzy pre-filter plus LLM verification on short snippets "improves
  fuzzy match accuracy by up to 50 points"; 95-100% rejection of absent
  targets at 50K-900K.

### 5. Probes and labelled one-pass routes

- **Know but don't tell**, Lu, Gao, Yu, Byerly, Khashabi, [2406.14673](https://arxiv.org/abs/2406.14673).
  A linear probe per layer on the **last token's** embedding predicts the
  gold key-value pair (100 pairs) or document (30) ID. "LLMs encode the
  position of target information, [but] often fail to leverage this in
  generating accurate responses". On key-value pairs probing "reaches
  perfect accuracy at layer 13"; on MDQA "most gold IDs achieve peak
  accuracy around layer 18" (the main text's model is
  Mistral-7B-Instruct-v0.3);
  "early-layer information localization leads to higher generation
  accuracy" (p < 5e-5).
- **BAP**, Stein et al., [2502.13966](https://arxiv.org/abs/2502.13966): a
  one-layer transformer-decoder probe over a frozen LLM's hidden states,
  trained with "only weak supervision" (bug / no-bug labels); a line's
  score is the sum of the probe's attention over its tokens (architecture
  *extraction only*). +34.6% top-1 over the
  strongest baseline across eight datasets (abstract); Defects4J top-1
  0.334. "On code fragments of 60 lines and longer, all methods perform
  near random."
- **LOVA**, Li et al., [2410.15288](https://arxiv.org/abs/2410.15288): the
  last prompt token's attention summed per line per layer, the difference
  between a prompt with a line highlighted and the base prompt (one pass
  per highlighted line), then a trained Bi-LSTM *(extraction only)*; "up to
  a 5.3x improvement in F1" (abstract).
- **Binding and lookback IDs**: entities and attributes carry binding ID
  vectors whose "distances … reflect their discernability" (Feng &
  Steinhardt, [2310.17191](https://arxiv.org/abs/2310.17191)); ordering IDs
  and a binding lookback then an answer lookback (Prakash et al.,
  [2505.14685](https://arxiv.org/abs/2505.14685)).
- **Labelled options in one pass**: multiple-choice symbol binding — the
  natural "question and options, output the symbol" format beats cloze
  scoring when the model binds symbols well, an ability that "varies
  greatly by model" (Robinson, Rytting, Wingate, [2210.12353](https://arxiv.org/abs/2210.12353),
  ICLR 2023); selection bias comes from "token bias" on option IDs, removed
  label-free by a prior estimated from permuted options (PriDe, Zheng et
  al., [2309.03882](https://arxiv.org/abs/2309.03882), ICLR 2024); the
  first identifier's logits rank every candidate, "accelerat[ing]
  inference by 50%" with a learning-to-rank loss (FIRST, Reddy et al.,
  [2406.15657](https://arxiv.org/abs/2406.15657)); Chinchilla's
  correct-letter heads read the options' summary tokens *(verified 09-27)*.

### 6. Near-duplicates and interference

- **RULER**, Hsieh et al., [2404.06654](https://arxiv.org/abs/2404.06654),
  COLM 2024. In MK-NIAH "the additional 'needles' are hard distractors";
  "increasing the number of distracting needles steadily lowers
  performance, with Yi dropping by ∼40 points at 256K in the extreme
  version, where the context is full of irrelevant needles"; Yi "incorrectly
  retrieves values associated with the distractor keys" *(extraction only)*.
- **PI-LLM**, Wang & Sun, [2506.08184](https://arxiv.org/abs/2506.08184),
  ICML 2025 LCFM workshop. Key-value updates streamed, only the last value
  asked, "clearly positioned just before the query": accuracy "declines
  log-linearly toward zero as interference accumulates; errors arise from
  retrieving previously overwritten values"; prompting to forget yields
  "limited success" (46 keys, 3-400 updates each, *extraction only*).
- **The First Drop of Ink**, Gao, Chen, Huang, [2605.10828](https://arxiv.org/abs/2605.10828):
  "hard distractors capture disproportionate attention even at small
  proportions"; "filtering gains mainly come from context-length reduction
  rather than distractor removal; substantial recovery requires reducing
  the hard-distractor proportion to near zero".
- **NoLiMa** ([2502.05167](https://arxiv.org/abs/2502.05167), ICML 2025):
  without literal matches GPT-4o falls from 99.3% to 69.7% at 32K, which
  its authors attribute to "the increased difficulty the attention
  mechanism faces in longer contexts".

### 7. Log parsing and log question answering

- **Drain**, He, Zhu, Zheng, Lyu, ICWS 2017 (dblp): online, a
  fixed-depth parse tree; the next node is chosen "by the tokens in the
  beginning positions of the log message", then the similarity to each
  group's template decides the group; regex preprocessing (logparser
  documentation).
- **LogPAI benchmark**, Zhu et al., [1811.03509](https://arxiv.org/abs/1811.03509),
  ICSE 2019: 13 parsers on 16 LogHub datasets, 2,000 hand-labelled
  messages each; parsing accuracy = the fraction of messages whose template
  groups the same messages as the ground truth. Drain has the highest
  average, 0.865 (Table IV), high (> 0.9) on 9 of 16 datasets, and the
  smallest variance.
- **Loghub**, Zhu, He, He, Liu, Lyu, [2008.06448](https://arxiv.org/abs/2008.06448),
  ISSRE 2023: 19 real-world datasets. **Loghub-2.0**, Jiang et al.,
  [2308.10828](https://arxiv.org/abs/2308.10828), ISSTA 2024: 14 datasets,
  3.6 million lines each on average; "all existing parsers demonstrate a
  significant degradation"; Drain's average FGA drops "from 0.75 to
  approximately 0.55", still the highest GA and FGA; Drain "achieves an
  average score exceeding 0.95 on all four metrics on logs without
  parameters".
- **LILAC**, Jiang et al., [2310.01796](https://arxiv.org/abs/2310.01796),
  FSE 2024: LLM parsing with in-context demonstrations and an adaptive
  template cache, +69.5% average F1 of template accuracy, LLM queries cut
  "by several orders of magnitude". ChatGPT as a parser: Le & Zhang,
  [2306.01590](https://arxiv.org/abs/2306.01590), ASE 2023 NIER.
- **Log QA**: LogQA (Huang et al., [2303.11715](https://arxiv.org/abs/2303.11715)),
  a trained retriever and an extractive reader, 247 / 188 / 397 labelled
  questions on HDFS / OpenSSH / Spark, answers are spans in log lines.
  LogEval ([2407.01896](https://arxiv.org/abs/2407.01896)): parsing,
  anomaly detection, fault diagnosis, summarisation, 4,000 entries and 15
  prompts per task — no locate task. LogNLQ ([2607.03884](https://arxiv.org/abs/2607.03884)):
  logs parsed into "template-partitioned relational tables", templates and
  parameter columns annotated, the LLM writes SQL; LogNLQ-Bench, 8,895
  execution-verified queries over four datasets. LogRouter
  ([2605.18015](https://arxiv.org/abs/2605.18015)): Drain3 ingestion,
  routing among keyword search, template lookup with SQL and semantic
  retrieval; 70 questions on four LogHub datasets, router accuracy 88.4%.
  LLM4Log ([2604.16359](https://arxiv.org/abs/2604.16359)) reviews 145
  papers to November 2025; its abstract's task list names neither QA nor
  locating a line.
- **Not found**: a public benchmark of locating one line among
  near-duplicates in a long raw log, or an attention-based log-line
  locator.

## Finding

### Observed in the sources

1. **Every training-free zero-decode relevance reading that works at length
   reads more than the last position, or pools**: all query tokens (ICR,
   QRHead), a 16-64-token window with a local max-pool (SnapKV), the
   question's tokens with a phrase kernel (InfiniRetri). The last position
   alone is ICR's weakest measured cell (−10.8 / −17.1 points).
2. **What these methods return is a region** — a document, a page, a
   chunk, a sentence — and they hand it to generation. None reports picking
   one instance among near-duplicates; their needles sit in haystacks
   without siblings.
3. **Query-aware beats query-agnostic by a wide margin** (Quest 96% against
   H2O / TOVA 2% at 100K), and a null or position calibration is part of
   every attention relevance method.
4. **Heads**: a minority carries long-range retrieval (< 5% by copy-paste,
   ~15% by induction and echo, 25-50% by an optimised gate); heads chosen
   by query-focused scoring on real tasks beat copy-paste retrieval heads
   (QRHead); an optimisation on synthetic data beats attention profiling
   (DuoAttention); averaging a layer's heads beats single heads for
   eviction (TOVA, preliminary).
5. **Segment ends and separators collect segment information** in many
   models (SepLLM, label words, Gather-and-Aggregate, token erasure,
   concept induction heads), not in all (Gemma, per Punctuation and
   Predicates), and a separator can also carry the *next* segment.
6. **Near-duplicate interference defeats generation too**: RULER's
   distractor needles (~40 points at 256K), PI-LLM's overwritten values
   (log-linear to zero), the first drop of ink. The measured recoveries
   come from a shorter context or fewer distractors, not from prompting.
7. **Putting the question before the data** rescues exact key-value
   retrieval and list indexing (Lost in the Middle; NameIndex 21 → 97%;
   TimeStampEval +3-20) but not multi-document QA; SnapKV's selection
   barely depends on where the instruction sits.
8. **Positions are linearly decodable** from the last token's residual when
   the model fails to output them (Lu et al., 30-100 fixed slots); a
   line-level attention probe goes near random past 60 lines (BAP).
9. **Logs**: Drain is the strongest classical parser and degrades on
   realistic volumes (FGA 0.75 → ~0.55); parser-induced tables (LogNLQ)
   and Drain3-based routing (LogRouter) already serve log QA; no line-locate
   benchmark exists.

### Inferred (not tested here or anywhere found)

- **Our failure is RULER's and PI-LLM's interference, seen through
  attention.** With the question after the log, near-duplicate lines'
  keys and end tokens were computed without it, and the scaffold's query
  has only the values to separate them. The literature separates
  instances in three ways, each consistent with our own data: **shorten**
  (re-read a shortlist; "length reduction" drives recovery), **let the
  keys see the question** (query-aware render), **copy** (the instance
  emerges where the written prefix becomes unique).
- **"No head marks the start" is what a causal model predicts.** A line's
  first token cannot encode the line; the line is summarised at or after
  its end (token erasure, separators), and in this hybrid the lag-one
  carrying of the GDN layer puts that summary on the next key — the
  observed peak on "the separator or first keys after it". The start is
  recoverable as the previous line's end plus one.
- **The 1/length share does not by itself explain a head's miss.** Within
  one head, every line's share has the same denominator, so the argmax is
  the argmax of the summed `exp(q·k)`. The fall matters for the model's own
  read, for combining heads with weights, and for any found/absent
  threshold — a calibration problem, not the instance problem.

## Implications

Ranked by expected gain on the measured failure (the instance among
near-duplicates at 16K-200K) per unit of cost; *established* means the
technique is published and works in its setting, never that it works here.

1. **Shortlist, then re-read short.** Signal: the served vote (or the
   lift) over the whole log yields its top-K lines — the heads already find
   the right kind; the K lines, with their original indices and order, form
   a new short state, read again by the vote (91-94% at < 4.5K) or the
   labelled `choice`. Precedent: InfiniRetri (select by attention, keep the
   sentence, re-read), QRRetriever, Quest's top-K pages; the first drop of
   ink attributes recovery to shortening. Engine: nothing new — a second
   request. Offline gate first: recall@K of the vote on R and R2 (K = 8,
   16, 32, 64, by sibling bin) from the 32-head full rows. Benefit: bounded
   by recall@K; the siblings survive into the short state, but at the length
   where the vote's instance reading is best. Cost: one extra prefill of
   ~K×40 tokens. Status: pattern established for documents, untested on
   lines.
2. **A question-aware render of the short state.** Place the instruction
   before *and* after the shortlist (or folded level-2 rows); the
   content-free twin blanks both. Signal: the lines' keys and end tokens,
   and the recurrent state, computed in the question's light. Precedent:
   Lost in the Middle's key-value retrieval (worst case 45.6% → near
   perfect), NameIndex, TimeStampEval, BlockRank's `Inst` + query. Counter
   evidence: multi-document QA unchanged; SnapKV insensitive. Engine: a
   render change. Cost: it forfeits reuse of the state across questions,
   so it is affordable only on short texts — never the 200K log (47-139 s
   prefill per question). Status: speculative here; our task is closer to
   key-value retrieval (an exact instance) than to multi-document QA.
3. **Candidate-prefix likelihood: the copy without sampling.** For each
   shortlisted (or level-2) candidate, force the scaffold plus the
   candidate's first tokens up to the depth where its prefix becomes unique
   among the candidates (a trie; the prefix stopped matching other lines
   after a median ~24-26 tokens on set R), and pick the
   largest summed log-probability. Signal: the mechanism measured on R2 —
   the instance resolves where the prefix becomes unique (rho 0.59) — and
   generation's 86-95%. Precedent: cloze scoring against symbol binding
   (Robinson et al.), FIRST's first-identifier logits. Engine: the forced
   tokens' log-probabilities over the retained state (to check whether the
   readout seam exposes them; if not, a small addition), or one
   tree-attention verify over the trie. Cost: K short prefills, or one tree
   pass. Status: highest ceiling of the list; a likelihood route, not an
   attention one.
4. **Kind × value: read attention from the question's discriminating
   tokens.** At the instruction's own value tokens (a host, a job, an id, a
   time) duplicate-token / echo heads land on that value's copies in the
   log (offset 0) and induction heads one token after (+1); intersect or
   multiply with the scaffold vote's per-line kind score. Precedent: ICR
   (its query-token lexical bias is here the wanted signal), IOI's
   duplicate-token heads, Razor's echo heads. Engine: reads at non-last
   positions — one truncated request per value token over the retained
   state, or an engine change to read several positions of one prefill.
   Cost: a question-length prefill per position (R2's retained-prefix
   question cost ~2.4 s including generation). Benefit: only for questions
   that name a value the siblings differ in (R2 has 35 combo or lexical
   present questions of 58; which name such a value is not recorded); none
   for ordinal or temporal questions. Status: speculative.
5. **Line-end (landmark) scoring.** Score a line by the keys at its end —
   last content token, the `\n` escape, the next line's first key — with
   the heads that mark ends, beside the whole-line sum and max. Precedent:
   SepLLM, Gather-and-Aggregate, label words, token erasure, InfLLM's
   representatives, lag-one carrying; on synthetic logs our end-marker
   reading lifted the vote from 91 to 94 (97 on L). Engine: nothing — the
   32-head full rows hold it. Cost: offline, free. Benefit: removes
   line-length bias; whether line-end summaries separate near-duplicates
   (rather than encode the kind) is untested. Status: speculative, and the
   summary role is model-specific.
6. **Re-select heads for length and instance.** Score heads QRHead-style
   (gold-line mass less the null) on long real-log questions; follow
   DuoAttention's lesson and select on synthetic logs built to exercise the
   failure (controlled sibling counts at 16-200K), with Razor's echo and
   induction scores as a prior and local-only heads dropped. Engine:
   nothing — the 384-head per-segment shares suffice. Cost: offline, free;
   ~120 questions demand nested CV. Benefit: modest (learned weights added
   ≤ 1.8 points on short sets; head utility is dataset-dependent), but the
   served 32 were chosen below 4.5K. Status: established technique.
7. **Folded + labelled `choice`.** Labels on the level-1 templates and the
   level-2 rows — few labels over short texts — read from the next-token
   distribution, with a PriDe-style permutation prior against label
   position bias. Precedent: MCSB, FIRST, PriDe, correct-letter heads.
   Engine: the existing labelled route. Cost: two short prefills. Benefit:
   uncertain — folded + vote read 28/58 and the split of its misses between
   level 1 and level 2 is not recorded; labels tied the vote on short sets.
   Status: established technique, cheapest GPU test on the list.
8. **Pooling and length normalisation.** SnapKV's local max-pool before
   the per-line reduction, Quest's per-line upper bound (a max logit),
   AttnTrace's top-k mean, a per-line log-sum-exp of the lift, heights
   calibrated per length. Engine: nothing. Cost: offline, free. Benefit:
   small for top-1 (per-head argmax is invariant to the denominator);
   useful for weighting heads and for the found/absent threshold (the
   vote's own confidence separates absent questions at AUC 0.72-0.73). Status: established in eviction.
9. **A learned pointer probe.** A small bilinear or attention probe
   (BAP-shaped) from the scaffold's residual against frozen line-end
   states, or a Lu-style linear probe over bucketed positions. Engine: new
   taps (hidden states or keys at line ends) and labelled data at length.
   Cost: high. Benefit: unknown — positions are decodable over 30-100 fixed
   slots, a line probe fails past 60 lines. Status: speculative.
10. **Value-, gate- and gradient-weighted readouts.**
    `α·‖W_O^h(σ(g) ⊙ v)‖`, ALTI, AttnLRP, AT2's ablation fits. Engine:
    value and gate taps, or a backward pass that does not exist. Cost:
    high. Benefit: low for this failure — they reweight what a head reads,
    they do not make near-duplicate keys separable. Status: established for
    attribution, off-target here.

1-3 compose into one zero-decode pipeline — shortlist by attention,
question-aware short render, then vote or candidate likelihood — whose
every step is short except the first read of the log, which the retained
state already pays.

**For the method paper**: Drain / LogPAI / Loghub-2.0 place the folding
step (Drain the reference parser, FGA ~0.55 at scale as the honest caveat);
LILAC, LogNLQ and LogRouter are the nearest LLM uses of templates; LogQA,
LogEval and TimeStampEval the nearest benchmarks, none a line-locate among
near-duplicates; RULER, PI-LLM and the first drop of ink the external
evidence that near-duplicate interference defeats long-context retrieval
in general.

## Limits and unknowns

- **Nothing here was measured on this model.** Almost every number comes
  from 0.5B-8B dense models (Llama, Mistral, Qwen2, GPT-2), document-level
  units of hundreds of tokens, and needles without siblings.
- The engine constraints are as the caller stated them (last-position
  `q·k`, ≤ 32 heads live; 384-head per-segment and 32-head full-row dumps
  offline); whether the readout seam can return forced tokens'
  log-probabilities (item 3) was not checked.
- Rows marked *(extraction only)* rest on one summarising read. Figure-only
  results (SnapKV's pooling ablation, DuoAttention's Fig. 13, Razor's
  Fig. 5) are cited as their authors' sentences, without numbers.
- Several sources are 2025-2026 preprints (InfiniRetri, PI-LLM, the first
  drop of ink, Prompt Repetition, PartRep, LogNLQ, LogRouter), not known to
  be peer reviewed. BlockRank is a fine-tuned model; its attention result
  does not transfer to zero-shot by itself.
- The rankings above are judgement from these sources and the repository's
  findings, not a measurement.
- **Not found**: a zero-decode method evaluated on near-duplicate
  instances; a line-level summary analysis in a Qwen3.x or GDN hybrid; a
  line-locate benchmark over raw logs.

## Follow-ups

- Offline, before any GPU run: the vote's recall@K on R / R2 (item 1),
  line-end scoring (5), head re-selection at length (6) and pooling (8) on
  the existing dumps; and the level-1 / level-2 split of folded + vote's
  30 misses from R2's rows (7).
- Then, on the GPU: the shortlist re-read under both renders (1-2), the
  candidate likelihood (3) and the question-token reading (4), on a set
  registered before it is asked, as spec 21 was.

## Sources

Retrieval and re-ranking without decoding:

- [ICR, arXiv:2410.02642](https://arxiv.org/abs/2410.02642); [QRHead, arXiv:2506.09944](https://arxiv.org/abs/2506.09944); [Retrieval Head, arXiv:2404.15574](https://arxiv.org/abs/2404.15574); [InfiniRetri, arXiv:2502.12962](https://arxiv.org/abs/2502.12962); [Attention Sorting, arXiv:2310.01427](https://arxiv.org/abs/2310.01427); [AttentionRAG, arXiv:2503.10720](https://arxiv.org/abs/2503.10720); [AT2, arXiv:2504.13752](https://arxiv.org/abs/2504.13752); [Found in the Middle, arXiv:2406.16008](https://arxiv.org/abs/2406.16008); [BlockRank, arXiv:2510.05396](https://arxiv.org/abs/2510.05396); [Atlas of ICL, arXiv:2505.15807](https://arxiv.org/abs/2505.15807)

KV-cache selection:

- [H2O, arXiv:2306.14048](https://arxiv.org/abs/2306.14048); [TOVA, arXiv:2401.06104](https://arxiv.org/abs/2401.06104); [SnapKV, arXiv:2404.14469](https://arxiv.org/abs/2404.14469); [Quest, arXiv:2406.10774](https://arxiv.org/abs/2406.10774); [PyramidKV, arXiv:2406.02069](https://arxiv.org/abs/2406.02069); [DuoAttention, arXiv:2410.10819](https://arxiv.org/abs/2410.10819); [RazorAttention, arXiv:2407.15891](https://arxiv.org/abs/2407.15891); [HeadKV, arXiv:2410.19258](https://arxiv.org/abs/2410.19258); [InfLLM, arXiv:2402.04617](https://arxiv.org/abs/2402.04617); [LongHeads, arXiv:2402.10685](https://arxiv.org/abs/2402.10685)

Separators, anchors, landmarks, span ends:

- [SepLLM, arXiv:2412.12094](https://arxiv.org/abs/2412.12094); [Label Words are Anchors, arXiv:2305.14160](https://arxiv.org/abs/2305.14160); [Gather-and-Aggregate, arXiv:2504.18574](https://arxiv.org/abs/2504.18574); [LLM-Microscope, arXiv:2502.15007](https://arxiv.org/abs/2502.15007); [Punctuation and Predicates, arXiv:2508.14067](https://arxiv.org/abs/2508.14067); [Extracting Paragraphs from LLM Token Activations, arXiv:2409.06328](https://arxiv.org/abs/2409.06328); [Token Erasure, arXiv:2406.20086](https://arxiv.org/abs/2406.20086)
- [Induction Heads, arXiv:2209.11895](https://arxiv.org/abs/2209.11895); [Dual-Route Model of Induction, arXiv:2504.03022](https://arxiv.org/abs/2504.03022); [Copy Suppression, arXiv:2310.04625](https://arxiv.org/abs/2310.04625); [The Token Before the Value Is the Key, arXiv:2609.15545](https://arxiv.org/abs/2609.15545)
- [Landmark Attention, arXiv:2305.16300](https://arxiv.org/abs/2305.16300); [Anchor-based LLMs, arXiv:2402.07616](https://arxiv.org/abs/2402.07616); [Vision Transformers Need Registers, arXiv:2309.16588](https://arxiv.org/abs/2309.16588); [StreamingLLM, arXiv:2309.17453](https://arxiv.org/abs/2309.17453)

Readouts and placement:

- [Kobayashi et al., arXiv:2004.10102](https://arxiv.org/abs/2004.10102); [Attention Flow, arXiv:2005.00928](https://arxiv.org/abs/2005.00928); [Value Aggregation / AlignedWVA, arXiv:2602.01572](https://arxiv.org/abs/2602.01572); [AttnLRP, arXiv:2402.05602](https://arxiv.org/abs/2402.05602); [Chefer et al., arXiv:2012.09838](https://arxiv.org/abs/2012.09838)
- [Lost in the Middle, arXiv:2307.03172](https://arxiv.org/abs/2307.03172); [Prompt Repetition, arXiv:2512.14982](https://arxiv.org/abs/2512.14982); [Echo Embeddings, arXiv:2402.15449](https://arxiv.org/abs/2402.15449); [Re2, arXiv:2309.06275](https://arxiv.org/abs/2309.06275); [PartRep, arXiv:2607.01792](https://arxiv.org/abs/2607.01792); [TimeStampEval, arXiv:2511.11594](https://arxiv.org/abs/2511.11594)

Probes and labelled routes:

- [Know but Don't Tell, arXiv:2406.14673](https://arxiv.org/abs/2406.14673); [BAP, arXiv:2502.13966](https://arxiv.org/abs/2502.13966); [LOVA, arXiv:2410.15288](https://arxiv.org/abs/2410.15288); [Binding IDs, arXiv:2310.17191](https://arxiv.org/abs/2310.17191); [Lookbacks, arXiv:2505.14685](https://arxiv.org/abs/2505.14685)
- [MCSB, arXiv:2210.12353](https://arxiv.org/abs/2210.12353); [PriDe, arXiv:2309.03882](https://arxiv.org/abs/2309.03882); [FIRST, arXiv:2406.15657](https://arxiv.org/abs/2406.15657)

Interference:

- [RULER, arXiv:2404.06654](https://arxiv.org/abs/2404.06654); [PI-LLM, arXiv:2506.08184](https://arxiv.org/abs/2506.08184); [The First Drop of Ink, arXiv:2605.10828](https://arxiv.org/abs/2605.10828); [NoLiMa, arXiv:2502.05167](https://arxiv.org/abs/2502.05167)

Logs:

- [Drain, ICWS 2017 (dblp)](https://dblp.org/rec/conf/icws/HeZZL17.html) and [logparser's Drain](https://logparser.readthedocs.io/en/latest/tools/Drain.html); [Tools and Benchmarks for Automated Log Parsing, arXiv:1811.03509](https://arxiv.org/abs/1811.03509); [Loghub, arXiv:2008.06448](https://arxiv.org/abs/2008.06448); [Loghub-2.0, arXiv:2308.10828](https://arxiv.org/abs/2308.10828); [LILAC, arXiv:2310.01796](https://arxiv.org/abs/2310.01796); [Log Parsing: How Far Can ChatGPT Go?, arXiv:2306.01590](https://arxiv.org/abs/2306.01590)
- [LogQA, arXiv:2303.11715](https://arxiv.org/abs/2303.11715); [LogEval, arXiv:2407.01896](https://arxiv.org/abs/2407.01896); [LogNLQ, arXiv:2607.03884](https://arxiv.org/abs/2607.03884); [LogRouter, arXiv:2605.18015](https://arxiv.org/abs/2605.18015); [LLM4Log, arXiv:2604.16359](https://arxiv.org/abs/2604.16359)
