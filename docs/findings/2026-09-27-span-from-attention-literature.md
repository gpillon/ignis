# A span read from one prefill's attention has no precedent, and the lexical misses have mechanisms to test

- Kind: research
- Status: current
- Observed: 2026-09-27
- Last verified: 2026-09-27
- Scope: `/v1/decide` / attention readout over a text state (spec 19 phase 0 step 0): spans, head combinations, values and gates, calibration, position bias, the lexical paradox
- Related: [GitHub #276](https://github.com/gpillon/ignis/issues/276); [spec 19](../specs/decide/19-a-span-read-from-attention.md); [`2026-09-26-locate-attention-no-go.md`](2026-09-26-locate-attention-no-go.md) (spec 18's result, which this reads against); [`2026-09-21-qwen-vl-grounding-primary-sources.md`](2026-09-21-qwen-vl-grounding-primary-sources.md) (the image-side literature pass)
- Superseded by: none

## Question

Spec 19 asks whether one prefill's attention can name the exact *span* of a
text state that answers an instruction, and why spec 18's reading failed the
lexical questions: one head (L39.h12) at the copy scaffold `{"quote":"`,
content-free baseline subtracted, found paraphrased targets 91% of the time
and missed 43% of the questions that repeat a rare word of their target —
far from it, often on an early line. Its phase 0 step 0 is this pass: what
the literature measured, on which models, with which query positions,
baselines and head aggregations, on eleven topics:

1. training-free answer-span extraction from attention;
2. value- and gate-aware attribution, and gated attention;
3. learned or selected head combinations (ICR, QRHead, AT2, ContextCite);
4. attention sinks and position bias in long context;
5. copy, induction and retrieval heads, and their offsets;
6. **the lexical paradox** (the priority): any report of a word shared by the
   question and the context *hurting* attention-based retrieval, and where
   the attention goes instead;
7. relevance profiles read as peaks: several answers, none;
8. span decoding (BERT's start and end);
9. contextual calibration and its transfer to a per-token log-ratio;
10. attention flow, and attention heads in hybrid (recurrent + attention)
    models;
11. TAG, cross-checked.

## Evidence

### How the sources were read

- Primary texts only: arXiv abstract, HTML and PDF, the ACL Anthology, the
  authors' code, and the served model's own code and config. No blog or
  third-party summary.
- **Checked against the raw text** (the HTML or PDF downloaded, converted to
  text, and the sentence found by search): ICR, IOI, NoLiMa, the
  extractive-QA circuit paper (AttnAttrib), Atlas §7 and App. D, Xu et al.'s
  Table 1, Gather-and-Aggregate §4-5, the Dual-Route model §3, Cheng et al.
  (hybrid induction), Afendulev et al. (hybrid recall), RoFormer §3.3, the
  hidden-sinks paper, and the arXiv metadata (title, authors) of every ID in
  spec 19's reference list.
- The rest was read through a fetcher that summarises. Quoted sentences were
  re-fetched until identical, and table rows were re-read one at a time.
  *(extraction only)* marks numbers read once through it and not confirmed by
  a second read. Numbers that a paper shows only in figures are not quoted.
- **Trap:** arXiv HTML prints inline math twice ("88 heads" is 8 heads,
  "top-1010" is top-10). **Trap:** a paper's v1 and v2 tables can differ
  (ICR's Table 1 does); the version is named where it matters.
- The downloads were kept outside the repository.

### 0. The served model's attention, from its own code

transformers 5.17.0, `models/qwen3_5/modeling_qwen3_5.py` (read locally in
`F:/ai/ngram-venv`), and `Y:/models/Qwen3.8-27B/config.json`:

- `:761-762`: `q_proj` emits `num_attention_heads * head_dim * 2`;
  `:787-790` splits it per head into the query and the **gate**. The gate is a
  256-vector per head, elementwise, computed from the query token's hidden
  state.
- `:792-793`: `q_norm` and `k_norm` (per-head RMSNorm) before RoPE. Values are
  not normalised (`:794`).
- `:698-700` with `partial_rotary_factor: 0.25`: RoPE rotates the first 64 of
  the 256 dimensions; the other 192 carry no position. `rope_theta: 10000000`.
- `:817-820`: `attn_output * torch.sigmoid(gate)`, then `o_proj`.
- 24 query heads, 4 KV heads (6 query heads per KV head), `head_dim` 256.

What follows by arithmetic: key `j`'s contribution to query head `h` at query
`i` is `α_ij · W_O^h (σ(g_i^h) ⊙ v_j)`. The gate reweights dimensions, so it
can change which key contributes most *within* a head, not only which heads
count. `‖v_j‖` belongs to the KV head and is shared by six query heads;
`‖W_O^h(σ(g) ⊙ v_j)‖` is per query head.

### 1. Spans read from attention (topics 1, 7, 8)

**Not found:** a training-free reading of an answer span — both edges — from
one prefill's attention in a decoder-only model, with a way to say "absent".
Every close work misses at least one of these:

| work | model | where attention is read | what it returns | training | numbers |
|---|---|---|---|---|---|
| Xu, Liang, Huang, Xiang, [2110.06393](https://arxiv.org/abs/2110.06393) | BART, T5 (encoder-decoder) | cross-attention of the last decoder layer at the **first and last generated** answer tokens, heads averaged: "a reasonable proxy for the probabilities of the start and end positions" | start and end | QA fine-tuned; the span read is zero-shot, or trained jointly | SQuAD 1.1 EM/F1, generate vs span from attention (Table 1): BART-base 79.96/88.14 vs 79.99/88.07; **BART-large 84.23/91.56 vs 62.94/73.82**; large, joint 84.17/91.69 vs 85.53/92.41 |
| Atlas, Kahardipraja, Achtibat, Wiegand, Samek, Lapuschkin, [2505.15807](https://arxiv.org/abs/2505.15807) §7, App. D | Llama-3.1-8B, Mistral-7B-v0.3, Gemma-2-9B | the **last prompt position** (before the first answer token), retrieval heads; each head's map weighted by a learned `w_h` times the head's logit-lens score for the predicted token (Eq. 11) | one token | linear probe on NQ-Swap | top-1 localisation 97 / 96 / 84% (Table 2); "a simple averaging of the attention maps … resulted in approximately 10% lower scores across all models" |
| AttnAttrib, Basu et al., [2502.08059](https://arxiv.org/abs/2502.08059) §3.3.3, §4 | Vicuna-7B, Llama-3-8B, Phi-3, Llama-3-70B | one low-entropy head of the context-faithfulness circuit, at **each generated** answer token | a fixed-length window around the maximum (`GetMaxSpan`), sentences ranked by their maximum | head found by path patching on a probe set | figures only |
| MultAttnAttrib, [2607.01420](https://arxiv.org/abs/2607.01420) | Qwen3-VL-30B-A3B | a prefill over the answer and question tokens (the answer must exist) | sliding windows; thresholds swept for F1; falls back to the argmax, never abstains | heads by causal mediation | *(extraction only)* |
| Attrieval [2503.09819](https://arxiv.org/abs/2503.09819); Ding et al. [2412.11404](https://arxiv.org/abs/2412.11404); AttnTrace [2508.03793](https://arxiv.org/abs/2508.03793) | Llama, Qwen2 7-8B | generated tokens | sentences, facts, passages | none | — |

AttnAttrib's heads "attend to the answer token in the context" (§3.3.3); its
evaluation gold is the sentence, not the span. Atlas is the only
prompt-position reading, and it returns the first answer token — an
initiator, with no terminator.

**Span decoding (trained).** BERT, [1810.04805](https://arxiv.org/abs/1810.04805)
§4.2: "The score of a candidate span from position i to position j is defined
as S·T_i + E·T_j, and the maximum scoring span where j ≥ i is used as a
prediction." §4.3, SQuAD 2.0: a null score `s_null = S·C + E·C` at `[CLS]`;
"We predict a non-null answer when ŝ_{i,j} > s_null + τ, where the threshold τ
is selected on the dev set to maximize F1." The paper states no maximum span
length. Wang & Jiang's boundary model (Match-LSTM,
[1608.07905](https://arxiv.org/abs/1608.07905)) capped spans at 15 tokens
and searched `p(a_s)·p(a_e)`: dev EM 61.1 → 63.0. Segal et al.
([1909.13375](https://arxiv.org/abs/1909.13375)) replace start and end with a
per-token tag for multi-span answers (+9.9 EM on DROP). SQuAD 2.0 (Rajpurkar,
Jia, Liang, ACL 2018, [1806.03822](https://arxiv.org/abs/1806.03822)):
unanswerable questions are "relevant to the paragraph" with a plausible
answer present, and abstaining scores 1 or 0.

**Metrics for spans and rationales.** ERASER (DeYoung et al.,
[1911.03429](https://arxiv.org/abs/1911.03429) §4.1): IOU F1 counts a
predicted span as a match "if it overlaps with any of the ground truth
rationales by more than some threshold (here, 0.5)"; soft scores are scored
by AUPRC, "constructed by sweeping a threshold over token scores".

**Peaks and abstention.** The only per-example peak rule found is MIRAGE's
(Qi et al., [2406.13663](https://arxiv.org/abs/2406.13663), gradient
saliency, not attention): keep scores above the example's mean plus one
standard deviation, or its top 3-20%. A calibrator for "is this answer right"
is Kamath, Jia, Liang ([2006.09462](https://arxiv.org/abs/2006.09462)): a
random forest over the span's probability, the top-5 softmax values and the
lengths lifts coverage at 80% accuracy from 48.2 to 56.1. Lookback Lens
(Chuang et al., [2407.07071](https://arxiv.org/abs/2407.07071)) detects
hallucinated *generated* spans with a logistic regression over every head's
context-vs-generated attention ratio: transfer AUROC 85.3 / 82.0 with all
heads, 71.2 / 79.2 with the top 10. **Not found:** a training-free
detector of unanswerable questions from question-to-context attention.

**The comparator's failure modes.** Xu et al.: 6.2% of FiD BART-large answers
on NQ and 11.2% on TriviaQA are "not spans from within the 100 context
passages". A 2026 clinical study
([2609.15964](https://arxiv.org/abs/2609.15964), *extraction only*) finds
exact verbatim quotes from 44.8% (claude-haiku-4.5) to 99.5% (claude-opus-5)
of the time.

### 2. Head combinations (topic 3)

| method | reading positions | heads | calibration | unit | models | numbers |
|---|---|---|---|---|---|---|
| ICR, Chen, Jiménez Gutiérrez, Su, [2410.02642](https://arxiv.org/abs/2410.02642), ICLR 2025 | **every query token**, averaged (Eq. 1) | **all layers and heads, summed** | `N/A` pass subtracted per token (Eq. 2), then tokens below mean − 2σ dropped | document (tokens summed) | Mistral-7B-Instruct-v0.2, Llama-3.1-8B-Instruct | BEIR micro nDCG@10 (v2 Table 1, Llama): ICR 60.4, RankGPT 52.0; DBPedia-Entity 35.3 vs 41.4; FEVER 84.5 vs 66.7 |
| QRHead, Zhang, Yin, Yen, Chen, Ye, [2506.09944](https://arxiv.org/abs/2506.09944), EMNLP 2025 | query tokens averaged | 16 heads (32 at 70B), ranked by attention mass on gold documents | `N/A` score subtracted at retrieval | document | Llama-3.2-3B, 3.1-8B, 3.1-70B, Qwen2.5-7B | BEIR avg (Llama-8B): QR 50.0, ICR 48.5; head choice (BEIR-shuffled): random 37.5, all 42.8, Wu et al.'s retrieval heads 43.4, QR 47.5; 16 → 256 heads on NQ: 58.6 → 56.6 |
| AT2, Cohen-Wang, Chuang, Madry, [2504.13752](https://arxiv.org/abs/2504.13752) | the response tokens, averaged | one learned coefficient per head, fitted to ablation effects (ContextCite's surrogate) | none | trained on single tokens, transfers to sentences | Llama-3.1-8B, Phi-3.5-mini, R1-Distill-Qwen-7B | beats averaged attention; bar charts only |
| ContextCite, Cohen-Wang, Shah, Georgiev, Madry, [2409.00729](https://arxiv.org/abs/2409.00729) | — (ablations) | — | — | sentence | Llama-3-8B, Phi-3-mini, … | averaged attention "approaches the performance of ContextCite with Llama-3-8B, it fares quite poorly with Phi-3-mini" |

ICR's own ablation (Table 4, Llama-3.1-8B, single-hop / multi-hop averages)
is the closest analogue to spec 18's reading position:

| ICR variant | single-hop | multi-hop |
|---|---|---|
| full (all query tokens, calibrated) | 57.1 | 64.3 |
| − calibration | 50.0 | 58.2 |
| − aggregation (**last query token only**, calibrated) | 46.3 | 47.2 |
| − both (last token, uncalibrated = attention sorting) | 40.5 | 45.7 |

Also verified by the sub-reading *(extraction only)*: CoRe heads
([2510.02219](https://arxiv.org/abs/2510.02219)) peak within the top 10
heads; on Qwen3-8B, a band of layers 18-21 beats all layers (.471 vs .447
nDCG@10, [2602.22591](https://arxiv.org/abs/2602.22591)); head utility is
"strongly dataset-dependent" ([2604.24608](https://arxiv.org/abs/2604.24608));
Lookback Lens keeps 100 heads of both signs and loses accuracy with only one
sign; AttnTrace averages only a text's top-K tokens, because attention
"spreads" across competing texts. No paper splits its questions into lexical
and paraphrase, and no paper reads a scaffold after the question.

### 3. Values, gates, and what sinks carry (topic 2)

- **Kobayashi, Kuribayashi, Yokoi, Inui, EMNLP 2020**
  ([2004.10102](https://arxiv.org/abs/2004.10102)). The measured quantity is
  `‖α f(x)‖` with `f(x) = (x W^V + b^V) W^O` — value *and* output projection,
  no residual, no layer norm. On BERT-base: "[CLS], [SEP], and punctuations
  — have remarkably large attention weights … our norm-based analysis
  demonstrated that the contributions of vectors corresponding to these tokens
  were generally small" (§4.2); Spearman ρ between α and `‖f(x)‖` is −0.69 for
  [SEP], −0.34 [CLS], −0.25 comma and period, −0.06 other tokens (Table 2);
  word-frequency rank against `‖f(x)‖`, ρ = 0.75: BERT reduces frequent words
  "by adjusting ‖f(x)‖ and not α" (§4.3). On NMT alignment the norm lowers the
  best layer's AER from 47.7 to 41.4 (AWO) and 29.8 to 25.0 (AWI) without
  moving the best layer (Table 3).
- **Kobayashi et al., EMNLP 2021** ([2109.07152](https://arxiv.org/abs/2109.07152))
  adds residual and layer norm: "the residual connections pass through much
  larger vectors than the vectors produced by the multi-head attention". Within
  one head at one query the residual adds only a self term, so it does not
  reorder keys.
- **ALTI** (Ferrando, Gállego, Costa-jussà, EMNLP 2022,
  [2203.04212](https://arxiv.org/abs/2203.04212)) and **ALTI-Logit** (ACL 2023,
  [2305.12535](https://arxiv.org/abs/2305.12535), decoder-only: GPT-2, OPT,
  BLOOM) project each head's transformed values onto the predicted token and
  chain layers; ALTI-Logit "aligns better at tasks where the tokens of the
  linguistic evidence are far from the prediction".
- **Value zeroing** (Mohebbi, Zuidema, Chrupała, Alishahi, EACL 2023,
  [2301.12971](https://arxiv.org/abs/2301.12971)), Spearman with blank-out,
  pre-trained / fine-tuned BERT-family (Table 3): attention −0.10 / −0.07,
  attention-norm 0.19 / 0.14, ALTI 0.17 / 0.19, value zeroing 0.26 / 0.31.
- **Gated attention** (Qiu et al., Qwen team,
  [2505.06708](https://arxiv.org/abs/2505.06708); NeurIPS 2025 per the
  authors' repository). The gate `Y′ = Y ⊙ σ(XW_θ)` after SDPA, from "the
  hidden states after pre-normalization", head-specific, elementwise — the
  shape of the served model's code (§0). Table 4: the first token's share of
  attention falls from 0.467 to 0.048, with a mean gate score of 0.116; Fig. 2:
  layer 21 from 83% to 4%. §4.2: the gate "may filter out irrelevant
  contextual information for the query"; App. A.2: "gating might serve a
  similar function as attention sink in filtering out irrelevant information";
  App. A.4: "different heads require different sparsity". The paper names no
  shut heads and does not say that a sharply attending head can be gated to
  nothing — that is arithmetic from the design.
- **Head collapse in a released gated model** (Fu et al.,
  [2602.01203](https://arxiv.org/abs/2602.01203), ICML 2026): in
  Qwen3-Next-80B-A3B "certain heads consistently exhibit high importance
  scores, while many others show very low activation" (importance = the mean
  gate). Figure only.
- **Sinks carry small values.** Gu et al. ([2410.10781](https://arxiv.org/abs/2410.10781)
  §3.1): "the ℓ2-norm of keys and values of the first token is significantly
  smaller than that of other tokens"; the sink "acts more like key biases …
  not contributing to the value computation". Guo et al.
  ([2410.13835](https://arxiv.org/abs/2410.13835)): "value-state drains" on
  Llama-3.1-8B. VATP ([2406.12335](https://arxiv.org/abs/2406.12335), EMNLP
  2024) already ranks KV entries by `S_k · ‖v_k‖_1` because sink tokens' ℓ1
  norms are near zero. Bondarenko et al. ([2306.12929](https://arxiv.org/abs/2306.12929)):
  heads that want to "not update" put their mass on tokens "that have a low
  information content".
- **Not found:** any published attention, sink or gate analysis of the
  Qwen3.5 / 3.6 / 3.8 dense 27B, or of any hybrid's attention heads by value
  or gate.

### 4. Sinks, position bias and length (topic 4)

- **Xiao et al.** ([2309.17453](https://arxiv.org/abs/2309.17453), ICLR 2024):
  beyond the bottom two layers "the model heavily attends to the initial token
  across all layers and heads"; the sink is positional (four `\n` tokens work
  almost as well: perplexity 5.60 against 5.40, Table 1).
- **Gu et al.** (Table 6): the normalisation, not the nonlinearity, makes the
  sink (softmax 18.18%, sigmoid without normalisation 0.44%).
- **Hidden sinks** (Yu, Wang, Fu, Shi, Shaikh, Lin,
  [2406.15765](https://arxiv.org/abs/2406.15765), ICML 2024): "attention sinks
  occur not only at the start of sequences but also within later tokens of the
  input", on "tokens of less semantic importance", "particularly during the
  intermediate layers". SepLLM ([2412.12094](https://arxiv.org/abs/2412.12094)):
  separator tokens "contribute disproportionately to attention scores".
- **Found in the Middle** (Hsieh et al., [2406.16008](https://arxiv.org/abs/2406.16008),
  Findings of ACL 2024; Vicuna-7B-16k, tulu-2-7b): attention at the last prompt
  position, averaged over a document's tokens, every layer and head, is
  U-shaped over positions, and "the U-shaped pattern persists even after
  randomly shuffling document order". Model: `Attn(x,k) = rel(x) + bias(k)`;
  calibration subtracts a dummy document's attention at the same position
  (Eq. 4), one extra pass per position. NQ recall@3 with the gold in the
  middle, 20 documents: 0.2052 raw, **0.6832 calibrated** (Table 3). The
  intervention is applied only to the last 16 of 32 layers.
- **Lost in the Middle** (Liu et al., [2307.03172](https://arxiv.org/abs/2307.03172),
  TACL): the U-shaped accuracy curve; the key-value task is "a minimal
  testbed for the basic ability to retrieve matching tokens"; §4.2:
  "decoder-only models cannot attend to query tokens when contextualizing
  documents or key-value pairs, since the query only appears at the end" —
  with the query also placed before the data, GPT-3.5-Turbo (16K) retrieves
  perfectly at 300 pairs against a worst case of 45.6% without. App. E: only
  the larger Llama-2 models show primacy; 7B is recency-only.
- **Recency** dominates average attention in long contexts for Llama-2-7B
  variants (attention sorting, Peysakhovich & Lerer,
  [2310.01427](https://arxiv.org/abs/2310.01427)); attention from the last
  question token over 50 key-value pairs is U-shaped in layers 15-20
  ([2406.02536](https://arxiv.org/abs/2406.02536)).
- **RoPE's decay is an upper bound.** RoFormer ([2104.09864](https://arxiv.org/abs/2104.09864)
  §3.4.3) shows a bound on the inner product that decays with relative
  distance, not a decay of every product. In this model it acts on 64 of 256
  dimensions at `rope_theta` 1e7 (§0).
- **Decoders over-attend to initial tokens** (Abnar & Zuidema, below): "there
  is more attention toward initial tokens … we should first normalize based on
  the receptive field of attention."
- **Length.** Softmax flattens as the key count grows (Scalable-Softmax,
  [2501.19399](https://arxiv.org/abs/2501.19399)). Found in the Middle's
  calibration cut the drop from 10 to 20 documents from 44% relative to about
  8%.

### 5. Content-free calibration and the lift (topic 9)

- **Calibrate Before Use** (Zhao, Wallace, Feng, Klein, Singh, ICML 2021,
  [2102.09690](https://arxiv.org/abs/2102.09690)): `W = diag(p̂_cf)^-1`, with
  `p̂_cf` averaged over "N/A", "[MASK]" and the empty string. The same paper:
  "An alternate solution is to set b to −p̂_cf and W to the identity.
  Empirically, this alternate solution yields higher accuracy for generation
  tasks (where the dimensionality of p̂ is large)."
- **Every attention-relevance method found subtracts**: ICR per token (Eq. 2)
  and then filters "tokens with abnormally negative calibrated scores"; QRHead
  per document; Found in the Middle per position. **None uses a log-ratio per
  key.** ICR's calibration removes "a strong position bias towards documents
  placed at the beginning and the end of the input" and a bias "towards
  titles, entities and punctuation" (§5.1); its overhead is about 30% because
  the documents' KV is shared.
- **The null is not always null.** Batch Calibration
  ([2309.17249](https://arxiv.org/abs/2309.17249)) finds "N/A" read as content
  by surface equivalence. Karypis et al. ([2609.17764](https://arxiv.org/abs/2609.17764),
  Sep 2026): when instructions stay in the null pass it becomes
  "relevance-aware rather than null", and standard calibration hurts — Gemma-3-4B
  on InstructIR 0.578 uncalibrated, 0.275 calibrated, 0.825 with the
  instruction kept out of the null (*extraction only* for the numbers).
- **Arithmetic under layout L1 (not a published result).** The state's keys
  are identical in the question pass and the `N/A` pass, so for one head

  `log α_q(x) − log α_NA(x) = s_q(x) − s_NA(x) − (log Z_q − log Z_NA)`,

  where `s` is the pre-softmax score and `Z` the full row's partition. Within
  one question the last term is the same for every key: **the lift's ranking
  is the difference of the raw scores spec 18's dumps already hold** (`q·k/16`
  over the span). Its height across questions needs `log Z`, i.e. the full
  row (phase 1). On the 192 unrotated dimensions the difference is
  `(q_q − q_NA) · k_x / 16`: a projection of the RMS-normalised key onto what
  the question changed in the query; the 64 rotary dimensions add a term,
  because the scaffold sits at different absolute positions in the two
  prompts. Computed from scores there is no near-zero-probability blow-up: a
  log-probability is a score minus `log Z`.

### 6. Copy, induction and retrieval heads, and offsets (topic 5)

- **Induction heads** (Olsson et al., [2209.11895](https://arxiv.org/abs/2209.11895),
  26 authors): "Prefix matching: The head attends back to previous tokens that
  were followed by the current and/or recent tokens. Copying: The head's output
  increases the logit corresponding to the attended-to token." The query sits
  on the second `[A]` and lands on `[B]`: **+1 token** after the match, through
  a previous-token head's output in the key (K-composition). A "fuzzy" version
  completes `[A*][B*]…[A] → [B]`.
- **The IOI circuit** (Wang, Variengien, Conmy, Shlegeris, Steinhardt,
  [2211.00593](https://arxiv.org/abs/2211.00593), GPT-2 small §3.3):
  duplicate-token heads at the repeated name S2 attend to its first occurrence
  S1 (offset 0) and "copy the position of this previous occurrence"; induction
  heads at S2 attend to S1+1; previous-token heads sit at S1+1.
- **Retrieval heads** (Wu, Wang, Xiao, Peng, Fu, [2404.15574](https://arxiv.org/abs/2404.15574)):
  measured **during decoding** of the needle — the head's most-attended input
  token "is a token within the needle and is the same token as the currently
  generated token"; threshold 0.1; "only about 3% to 6%" of heads; masking 50
  drops every model's needle score below 50. The score is exact token identity:
  lexical by construction. Models: Llama-2, Mistral, Mixtral, Yi, Qwen1.5.
- **A span-end head** (Feucht, Todd, Wallace, Bau, "The Dual-Route Model of
  Induction", [2504.03022](https://arxiv.org/abs/2504.03022), COLM 2025):
  "Concept induction heads learn to attend to the ends of multi-token words";
  "token copier heads attend to the next token … whereas concept copier heads
  attend to the end of the next word"; concept heads sit in "mid-early layers",
  token heads more often late; ablating token heads makes models "paraphrase
  where they would otherwise copy verbatim". Word level only.
- **Where a segment's summary lives** (Bick, Xing, Gu, Gather-and-Aggregate,
  [2504.18574](https://arxiv.org/abs/2504.18574) §4): a Gather head makes "the
  final token of each segment 'attend to' all prior tokens in the same
  segment"; "the newline token (\n) at the end of each answer choice acts as a
  summary token"; Aggregate heads read those summary tokens and were
  "previously termed 'Correct Letter' Heads (Lieberum et al., 2023)" — the
  Chinchilla-70B multiple-choice heads ([2307.09458](https://arxiv.org/abs/2307.09458)).
  Label words collect a demonstration's information in shallow layers and are
  read by the prediction in deep ones (Wang et al., EMNLP 2023,
  [2305.14160](https://arxiv.org/abs/2305.14160)).
- **Other offsets.** Copying transformers hash n-grams and output "the
  succeeding token" (Jelassi et al., [2402.01032](https://arxiv.org/abs/2402.01032)).
  Successor heads ([2312.09230](https://arxiv.org/abs/2312.09230)) increment
  in their output, not in where they attend.
- **Not found:** a head defined as marking where a copied span *begins*. At
  the first copy step the retrieval-head and induction definitions put the
  attention on the span's first token, so an initiator falls out of them.

### 7. The lexical paradox (topic 6)

**Where ICR's lexical bias is measured.** ICR reads from the query's own
tokens and sums every head: "we aggregate the attention weights over all query
tokens, rather than only considering the last token as in previous work"
(§2). Its lexical finding (App. C.1, "ICR still suffers from lexical bias"):
"most of the re-ranking score comes from a small number of tokens in a few
documents … this signal is mostly concentrated in phrases that are lexically
similar to the query." Its lexical *failures* are distractors: "many
distractor documents containing entities with high lexical overlap with the
query, and therefore getting higher scores from ICR" (DBPedia-Entity,
2WikiMultihopQA). ICR is strongest where evidence has "lower lexical overlap
with the query" (FEVER, SciFact, §5.2). ReAttn
([2602.19969](https://arxiv.org/abs/2602.19969)) restates the bias and
down-weights query tokens that recur across candidates (IDF-like).

**A repeated token is actively avoided (IOI).** The algorithm GPT-2 small
implements: "1. Identify all previous names in the sentence … 2. Remove all
names that are duplicated … 3. Output the remaining name." S-Inhibition heads
"are active at the END token, attend to the S2 token, and write in the query of
the Name Mover Heads, inhibiting their attention to S1 and S2 tokens"; they
carry a token signal that "causes Name Mover Heads to avoid occurrences of that
token" and a position signal "causing Name Mover Heads to avoid the S1 position
no matter the value of the token at this position" — "position signals have a
greater effect than token signals" (§3.2). Name Movers put 0.59 of their
attention on the non-repeated name.

**A token can be attended and pushed down (copy suppression).** McDougall,
Conmy, Rushing, McGrath, Nanda ([2310.04625](https://arxiv.org/abs/2310.04625)):
"If components in earlier layers predict a certain token, and this token
appears earlier in the context, the head suppresses it" — the head *attends
back* to the earlier instance and writes a negative logit. IOI's Negative Name
Mover heads and anti-induction heads are copy suppression; it explains 76.9%
of GPT-2 small head 10.7's effect.

**A shared word misleads when it is not the answer (NoLiMa).** Modarressi et
al. ([2502.05167](https://arxiv.org/abs/2502.05167), ICML 2025): a distractor
sentence containing the question's keyword, "entirely irrelevant to both the
needle and the question's intent", cuts GPT-4o's effective length to 1K; "when
literal matches serve as distractors, they severely impair accuracy" (§4.4.4).
Literal matches where they are part of the fact make the task easy.

**The state cannot see the question.** Lost in the Middle §4.2 (above), and
Afendulev et al. (§8) on the hybrid.

**Not found:** any report that a word shared by the question and the target
*lowers* attention to the target when attention is read from a position after
the question. Each mechanism above is from a different setting; what they
predict for ours is in the Finding.

### 8. Hybrid models and attention flow (topic 10)

- **Qwen3.5, the served family** (Afendulev, Dontsov, Tutubalina, Korznikov,
  [2609.04434](https://arxiv.org/abs/2609.04434), Sep 2026; Qwen3.5 4B / 9B /
  27B — the 27B a single-seed extension — and Falcon-H1): keeping only the KV
  cache or only the recurrent state of a prefilled context, "Exact retrieval
  survives only through attention (64–98% of full accuracy) and collapses to
  zero through recurrence"; "The recurrent state preserves the semantic field
  of seen words rather than their identities" — recurrent-only generation
  "accepts words that were never in the context but share meaning or parts with
  seen items". No per-head analysis.
- **Induction in hybrids** (Cheng et al., "The Token Before the Value Is the
  Key", [2609.15545](https://arxiv.org/abs/2609.15545), Sep 2026): in
  GDN–attention hybrids, "Carrying concentrates in efficient layers and
  Matching in global receivers. The measured local contribution concentrates
  on lag one: the token immediately before the historical value." On released
  Qwen3.5-4B (`[GDN^3, FullAttn]^8`, short convolution 4): "the strongest
  Carrying lies in a GDN layer immediately before the strongest global
  Matching"; copying "align[s] with the key/value groups that carry the
  selected source" (Qwen3.5 KV H3, query heads 12-15).
- **Gather-and-Aggregate**: "pretrained hybrid models, where SSMs are combined
  with a few attention layers, delegate the role of Aggregate Heads to
  attention"; in Zamba-2-7B, disabling 9 attention heads cuts MMLU "from 64.3%
  to 34.9%, while knowledge task accuracy remains stable at 70%" (§5.3).
- **Jamba** ([2403.19887](https://arxiv.org/abs/2403.19887)), *extraction
  only*: 12 induction-like heads found "in all three attention layers" of a
  1.3B 1:7 hybrid, anecdotal. RecurrentGemma and Jamba-Mini keep near-perfect
  retrieval with about 15% of heads ([2510.19861](https://arxiv.org/abs/2510.19861),
  *extraction only*).
- **Attention flow** (Abnar & Zuidema, ACL 2020, [2005.00928](https://arxiv.org/abs/2005.00928)):
  rollout multiplies per-layer maps with a 0.5 identity for the residual;
  Spearman with blank-out at layer 6 of a small encoder: raw 0.29, rollout 0.71,
  flow 0.70 (Table 1). **Implicit attention exists for Mamba, Mamba-2, Griffin
  and RWKV** ([2403.01590](https://arxiv.org/abs/2403.01590),
  [2405.16504](https://arxiv.org/abs/2405.16504)), **not for Gated DeltaNet**:
  no rollout through this model's 48 recurrent layers has a published form.

### 9. TAG (topic 11)

[2412.10840](https://arxiv.org/abs/2412.10840) is "Attention-driven GUI
Grounding: Leveraging Pretrained Multimodal Large Language Models without
Fine-Tuning" (Xu, Chen, Wang, Liu, AAAI 2025); TAG stands for "Tuning-free
Attention-driven Grounding". The neighbour finding's claims hold: attention is
read from *generated* description tokens (not one pass), pooled over all
layers and heads, the top K = 10 heads chosen per generated token by their
attention to MiniCPM's resampler queries, ScreenSpot average 54.8 on
MiniCPM-Llama3-V 2.5.

### 10. Spec 19's reference list, checked

Every arXiv ID resolves to the title and authors the spec implies. What needs
correcting or adding:

1. **§ Why a second study, point 4** says the literature predicts the opposite
   of the lexical failure. It does not, for this reading position: ICR's bias
   is measured from the query's own tokens with every head summed, and ICR's
   last-query-token reading — the class spec 18 read in — loses 10.8 and 17.1
   points (§2). What ICR reports failing is distractors that share the
   query's words.
2. **ICR's calibration** is a per-token *subtraction* followed by a mean − 2σ
   filter, not a ratio. No attention-relevance paper uses the log-ratio
   "lift"; Calibrate Before Use's default is the ratio, but its own text
   prefers subtraction for large output spaces (§5).
3. **Kobayashi 2020** (arXiv 2004.10102) measures `‖α f(x)‖` with
   `f(x) = (xW^V + b^V)W^O`: the output projection is included. The spec's
   `α_h(x)·‖v_h(x)‖` is a weaker, KV-head-level quantity.
4. **The gate is a 256-vector per head**, elementwise, not a scalar `g_h`
   (§0). The gate-weighted ordinate is `α_j·‖W_O^h(σ(g^h) ⊙ v_j)‖`; a scalar
   per head (the mean gate) only reorders heads.
5. **Olsson et al.'s +1 is one token.** A +1 *line* needs the match to sit at
   the target line's last token, where the literature puts a segment's summary
   (Gather-and-Aggregate's newline, SepLLM's separators) and where concept
   induction heads land (the end of a lexical unit).
6. **Retrieval heads** (2404.15574) are defined during decoding, by the
   generated token; they are not a prompt-position reading.
7. **AT2** (2504.13752): coefficients are fitted to ablation effects, the
   features are attention weights only, averaged over the response's tokens.
8. **TAG**'s title and acronym as in §9.
9. **Xiao et al.'s sinks** may not be large in this model: gated attention cut
   the first token's share from 46.7% to 4.8% in Qiu et al.'s models. Measure
   before attributing early-line misses to them.
10. **Abnar & Zuidema** cannot be applied through the 48 Gated DeltaNet
    layers as published.
11. **SQuAD 2.0** is Rajpurkar, Jia, Liang, ACL 2018 (arXiv 1806.03822). The
    CC BY-SA 4.0 licences of SQuAD 2.0 and HotpotQA were confirmed on
    `rajpurkar.github.io/SQuAD-explorer` and `hotpotqa.github.io`.
12. **To add:** Atlas (2505.15807) and Xu et al. (2110.06393) as the nearest
    span precedents; IOI (2211.00593), copy suppression (2310.04625) and
    NoLiMa (2502.05167) for Q4; Found in the Middle (2406.16008) for position
    calibration; Gated Attention (2505.06708) for the gate; Gather-and-Aggregate
    (2504.18574), Cheng et al. (2609.15545) and Afendulev et al. (2609.04434)
    for the hybrid; the Dual-Route model (2504.03022) for span ends; ERASER
    (1911.03429) for the metrics; Karypis et al. (2609.17764) for the null
    prompt's content.

## Finding

### Observed in the sources

1. **Nobody reads a span from one prefill.** The nearest works either read
   start and end at the first and last *generated* tokens (Xu et al., an
   encoder-decoder; BART-large drops from 84.2 to 62.9 EM unless trained
   jointly), or read one token at the last prompt position (Atlas, 84-97%
   top-1, with a learned head weighting). No training-free method abstains.
2. **Everywhere attention serves as relevance**, a few selected heads beat
   all heads (QRHead, CoRe, a layer band on Qwen3-8B), learned or signed
   combinations beat small subsets (AT2, Atlas, Lookback Lens), averaging
   over the question's tokens beats the last position (ICR: +10.8 / +17.1),
   and calibration subtracts a null pass.
3. **ICR's lexical bias is a property of reading from the question's tokens.**
   Its failures are other documents sharing the query's words.
4. **Circuits exist that act against a repeated token.** IOI's S-inhibition
   removes the duplicated name's *position* from the copying heads'
   attention; copy suppression attends to a token and writes it down.
5. **Copy-type heads land one token after their match**; concept heads land
   on the end of a lexical unit; a segment's summary sits on its last token or
   separator, and aggregate heads read it.
6. **In the served family, exact retrieval runs through attention only**, and
   the lag-one "carrying" that prepares an attention layer's keys is done by
   the Gated DeltaNet layer just before it.
7. **This model's gate is per head and per dimension**, from the query token;
   output gating removed the first-token sink in its authors' models, and
   Qwen3-Next shows many heads with a low mean gate.
8. **Sinks carry small values** in every dense model measured.

### Inferred (untested here or anywhere)

- **The paradox is two routes, not a contradiction.** With a rare word shared
  by the question and the target, the match can be made where the word
  repeats — at the question's own tokens, by duplicate-token or induction
  heads, which record the earlier occurrence's position there (IOI's position
  signal). A later position then has no reason to search the state. A
  paraphrase has no repeated token, so the search happens at the answer
  position and is visible in the state. That predicts attention from the
  instruction's rare word should find the target — ICR's lexical bias — while
  the scaffold's in-span profile on lexical questions carries little question
  signal.
- **Where the reading then lands.** If the question adds little to the
  scaffold's row over the span, the per-line difference of shares is
  dominated by what the `N/A` pass already does: position and separator
  preferences. That would put the winner where the null profile is largest —
  early lines, separators — consistent with "far away, often early".
- **Inhibition is the competing account.** An IOI-like mechanism would push
  the scaffold's attention *below* its null level on the target's shared word,
  and toward same-type tokens in other lines.
- **L39.h10's next-line landing** fits a head that reads line summaries: in a
  causal model a line's summary can only form at or after its last token, and
  lag-one carrying moves it one position on. That makes it a terminator
  candidate and gives no reason to call it an induction head.
- **Value and gate weighting are unlikely to help by themselves.** Sinks carry
  small values, which `‖W_O v‖` will discount. But the lexical misses are not
  on sinks. And a head whose attention is right but whose gate is shut is not
  wrong for localisation, only unused by the model. Gate-weighting is a head
  selector, not a sharpener. Only a signed quantity — the head's direct effect
  on the predicted token, as in Atlas — can tell a copier from a suppressor.

## Implications

### The lexical paradox: hypotheses and what each predicts

| | hypothesis | source | predicts, on lexical questions | kill condition | where it is measured |
|---|---|---|---|---|---|
| **H1** | the match is made at the instruction's copy of the word; the scaffold reads the result from there | IOI duplicate-token and induction heads; ICR's query-token lexical bias | at the scaffold, the target's lift (raw score minus `N/A`) is near zero in the heads that read paraphrases, and the in-span lift profile is flat; the scaffold row's mass on the instruction's shared-word tokens exceeds the paraphrase level; attention *from* those tokens lands on the target's copy (offset 0) or the token after it (+1) | target lift on lexical as high as on paraphrase, or no extra instruction mass | lift: phase 0 (span dumps); instruction mass and the instruction-position reading: phase 1 |
| **H2** | the repeated position is inhibited | IOI S-inhibition | the target's shared-word tokens have *negative* lift; the displaced mass goes to lines holding same-type tokens (other ids, codes) — the manifest's `distractors` | lift at the shared-word tokens ≥ 0 on the misses | phase 0 |
| **H3** | the target is attended but written down | copy suppression | some heads put high α on the target with a negative direct logit effect on the predicted token; an α-only head choice can keep them | no head with high target α and negative effect | phase 1 (vehicle: head output, `W_O`, unembedding) |
| **H4** | when the question adds little, the null prior wins | Found in the Middle; ICR §5.1; hidden sinks; separators | the miss lines rank high in the `N/A`-only profile, whatever the target's position; they carry separators or early positions | misses uncorrelated with the null profile | phase 0 |

H1 or H2 would explain why the target is lost; H4 would explain where the
reading goes. They can hold together.

### Phase 0 (on A+B's span-only dumps, no GPU)

- **Step 4 (Q4) becomes the four checks above.** It needs the shared rare
  word's token positions in the target and in the instruction, recoverable from
  the manifest's `instruction` and `targets` with the tokenizer's offsets.
  Report per split, and for several heads, not only L39.h12.
- **Compute the lift from the raw scores**, per token: `s_q − s_NA` ranks
  keys exactly within a question (§5). Report it beside spec 18's per-line
  share difference, and beside a subtractive lift on renormalised in-span
  probabilities (Calibrate Before Use's large-output advice, and ICR's form).
  Heights across questions wait for phase 1's full row.
- **Per-line aggregation is a choice**, not only a sum: mean (Found in the
  Middle), top-K tokens' mean (AttnTrace), the maximum (AttnAttrib), and the
  **line's last token or its separator** (Gather-and-Aggregate's summary
  token). The last is the label-free analogue of what the labelled route
  offers its aggregate heads.
- **Step 1 (Q5) at token level**, under both scaffolds: where each head's
  peak key sits relative to the target — its first token, the last token of
  its first word (concept heads), its last content token, the `\n` escape
  after it, the next line's first token. The summary-token account predicts
  L39.h10 under `{"line":` peaks at the escape or the next line's first token;
  the copy scaffold's retrieval-type heads should peak on the target's first
  token (an initiator). Check how the tokenizer splits the escape and the
  punctuation next to it.
- **Step 2 (Q2)**: add per-head top-1 split by lexical and paraphrase, to
  find heads that are good on the half L39.h12 loses (complementarity), and
  the combinations with precedent — a QRHead-style set scored by gold-line
  mass, a logistic regression over heads with signed weights (Lookback Lens,
  AT2) — beside the anchored set.
- **The `N/A` pass alone as a reading** (Karypis et al.): the null prompt
  keeps the kind text and the scaffold. If it already points at gold, the
  baseline subtracts signal; try a null that also blanks the kind text in
  phase 1.

### Phase 1 (instruments)

- **Full-row dumps** with region labels, and the instruction's shared-word
  tokens labelled as their own region: the single measurement that confirms
  or kills H1.
- **Query positions**: the instruction's shared rare-word tokens (duplicate-token
  and induction heads: expect the target's copy and the token after it); all
  question tokens averaged, per head (ICR's Eq. 1 without the sum over
  heads); the scaffold's last token; teacher-forced quote tokens (retrieval
  heads at each copied token; concept heads at word ends).
- **Values and gates**: capture `‖v_j‖` per KV head, and the gate *vector* per
  head at each query position; compute `α_j·‖W_O^h(σ(g) ⊙ v_j)‖` offline from
  the weights. For H3 and for head choice, the head's signed direct effect on
  the model's own top next token at the scaffold (Atlas's weighting) — the
  prefill already has that token, but the head output, `W_O` and the
  unembedding need the vehicle or a new tap.
- **Measure the sink before blaming it**: the first-token and separator shares
  per head in this gated model.
- **The instruction-before-state render** (Q4's diagnostic): Lost in the
  Middle's query-aware placement predicts the lexical gap closes there. That
  would also support H1, since L1 is exactly the layout in which the state
  cannot be read in the question's light.
- **Metrics and rules for spans and peaks**: ERASER's IOU F1 at 0.5 overlap
  and token AUPRC; an edge pair decoded as BERT's best `i ≤ j`, with a
  maximum length set from the gold spans in characters (the tokenizer splits
  every digit, so a token cap is not portable); a null threshold `τ` tuned on
  E1+E2 for the absent decision (BERT §4.3); mean + σ as the baseline peak rule
  (MIRAGE); peak heights calibrated on development sets (Kamath et al.'s shape).
- **The generation comparator** must count "not found in the state" as its
  own outcome, beside "ambiguous": Xu et al. saw 6-11% of generated answers
  that were not spans.

## Limits and unknowns

- **Nothing here was measured on this model.** The circuits (IOI, copy
  suppression) are GPT-2 small; copy suppression is weaker in models trained
  without dropout. The hybrid results are on Qwen3.5 (different weights, the
  27B a single seed) and on small trained-from-scratch models.
- **No source splits lexical from paraphrase questions at a fixed reading
  position.** H1-H4 are assembled from mechanisms found in other tasks.
- **Nearly every head-combination result is document-level reranking**, where
  the unit is a passage of hundreds of tokens; a span is a handful.
- **Figure-only numbers** (AT2, ContextCite, AttnAttrib, Fu et al.'s head
  collapse) were not quoted. Rows marked *(extraction only)* rest on the
  summarising fetcher.
- **Several sources are 2026 preprints** read within weeks of posting
  (2602-2609 IDs), none known to be peer reviewed.
- **The lift arithmetic** assumes the two passes read identical state keys:
  true under L1 with the same chunking. With the hq codec the decoded rows are
  shared between passes, so their error enters the difference only through
  `q_q − q_NA`. That is inferred, not measured.
- **Not found:** a published analysis of attention heads, sinks or gates in the
  Qwen3.5, 3.6 or 3.8 dense 27B; an implicit-attention form for Gated
  DeltaNet; a training-free span or abstention reading from one prefill.

## Follow-ups

- Spec 19's phase 0 step 4 and step 1 take the tests in § Implications;
  phase 1's harness change adds the instruction's shared-word region, the
  instruction-token query positions and the gate vector.
- Spec 19's § Why a second study (point 4), § Candidate ordinates (the value
  and gate ordinates) and § References take the corrections in §10.

## Sources

The served model:

- transformers 5.17.0, [`models/qwen3_5/modeling_qwen3_5.py`](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_5) (read locally), and the Qwen3.8-27B `config.json` (local snapshot)

Spans, peaks, abstention:

- [Xu et al., Attention-guided Generative Models for Extractive QA, arXiv:2110.06393](https://arxiv.org/abs/2110.06393)
- [Kahardipraja et al., The Atlas of In-Context Learning, arXiv:2505.15807](https://arxiv.org/abs/2505.15807)
- [Basu et al., On Mechanistic Circuits for Extractive Question-Answering, arXiv:2502.08059](https://arxiv.org/abs/2502.08059)
- [MultAttnAttrib, arXiv:2607.01420](https://arxiv.org/abs/2607.01420); [Attrieval, arXiv:2503.09819](https://arxiv.org/abs/2503.09819); [Ding et al., arXiv:2412.11404](https://arxiv.org/abs/2412.11404); [AttnTrace, arXiv:2508.03793](https://arxiv.org/abs/2508.03793)
- [BERT, arXiv:1810.04805](https://arxiv.org/abs/1810.04805); [Match-LSTM + Answer Pointer, arXiv:1608.07905](https://arxiv.org/abs/1608.07905); [Segal et al., multi-span, arXiv:1909.13375](https://arxiv.org/abs/1909.13375); [SQuAD 2.0, arXiv:1806.03822](https://arxiv.org/abs/1806.03822)
- [ERASER, arXiv:1911.03429](https://arxiv.org/abs/1911.03429); [MIRAGE, arXiv:2406.13663](https://arxiv.org/abs/2406.13663); [Kamath et al., Selective QA, arXiv:2006.09462](https://arxiv.org/abs/2006.09462); [Lookback Lens, arXiv:2407.07071](https://arxiv.org/abs/2407.07071); [Verifiable by Construction, arXiv:2609.15964](https://arxiv.org/abs/2609.15964)

Head combinations and calibration:

- [ICR, arXiv:2410.02642](https://arxiv.org/abs/2410.02642); [QRHead, arXiv:2506.09944](https://arxiv.org/abs/2506.09944); [AT2, arXiv:2504.13752](https://arxiv.org/abs/2504.13752); [ContextCite, arXiv:2409.00729](https://arxiv.org/abs/2409.00729)
- [CoRe heads, arXiv:2510.02219](https://arxiv.org/abs/2510.02219); [Where Relevance Emerges, arXiv:2602.22591](https://arxiv.org/abs/2602.22591); [RouteHead, arXiv:2604.24608](https://arxiv.org/abs/2604.24608); [ReAttn, arXiv:2602.19969](https://arxiv.org/abs/2602.19969)
- [Calibrate Before Use, arXiv:2102.09690](https://arxiv.org/abs/2102.09690); [Batch Calibration, arXiv:2309.17249](https://arxiv.org/abs/2309.17249); [How Calibration Content Shapes Attention-Based Reranking, arXiv:2609.17764](https://arxiv.org/abs/2609.17764)

Values, gates, sinks, position:

- [Kobayashi et al. 2020, arXiv:2004.10102](https://arxiv.org/abs/2004.10102); [Kobayashi et al. 2021, arXiv:2109.07152](https://arxiv.org/abs/2109.07152); [ALTI, arXiv:2203.04212](https://arxiv.org/abs/2203.04212); [ALTI-Logit, arXiv:2305.12535](https://arxiv.org/abs/2305.12535); [Value Zeroing, arXiv:2301.12971](https://arxiv.org/abs/2301.12971)
- [Gated Attention for LLMs, arXiv:2505.06708](https://arxiv.org/abs/2505.06708) and [qiuzh20/gated_attention](https://github.com/qiuzh20/gated_attention); [Attention Sink Forges Native MoE, arXiv:2602.01203](https://arxiv.org/abs/2602.01203)
- [StreamingLLM, arXiv:2309.17453](https://arxiv.org/abs/2309.17453); [When Attention Sink Emerges, arXiv:2410.10781](https://arxiv.org/abs/2410.10781); [Active-Dormant Attention Heads, arXiv:2410.13835](https://arxiv.org/abs/2410.13835); [VATP, arXiv:2406.12335](https://arxiv.org/abs/2406.12335); [Quantizable Transformers, arXiv:2306.12929](https://arxiv.org/abs/2306.12929); [Hidden Attention Sinks, arXiv:2406.15765](https://arxiv.org/abs/2406.15765); [SepLLM, arXiv:2412.12094](https://arxiv.org/abs/2412.12094)
- [Found in the Middle, arXiv:2406.16008](https://arxiv.org/abs/2406.16008); [Lost in the Middle, arXiv:2307.03172](https://arxiv.org/abs/2307.03172); [Attention Sorting, arXiv:2310.01427](https://arxiv.org/abs/2310.01427); [Scaling a Single Dimension, arXiv:2406.02536](https://arxiv.org/abs/2406.02536); [RoFormer, arXiv:2104.09864](https://arxiv.org/abs/2104.09864); [Scalable-Softmax, arXiv:2501.19399](https://arxiv.org/abs/2501.19399)

Heads, circuits, the lexical paradox:

- [In-context Learning and Induction Heads, arXiv:2209.11895](https://arxiv.org/abs/2209.11895); [IOI, arXiv:2211.00593](https://arxiv.org/abs/2211.00593); [Copy Suppression, arXiv:2310.04625](https://arxiv.org/abs/2310.04625); [Retrieval Heads, arXiv:2404.15574](https://arxiv.org/abs/2404.15574); [Dual-Route Model of Induction, arXiv:2504.03022](https://arxiv.org/abs/2504.03022); [Repeat After Me, arXiv:2402.01032](https://arxiv.org/abs/2402.01032); [Successor Heads, arXiv:2312.09230](https://arxiv.org/abs/2312.09230)
- [Label Words are Anchors, arXiv:2305.14160](https://arxiv.org/abs/2305.14160); [Chinchilla multiple choice circuits, arXiv:2307.09458](https://arxiv.org/abs/2307.09458); [NoLiMa, arXiv:2502.05167](https://arxiv.org/abs/2502.05167)

Hybrids and flow:

- [Gather-and-Aggregate, arXiv:2504.18574](https://arxiv.org/abs/2504.18574); [What Attention Recalls and Recurrence Controls, arXiv:2609.04434](https://arxiv.org/abs/2609.04434); [The Token Before the Value Is the Key, arXiv:2609.15545](https://arxiv.org/abs/2609.15545); [Jamba, arXiv:2403.19887](https://arxiv.org/abs/2403.19887); [Some Attention is All You Need for Retrieval, arXiv:2510.19861](https://arxiv.org/abs/2510.19861)
- [Attention Flow, arXiv:2005.00928](https://arxiv.org/abs/2005.00928); [Hidden Attention of Mamba, arXiv:2403.01590](https://arxiv.org/abs/2403.01590); [Unified Implicit Attention for Gated-Linear RNNs, arXiv:2405.16504](https://arxiv.org/abs/2405.16504)
- [TAG, arXiv:2412.10840](https://arxiv.org/abs/2412.10840)
