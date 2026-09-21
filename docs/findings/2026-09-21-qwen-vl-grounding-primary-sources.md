# Grounding attention sits where this model still has attention, and the tail is not what zoom fixes

- Kind: research
- Status: current
- Observed: 2026-09-21
- Last verified: 2026-09-21
- Scope: serving / decision readout primitives, one-pass pointing (spec 11 C5), the two-pass box, `/v1/decide` `point` and `box`
- Related: [`2026-09-21-what-the-literature-says-about-pointing.md`](2026-09-21-what-the-literature-says-about-pointing.md)
  (the shallow first pass this deepens; four corrections to it are listed below),
  [spec 11](../specs/decide/11-point-in-one-pass.md),
  [spec 12](../specs/decide/12-the-coordinate-in-the-latent.md),
  [`2026-09-21-vision-tower-cost-at-width.md`](2026-09-21-vision-tower-cost-at-width.md),
  [`2026-09-21-the-point-is-assembled-as-it-is-written.md`](2026-09-21-the-point-is-assembled-as-it-is-written.md),
  [`2026-09-21-a-declared-grid-is-read-to-one-part-in-ten.md`](2026-09-21-a-declared-grid-is-read-to-one-part-in-ten.md),
  [`2026-09-19-constrained-digit-readout-points.md`](2026-09-19-constrained-digit-readout-points.md)
- Superseded by: none

## Question

The first literature pass named the mechanism (attention from a text token to
the image tokens) and moved on. This one reads the primary sources behind it,
and it serves two product questions that pull in different directions:

- **COST.** `point` costs one prefill plus 10 decode rounds and `box` costs 25.
  The owner wants a point in **one** forward pass and a box in **two**. The
  candidate is spec 11's C5, attention as the answer. On this model, softmax
  attention exists in only 16 of the 64 layers (0-indexed 3, 7, ..., 63). The
  fused kernel never materializes it. Is C5 plausible there, and what would it
  buy?
- **TAIL.** The chain is exact or catastrophic. It lands inside the button on
  218 of 240 scenes, with a median error of 0.8 units of 999. The other 22 are
  gross failures, and the failures are small targets. The literature's remedy
  for small targets is test-time zoom: crop around a first estimate and ground
  again. That is also the natural pass 2 of the two-pass box. Does it fix our
  tail, and what does it cost on this engine?

Seven sub-questions. The tag after each says which product question it serves:

1. Where does grounding attention live, by layer and head, in Qwen-VL LLMs? (COST)
2. How does test-time zoom behave on small targets, and at what price? (TAIL)
3. How does the family train and emit grounding? (both)
4. How does Qwen3.8 differ from Qwen3-VL, especially on DeepStack? (COST)
5. Has anyone read attention tuning-free on a Qwen backbone, and what is
   GUI-Actor exactly? (COST)
6. What resolution does GUI grounding want? (TAIL)
7. Where do probes find spatial information in VLM hidden states? (context)

## Evidence

**Tags.** Every claim carries the model generation it is about:

- [Qwen2-VL], [Qwen2.5-VL], [Qwen3-VL];
- [Qwen3.5] and [Qwen3.6]: the same `qwen3_5` architecture and 27B layout as
  the served model, but different weights;
- [Qwen3.8]: the served artifact;
- [other: base]: a non-Qwen base model;
- [here]: this repository's own measurements.

**How the numbers were checked.**

- Tables were parsed from the arXiv HTML itself, not from a summary. A
  summarizer read the Dev column as Overall on two tables in this pass (the
  ZoomClick main table and ScreenSpot-Pro Table 4), so nothing below rests on a
  summary unless marked *(extraction only)*.
- The Qwen3-VL benchmark numbers are published only as images on the model
  card. They were read from the images.
- Raw downloads are in `.scratch/lit/`, which is untracked.

**Settled here, not re-derived** [here]:

- the model answers on a 0-999 scale and volunteers `bbox_2d`;
- the native `point_2d` format and ours agree on 240 of 240 scenes;
- the 22 failures are small targets;
- the best residual probe site is L47, and the probe lands inside the button on
  97-111 of 240;
- a declared grid is read to about 1/10 of the axis;
- set-of-mark hurts open-weight models;
- MRoPE is interleaved.

### 0. The served model, from its own files [Qwen3.8]

These are primary sources for the architecture. They replace the MindStudio
citation in the first pass.

**Model card** (`Qwen/Qwen3.8-27B`, `README.md`; the pinned local snapshot is
byte-identical to HF `raw/main` today):

- "Hidden Layout: 16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))"
- "Number of Attention Heads: 24 for Q and 4 for KV", "Head Dimension: 256",
  "Rotary Position Embedding Dimension: 64"

**`config.json`:**

- `layer_types` has `full_attention` exactly at 0-indexed 3, 7, ..., 63.
- `attn_output_gate: true`
- `rope_parameters.mrope_interleaved: true`, `mrope_section: [11, 11, 10]`
- `vision_config.deepstack_visual_indexes: []`, `depth: 27`, `patch_size: 16`,
  `spatial_merge_size: 2`, `out_hidden_size: 5120`
- `preprocessor_config.json` has `size.longest_edge: 16777216` (4096²) and
  `shortest_edge: 65536` (256²).

**What that means for C5.** A forward pass computes 16 × 24 = **384**
query-head attention maps from text to image, and the fused kernel keeps none
of them.

**transformers 5.17.0, `models/qwen3_5/modeling_qwen3_5.py`:**

- `:818`: `attn_output = attn_output * torch.sigmoid(gate)`. Every head's output
  is gated **after** softmax·V, so an attention weight is not the head's
  contribution.
- `:926-929`: `_can_record_outputs` includes `"attentions": Qwen3_5Attention`.
  The PyTorch vehicle can therefore return all 16 maps under eager attention.
- `:206-215`: `recomposition_frequencies` writes the H and W frequencies at
  stride 3 (`slice(offset, length, 3)`). That line is the interleaving itself.

**DeepStack is deleted, not just unconfigured:**

- `modular_qwen3_5.py:152`: `deepstack_visual_indexes = AttributeError()`
- `:440-441`: `del self.deepstack_visual_indexes` and
  `del self.deepstack_merger_list`
- The text model's forward (`:1222-1308`) has no injection.
- Qwen3-VL's does: `qwen3_vl/modeling_qwen3_vl.py:846-852` adds visual features
  to the hidden states of the first `len(deepstack_visual_embeds)` layers, with
  default indexes `(8, 16, 24)`.

**Benchmarks on the card.** It reports **no grounding benchmark**: no RefCOCO,
CountBench, ODinW or ScreenSpot row. The agentic rows are OSWorld-Verified 84.3
and AndroidWorld 81.9. Thinking is on by default.

**Tokenizer.** `<|object_ref_start|>`, `<|object_ref_end|>`, `<|box_start|>`,
`<|box_end|>`, `<|quad_start|>` and `<|quad_end|>` are special tokens 248047-248052.

### 1. Where grounding attention lives, by layer (COST)

| model | LLM layers | layers named | depth | chosen by | used for | source |
|---|---|---|---|---|---|---|
| [Qwen2.5-VL] 7B | 28 | {18, 22, 24} | 0.64-0.86 | top 10% of layers by AUROC of head-averaged attention against evidence boxes, 100 TextVQA items; AUROC 89.09 on these, 80.07 over all layers | VQA evidence | 2510.17771 Table 4 |
| [Qwen2.5-VL] 32B | **64** | **{49, 50, 51, 52, 53, 55, 56}** | 0.77-0.88 | same; 88.18 against 73.08 | same | 2510.17771 Table 4 |
| [Qwen2.5-VL] 7B | 28 | 14-21, 24, 25 | 0.5-0.9 | gradient attribution, top 10 | ranking candidate boxes | 2606.01612 Table 1 |
| [Qwen2.5-VL] 7B / 72B | 28 / 80 | 20 / 60 | ~0.71 / 0.75 | Recall@K of the needle tokens | video token selection | FlexSelect 2506.00993 |
| [Qwen2.5-VL] 7B | 28 | 15 (1-indexed) | 0.54 | fixed per model | crop seed | 2606.16158 appendix |
| [Qwen2.5-VL] 3B | 36 | 22 | 0.61 | `# currently select 22 but feel free to try other layers` | crop seed (code only, no paper numbers) | `mllms_know/qwen2_5_methods.py:9` |
| [Qwen3-VL] 8B / 32B | 36 / **64** | 24 / **48** | 0.67 / 0.75 | "According to FlexSelect" | crop-view seed | MVP 2512.08529 |
| [Qwen3-VL] 8B | 36 | 20-28 (100 of 1,152 heads) | 0.56-0.78 | "gaze score" | steering, not localization | 2606.14703 §5.1 |
| [Qwen3-VL] 8B | 36 | **all** | - | "DeepStack multi-layer visual injection distributes grounding cues across depths so a single layer is insufficient" | crop seed | 2606.16158 appendix |

2510.17771, 2606.01612 and 2606.14703 do not state whether their layer indices
are 0- or 1-based.

**Every Qwen measurement puts the localizing attention between about half and
nine-tenths of the depth, and the readouts that work use one layer or a
handful, never the last.** FlexSelect states the mechanism: "Very deep layers
also not highlight the needle frames as the model has already consolidated the
critical visual information into the final token for next token generation"
[Qwen2.5-VL and others]. 2510.17771's abstract: "shallow layers focus primarily
on text, while deeper layers sparsely but reliably attend to localized evidence
regions" [Qwen2.5-VL 7B/32B, LLaVA-1.5, Gemma-3, InternVL3.5].

**Mapped onto this model** (arithmetic, and an analogy between two models, not a
measurement). The GQA layers sit at depths (i+1)/64: 35 → 0.56, 39 → 0.63,
43 → 0.69, 47 → 0.75, 51 → 0.81, 55 → 0.88.

- Six of the sixteen attention layers fall in the band. That is 144 of the 384
  query heads. Layer 31, at exactly 0.50, sits on the band's edge.
- Qwen2.5-VL-32B has the same 64-layer depth. Its grounding set contains 51 and
  55 under either indexing convention.
- MVP's layer 48 on the 64-layer Qwen3-VL-32B lies between Qwen3.8's GQA layers
  47 and 51.

**Few heads carry it.**

- [Qwen3-VL] 100 of 1,152 heads (8.7%) are the "gaze heads". Steering them
  redirects a description 83.1% of the time, against 14.6% for non-gaze heads
  and 0.9% for all heads (2606.14703 Table 2).
- Their accuracy falls with object size: 80.3% on large COCO objects, 61.7% on
  small ones, 27-32 px (Table 1).
- [Qwen2.5-VL] Trifuse: "Aggregating attention from all heads leads to
  substantial performance degradation, indicating that many attention heads
  encode information that is irrelevant to spatial localization." It keeps the
  top 6 heads by spatial entropy.
- [other: LLaVA-1.5, DeepSeek-VL, InternVL] Kang et al. (2503.06287) fix k = 3
  "localization heads" for every model they test. **No Qwen model** is among
  them *(extraction only)*.

**The query position matters.**

- [Qwen3-VL] MVP reads from "the center comma token (',') from the predicted
  coordinate format (e.g. '(123,456)')" because it "demonstrates better region
  localization performance". That is a position after x has been written.
- [Qwen2.5-VL] The ViCrop code reads the **last prompt token**:
  `outputs['attentions'][ATT_LAYER][0, :, -1, pos:pos_end]`. It then divides
  by the same row from a second pass with a generic prompt, so it is two
  passes.
- A zero-round C5 can use only a prompt position.

**No published attention-map, head or layer analysis of grounding exists for
any hybrid linear-attention / softmax VLM, Qwen3.5, 3.6 and 3.8 included.** That
gap is the finding for question 1. Two adjacent results point in opposite
directions:

- [other: Pythia-VL and Mamba-VL, 1-2.8B] 2409.05395 Table 2, RefCOCO testA:
  82.43 against 76.60 at 1.4B, and 85.39 against 79.29 at 2.8B, while Mamba
  leads on TextVQA. "Transformers achieve greater performance in visual
  grounding and the performance gap widens with scale." A recurrent state costs
  grounding.
- [other: Zamba2-7B, text-only] 2504.18574: "Hybrid models delegate the role of
  Aggregate Heads to attention" (Zamba2 has a 6:1 ratio of Mamba2 to attention
  layers). If that holds for a VLM, the retrieval-shaped heads are concentrated
  in exactly the layers that can be read.

### 2. What the one-pass readouts actually are (COST)

**TAG** [other: MiniCPM-Llama3-V 2.5, 8.5B; arXiv 2412.10840]

- **Not one pass.** §3.3 builds "the query prompt to prompt the model to first
  explicitly generate a description of the content or elements relevant to the
  query. We then use the attention between these descriptive tokens … and
  visual tokens". The attention is read from **generated** tokens.
- It pools every layer: "N is the total number of multi-head self-attention
  (MHA) layers multiplied by the number of attention heads per MHA".
- It keeps the top K heads **per generated token**, ranked by attention mass on
  the visual tokens: "we use K=10 in all experiments".
- The image reaches the LLM through MiniCPM's resampler, so the relevance map is
  the LLM attention multiplied by the resampler's cross-attention.
- ScreenSpot (Table 2): TAG averages **54.8** against 36.0 for the same model
  generating coordinates, and 53.4 for SeeClick (SFT). The icon cells are 29.3,
  28.6 and 29.1. Resolution is not stated; the model supports up to 1344×1344.
- The only other backbone is Qwen-VL-Chat [Qwen-VL, first generation], on one
  OCG aspect ratio: 2.7% to 10.2%.
- **TAG has never been run on Qwen2-VL or later.**

**GUI-Actor** [Qwen2-VL, Qwen2.5-VL; arXiv 2506.03143, code `microsoft/GUI-Actor`]

- "The final-layer hidden state of <ACTOR>, i.e., h<ACTOR>, is a contextual
  anchor for computing action attention over the visual patch tokens."
- In the released `modeling_qwen25vl.py`:
  - the keys are `inputs_embeds[i][visual_indices]`, the vision tower's merged
    output as it enters the LLM, before any decoder layer;
  - the head is a **new** `VisionHead_MultiPatch`: one 8-head
    `nn.MultiheadAttention` over the patch features, plus two
    `Linear(d, d) → GELU → Linear(d, d)` MLPs.
- **It never reads the backbone's attention weights.**

LiteTrain (Table 5; Qwen2-VL frozen, only the head and special tokens trained):

| variant | trained params | ScreenSpot-Pro | ScreenSpot | ScreenSpot-v2 |
|---|---|---|---|---|
| 2B-LiteTrain | 19M | 25.4 | 71.4 | 73.9 |
| 2B-LiteTrain + Verifier | 19M | 34.0 | 79.2 | 82.3 |
| 7B-LiteTrain | 103M | 22.9 | 73.5 | 74.9 |
| 7B-LiteTrain + Verifier | 103M | 35.8 | 81.3 | 83.8 |

- **Full training** (Table 1) gets further: Qwen2-VL-7B base 40.7 on
  ScreenSpot-Pro (44.2 with the verifier), Qwen2.5-VL-3B 42.2 (45.9), and
  Qwen2.5-VL-7B 44.6 (47.7). The data is "~1M screenshots" (9.6M elements),
  1 epoch.
- **Same backbone and data, coordinates against the head** (Table 6,
  Qwen2-VL-7B): Aguvis with point supervision scores 15.6 on ScreenSpot-Pro and
  85.1 on ScreenSpot. GUI-Actor scores 40.7 and 88.3.
- **The verifier is a second model.** It is "finetuned from UI-TARS-2B-SFT" and
  sees a pool of K = 20 candidates. "Candidates are evaluated in descending
  order of attention weights, and we return the first one exceeding a
  confidence threshold". So the "+ Verifier" rows are not one pass.
- **Its own limitation:** "a fixed patch size of 28×28 pixels. This poses
  challenges when dealing with very small interface elements (e.g., icons
  smaller than 10×10 pixels)".
- **By target size** (V2P 2508.13634 Table 4, [Qwen2.5-VL] 7B, full training;
  bins by area relative to the patch area n):

| bin | ScreenSpot-Pro | ScreenSpot-v2 |
|---|---|---|
| small (n ≤ A < 4n) | 17.5 | 50.0 |
| medium | 43.1 | 71.4 |
| large | 60.3 | 93.2 |

**Trifuse** [Qwen2.5-VL, 3B; training-free; arXiv 2602.06351]

- Its native-attention-only row (Table 7) reads the top-1 instruction token
  over the top-6 heads by spatial entropy.
- It scores **58.5 on ScreenSpot** (text 66.5, icon 48.7), 54.7 on
  ScreenSpot-v2, **8.0 on ScreenSpot-Pro** and 28.0 on OSWorld-G.
- The same backbone emitting coordinates zero-shot, in the same paper
  (Tables 2-4), scores **46.9**, 55.0 and **16.1**.
- Fusing OCR and a captioner gives 81.1, 82.6 and 18.9. Its own zoom stage
  (Table 8) takes ScreenSpot-Pro from 9.2 to 18.9.
- Resolution is not stated.

**GUI-AIMA** [Qwen2.5-VL, 3B; all parameters fine-tuned; arXiv 2511.00810]

- It supervises the model's **own** multi-head attention from an appended
  `<ANCHOR>` token, summed over all layers and heads with computed weights
  (Eq. 2-3).
- At 45k training samples (Table 4), "Vanilla Attention Grounding" scores 37.51
  on ScreenSpot-Pro against 35.99 for GUI-Actor-3B on the same data. The best
  variant scores 43.39.
- Full training: **53.8**, and **61.5** with a second, zoomed pass.

**The only training-free native-attention numbers on a Qwen backbone beat
coordinate output on ScreenSpot and halve it on ScreenSpot-Pro** [Qwen2.5-VL
3B]. Every attention-style readout above is patch-granular, and each one
reports its weakest numbers on the smallest targets.

### 3. Test-time zoom for small targets (TAIL, and pass 2 of the box)

ScreenSpot-Pro (SS-Pro) text / icon / average, before and after, each within
its own paper. The same backbone's single-pass SS-Pro score differs between
papers (Qwen3-VL-8B: 49.9 in MAI-UI, 55.0 in MVP, 54.6 on the card), so the
comparison is only meaningful inside a paper.

| method | base | seed | passes | crop rule | before → after |
|---|---|---|---|---|---|
| ReGround (2504.07981) | [other: OS-Atlas-7B] | point | 2 | fixed square around the first prediction | 28.1/4.0/18.9 → 55.7/15.1/**40.2** |
| ScreenSeekeR (2504.07981) | [other: OS-Atlas-7B + GPT-4o planner] | planner boxes | recursive | search until the patch is ≤ 1280 px | → 64.1/22.4/**48.1** |
| RegionFocus (2505.00684) | [Qwen2.5-VL] 7B | point | 6 calls (1 + 4 + 1) | 4 boxes at ratios [0.5,0.5], [0.3,0.3], [0.4,0.8], [0.8,0.4] of the image, upsampled | 39.3/6.6/26.8 → 46.6/8.8/32.1 |
| RegionFocus | [Qwen2.5-VL] 72B | point | 6 calls | same | 64.9/20.2/47.8 → 78.6/34.1/**61.6** |
| InnerZoom's "+ Zoom-In" baseline (2606.30084) | [Qwen3-VL] 2B / 4B, off the shelf | point | 2 | not stated | 38.3 → 50.9 / 53.1 → **63.1** |
| MAI-UI (2512.22047) | [Qwen3-VL] fine-tunes 2B / 8B / 32B | point | 2 | "width and height equal to half of the original image dimensions", resized back to the original resolution | 57.4 → 62.8 / 65.8 → 70.9 / 67.9 → 73.5 |
| ZoomClick (2512.05941) | [Qwen3-VL] 32B, training-free | point, plus a 2×2 pre-zoom | depth 3 | ×0.5 per step, "minimum crop size of m=768 pixels" | 72.6/24.0/54.0 → 86.5/48.8/**72.1** |
| GUI-AIMA (2511.00810) | [Qwen2.5-VL] 3B, trained | point | 2 | "crop size to 616 pixels", 2× zoom | 62.1/40.2/53.8 → 70.0/47.9/61.5 |
| MVP (2512.08529) | [Qwen3-VL] 8B / 32B | attention map (comma token, layer 24/48) | 1 + 2-4 views | 1280×720 views, "resized to 2560 × 1440" | 55.0 → 65.3 / 55.3 → **74.0** |

**Where the gain lands**

- **For strong models the gain is concentrated on icons, the small targets:**
  - ZoomClick, Qwen3-VL-32B: icon +24.8, text +13.9.
  - RegionFocus, Qwen2.5-VL-72B: icon +13.9, text +13.7.
- At 7B it is not: RegionFocus on Qwen2.5-VL-7B gains icon +2.2 and text +7.3.
- MAI-UI-32B: icon accuracy rises in all six domains, while text falls in Dev
  (86.4 → 84.4) and Office (90.4 → 88.1).

**The crop needs a context floor**

- ReGround crop size (SS-Pro Table 5), OS-Atlas-7B: 25.1 / 34.2 / 40.2 / 40.1 at
  512 / 768 / 1024 / 1280 px.
- UGround-7B on the same sizes: 27.0 / 28.8 / 28.2 / 26.3.
- ZoomClick's floor (Table 8), Qwen3-VL-32B: 65.2 with no floor, 72.1 at 768,
  72.3 at 1024. UI-Venus-7B instead peaks at 384 (65.7).
- ScreenSpot-Pro's authors: "when images are too small, crucial context is
  lost".

**Two passes get most of it.** ZoomClick at depth 2 / 3 / 4, with pre-zoom, on
Qwen3-VL-32B: 71.3 / 72.1 / 71.8.

**A crop around a wrong seed misses the target.** [Qwen2.5-VL-7B / UI-TARS-1.5-7B
based] GUI-Cursor (2509.21552): "the target could be out of the focused area
when the initial prediction is far away from the target; we find that 10.3% of
examples in ScreenSpot-Pro have this issue". Its crop is 1920×1080.

**The price is a second pass over a new image.**

- [Qwen3-VL 2B/4B] InnerZoom: the two-pass zoom "relies on a conditional second
  forward pass, leading to substantially higher overhead with 1.56-1.94×
  latency and 1.57-1.75× TFLOPs".
- [other: LLaVA-1.5] ViCrop is the one design that keeps the first image: "we
  directly extend the original image tokens by concatenating the visually
  cropped image tokens".
- ViCrop also shows small targets are the ones a crop rescues. On TextVQA with
  a human crop, LLaVA-1.5-7B goes from 39.38 to 69.95 on small answers against
  50.65 to 56.96 on large ones (Table 1).

**Our 22 failures against these crop rules** [here]

Computed for this finding, CPU only, from the per-scene rows of the
native-format A/B run (`.scratch/latent-probe/results/format-ab.json`, `ours`
arm). Coordinates were rescaled from 0-999 to the 1024 px side, and each crop
was clamped to the image.

- **The failures are not near misses.** None of the 22 lands within 2% of the
  side of its button on both axes.
- **They are mostly the right column and the wrong row.** The centre offset has
  a median |dx| of 0.001 of the side and a median |dy| of 0.298 (p75: 0.595 and
  0.640).

A square crop of side s, centred on the chain's answer, contains the true
button's centre on:

| s (fraction of side) | px at 1024 | target centre inside | whole button inside |
|---|---|---|---|
| 0.25 | 256 | 1 / 22 | 1 / 22 |
| 0.375 | 384 | 2 / 22 | 1 / 22 |
| 0.5 | 512 | 3 / 22 | 3 / 22 |
| 0.75 | 768 | 5 / 22 | 5 / 22 |

Crops that do not depend on the wrong answer's y:

- a full-height column at the answer's x contains the whole button on 13/22 at
  a quarter of the width, and 15/22 at half the width;
- **the 2×2 quadrants contain it on 17/22.** The other five straddle a quadrant
  boundary.

The failing buttons measure a median 13.5% × 3.6% of the side. At 1024 px that
is **4.3 × 1.1 merged tokens**.

### 4. How the family trains and emits grounding (both)

**[Qwen2-VL]** §2.2.1 of arXiv 2409.12191:

- "bounding box coordinates are normalized within [0, 1000) and represented as
  '(X_top left, Y_top left),(X_bottom right, Y_bottom right)'. Tokens
  <|box_start|> and <|box_end|> are utilized to demarcate bounding box text. …
  we introduce tokens <|object_ref_start|> and <|object_ref_end|> to indicate
  the content that the bounding box references".
- `quad` appears nowhere in the paper. The tokens are in the vocabulary only.

**[Qwen2.5-VL]** arXiv 2502.13923 **switched to absolute pixels**:

- "Unlike traditional approaches that normalize coordinates, our model directly
  uses the actual dimensions of the input image to represent bounding boxes,
  points, and other spatial features".
- The paper never mentions the special tokens.
- **Points became a trained task of their own:** "we have constructed a
  comprehensive pointing dataset comprising both publicly available and
  synthetic data. Specifically, the data source includes public pointing and
  counting data from PixMo".
- Table 6, PointGrounding: 58.3 / 67.3 / 67.5 for 3B / 7B / 72B. RefCOCO val:
  89.1 / 90.0 / 92.7.
- GUI data: "A synthetic data engine is used to generate screenshot captions
  and UI element grounding annotations."
- The JSON `bbox_2d` / `point_2d` shape is shown in the official blog
  (qwenlm.github.io/blog/qwen2.5-vl), not in the paper.

**[Qwen3-VL]** §3.2.4 of arXiv 2511.21631 **switched back to a normalized
scale**:

- "we systematically enhance grounding proficiency and support two grounding
  modalities: bounding boxes and points".
- "Different from Qwen2.5-VL, we adopt a normalized coordinate system scaled to
  the range [0,1000] in this version."
- Counting has "direct counting, box-based counting, and point-based counting".
- GUI data (§3.2.9): "element description, dense captioning, and dense
  grounding".
- The paper gives no output syntax and no special tokens.

The official cookbooks (`QwenLM/Qwen3-VL`, `main`) show the output syntax and
the scale:

- `2d_grounding.ipynb` emits `{"bbox_2d": [...]}` and `{"point_2d": [x, y]}` and
  divides by 1000.
- `computer_use.ipynb` sets `cfg={"display_width_px": 1000, "display_height_px": 1000}`
  and `max_pixels=12845056`.
- **`mobile_agent.ipynb` tells the model "The screen's resolution is 999x999."
  and rescales with `point[0]/999*width`.** The 0-999 scale we measured is one
  of the model's own trained conventions.

Qwen3-VL on the official card images (Instruct / Thinking):

| size | ScreenSpot | ScreenSpot-Pro | CountBench | ODinW13 |
|---|---|---|---|---|
| 2B | 91.4 / 88.9 | **48.5 / 32.2** | 88.4 / 84.1 | 43.4 / 36.0 |
| 4B | 94.0 / 92.9 | **59.5 / 49.2** | 84.9 / 89.4 | 48.2 / 39.4 |
| 8B | 94.4 / 93.6 | **54.6 / 46.6** | 80.5 / 91.5 | 44.7 / 39.8 |
| 32B | 95.8 / 95.7 | **57.9 / 57.1** | 94.9 / 94.1 | 46.6 / 41.8 |
| 235B-A22B | 95.4 / not read | 62.0 / not read | 93.0 / - | 48.6 / - |

- 235B Instruct also reports RefCOCO(avg) 91.9. The cards note "The default
  evaluation is conducted through API calls"; resolution is not stated.
- **Thinking scores below Instruct on ScreenSpot-Pro at every size read.**

**[Qwen3.5]** The `Qwen/Qwen3.5-27B` card uses the same Hidden Layout line and
`deepstack_visual_indexes: []`, and it does report grounding:

- **ScreenSpot Pro 70.3**
- RefCOCO(avg) 90.9, CountBench 97.8, ODInW13 41.1
- OSWorld-Verified 56.2, AndroidWorld 64.2
- Evaluation settings for these rows are not stated. The card says the model
  "operate[s] in thinking mode by default".
- Its comparison column for Qwen3-VL-235B-A22B mixes variants. Its ODinW 43.2
  is the Thinking number; Instruct is 48.6.

**[Qwen3.6]** The `Qwen/Qwen3.6-27B` card reports RefCOCO avg 92.5 and
CountBench 97.8, with no ScreenSpot row.

**[Qwen3.8]** The card has no grounding row at all (§0).

### 5. Qwen3.8 against Qwen3-VL: DeepStack (COST)

**[Qwen3-VL]** §2.2: "we select features from three distinct levels of the
vision encoder. Subsequently, dedicated vision–language merger modules project
these multi-level features into visual tokens, which are then added directly to
the corresponding hidden states of the first three LLM layers." The configs put
those levels at ViT blocks `[8, 16, 24]` for 8B and 32B.

**What DeepStack is for, as stated:**

- Table 9 is an ablation on an internal 15B-A2B model, pretrained only.
  DeepStack lifts the average from 74.7 to 76.0, InfoVQA from 71.9 to 74.2 and
  DocVQA from 89.5 to 91.1.
- §5.12.2 attributes it to "fine-grained visual understanding, such as on the
  InfoVQA and DocVQA benchmarks".
- **The ablation has no grounding benchmark.**
- [other: LLaVA-1.5-7B] The DeepStack paper (2406.04334): "gains are
  particularly pronounced on high-resolution tasks, e.g., 4.2, 11.0, and 4.0
  improvements on TextVQA, DocVQA, and InfoVQA".

**[Qwen3.5 / 3.6 / 3.8] DeepStack is gone:**

- every 27B config has `deepstack_visual_indexes: []`, and transformers deletes
  the mergers (§0);
- the Qwen3.5 README describes the vision side as "Early fusion training on
  multimodal tokens";
- **no official Qwen statement says DeepStack was dropped or why, or says
  anything about grounding under the hybrid architecture.**

**Consequence (inference).** In this model the image enters the LLM only as the
input embeddings of its tokens, as in Qwen2.5-VL. 2606.16158's warning that
Qwen3-VL needs every layer because DeepStack spreads the cues over depth
therefore has no mechanism here. The single-layer Qwen2.5-VL readouts of §1 are
the closer analogue.

### 6. Resolution for GUI grounding (TAIL)

- **[other: benchmark]** ScreenSpot-Pro: "targets in ScreenSpot-Pro occupy
  0.07% of the screenshot area on average, a significant reduction compared to
  2.01% of ScreenSpot". Screens above 1080p were prioritized.
- **No published curve of accuracy against `max_pixels` exists for any Qwen
  backbone.** The crop-size sweeps of §3 are the only resolution-shaped
  evidence.
- **[Qwen3-VL] Official guidance is about budget, not grounding accuracy:**
  - The README: "since the compression ratio is 32 for Qwen3-VL, we can set the
    number of visual tokens of a single image to 256-1280", with
    `{"longest_edge": 1280*32*32, "shortest_edge": 256*32*32}`.
  - The computer-use cookbook uses `max_pixels=12845056`, i.e. 12,544 merged
    tokens.
  - The shipped default is 4096² for Qwen3-VL-8B and, identically, for Qwen3.8
    (§0).
- **[here] What we have measured:**
  - `box` is most accurate at 2048 px on the three fixture scenes, with a y
    error of 6-10 px against 28-67 px at 4096.
  - `point` lands inside at every width from 768 px.
  - Encoding costs 0.040 s at 1024 px, 0.308 s at 2048 and 3.665 s at 4096
    (`2026-09-21-vision-tower-cost-at-width.md` §3, §5).
- **[here] Our "small" is not the literature's small.** The median failing
  button covers about 0.49% of the image area. That sits between ScreenSpot's
  2.01% and ScreenSpot-Pro's 0.07%.

### 7. Probing hidden states for position (context)

- [Qwen2.5-VL] 7B, 2606.31257: 5-fold logistic probes on the decision token's
  residual, for binary within-axis relations.
  - "Decoding rises sharply at L16–24 for every axis. Vertical peaks highest
    (94 at L23), above horizontal (85)".
  - "We index hidden_states[i] as the output of decoder layer i−1."
  - The paper's point is that decodable is not the same as used.
- [Qwen2.5-VL] 7B, 2607.06445: "representations in the final layer are notably
  poor at conveying spatial details". Middle layers L = {15, …, 24} converge
  fastest. The probe is a trained proxy, not a linear one.
- **No linear probe regressing absolute coordinates per layer on a Qwen model
  was found.**
- [here] L47 was the best of {19, 32, 47, 64} and the final layer the worst.
  That is the same mid-to-late band.

### What this corrects in the first pass

The first pass is kept as written. Four things in it do not survive the
primary sources:

1. **TAG is not a one-pass method.** It reads attention from tokens it first
   *generates* (§2). The one-pass precedents are GUI-Actor, GUI-AIMA, and the
   direct stage of Trifuse.
2. **GUI-Actor does not "train a head on exactly that signal".** It trains a
   new attention module over the vision embeddings, queried by the last layer's
   hidden state (§2).
3. **The numbers were conflated.** The "lightweight variant, 19-103M parameters"
   scores 25.4 and 22.9 on ScreenSpot-Pro. The 40.7 and 44.6 quoted beside it
   are full-training numbers.
4. **The architecture citation was a blog.** It is replaced by the card and the
   config (§0).

## Finding

### COST: C5, and what else a one-pass point could be

**Where the signal lives is not what blocks C5 on this model** (inference from
dense models). Every Qwen measurement puts the localizing attention at about 0.5
to 0.9 of the depth, and the readouts that work use one layer or a handful.
This model has six attention layers in that band: 35, 39, 43, 47, 51 and 55. A
64-layer Qwen2.5-VL grounds with a set that contains two of them. The first
pass framed "only 16 of 64 layers" as the architectural blocker. As a question
of *where*, it is not.

**Whether a hybrid's attention layers carry the map is unmeasured by anyone.**
That is the plain answer to question 1. The two nearest results pull in
opposite directions:

- a pure recurrent state loses about 6 RefCOCO points to a transformer
  [other];
- hybrids push their retrieval heads into the attention layers [other,
  text-only].

**The hybrid itself grounds well when it emits coordinates** [Qwen3.5]. The same
27B layout without DeepStack scores 70.3 on ScreenSpot-Pro, above Qwen3-VL-32B
Instruct (57.9) and 235B Instruct (62.0). That settles that the architecture
can ground. It says nothing about whether its attention maps can be read.

**The accuracy side is the real risk, and it lands on the tail.** The one
training-free native-attention number on a Qwen backbone beats coordinates on
ScreenSpot (58.5 against 46.9) and halves them on ScreenSpot-Pro (8.0 against
16.1). Every attention readout is patch-granular and reports its worst numbers
on the smallest bin. Our failing buttons are 1.1 merged tokens tall at 1024 px.

- Observed: that is the published pattern.
- Inferred: a C5 that matches the chain on the 218 would still lose ground on
  the 22.
- So C5 is a COST candidate that has to be scored separately on the
  small-target subset. It is not a TAIL fix.

**Two transfer caveats specific to this model** (inferred; untested anywhere):

- **The output gate.** Each head's output is multiplied by a sigmoid gate after
  softmax·V. Selecting heads by attention mass (TAG, Trifuse) can pick a head
  the gate then silences.
- **The query position.** MVP's best query sits *inside* the answer, after x.
  A zero-round readout has only prompt positions.

**The first pass mis-described the mechanism with no 16/64 problem at all.**
GUI-Actor's head needs two inputs:

- the final hidden state at one anchor position, which ignis already computes
  (`final_residual`);
- the vision tower's merged embeddings, which ignis already keeps (#243's
  embedding cache).

It needs no attention matrix and no GDN special case. Measured on Qwen2-VL, the
frozen-backbone version is modest: 22.9-25.4 on ScreenSpot-Pro, reaching 34-36
only with a second verifier model. What it would take here:

- Inferred: at d = 5120 the head is about 8d² ≈ 210M parameters (one d-wide MHA
  plus two d→d→d MLPs, the released structure).
- Inferred: spec 12's E-P1 bound does **not** cover it. E-P1 bounds a
  *regression* of (x, y) from the residual. This head is a *bilinear match*
  between the anchor state and each patch. It needs the anchor to encode what
  the target looks like, not where it is.

### TAIL: what zoom fixes, and what it does not

**Test-time zoom is the literature's answer to small targets, and it works.**
Two passes get most of the gain, which for 32B-and-up models lands mostly on
icons: ZoomClick +24.8 icon on Qwen3-VL-32B, and +10.0 average on an
off-the-shelf Qwen3-VL-4B. The crop needs a context floor of roughly 768-1024 px.

**But our tail is not the failure zoom repairs** (computed here). Zoom repairs
near misses and unreadably small targets inside a crop that contains them. Our
22 failures are wrong-element errors:

- none lands within 2% of the side of its button;
- the median failure is in the right column and 0.30 of the side off in y;
- a crop centred on the chain's answer contains the target on 3 of 22 at half
  the side and 5 of 22 at three quarters.

**So crop-and-reground seeded by the chain's own point can rescue at most about
five of the 22.** The literature knows this failure mode (10.3% of
ScreenSpot-Pro is out of crop). Here it is the majority case, because our tail
is the model picking a different button, not missing the right one by a few
pixels.

**What could reach the tail is a seed that does not trust the wrong answer**
(inferred; unmeasured here):

- **Tiles.** The 2×2 quadrants contain the whole button on 17 of 22. That is
  ZoomClick's pre-zoom, RankGround's tiles and UI-AGILE's sub-images, at the
  cost of several passes.
- **A column at the answer's x.** 15 of 22 at half width, because x is usually
  right.
- **The attention map.** Two papers report that the attention lands on the
  right region even when the answer is wrong: 2502.17422 on LLaVA-1.5 and
  InstructBLIP, and 2510.17771 on families including Qwen2.5-VL 7B/32B.
- **The whole image upsampled.** A zoom that cannot miss, and it doubles the
  tokens per button. At 2048 px the encode costs 0.308 s.

**The two-pass box's certificate cannot see this tail** (inference from
measured data). The scaffold pass proves chain-equivalence, not correctness. The
format A/B showed the model reproduces its own wrong answers across two output
formats on 240 of 240 scenes, so the certificate would certify all 22 failures.

**A crop pass costs a fresh vision encode plus a fresh prefill of the crop's
tokens, unless the crop is appended rather than substituted.**

- Measured here: a 1024 px crop encodes in 0.040 s; the prefill scales with its
  about 1k tokens.
- Inferred: ViCrop's design, the original image followed by the crop, leaves
  the first image's prefix intact. Whether the retained prefix claims across an
  appended image is for #193's media rule to decide, and it is unverified here.

## Implications

**For spec 11's C5:**

- Score per layer on the six band layers (35-55) before the other ten.
- Score per head, since it is 384 maps, not an average.
- Score on the small-target subset, not only on the 240.
- Record the gate value at the query position next to each map.
- A head set chosen offline has to be keyed to the artifact, the way the answer
  alphabet and spec 12's probe are.

**A candidate to add beside C5: GUI-Actor's head.** It sidesteps the fused
kernel and the GDN layers by construction, and costs a trained ~210M-parameter
module keyed to the artifact. Frozen-backbone precedent: modest, on Qwen2-VL.

**For the two-pass box:**

- A crop centred on pass 1's point is not the tail fix. Measure tiles, a column
  crop, or the attention map as the pass-2 seed.
- The crop side should stay at or above ~768 px of context.
- The scaffold certificate should be described as chain-equivalence, which is
  what it proves.

**What goes to the back of the queue:**

- `<|box_start|>` and the other special tokens stay last. No primary source
  after Qwen2-VL documents their use, and the output format was already shown
  not to be load-bearing here.
- Thinking stays off for decisions. It costs ScreenSpot-Pro points at every
  Qwen3-VL size read.

## Transfer table

| claim | generation it is about | transfers to Qwen3.8-hybrid? | what it changes in spec 11 | cheapest experiment to check it |
|---|---|---|---|---|
| Q1. Localizing text→image attention peaks at ~0.5-0.9 depth, and one mid-late layer is enough for a readout | Qwen2.5-VL, Qwen3-VL | **unknown**: six GQA layers (35-55) sit in the band, but no hybrid has been measured | C5 reads layers 35-55 first; "16 of 64" stops being the stated blocker | PyTorch vehicle at 1024 px, `attn_implementation="eager"`, `output_attentions` on the 16 GQA layers, 240 scenes, inside-rate per layer. **GPU** (vehicle), no engine change |
| Q1. A few heads carry it; averaging all heads degrades (top-6, 8.7%, k = 3) | Qwen2.5-VL, Qwen3-VL, other (LLaVA) | unknown | C5 needs a head set fixed offline and keyed to the artifact, not a per-request top-K over 384 maps | the same run: rank heads on half the scenes, score on the other half. **GPU** |
| Q1. Grounding cues spread over depth, so a single layer is insufficient | Qwen3-VL (because of DeepStack) | **no**: this model has no DeepStack (config `[]`, transformers deletes the mergers) | removes the argument for aggregating all 16 layers | none needed; primary source read. **No GPU** |
| Q1. Hybrids delegate retrieval heads to their attention layers; a pure recurrent state loses ~6 RefCOCO points | other (Zamba2 text-only; Pythia vs Mamba) | unknown; weak prior for C5 and against a pure recurrent state | recorded as the only hybrid evidence; neither predicts C5's accuracy | covered by the per-layer run above. **GPU** |
| Q1/Q5. Attention weights select heads, but this model gates each head's output with a sigmoid after softmax·V | Qwen3.8 (its own transformers source) | **yes**: it is this model | head selection must weigh the gate, or it can pick silenced heads | record `gate` at the query position alongside the maps in the same run. **GPU** |
| Q1. The best query token is inside the answer (MVP's comma), not a prompt token | Qwen3-VL | unknown | a zero-round C5 must pick among prompt positions (last token, or a forced anchor) and say which | the same run: last prompt token against a forced literal. **GPU** |
| Q5. TAG is not one pass; it reads attention from tokens it first generates | other (MiniCPM-Llama3-V 2.5) | n/a: corrects our first pass | C5's tuning-free precedent is Trifuse/ViCrop-style (prompt token, fixed layer), not TAG | none. **No GPU** |
| Q5. Training-free native attention alone beats coordinates on ScreenSpot (58.5 vs 46.9) and halves them on ScreenSpot-Pro (8.0 vs 16.1) | Qwen2.5-VL 3B | unknown; the direction predicts C5 is weakest on small targets | C5's acceptance must include the 22-scene small-target subset; C5 is not a tail fix | score the vehicle run split by button size. **GPU** |
| Q5. GUI-Actor's head uses the final hidden state and the vision embeddings, not backbone attention; LiteTrain 19M/103M on a frozen Qwen2-VL scores 25.4/22.9 on ScreenSpot-Pro (34.0/35.8 with a verifier) | Qwen2-VL (full: Qwen2.5-VL) | **yes** for the architecture (both inputs exist in ignis); **unknown** for accuracy | new candidate beside C5: a trained pointer head (~210M params at d = 5120, inferred) that has no 16/64 or fused-kernel problem; E-P1's bound does not cover it | vehicle: cache the anchor's final state and the merged vision embeddings for the 240 scenes plus more generated ones, and train the head offline. **GPU** for extraction only |
| Q2. A second pass on a crop lifts small targets most for ≥32B models (ZoomClick icon +24.8; Qwen3-VL-4B +10.0 average) | Qwen3-VL, Qwen2.5-VL | likely for near misses; the same family, and the crop gives more tokens per target | pass 2 of the box can buy resolution the scaffold pass cannot | crop pass over the 240 at 2× in the vehicle. **GPU** |
| Q2. A crop around a wrong seed misses the target (10.3% of ScreenSpot-Pro) | Qwen2.5-VL-based (GUI-Cursor) | **yes, and worse here**: computed 3/22 at 512 px, 5/22 at 768 px, 0/22 near misses | a crop centred on the chain's point is not the tail fix; seed pass 2 with tiles, a column at x, or the attention map | done here for containment (CPU). Accuracy of each seed: **GPU** |
| Q2. The crop needs a context floor (512 → 1024 px: 25.1 → 40.2; no floor → 768: 65.2 → 72.1) | other (OS-Atlas), Qwen3-VL-32B | likely | the pass-2 crop side stays ≥ 768 px of source context, not a fixed fraction | a crop-side sweep in the same crop run. **GPU** |
| Q2. A crop is a second encode plus a fresh prefill (1.56-1.94× latency); ViCrop appends the crop after the original image | Qwen3-VL (InnerZoom), other (ViCrop) | **yes** for the cost (1024 px crop: 0.040 s encode, measured here); the append variant is unverified against #193 | pass 2 prices at crop encode + crop prefill; an appended crop may keep pass 1's image prefix | engine read of #193's media rule; one timing run. **GPU** for the timing |
| Q3. Qwen3-VL trains points and boxes on 0-1000; Qwen2.5-VL used absolute pixels; the special tokens are documented only for Qwen2-VL (quad never) | Qwen2-VL, Qwen2.5-VL, Qwen3-VL | yes for the scale (0-999 measured here; the mobile cookbook uses 999); the tokens exist in this tokenizer with no documented use | the `<|box_start|>` route stays last; not worth a run before C5 or the crop | one vehicle run on the 240 with a `<|box_start|>` prompt. **GPU**; recommended skip |
| Q3. Thinking scores below Instruct on ScreenSpot-Pro at every Qwen3-VL size read (8B: 46.6 vs 54.6) | Qwen3-VL | unknown; this model thinks by default | keep decisions non-thinking; do not try reasoning to fix the tail | none planned |
| Q4. The same hybrid layout, without DeepStack, scores 70.3 on ScreenSpot-Pro, above Qwen3-VL-32B Instruct's 57.9 | Qwen3.5 | likely (same architecture, different weights; Qwen3.8's card has no grounding row) | the architecture is not a grounding handicap for coordinate output; says nothing about attention readout | none. **No GPU** |
| Q4. DeepStack injects three ViT levels into LLM layers 0-2 "for fine-grained visual understanding"; its ablation has no grounding benchmark | Qwen3-VL | **no** (absent here); no official statement why | image features reach the LLM only at the input embeddings, which is where GUI-Actor's keys are taken | none. **No GPU** |
| Q6. No accuracy-vs-`max_pixels` curve exists for Qwen; ScreenSpot-Pro targets average 0.07% of the area; crop sweeps favour 768-1280 px windows | other, Qwen3-VL | unknown; our failing targets are ~0.49% of the area, between the benchmarks | keep 2048 px (our `box` optimum) as the only local evidence; whole-image 2× upsample becomes a pass-2 seed candidate | the tail subset at 1024 and 2048 px in the vehicle. **GPU** |
| Q7. Spatial information peaks mid-to-late and the final layer is poor | Qwen2.5-VL | consistent with our L47 | nothing new; confirms spec 12's tap-not-final conclusion | none |

## Limits and unknowns

**The core transfer is not measured.** No source measures attention-based
grounding in a hybrid model. Every layer claim in §1 is about a dense model, and
the mapping onto layers 35-55 is an analogy.

**Some sources are soft:**

- Several of them are 2026 preprints read within weeks of posting
  (2606.xxxxx, 2607.xxxxx). Their tables were parsed, but none is known to be
  peer-reviewed.
- Layer-index conventions are unstated in 2510.17771, 2606.01612 and
  2606.14703, so every layer number there is ±1.

**Benchmark numbers do not compare cleanly across papers.** The same backbone's
ScreenSpot-Pro score varies 49.9-55.0 for Qwen3-VL-8B between papers, so only
within-paper deltas are quoted as gains.

**The Qwen3.5 comparison is looser than it looks.** Its 70.3 is a different
checkpoint from the served one, evaluated at a resolution and mode the card
does not state. Qwen3.6 and Qwen3.8 dropped the row, so there is no grounding
number for the served generation at all.

**The crop containment counts** in §3 are about 22 failures on synthetic
1024 px scenes, from one run. They are geometry, not accuracy: whether the model
then finds the button inside a crop that contains it is exactly what is
unmeasured.

**Some sources were not read, or were read only in part:**

- the Qwen3-VL report's grounding tables, which were truncated in the HTML (the
  card images were used instead);
- the 235B Thinking card image;
- the actual prompt strings in `2d_grounding.ipynb`, cut off by embedded
  images;
- Kang et al.'s tables, marked *(extraction only)*.

**Not found:**

- any official Qwen statement that DeepStack was dropped, or on grounding under
  the hybrid;
- a Qwen3.5 technical report on arXiv (only Qwen3.5-Omni exists);
- any Qwen3-VL training-free attention readout with GUI numbers;
- any linear probe regressing absolute coordinates per layer on a Qwen model.

**The cost estimates are partial.** The ~210M figure for a GUI-Actor head at
d = 5120 is arithmetic from the released code's structure, not a published
number. Crop-pass prefill cost is described by scale only. Only the encode
times are measured here.

## Follow-ups

- An attention-readout run (`c5.py` and siblings) was in flight in
  `.scratch/latent-probe/` while this was written. **This finding does not
  incorporate it.** The transfer table's C5 rows are what it should be checked
  against.
- Spec 11's candidate list should gain GUI-Actor's pointer head beside C5, and
  E6 should compare tiles, a column crop, the attention map and a whole-image
  2× upsample as pass-2 seeds, not only a crop at the chain's point.
- `docs/specs/decide/11-point-in-one-pass.md`'s two-pass-box section should say
  the certificate proves chain-equivalence and cannot flag the 22.

## Sources

The served model and the Qwen family:

- [Qwen/Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B) and [config.json](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json) (read from the pinned snapshot, byte-identical to `raw/main` on 2026-09-21)
- [Qwen/Qwen3.5-27B model card](https://huggingface.co/Qwen/Qwen3.5-27B) and [config.json](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/config.json)
- [Qwen/Qwen3.6-27B model card](https://huggingface.co/Qwen/Qwen3.6-27B)
- [QwenLM/Qwen3.5 README](https://github.com/QwenLM/Qwen3.5)
- [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) and [config.json](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/config.json); benchmark images [2B-32B Instruct](https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-VL/qwen3vl_2b_32b_vl_instruct.jpg), [2B-32B Thinking](https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-VL/qwen3vl_2b_32b_vl_thinking.jpg), [235B Instruct](https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-VL/table_nothinking_vl.jpg)
- [Qwen3-VL Technical Report, arXiv:2511.21631](https://arxiv.org/abs/2511.21631)
- [Qwen2.5-VL Technical Report, arXiv:2502.13923](https://arxiv.org/abs/2502.13923) and the [Qwen2.5-VL blog](https://qwenlm.github.io/blog/qwen2.5-vl/)
- [Qwen2-VL, arXiv:2409.12191](https://arxiv.org/abs/2409.12191)
- [QwenLM/Qwen3-VL README](https://github.com/QwenLM/Qwen3-VL/blob/main/README.md); cookbooks [2d_grounding](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb), [computer_use](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/computer_use.ipynb), [mobile_agent](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/mobile_agent.ipynb)
- transformers 5.17.0, [`models/qwen3_5`](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_5) (read locally: `F:\ai\ngram-venv\Lib\site-packages\transformers\models\qwen3_5\`) and [`models/qwen3_vl`](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_vl)
- [DeepStack, arXiv:2406.04334](https://arxiv.org/abs/2406.04334)

Where attention localizes, and hybrids:

- [Seeing but Not Believing, arXiv:2510.17771](https://arxiv.org/abs/2510.17771)
- [Self-Improving Small Object Grounding in LVLMs, arXiv:2606.01612](https://arxiv.org/abs/2606.01612)
- [FlexSelect, arXiv:2506.00993](https://arxiv.org/abs/2506.00993)
- [Focus When Necessary, arXiv:2606.16158](https://arxiv.org/abs/2606.16158)
- [Gaze Heads, arXiv:2606.14703](https://arxiv.org/abs/2606.14703)
- [MLLMs Know Where to Look (ViCrop), arXiv:2502.17422](https://arxiv.org/abs/2502.17422) and [code, `qwen2_5_methods.py`](https://github.com/saccharomycetes/mllms_know/blob/main/qwen2_5_methods.py)
- [Your LVLM Only Needs A Few Attention Heads for Visual Grounding, arXiv:2503.06287](https://arxiv.org/abs/2503.06287)
- [Shaking Up VLMs (Transformers vs Mamba), arXiv:2409.05395](https://arxiv.org/abs/2409.05395)
- [Understanding the Skill Gap in Recurrent Language Models (Gather-and-Aggregate), arXiv:2504.18574](https://arxiv.org/abs/2504.18574)

One-pass readouts:

- [TAG, arXiv:2412.10840](https://arxiv.org/abs/2412.10840)
- [GUI-Actor, arXiv:2506.03143](https://arxiv.org/abs/2506.03143) and [microsoft/GUI-Actor](https://github.com/microsoft/GUI-Actor) (`src/gui_actor/modeling_qwen25vl.py`)
- [Trifuse, arXiv:2602.06351](https://arxiv.org/abs/2602.06351)
- [GUI-AIMA, arXiv:2511.00810](https://arxiv.org/abs/2511.00810)
- [V2P, arXiv:2508.13634](https://arxiv.org/abs/2508.13634)

Test-time zoom and resolution:

- [ScreenSpot-Pro (ReGround, ScreenSeekeR), arXiv:2504.07981](https://arxiv.org/abs/2504.07981)
- [RegionFocus (Visual Test-time Scaling for GUI Agent Grounding), arXiv:2505.00684](https://arxiv.org/abs/2505.00684)
- [ZoomClick, arXiv:2512.05941](https://arxiv.org/abs/2512.05941)
- [MAI-UI, arXiv:2512.22047](https://arxiv.org/abs/2512.22047)
- [MVP, arXiv:2512.08529](https://arxiv.org/abs/2512.08529)
- [InnerZoom (One Forward Beats Two), arXiv:2606.30084](https://arxiv.org/abs/2606.30084)
- [GUI-Cursor (Learning GUI Grounding with Spatial Reasoning from Visual Feedback), arXiv:2509.21552](https://arxiv.org/abs/2509.21552)

Probing:

- [Decodable Is Not Grounded, arXiv:2606.31257](https://arxiv.org/abs/2606.31257)
- [Analysis-by-Proxy, arXiv:2607.06445](https://arxiv.org/abs/2607.06445)

Settled here, cited but not re-read:

- [UGround (set-of-mark decreases open-weight models), arXiv:2410.05243](https://arxiv.org/abs/2410.05243)
