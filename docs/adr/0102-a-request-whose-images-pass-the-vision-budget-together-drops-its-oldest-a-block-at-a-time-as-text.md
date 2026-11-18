---
status: accepted
amends: [ADR-0099]
---

# A request whose images pass the vision budget together drops its oldest, a block at a time, as text

An agent keeps the images it looked at in its history, and its harness sends the whole history again every turn. Once
those images together passed `vision.max_tokens`, ignis refused every later turn `400 media_budget_exceeded` and the
session died (#673: opencode, 36 crops, 32,557 tokens, then one crop more). ignis already shrank a single image too
large for the budget (#248); several that passed it only together were refused. The owner chose, at triage on
2026-11-17, to drop the oldest images in blocks, as the default, with no field to turn it off, and to apply the same
rule to the byte budget (`max_encoded_media_bytes`). An image left in a harness's history past `vision.max_tokens` is
therefore no longer refused on every later turn, as [ADR 0099](0099-a-harness-reaches-ignis-through-an-mcp-server-inside-ignis-and-a-skill-ignis-serves-whose-scripts-carry-the-files.md)
has it for the connector's previews.

- **Blocks.** A request's image parts, in prompt order, form blocks: a block takes images until the next would carry
  it past half of `vision.max_tokens` in tokens (raw patches with them), or past half of the byte budget in bytes. An
  image past half alone is a block of its own. An image weighs the tokens it takes after the single-image shrink, and
  its encoded bytes.
- **The drop.** When the images together pass either budget, the fewest whole blocks, oldest first, that bring the
  rest within both are dropped. The last block never is. A request within both is unchanged.
- **Why blocks, and why half.** A block's bounds depend only on the images before it, so on a history that grows at
  its end a closed block never changes and each is dropped once: the prompt changes at its first image once a block,
  about every half a budget of new images, and prefix reuse holds between drops. Half is the owner's: it follows
  `vision.max_tokens`, so it is no field of its own, a deliberate exception to "Limits are config fields"
  (`CODING_STANDARDS.md`, [ADR 0046](0046-one-declaration-per-config-field.md)) decided at triage, like
  having no field to turn the rule off.
- **The placeholder.** A dropped image renders where it was as the text
  `[image omitted: past this request's vision budget]`, not as vision placeholders: the model reads that an image was
  there. The request's parts are rewritten before the template renders them, so every route that takes images (chat
  completions, Responses, decide) follows the rule, and the template pairs the kept images with their parts in order.
- **No work for a dropped image.** It is weighed from its header (its dimensions, the format's own, a JPEG's EXIF
  orientation applied) and its encoded length, and never prepared: no patching, no encode, no media-cache entry. What
  the decoder would find wrong past the header is therefore not seen in a dropped image.
- **Still refused**, wherever the image sits: invalid media its header shows, a failed or timed-out fetch, an aspect
  ratio no size can fit, one image past the byte budget or the decoded-pixel limit, an item count past the item cap,
  and the request's fetches together past the byte budget (a fetch has cost its bytes once made; harnesses send
  images inline).
- **Trace.** `ignis.request.admitted`, `done` and `cancelled` carry `media.dropped_items`,
  `media.dropped_vision_tokens` and `media.dropped_bytes`; `media.items`, `media.vision_tokens` and `media.bytes`
  count the images the model saw. A request refused for its media, answered before admission, leaves a WARN event
  `ignis.media.refused` with the wire code, the part and, for a budget, the budget, its limit and the amount requested;
  never the message the client read, which for a fetch names its URL, signature and all.

## Considered options

- **Shrink every image in proportion.** Every image kept, with less detail; but once the budget is full every turn
  changes every image, a full prefill and a full re-encode each turn, and it blurs the detail an agent cropped for.
- **Shrink the older images in steps.** Each image keeps some detail, at the price of the most rules of the four.
- **Keep refusing, with a higher default and advice in the message.** The session still dies, only later.
