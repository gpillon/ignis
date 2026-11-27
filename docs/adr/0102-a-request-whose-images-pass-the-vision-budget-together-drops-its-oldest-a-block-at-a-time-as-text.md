---
status: accepted
---

# A request whose images pass the vision budget together drops its oldest, a block at a time, as text

An agent keeps the images it looked at in its history, and its harness sends the whole history again every turn. A
request whose images together pass `vision.max_tokens` is not refused: its oldest images are dropped, in blocks, and
the same rule holds for the byte budget (`max_encoded_media_bytes`). Refusing killed the session: once the images
together passed the budget, every later turn was refused `400 media_budget_exceeded`. The owner chose the drop as the
default, with no field to turn it off. A single image too large for the budget is shrunk, as before the rule.

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
  (`CODING_STANDARDS.md`, ADR 0046), like having no field to turn the rule off.
- **The placeholder.** A dropped image renders where it was as a line of text, not as vision placeholders: the model
  reads that an image was there. The request's parts are rewritten before the template renders them, so every route
  that takes images (chat completions, Responses, decide) follows the rule, and the template pairs the kept images
  with their parts in order.
- **No work for a dropped image.** It is weighed from its header (its dimensions, the format's own, a JPEG's EXIF
  orientation applied) and its encoded length, and never prepared: no patching, no encode, no media-cache entry. What
  the decoder would find wrong past the header is therefore not seen in a dropped image.
- **Still refused**, wherever the image sits: invalid media its header shows, a failed or timed-out fetch, an aspect
  ratio no size can fit, one image past the byte budget or the decoded-pixel limit, an item count past the item cap,
  and the request's fetches together past the byte budget (a fetch has cost its bytes once made; harnesses send
  images inline).
- **Trace.** A request's events count the images dropped beside the images the model saw. A request refused for its
  media, answered before admission, leaves a WARN event of its own with the wire code, the part and, for a budget,
  the budget, its limit and the amount requested; never the message the client read, which for a fetch names its URL,
  signature and all.

## Considered Options

- **Shrink every image in proportion.** Every image kept, with less detail; but once the budget is full every turn
  changes every image, a full prefill and a full re-encode each turn, and it blurs the detail an agent cropped for.
- **Shrink the older images in steps.** Each image keeps some detail, at the price of the most rules of the four.
- **Keep refusing, with a higher default and advice in the message.** The session still dies, only later.
