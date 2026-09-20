# "Evidence first" was never in the bytes

- Kind: defect
- Status: current
- Observed: 2026-09-20
- Last verified: 2026-09-20
- Scope: serving / decisions, the `/v1/decide` prompt, shared-prefix reuse
- Related: `docs/findings/2026-09-19-typed-option-logit-readout.md`, `crates/server/tests/classify_readout_gpu.rs`, `classify_option_ceiling_gpu.rs`, `classify_vision_readout_gpu.rs`, `crates/server/src/decide.rs`, GitHub #239, #240
- Superseded by: none (followed by `2026-09-20-the-evidence-belongs-in-the-system-block.md`, which takes the re-measurement this one leaves open and shows the byte order was not what the reuse turned on)

## Question

The readout finding rests on one sentence: *"Evidence-first costs nothing and
keeps reuse possible. Jev's shape is one `state` and many questions; with the
evidence at the head of the payload that is one shared token prefix and N
short suffixes."*

(ADR 0034 does **not** say this — it is the finding's implication alone. An
earlier draft of this document, and the commit message that shipped it,
attributed it to the ADR as well; they were wrong, and the ADR's own text is
silent about prompt layout.)

Spec 04's whole fan-out design is that sentence made mechanical — twenty
questions over one `state` prefill the state once. While implementing #239 a
test asserted the obvious precondition: that the rendered prompt actually
begins with the evidence.

It does not.

## Evidence

Every measurement of this technique built its user message the same way
(`classify_readout_gpu.rs:150`, and identically in the ceiling and vision
tests):

```rust
let payload = json!({
    "evidence": row.state,
    "criterion": row.question,
    "options": options,
});
// ...
ChatMessage::text("user", payload.to_string())
```

`serde_json::Map` is a `BTreeMap` in this workspace — `preserve_order` is
enabled only for `minijinja` in `crates/artifact`, not for `serde_json`
itself. So `payload.to_string()` emits its keys **sorted**:

```json
{"criterion":"…","evidence":"…","options":[…]}
```

The source order in the `json!` literal is discarded. Two questions over one
`state` therefore share exactly the characters `{"criterion":"` and then
diverge, because the criterion is what differs between them. The evidence —
the long part, the part worth reusing, the part that is 16K tokens when it is
an image — sits *after* the thing that makes each question unique.

Reproduced in `crates/server/tests/decide_wire.rs`
(`text_evidence_leads_the_payload`): the assertion
`text.find("evidence") < text.find("criterion")` fails against a `json!`-built
payload and passes against an explicitly ordered one.

## Finding

**The intent was documented; the bytes never matched it.** "Evidence first"
describes the order the `json!` literal is *written* in, which is not the
order it serializes in. Nothing caught this because nothing had yet asked the
prompt to be a shared prefix: the finding measured accuracy and answer mass,
both of which are indifferent to key order, and the reuse claim was an
implication drawn from the source rather than from the output.

Two consequences, and they point in opposite directions.

**For reuse, this is a real defect.** Every claim about one `state` becoming a
shared token prefix — the readout finding's "evidence-first costs nothing and
keeps reuse possible", spec 04's acceptance 1 ("twenty questions over one text
`state` prefill the state once") and acceptance 2 (an image encoded once) —
was false of the prompt as it was actually being built. A fan-out over the sorted
payload would have re-prefilled the whole evidence per question and
re-encoded the image per question.

**For the accuracy numbers, nothing moves — but they are numbers about a
different prompt than the one we now ship.** The 0.934 balanced accuracy, the
median 99.8% answer mass and the 100% in-slot rate at every width out to 256
were all measured with the criterion in front of the evidence. `/v1/decide`
now emits the evidence first, which is the prompt the design calls for and
*not* the prompt those numbers came from. There is no reason to expect the
model to care — the payload is a JSON object either way and both orders are
well-formed — but "no reason to expect" is not a measurement.

## Implications

**What is fixed, and what is not.** The defect is fixed in the only place
that serves traffic: `crates/server/src/decide.rs` builds the payload with
`payload_text`, which emits the keys in the order given rather than through
`serde_json::Map`. The order is load-bearing, so it is now stated in code
rather than implied by the order of a literal, and
`decide_wire.rs::text_evidence_leads_the_payload` fails if that stops being
true (verified by mutation).

Three things are deliberately *not* fixed:

- **The GPU tests still sort.** They are the record of what was measured, and
  rewriting them would silently restate history. A re-run of
  `classify_readout_gpu.rs` against the evidence-first payload is the
  measurement that would close the gap between the shipped prompt and the
  published accuracy, and it is not done here. *(Taken on 2026-09-20 by
  GitHub #240: that sweep now runs **both** layouts over the same rows on one
  load, the sorted one kept verbatim as the record —
  `2026-09-20-the-evidence-belongs-in-the-system-block.md`.)*
- **No re-measurement has been taken**, so the accuracy numbers still
  describe the old prompt. *(Since taken, and the shipped prompt is not the
  evidence-first payload this document proposed either: #240 moved the
  evidence into the system block, where the reuse actually lives. Balanced
  accuracy 0.934 -> 0.963.)*
- **The claim still stands in the readout finding's own Implications**, with
  a pointer to this document beside it rather than a rewrite. Erasing it
  would erase that the claim was made and believed, which is the part worth
  keeping.

Two things worth carrying elsewhere:

- Enabling `serde_json/preserve_order` workspace-wide would have fixed this
  invisibly and reordered the keys of every other JSON the server emits. The
  explicit builder is three lines and changes nothing else.
- A `json!` literal reads like an ordered document and is not one. Anywhere
  key order carries meaning — a prompt, a cache key, a signature — the order
  has to be asserted by a test, because the source will keep looking right.
