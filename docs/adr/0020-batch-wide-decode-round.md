---
status: accepted
---

# the decode round is one batch-wide model traversal

A decode round is one traversal of the model at batch width W, for every exact width a round serves: one call per
op per layer with the lanes on the batch axis, so that W lanes stream the model's weights once rather than W times.
The spec of the serving loop (GitHub #64) asked for exactly that, "one decode call spanning every decode-ready
lane, so that eight lanes stream the model's weights once rather than eight times", and the round was first the
opposite, W complete single-sequence forward passes in program order, sharing only the final sample. That round was
the whole gap to the reference in ITL p95, a decoding lane's blocked inter-token interval being a prefill chunk
plus a decode round: at four lanes the round cost 70.2 ms against the reference's 17.3 ms (4.06 times), 4.81 times
its own one-lane round where the reference paid 1.07, the signature of streaming the weights once per lane, while
the 1,024-token prefill chunk beside it, at 32K context, was 21% faster than the reference's (116.3 ms against
141.2 ms), the sum being 186.5 ms against 158.5 ms.

- **Isolation is a property of the row index, not of separate calls.** Every per-sequence input the traversal
  needs (token id, absolute position, physical pool slot) is read from the staging buffers of ADR 0019 as a
  contiguous vector indexed by row, and row `b` addresses lane `b`'s KV pages, GDN slot and conv taps through the
  same device-resident selectors.
- **The staged positions are the positions vector the rotation and the attention read**, with no base to offset
  from: the lanes' positions are unrelated to one another.
- **One traversal serves both paths.** Graph capture records that traversal, and a round at a width whose capture
  failed enqueues the identical call sequence directly. The eager path is still a fallback, a capture failure
  degrades performance and never refuses service, but it is not a second, differently shaped implementation of the
  round.
- **Only the decode round is batched.** Prefill remains one sequence per call, the chunk-level interleaving of
  ADR 0018 being what shares the device between a prefilling request and decoding lanes.
- No vendored kernel changes (ADR 0010): the vendored attention, recurrence, projection, norm and SwiGLU ops
  already take the lanes on their batch axis. The decode call's signature, the step ABI and every Rust call site
  are unchanged.

## Consequences

- The round's dispatch counter (`kernel_count` in the program statistics) is the layer count at every width and on
  both paths: at more than one lane it must equal the one-lane round's, which is the leaf's own evidence of one
  traversal per round.
- The replay-against-eager equality test, run at every exact width, no longer compares two independent
  implementations. It still earns its place, catching anything the capture bakes in that a direct run would have
  read fresh, but isolation needs a test of its own: the same sequences presented to the round in reverse row
  order, each one's token stream unchanged.
- The decode call advances each sequence's per-layer attention position counter at the end of a successful round,
  on both paths, so the counter has one owner and stays truthful for the per-token prefill route and for the layer
  bodies' KV-capacity check.
