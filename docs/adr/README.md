# Architecture decision records

An ADR records a decision that is hard to reverse, surprising without its
context and the result of a real trade-off, and why it was taken: the project's
memory of why ignis is built the way it is.

- **Writing one.** A new ADR is in the `/domain-modeling` skill's short format,
  its heading the bare title, with the frontmatter
  [ADR 0081](0081-an-adrs-status-is-in-its-frontmatter-and-a-changed-decision-is-a-new-adr.md)
  describes. [ADR 0070](0070-a-spec-is-a-github-issue-and-its-tickets-are-sub-issues.md)
  lists where the repo differs from the skill, and
  [docs/agents/domain.md](../agents/domain.md) gives the numbering rule.
- **A decision that changes** is recorded by a new ADR that amends or
  supersedes the old one, whose text then stays as it was: only its
  frontmatter changes (`amended-by`, or a status of `superseded by`).
  `make records-check` refuses any other change to an ADR that is no longer
  `proposed`, a typo fix aside when its commit message says `[typo ADR-NNNN]`.
- **This file is generated** from the frontmatter by `mk/adr-index.py`
  (`make records`), and `make records-check` fails when it is out of date:
  never edit it by hand.

| ADR | Title | Status | Amended by |
|---|---|---|---|
| [0001](0001-cpp-kernel-leaf-c-abi.md) | C++ kernel leaf behind a C ABI | accepted | [0009](0009-step-level-device-resident-abi.md) |
| [0002](0002-load-ninfer-artifact.md) | The engine loads `.ninfer` artifacts directly | accepted | |
| [0003](0003-parity-gate-99pct.md) | 99% token-agreement parity gate | superseded by [0007](0007-performance-gate-not-parity.md) | |
| [0004](0004-full-admission-in-v1.md) | Full admission state machine in v1 | accepted | |
| [0005](0005-performance-first-principle.md) | Performance-first as the organizing principle | accepted | [0050](0050-ignis-serves-an-image-model.md), [0082](0082-the-gate-is-kld-against-bf16-dflash2-on-against-off-and-the-loop-rate.md), [0083](0083-the-north-star-is-the-best-local-engine-for-coding-agentic-work-and-as-a-companion.md), [0084](0084-ignis-stands-on-its-own.md) |
| [0006](0006-exclusive-gpu-testing.md) | Exclusive GPU testing (no concurrent engines) | accepted | [0084](0084-ignis-stands-on-its-own.md) |
| [0007](0007-performance-gate-not-parity.md) | The 99% gate is a performance gate, not a parity gate | accepted | [0050](0050-ignis-serves-an-image-model.md), [0082](0082-the-gate-is-kld-against-bf16-dflash2-on-against-off-and-the-loop-rate.md) |
| [0008](0008-cuda-graph-staging-buffers.md) | CUDA-graph decode replay: persistent device staging buffers | superseded by [0009](0009-step-level-device-resident-abi.md) | [0019](0019-decode-cuda-graph-slot-indirection.md) |
| [0009](0009-step-level-device-resident-abi.md) | Step-level, device-resident C ABI (the forward pass lives in the leaf) | accepted | [0016](0016-step-abi-extensible-options.md) |
| [0010](0010-vendored-reference-kernels.md) | Kernel policy: vendor the reference ops verbatim, under a manifest | accepted | |
| [0011](0011-tracing-for-structured-logging.md) | `tracing` + `tracing-subscriber` as the structured logging foundation | accepted | [0025](0025-scheduler-interval-as-a-log-event.md) |
| [0012](0012-request-id-as-trace-id.md) | Internal tracing spans reuse `request.id` as the OTel `trace_id` | accepted | |
| [0013](0013-no-typed-attribute-registry.md) | No typed attribute registry; discipline + regression tests only | accepted | |
| [0014](0014-g1-teacher-forced-canary-agreement.md) | G1's cross-engine canary floor is teacher-forced next-token agreement | superseded by [0082](0082-the-gate-is-kld-against-bf16-dflash2-on-against-off-and-the-loop-rate.md) | [0043](0043-a-second-model-without-a-ninfer-reference.md) |
| [0015](0015-g2-live-live-cold-prefix-gate.md) | G2 is judged live/live against the reference, on cold-prefix samples | superseded by [0082](0082-the-gate-is-kld-against-bf16-dflash2-on-against-off-and-the-loop-rate.md) | [0021](0021-live-live-launch-pooling.md), [0043](0043-a-second-model-without-a-ninfer-reference.md) |
| [0016](0016-step-abi-extensible-options.md) | Step-ABI calls take extensible options structs, and G2 spends the one extension | accepted | [0024](0024-sequence-state-transfer.md) |
| [0017](0017-prometheus-metrics.md) | Opt-in Prometheus metrics with zero inference-path work | accepted | [0025](0025-scheduler-interval-as-a-log-event.md), [0030](0030-device-memory-reserved-at-load.md), [0045](0045-the-kv-pool-follows-residency-and-state-goes-down-to-disk.md), [0046](0046-one-declaration-per-config-field.md), [0058](0058-every-route-has-a-deadline-and-a-passed-one-answers-what-was-generated.md), [0089](0089-a-grammar-checks-each-sampled-token-and-repairs-a-miss-and-the-model-is-told-the-schema.md) |
| [0018](0018-chunk-level-prefill-decode-interleaving.md) | Prefill and decode interleave at chunk granularity, driven by the Rust scheduler | accepted | |
| [0019](0019-decode-cuda-graph-slot-indirection.md) | decode CUDA graph replay via device-resident slot indirection | accepted | [0020](0020-batch-wide-decode-round.md) |
| [0020](0020-batch-wide-decode-round.md) | the decode round is one batch-wide model traversal | accepted | |
| [0021](0021-live-live-launch-pooling.md) | a live/live gate pools at least two independent process launches per engine | superseded by [0082](0082-the-gate-is-kld-against-bf16-dflash2-on-against-off-and-the-loop-rate.md) | |
| [0022](0022-two-kv-formats-bf16-as-oracle.md) | two KV formats, with BF16 retained as the correctness oracle | accepted | |
| [0023](0023-unified-eviction-priority.md) | one eviction priority across GPU residency and the host tier | accepted | [0029](0029-cross-request-state-reuse.md), [0045](0045-the-kv-pool-follows-residency-and-state-goes-down-to-disk.md) |
| [0024](0024-sequence-state-transfer.md) | sequence state transfer: an opaque versioned blob, and device-to-device cloning for prefix reuse | accepted | [0029](0029-cross-request-state-reuse.md), [0045](0045-the-kv-pool-follows-residency-and-state-goes-down-to-disk.md) |
| [0025](0025-scheduler-interval-as-a-log-event.md) | Scheduler interval counters are a DEBUG log event | accepted | |
| [0026](0026-playground-embedded-when-built.md) | The Playground is embedded when built, served by default | accepted | |
| [0027](0027-make-as-the-dev-entrypoint.md) | make is the development entry point, with per-OS hooks | accepted | |
| [0028](0028-expose-modes-always-keyed.md) | exposure is a named mode, and an exposed server always has a key | accepted | |
| [0029](0029-cross-request-state-reuse.md) | cross-request state reuse: content-matched retained state across three residency tiers | accepted | [0045](0045-the-kv-pool-follows-residency-and-state-goes-down-to-disk.md) |
| [0030](0030-device-memory-reserved-at-load.md) | device memory reserved at load, within a VRAM budget | accepted | [0035](0035-the-vision-embedding-outlives-its-encode.md), [0045](0045-the-kv-pool-follows-residency-and-state-goes-down-to-disk.md), [0092](0092-the-cuda-context-is-measured-when-it-is-made-and-a-reload-adds-back-what-the-driver-reports.md) |
| [0031](0031-vendored-kernel-bottleneck-exemption.md) | a vendored kernel that is measured as the bottleneck is not untouchable | accepted | |
| [0032](0032-release-through-the-container-build.md) | the Linux release is the container build, and it is one pipeline | accepted | [0086](0086-the-builds-run-beside-the-tests-and-nothing-is-published-before-they-pass.md) |
| [0033](0033-the-server-fetches-its-own-model.md) | the server fetches its own model, and asks first only when somebody is there | accepted | [0047](0047-a-catalog-the-operator-can-extend.md) |
| [0034](0034-the-leaf-answers-without-generating.md) | the leaf answers without generating | accepted | [0089](0089-a-grammar-checks-each-sampled-token-and-repairs-a-miss-and-the-model-is-told-the-schema.md) |
| [0035](0035-the-vision-embedding-outlives-its-encode.md) | the vision embedding outlives its encode | accepted | [0075](0075-the-vision-tower-is-staged-and-the-embedding-pool-host-mapped-unless-the-load-asks-for-the-device.md) |
| [0036](0036-the-v1-surface-documents-itself.md) | the `/v1` surface documents itself, and its reference needs no key | accepted | |
| [0037](0037-a-vendored-file-may-carry-a-correctness-patch.md) | a vendored file may carry a patch that fixes a known correctness bug | accepted | |
| [0038](0038-the-seam-carries-one-attention-head.md) | the `Compute` seam carries one attention head's scores | accepted | |
| [0039](0039-the-seam-carries-a-head-set.md) | the attention readout carries a head set | accepted | |
| [0040](0040-the-readout-carries-the-peak-s-neighbours.md) | the attention readout carries each peak's neighbours | accepted | |
| [0041](0041-the-readout-reads-a-text-span.md) | the attention readout reads a text span, in whole rows | accepted | |
| [0042](0042-locate-copies-over-a-folded-state.md) | a locate narrows by attention heads and decides by a labelled choice, by kind of text | accepted | |
| [0043](0043-a-second-model-without-a-ninfer-reference.md) | a second model, Qwen3.8-Flash-Next, served without a ninfer reference | accepted | [0051](0051-kernel-code-is-ours-adapted-or-vendored.md), [0082](0082-the-gate-is-kld-against-bf16-dflash2-on-against-off-and-the-loop-rate.md) |
| [0044](0044-experts-in-trellis-with-a-k-per-expert.md) | Flash-Next's experts are trellis-coded with a bit width per expert, on our own kernels | accepted | |
| [0045](0045-the-kv-pool-follows-residency-and-state-goes-down-to-disk.md) | the KV pool is sized by whether the model is resident, and evicted state goes down to disk | accepted | [0054](0054-a-resident-load-keeps-its-experts-only-in-vram.md) |
| [0046](0046-one-declaration-per-config-field.md) | one declaration per config field: flag, env, file key, default, validator | accepted | |
| [0047](0047-a-catalog-the-operator-can-extend.md) | a model catalog the operator can extend, and a `model` command to fetch from it | accepted | [0064](0064-a-catalog-entry-states-its-card-class.md), [0074](0074-a-catalog-entry-may-state-its-family-and-an-alias-may-be-another-entrys-id.md) |
| [0048](0048-an-end-of-turn-inside-the-reasoning-closes-it.md) | an end of turn drawn inside an open reasoning block closes the block instead | accepted | |
| [0049](0049-an-idle-model-is-unloaded-and-loads-again-on-demand.md) | an idle model is unloaded, and the next request loads it again | accepted | [0067](0067-the-speech-model-waits-on-the-host-and-borrows-free-kv-pages.md) |
| [0050](0050-ignis-serves-an-image-model.md) | ignis serves an image model, one model at a time | accepted | [0067](0067-the-speech-model-waits-on-the-host-and-borrows-free-kv-pages.md) |
| [0051](0051-kernel-code-is-ours-adapted-or-vendored.md) | kernel code is ours, adapted from a named source, or vendored | accepted | |
| [0052](0052-a-reasoning-loop-is-taken-back-and-nudged.md) | a reasoning loop is taken back, the model is nudged, and the client is told | accepted | [0053](0053-a-loop-is-acted-on-once-it-persists.md) |
| [0053](0053-a-loop-is-acted-on-once-it-persists.md) | a loop is acted on once it persists, and a rewind keeps something | accepted | |
| [0054](0054-a-resident-load-keeps-its-experts-only-in-vram.md) | a resident Flash-Next load keeps its experts only in VRAM | accepted | |
| [0055](0055-a-pinned-hot-set-is-opt-in.md) | a pinned hot set of experts: a field off by default, 2 GiB in the rtx5090 profile | accepted | |
| [0056](0056-speculation-past-eight-verify-rows-is-numerically-equivalent.md) | speculation past 8 verify rows is numerically equivalent, not bit-exact | accepted | |
| [0057](0057-rope-scaling-is-auto-by-default.md) | rope scaling is `auto` by default: the YaRN factor follows the context | accepted | |
| [0058](0058-every-route-has-a-deadline-and-a-passed-one-answers-what-was-generated.md) | every route has a deadline, and a passed one answers what was generated | accepted | |
| [0059](0059-flash-next-applies-yarn-on-q-and-k.md) | Flash-Next applies YaRN, the attention factor folded into cos and sin on q and k | accepted | |
| [0060](0060-the-27b-has-a-trellis-body-for-small-cards.md) | the 27B gets a second body format, trellis, for the 16 GB card class | accepted | |
| [0061](0061-prefill-attention-reads-the-hq-codec.md) | prefill attention reads the hq-e8-2b codec: no prompt planes, on every card | accepted | |
| [0062](0062-the-token-embedding-lives-in-host-memory.md) | the 27B's token embedding lives in pinned host memory, on every card | accepted | |
| [0063](0063-the-profile-follows-the-card-class.md) | the profile follows the card class, and its defaults adapt to the budget | accepted | [0075](0075-the-vision-tower-is-staged-and-the-embedding-pool-host-mapped-unless-the-load-asks-for-the-device.md), [0079](0079-a-catalog-entry-states-the-oldest-gpu-generation-it-loads-on.md) |
| [0064](0064-a-catalog-entry-states-its-card-class.md) | a catalog entry states the smallest card class it loads on; aliases resolve within a family; a switch refuses before teardown | accepted | [0074](0074-a-catalog-entry-may-state-its-family-and-an-alias-may-be-another-entrys-id.md), [0079](0079-a-catalog-entry-states-the-oldest-gpu-generation-it-loads-on.md), [0084](0084-ignis-stands-on-its-own.md) |
| [0065](0065-the-bf16-exception-tensors-can-be-held-in-fp8.md) | the 27B's BF16 exception tensors can be held in FP8, converted at load | accepted | |
| [0066](0066-the-user-guide-is-a-site-built-from-docs-user.md) | the user guide is a Starlight site built from `docs/user/` alone, published on a release tag | accepted | |
| [0067](0067-the-speech-model-waits-on-the-host-and-borrows-free-kv-pages.md) | the speech model waits on the host and borrows free KV pages | accepted | |
| [0068](0068-the-speech-models-cpu-executor-is-ours-in-rust.md) | the speech model's CPU executor is ours, in Rust | accepted | |
| [0069](0069-the-speakers-come-from-nemotron-diarization-beside-qwen3-asr.md) | the speakers come from Nemotron 3 Diarization, beside Qwen3-ASR | accepted | |
| [0070](0070-a-spec-is-a-github-issue-and-its-tickets-are-sub-issues.md) | A spec is a GitHub issue and its tickets are sub-issues | accepted | [0081](0081-an-adrs-status-is-in-its-frontmatter-and-a-changed-decision-is-a-new-adr.md), [0084](0084-ignis-stands-on-its-own.md) |
| [0071](0071-the-server-downloads-and-removes-models-from-the-server-view.md) | The server downloads and removes models after start, from the Server view | accepted | |
| [0072](0072-the-playgrounds-system-prompt-never-changes.md) | The Playground's system prompt never changes: what changes rides after the history | accepted | |
| [0073](0073-a-workspace-is-kept-by-the-page-and-ignis-stores-no-files.md) | A workspace is kept by the page and ignis stores no files | accepted | |
| [0074](0074-a-catalog-entry-may-state-its-family-and-an-alias-may-be-another-entrys-id.md) | A catalog entry may state its family, and an alias may be another entry's id | accepted | |
| [0075](0075-the-vision-tower-is-staged-and-the-embedding-pool-host-mapped-unless-the-load-asks-for-the-device.md) | The vision tower is staged and the embedding pool host-mapped, unless the load asks for the device | accepted | |
| [0076](0076-the-running-notes-send-the-notes-back-and-the-recap-does-not.md) | The running notes send the notes back to the text model and the recap does not | accepted | |
| [0077](0077-a-vendored-file-may-carry-a-portability-patch.md) | A vendored file may carry a portability patch | accepted | [0080](0080-on-sm8x-the-trellis-prefill-decodes-each-matrix-once-per-chunk-and-multiplies-on-the-integer-mma.md), [0091](0091-on-a-card-smaller-than-the-rtx-5090-the-gdn-gating-planner-takes-the-largest-split-it-holds-and-its-test-caps-the-sm-count.md) |
| [0078](0078-the-built-in-catalog-names-each-entrys-license-and-may-ship-restricted-weights.md) | The built-in catalog names each entry's license and may ship weights whose license restricts commercial use | accepted | |
| [0079](0079-a-catalog-entry-states-the-oldest-gpu-generation-it-loads-on.md) | A catalog entry states the oldest GPU generation it loads on | accepted | |
| [0080](0080-on-sm8x-the-trellis-prefill-decodes-each-matrix-once-per-chunk-and-multiplies-on-the-integer-mma.md) | On sm_8x the trellis prefill decodes each matrix once per chunk and multiplies on the integer MMA | accepted | [0085](0085-on-blackwell-too-the-27bs-gdn-prefill-runs-ignis-own-kernels.md) |
| [0081](0081-an-adrs-status-is-in-its-frontmatter-and-a-changed-decision-is-a-new-adr.md) | An ADR's status is in its frontmatter, and a decision that changes is recorded by a new ADR | accepted | |
| [0082](0082-the-gate-is-kld-against-bf16-dflash2-on-against-off-and-the-loop-rate.md) | The gate is KLD against BF16, DFlash2 on against off and the loop rate | accepted | |
| [0083](0083-the-north-star-is-the-best-local-engine-for-coding-agentic-work-and-as-a-companion.md) | The north-star is the best local engine for coding, for agentic work and as a companion | accepted | |
| [0084](0084-ignis-stands-on-its-own.md) | ignis stands on its own | accepted | |
| [0085](0085-on-blackwell-too-the-27bs-gdn-prefill-runs-ignis-own-kernels.md) | On Blackwell too, the 27B's GDN prefill runs ignis's own kernels | accepted | |
| [0086](0086-the-builds-run-beside-the-tests-and-nothing-is-published-before-they-pass.md) | The builds run beside the tests, and nothing is published before they pass | accepted | |
| [0087](0087-flash-nexts-vision-encoder-ships-in-a-companion-container-and-encodes-in-memory-its-prefill-already-reserves.md) | Flash-Next's vision encoder ships in a companion container and encodes in memory its prefill already reserves | accepted | |
| [0088](0088-a-vendored-registry-that-lacks-a-case-ignis-needs-may-be-extended-or-rewritten-when-small.md) | A vendored registry that lacks a case ignis needs may be extended, or rewritten when it is small | accepted | |
| [0089](0089-a-grammar-checks-each-sampled-token-and-repairs-a-miss-and-the-model-is-told-the-schema.md) | A grammar checks each sampled token and repairs a miss, and the model is told the schema | accepted | |
| [0090](0090-the-27bs-sampled-lanes-draw-their-drafts-and-take-them-by-an-accept-of-ignis-s-own.md) | The 27B's sampled lanes draw their drafts and take them by an accept of ignis's own | accepted | |
| [0091](0091-on-a-card-smaller-than-the-rtx-5090-the-gdn-gating-planner-takes-the-largest-split-it-holds-and-its-test-caps-the-sm-count.md) | On a card smaller than the RTX 5090 the GDN gating planner takes the largest split it holds, and its test caps the SM count | accepted | |
| [0092](0092-the-cuda-context-is-measured-when-it-is-made-and-a-reload-adds-back-what-the-driver-reports.md) | The CUDA context is measured when it is made, and a reload adds back what the driver reports | accepted | |
