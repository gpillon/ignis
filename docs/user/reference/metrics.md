---
title: "Metrics"
description: "Every Prometheus metric ignis-server exports: its name, type, meaning, and when it appears."
sidebar:
  order: 4
  badge:
    text: generated
    variant: note
editUrl: false
---

:::note
Generated from the code by `make docs-reference` (`crates/server/src/user_docs.rs`); an edit here is overwritten, and `cargo test` fails while this page and the code disagree.
:::

The metrics listener (on by default, keyless, on loopback) serves these in the Prometheus text format, and the Playground's Monitor reads the same set at `/ui/metrics` ([Telemetry and metrics](../guides/telemetry.md)). A family marked *always* is in every scrape from the start, zeros included; the others appear once what the last column names has happened, and then stay.

| Metric | Type | Meaning | Exported |
|---|---|---|---|
| `ignis_build_info` | gauge | Constant build identity with value 1. | always |
| `ignis_condition_budget_used_bytes` | gauge | Bytes of the condition prefixes the image model's jobs hold now. | always |
| `ignis_decision_answer_mass` | histogram | Share of the next-token distribution held by a decision's declared options. | after the first decision |
| `ignis_decisions_total` | counter | Questions answered by a readout, by typed primitive. | after the first decision |
| `ignis_decoded_tokens_total` | counter | Tokens generated so far, counted as each one is emitted. | always |
| `ignis_expert_bytes_moved_total` | counter | Bytes of expert projections copied host-to-device, misses and prefetches, by phase. | on a Flash-Next load |
| `ignis_expert_cache_hits_total` | counter | Selected expert projections already resident in the VRAM expert cache, or staged for their layer, by K class and phase. | on a Flash-Next load |
| `ignis_expert_cache_misses_total` | counter | Selected expert projections copied in from the pinned host pool by their own step, by K class and phase. | on a Flash-Next load |
| `ignis_expert_cache_slots` | gauge | VRAM expert cache LRU slots per K class: capacity reserved at load, and in use. A pinned hot set's slots are not among them (the load's ignis.runtime.flash_next_pinned_experts event names them). | on a Flash-Next load |
| `ignis_expert_prefetches_issued_total` | counter | Expert projections copied ahead for the next layer by the router lookahead. | on a Flash-Next load |
| `ignis_expert_prefetches_used_total` | counter | Prefetched expert projections at their first use, whenever it comes. | on a Flash-Next load |
| `ignis_expert_residency_stall_seconds_total` | counter | Device time the expert kernels waited on their step's demand copies (the misses), by phase; a step with no miss adds nothing. | on a Flash-Next load |
| `ignis_generated_tokens_total` | counter | Generated tokens on completed requests. | always |
| `ignis_image_images_total` | counter | Images the image jobs made. | on a server that has seen an image job |
| `ignis_image_job_duration_seconds` | histogram | Image job duration from its turn to its last image, completed jobs only. | on a server that has seen an image job |
| `ignis_image_job_phase_seconds_total` | counter | Seconds the image jobs spent in each phase: waiting for their turn, encoding the condition, denoising, decoding. | on a server that has seen an image job |
| `ignis_image_jobs_total` | counter | Image jobs ended, by outcome. | on a server that has seen an image job |
| `ignis_image_queue_jobs` | gauge | Image jobs waiting for their turn and running (at most one). | on a server that has seen an image job |
| `ignis_kv_cache_evictions_total` | counter | Cumulative host-tier evictions. | always |
| `ignis_kv_disk_bytes` | gauge | The KV-disk tier: its effective budget, and the bytes its files hold. | on a load with the KV-disk tier |
| `ignis_kv_disk_failures_total` | counter | KV-disk writes refused or failed (nothing lost), and files that failed their check on the way back. | on a load with the KV-disk tier |
| `ignis_kv_disk_spills_total` | counter | Live snapshots written to KV-disk, by the tier they came from. | on a load with the KV-disk tier |
| `ignis_kv_page_bytes` | gauge | One KV page's bytes. | always |
| `ignis_kv_pool_pages` | gauge | Pages the KV pool holds. | always |
| `ignis_kv_pool_used_pages` | gauge | KV pool pages reserved by running requests and retained state. | always |
| `ignis_kv_ram_arena_bytes` | gauge | The pinned host KV-RAM arena: what it holds, and what is used of it. | always |
| `ignis_kv_ram_evictions_total` | counter | Live host-tier snapshots dropped from KV-RAM to make room; the request re-prefills from the start. | always |
| `ignis_locates_total` | counter | Locate questions answered, by resolved kind, method, compression and whether they found an answer. | after the first `locate` |
| `ignis_loop_guard_looped_requests_total` | counter | Watched requests with at least one loop. | once the loop guard has watched a request |
| `ignis_loop_guard_loops_total` | counter | Loops the guard saw, by what it did about them. | once the loop guard has watched a request |
| `ignis_loop_guard_min_ratio` | histogram | Lowest loop score each watched request reached. | once the loop guard has watched a request |
| `ignis_loop_guard_rewound_tokens_total` | counter | Generated tokens the guard's rewinds took back. | once the loop guard has watched a request |
| `ignis_loop_guard_watched_requests_total` | counter | Requests the loop guard watched. | once the loop guard has watched a request |
| `ignis_ngram_read_bytes_total` | counter | Bytes of the n-gram reads issued to the artifact file. | on a Flash-Next load |
| `ignis_ngram_reads_total` | counter | Reads issued to the artifact file for n-gram rows the hot-row cache did not hold. | on a Flash-Next load |
| `ignis_ngram_rows_total` | counter | N-gram embedding rows staged for prefill spans and decode rounds, by source: the RAM hot-row cache or the artifact file. | on a Flash-Next load |
| `ignis_prefix_reused_tokens_total` | counter | Cumulative tokens skipped through sibling-prefix reuse. | always |
| `ignis_request_duration_seconds` | histogram | Submission-to-completion latency. | always |
| `ignis_request_ttft_seconds` | histogram | Submission-to-first-token latency. | always |
| `ignis_requests_accepted_total` | counter | Accepted submissions. | always |
| `ignis_requests_cancelled_total` | counter | Accepted requests cancelled before completion. | always |
| `ignis_requests_completed_total` | counter | Completed requests. | always |
| `ignis_requests_rejected_total` | counter | Rejected submissions by fixed reason. | always |
| `ignis_responses_queued_requests` | gauge | Responses WebSocket requests waiting in the server-wide admission queue. | always |
| `ignis_responses_sockets` | gauge | Open Responses API WebSocket connections. | always |
| `ignis_retained_host_bytes` | gauge | The pinned host block holding the host retained slots' images. | always |
| `ignis_retained_host_slots` | gauge | Of the retained slots, those whose images live in the pinned host block. | always |
| `ignis_retained_reused_tokens_total` | counter | Cumulative tokens skipped through retained state, by residency tier. | always |
| `ignis_retained_slot_skips_total` | counter | Publishes and captures that found no retained slot or no tail page. | always |
| `ignis_retained_slots` | gauge | Retained slots this load hands out, and how many hold an image. | always |
| `ignis_retained_state_discards_total` | counter | Retained checkpoints and prefixes discarded from the tier. | always |
| `ignis_retained_state_hits_total` | counter | Retained state chosen to resume from or brought back, by residency tier. | always |
| `ignis_retained_state_misses_total` | counter | First prefill chunks with no retained checkpoint, or no retained prefix, matching in the tier. | always |
| `ignis_retained_state_restores_total` | counter | Retained state restored from the tier. | always |
| `ignis_retained_state_spills_total` | counter | Retained checkpoints and prefixes spilled into the tier. | always |
| `ignis_scheduler_requests` | gauge | Current requests by observable scheduler state. | always |
| `ignis_speculative_accepted_tokens_total` | counter | Draft tokens committed. | after the first request that ran speculative verify rounds |
| `ignis_speculative_drafted_tokens_total` | counter | Draft tokens verified. | after the first request that ran speculative verify rounds |
| `ignis_speculative_position_accepted_total` | counter | Verify rounds that committed the draft at the position. | after the first request that ran speculative verify rounds |
| `ignis_speculative_position_drafted_total` | counter | Verify rounds that proposed a draft at the position. | after the first request that ran speculative verify rounds |
| `ignis_speculative_rounds_total` | counter | Verify rounds run. | after the first request that ran speculative verify rounds |
| `ignis_speech_audio_seconds_total` | counter | Seconds of audio the completed transcriptions covered. | on a server that has been asked for a transcription |
| `ignis_speech_commit_wait_seconds` | histogram | Wait from a transcription's commit (a batch request's last byte received) to its transcript, completed ones only. | on a server that has been asked for a transcription |
| `ignis_speech_queue_length` | gauge | Batch transcriptions waiting behind the one running. | on a server that has been asked for a transcription |
| `ignis_speech_sessions_open` | gauge | Realtime transcription sessions open. | on a server that has been asked for a transcription |
| `ignis_speech_transcriptions_total` | counter | Transcriptions ended, by executor and outcome. | on a server that has been asked for a transcription |
| `ignis_thinking_forced_closes_total` | counter | Completed requests whose reasoning block the thinking budget closed. | once the thinking budget has closed a reasoning block |
| `ignis_tokenize_requests_total` | counter | Calls to the counting routes, which never enter the scheduler, by route. | after the first call to a counting route |
| `ignis_vram_budget_bytes` | gauge | The device budget the plan was laid out inside. | always |
| `ignis_vram_reserved_bytes` | gauge | Device bytes this load reserved, by the plan line that reserved them. | always |
