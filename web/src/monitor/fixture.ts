// A scrape of `GET /ui/metrics` in the shape `crates/server/src/metrics.rs`
// renders (GitHub #89/#90, ADR 0017 and ADR 0030 §Observability): every
// contract series, in the order and with the labels `Metrics::render` writes
// them, on a load that has served traffic and retained state.
// Test data only — nothing in the page imports it.

export const IGNIS_EXPOSITION = `# HELP ignis_build_info Constant build identity with value 1.
# TYPE ignis_build_info gauge
ignis_build_info{version="0.1.0"} 1
# HELP ignis_scheduler_requests Current requests by observable scheduler state.
# TYPE ignis_scheduler_requests gauge
ignis_scheduler_requests{state="waiting"} 2
ignis_scheduler_requests{state="running"} 5
# HELP ignis_requests_accepted_total Accepted submissions.
# TYPE ignis_requests_accepted_total counter
ignis_requests_accepted_total 42
# HELP ignis_requests_completed_total Completed requests.
# TYPE ignis_requests_completed_total counter
ignis_requests_completed_total 30
# HELP ignis_requests_cancelled_total Accepted requests cancelled before completion.
# TYPE ignis_requests_cancelled_total counter
ignis_requests_cancelled_total 3
# HELP ignis_generated_tokens_total Generated tokens on completed requests.
# TYPE ignis_generated_tokens_total counter
ignis_generated_tokens_total 12345
# HELP ignis_decoded_tokens_total Tokens generated so far, counted as each one is emitted.
# TYPE ignis_decoded_tokens_total counter
ignis_decoded_tokens_total 12400
# HELP ignis_kv_cache_evictions_total Cumulative host-tier evictions.
# TYPE ignis_kv_cache_evictions_total counter
ignis_kv_cache_evictions_total 4
# HELP ignis_prefix_reused_tokens_total Cumulative tokens skipped through sibling-prefix reuse.
# TYPE ignis_prefix_reused_tokens_total counter
ignis_prefix_reused_tokens_total 8192
# HELP ignis_retained_reused_tokens_total Cumulative tokens skipped through retained state, by residency tier.
# TYPE ignis_retained_reused_tokens_total counter
ignis_retained_reused_tokens_total{tier="device",kind="checkpoint"} 41200
ignis_retained_reused_tokens_total{tier="device",kind="prefix"} 9800
ignis_retained_reused_tokens_total{tier="kv_ram",kind="checkpoint"} 6400
ignis_retained_reused_tokens_total{tier="kv_ram",kind="prefix"} 0
# HELP ignis_retained_state_hits_total Retained state chosen to resume from or brought back, by residency tier.
# TYPE ignis_retained_state_hits_total counter
ignis_retained_state_hits_total{tier="device",kind="checkpoint"} 18
ignis_retained_state_hits_total{tier="device",kind="prefix"} 7
ignis_retained_state_hits_total{tier="kv_ram",kind="checkpoint"} 3
ignis_retained_state_hits_total{tier="kv_ram",kind="prefix"} 0
# HELP ignis_retained_state_misses_total First prefill chunks with no retained checkpoint matching in the tier.
# TYPE ignis_retained_state_misses_total counter
ignis_retained_state_misses_total{tier="device",kind="checkpoint"} 11
ignis_retained_state_misses_total{tier="device",kind="prefix"} 0
ignis_retained_state_misses_total{tier="kv_ram",kind="checkpoint"} 5
ignis_retained_state_misses_total{tier="kv_ram",kind="prefix"} 0
# HELP ignis_retained_state_spills_total Retained checkpoints and prefixes spilled into the tier.
# TYPE ignis_retained_state_spills_total counter
ignis_retained_state_spills_total{tier="device",kind="checkpoint"} 0
ignis_retained_state_spills_total{tier="device",kind="prefix"} 0
ignis_retained_state_spills_total{tier="kv_ram",kind="checkpoint"} 6
ignis_retained_state_spills_total{tier="kv_ram",kind="prefix"} 2
# HELP ignis_retained_state_discards_total Retained checkpoints and prefixes discarded from the tier.
# TYPE ignis_retained_state_discards_total counter
ignis_retained_state_discards_total{tier="device",kind="checkpoint"} 4
ignis_retained_state_discards_total{tier="device",kind="prefix"} 9
ignis_retained_state_discards_total{tier="kv_ram",kind="checkpoint"} 1
ignis_retained_state_discards_total{tier="kv_ram",kind="prefix"} 0
# HELP ignis_retained_state_restores_total Retained state restored from the tier.
# TYPE ignis_retained_state_restores_total counter
ignis_retained_state_restores_total{tier="device",kind="checkpoint"} 0
ignis_retained_state_restores_total{tier="device",kind="prefix"} 0
ignis_retained_state_restores_total{tier="kv_ram",kind="checkpoint"} 3
ignis_retained_state_restores_total{tier="kv_ram",kind="prefix"} 1
# HELP ignis_retained_slot_skips_total Publishes and captures that found no retained slot or no tail page.
# TYPE ignis_retained_slot_skips_total counter
ignis_retained_slot_skips_total{reason="publish_skipped_no_slot"} 12
ignis_retained_slot_skips_total{reason="capture_skipped_no_slot"} 5
ignis_retained_slot_skips_total{reason="capture_skipped_no_page"} 1
# HELP ignis_vram_reserved_bytes Device bytes this load reserved, by the plan line that reserved them.
# TYPE ignis_vram_reserved_bytes gauge
ignis_vram_reserved_bytes{line="weights"} 17179869184
ignis_vram_reserved_bytes{line="cuda_context"} 587202560
ignis_vram_reserved_bytes{line="workspace"} 1342177280
ignis_vram_reserved_bytes{line="media_embedding"} 402653184
ignis_vram_reserved_bytes{line="sampling"} 33554432
ignis_vram_reserved_bytes{line="decode_graph"} 16777216
ignis_vram_reserved_bytes{line="verify_round"} 268435456
ignis_vram_reserved_bytes{line="drafter_round"} 134217728
ignis_vram_reserved_bytes{line="lane_state"} 1073741824
ignis_vram_reserved_bytes{line="retained_slots"} 2415919104
ignis_vram_reserved_bytes{line="residual"} 268435456
# HELP ignis_vram_budget_bytes The device budget the plan was laid out inside.
# TYPE ignis_vram_budget_bytes gauge
ignis_vram_budget_bytes 31138512896
# HELP ignis_kv_pool_pages Pages the KV pool holds.
# TYPE ignis_kv_pool_pages gauge
ignis_kv_pool_pages 4032
# HELP ignis_kv_page_bytes One KV page's bytes.
# TYPE ignis_kv_page_bytes gauge
ignis_kv_page_bytes 1835008
# HELP ignis_kv_pool_used_pages KV pool pages reserved by running requests and retained state.
# TYPE ignis_kv_pool_used_pages gauge
ignis_kv_pool_used_pages 1536
# HELP ignis_kv_ram_arena_bytes The pinned host KV-RAM arena: what it holds, and what is used of it.
# TYPE ignis_kv_ram_arena_bytes gauge
ignis_kv_ram_arena_bytes{state="capacity"} 8589934592
ignis_kv_ram_arena_bytes{state="used"} 2147483648
# HELP ignis_retained_slots Retained slots this load hands out, and how many hold an image.
# TYPE ignis_retained_slots gauge
ignis_retained_slots{state="capacity"} 10
ignis_retained_slots{state="in_use"} 7
# HELP ignis_requests_rejected_total Rejected submissions by fixed reason.
# TYPE ignis_requests_rejected_total counter
ignis_requests_rejected_total{reason="full"} 6
ignis_requests_rejected_total{reason="unknown_model"} 1
ignis_requests_rejected_total{reason="oversized"} 0
# HELP ignis_request_ttft_seconds Submission-to-first-token latency.
# TYPE ignis_request_ttft_seconds histogram
ignis_request_ttft_seconds_bucket{le="0.05"} 1
ignis_request_ttft_seconds_bucket{le="0.1"} 4
ignis_request_ttft_seconds_bucket{le="0.25"} 10
ignis_request_ttft_seconds_bucket{le="0.5"} 20
ignis_request_ttft_seconds_bucket{le="1"} 28
ignis_request_ttft_seconds_bucket{le="2.5"} 30
ignis_request_ttft_seconds_bucket{le="5"} 30
ignis_request_ttft_seconds_bucket{le="10"} 30
ignis_request_ttft_seconds_bucket{le="30"} 30
ignis_request_ttft_seconds_bucket{le="60"} 30
ignis_request_ttft_seconds_bucket{le="120"} 30
ignis_request_ttft_seconds_bucket{le="300"} 30
ignis_request_ttft_seconds_bucket{le="+Inf"} 30
ignis_request_ttft_seconds_sum 15.3
ignis_request_ttft_seconds_count 30
# HELP ignis_request_duration_seconds Submission-to-completion latency.
# TYPE ignis_request_duration_seconds histogram
ignis_request_duration_seconds_bucket{le="0.1"} 0
ignis_request_duration_seconds_bucket{le="0.25"} 0
ignis_request_duration_seconds_bucket{le="0.5"} 1
ignis_request_duration_seconds_bucket{le="1"} 2
ignis_request_duration_seconds_bucket{le="2.5"} 8
ignis_request_duration_seconds_bucket{le="5"} 15
ignis_request_duration_seconds_bucket{le="10"} 24
ignis_request_duration_seconds_bucket{le="30"} 29
ignis_request_duration_seconds_bucket{le="60"} 30
ignis_request_duration_seconds_bucket{le="120"} 30
ignis_request_duration_seconds_bucket{le="300"} 30
ignis_request_duration_seconds_bucket{le="600"} 30
ignis_request_duration_seconds_bucket{le="+Inf"} 30
ignis_request_duration_seconds_sum 210.75
ignis_request_duration_seconds_count 30
`;
