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
# HELP ignis_kv_ram_evictions_total Live host-tier snapshots dropped from KV-RAM to make room; the request re-prefills from the start.
# TYPE ignis_kv_ram_evictions_total counter
ignis_kv_ram_evictions_total 2
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
# HELP ignis_retained_state_misses_total Prefill runs whose first chunk landed with no retained checkpoint, or no retained prefix, matching in the tier (a re-queued request counts again).
# TYPE ignis_retained_state_misses_total counter
ignis_retained_state_misses_total{tier="device",kind="checkpoint"} 11
ignis_retained_state_misses_total{tier="device",kind="prefix"} 14
ignis_retained_state_misses_total{tier="kv_ram",kind="checkpoint"} 5
ignis_retained_state_misses_total{tier="kv_ram",kind="prefix"} 16
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
ignis_vram_reserved_bytes{line="trellis_workspace"} 0
ignis_vram_reserved_bytes{line="media_embedding"} 402653184
ignis_vram_reserved_bytes{line="sampling"} 33554432
ignis_vram_reserved_bytes{line="decode_graph"} 16777216
ignis_vram_reserved_bytes{line="verify_round"} 268435456
ignis_vram_reserved_bytes{line="drafter_round"} 134217728
ignis_vram_reserved_bytes{line="lane_state"} 1073741824
ignis_vram_reserved_bytes{line="retained_slots"} 1845491712
ignis_vram_reserved_bytes{line="hq_residual_window"} 570427392
ignis_vram_reserved_bytes{line="residual"} 268435456
ignis_vram_reserved_bytes{line="residency"} 0
ignis_vram_reserved_bytes{line="expert_cache"} 0
ignis_vram_reserved_bytes{line="condition_budget"} 0
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
# HELP ignis_condition_budget_used_bytes Bytes of the condition prefixes the image model's jobs hold now.
# TYPE ignis_condition_budget_used_bytes gauge
ignis_condition_budget_used_bytes 0
# HELP ignis_kv_ram_arena_bytes The pinned host KV-RAM arena: what it holds, and what is used of it.
# TYPE ignis_kv_ram_arena_bytes gauge
ignis_kv_ram_arena_bytes{state="capacity"} 8589934592
ignis_kv_ram_arena_bytes{state="used"} 2147483648
# HELP ignis_retained_slots Retained slots this load hands out, and how many hold an image.
# TYPE ignis_retained_slots gauge
ignis_retained_slots{state="capacity"} 10
ignis_retained_slots{state="in_use"} 7
# HELP ignis_retained_host_slots Of the retained slots, those whose images live in the pinned host block.
# TYPE ignis_retained_host_slots gauge
ignis_retained_host_slots 8
# HELP ignis_retained_host_bytes The pinned host block holding the host retained slots' images.
# TYPE ignis_retained_host_bytes gauge
ignis_retained_host_bytes 1860257792
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

// A Flash-Next load's scrape (GitHub #301, #302): the same exposition with
// the expert residency and n-gram families after it, where `render()` writes
// them. Decode: 10,000 hits and 1,000 misses over four classes in use.
export const FLASH_NEXT_EXPOSITION = `${IGNIS_EXPOSITION}# HELP ignis_expert_cache_hits_total Selected expert projections already resident in the VRAM expert cache, or staged for their layer, by K class and phase.
# TYPE ignis_expert_cache_hits_total counter
ignis_expert_cache_hits_total{class="gate_up_k2",phase="decode"} 4000
ignis_expert_cache_hits_total{class="gate_up_k2",phase="prefill"} 1200
ignis_expert_cache_hits_total{class="gate_up_k2_5",phase="decode"} 0
ignis_expert_cache_hits_total{class="gate_up_k2_5",phase="prefill"} 0
ignis_expert_cache_hits_total{class="gate_up_k3",phase="decode"} 0
ignis_expert_cache_hits_total{class="gate_up_k3",phase="prefill"} 0
ignis_expert_cache_hits_total{class="gate_up_k4",phase="decode"} 500
ignis_expert_cache_hits_total{class="gate_up_k4",phase="prefill"} 300
ignis_expert_cache_hits_total{class="down_k2",phase="decode"} 4000
ignis_expert_cache_hits_total{class="down_k2",phase="prefill"} 1200
ignis_expert_cache_hits_total{class="down_k2_5",phase="decode"} 0
ignis_expert_cache_hits_total{class="down_k2_5",phase="prefill"} 0
ignis_expert_cache_hits_total{class="down_k3",phase="decode"} 1500
ignis_expert_cache_hits_total{class="down_k3",phase="prefill"} 0
ignis_expert_cache_hits_total{class="down_k4",phase="decode"} 0
ignis_expert_cache_hits_total{class="down_k4",phase="prefill"} 0
# HELP ignis_expert_cache_misses_total Selected expert projections copied in from the pinned host pool by their own step, by K class and phase.
# TYPE ignis_expert_cache_misses_total counter
ignis_expert_cache_misses_total{class="gate_up_k2",phase="decode"} 300
ignis_expert_cache_misses_total{class="gate_up_k2",phase="prefill"} 800
ignis_expert_cache_misses_total{class="gate_up_k2_5",phase="decode"} 0
ignis_expert_cache_misses_total{class="gate_up_k2_5",phase="prefill"} 0
ignis_expert_cache_misses_total{class="gate_up_k3",phase="decode"} 0
ignis_expert_cache_misses_total{class="gate_up_k3",phase="prefill"} 0
ignis_expert_cache_misses_total{class="gate_up_k4",phase="decode"} 100
ignis_expert_cache_misses_total{class="gate_up_k4",phase="prefill"} 40
ignis_expert_cache_misses_total{class="down_k2",phase="decode"} 300
ignis_expert_cache_misses_total{class="down_k2",phase="prefill"} 800
ignis_expert_cache_misses_total{class="down_k2_5",phase="decode"} 0
ignis_expert_cache_misses_total{class="down_k2_5",phase="prefill"} 0
ignis_expert_cache_misses_total{class="down_k3",phase="decode"} 300
ignis_expert_cache_misses_total{class="down_k3",phase="prefill"} 0
ignis_expert_cache_misses_total{class="down_k4",phase="decode"} 0
ignis_expert_cache_misses_total{class="down_k4",phase="prefill"} 0
# HELP ignis_expert_prefetches_issued_total Expert projections copied ahead for the next layer by the router lookahead.
# TYPE ignis_expert_prefetches_issued_total counter
ignis_expert_prefetches_issued_total 900
# HELP ignis_expert_prefetches_used_total Prefetched expert projections at their first use, whenever it comes.
# TYPE ignis_expert_prefetches_used_total counter
ignis_expert_prefetches_used_total 720
# HELP ignis_expert_bytes_moved_total Bytes of expert projections copied host-to-device, misses and prefetches, by phase.
# TYPE ignis_expert_bytes_moved_total counter
ignis_expert_bytes_moved_total{phase="decode"} 6000000000
ignis_expert_bytes_moved_total{phase="prefill"} 9000000000
# HELP ignis_expert_residency_stall_seconds_total Device time the expert kernels waited on their step's demand copies (the misses), by phase; a step with no miss adds nothing.
# TYPE ignis_expert_residency_stall_seconds_total counter
ignis_expert_residency_stall_seconds_total{phase="decode"} 18.6
ignis_expert_residency_stall_seconds_total{phase="prefill"} 40.25
# HELP ignis_expert_cache_slots VRAM expert cache LRU slots per K class: capacity reserved at load, and in use. A pinned hot set's slots are not among them (the load's ignis.runtime.flash_next_pinned_experts event names them).
# TYPE ignis_expert_cache_slots gauge
ignis_expert_cache_slots{class="gate_up_k2",state="capacity"} 3000
ignis_expert_cache_slots{class="gate_up_k2",state="in_use"} 3000
ignis_expert_cache_slots{class="gate_up_k2_5",state="capacity"} 0
ignis_expert_cache_slots{class="gate_up_k2_5",state="in_use"} 0
ignis_expert_cache_slots{class="gate_up_k3",state="capacity"} 0
ignis_expert_cache_slots{class="gate_up_k3",state="in_use"} 0
ignis_expert_cache_slots{class="gate_up_k4",state="capacity"} 400
ignis_expert_cache_slots{class="gate_up_k4",state="in_use"} 250
ignis_expert_cache_slots{class="down_k2",state="capacity"} 3000
ignis_expert_cache_slots{class="down_k2",state="in_use"} 2999
ignis_expert_cache_slots{class="down_k2_5",state="capacity"} 0
ignis_expert_cache_slots{class="down_k2_5",state="in_use"} 0
ignis_expert_cache_slots{class="down_k3",state="capacity"} 1200
ignis_expert_cache_slots{class="down_k3",state="in_use"} 1200
ignis_expert_cache_slots{class="down_k4",state="capacity"} 0
ignis_expert_cache_slots{class="down_k4",state="in_use"} 0
# HELP ignis_ngram_rows_total N-gram embedding rows staged for prefill spans and decode rounds, by source: the RAM hot-row cache or the artifact file.
# TYPE ignis_ngram_rows_total counter
ignis_ngram_rows_total{source="hot"} 190000
ignis_ngram_rows_total{source="file"} 10000
# HELP ignis_ngram_reads_total Reads issued to the artifact file for n-gram rows the hot-row cache did not hold.
# TYPE ignis_ngram_reads_total counter
ignis_ngram_reads_total 6000
# HELP ignis_ngram_read_bytes_total Bytes of the n-gram reads issued to the artifact file.
# TYPE ignis_ngram_read_bytes_total counter
ignis_ngram_read_bytes_total 24576000
`;

// A load with the KV-disk tier (spec vram-budget/03, ADR 0017): the 27B-shaped
// exposition plus the disk families and the `tier="disk"` retained rows, which
// the server renders from the first scrape (zeros included) and only then.
export const KV_DISK_EXPOSITION = `${IGNIS_EXPOSITION.replace(
  /(ignis_retained_(?:state_)?\w+_total)\{tier="kv_ram",kind="prefix"\} (\d+)\n/g,
  (row, name) => `${row}${name}{tier="disk",kind="checkpoint"} ${name.includes("discards") ? 3 : name.includes("spills") ? 4 : 0}\n${name}{tier="disk",kind="prefix"} ${name.includes("discards") ? 1 : name.includes("spills") ? 2 : 0}\n`,
)}# HELP ignis_kv_disk_bytes The KV-disk tier: its effective budget, and the bytes the committed files hold.
# TYPE ignis_kv_disk_bytes gauge
ignis_kv_disk_bytes{state="capacity"} 107374182400
ignis_kv_disk_bytes{state="used"} 26843545600
# HELP ignis_kv_disk_spills_total Live sequences written into the disk tier, by where they came from.
# TYPE ignis_kv_disk_spills_total counter
ignis_kv_disk_spills_total{from="device"} 5
ignis_kv_disk_spills_total{from="kv_ram"} 7
# HELP ignis_kv_disk_failures_total Disk operations refused, by operation.
# TYPE ignis_kv_disk_failures_total counter
ignis_kv_disk_failures_total{op="write"} 1
ignis_kv_disk_failures_total{op="read"} 0
`;

// The image families (GitHub #324, ADR 0017), which the server renders from
// the first admitted job (zeros included) and only then. Twelve jobs ended:
// ten completed (twenty images, 2 s median), one cancelled, one failed.
const IMAGE_JOB_FAMILIES = `# HELP ignis_image_jobs_total Image jobs ended, by outcome.
# TYPE ignis_image_jobs_total counter
ignis_image_jobs_total{outcome="completed"} 10
ignis_image_jobs_total{outcome="cancelled"} 1
ignis_image_jobs_total{outcome="stopped"} 0
ignis_image_jobs_total{outcome="refused"} 0
ignis_image_jobs_total{outcome="failed"} 1
ignis_image_jobs_total{outcome="timed_out"} 0
# HELP ignis_image_images_total Images the image jobs made.
# TYPE ignis_image_images_total counter
ignis_image_images_total 21
# HELP ignis_image_job_duration_seconds Image job duration from its turn to its last image, completed jobs only.
# TYPE ignis_image_job_duration_seconds histogram
ignis_image_job_duration_seconds_bucket{le="0.1"} 0
ignis_image_job_duration_seconds_bucket{le="0.25"} 0
ignis_image_job_duration_seconds_bucket{le="0.5"} 0
ignis_image_job_duration_seconds_bucket{le="1"} 0
ignis_image_job_duration_seconds_bucket{le="2.5"} 10
ignis_image_job_duration_seconds_bucket{le="5"} 10
ignis_image_job_duration_seconds_bucket{le="10"} 10
ignis_image_job_duration_seconds_bucket{le="30"} 10
ignis_image_job_duration_seconds_bucket{le="60"} 10
ignis_image_job_duration_seconds_bucket{le="120"} 10
ignis_image_job_duration_seconds_bucket{le="300"} 10
ignis_image_job_duration_seconds_bucket{le="600"} 10
ignis_image_job_duration_seconds_bucket{le="+Inf"} 10
ignis_image_job_duration_seconds_sum 20
ignis_image_job_duration_seconds_count 10
# HELP ignis_image_job_phase_seconds_total Seconds the image jobs spent in each phase: waiting for their turn, encoding the condition, denoising, decoding.
# TYPE ignis_image_job_phase_seconds_total counter
ignis_image_job_phase_seconds_total{phase="queued"} 5
ignis_image_job_phase_seconds_total{phase="encoding"} 3
ignis_image_job_phase_seconds_total{phase="denoising"} 14
ignis_image_job_phase_seconds_total{phase="decoding"} 3
# HELP ignis_image_queue_jobs Image jobs waiting for their turn and running (at most one).
# TYPE ignis_image_queue_jobs gauge
ignis_image_queue_jobs{state="waiting"} 2
ignis_image_queue_jobs{state="running"} 1
`;

// A server that ran image jobs: the 27B-shaped exposition plus the image families.
export const IMAGE_EXPOSITION = `${IGNIS_EXPOSITION}${IMAGE_JOB_FAMILIES}`;

/**
 * The image model's load as the server renders it (GitHub #324): its plan's
 * lines in the text models' family -- 15.46 GB of weights, a 1.32 GB arena
 * (`workspace`), a 12.3 GB condition budget -- in a 29.1 GB budget, every
 * other line, the KV pool, the KV-RAM arena and the retained slots at zero,
 * and a running job's 1.12 GB condition prefix; then its job families.
 */
const IMAGE_LOAD: Record<string, number> = {
  'ignis_vram_reserved_bytes{line="weights"}': 15_460_000_000,
  'ignis_vram_reserved_bytes{line="workspace"}': 1_320_000_000,
  'ignis_vram_reserved_bytes{line="condition_budget"}': 12_300_000_000,
  ignis_vram_budget_bytes: 29_100_000_000,
  ignis_kv_pool_pages: 0,
  ignis_kv_page_bytes: 0,
  ignis_kv_pool_used_pages: 0,
  ignis_condition_budget_used_bytes: 1_120_000_000,
  'ignis_kv_ram_arena_bytes{state="capacity"}': 0,
  'ignis_kv_ram_arena_bytes{state="used"}': 0,
  'ignis_retained_slots{state="capacity"}': 0,
  'ignis_retained_slots{state="in_use"}': 0,
  ignis_retained_host_slots: 0,
  ignis_retained_host_bytes: 0,
};
export const IMAGE_LOAD_EXPOSITION = `${IGNIS_EXPOSITION.split("\n")
  .map((line) => {
    const series = line.slice(0, line.lastIndexOf(" "));
    if (series in IMAGE_LOAD) return `${series} ${IMAGE_LOAD[series]}`;
    return series.startsWith("ignis_vram_reserved_bytes{") ? `${series} 0` : line;
  })
  .join("\n")}${IMAGE_JOB_FAMILIES}`;

// The loop guard's families as the server renders them once it has watched a
// request (GitHub #330): every series, zeros included. A server with the guard
// off renders none of them, so this is the 27B-shaped exposition plus the
// family, and the thinking budget's own close counter beside it.
export const LOOP_GUARD_FAMILIES = `# HELP ignis_loop_guard_watched_requests_total Requests the loop guard watched.
# TYPE ignis_loop_guard_watched_requests_total counter
ignis_loop_guard_watched_requests_total 40
# HELP ignis_loop_guard_looped_requests_total Watched requests with at least one loop.
# TYPE ignis_loop_guard_looped_requests_total counter
ignis_loop_guard_looped_requests_total 6
# HELP ignis_loop_guard_loops_total Loops the guard saw, by what it did about them.
# TYPE ignis_loop_guard_loops_total counter
ignis_loop_guard_loops_total{outcome="rewound"} 7
ignis_loop_guard_loops_total{outcome="closed"} 1
ignis_loop_guard_loops_total{outcome="detected"} 0
ignis_loop_guard_loops_total{outcome="too_little_kept"} 2
ignis_loop_guard_loops_total{outcome="no_room"} 0
ignis_loop_guard_loops_total{outcome="no_tokenizer"} 0
ignis_loop_guard_loops_total{outcome="submit_refused"} 0
# HELP ignis_loop_guard_rewound_tokens_total Generated tokens the guard's rewinds took back.
# TYPE ignis_loop_guard_rewound_tokens_total counter
ignis_loop_guard_rewound_tokens_total 5120
# HELP ignis_loop_guard_min_ratio Lowest loop score each watched request reached.
# TYPE ignis_loop_guard_min_ratio histogram
ignis_loop_guard_min_ratio_bucket{le="0.1"} 0
ignis_loop_guard_min_ratio_bucket{le="0.15"} 2
ignis_loop_guard_min_ratio_bucket{le="0.2"} 4
ignis_loop_guard_min_ratio_bucket{le="0.25"} 6
ignis_loop_guard_min_ratio_bucket{le="0.3"} 9
ignis_loop_guard_min_ratio_bucket{le="0.35"} 15
ignis_loop_guard_min_ratio_bucket{le="0.4"} 24
ignis_loop_guard_min_ratio_bucket{le="0.45"} 33
ignis_loop_guard_min_ratio_bucket{le="0.5"} 38
ignis_loop_guard_min_ratio_bucket{le="1"} 38
ignis_loop_guard_min_ratio_bucket{le="+Inf"} 38
ignis_loop_guard_min_ratio_sum 13.42
ignis_loop_guard_min_ratio_count 38
`;

export const THINKING_FORCED_CLOSES = `# HELP ignis_thinking_forced_closes_total Reasonings the thinking budget closed.
# TYPE ignis_thinking_forced_closes_total counter
ignis_thinking_forced_closes_total 3
`;

export const LOOP_GUARD_EXPOSITION = `${IGNIS_EXPOSITION}${LOOP_GUARD_FAMILIES}${THINKING_FORCED_CLOSES}`;

// The speech families (spec speech/02, ADR 0017's amendment of 2026-10-24),
// which the server renders from the first transcription (zeros included) and
// only then. Five transcriptions on the CPU ended: four completed (80 s of
// audio, a median wait of 2.5 s or so), one failed; one waits behind the one
// running.
const SPEECH_FAMILIES = `# HELP ignis_speech_transcriptions_total Transcriptions ended, by executor and outcome.
# TYPE ignis_speech_transcriptions_total counter
ignis_speech_transcriptions_total{executor="cpu",outcome="completed"} 4
ignis_speech_transcriptions_total{executor="cpu",outcome="cancelled"} 0
ignis_speech_transcriptions_total{executor="cpu",outcome="failed"} 1
ignis_speech_transcriptions_total{executor="cpu",outcome="timed_out"} 0
ignis_speech_transcriptions_total{executor="gpu_loan",outcome="completed"} 0
ignis_speech_transcriptions_total{executor="gpu_loan",outcome="cancelled"} 0
ignis_speech_transcriptions_total{executor="gpu_loan",outcome="failed"} 0
ignis_speech_transcriptions_total{executor="gpu_loan",outcome="timed_out"} 0
# HELP ignis_speech_audio_seconds_total Seconds of audio the completed transcriptions covered.
# TYPE ignis_speech_audio_seconds_total counter
ignis_speech_audio_seconds_total 80
# HELP ignis_speech_commit_wait_seconds Wait from a transcription's commit (a batch request's last byte received) to its transcript, completed ones only.
# TYPE ignis_speech_commit_wait_seconds histogram
ignis_speech_commit_wait_seconds_bucket{le="0.1"} 0
ignis_speech_commit_wait_seconds_bucket{le="0.25"} 0
ignis_speech_commit_wait_seconds_bucket{le="0.5"} 0
ignis_speech_commit_wait_seconds_bucket{le="1"} 0
ignis_speech_commit_wait_seconds_bucket{le="2.5"} 2
ignis_speech_commit_wait_seconds_bucket{le="5"} 4
ignis_speech_commit_wait_seconds_bucket{le="10"} 4
ignis_speech_commit_wait_seconds_bucket{le="30"} 4
ignis_speech_commit_wait_seconds_bucket{le="60"} 4
ignis_speech_commit_wait_seconds_bucket{le="120"} 4
ignis_speech_commit_wait_seconds_bucket{le="300"} 4
ignis_speech_commit_wait_seconds_bucket{le="600"} 4
ignis_speech_commit_wait_seconds_bucket{le="+Inf"} 4
ignis_speech_commit_wait_seconds_sum 11
ignis_speech_commit_wait_seconds_count 4
# HELP ignis_speech_sessions_open Realtime transcription sessions open.
# TYPE ignis_speech_sessions_open gauge
ignis_speech_sessions_open 0
# HELP ignis_speech_queue_length Batch transcriptions waiting behind the one running.
# TYPE ignis_speech_queue_length gauge
ignis_speech_queue_length 1
`;

// A server that transcribed: the 27B-shaped exposition plus the speech families.
export const SPEECH_EXPOSITION = `${IGNIS_EXPOSITION}${SPEECH_FAMILIES}`;
