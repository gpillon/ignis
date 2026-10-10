//! A Flash-Next artifact served (spec flash-next/04, GitHub #302): the
//! scheduler the server builds for it (`runtime::flash_next_scheduler`, the
//! leaf `ignis_runtime::FlashNextLeaf`) generates for three requests at once
//! and finishes each -- prefill chunks, decode rounds over three lanes and
//! the n-gram rows staged on the host for every step -- and, with its
//! default prompt reuse (spec flash-next/05), a second turn resumes from the
//! first's checkpoint and says so on the events the request log reads.
//!
//! Machine-local: `F:/ai/models/Qwen3.8-Flash-Next-ignis/` (or
//! `IGNIS_FLASH_NEXT_DIR`). Explicit GPU profile (ADR 0006): a missing
//! artifact or GPU is a skip outside it, a failure under it. Pins the ~38 GB
//! expert pool.

#![cfg(feature = "cuda")]

use std::path::PathBuf;

use ignis_artifact::packer::ARTIFACT_FILE_NAME;
use ignis_core::checkpoint::ReuseSource;
use ignis_core::types::{DecodeParams, FinishReason, RequestClass, RequestInput, SchedEvent};
use ignis_core::{gpu_profile, ConcreteScheduler, Scheduler};
use ignis_server::runtime::{EngineShape, flash_next_scheduler};

const MODEL_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
const MODEL: &str = "qwen3.8-flash-next";
/// Flash-Next's EOS (generation_config.json).
const EOS: u32 = 248_044;

fn input(prompt: Vec<u32>) -> RequestInput {
    RequestInput {
        decision: None,
        model: MODEL.into(),
        tokens: prompt,
        params: DecodeParams {
            max_tokens: Some(16),
            ..DecodeParams::default()
        },
        multimodal: None,
        opener_tokens: None,
        user_turn_tokens: None,
        system_block_tokens: None,
        reuse_boundaries: Vec::new(),
        constrained: None,
        forced_literal: None,
        warm_up: false,
    }
}

#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact"]
fn three_requests_generate_and_finish_on_three_lanes() {
    let dir = std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(MODEL_DIR), PathBuf::from);
    let path = dir.join(ARTIFACT_FILE_NAME);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("no Flash-Next artifact at {}", path.display())) {
        return;
    }
    let shape = EngineShape { max_context: 8192, prefill_chunk: 2048, ..EngineShape::default() };
    let (mut sched, reserved) = match flash_next_scheduler(&path, MODEL.into(), EOS, shape, None) {
        Ok(loaded) => loaded,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("load the Flash-Next scheduler: {e}"));
            return;
        }
    };
    println!("plan: {:?}", reserved.lines);
    let requests: Vec<_> = (0..3u32)
        .map(|r| {
            let prompt: Vec<u32> = (0..200 + 50 * r).map(|i| 1000 + (i * 7919 + r * 104_729) % 60_000).collect();
            sched.submit(input(prompt), RequestClass::Interactive).expect("submit")
        })
        .collect();
    let mut tokens = vec![0usize; requests.len()];
    let mut finished = vec![None; requests.len()];
    while !sched.is_idle() {
        for event in sched.advance() {
            match event {
                SchedEvent::Token { request, .. } => {
                    if let Some(i) = requests.iter().position(|&r| r == request) {
                        tokens[i] += 1;
                    }
                }
                SchedEvent::Done { request, reason, .. } => {
                    if let Some(i) = requests.iter().position(|&r| r == request) {
                        finished[i] = Some(reason);
                    }
                }
                _ => {}
            }
        }
    }
    for (i, reason) in finished.iter().enumerate() {
        println!("request {i}: {} tokens, {reason:?}", tokens[i]);
        assert!(tokens[i] > 0, "request {i} generated nothing");
        assert!(
            matches!(reason, Some(FinishReason::Length) | Some(FinishReason::Stop)),
            "request {i} finished as {reason:?}"
        );
    }
}

/// Every log record `f` emits on this thread, as JSON, beside its result.
fn captured<T>(f: impl FnOnce() -> T) -> (Vec<serde_json::Value>, T) {
    use tracing_subscriber::layer::SubscriberExt;
    let sink = std::sync::Arc::new(ignis_logging::MemorySink::new());
    let subscriber = tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone()));
    let out = tracing::subscriber::with_default(subscriber, f);
    (sink.lines().iter().map(|line| serde_json::from_str(line).expect("json")).collect(), out)
}

/// The attributes of the one record named `name`.
fn attributes(records: &[serde_json::Value], name: &str) -> serde_json::Value {
    let found: Vec<_> = records.iter().filter(|r| r["event_name"] == name).collect();
    assert_eq!(found.len(), 1, "one `{name}` in {records:?}");
    found[0]["attributes"].clone()
}

/// ADR 0045 on the card (spec vram-budget/03 ACs 2, 6, 7): at the make
/// default of its day -- 262,144 tokens, three lanes, a 4 GiB headroom (1 GiB
/// since 2026-10-09; the test keeps 4 GiB, its plan is relative) -- the pool is
/// offloaded at 524,288 tokens, the leaf builds those pages, and the plan
/// event says so; an explicit budget that leaves the expert cache below its
/// floor is refused naming the opt-in, and with it starts and warns. The
/// KV-RAM arena and the host retained slots are off, so the load fits a host
/// with ~44 GB available; neither moves VRAM.
#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact"]
fn the_default_plan_is_offloaded_at_524288_tokens_and_the_floor_opt_in_warns() {
    use ignis_core::residency::{available_physical_bytes, EXPERT_CACHE_FLOOR_BYTES};
    let dir = std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(MODEL_DIR), PathBuf::from);
    let path = dir.join(ARTIFACT_FILE_NAME);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("no Flash-Next artifact at {}", path.display())) {
        return;
    }
    let shape = EngineShape {
        max_context: 262_144,
        prefill_chunk: 8192,
        decode_lanes: 3,
        vram: ignis_core::VramMode::Derived { headroom_bytes: 4 << 30 },
        host_pool_bytes: 0,
        retained_host_slots: 0,
        retained_host_named: true,
        ..EngineShape::default()
    };
    let available_before = available_physical_bytes().unwrap_or(0);
    let (records, loaded) = captured(|| flash_next_scheduler(&path, MODEL.into(), EOS, shape, None));
    let (sched, reserved) = match loaded {
        Ok(loaded) => loaded,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("load the Flash-Next scheduler: {e}"));
            return;
        }
    };
    let plan = attributes(&records, "ignis.runtime.flash_next_plan");
    println!("plan: {plan}");
    assert_eq!(plan["kv_pool_policy"], "offloaded", "{plan}");
    assert_eq!(plan["kv_pool_pages"], 8_192, "{plan}");
    assert_eq!(plan["kv_pool_tokens"], 524_288, "{plan}");
    assert_eq!(reserved.kv_pool_pages, 8_192, "the leaf built the plan's pages");
    assert_eq!(reserved.expert_cache_bytes, plan["expert_cache_bytes"].as_u64().unwrap());
    assert!(records.iter().all(|r| r["event_name"] != "ignis.runtime.expert_cache_below_floor"));
    let budget = plan["budget_bytes"].as_u64().unwrap();
    let cache = plan["expert_cache_bytes"].as_u64().unwrap();
    assert!(cache > EXPERT_CACHE_FLOOR_BYTES + (2 << 30), "the default plan's cache clears the floor: {plan}");
    drop(sched);

    // A budget 2 GiB short of what the floor needs: refused by name, before
    // anything is pinned, unless the operator opts in.
    let below = ignis_core::VramMode::Explicit {
        budget_bytes: budget - (cache - EXPERT_CACHE_FLOOR_BYTES) - (2 << 30),
        allow_oversubscription: false,
    };
    let refused = flash_next_scheduler(&path, MODEL.into(), EOS, EngineShape { vram: below, ..shape }, None)
        .err()
        .expect("below the floor without the opt-in");
    assert!(refused.contains("--vram-allow-expert-cache-below-floor"), "{refused}");
    // The first load's pinned pool frees asynchronously.
    for _ in 0..60 {
        if available_physical_bytes().unwrap_or(0) + (1 << 30) >= available_before {
            break;
        }
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
    let opted = EngineShape { vram: below, allow_expert_cache_below_floor: true, ..shape };
    let (records, loaded) = captured(|| flash_next_scheduler(&path, MODEL.into(), EOS, opted, None));
    let (_sched, reserved) = loaded.expect("the opt-in starts below the floor");
    let warning = attributes(&records, "ignis.runtime.expert_cache_below_floor");
    println!("warning: {warning}");
    let below_floor = warning["expert_cache_bytes"].as_u64().unwrap();
    assert!(below_floor < EXPERT_CACHE_FLOOR_BYTES, "{warning}");
    assert_eq!(below_floor, reserved.expert_cache_bytes);
    assert_eq!(warning["floor_bytes"], EXPERT_CACHE_FLOOR_BYTES);
    assert!(warning["knobs"].as_str().unwrap().contains("--vram-kv-pool-bytes"), "{warning}");
}

fn load(shape: EngineShape) -> Option<ConcreteScheduler> {
    let dir = std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(MODEL_DIR), PathBuf::from);
    let path = dir.join(ARTIFACT_FILE_NAME);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("no Flash-Next artifact at {}", path.display())) {
        return None;
    }
    match flash_next_scheduler(&path, MODEL.into(), EOS, shape, None) {
        Ok((sched, reserved)) => {
            println!("plan: {:?}; host slots {}, arena {} bytes", reserved.lines, reserved.retained_host_slots, reserved.kv_ram_arena_bytes);
            // The load reads the program's stats for the pool's pages: on a
            // Flash-Next model, which has no 27B decode-graph staging, that
            // read once dereferenced the 27B's null buffers.
            assert!(reserved.kv_pool_pages > 0, "the program's stats name the pool's pages");
            Some(sched)
        }
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("load the Flash-Next scheduler: {e}"));
            None
        }
    }
}

/// Run `input` to its end: the tokens it generated and every event it
/// produced.
fn run(sched: &mut ConcreteScheduler, input: RequestInput) -> (Vec<u32>, Vec<SchedEvent>) {
    let request = sched.submit(input, RequestClass::Agent).expect("submit");
    let (mut tokens, mut events) = (Vec::new(), Vec::new());
    while !sched.is_idle() {
        for event in sched.advance() {
            if let SchedEvent::Token { request: r, token, .. } = &event {
                if *r == request {
                    tokens.push(*token);
                }
            }
            events.push(event);
        }
    }
    (tokens, events)
}

#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact"]
fn a_second_turn_resumes_from_the_first_turns_checkpoint() {
    // The load's defaults: prompt reuse on, 8 host retained slots, 0 on the
    // device, the 2 GiB arena (spec flash-next/05).
    let shape = EngineShape { max_context: 8192, prefill_chunk: 2048, ..EngineShape::default() };
    let Some(mut sched) = load(shape) else { return };

    // Turn 1: a system block of four pages, a history past one chunk, the
    // opener at its end and the `<think>\n` a chat template places after the
    // opener (ADR 0029): an opener at the very end of a prompt captures
    // nothing, since the chunk that would capture is the one that samples.
    let history: Vec<u32> = (0..2500u32).map(|i| 1000 + (i * 7919 + 31 * 104_729) % 60_000).collect();
    let think = [3001u32, 3002, 3003];
    let first = RequestInput {
        opener_tokens: Some(history.len() as u32),
        system_block_tokens: Some(256),
        ..input([history.as_slice(), &think].concat())
    };
    let (generated, events) = run(&mut sched, first);
    assert!(!generated.is_empty(), "turn 1 generated nothing");
    assert!(
        !events.iter().any(|e| matches!(e, SchedEvent::StateReused { .. })),
        "turn 1 had nothing to resume from"
    );

    // Turn 2 re-sends the history, turn 1's answer and a tool result.
    let mut prompt = history.clone();
    prompt.extend(&generated);
    prompt.extend((0..300u32).map(|i| 2000 + i * 37 % 50_000));
    let opener = prompt.len() as u32;
    prompt.extend(think);
    let second = RequestInput { opener_tokens: Some(opener), ..input(prompt) };
    let (generated, events) = run(&mut sched, second);
    assert!(!generated.is_empty(), "turn 2 generated nothing");
    let resumed: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            SchedEvent::StateReused { source, tokens, restore_micros, .. } => Some((*source, *tokens, *restore_micros)),
            _ => None,
        })
        .collect();
    println!("turn 2 resumed: {resumed:?}");
    assert_eq!(resumed.len(), 1, "one resume per request: {resumed:?}");
    let (source, tokens, _) = resumed[0];
    assert_eq!(source, ReuseSource::Device, "the checkpoint was still on the device (host slot, device pages)");
    assert_eq!(tokens, history.len() as u32, "everything up to turn 1's opener was skipped");
    let retained: Vec<_> = events.iter().filter(|e| matches!(e, SchedEvent::RetainedState { .. })).collect();
    println!("turn 2 retained-state operations: {retained:?}");
    assert!(!retained.is_empty(), "the per-tier counters see the hit");
}
