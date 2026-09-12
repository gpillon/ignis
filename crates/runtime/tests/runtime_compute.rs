use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use ignis_core::{
    Compute, ComputeError, ConcreteScheduler, DecodeJob, DecodeOutcome, DecodeParams, FinishReason,
    N_DECODE_LANES, PrefillJob, RequestClass, RequestInput, Scheduler, SchedulerConfig,
};
use ignis_runtime::{Model, RuntimeCompute, RuntimeStats, StepLeaf};

#[derive(Default)]
struct Calls {
    models_released: u32,
    stats_calls: u32,
    sequences_allocated: Vec<u32>,
    sequences_released: u32,
    prefill_positions: Vec<u32>,
    prefill_params: Vec<DecodeParams>,
    decode_batch_sizes: Vec<usize>,
    decode_params: Vec<Vec<DecodeParams>>,
    /// P4-10 (GitHub #126): the prefixes published (their token counts) and
    /// the claims served (the context each claimant reserved), so a test can
    /// see that a claimant was allocated *against* a prefix rather than
    /// allocated normally and prefilled from a hole.
    prefixes_published: Vec<u32>,
    shared_allocations: Vec<u32>,
    prefixes_released: u32,
}

struct StubLeaf {
    calls: Mutex<Calls>,
    tokens: Mutex<VecDeque<u32>>,
    allocation_error: Option<i32>,
    prefill_error: Option<i32>,
    prefill_error_on_call: Option<(usize, i32)>,
    decode_error: Option<i32>,
}

impl StubLeaf {
    fn with_tokens(tokens: impl IntoIterator<Item = u32>) -> Self {
        Self {
            calls: Mutex::new(Calls::default()),
            tokens: Mutex::new(tokens.into_iter().collect()),
            allocation_error: None,
            prefill_error: None,
            prefill_error_on_call: None,
            decode_error: None,
        }
    }

    fn failing_prefill(code: i32) -> Self {
        Self {
            calls: Mutex::new(Calls::default()),
            tokens: Mutex::new(VecDeque::new()),
            allocation_error: None,
            prefill_error: Some(code),
            prefill_error_on_call: None,
            decode_error: None,
        }
    }

    fn failing_allocation(code: i32) -> Self {
        Self {
            calls: Mutex::new(Calls::default()),
            tokens: Mutex::new(VecDeque::new()),
            allocation_error: Some(code),
            prefill_error: None,
            prefill_error_on_call: None,
            decode_error: None,
        }
    }

    fn failing_second_prefill(code: i32) -> Self {
        Self {
            calls: Mutex::new(Calls::default()),
            tokens: Mutex::new(VecDeque::new()),
            allocation_error: None,
            prefill_error: None,
            prefill_error_on_call: Some((2, code)),
            decode_error: None,
        }
    }

    fn failing_decode(code: i32) -> Self {
        Self {
            calls: Mutex::new(Calls::default()),
            tokens: Mutex::new(VecDeque::new()),
            allocation_error: None,
            prefill_error: None,
            prefill_error_on_call: None,
            decode_error: Some(code),
        }
    }
}

impl StepLeaf for StubLeaf {
    type Model = ();
    type Sequence = ();
    /// The tokens the prefix covers — enough for a stub to prove a claimant
    /// was handed the right entry.
    type Prefix = u32;

    fn load_model(&self) -> Result<Self::Model, i32> {
        Ok(())
    }

    fn release_model(&self, _model: Self::Model) {
        self.calls.lock().unwrap().models_released += 1;
    }

    fn stats(&self, _model: &Self::Model) -> Result<RuntimeStats, i32> {
        self.calls.lock().unwrap().stats_calls += 1;
        Ok(RuntimeStats {
            vram_bytes: 4096,
            kv_page_tokens: 64,
            kv_page_bytes: 8192,
            kv_page_count: 128,
            last_step_micros: 13,
            kernel_count: 7,
            graph_launches: 0,
        })
    }

    fn allocate_sequence(
        &self,
        _model: &Self::Model,
        context_tokens: u32,
    ) -> Result<Self::Sequence, i32> {
        if let Some(code) = self.allocation_error {
            return Err(code);
        }
        self.calls
            .lock()
            .unwrap()
            .sequences_allocated
            .push(context_tokens);
        Ok(())
    }

    fn release_sequence(&self, _model: &Self::Model, _sequence: Self::Sequence) {
        self.calls.lock().unwrap().sequences_released += 1;
    }

    fn allocate_sequence_shared(
        &self,
        _model: &Self::Model,
        context_tokens: u32,
        _prefix: &Self::Prefix,
    ) -> Result<Self::Sequence, i32> {
        if let Some(code) = self.allocation_error {
            return Err(code);
        }
        let mut calls = self.calls.lock().unwrap();
        calls.sequences_allocated.push(context_tokens);
        calls.shared_allocations.push(context_tokens);
        Ok(())
    }

    fn publish_prefix(
        &self,
        _model: &Self::Model,
        _sequence: &mut Self::Sequence,
        prefix_tokens: u32,
    ) -> Result<Self::Prefix, i32> {
        self.calls.lock().unwrap().prefixes_published.push(prefix_tokens);
        Ok(prefix_tokens)
    }

    fn release_prefix(&self, _model: &Self::Model, _prefix: Self::Prefix) {
        self.calls.lock().unwrap().prefixes_released += 1;
    }

    fn prefill(
        &self,
        _model: &Self::Model,
        _sequence: &mut Self::Sequence,
        _tokens: &[u32],
        start_position: u32,
        params: DecodeParams,
    ) -> Result<(), i32> {
        let call = {
            let mut calls = self.calls.lock().unwrap();
            calls.prefill_positions.push(start_position);
            calls.prefill_params.push(params);
            calls.prefill_positions.len()
        };
        self.prefill_error
            .or_else(|| {
                self.prefill_error_on_call
                    .filter(|(expected, _)| call == *expected)
                    .map(|(_, code)| code)
            })
            .map_or(Ok(()), Err)
    }

    fn decode(
        &self,
        _model: &Self::Model,
        sequences: &mut [&mut Self::Sequence],
        params: &[DecodeParams],
    ) -> Result<Vec<u32>, i32> {
        if let Some(code) = self.decode_error {
            return Err(code);
        }
        let mut calls = self.calls.lock().unwrap();
        calls.decode_batch_sizes.push(sequences.len());
        calls.decode_params.push(params.to_vec());
        drop(calls);
        let mut tokens = self.tokens.lock().unwrap();
        Ok(sequences
            .iter()
            .map(|_| tokens.pop_front().unwrap_or(7))
            .collect())
    }
}

#[test]
fn runtime_threads_each_requests_sampling_params_to_the_leaf_batch() {
    let leaf = Arc::new(StubLeaf::with_tokens([7, 8]));
    let model = Arc::new(Model::load(leaf.clone()).unwrap());
    let compute = RuntimeCompute::new(model, 99);
    let left = DecodeParams {
        max_tokens: Some(3),
        temperature: 0.5,
        top_p: 0.7,
        top_k: 7,
        presence_penalty: 0.4,
        frequency_penalty: -0.4,
        seed: 9,
        ignore_eos: false,
    };
    let right = DecodeParams {
        max_tokens: Some(4),
        temperature: 1.2,
        top_p: 0.8,
        top_k: 12,
        presence_penalty: -0.3,
        frequency_penalty: 0.6,
        seed: 11,
        ignore_eos: false,
    };
    compute
        .prefill_step(&[
            PrefillJob {
                request: 1,
                tokens: vec![4, 5],
                context_tokens: 9,
                start_position: 0,
                params: left,
                shared_prefix: None,
                publish_prefix_tokens: None,
            },
            PrefillJob {
                request: 2,
                tokens: vec![4, 5],
                context_tokens: 9,
                start_position: 0,
                params: right,
                shared_prefix: None,
                publish_prefix_tokens: None,
            },
        ])
        .unwrap();

    compute
        .decode_step(&[
            DecodeJob {
                request: 1,
                lane: 0,
                params: left,
            },
            DecodeJob {
                request: 2,
                lane: 1,
                params: right,
            },
        ])
        .unwrap();

    let calls = leaf.calls.lock().unwrap();
    assert_eq!(calls.prefill_params, vec![left, right]);
    assert_eq!(calls.decode_params, vec![vec![left, right]]);
}

#[test]
fn adapter_rejects_decode_batches_larger_than_the_resident_lane_bound() {
    let leaf = Arc::new(StubLeaf::with_tokens([]));
    let model = Arc::new(Model::load(leaf).unwrap());
    let compute = RuntimeCompute::new(model, 99);
    let jobs: Vec<_> = (0..=N_DECODE_LANES)
        .map(|request| DecodeJob {
            request: request as u64,
            lane: request,
            params: DecodeParams::default(),
        })
        .collect();

    assert_eq!(compute.decode_step(&jobs), Err(ComputeError::Kernel(-1)));
    assert_eq!(compute.live_sequences(), 0);
}

fn prefill(request: u64, max_tokens: Option<u32>) -> PrefillJob {
    PrefillJob {
        request,
        tokens: vec![4, 5],
        context_tokens: 9,
        start_position: 0,
        params: DecodeParams {
            max_tokens,
            ..DecodeParams::default()
        },
        shared_prefix: None,
        publish_prefix_tokens: None,
    }
}

#[test]
fn model_and_sequence_handles_release_at_their_ownership_boundaries() {
    let leaf = Arc::new(StubLeaf::with_tokens([7]));
    let model = Arc::new(Model::load(leaf.clone()).expect("stub model loads"));
    let compute = RuntimeCompute::new(model.clone(), 99);

    compute
        .prefill_step(&[prefill(1, None)])
        .expect("prefill succeeds");
    assert_eq!(compute.live_sequences(), 1);
    assert_eq!(leaf.calls.lock().unwrap().sequences_allocated, vec![9]);
    assert_eq!(model.stats().unwrap().kv_page_tokens, 64);
    assert_eq!(leaf.calls.lock().unwrap().stats_calls, 1);

    compute.release(1);
    assert_eq!(compute.live_sequences(), 0);
    assert_eq!(leaf.calls.lock().unwrap().sequences_released, 1);

    drop(compute);
    drop(model);
    assert_eq!(leaf.calls.lock().unwrap().models_released, 1);
}

#[test]
fn dropping_the_adapter_releases_an_unfinished_sequence() {
    let leaf = Arc::new(StubLeaf::with_tokens([7]));
    let model = Arc::new(Model::load(leaf.clone()).unwrap());
    let compute = RuntimeCompute::new(model, 99);
    compute.prefill_step(&[prefill(1, None)]).unwrap();

    drop(compute);

    assert_eq!(leaf.calls.lock().unwrap().sequences_released, 1);
}

#[test]
fn adapter_maps_leaf_errors_and_releases_the_failed_prefill_sequence() {
    let leaf = Arc::new(StubLeaf::failing_prefill(-17));
    let model = Arc::new(Model::load(leaf.clone()).expect("stub model loads"));
    let compute = RuntimeCompute::new(model, 99);

    assert_eq!(
        compute.prefill_step(&[prefill(1, None)]),
        Err(ComputeError::Kernel(-17))
    );
    let calls = leaf.calls.lock().unwrap();
    assert_eq!(calls.sequences_allocated, vec![9]);
    assert_eq!(calls.sequences_released, 1);
}

#[test]
fn adapter_maps_allocation_and_decode_errors_without_losing_live_state() {
    let allocation_leaf = Arc::new(StubLeaf::failing_allocation(-11));
    let allocation_model = Arc::new(Model::load(allocation_leaf.clone()).unwrap());
    let allocation_compute = RuntimeCompute::new(allocation_model, 99);
    assert_eq!(
        allocation_compute.prefill_step(&[prefill(1, None)]),
        Err(ComputeError::Kernel(-11))
    );
    assert_eq!(allocation_compute.live_sequences(), 0);

    let decode_leaf = Arc::new(StubLeaf::failing_decode(-19));
    let decode_model = Arc::new(Model::load(decode_leaf.clone()).unwrap());
    let decode_compute = RuntimeCompute::new(decode_model, 99);
    decode_compute.prefill_step(&[prefill(1, None)]).unwrap();
    assert_eq!(
        decode_compute.decode_step(&[DecodeJob {
            request: 1,
            lane: 0,
            params: DecodeParams::default(),
        }]),
        Err(ComputeError::Kernel(-19))
    );
    assert_eq!(decode_compute.live_sequences(), 1);
}

#[test]
fn a_partial_prefill_failure_releases_the_entire_retry_batch() {
    let leaf = Arc::new(StubLeaf::failing_second_prefill(-18));
    let model = Arc::new(Model::load(leaf.clone()).unwrap());
    let compute = RuntimeCompute::new(model, 99);

    assert_eq!(
        compute.prefill_step(&[prefill(1, None), prefill(2, None)]),
        Err(ComputeError::Kernel(-18))
    );
    assert_eq!(compute.live_sequences(), 0);
    assert_eq!(leaf.calls.lock().unwrap().sequences_released, 2);
}

#[test]
fn adapter_enforces_max_tokens_and_eos() {
    let leaf = Arc::new(StubLeaf::with_tokens([7, 99]));
    let model = Arc::new(Model::load(leaf.clone()).expect("stub model loads"));
    let compute = RuntimeCompute::new(model, 99);

    compute.prefill_step(&[prefill(1, Some(1))]).unwrap();
    let job = DecodeJob {
        request: 1,
        lane: 0,
        params: DecodeParams {
            max_tokens: Some(1),
            ..DecodeParams::default()
        },
    };
    assert_eq!(
        compute.decode_step(&[job.clone()]).unwrap(),
        vec![DecodeOutcome::Token(7)]
    );
    // The cap (`max_tokens: Some(1)`) was already reached by the first
    // token: `Length`, not `Stop` (the leaf never even runs — see
    // `decode_batch_sizes` below).
    assert_eq!(
        compute.decode_step(&[job]).unwrap(),
        vec![DecodeOutcome::Finished(FinishReason::Length)]
    );

    // Request 2 has no `max_tokens`: the only way it stops is the leaf's
    // next token (99) matching this adapter's configured EOS (99) — `Stop`.
    compute.prefill_step(&[prefill(2, None)]).unwrap();
    assert_eq!(
        compute
            .decode_step(&[DecodeJob {
                request: 2,
                lane: 1,
                params: DecodeParams::default(),
            }])
            .unwrap(),
        vec![DecodeOutcome::Finished(FinishReason::Stop)]
    );
    assert_eq!(leaf.calls.lock().unwrap().sequences_released, 2);
}

#[test]
fn adapter_can_keep_a_measurement_lane_alive_past_eos() {
    let leaf = Arc::new(StubLeaf::with_tokens([99, 7]));
    let model = Arc::new(Model::load(leaf).expect("stub model loads"));
    let compute = RuntimeCompute::new(model, 99);
    compute.prefill_step(&[prefill(1, None)]).unwrap();
    let job = DecodeJob {
        request: 1,
        lane: 0,
        params: DecodeParams {
            ignore_eos: true,
            ..DecodeParams::default()
        },
    };

    assert_eq!(
        compute.decode_step(std::slice::from_ref(&job)).unwrap(),
        vec![DecodeOutcome::Token(99)]
    );
    assert_eq!(compute.decode_step(&[job]).unwrap(), vec![DecodeOutcome::Token(7)]);
    assert_eq!(compute.live_sequences(), 1);
}

#[test]
fn adapter_decodes_multiple_requests_in_one_ordered_leaf_round() {
    let leaf = Arc::new(StubLeaf::with_tokens([17, 23]));
    let model = Arc::new(Model::load(leaf.clone()).unwrap());
    let compute = RuntimeCompute::new(model, 99);
    compute
        .prefill_step(&[prefill(1, None), prefill(2, None)])
        .unwrap();

    assert_eq!(
        compute
            .decode_step(&[
                DecodeJob {
                    request: 1,
                    lane: 0,
                    params: DecodeParams::default(),
                },
                DecodeJob {
                    request: 2,
                    lane: 1,
                    params: DecodeParams::default(),
                },
            ])
            .unwrap(),
        vec![DecodeOutcome::Token(17), DecodeOutcome::Token(23)]
    );
    assert_eq!(leaf.calls.lock().unwrap().decode_batch_sizes, vec![2]);
    assert_eq!(compute.live_sequences(), 2);
}

#[test]
fn scheduler_releases_the_adapter_sequence_when_a_request_completes() {
    let leaf = Arc::new(StubLeaf::with_tokens([7]));
    let model = Arc::new(Model::load(leaf.clone()).unwrap());
    let compute = Arc::new(RuntimeCompute::new(model, 99));
    let mut scheduler = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: "stub".into(),
            ..SchedulerConfig::default()
        },
        compute,
    );
    scheduler
        .submit(
            RequestInput {
                model: "stub".into(),
                tokens: vec![1, 2],
                params: DecodeParams {
                    max_tokens: Some(1),
                    ..DecodeParams::default()
                },
            },
            RequestClass::Interactive,
        )
        .unwrap();

    scheduler.advance();
    assert_eq!(leaf.calls.lock().unwrap().sequences_released, 1);
}

#[test]
fn scheduler_passes_the_full_sequence_reservation_to_first_prefill() {
    let leaf = Arc::new(StubLeaf::with_tokens([7]));
    let model = Arc::new(Model::load(leaf.clone()).unwrap());
    let compute = Arc::new(RuntimeCompute::new(model, 99));
    let mut scheduler = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: "stub".into(),
            max_sequence_tokens: 20,
            ..SchedulerConfig::default()
        },
        compute,
    );
    scheduler
        .submit(
            RequestInput {
                model: "stub".into(),
                tokens: vec![1, 2, 3],
                params: DecodeParams::default(),
            },
            RequestClass::Interactive,
        )
        .unwrap();

    scheduler.advance();
    let calls = leaf.calls.lock().unwrap();
    assert_eq!(calls.sequences_allocated, vec![23]);
    assert_eq!(calls.prefill_positions, vec![0]);
}

#[test]
fn scheduler_passes_the_shared_prefix_boundary_to_prefill() {
    // P4-10 (GitHub #126): a claimant's sequence is allocated *against* the
    // leaf's prefix — that is what shares the pages and clones the mutable
    // state — and its prefill starts where the prefix ends. With a tail to
    // warm, both are visible: one shared allocation, and a prefill at the
    // boundary rather than at 0.
    let leaf = Arc::new(StubLeaf::with_tokens([]));
    let model = Arc::new(Model::load(leaf.clone()).unwrap());
    let compute = Arc::new(RuntimeCompute::new(model, 99));
    let mut scheduler = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: "stub".into(),
            kv_page_tokens: 4,
            ..SchedulerConfig::default()
        },
        compute,
    );
    let input = |tokens: Vec<u32>| RequestInput {
        model: "stub".into(),
        tokens,
        params: DecodeParams {
            max_tokens: Some(3),
            ..DecodeParams::default()
        },
    };
    scheduler
        .submit(input(vec![1, 2, 3, 4]), RequestClass::Interactive)
        .unwrap();
    scheduler.advance();
    assert_eq!(
        leaf.calls.lock().unwrap().prefixes_published,
        vec![4],
        "the publisher's chunk lands on its 4-token head and publishes it"
    );
    scheduler
        .submit(input(vec![1, 2, 3, 4, 5, 6]), RequestClass::Interactive)
        .unwrap();
    scheduler.advance();

    let calls = leaf.calls.lock().unwrap();
    assert_eq!(
        calls.prefill_positions,
        vec![0, 4],
        "the claimant's prefill starts where the prefix ends"
    );
    assert_eq!(
        calls.shared_allocations.len(),
        1,
        "and its sequence was built against the prefix, not allocated fresh"
    );
}

#[test]
fn a_full_prompt_match_is_allocated_against_the_prefix_and_never_prefilled() {
    // The claimant's prompt IS the cached prefix, so the clone alone puts it
    // where it needs to be — pending token included. There is no span left to
    // warm, and the leaf rejects an empty one ("null argument or empty
    // batch"), so the adapter must not make the call at all.
    let leaf = Arc::new(StubLeaf::with_tokens([]));
    let model = Arc::new(Model::load(leaf.clone()).unwrap());
    let compute = Arc::new(RuntimeCompute::new(model, 99));
    let mut scheduler = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: "stub".into(),
            kv_page_tokens: 4,
            ..SchedulerConfig::default()
        },
        compute,
    );
    let input = || RequestInput {
        model: "stub".into(),
        tokens: vec![1, 2, 3, 4],
        params: DecodeParams {
            max_tokens: Some(3),
            ..DecodeParams::default()
        },
    };
    scheduler
        .submit(input(), RequestClass::Interactive)
        .unwrap();
    scheduler.advance();
    scheduler
        .submit(input(), RequestClass::Interactive)
        .unwrap();
    scheduler.advance();

    let calls = leaf.calls.lock().unwrap();
    assert_eq!(
        calls.prefill_positions,
        vec![0],
        "only the publisher prefilled; the twin had nothing to warm"
    );
    assert_eq!(
        calls.shared_allocations.len(),
        1,
        "the twin still got a sequence — built against the prefix"
    );
    assert_eq!(calls.sequences_allocated.len(), 2, "two sequences in all");
}

#[test]
fn scheduler_releases_the_adapter_sequence_when_a_request_is_evicted() {
    let leaf = Arc::new(StubLeaf::with_tokens([]));
    let model = Arc::new(Model::load(leaf.clone()).unwrap());
    let compute = Arc::new(RuntimeCompute::new(model, 99));
    let mut scheduler = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: "stub".into(),
            max_in_flight: 9,
            host_capacity_pages: 64,
            ..SchedulerConfig::default()
        },
        compute,
    );
    for token in 0..9 {
        scheduler
            .submit(
                RequestInput {
                    model: "stub".into(),
                    tokens: vec![token],
                    params: DecodeParams {
                        max_tokens: Some(8),
                        ..DecodeParams::default()
                    },
                },
                RequestClass::Agent,
            )
            .unwrap();
    }

    scheduler.advance();
    let events = scheduler.advance();

    assert!(
        events
            .iter()
            .any(|event| matches!(event, ignis_core::SchedEvent::Evicted { .. }))
    );
    assert_eq!(leaf.calls.lock().unwrap().sequences_released, 1);
}
