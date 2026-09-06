use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use ignis_core::{
    Compute, ComputeError, ConcreteScheduler, DecodeJob, DecodeParams, PrefillJob, RequestClass,
    RequestInput, Scheduler, SchedulerConfig,
};
use ignis_runtime::{Model, RuntimeCompute, RuntimeStats, StepLeaf};

#[derive(Default)]
struct Calls {
    models_released: u32,
    stats_calls: u32,
    sequences_allocated: Vec<u32>,
    sequences_released: u32,
    prefill_positions: Vec<u32>,
    decode_batch_sizes: Vec<usize>,
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

    fn prefill(
        &self,
        _model: &Self::Model,
        _sequence: &mut Self::Sequence,
        _tokens: &[u32],
        start_position: u32,
    ) -> Result<(), i32> {
        let call = {
            let mut calls = self.calls.lock().unwrap();
            calls.prefill_positions.push(start_position);
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
    ) -> Result<Vec<u32>, i32> {
        if let Some(code) = self.decode_error {
            return Err(code);
        }
        self.calls
            .lock()
            .unwrap()
            .decode_batch_sizes
            .push(sequences.len());
        let mut tokens = self.tokens.lock().unwrap();
        Ok(sequences
            .iter()
            .map(|_| tokens.pop_front().unwrap_or(7))
            .collect())
    }
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
    assert_eq!(compute.decode_step(&[job.clone()]).unwrap(), vec![Some(7)]);
    assert_eq!(compute.decode_step(&[job]).unwrap(), vec![None]);

    compute.prefill_step(&[prefill(2, None)]).unwrap();
    assert_eq!(
        compute
            .decode_step(&[DecodeJob {
                request: 2,
                lane: 1,
                params: DecodeParams::default(),
            }])
            .unwrap(),
        vec![None]
    );
    assert_eq!(leaf.calls.lock().unwrap().sequences_released, 2);
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
        vec![Some(17), Some(23)]
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

    assert_eq!(leaf.calls.lock().unwrap().prefill_positions, vec![0, 4]);
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
