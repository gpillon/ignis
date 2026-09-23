use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use ignis_core::{
    Compute, ComputeError, ConcreteScheduler, DecodeJob, DecodeOutcome, DecodeParams, FinishReason,
    N_DECODE_LANES, PrefillJob, PrefillOutcome, RequestClass, RequestInput, RetainedAt, Scheduler,
    SchedulerConfig, SpecCounters,
};
use ignis_core::checkpoint::ReuseSource;
use ignis_core::scheduler::CheckpointClaim;
use ignis_core::vision::{Grid, MediaItem, Multimodal, TokenSpan};
use ignis_core::pointing::{AttentionQuery, PointingHead, SetQuery};
use ignis_runtime::{
    AttentionRead, DecodeLane, LaneRun, Model, MultimodalSpan, RuntimeCompute, RuntimeStats,
    StepLeaf,
};

/// The stub's output-head width (GitHub #237). Small enough to write out in
/// a test, wide enough that an answer token past it is a distinct case.
const STUB_VOCAB: u32 = 16;
/// What the stub reports for a constrained draw (GitHub #242): a real
/// number, neither absent nor certain, so a caller that never reads the
/// trace cannot pass by reporting zero uncertainty.
const STUB_PERMITTED_PROBABILITY: f32 = 0.75;

#[derive(Default)]
struct Calls {
    models_released: u32,
    stats_calls: u32,
    sequences_allocated: Vec<u32>,
    sequences_released: u32,
    prefill_positions: Vec<u32>,
    prefill_params: Vec<DecodeParams>,
    /// The permitted set of each prefill call, in order (GitHub #242).
    prefill_permitted: Vec<Vec<u32>>,
    /// GitHub #237: whether each prefill call was handed a logits buffer,
    /// and how long it was — a job that asked for no readout must not cost
    /// one, which is a claim about the *call*, not about the outcome.
    prefill_logit_buffers: Vec<Option<usize>>,
    decode_batch_sizes: Vec<usize>,
    decode_params: Vec<Vec<DecodeParams>>,
    /// P5-06 (GitHub #154): each decode round's per-lane budget and stop ids.
    decode_lanes: Vec<Vec<(u32, Vec<u32>)>>,
    /// The permitted set of each round's lanes, in order (GitHub #242).
    decode_permitted: Vec<Vec<Vec<u32>>>,
    /// P4-10 (GitHub #126): the prefixes published (their token counts) and
    /// the claims served (the context each claimant reserved), so a test can
    /// see that a claimant was allocated *against* a prefix rather than
    /// allocated normally and prefilled from a hole.
    prefixes_published: Vec<u32>,
    /// GitHub #215: the retained slot each publish and each capture was named,
    /// in call order — what the scheduler decided, reaching the leaf.
    publish_slots: Vec<u32>,
    capture_slots: Vec<u32>,
    shared_allocations: Vec<u32>,
    /// The prefix each shared allocation stood on (the stub's prefix is its
    /// token count), so a test can see *which* of a publisher's heads a
    /// claimant was given.
    claimed_prefixes: Vec<u32>,
    prefixes_released: u32,
    /// GitHub #186: the prompt checkpoints captured (their opener token
    /// counts), the claims served (each claimant's reservation), and the
    /// releases — so a test can see that a claimant was stood up *on a
    /// checkpoint* rather than allocated normally and prefilled from a hole,
    /// and that no device image outlives the ledger that names it.
    checkpoints_captured: Vec<u32>,
    checkpoint_allocations: Vec<u32>,
    checkpoints_released: u32,
    snapshots_taken: u32,
    restores: u32,
    /// GitHub #190: fail every checkpoint materialization while set.
    fail_checkpoint_snapshots: bool,
    /// GitHub #178: the token span begin of each encoded media item, the
    /// embeddings released (by encode order, from 1), and every multimodal
    /// prefill span.
    media_encoded: Vec<u32>,
    media_released: Vec<u32>,
    /// GitHub #243: the columns each embedding this stub has not released
    /// occupies, by encode order — the stub's side of the pool, which is
    /// what lets it answer `MEDIA_ENCODE_POOL_FULL`.
    media_live: std::collections::HashMap<u32, u64>,
    multimodal_spans: Vec<SpanCall>,
    /// GitHub #260: the attention readout each multimodal span was handed,
    /// `None` for a span that asked for none.
    attention_asked: Vec<Option<AttentionQuery>>,
    /// GitHub #260: the leaf cannot read the keys while set — it leaves the
    /// readout unread, as a real leaf does on a route that materialized none.
    attention_unreadable: bool,
}

#[derive(Debug, Clone, PartialEq)]
struct SpanCall {
    start: u32,
    positions: Vec<i32>,
    rope_delta: i32,
    /// (embedding, first column, scatter indices)
    media: Option<(u32, u32, Vec<i32>)>,
}

struct StubLeaf {
    calls: Mutex<Calls>,
    tokens: Mutex<VecDeque<u32>>,
    /// Whole runs to commit, one per lane, before `tokens` is drawn from.
    runs: Mutex<VecDeque<LaneRun>>,
    allocation_error: Option<i32>,
    prefill_error: Option<i32>,
    prefill_error_on_call: Option<(usize, i32)>,
    decode_error: Option<i32>,
    /// GitHub #186: the leaf refuses the next checkpoint capture with this
    /// code. A capture is a bet the leaf may decline — no room in its image
    /// pool, a sequence it will not capture — and declining must leave the
    /// batch alone.
    capture_error: Option<i32>,
    /// GitHub #243: the merged columns this stub's embedding pool holds, or
    /// `None` for a pool nothing fills. A bounded stub is what proves the
    /// eviction loop, since the real refusal comes from the leaf's bytes and
    /// this side owns only the policy.
    media_pool_columns: Option<u64>,
}

impl StubLeaf {
    /// The same stub with an embedding pool of `columns` merged columns
    /// (GitHub #243).
    fn with_media_pool(mut self, columns: u64) -> Self {
        self.media_pool_columns = Some(columns);
        self
    }

}

impl StubLeaf {
    fn with_tokens(tokens: impl IntoIterator<Item = u32>) -> Self {
        Self {
            calls: Mutex::new(Calls::default()),
            tokens: Mutex::new(tokens.into_iter().collect()),
            runs: Mutex::new(VecDeque::new()),
            allocation_error: None,
            prefill_error: None,
            prefill_error_on_call: None,
            decode_error: None,
            capture_error: None,
            media_pool_columns: None,
        }
    }

    fn with_runs(runs: impl IntoIterator<Item = LaneRun>) -> Self {
        let leaf = Self::with_tokens([]);
        *leaf.runs.lock().unwrap() = runs.into_iter().collect();
        leaf
    }

    fn failing_prefill(code: i32) -> Self {
        Self {
            calls: Mutex::new(Calls::default()),
            tokens: Mutex::new(VecDeque::new()),
            runs: Mutex::new(VecDeque::new()),
            allocation_error: None,
            prefill_error: Some(code),
            prefill_error_on_call: None,
            decode_error: None,
            capture_error: None,
            media_pool_columns: None,
        }
    }

    fn failing_allocation(code: i32) -> Self {
        Self {
            calls: Mutex::new(Calls::default()),
            tokens: Mutex::new(VecDeque::new()),
            runs: Mutex::new(VecDeque::new()),
            allocation_error: Some(code),
            prefill_error: None,
            prefill_error_on_call: None,
            decode_error: None,
            capture_error: None,
            media_pool_columns: None,
        }
    }

    fn failing_second_prefill(code: i32) -> Self {
        Self {
            calls: Mutex::new(Calls::default()),
            tokens: Mutex::new(VecDeque::new()),
            runs: Mutex::new(VecDeque::new()),
            allocation_error: None,
            prefill_error: None,
            prefill_error_on_call: Some((2, code)),
            decode_error: None,
            capture_error: None,
            media_pool_columns: None,
        }
    }

    fn failing_decode(code: i32) -> Self {
        Self {
            calls: Mutex::new(Calls::default()),
            tokens: Mutex::new(VecDeque::new()),
            runs: Mutex::new(VecDeque::new()),
            allocation_error: None,
            prefill_error: None,
            prefill_error_on_call: None,
            decode_error: Some(code),
            capture_error: None,
            media_pool_columns: None,
        }
    }
}

impl StepLeaf for StubLeaf {
    type Model = ();
    type Sequence = ();
    /// The tokens the prefix covers — enough for a stub to prove a claimant
    /// was handed the right entry.
    type Prefix = u32;
    type SnapshotBuf = Vec<u8>;
    /// The embedding's encode order, from 1.
    type Media = u32;
    /// The opener the checkpoint was captured at (GitHub #186) — enough for
    /// a stub to prove a claimant was handed the right entry.
    type Checkpoint = u32;

    fn load_model(&self) -> Result<Self::Model, i32> {
        Ok(())
    }

    fn encode_media(&self, _model: &Self::Model, item: &MediaItem) -> Result<Self::Media, i32> {
        let mut calls = self.calls.lock().unwrap();
        let columns = item.grid.vision_tokens();
        // GitHub #243: the pool refuses the item that does not fit what is
        // free *now*, by the code that means "release something and ask
        // again". The caller's eviction loop is what this models.
        if let Some(capacity) = self.media_pool_columns {
            let live: u64 = calls.media_live.values().sum();
            if live + columns > capacity {
                return Err(ignis_core::vision::MEDIA_ENCODE_POOL_FULL);
            }
        }
        calls.media_encoded.push(item.token_span.begin as u32);
        let handle = calls.media_encoded.len() as u32;
        calls.media_live.insert(handle, columns);
        Ok(handle)
    }

    fn release_media(&self, _model: &Self::Model, media: Self::Media) {
        let mut calls = self.calls.lock().unwrap();
        calls.media_live.remove(&media);
        calls.media_released.push(media);
    }

    fn prefill_multimodal(
        &self,
        model: &Self::Model,
        sequence: &mut Self::Sequence,
        tokens: &[u32],
        start_position: u32,
        params: DecodeParams,
        permitted: &[u32],
        span: MultimodalSpan<'_, Self::Media>,
        out_logits: Option<&mut [f32]>,
        attention: Option<&mut AttentionRead>,
    ) -> Result<f32, i32> {
        {
            let mut calls = self.calls.lock().unwrap();
            calls.attention_asked.push(attention.as_ref().map(|read| read.query.clone()));
            // A deterministic map the test can predict: each key scores its
            // own absolute position, and head `i` of a set peaks on key `2i`.
            if let Some(read) = attention
                && !calls.attention_unreadable
            {
                for (k, score) in read.scores.iter_mut().enumerate() {
                    *score = (read.query.key_begin + k as u32) as f32;
                }
                for (i, key) in read.set_argmax.iter_mut().enumerate() {
                    *key = 2 * i as u32;
                }
                // Spec 15: each peak's score, and the four around it. The
                // grid's columns say which exist -- NaN is "off the grid".
                let cols = read.query.set.as_ref().map_or(1, |set| set.grid_cols.max(1));
                let rows = read.query.key_count.div_ceil(cols);
                let peaks: Vec<u32> = read.set_argmax.clone();
                for (i, peak) in read.set_peak.iter_mut().enumerate() {
                    *peak = peaks[i] as f32 + 100.0;
                }
                for (i, key) in peaks.iter().enumerate() {
                    let (col, row) = (key % cols, key / cols);
                    let around = [
                        (col > 0).then(|| key - 1),
                        (col + 1 < cols).then(|| key + 1),
                        (row > 0).then(|| key - cols),
                        (row + 1 < rows).then(|| key + cols),
                    ];
                    for (j, neighbour) in around.into_iter().enumerate() {
                        read.set_neighbours[4 * i + j] =
                            neighbour.map_or(f32::NAN, |key| key as f32);
                    }
                }
                read.read = true;
            }
        }
        self.calls.lock().unwrap().multimodal_spans.push(SpanCall {
            start: start_position,
            positions: span.positions.to_vec(),
            rope_delta: span.rope_delta,
            media: span
                .media
                .map(|media| (*media.embedding, media.first_column, media.scatter_indices.to_vec())),
        });
        self.prefill(model, sequence, tokens, start_position, params, permitted, out_logits)
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
            free_vram_bytes: 0,
            reserved: Default::default(),
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
        prefix: &Self::Prefix,
    ) -> Result<Self::Sequence, i32> {
        if let Some(code) = self.allocation_error {
            return Err(code);
        }
        let mut calls = self.calls.lock().unwrap();
        calls.sequences_allocated.push(context_tokens);
        calls.shared_allocations.push(context_tokens);
        calls.claimed_prefixes.push(*prefix);
        Ok(())
    }

    fn publish_prefix(
        &self,
        _model: &Self::Model,
        _sequence: &mut Self::Sequence,
        prefix_tokens: u32,
        retained_slot: u32,
    ) -> Result<Self::Prefix, i32> {
        let mut calls = self.calls.lock().unwrap();
        calls.prefixes_published.push(prefix_tokens);
        calls.publish_slots.push(retained_slot);
        Ok(prefix_tokens)
    }

    fn release_prefix(&self, _model: &Self::Model, _prefix: Self::Prefix) {
        self.calls.lock().unwrap().prefixes_released += 1;
    }

    fn prefix_snapshot_bytes(&self, _model: &Self::Model, _prefix: &Self::Prefix) -> Result<u64, i32> {
        Ok(SNAPSHOT_MARKER.len() as u64)
    }

    fn prefix_snapshot_into(
        &self,
        _model: &Self::Model,
        _prefix: &Self::Prefix,
        dst: &mut [u8],
    ) -> Result<(), i32> {
        dst.copy_from_slice(SNAPSHOT_MARKER);
        self.calls.lock().unwrap().snapshots_taken += 1;
        Ok(())
    }

    fn capture_checkpoint(
        &self,
        _model: &Self::Model,
        _sequence: &mut Self::Sequence,
        opener_tokens: u32,
        retained_slot: u32,
    ) -> Result<Self::Checkpoint, i32> {
        if let Some(code) = self.capture_error {
            return Err(code);
        }
        let mut calls = self.calls.lock().unwrap();
        calls.checkpoints_captured.push(opener_tokens);
        calls.capture_slots.push(retained_slot);
        Ok(opener_tokens)
    }

    fn allocate_sequence_from_checkpoint(
        &self,
        _model: &Self::Model,
        context_tokens: u32,
        _checkpoint: &Self::Checkpoint,
    ) -> Result<(Self::Sequence, u64), i32> {
        if let Some(code) = self.allocation_error {
            return Err(code);
        }
        let mut calls = self.calls.lock().unwrap();
        calls.sequences_allocated.push(context_tokens);
        calls.checkpoint_allocations.push(context_tokens);
        // A nominal, deterministic restore cost, so `restore_micros` is
        // observable without a clock.
        Ok(((), 7))
    }

    fn release_checkpoint(&self, _model: &Self::Model, _checkpoint: Self::Checkpoint) {
        self.calls.lock().unwrap().checkpoints_released += 1;
    }

    fn checkpoint_snapshot_bytes(
        &self,
        _model: &Self::Model,
        _checkpoint: &Self::Checkpoint,
    ) -> Result<u64, i32> {
        Ok(SNAPSHOT_MARKER.len() as u64)
    }

    fn checkpoint_snapshot_into(
        &self,
        _model: &Self::Model,
        _checkpoint: &Self::Checkpoint,
        dst: &mut [u8],
    ) -> Result<(), i32> {
        let mut calls = self.calls.lock().unwrap();
        if calls.fail_checkpoint_snapshots {
            return Err(-7);
        }
        dst.copy_from_slice(SNAPSHOT_MARKER);
        calls.snapshots_taken += 1;
        Ok(())
    }

    fn vocab(&self, _model: &Self::Model) -> u32 {
        STUB_VOCAB
    }

    fn prefill(
        &self,
        _model: &Self::Model,
        _sequence: &mut Self::Sequence,
        _tokens: &[u32],
        start_position: u32,
        params: DecodeParams,
        permitted: &[u32],
        out_logits: Option<&mut [f32]>,
    ) -> Result<f32, i32> {
        let call = {
            let mut calls = self.calls.lock().unwrap();
            calls.prefill_positions.push(start_position);
            calls.prefill_params.push(params);
            calls.prefill_permitted.push(permitted.to_vec());
            calls
                .prefill_logit_buffers
                .push(out_logits.as_ref().map(|buffer| buffer.len()));
            calls.prefill_positions.len()
        };
        if let Some(buffer) = out_logits {
            // A ramp, so a gather that read the wrong column or the wrong
            // position would produce a wrong number rather than a
            // plausible one: column `i` is worth `i / 2`.
            for (column, logit) in buffer.iter_mut().enumerate() {
                *logit = column as f32 / 2.0;
            }
        }
        if let Some(code) = self.prefill_error.or_else(|| {
            self.prefill_error_on_call
                .filter(|(expected, _)| call == *expected)
                .map(|(_, code)| code)
        }) {
            return Err(code);
        }
        // GitHub #242: the stub draws under constraint the way the leaf
        // does — the *first* permitted id, which is enough to tell one
        // step's set from another's, at a probability that is neither 0 nor
        // 1 so a caller that ignores the trace cannot pass by accident.
        Ok(match permitted.is_empty() {
            true => 0.0,
            false => STUB_PERMITTED_PROBABILITY,
        })
    }

    fn decode(
        &self,
        _model: &Self::Model,
        sequences: &mut [&mut Self::Sequence],
        lanes: &[DecodeLane<'_>],
    ) -> Result<Vec<LaneRun>, i32> {
        if let Some(code) = self.decode_error {
            return Err(code);
        }
        let mut calls = self.calls.lock().unwrap();
        calls.decode_batch_sizes.push(sequences.len());
        calls.decode_params.push(lanes.iter().map(|lane| lane.params).collect());
        calls.decode_lanes.push(
            lanes
                .iter()
                .map(|lane| (lane.remaining_tokens, lane.stop_ids.to_vec()))
                .collect(),
        );
        calls
            .decode_permitted
            .push(lanes.iter().map(|lane| lane.permitted.to_vec()).collect());
        drop(calls);
        let mut runs = self.runs.lock().unwrap();
        let mut tokens = self.tokens.lock().unwrap();
        Ok(sequences
            .iter()
            .zip(lanes)
            .map(|(_, lane)| {
                let run = runs
                    .pop_front()
                    .unwrap_or_else(|| LaneRun::token(tokens.pop_front().unwrap_or(7)));
                // The draw this round made, for the token the *next* round
                // returns (GitHub #242).
                LaneRun {
                    drawn_probability: (!lane.permitted.is_empty())
                        .then_some(STUB_PERMITTED_PROBABILITY),
                    ..run
                }
            })
            .collect())
    }

    fn alloc_snapshot_buf(&self, bytes: u64) -> Result<Self::SnapshotBuf, i32> {
        Ok(vec![0u8; bytes as usize])
    }

    fn snapshot_bytes(&self, _model: &Self::Model, _sequence: &Self::Sequence) -> Result<u64, i32> {
        Ok(SNAPSHOT_MARKER.len() as u64)
    }

    fn snapshot_into(
        &self,
        _model: &Self::Model,
        _sequence: &Self::Sequence,
        dst: &mut [u8],
    ) -> Result<(), i32> {
        dst.copy_from_slice(SNAPSHOT_MARKER);
        self.calls.lock().unwrap().snapshots_taken += 1;
        Ok(())
    }

    fn restore_sequence(
        &self,
        _model: &Self::Model,
        _sequence: &mut Self::Sequence,
        src: &[u8],
    ) -> Result<(), i32> {
        if src != SNAPSHOT_MARKER {
            return Err(-4); // BAD_SNAPSHOT-equivalent for this stub
        }
        self.calls.lock().unwrap().restores += 1;
        Ok(())
    }
}

/// The stub leaf has no real device state (`Sequence = ()`), so it writes a
/// fixed marker instead of real bytes — enough to prove `RuntimeCompute`
/// threads a snapshot through `alloc_snapshot_buf` → `snapshot_into` →
/// `restore_sequence` intact, without needing a real sequence to snapshot.
const SNAPSHOT_MARKER: &[u8] = &[0xAB, 0xCD, 0xEF, 0x01];

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
                checkpoint: None,
                capture_checkpoint: None,
                multimodal: None,
                readout: None,
                request: 1,
                tokens: vec![4, 5],
                context_tokens: 9,
                start_position: 0,
                params: left,
                shared_prefix: None,
                publish_prefix: None,
                permitted: None,
                attention: None,
            },
            PrefillJob {
                checkpoint: None,
                capture_checkpoint: None,
                multimodal: None,
                readout: None,
                request: 2,
                tokens: vec![4, 5],
                context_tokens: 9,
                start_position: 0,
                params: right,
                shared_prefix: None,
                publish_prefix: None,
                permitted: None,
                attention: None,
            },
        ])
        .unwrap();

    compute
        .decode_step(&[
            DecodeJob {
                request: 1,
                lane: 0,
                params: left,
                remaining_tokens: 3,
                permitted: None,
            },
            DecodeJob {
                request: 2,
                lane: 1,
                params: right,
                remaining_tokens: 4,
                permitted: None,
            },
        ])
        .unwrap();

    let calls = leaf.calls.lock().unwrap();
    assert_eq!(calls.prefill_params, vec![left, right]);
    assert_eq!(calls.decode_params, vec![vec![left, right]]);
}

/// GitHub #242: the adapter holds the leaf's one-round lag so the scheduler
/// does not have to.
///
/// The leaf draws at the *end* of a call and returns at the *start* of the
/// next one, and a constrained draw's probability exists only at the moment
/// of the draw — at most 32 logits, gone by the round that emits the token.
/// So a round reports the probability held from the round before, and holds
/// the one it just made. An adapter that reported its own round's draw would
/// pair every digit with the next digit's confidence, which on a number is
/// wrong by a factor of ten and looks entirely plausible.
///
/// The stub reports a probability for a constrained draw and nothing for a
/// free one, so what this pins is *which round the number appears on*.
#[test]
fn the_adapter_holds_a_constrained_draw_for_the_round_that_emits_it() {
    let leaf = Arc::new(StubLeaf::with_tokens([11, 12, 13]));
    let model = Arc::new(Model::load(leaf.clone()).unwrap());
    let compute = RuntimeCompute::new(model, 99);
    let digits: ignis_core::constrained::PermittedSet = vec![3, 4, 5].into();

    // The prefill draws the run's first token under the first step's set.
    compute
        .prefill_step(&[PrefillJob {
            permitted: Some(digits.clone()),
            ..prefill(1, Some(8))
        }])
        .unwrap();
    assert_eq!(
        leaf.calls.lock().unwrap().prefill_permitted,
        vec![vec![3, 4, 5]],
        "the set reached the leaf's prefill, which is where a constrained run starts"
    );

    let round = |permitted: Option<ignis_core::constrained::PermittedSet>| {
        compute
            .decode_step(&[DecodeJob {
                request: 1,
                lane: 0,
                params: DecodeParams::default(),
                remaining_tokens: 8,
                permitted,
            }])
            .unwrap()
            .remove(0)
    };

    // Round one returns the prefill's draw, with the prefill's probability.
    let first = round(Some(digits.clone()));
    assert_eq!(first.tokens, vec![11]);
    assert_eq!(
        first.probabilities,
        vec![STUB_PERMITTED_PROBABILITY],
        "the probability of the token this round EMITS, which the prefill drew"
    );

    // The last round carries no set: its own draw is discarded, and the
    // token it returns still carries the probability of the round before.
    let last = round(None);
    assert_eq!(last.tokens, vec![12]);
    assert_eq!(last.probabilities, vec![STUB_PERMITTED_PROBABILITY]);

    // And once the schedule is spent nothing is left holding a number.
    let after = round(None);
    assert_eq!(after.tokens, vec![13]);
    assert!(
        after.probabilities.is_empty(),
        "a round after an unconstrained one reports nothing: the entry is removed when it is used, not left for the next lane to pick up"
    );
    assert_eq!(
        leaf.calls.lock().unwrap().decode_permitted,
        vec![vec![vec![3, 4, 5]], vec![Vec::<u32>::new()], vec![Vec::<u32>::new()]],
        "and each round carried its own step's set, not the one before it"
    );
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
            remaining_tokens: 8,
            permitted: None,
        })
        .collect();

    assert_eq!(compute.decode_step(&jobs), Err(ComputeError::Kernel(-1)));
    assert_eq!(compute.live_sequences(), 0);
}

fn prefill(request: u64, max_tokens: Option<u32>) -> PrefillJob {
    PrefillJob {
        checkpoint: None,
        capture_checkpoint: None,
        multimodal: None,
        readout: None,
        request,
        tokens: vec![4, 5],
        context_tokens: 9,
        start_position: 0,
        params: DecodeParams {
            max_tokens,
            ..DecodeParams::default()
        },
        shared_prefix: None,
        publish_prefix: None,
        permitted: None,
        attention: None,
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
            remaining_tokens: 8,
            permitted: None,
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
        remaining_tokens: 1,
        permitted: None,
};
    assert_eq!(
        compute.decode_step(&[job.clone()]).unwrap(),
        vec![DecodeOutcome::token(7)]
    );
    // The cap (`max_tokens: Some(1)`) was already reached by the first
    // token: `Length`, not `Stop` (the leaf never even runs — see
    // `decode_batch_sizes` below).
    assert_eq!(
        compute.decode_step(&[job]).unwrap(),
        vec![DecodeOutcome::finished(FinishReason::Length)]
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
                remaining_tokens: 8,
                permitted: None,
}])
            .unwrap(),
        vec![DecodeOutcome::finished(FinishReason::Stop)]
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
        remaining_tokens: 8,
        permitted: None,
};

    assert_eq!(
        compute.decode_step(std::slice::from_ref(&job)).unwrap(),
        vec![DecodeOutcome::token(99)]
    );
    assert_eq!(compute.decode_step(&[job]).unwrap(), vec![DecodeOutcome::token(7)]);
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
                    remaining_tokens: 8,
                    permitted: None,
                },
                DecodeJob {
                    request: 2,
                    lane: 1,
                    params: DecodeParams::default(),
                    remaining_tokens: 8,
                    permitted: None,
                },
            ])
            .unwrap(),
        vec![DecodeOutcome::token(17), DecodeOutcome::token(23)]
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
                decision: None,
                multimodal: None,
                opener_tokens: None,
                user_turn_tokens: None,
                system_block_tokens: None,
                model: "stub".into(),
                tokens: vec![1, 2],
                params: DecodeParams {
                    max_tokens: Some(1),
                    ..DecodeParams::default()
                },
                constrained: None,
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
                decision: None,
                multimodal: None,
                opener_tokens: None,
                user_turn_tokens: None,
                system_block_tokens: None,
                model: "stub".into(),
                tokens: vec![1, 2, 3],
                params: DecodeParams::default(),
                constrained: None,
            },
            RequestClass::Interactive,
        )
        .unwrap();

    scheduler.advance();
    let calls = leaf.calls.lock().unwrap();
    // GitHub #166: without `max_tokens` the reservation is the sequence
    // limit itself (prompt included), never prompt + limit — the leaf
    // refuses anything past its `max_context`.
    assert_eq!(calls.sequences_allocated, vec![20]);
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
        decision: None,
        multimodal: None,
        opener_tokens: None,
        user_turn_tokens: None,
        system_block_tokens: None,
        model: "stub".into(),
        tokens,
        params: DecodeParams {
            max_tokens: Some(3),
            ..DecodeParams::default()
        },
        constrained: None,
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
        decision: None,
        multimodal: None,
        opener_tokens: None,
        user_turn_tokens: None,
        system_block_tokens: None,
        model: "stub".into(),
        tokens: vec![1, 2, 3, 4],
        params: DecodeParams {
            max_tokens: Some(3),
            ..DecodeParams::default()
        },
        constrained: None,
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

// ── GitHub #186: prompt checkpoints through the adapter ────────────────

/// A scheduler over `leaf` with 4-token KV pages, so a short prompt still
/// has a whole page under its opener.
fn checkpoint_scheduler(leaf: Arc<StubLeaf>) -> ConcreteScheduler {
    let model = Arc::new(Model::load(leaf).unwrap());
    let compute = Arc::new(RuntimeCompute::new(model, 99));
    ConcreteScheduler::with_config(
        SchedulerConfig {
            model: "stub".into(),
            kv_page_tokens: 4,
            ..SchedulerConfig::default()
        },
        compute,
    )
}

/// An 8-token prompt whose generation opener ends at 6 — two whole pages
/// below it, and two tokens past it, the shape a rendered chat prompt has.
fn checkpoint_input(tokens: Vec<u32>, opener: Option<u32>) -> RequestInput {
    RequestInput {
        decision: None,
        multimodal: None,
        opener_tokens: opener,
        user_turn_tokens: None,
        system_block_tokens: None,
        model: "stub".into(),
        tokens,
        params: DecodeParams {
            max_tokens: Some(3),
            ..DecodeParams::default()
        },
        constrained: None,
    }
}

#[test]
fn a_later_request_is_allocated_against_the_checkpoint_the_first_one_left() {
    // The whole seam, through the adapter rather than the mock: the capture
    // lands on the chunk that ends on the opener, and the next request's
    // sequence is built *on the checkpoint* — not allocated fresh and
    // prefilled from a hole, which is what it would be if the claim were
    // dropped anywhere between the scheduler and the leaf.
    let leaf = Arc::new(StubLeaf::with_tokens([]));
    let mut scheduler = checkpoint_scheduler(leaf.clone());
    scheduler
        .submit(
            checkpoint_input(vec![1, 2, 3, 4, 5, 6, 7, 8], Some(6)),
            RequestClass::Interactive,
        )
        .unwrap();
    while !scheduler.is_idle() {
        scheduler.advance();
    }
    assert_eq!(
        leaf.calls.lock().unwrap().checkpoints_captured,
        vec![6],
        "captured at the opener, not at the page boundary below it"
    );
    assert_eq!(
        leaf.calls.lock().unwrap().prefixes_published,
        vec![4],
        "on the shared prefix holding the whole pages under it"
    );

    // The next turn: the same history up to the opener, then new text.
    scheduler
        .submit(
            checkpoint_input(vec![1, 2, 3, 4, 5, 6, 90, 91, 92, 93], Some(9)),
            RequestClass::Interactive,
        )
        .unwrap();
    while !scheduler.is_idle() {
        scheduler.advance();
    }

    let (from_checkpoint, shared, positions) = {
        let calls = leaf.calls.lock().unwrap();
        (
            calls.checkpoint_allocations.len(),
            calls.shared_allocations.len(),
            calls.prefill_positions.clone(),
        )
    };
    assert_eq!(from_checkpoint, 1, "the later request stood up on the checkpoint");
    assert_eq!(
        shared, 0,
        "and not on a plain sibling prefix, which would stop a page short of the opener"
    );
    assert!(
        positions.contains(&6),
        "its prefill resumes at the opener: {positions:?}"
    );
}

#[test]
fn a_declined_capture_leaves_the_batch_and_the_request_alone() {
    // A capture is a bet. A leaf that will not take it — no room in its own
    // image pool, a sequence it refuses — must not cost the request that was
    // only trying to prefill its prompt.
    let leaf = Arc::new(StubLeaf {
        capture_error: Some(-1),
        media_pool_columns: None,
        ..StubLeaf::with_tokens([])
    });
    let mut scheduler = checkpoint_scheduler(leaf.clone());
    let request = scheduler
        .submit(
            checkpoint_input(vec![1, 2, 3, 4, 5, 6, 7, 8], Some(6)),
            RequestClass::Interactive,
        )
        .unwrap();
    let mut done = false;
    while !scheduler.is_idle() {
        for event in scheduler.advance() {
            if let ignis_core::SchedEvent::Done { request: r, reason, .. } = event {
                assert_eq!(r, request);
                assert_ne!(
                    reason,
                    ignis_core::FinishReason::Error,
                    "a refused capture must not fail the request"
                );
                done = true;
            }
        }
    }
    assert!(done, "the request completed");
    let (captured, positions) = {
        let calls = leaf.calls.lock().unwrap();
        (calls.checkpoints_captured.clone(), calls.prefill_positions.clone())
    };
    assert!(captured.is_empty(), "nothing was captured");
    assert_eq!(
        positions,
        vec![0, 4, 6],
        "and its prompt was prefilled in full, cuts and all"
    );
}

#[test]
fn a_reclaimed_checkpoint_spills_to_kv_ram_and_releases_the_device_handle() {
    // The scheduler discarding a retained entry must reach the leaf: an
    // image nothing releases is device memory nothing will ever free.
    let leaf = Arc::new(StubLeaf::with_tokens([]));
    let model = Arc::new(Model::load(leaf.clone()).unwrap());
    let compute = Arc::new(RuntimeCompute::new(model, 99));
    let observe = compute.clone();
    // A pool with room for exactly one sequence's worth of pages, so the
    // second request has to take the first's retained pages back.
    let mut scheduler = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: "stub".into(),
            kv_page_tokens: 4,
            max_sequence_tokens: 16,
            kv_capacity_pages: 4,
            ..SchedulerConfig::default()
        },
        compute,
    );
    scheduler
        .submit(
            checkpoint_input(vec![1, 2, 3, 4, 5, 6, 7, 8], Some(6)),
            RequestClass::Interactive,
        )
        .unwrap();
    while !scheduler.is_idle() {
        scheduler.advance();
    }
    assert_eq!(observe.live_checkpoints(), 1, "one image is held");

    scheduler
        .submit(
            checkpoint_input((100..113).collect(), None),
            RequestClass::Interactive,
        )
        .unwrap();
    while !scheduler.is_idle() {
        scheduler.advance();
    }
    assert_eq!(
        leaf.calls.lock().unwrap().checkpoints_released,
        1,
        "the reclaimed entry's image went back to the leaf"
    );
    assert_eq!(observe.live_checkpoints(), 0, "and the adapter holds none");
    assert_eq!(
        observe.retained_checkpoints(),
        1,
        "the materialized blob stays in KV-RAM for a later non-consuming restore"
    );
    assert_eq!(
        leaf.calls.lock().unwrap().snapshots_taken,
        1,
        "spill is lazy: exactly the reclaimed checkpoint crossed PCIe"
    );
}

#[test]
fn a_spill_the_leaf_fails_leaves_the_device_image_to_discard() {
    // GitHub #190: every step before the release leaves the image where it
    // was, so the scheduler's discard that follows still has a handle to drop
    // — and a later spill of the same checkpoint can still succeed.
    let leaf = Arc::new(StubLeaf::with_tokens([]));
    let model = Arc::new(Model::load(leaf.clone()).unwrap());
    let compute = RuntimeCompute::new(model, 99);
    compute
        .prefill_step(&[PrefillJob {
            request: 1,
            tokens: vec![1, 2, 3, 4, 5, 6],
            context_tokens: 16,
            start_position: 0,
            params: DecodeParams::default(),
            shared_prefix: None,
            publish_prefix: None,
            checkpoint: None,
            capture_checkpoint: Some(RetainedAt { tokens: 6, slot: 0 }),
            multimodal: None,
            readout: None,
            permitted: None,
            attention: None,
}])
        .unwrap();

    leaf.calls.lock().unwrap().fail_checkpoint_snapshots = true;
    assert!(compute.spill_checkpoint(1).is_err());
    assert_eq!(compute.live_checkpoints(), 1, "the device image is still held");
    assert_eq!(compute.retained_checkpoints(), 0, "and no blob was kept");
    assert_eq!(leaf.calls.lock().unwrap().checkpoints_released, 0);

    leaf.calls.lock().unwrap().fail_checkpoint_snapshots = false;
    assert!(compute.spill_checkpoint(1).is_ok(), "the same checkpoint spills once the leaf can");
    assert_eq!(compute.live_checkpoints(), 0);
    assert_eq!(compute.retained_checkpoints(), 1);
}

#[test]
fn a_kv_ram_checkpoint_restores_repeatedly_without_consuming_its_blob() {
    let leaf = Arc::new(StubLeaf::with_tokens([]));
    let model = Arc::new(Model::load(leaf.clone()).unwrap());
    let compute = RuntimeCompute::new(model, 99);
    let params = DecodeParams::default();
    compute
        .prefill_step(&[PrefillJob {
            request: 1,
            tokens: vec![1, 2, 3, 4, 5, 6],
            context_tokens: 16,
            start_position: 0,
            params,
            shared_prefix: None,
            publish_prefix: None,
            checkpoint: None,
            capture_checkpoint: Some(RetainedAt { tokens: 6, slot: 2 }),
            multimodal: None,
            readout: None,
            permitted: None,
            attention: None,
}])
        .unwrap();
    assert_eq!(leaf.calls.lock().unwrap().capture_slots, vec![2], "the slot the job named");
    compute.spill_checkpoint(1).unwrap();

    for request in [2, 3] {
        compute
            .prefill_step(&[PrefillJob {
                request,
                tokens: vec![90, 91],
                context_tokens: 16,
                start_position: 6,
                params,
                shared_prefix: None,
                publish_prefix: None,
                checkpoint: Some(CheckpointClaim {
                    publisher: 1,
                    tokens: 6,
                    source: ReuseSource::KvRam,
                }),
                capture_checkpoint: None,
                multimodal: None,
                readout: None,
                permitted: None,
                attention: None,
}])
            .unwrap();
    }

    assert_eq!(leaf.calls.lock().unwrap().restores, 2);
    assert_eq!(compute.retained_checkpoints(), 1, "restore never consumes the blob");
    assert_eq!(
        leaf.calls.lock().unwrap().checkpoint_allocations.len(),
        0,
        "KV-RAM restores into fresh standalone sequences, not a dead device checkpoint"
    );
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
            host_capacity_bytes: 64,
            ..SchedulerConfig::default()
        },
        compute,
    );
    for token in 0..9 {
        scheduler
            .submit(
                RequestInput {
                    decision: None,
                    multimodal: None,
                    opener_tokens: None,
                    user_turn_tokens: None,
                    system_block_tokens: None,
                    model: "stub".into(),
                    tokens: vec![token],
                    params: DecodeParams {
                        max_tokens: Some(8),
                        ..DecodeParams::default()
                    },
                    constrained: None,
                },
                RequestClass::Agent,
            )
            .unwrap();
    }

    scheduler.advance();
    let mut events = scheduler.advance();

    assert!(
        events
            .iter()
            .any(|event| matches!(event, ignis_core::SchedEvent::Evicted { .. }))
    );
    assert_eq!(leaf.calls.lock().unwrap().sequences_released, 1);
    // P4-07, GitHub #125: eviction snapshots through the real leaf seam
    // (`StepLeaf::snapshot_bytes` / `alloc_snapshot_buf` / `snapshot_into`)
    // before releasing the sequence — not a bare `release`.
    assert_eq!(
        leaf.calls.lock().unwrap().snapshots_taken, 1,
        "eviction takes a real snapshot through the leaf before releasing"
    );

    // Run to idle: the evicted request is restored (through
    // `StepLeaf::restore_sequence`) and completes.
    while !scheduler.is_idle() {
        events.extend(scheduler.advance());
    }
    assert!(
        events
            .iter()
            .any(|event| matches!(event, ignis_core::SchedEvent::Restored { .. })),
        "the evicted request is restored, not re-prefilled"
    );
    assert_eq!(
        leaf.calls.lock().unwrap().restores, 1,
        "restore threads the snapshot back through `StepLeaf::restore_sequence`"
    );
}

// ── P5-06 (GitHub #154): a decode round commits a run ───────────────────

fn job(request: u64, params: DecodeParams, remaining_tokens: u32) -> DecodeJob {
    DecodeJob {
        request,
        lane: request as usize,
        params,
        remaining_tokens,
        permitted: None,
    }
}

#[test]
fn the_leaf_is_handed_each_lanes_budget_and_stop_ids() {
    let leaf = Arc::new(StubLeaf::with_runs([
        LaneRun {
            tokens: vec![5, 6, 7],
            spec: None,
                    drawn_probability: None,
},
        LaneRun::token(8),
    ]));
    let model = Arc::new(Model::load(leaf.clone()).unwrap());
    let compute = RuntimeCompute::new(model, 99);
    let capped = DecodeParams {
        max_tokens: Some(10),
        ..DecodeParams::default()
    };
    let past_eos = DecodeParams {
        ignore_eos: true,
        ..DecodeParams::default()
    };
    compute
        .prefill_step(&[prefill(1, Some(10)), prefill(2, None)])
        .unwrap();

    compute
        .decode_step(&[job(1, capped, 4), job(2, past_eos, 6)])
        .unwrap();
    // Request 1 committed three of its ten: the scheduler's budget (20) is
    // no longer the tighter one, its own `max_tokens` is.
    compute
        .decode_step(&[job(1, capped, 20), job(2, past_eos, 5)])
        .unwrap();

    assert_eq!(
        leaf.calls.lock().unwrap().decode_lanes,
        vec![
            vec![(4, vec![99]), (6, vec![])],
            vec![(7, vec![99]), (5, vec![])],
        ],
        "the budget is the tighter of the scheduler's and max_tokens; a lane past EOS has no stop"
    );
}

#[test]
fn a_run_cut_at_eos_emits_the_tokens_before_it_and_finishes_with_stop() {
    let leaf = Arc::new(StubLeaf::with_runs([LaneRun {
        tokens: vec![5, 6, 99],
        spec: None,
            drawn_probability: None,
}]));
    let model = Arc::new(Model::load(leaf.clone()).unwrap());
    let compute = RuntimeCompute::new(model, 99);
    compute.prefill_step(&[prefill(1, None)]).unwrap();

    assert_eq!(
        compute
            .decode_step(&[job(1, DecodeParams::default(), 8)])
            .unwrap(),
        vec![DecodeOutcome::run_then_finished(vec![5, 6], FinishReason::Stop)]
    );
    assert_eq!(compute.live_sequences(), 0);
    assert_eq!(leaf.calls.lock().unwrap().sequences_released, 1);
}

#[test]
fn a_committed_run_counts_toward_max_tokens() {
    let leaf = Arc::new(StubLeaf::with_runs([LaneRun {
        tokens: vec![5, 6, 7],
        spec: None,
            drawn_probability: None,
}]));
    let model = Arc::new(Model::load(leaf.clone()).unwrap());
    let compute = RuntimeCompute::new(model, 99);
    let params = DecodeParams {
        max_tokens: Some(3),
        ..DecodeParams::default()
    };
    compute.prefill_step(&[prefill(1, Some(3))]).unwrap();

    assert_eq!(
        compute.decode_step(&[job(1, params, 3)]).unwrap(),
        vec![DecodeOutcome::run(vec![5, 6, 7])]
    );
    assert_eq!(
        compute.decode_step(&[job(1, params, 3)]).unwrap(),
        vec![DecodeOutcome::finished(FinishReason::Length)]
    );
    assert_eq!(
        leaf.calls.lock().unwrap().decode_batch_sizes,
        vec![1],
        "the capped request never reaches the leaf again"
    );
}

#[test]
fn a_verify_rounds_counters_ride_its_outcome() {
    let spec = SpecCounters::round(3, 2);
    let leaf = Arc::new(StubLeaf::with_runs([LaneRun {
        tokens: vec![5, 6, 7],
        spec: Some(spec),
            drawn_probability: None,
}]));
    let model = Arc::new(Model::load(leaf).unwrap());
    let compute = RuntimeCompute::new(model, 99);
    compute.prefill_step(&[prefill(1, None)]).unwrap();

    assert_eq!(
        compute
            .decode_step(&[job(1, DecodeParams::default(), 8)])
            .unwrap(),
        vec![DecodeOutcome::run(vec![5, 6, 7]).with_spec(spec)]
    );
}

#[test]
fn an_empty_run_is_refused_without_losing_the_sequence() {
    let leaf = Arc::new(StubLeaf::with_runs([LaneRun {
        tokens: vec![],
        spec: None,
            drawn_probability: None,
}]));
    let model = Arc::new(Model::load(leaf).unwrap());
    let compute = RuntimeCompute::new(model, 99);
    compute.prefill_step(&[prefill(1, None)]).unwrap();

    assert_eq!(
        compute.decode_step(&[job(1, DecodeParams::default(), 8)]),
        Err(ComputeError::Kernel(-1))
    );
    assert_eq!(compute.live_sequences(), 1);
}

// ── GitHub #178: media embeddings across a multimodal prompt's chunks ───────

/// An image of `count` merged tokens at prompt tokens `begin..begin+count`.
/// A distinct picture at `begin`. The digest is derived from `begin` because
/// the cache (GitHub #243) keys on it: a fixture that left every item at
/// `[0; 32]` would make two different pictures one entry and hide the bug it
/// is meant to catch.
fn image(begin: usize, count: usize) -> MediaItem {
    image_of(begin as u8, begin, count)
}

/// Picture `digest`, whose placeholders start at `begin`. Two calls with one
/// digest and one `count` are the same bytes at the same grid asked about in
/// two places — a fan-out.
fn image_of(digest: u8, begin: usize, count: usize) -> MediaItem {
    let mut content_digest = [0; 32];
    content_digest[0] = digest;
    MediaItem {
        grid: Grid { t: 1, h: 2, w: 2 * count as u32 },
        token_span: TokenSpan { begin, count },
        patches: Vec::new(),
        content_digest,
    }
}

fn multimodal(tokens: usize, media: Vec<MediaItem>) -> Arc<Multimodal> {
    let positions = (0..3).flat_map(|axis| (0..tokens as i32).map(move |t| axis * 100 + t)).collect();
    Arc::new(Multimodal { positions, rope_delta: -2, media })
}

fn multimodal_job(request: u64, prompt: &Arc<Multimodal>, start: u32, len: u32) -> PrefillJob {
    PrefillJob {
        request,
        tokens: (start..start + len).collect(),
        context_tokens: 64,
        start_position: start,
        params: DecodeParams::default(),
        shared_prefix: None,
        publish_prefix: None,
        checkpoint: None,
        capture_checkpoint: None,
        multimodal: Some(prompt.clone()),
        readout: None,
        permitted: None,
        attention: None,
    }
}

fn stub_compute(leaf: StubLeaf) -> (Arc<StubLeaf>, RuntimeCompute<StubLeaf>) {
    let leaf = Arc::new(leaf);
    let model = Arc::new(Model::load(leaf.clone()).unwrap());
    (leaf, RuntimeCompute::new(model, 99))
}

#[test]
fn a_media_item_is_encoded_once_and_let_go_after_its_last_placeholder() {
    let (leaf, compute) = stub_compute(StubLeaf::with_tokens([]));
    let prompt = multimodal(40, vec![image(10, 20)]);

    let encoding = compute.prefill_step(&[multimodal_job(1, &prompt, 0, 18)]).unwrap();
    assert_eq!(encoding.len(), 1, "one outcome per job (GitHub #192)");
    assert_eq!(compute.live_media(), 1, "the item spans into the next chunk");
    let continuation = compute.prefill_step(&[multimodal_job(1, &prompt, 18, 22)]).unwrap();
    assert_eq!(compute.live_media(), 0);
    // GitHub #192: the continuation reuses the live embedding, so it encodes
    // nothing and reports no encode time. (The encoding chunk's own
    // microseconds are whatever the stub leaf took — near zero, so asserting
    // a lower bound on it would only be flaky.)
    assert_eq!(continuation, [PrefillOutcome::default()]);
    // GitHub #243: the last placeholder ends the *hold*, not the embedding.
    // Nothing here has asked for the pool's room back, so the picture is
    // still encoded — which is the whole of what a fan-out reuses.
    assert_eq!(compute.cached_media(), 1);
    assert_eq!(compute.cached_media_columns(), 20);

    let calls = leaf.calls.lock().unwrap();
    assert_eq!(calls.media_encoded, [10], "one encode for the whole item");
    assert!(calls.media_released.is_empty(), "nothing needed the room");
    assert_eq!(
        calls.multimodal_spans,
        [
            SpanCall {
                start: 0,
                positions: prompt.span_positions(0, 18),
                rope_delta: -2,
                media: Some((1, 0, (10..18).collect())),
            },
            SpanCall {
                start: 18,
                positions: prompt.span_positions(18, 22),
                rope_delta: -2,
                media: Some((1, 8, (0..12).collect())),
            },
        ]
    );
    assert!(calls.prefill_positions == [0, 18], "the multimodal spans warm the sequence");
}

#[test]
fn a_text_chunk_of_a_multimodal_prompt_carries_positions_but_no_media() {
    let (leaf, compute) = stub_compute(StubLeaf::with_tokens([]));
    let prompt = multimodal(40, vec![image(30, 4)]);
    let outcomes = compute.prefill_step(&[multimodal_job(1, &prompt, 0, 30)]).unwrap();
    assert_eq!(outcomes, [PrefillOutcome::default()], "a text chunk encodes nothing");
    let calls = leaf.calls.lock().unwrap();
    assert!(calls.media_encoded.is_empty());
    assert_eq!(calls.multimodal_spans[0].media, None);
    assert_eq!(calls.multimodal_spans[0].positions, prompt.span_positions(0, 30));
}

#[test]
fn a_cancelled_request_lets_go_of_its_media_without_dropping_it() {
    let (leaf, compute) = stub_compute(StubLeaf::with_tokens([]));
    let prompt = multimodal(40, vec![image(10, 20)]);
    compute.prefill_step(&[multimodal_job(1, &prompt, 0, 18)]).unwrap();
    compute.release(1);
    assert_eq!(compute.live_media(), 0);
    assert_eq!(compute.live_sequences(), 0);
    // GitHub #243: a cancel is the moment a sibling asking the next question
    // about the same picture is about to want it, so the entry stays.
    assert_eq!(compute.cached_media(), 1);
    assert!(leaf.calls.lock().unwrap().media_released.is_empty());
}

#[test]
fn an_evicted_request_lets_go_of_its_media_and_finds_it_again_after_restore() {
    // GitHub #194: a request evicted mid-item keeps no vision state while it
    // sits in KV-RAM — a *hold* on an embedding is state the eviction cannot
    // carry. GitHub #243: the embedding it let go of is still in the cache,
    // so the restored continuation picks it back up instead of running the
    // tower a second time. That was the old behaviour's price, paid because
    // the load reserved room for exactly one item; the pool is what stopped
    // it being the only option.
    let (leaf, compute) = stub_compute(StubLeaf::with_tokens([]));
    let prompt = multimodal(40, vec![image(10, 20)]);
    compute.prefill_step(&[multimodal_job(1, &prompt, 0, 18)]).unwrap();
    assert_eq!(compute.live_media(), 1);

    compute.evict(1).unwrap();
    assert_eq!(compute.live_media(), 0, "an evicted request holds no embedding");
    assert!(leaf.calls.lock().unwrap().media_released.is_empty());

    compute.restore(1, 64).unwrap();
    compute.prefill_step(&[multimodal_job(1, &prompt, 18, 22)]).unwrap();
    assert_eq!(compute.live_media(), 0);
    let calls = leaf.calls.lock().unwrap();
    assert_eq!(calls.media_encoded, [10], "the continuation reuses the encode");
    assert_eq!(calls.multimodal_spans[1].media, Some((1, 8, (0..12).collect())));
}

#[test]
fn a_failed_multimodal_chunk_keeps_the_media_it_encoded_for_the_retry() {
    // The encode succeeded and the item has not changed; only the prefill
    // failed. GitHub #243: the batch unwind gives back the *hold*, so the
    // retry the scheduler makes (`MAX_PREFILL_ATTEMPTS`) finds the embedding
    // rather than paying the tower again for the same picture.
    let (leaf, compute) = stub_compute(StubLeaf::failing_prefill(-3));
    let prompt = multimodal(40, vec![image(10, 20)]);
    assert!(compute.prefill_step(&[multimodal_job(1, &prompt, 0, 18)]).is_err());
    assert_eq!(compute.live_media(), 0);
    assert_eq!(compute.cached_media(), 1);
    assert!(leaf.calls.lock().unwrap().media_released.is_empty());

    assert!(compute.prefill_step(&[multimodal_job(1, &prompt, 0, 18)]).is_err());
    assert_eq!(leaf.calls.lock().unwrap().media_encoded, [10], "one encode across both attempts");
}

#[test]
fn a_scheduled_multimodal_request_encodes_and_releases_every_item() {
    let leaf = Arc::new(StubLeaf::with_tokens([]));
    let model = Arc::new(Model::load(leaf.clone()).unwrap());
    let compute = Arc::new(RuntimeCompute::new(model, 99));
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: "stub".into(),
            serving_chunk_tokens: 16,
            ..SchedulerConfig::default()
        },
        compute.clone(),
    );
    sched
        .submit(
            RequestInput {
                decision: None,
                model: "stub".into(),
                tokens: (0..40).collect(),
                params: DecodeParams { max_tokens: Some(2), ..DecodeParams::default() },
                multimodal: Some(multimodal(40, vec![image(5, 10), image(20, 10)])),
                opener_tokens: None,
                user_turn_tokens: None,
                system_block_tokens: None,
                constrained: None,
            },
            RequestClass::Agent,
        )
        .unwrap();
    while !sched.is_idle() {
        sched.advance();
    }
    let calls = leaf.calls.lock().unwrap();
    assert_eq!(calls.media_encoded, [5, 20]);
    assert_eq!(compute.live_media(), 0, "nothing is holding either item");
    // GitHub #243: two pictures, two entries. They are two because their
    // digests differ — the cache keys on the bytes, not on where the
    // placeholders landed.
    assert_eq!(compute.cached_media(), 2);
    assert!(calls.media_released.is_empty());
}

#[test]
fn a_second_question_about_one_picture_does_not_encode_it_again() {
    // GitHub #243, acceptance 1. The two questions never overlap — exactly
    // one request holds multi-tick prefill progress at a time, and the first
    // has let its item go before the second's first chunk runs — so this is
    // not reference counting working. It is the entry outliving its last
    // holder, which is the mechanism the slice adds.
    let (leaf, compute) = stub_compute(StubLeaf::with_tokens([]));
    let picture = image_of(7, 10, 20);
    let first = multimodal(40, vec![picture.clone()]);
    let second = multimodal(40, vec![picture]);

    compute.prefill_step(&[multimodal_job(1, &first, 0, 18)]).unwrap();
    compute.prefill_step(&[multimodal_job(1, &first, 18, 22)]).unwrap();
    compute.release(1);

    let opening = compute.prefill_step(&[multimodal_job(2, &second, 0, 18)]).unwrap();
    assert_eq!(
        opening,
        [PrefillOutcome::default()],
        "the second question reports no encode time"
    );
    compute.prefill_step(&[multimodal_job(2, &second, 18, 22)]).unwrap();

    let calls = leaf.calls.lock().unwrap();
    assert_eq!(calls.media_encoded, [10], "one encode for both questions");
    // Both questions were prefilled against the same embedding, at the same
    // columns: the answer cannot differ because the columns did not.
    assert_eq!(calls.multimodal_spans[0].media, calls.multimodal_spans[2].media);
    assert_eq!(calls.multimodal_spans[1].media, calls.multimodal_spans[3].media);
}

#[test]
fn the_same_bytes_at_another_grid_are_another_embedding() {
    // The digest alone is not the key: the same picture packed into a
    // different grid expands to different placeholders, so the encoder's
    // output is a different thing (GitHub #243, and `concrete::media_keys`
    // for the same reasoning on the prefix side).
    let (leaf, compute) = stub_compute(StubLeaf::with_tokens([]));
    let wide = multimodal(40, vec![image_of(7, 10, 20)]);
    let narrow = multimodal(40, vec![image_of(7, 10, 10)]);

    compute.prefill_step(&[multimodal_job(1, &wide, 0, 18)]).unwrap();
    compute.prefill_step(&[multimodal_job(1, &wide, 18, 22)]).unwrap();
    compute.prefill_step(&[multimodal_job(2, &narrow, 0, 18)]).unwrap();

    assert_eq!(leaf.calls.lock().unwrap().media_encoded, [10, 10]);
    assert_eq!(compute.cached_media(), 2);
}

#[test]
fn a_full_pool_gives_up_an_unheld_embedding_rather_than_refusing() {
    // GitHub #243: the leaf owns the bytes and answers only "full"; the
    // policy is `RuntimeCompute`'s. A pool of exactly one item, then a
    // second picture: the first is nobody's, so it goes.
    let (leaf, compute) = stub_compute(StubLeaf::with_tokens([]).with_media_pool(20));
    let first = multimodal(40, vec![image_of(1, 10, 20)]);
    let second = multimodal(40, vec![image_of(2, 10, 20)]);

    compute.prefill_step(&[multimodal_job(1, &first, 0, 18)]).unwrap();
    compute.prefill_step(&[multimodal_job(1, &first, 18, 22)]).unwrap();
    assert_eq!(compute.cached_media(), 1);

    compute.prefill_step(&[multimodal_job(2, &second, 0, 18)]).unwrap();
    // Acceptance 3: a fan-out over a *different* image replaces, it does not
    // grow. The bound is the leaf's pool, and nothing here counts bytes.
    assert_eq!(compute.cached_media(), 1);
    assert_eq!(compute.cached_media_columns(), 20);
    let calls = leaf.calls.lock().unwrap();
    assert_eq!(calls.media_encoded, [10, 10]);
    assert_eq!(calls.media_released, [1], "the unheld first picture paid for the second");
    assert_eq!(
        calls.media_live.values().sum::<u64>(),
        20,
        "the leaf's pool never held two at once"
    );
}

#[test]
fn a_full_pool_gives_up_the_least_recently_used_embedding() {
    let (leaf, compute) = stub_compute(StubLeaf::with_tokens([]).with_media_pool(40));
    let a = multimodal(40, vec![image_of(1, 10, 20)]);
    let b = multimodal(40, vec![image_of(2, 10, 20)]);
    let c = multimodal(40, vec![image_of(3, 10, 20)]);

    for (request, prompt) in [(1u64, &a), (2, &b)] {
        compute.prefill_step(&[multimodal_job(request, prompt, 0, 18)]).unwrap();
        compute.prefill_step(&[multimodal_job(request, prompt, 18, 22)]).unwrap();
    }
    // Touch A again, so B is the older release.
    compute.prefill_step(&[multimodal_job(3, &a, 0, 18)]).unwrap();
    compute.prefill_step(&[multimodal_job(3, &a, 18, 22)]).unwrap();

    compute.prefill_step(&[multimodal_job(4, &c, 0, 18)]).unwrap();
    assert_eq!(
        leaf.calls.lock().unwrap().media_released,
        [2],
        "B was released longest ago, so B pays"
    );
}

#[test]
fn a_pool_that_cannot_be_freed_returns_the_leafs_refusal() {
    // The loop that evicts has to end. Every entry held and no room is the
    // only shape in which it cannot, so the refusal comes back out as a leaf
    // error rather than spinning. A load floors the pool at one
    // envelope-wide item, so reaching this needs two items in one prefill
    // batch that together overflow it.
    let (leaf, compute) = stub_compute(StubLeaf::with_tokens([]).with_media_pool(20));
    let a = multimodal(40, vec![image_of(1, 10, 20)]);
    let b = multimodal(40, vec![image_of(2, 10, 20)]);
    // Both jobs in one batch: the first keeps its hold while the second asks.
    let batch = [multimodal_job(1, &a, 0, 18), multimodal_job(2, &b, 0, 18)];
    let failed = compute.prefill_step(&batch).unwrap_err();
    assert!(format!("{failed}").contains("-2"), "the leaf's code reaches the caller: {failed}");
    assert_eq!(compute.live_media(), 0, "the unwind gave every hold back");
    assert_eq!(leaf.calls.lock().unwrap().media_encoded, [10]);
}

#[test]
fn a_request_that_publishes_a_block_and_chains_over_it_keeps_both_heads_claimable() {
    // GitHub #187 x #188: one request publishes its system block, then chains
    // its opener's page over it. A later burst member claims the *block* —
    // and must be stood up on the block, not on the longer chained head the
    // same publisher also owns.
    let leaf = Arc::new(StubLeaf::with_tokens([]));
    let model = Arc::new(Model::load(leaf.clone()).unwrap());
    let compute = RuntimeCompute::new(model, 99);
    let job = |request, tokens: Vec<u32>, start, publish: Option<u32>| PrefillJob {
        request,
        tokens,
        context_tokens: 32,
        start_position: start,
        params: DecodeParams::default(),
        shared_prefix: None,
        publish_prefix: publish.map(|tokens| RetainedAt { tokens, slot: tokens / 4 }),
        checkpoint: None,
        capture_checkpoint: None,
        multimodal: None,
        readout: None,
        permitted: None,
        attention: None,
};
    compute.prefill_step(&[job(1, vec![1, 2, 3, 4], 0, Some(4))]).unwrap();
    compute.prefill_step(&[job(1, vec![5, 6, 7, 8], 4, Some(8))]).unwrap();
    assert_eq!(leaf.calls.lock().unwrap().prefixes_released, 0, "neither head was dropped");

    compute
        .prefill_step(&[PrefillJob {
            shared_prefix: Some(ignis_core::scheduler::SharedPrefixClaim { publisher: 1, tokens: 4 }),
            ..job(2, vec![50, 51], 4, None)
        }])
        .unwrap();
    assert_eq!(leaf.calls.lock().unwrap().claimed_prefixes, vec![4], "the block, not the chain");

    compute.release_prefix(1, 8);
    assert_eq!(leaf.calls.lock().unwrap().prefixes_released, 1);
    compute.release_prefix(1, 4);
    assert_eq!(leaf.calls.lock().unwrap().prefixes_released, 2, "each head is released by name");
}

#[test]
fn a_spilled_prefix_comes_back_under_its_own_name_and_keeps_its_blob() {
    // GitHub #190: spilled while its device handle still exists, released
    // the usual way, then published again from the blob by a carrier
    // sequence that goes as soon as the prefix exists.
    let leaf = Arc::new(StubLeaf::with_tokens([]));
    let model = Arc::new(Model::load(leaf.clone()).unwrap());
    let compute = RuntimeCompute::new(model, 99);
    compute
        .prefill_step(&[PrefillJob {
            request: 1,
            tokens: vec![1, 2, 3, 4],
            context_tokens: 32,
            start_position: 0,
            params: DecodeParams::default(),
            shared_prefix: None,
            publish_prefix: Some(RetainedAt { tokens: 4, slot: 3 }),
            checkpoint: None,
            capture_checkpoint: None,
            multimodal: None,
            readout: None,
            permitted: None,
            attention: None,
}])
        .unwrap();
    compute.release(1);

    assert!(compute.restore_prefix(1, 4, 5).is_err(), "nothing spilled yet");
    compute.spill_prefix(1, 4).unwrap();
    assert_eq!(compute.spilled_prefixes(), 1);
    assert!(compute.restore_prefix(1, 4, 5).is_err(), "it is still on the device");
    compute.release_prefix(1, 4);
    assert_eq!(compute.live_prefixes(), 0);

    let sequences_before = leaf.calls.lock().unwrap().sequences_released;
    compute.restore_prefix(1, 4, 5).unwrap();
    assert_eq!(
        leaf.calls.lock().unwrap().publish_slots,
        vec![3, 5],
        "the carrier publishes into the slot the scheduler named for the return (GitHub #215)"
    );
    assert_eq!(compute.live_prefixes(), 1, "the prefix is back");
    assert_eq!(compute.live_sequences(), 0, "and its carrier is gone");
    assert_eq!(leaf.calls.lock().unwrap().sequences_released, sequences_before + 1);
    assert_eq!(leaf.calls.lock().unwrap().restores, 1);
    assert_eq!(compute.spilled_prefixes(), 1, "the blob stays");

    compute
        .prefill_step(&[PrefillJob {
            request: 2,
            tokens: vec![9],
            context_tokens: 32,
            start_position: 4,
            params: DecodeParams::default(),
            shared_prefix: Some(ignis_core::scheduler::SharedPrefixClaim { publisher: 1, tokens: 4 }),
            publish_prefix: None,
            checkpoint: None,
            capture_checkpoint: None,
            multimodal: None,
            readout: None,
            permitted: None,
            attention: None,
}])
        .unwrap();
    assert_eq!(leaf.calls.lock().unwrap().claimed_prefixes, vec![4], "claimable by name");

    compute.discard_spilled_prefix(1, 4);
    assert_eq!(compute.spilled_prefixes(), 0);
}

// ── the readout seam (GitHub #237, ADR 0034) ─────────────────────────────

/// A prefill job that reads `answers` out at its last position.
fn readout_job(request: u64, tokens: Vec<u32>, answers: &[u32]) -> PrefillJob {
    PrefillJob {
        readout: Some(std::sync::Arc::from(answers.to_vec())),
        tokens,
        ..prefill(request, None)
    }
}

#[test]
fn a_readout_job_is_gathered_on_the_leafs_side_of_the_seam() {
    let leaf = Arc::new(StubLeaf::with_tokens([7]));
    let model = Arc::new(Model::load(leaf.clone()).expect("stub model loads"));
    let compute = RuntimeCompute::new(model, 99);

    let outcomes = compute
        .prefill_step(&[readout_job(1, vec![4, 5], &[2, 5])])
        .expect("a readout prefill succeeds");

    assert_eq!(
        leaf.calls.lock().unwrap().prefill_logit_buffers,
        vec![Some(STUB_VOCAB as usize)],
        "the leaf is handed a buffer exactly as wide as its output head"
    );
    let readout = outcomes[0].readout.as_ref().expect("the readout comes back");
    assert_eq!(
        readout.logits,
        vec![1.0, 2.5],
        "the named columns of the stub's ramp, in the order asked for"
    );
    assert_eq!(
        readout.full_argmax,
        STUB_VOCAB - 1,
        "the unrestricted winner is the ramp's top column, which no answer named"
    );
    let mass = readout.answer_mass();
    assert!(
        (0.0..=1.0).contains(&mass),
        "answer mass is a probability, got {mass}"
    );
    assert!(
        mass < 0.1,
        "the ramp keeps most of its mass outside the two answers, got {mass}"
    );
    assert_eq!(readout.winner(), Some(1), "column 5 outranks column 2");
}

#[test]
fn a_job_that_asks_for_no_readout_is_handed_no_buffer() {
    let leaf = Arc::new(StubLeaf::with_tokens([7]));
    let model = Arc::new(Model::load(leaf.clone()).expect("stub model loads"));
    let compute = RuntimeCompute::new(model, 99);

    let outcomes = compute.prefill_step(&[prefill(1, None)]).expect("prefill");

    assert_eq!(
        leaf.calls.lock().unwrap().prefill_logit_buffers,
        vec![None],
        "no readout was asked for, so the 248,320-wide buffer is never allocated"
    );
    assert!(outcomes[0].readout.is_none(), "and none comes back");
}

#[test]
fn one_readout_job_in_a_batch_does_not_give_the_others_one() {
    let leaf = Arc::new(StubLeaf::with_tokens([7]));
    let model = Arc::new(Model::load(leaf.clone()).expect("stub model loads"));
    let compute = RuntimeCompute::new(model, 99);

    let outcomes = compute
        .prefill_step(&[
            prefill(1, None),
            readout_job(2, vec![4, 5], &[3]),
            prefill(3, None),
        ])
        .expect("a mixed batch succeeds");

    assert_eq!(
        leaf.calls.lock().unwrap().prefill_logit_buffers,
        vec![None, Some(STUB_VOCAB as usize), None],
        "a decision batched beside ordinary prefills costs only itself"
    );
    assert!(outcomes[0].readout.is_none());
    assert_eq!(
        outcomes[1].readout.as_ref().expect("the decision's readout").logits,
        vec![1.5]
    );
    assert!(outcomes[2].readout.is_none());
}

#[test]
fn a_readout_job_with_nothing_to_prefill_fails_loudly() {
    // A chunk with no tokens runs no forward pass, so there are no logits
    // at this position to read. Returning `None` here would hand the caller
    // a decision with no answer and no error; the batch fails instead.
    let leaf = Arc::new(StubLeaf::with_tokens([7]));
    let model = Arc::new(Model::load(leaf.clone()).expect("stub model loads"));
    let compute = RuntimeCompute::new(model, 99);

    let error = compute
        .prefill_step(&[readout_job(1, Vec::new(), &[2, 5])])
        .expect_err("an empty readout chunk is refused");
    assert!(
        matches!(
            error,
            ComputeError::Kernel(code) if code == ignis_core::scheduler::READOUT_WITHOUT_TOKENS
        ),
        "the refusal names itself rather than passing for a kernel fault: {error}"
    );
    assert_eq!(
        compute.live_sequences(),
        0,
        "the failed batch leaves nothing behind for the retry to prefill twice"
    );
    assert!(
        leaf.calls.lock().unwrap().prefill_logit_buffers.is_empty(),
        "the leaf was never called at all"
    );
}

#[test]
fn an_answer_token_past_the_output_head_does_not_take_the_prefill_down() {
    let leaf = Arc::new(StubLeaf::with_tokens([7]));
    let model = Arc::new(Model::load(leaf.clone()).expect("stub model loads"));
    let compute = RuntimeCompute::new(model, 99);

    let outcomes = compute
        .prefill_step(&[readout_job(1, vec![4, 5], &[4, STUB_VOCAB + 100])])
        .expect("the prefill itself is unaffected");

    let readout = outcomes[0].readout.as_ref().expect("a readout");
    assert_eq!(readout.logits[0], 2.0);
    assert!(
        readout.logits[1].is_infinite() && readout.logits[1] < 0.0,
        "a column the head does not have loses every comparison: {:?}",
        readout.logits
    );
    assert_eq!(readout.winner(), Some(0));
}

// ── GitHub #260: the attention readout ───────────────────────────────────

const HEAD: PointingHead = PointingHead {
    gqa_ordinal: 9,
    query_head: 10,
};

fn attention_job(request: u64, prompt: &Arc<Multimodal>, start: u32, len: u32) -> PrefillJob {
    PrefillJob {
        attention: Some(AttentionQuery {
            head: HEAD,
            key_begin: 10,
            key_count: 20,
            set: None,
        }),
        ..multimodal_job(request, prompt, start, len)
    }
}

#[test]
fn an_attention_readout_comes_back_as_one_score_per_key_of_its_span() {
    let (leaf, compute) = stub_compute(StubLeaf::with_tokens([]));
    let prompt = multimodal(40, vec![image(10, 20)]);
    let outcomes = compute
        .prefill_step(&[multimodal_job(1, &prompt, 0, 30), attention_job(2, &prompt, 0, 40)])
        .expect("a batch with a head point succeeds");

    assert_eq!(outcomes[0].attention, None, "the job that asked for none gets none");
    let attention = outcomes[1].attention.as_ref().expect("the head point's scores come back");
    assert_eq!(&*attention.scores, (10..30).map(|k| k as f32).collect::<Vec<_>>().as_slice());
    assert_eq!(attention.set_argmax, None, "it named no head set");
    assert_eq!(outcomes[1].readout, None, "and no logits readout rides with it");
    let calls = leaf.calls.lock().unwrap();
    assert_eq!(
        calls.attention_asked,
        [
            None,
            Some(AttentionQuery {
                head: HEAD,
                key_begin: 10,
                key_count: 20,
                set: None,
            })
        ],
        "only the job that asked hands the leaf a readout"
    );
    assert_eq!(calls.prefill_logit_buffers, [None, None], "a head point asks for no logits row");
}

/// Spec 14: a readout naming a head set hands the leaf the set whole, one
/// argmax slot per head, and exactly those indices cross back beside the
/// pointing head's scores.
#[test]
fn a_head_sets_argmax_comes_back_one_key_per_head() {
    let (leaf, compute) = stub_compute(StubLeaf::with_tokens([]));
    let prompt = multimodal(40, vec![image(10, 20)]);
    let set = SetQuery {
        heads: Arc::from(vec![HEAD, PointingHead { gqa_ordinal: 15, query_head: 18 }, HEAD]),
        excluded: Arc::from(vec![0, 19]),
        grid_cols: 5,
    };
    let job = PrefillJob {
        attention: Some(AttentionQuery {
            head: HEAD,
            key_begin: 10,
            key_count: 20,
            set: Some(set.clone()),
        }),
        ..multimodal_job(1, &prompt, 0, 40)
    };
    let outcomes = compute.prefill_step(&[job]).expect("a head set reads");
    let attention = outcomes[0].attention.as_ref().expect("read");
    assert_eq!(attention.scores.len(), 20);
    assert_eq!(attention.set_argmax.as_deref(), Some(&[0, 2, 4][..]), "one key per head, in the set's order");
    assert_eq!(attention.set_peak.as_deref(), Some(&[100.0, 102.0, 104.0][..]), "each peak's score");
    // Spec 15: four neighbours per head, and a NaN from the leaf crosses as
    // `None`. The span is 20 keys on a 5-column grid, so key 0 is the top
    // left corner: no left neighbour and none above.
    let around = attention.set_neighbours.as_deref().expect("four neighbours per head");
    assert_eq!(around.len(), 12);
    assert_eq!(&around[..4], &[None, Some(1.0), None, Some(5.0)], "key 0 is the corner");
    assert_eq!(&around[4..8], &[Some(1.0), Some(3.0), None, Some(7.0)], "key 2 is the top row");
    // Key 4 is column 4 of five: the top right corner, with nothing to its
    // right and nothing above it.
    assert_eq!(&around[8..], &[Some(3.0), None, None, Some(9.0)], "key 4 is the top right corner");
    let calls = leaf.calls.lock().unwrap();
    let asked = calls.attention_asked[0].as_ref().expect("the leaf was handed the readout");
    assert_eq!(asked.set.as_ref(), Some(&set), "with the set, heads and excluded keys alike");
}

/// Spec 13 acceptance 4, the host's half: a span that asks for no attention
/// readout is handed no score buffer at all.
#[test]
fn a_span_that_asks_for_no_attention_readout_is_handed_nothing() {
    let (leaf, compute) = stub_compute(StubLeaf::with_tokens([]));
    let prompt = multimodal(40, vec![image(10, 20)]);
    compute.prefill_step(&[multimodal_job(1, &prompt, 0, 40)]).unwrap();
    assert_eq!(leaf.calls.lock().unwrap().attention_asked, [None]);
}

/// A leaf that could not read the keys the layer's attention read leaves
/// the readout unread. The chunk stands — the sequence really did prefill —
/// and no scores cross the seam, so the scheduler fails the question.
#[test]
fn an_attention_readout_the_leaf_could_not_read_crosses_nothing() {
    let (leaf, compute) = stub_compute(StubLeaf::with_tokens([]));
    leaf.calls.lock().unwrap().attention_unreadable = true;
    let prompt = multimodal(40, vec![image(10, 20)]);
    let outcomes = compute
        .prefill_step(&[attention_job(1, &prompt, 0, 40)])
        .expect("an unread readout does not fail the batch");
    assert_eq!(outcomes[0].attention, None);
    assert_eq!(compute.live_sequences(), 1, "the prefilled sequence is kept");
}

/// The span an attention readout reads is an image's, so a job without a
/// multimodal span asking for one is incoherent — the server never renders
/// one — and the batch fails with its own code rather than answering.
#[test]
fn an_attention_readout_without_an_image_fails_loudly() {
    let (leaf, compute) = stub_compute(StubLeaf::with_tokens([]));
    let job = PrefillJob {
        attention: Some(AttentionQuery {
            head: HEAD,
            key_begin: 0,
            key_count: 2,
            set: None,
        }),
        ..prefill(1, None)
    };
    let error = compute.prefill_step(&[job]).expect_err("refused");
    assert!(
        matches!(
            error,
            ComputeError::Kernel(code) if code == ignis_core::scheduler::ATTENTION_WITHOUT_IMAGE
        ),
        "{error}"
    );
    assert_eq!(compute.live_sequences(), 0);
    assert!(leaf.calls.lock().unwrap().prefill_positions.is_empty(), "the leaf was never called");
}
