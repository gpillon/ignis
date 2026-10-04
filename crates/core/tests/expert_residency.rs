//! Expert residency's CPU policy model (spec flash-next/03, GitHub #301): fed
//! pool capacities, class sizes and a routing trace, it produces the exact
//! hit/miss/evict/prefetch sequence the GPU implementation is later tested
//! against. Every test here drives the model through a trace and checks what
//! residency did — which projections were resident when the kernel ran and
//! how many bytes moved — never a slot index.

use ignis_core::residency::{ExpertCatalog, KBits, KClass, Projection, ProjectionId};

/// Two layers of four experts: gate/up at K = 2, 2.5, 3, 4 by expert id,
/// down at 2 everywhere. Class sizes are round numbers so a byte count reads
/// as a sum of projections.
fn small_catalog() -> ExpertCatalog {
    let ks = [KBits::K2, KBits::K2_5, KBits::K3, KBits::K4];
    let map = (0..2)
        .flat_map(|_| ks.iter().map(|&k| (k, KBits::K2)))
        .collect();
    ExpertCatalog::new(2, 4, map, slot_bytes()).expect("catalog")
}

/// gate/up K2, K2.5, K3, K4, then down K2, K2.5, K3, K4.
fn slot_bytes() -> [u64; 8] {
    [200, 250, 300, 400, 100, 125, 150, 200]
}

fn gate_up(layer: u16, expert: u16) -> ProjectionId {
    ProjectionId::new(layer, expert, Projection::GateUp)
}

fn down(layer: u16, expert: u16) -> ProjectionId {
    ProjectionId::new(layer, expert, Projection::Down)
}

#[test]
fn the_eight_classes_have_fixed_spellings_in_a_fixed_order() {
    let names: Vec<_> = KClass::ALL.iter().map(|c| c.as_str()).collect();
    assert_eq!(
        names,
        [
            "gate_up_k2",
            "gate_up_k2_5",
            "gate_up_k3",
            "gate_up_k4",
            "down_k2",
            "down_k2_5",
            "down_k3",
            "down_k4",
        ]
    );
    for (i, class) in KClass::ALL.iter().enumerate() {
        assert_eq!(class.index(), i);
    }
}

#[test]
fn a_projection_s_class_and_bytes_come_from_its_shape_and_its_expert_s_k() {
    let catalog = small_catalog();
    assert_eq!(
        catalog.class_of(gate_up(1, 2)),
        KClass::new(Projection::GateUp, KBits::K3)
    );
    assert_eq!(catalog.bytes(gate_up(1, 2)), 300);
    assert_eq!(catalog.class_of(down(1, 2)), KClass::new(Projection::Down, KBits::K2));
    assert_eq!(catalog.bytes(down(0, 3)), 100);
    // Every projection once: 2 x (200 + 250 + 300 + 400) + 16 x 100.
    assert_eq!(catalog.total_bytes(), 2 * 1150 + 8 * 100);
    assert_eq!(catalog.class_counts(), [2, 2, 2, 2, 8, 0, 0, 0]);
}

#[test]
fn a_k_map_of_the_wrong_length_is_refused() {
    let short = vec![(KBits::K2, KBits::K2); 7];
    assert!(ExpertCatalog::new(2, 4, short, slot_bytes()).is_err());
}

// ---- the policy model -------------------------------------------------------

use ignis_core::residency::{
    Admission, LayerStep, PolicyConfig, ResidencyModel, StepError, StepOutcome,
};

/// Capacity per class, indexed like `KClass::ALL`.
fn model(capacity: [u32; 8], prefetch_width: usize) -> ResidencyModel {
    ResidencyModel::new(
        small_catalog(),
        PolicyConfig {
            capacity,
            prefetch_width,
            prefetch_budget_bytes: None,
        },
    )
}

/// Enough room everywhere that nothing is ever evicted.
const ROOMY: [u32; 8] = [8, 8, 8, 8, 16, 16, 16, 16];

/// `ROOMY` with the down K2 class (every down of `small_catalog`) cut to
/// `slots`.
fn tight_downs(slots: u32) -> [u32; 8] {
    let mut capacity = ROOMY;
    capacity[4] = slots;
    capacity
}

fn decode(m: &mut ResidencyModel, layer: u16, selected: &[u16]) -> StepOutcome {
    m.step(&LayerStep::decode(layer, selected)).expect("step")
}

fn prefill(m: &mut ResidencyModel, layer: u16, selected: &[u16]) -> StepOutcome {
    m.step(&LayerStep::prefill(layer, selected)).expect("step")
}

#[test]
fn a_cold_selection_misses_and_moves_its_bytes_and_the_same_selection_again_hits_for_free() {
    let mut m = model(ROOMY, 0);
    let first = decode(&mut m, 0, &[1, 3]);
    assert!(first.hits.is_empty());
    assert_eq!(
        first.misses,
        vec![
            (gate_up(0, 1), Admission::Slot),
            (down(0, 1), Admission::Slot),
            (gate_up(0, 3), Admission::Slot),
            (down(0, 3), Admission::Slot),
        ]
    );
    // gate/up K2.5 + K4, two downs at K2.
    assert_eq!(first.bytes_moved, 250 + 400 + 2 * 100);
    assert!(first.evictions.is_empty());

    let again = decode(&mut m, 0, &[3, 1, 1]);
    assert_eq!(again.hits, vec![gate_up(0, 1), down(0, 1), gate_up(0, 3), down(0, 3)]);
    assert!(again.misses.is_empty());
    assert_eq!(again.bytes_moved, 0);
}

#[test]
fn a_full_class_evicts_its_least_recently_used_projection() {
    let mut m = model(tight_downs(2), 0);
    decode(&mut m, 0, &[0]);
    decode(&mut m, 0, &[1]);
    decode(&mut m, 0, &[0]); // 0 is now more recent than 1
    let step = decode(&mut m, 1, &[2]);
    assert_eq!(step.evictions, vec![down(0, 1)]);
    assert_eq!(step.misses, vec![(gate_up(1, 2), Admission::Slot), (down(1, 2), Admission::Slot)]);
    assert_eq!(m.occupancy()[4], 2);
    // Expert 1's gate/up was never evicted (its class has room), its down was.
    let back = decode(&mut m, 0, &[1]);
    assert_eq!(back.hits, vec![gate_up(0, 1)]);
    assert_eq!(back.misses, vec![(down(0, 1), Admission::Slot)]);
    assert_eq!(back.evictions, vec![down(0, 0)]);
}

#[test]
fn equal_recency_is_broken_by_the_canonical_key_so_the_lower_projection_goes_first() {
    let mut m = model(tight_downs(2), 0);
    decode(&mut m, 0, &[2, 1]); // both downs stamped in one step
    let step = decode(&mut m, 1, &[0]);
    assert_eq!(step.evictions, vec![down(0, 1)]);
}

#[test]
fn a_step_never_evicts_what_it_selected_and_says_so_when_its_class_cannot_hold_it() {
    let mut m = model(tight_downs(2), 0);
    decode(&mut m, 0, &[0]);
    // Down K2 now holds (0, 0) and one free slot; this step needs three downs.
    let err = m.step(&LayerStep::decode(0, &[0, 1, 2])).expect_err("no evictable slot");
    assert_eq!(
        err,
        StepError::NoEvictableSlot {
            class: KClass::new(Projection::Down, KBits::K2),
            layer: 0,
        }
    );
    // Refused whole: nothing moved, and the next step sees the old state.
    let step = decode(&mut m, 0, &[0]);
    assert_eq!(step.hits, vec![gate_up(0, 0), down(0, 0)]);
    assert_eq!(m.occupancy()[4], 1);
}

#[test]
fn an_expert_outside_the_catalog_is_refused() {
    let mut m = model(ROOMY, 0);
    assert_eq!(
        m.step(&LayerStep::decode(0, &[4])).expect_err("out of range"),
        StepError::OutOfCatalog { layer: 0, expert: 4 }
    );
    assert_eq!(
        m.step(&LayerStep::decode(2, &[0])).expect_err("out of range"),
        StepError::OutOfCatalog { layer: 2, expert: 0 }
    );
}

#[test]
fn a_prefetched_projection_used_by_the_next_layer_is_a_hit_that_moved_its_bytes_once() {
    let mut m = model(ROOMY, 16);
    let lane: &[u16] = &[2, 3];
    let step = m.step(&LayerStep::decode(0, &[0]).lookahead(&[lane])).expect("step");
    assert_eq!(
        step.prefetches,
        vec![
            (gate_up(1, 2), Admission::Slot),
            (down(1, 2), Admission::Slot),
            (gate_up(1, 3), Admission::Slot),
            (down(1, 3), Admission::Slot),
        ]
    );
    // Its own two misses, then the four prefetches.
    assert_eq!(step.bytes_moved, 200 + 100 + 300 + 400 + 2 * 100);
    let next = decode(&mut m, 1, &[3]);
    assert_eq!(next.hits, vec![gate_up(1, 3), down(1, 3)]);
    assert_eq!(next.prefetch_hits, vec![gate_up(1, 3), down(1, 3)]);
    assert_eq!(next.bytes_moved, 0);
    let counters = m.counters();
    assert_eq!(counters.prefetch_issued, 4);
    assert_eq!(counters.prefetch_used, 2);
    // A second use is an ordinary hit, not a second prefetch used.
    let third = decode(&mut m, 1, &[3]);
    assert!(third.prefetch_hits.is_empty());
    assert_eq!(m.counters().prefetch_used, 2);
}

#[test]
fn the_prefetch_width_takes_the_top_of_each_lane_s_ranking() {
    let mut m = model(ROOMY, 1);
    let a: &[u16] = &[2, 3];
    let b: &[u16] = &[1, 2];
    let step = m.step(&LayerStep::decode(0, &[0]).lookahead(&[a, b])).expect("step");
    let experts: Vec<_> = step.prefetches.iter().map(|(p, _)| p.expert).collect();
    assert_eq!(experts, vec![1, 1, 2, 2]);
    // The last layer has no next router: its lookahead is ignored.
    let last = m.step(&LayerStep::decode(1, &[0]).lookahead(&[a])).expect("step");
    assert!(last.prefetches.is_empty());
}

#[test]
fn a_decode_step_s_prefetches_stop_at_its_budget_best_ranked_first() {
    let mut m = ResidencyModel::new(
        small_catalog(),
        PolicyConfig {
            capacity: ROOMY,
            prefetch_width: 16,
            prefetch_budget_bytes: Some(350),
        },
    );
    let lane: &[u16] = &[2, 3];
    let step = m.step(&LayerStep::decode(0, &[0]).lookahead(&[lane])).expect("step");
    // Rank order: expert 2's gate/up (K3, 300 B) fits; its down (100 B)
    // would pass 350, and so would everything after it.
    assert_eq!(step.prefetches, vec![(gate_up(1, 2), Admission::Slot)]);
    assert_eq!(step.prefetch_dropped, vec![down(1, 2), gate_up(1, 3), down(1, 3)]);
    // The step's own misses are never budgeted.
    assert_eq!(step.bytes_moved, 200 + 100 + 300);
    // A prefill streams its lookahead whole.
    let step = m.step(&LayerStep::prefill(0, &[1]).lookahead(&[&[3, 0][..]])).expect("step");
    assert_eq!(step.prefetches.len(), 4);
}

#[test]
fn a_prefetch_never_evicts_the_current_step_and_is_dropped_when_nothing_else_can_go() {
    let mut m = model(tight_downs(2), 16);
    decode(&mut m, 1, &[0]); // an old down that a prefetch may evict
    let lane: &[u16] = &[1, 2];
    let step = m.step(&LayerStep::decode(0, &[0]).lookahead(&[lane])).expect("step");
    // Down K2: (0, 0) is this step's and pinned; (1, 0) is old. One prefetched
    // down evicts (1, 0); the other has nowhere to go and is dropped.
    assert_eq!(step.evictions, vec![down(1, 0)]);
    assert!(step.prefetches.contains(&(down(1, 1), Admission::Slot)));
    assert_eq!(step.prefetch_dropped, vec![down(1, 2)]);
}

#[test]
fn a_step_s_misses_spare_what_its_lookahead_names_for_the_next_layer() {
    let mut m = model(tight_downs(2), 16);
    decode(&mut m, 1, &[0]); // down(1, 0): the oldest down
    decode(&mut m, 0, &[1]); // down(0, 1): full now
    let lane: &[u16] = &[0];
    let step = m.step(&LayerStep::decode(0, &[2]).lookahead(&[lane])).expect("step");
    // The lookahead names expert 0 of layer 1, so its old down is spared and
    // the younger down(0, 1) goes; nothing is fetched back.
    assert_eq!(step.evictions, vec![down(0, 1)]);
    assert!(step.prefetches.is_empty());
}

#[test]
fn a_miss_takes_a_protected_projection_only_when_nothing_else_can_go_and_does_not_refetch_it() {
    let mut m = model(tight_downs(1), 16);
    decode(&mut m, 1, &[0]);
    let lane: &[u16] = &[0];
    let step = m.step(&LayerStep::decode(0, &[2]).lookahead(&[lane])).expect("step");
    assert_eq!(step.evictions, vec![down(1, 0)]);
    assert_eq!(step.prefetch_dropped, vec![down(1, 0)]);
    assert!(step.prefetches.is_empty());
}

#[test]
fn a_prefill_miss_takes_a_free_slot_first_then_the_staging_ring_and_never_evicts() {
    let mut m = model(tight_downs(3), 0);
    decode(&mut m, 0, &[0, 1]); // the decode working set: two downs
    let step = prefill(&mut m, 0, &[0, 1, 2, 3]);
    assert!(step.evictions.is_empty());
    assert_eq!(step.hits, vec![gate_up(0, 0), down(0, 0), gate_up(0, 1), down(0, 1)]);
    // Free slots go to the lowest keys first.
    assert_eq!(
        step.misses,
        vec![
            (gate_up(0, 2), Admission::Slot),
            (down(0, 2), Admission::Slot),
            (gate_up(0, 3), Admission::Slot),
            (down(0, 3), Admission::Staging),
        ]
    );
    // Every miss moved its bytes, staged or not.
    assert_eq!(step.bytes_moved, 300 + 400 + 2 * 100);
    // The staged down never entered the cache; the free-slot ones did.
    let after = decode(&mut m, 0, &[2, 3]);
    assert_eq!(after.hits, vec![gate_up(0, 2), down(0, 2), gate_up(0, 3)]);
    assert_eq!(after.misses, vec![(down(0, 3), Admission::Slot)]);
}

#[test]
fn a_long_prefill_leaves_the_decode_working_set_resident() {
    // Room for the working set and nothing more; the prefill touches every
    // expert of both layers, twice.
    let mut m = model(tight_downs(2), 0);
    decode(&mut m, 0, &[1]);
    decode(&mut m, 1, &[2]);
    for _ in 0..2 {
        for layer in 0..2 {
            let step = prefill(&mut m, layer, &[0, 1, 2, 3]);
            assert!(step.evictions.is_empty());
        }
    }
    assert!(decode(&mut m, 0, &[1]).misses.is_empty());
    assert!(decode(&mut m, 1, &[2]).misses.is_empty());
}

#[test]
fn prefill_hits_refresh_recency() {
    let mut m = model(tight_downs(2), 0);
    decode(&mut m, 0, &[0]);
    decode(&mut m, 0, &[1]);
    prefill(&mut m, 0, &[0]); // 0 is now more recent than 1
    let step = decode(&mut m, 1, &[3]);
    assert_eq!(step.evictions, vec![down(0, 1)]);
}

#[test]
fn a_prefill_lookahead_stages_the_next_layer_and_its_use_moves_nothing_more() {
    let mut m = model(tight_downs(0), 16);
    let lane: &[u16] = &[0, 1];
    let step = m.step(&LayerStep::prefill(0, &[3]).lookahead(&[lane])).expect("step");
    assert!(step.prefetches.contains(&(gate_up(1, 0), Admission::Slot)));
    assert!(step.prefetches.contains(&(down(1, 0), Admission::Staging)));
    let next = prefill(&mut m, 1, &[0]);
    assert_eq!(next.hits, vec![gate_up(1, 0), down(1, 0)]);
    assert_eq!(next.prefetch_hits, vec![gate_up(1, 0), down(1, 0)]);
    assert_eq!(next.bytes_moved, 0);
    // The ring is released once its layer ran: the down is gone.
    let later = prefill(&mut m, 1, &[0]);
    assert_eq!(later.misses, vec![(down(1, 0), Admission::Staging)]);
}

#[test]
fn a_warm_start_fills_each_class_hottest_first_and_the_least_hot_goes_first() {
    let mut m = model(tight_downs(2), 0);
    let hottest_first = [down(1, 3), down(0, 0), down(1, 1), gate_up(0, 0)];
    assert_eq!(m.warm_start(&hottest_first), 3);
    assert_eq!(m.occupancy()[4], 2);
    assert_eq!(m.occupancy()[0], 1);
    // down(1, 1) did not fit; the first eviction takes down(0, 0), the less
    // hot of the two that did.
    let step = decode(&mut m, 1, &[1]);
    assert_eq!(step.evictions, vec![down(0, 0)]);
    assert!(step.hits.is_empty());
}

// ---- properties over random traces -----------------------------------------

/// splitmix64: a fixed, dependency-free stream, so every trace below is the
/// same on every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn experts(&mut self, count: usize, experts: u16) -> Vec<u16> {
        (0..count).map(|_| self.below(u64::from(experts)) as u16).collect()
    }
}

const LAYERS: u16 = 3;
const EXPERTS: u16 = 12;
const TOP_K: usize = 2;
const MAX_LANES: usize = 3;
const WIDTH: usize = 2;

fn random_catalog(rng: &mut Rng) -> ExpertCatalog {
    let map = (0..usize::from(LAYERS) * usize::from(EXPERTS))
        .map(|_| {
            (
                KBits::ALL[rng.below(4) as usize],
                KBits::ALL[rng.below(4) as usize],
            )
        })
        .collect();
    ExpertCatalog::new(LAYERS, EXPERTS, map, slot_bytes()).expect("catalog")
}

/// The smallest pools a decode step of `MAX_LANES` lanes with a lookahead of
/// `WIDTH` can never overflow, plus a random margin: tight enough to evict.
fn random_capacity(rng: &mut Rng) -> [u32; 8] {
    let floor = (MAX_LANES * TOP_K + MAX_LANES * WIDTH) as u32;
    std::array::from_fn(|_| floor + rng.below(4) as u32)
}

/// One step of a random trace, owning its lists.
struct TraceStep {
    layer: u16,
    prefill: bool,
    selected: Vec<u16>,
    lookahead: Vec<Vec<u16>>,
}

/// Rounds of decode (1-3 lanes, top-2 each, lookahead width 2) and prefill
/// chunks (a random slice of each layer's experts), every round walking the
/// layers in order.
fn random_trace(rng: &mut Rng, rounds: usize) -> Vec<TraceStep> {
    let mut trace = Vec::new();
    for _ in 0..rounds {
        let prefill = rng.below(5) == 0;
        let lanes = 1 + rng.below(MAX_LANES as u64) as usize;
        for layer in 0..LAYERS {
            let selected = if prefill {
                let n = 1 + rng.below(u64::from(EXPERTS)) as usize;
                rng.experts(n, EXPERTS)
            } else {
                rng.experts(lanes * TOP_K, EXPERTS)
            };
            let lookahead = (0..if prefill { 4 } else { lanes })
                .map(|_| rng.experts(3, EXPERTS))
                .collect();
            trace.push(TraceStep {
                layer,
                prefill,
                selected,
                lookahead,
            });
        }
    }
    trace
}

fn run(m: &mut ResidencyModel, s: &TraceStep) -> StepOutcome {
    let lanes: Vec<&[u16]> = s.lookahead.iter().map(|l| l.as_slice()).collect();
    let step = if s.prefill {
        LayerStep::prefill(s.layer, &s.selected)
    } else {
        LayerStep::decode(s.layer, &s.selected)
    };
    m.step(&step.lookahead(&lanes)).expect("pools sized for every step")
}

fn selected_projections(s: &TraceStep) -> std::collections::BTreeSet<ProjectionId> {
    s.selected
        .iter()
        .flat_map(|&e| [gate_up(s.layer, e), down(s.layer, e)])
        .collect()
}

#[test]
fn over_random_traces_the_model_keeps_every_promise_of_its_contract() {
    for seed in 0..40 {
        let mut rng = Rng(seed);
        let catalog = random_catalog(&mut rng);
        let capacity = random_capacity(&mut rng);
        let mut m = ResidencyModel::new(
            catalog.clone(),
            PolicyConfig {
                capacity,
                prefetch_width: WIDTH,
                prefetch_budget_bytes: None,
            },
        );
        let trace = random_trace(&mut rng, 60);
        let mut previous: Option<StepOutcome> = None;
        for s in &trace {
            let out = run(&mut m, s);
            let selected = selected_projections(s);

            // Every selected projection is a hit or a miss, never both.
            let mut seen: Vec<ProjectionId> = out.hits.clone();
            seen.extend(out.misses.iter().map(|(p, _)| *p));
            seen.sort();
            assert_eq!(seen, selected.iter().copied().collect::<Vec<_>>(), "seed {seed}");

            // No eviction of a pinned projection: nothing the step selected
            // or prefetched is given up by that same step.
            for evicted in &out.evictions {
                assert!(!selected.contains(evicted), "seed {seed}: evicted a selected {evicted:?}");
                assert!(
                    !out.prefetches.iter().any(|(p, _)| p == evicted),
                    "seed {seed}: evicted its own prefetch {evicted:?}"
                );
            }
            // A prefill evicts nothing.
            if s.prefill {
                assert!(out.evictions.is_empty(), "seed {seed}");
            }

            // Capacity is never exceeded.
            let occupancy = m.occupancy();
            for class in 0..8 {
                assert!(occupancy[class] <= capacity[class], "seed {seed}: class {class}");
            }

            // A hit costs no bytes: what moved is exactly the misses and
            // the prefetches.
            let expected: u64 = out
                .misses
                .iter()
                .chain(out.prefetches.iter())
                .map(|(p, _)| catalog.bytes(*p))
                .sum();
            assert_eq!(out.bytes_moved, expected, "seed {seed}");

            // Prefetch-then-use is a hit: whatever the previous step
            // prefetched for this layer and this step selects, hits.
            if let Some(prev) = &previous {
                for (p, _) in &prev.prefetches {
                    if p.layer == s.layer && selected.contains(p) {
                        assert!(out.hits.contains(p), "seed {seed}: {p:?}");
                        assert!(out.prefetch_hits.contains(p), "seed {seed}: {p:?}");
                    }
                }
            }
            previous = Some(out);
        }
        let counters = m.counters();
        assert!(counters.prefetch_used <= counters.prefetch_issued, "seed {seed}");
    }
}

#[test]
fn the_same_trace_gives_the_same_sequence_every_time() {
    let mut rng = Rng(7);
    let catalog = random_catalog(&mut rng);
    let capacity = random_capacity(&mut rng);
    let trace = random_trace(&mut rng, 80);
    let replay = || {
        let mut m = ResidencyModel::new(
            catalog.clone(),
            PolicyConfig {
                capacity,
                prefetch_width: WIDTH,
                prefetch_budget_bytes: None,
            },
        );
        trace.iter().map(|s| run(&mut m, s)).collect::<Vec<_>>()
    };
    assert_eq!(replay(), replay());
}

/// A plain per-class LRU written as directly as possible — a list per class,
/// most recent last, scanned linearly — for decode without prefetch. The
/// model must produce the same hits, misses and evictions.
#[test]
fn decode_without_prefetch_is_a_plain_per_class_lru() {
    for seed in 100..130 {
        let mut rng = Rng(seed);
        let catalog = random_catalog(&mut rng);
        let capacity = random_capacity(&mut rng);
        let mut m = ResidencyModel::new(
            catalog.clone(),
            PolicyConfig {
                capacity,
                prefetch_width: 0,
                prefetch_budget_bytes: None,
            },
        );
        let mut lists: Vec<Vec<ProjectionId>> = vec![Vec::new(); 8];
        for s in random_trace(&mut rng, 60).iter().filter(|s| !s.prefill) {
            let out = m.step(&LayerStep::decode(s.layer, &s.selected)).expect("step");
            let selected = selected_projections(s);
            let (mut hits, mut misses, mut evictions) = (Vec::new(), Vec::new(), Vec::new());
            for &p in &selected {
                if lists[catalog.class_of(p).index()].contains(&p) {
                    hits.push(p);
                } else {
                    misses.push((p, Admission::Slot));
                }
            }
            for (class, list) in lists.iter_mut().enumerate() {
                // This step's projections of the class, in key order, go to
                // the back; the oldest of the rest make room at the front.
                let step: Vec<ProjectionId> = selected
                    .iter()
                    .copied()
                    .filter(|&p| catalog.class_of(p).index() == class)
                    .collect();
                list.retain(|p| !step.contains(p));
                let over = (list.len() + step.len()).saturating_sub(capacity[class] as usize);
                evictions.extend(list.drain(..over));
                list.extend(step);
            }
            evictions.sort();
            assert_eq!(out.hits, hits, "seed {seed}");
            assert_eq!(out.misses, misses, "seed {seed}");
            assert_eq!(out.evictions, evictions, "seed {seed}");
        }
    }
}

// ---- metrics ----------------------------------------------------------------

#[test]
fn the_metric_families_are_fixed_and_carry_what_the_model_counted() {
    use ignis_core::residency::FAMILIES;
    let names: Vec<_> = FAMILIES.iter().map(|f| f.name).collect();
    assert_eq!(
        names,
        [
            "ignis_expert_cache_hits_total",
            "ignis_expert_cache_misses_total",
            "ignis_expert_prefetches_issued_total",
            "ignis_expert_prefetches_used_total",
            "ignis_expert_bytes_moved_total",
            "ignis_expert_residency_stall_seconds_total",
            "ignis_expert_cache_slots",
        ]
    );
    for family in FAMILIES {
        assert_eq!(family.name.ends_with("_total"), family.kind == "counter", "{}", family.name);
    }

    let mut m = model(tight_downs(2), 16);
    let lane: &[u16] = &[1];
    m.step(&LayerStep::decode(0, &[0]).lookahead(&[lane])).expect("step");
    decode(&mut m, 1, &[1]);
    prefill(&mut m, 0, &[2]);
    let samples = m.counters().samples(&m.config().capacity, &m.occupancy());
    // 8 hit classes + 8 x 2 miss phases + 2 prefetch + 2 byte phases + 1
    // stall + 8 x 2 slot states, zeros included.
    assert_eq!(samples.len(), 45);
    let value = |name: &str, labels: &[(&str, &str)]| {
        samples
            .iter()
            .find(|s| s.name == name && s.labels.as_slice() == labels)
            .unwrap_or_else(|| panic!("{name} {labels:?}"))
            .value
    };
    // Decode: gate/up K2 + down of expert 0, prefetch of expert 1 (K2.5 +
    // down) used by the second step. Prefill: expert 2 (K3 + down).
    assert_eq!(value("ignis_expert_cache_hits_total", &[("class", "gate_up_k2_5")]), 1.0);
    assert_eq!(value("ignis_expert_cache_hits_total", &[("class", "down_k2")]), 1.0);
    assert_eq!(
        value("ignis_expert_cache_misses_total", &[("class", "down_k2"), ("phase", "decode")]),
        1.0
    );
    assert_eq!(
        value("ignis_expert_cache_misses_total", &[("class", "down_k2"), ("phase", "prefill")]),
        1.0
    );
    assert_eq!(value("ignis_expert_prefetches_issued_total", &[]), 2.0);
    assert_eq!(value("ignis_expert_prefetches_used_total", &[]), 2.0);
    assert_eq!(value("ignis_expert_bytes_moved_total", &[("phase", "decode")]), 650.0);
    assert_eq!(value("ignis_expert_bytes_moved_total", &[("phase", "prefill")]), 400.0);
    assert_eq!(value("ignis_expert_residency_stall_seconds_total", &[]), 0.0);
    assert_eq!(
        value("ignis_expert_cache_slots", &[("class", "down_k2"), ("state", "capacity")]),
        2.0
    );
    // Downs of experts 0 and 1, then the prefill's down found a full class
    // and was staged: still two.
    assert_eq!(
        value("ignis_expert_cache_slots", &[("class", "down_k2"), ("state", "in_use")]),
        2.0
    );
}
