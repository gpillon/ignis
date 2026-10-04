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
