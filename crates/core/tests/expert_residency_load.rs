//! The host expert pool filled from an artifact (spec flash-next/03,
//! acceptance 1, GitHub #301): on the Flash-Next binder's fixture artifact,
//! every expert projection lands in the pool at the offset the expert index
//! lays out, byte for byte what the file holds, and the residency catalog
//! read off the index carries each projection's K and its class's record
//! bytes.

use ignis_artifact::flash_next::{self, FlashNextGeometry, Projection as ArtifactProjection, fixture};
use ignis_artifact::Reader;
use ignis_core::residency::load::{catalog, fill_expert_pool, pool_layout};
use ignis_core::residency::{KClass, Projection, ProjectionId};

#[test]
fn every_expert_projection_lands_in_the_pool_where_the_index_puts_it() {
    let artifact = fixture::build("residency-pool").expect("fixture artifact");
    let reader = Reader::open(&artifact.path).expect("open");
    let plan = flash_next::bind(&reader, &FlashNextGeometry::fixture()).expect("bind");
    let index = &plan.experts;

    let layout = pool_layout(index);
    let keys = index.layers() * index.experts_per_layer() * 2;
    assert_eq!(layout.k2.len(), keys);
    assert_eq!(layout.offsets.len(), keys);
    assert_eq!(layout.bytes, plan.plan.expert_pool_capacity_bytes);

    let mut pool = vec![0u8; layout.bytes as usize];
    // 2 layers < READ_WORKERS: exercises workers left with no layer too.
    let read = fill_expert_pool(&artifact.path, index, &mut pool).expect("fill");
    let file_bytes = std::fs::read(&artifact.path).expect("read");

    let cat = catalog(index).expect("catalog");
    let mut total = 0;
    for layer in 0..index.layers() {
        for expert in 0..index.experts_per_layer() {
            for (p, projection) in ArtifactProjection::ALL.into_iter().enumerate() {
                let r = index.get(layer, expert, projection).expect("indexed");
                let at = r.pool_offset as usize;
                let from = r.file_offset as usize;
                assert_eq!(
                    &pool[at..at + r.bytes as usize],
                    &file_bytes[from..from + r.bytes as usize],
                    "layer {layer} expert {expert} projection {p}"
                );
                let key = (layer * index.experts_per_layer() + expert) * 2 + p;
                assert_eq!(layout.k2[key], r.k.k2());
                assert_eq!(layout.offsets[key], r.pool_offset);
                let id = ProjectionId::new(
                    layer as u16,
                    expert as u16,
                    if p == 0 { Projection::GateUp } else { Projection::Down },
                );
                assert_eq!(u32::from(r.k.k2()), cat.class_of(id).k.half_bits());
                assert_eq!(cat.bytes(id), r.bytes);
                total += r.bytes;
            }
        }
    }
    assert_eq!(read, total);
    // The fixture's K map spans several classes, as the real one does.
    let used = cat.class_counts().iter().filter(|&&n| n > 0).count();
    assert!(used >= 3, "{used} of {} classes used", KClass::COUNT);
}
