//! Flash-Next's vision companion container (ADR 0087, flash-next-layout.md
//! §14): the checkpoint's vision tower and its merger, beside the main
//! container, bound only when vision is on.
//!
//! The tower is encoded with the 27B tower's recipe: the same objects under
//! the same names, each object kind in the same format
//! ([`crate::nvfp4_27b::vision_tower`]), at Flash-Next's widths, where the
//! merger writes 2,560 columns (the text model's hidden width), not the
//! 27B's 5,120. Its identity is its own ([`VISION_MODEL_ID`]), and its
//! sidecar pins the one main container it belongs to (`pair.main`, the MTP
//! companion's shape, flash-next-layout.md §13.2).

use super::{fail, VISION_MODEL_ID};
use crate::binder::{Binder, MaterializationPlan, ObjectHandle};
use crate::nvfp4_27b::{vision_tower, Entry, VisionTower};
use crate::{ArtifactIdentity, Object, Reader, Result, VisionResidency};

/// The merger's output object, whose rows are the width the tower writes.
const MERGER_OUTPUT: &str = "vision/merger/fc2";

/// The companion's identity: the vision `model_id` and the build's weights
/// id (the 27B tower's grouped formats, from revision `de4b8e4`).
pub fn identity() -> ArtifactIdentity {
    ArtifactIdentity { model_id: VISION_MODEL_ID.into(), weights_id: "vision-w8q4q5q6-bf16-de4b8e4".into() }
}

/// Flash-Next's tower (the checkpoint's `vision_config`): 1,152 wide, 4,304
/// in the MLP, 16 x 16 patches over 2 frames of 3 channels, a 48 x 48
/// position table, the merger writing the text model's 2,560.
pub fn geometry() -> VisionTower {
    VisionTower { hidden: 1152, intermediate: 4304, patch: 3 * 2 * 16 * 16, positions: 48 * 48, out_hidden: 2560 }
}

/// The CPU tests' narrow tower (`tools/flash-next-vision`'s synthetic
/// checkpoint): every object of [`geometry`], each a few KB, the merger's
/// output narrower than its input as Flash-Next's is.
pub fn fixture_geometry() -> VisionTower {
    VisionTower { hidden: 32, intermediate: 64, patch: 3 * 2 * 4 * 4, positions: 4 * 4, out_hidden: 64 }
}

/// The companion's objects in container order, at `tower`: what the packer
/// holds the work tree to. At [`geometry`] they are the objects a load binds,
/// [`crate::inventory::vision_scope`]`(2560)`, which a test holds them to.
pub fn entries(tower: &VisionTower) -> Vec<Entry> {
    vision_tower(tower)
}

/// The companion's sidecar `pair.main` against the opened main container
/// `main`: `Ok` when its `model_id`, `weights_id` and `content_hash` are the
/// main container's, else the first field that differs, both values named.
pub fn check_pair(sidecar: &serde_json::Value, main: &Reader) -> Result<()> {
    super::check_companion_pair(sidecar, main, "vision")
}

/// Refuse a container that is not a vision companion, by its `model_id`
/// (the main container or the MTP companion handed in its place).
pub fn check_identity(reader: &Reader) -> Result<()> {
    let model_id = &reader.identity().model_id;
    if model_id != VISION_MODEL_ID {
        return Err(super::fail(format!("artifact model_id {model_id} is not a vision companion ({VISION_MODEL_ID})")));
    }
    Ok(())
}

/// The rows the companion's merger writes, as its directory records them:
/// the first dimension of `vision/merger/fc2` (GitHub #557). What a load
/// holds against the text model's hidden width before it binds anything.
pub fn merger_width(reader: &Reader) -> Result<u64> {
    match reader.find(MERGER_OUTPUT) {
        Some(Object::Tensor(tensor)) if !tensor.shape.is_empty() => Ok(tensor.shape[0]),
        _ => Err(fail(format!("the vision companion has no {MERGER_OUTPUT} tensor"))),
    }
}

/// What [`bind`] placed: every object of the tower, in [`entries`] order.
pub struct VisionPlan {
    pub plan: MaterializationPlan,
    pub handles: Vec<ObjectHandle>,
}

/// Bind the companion (GitHub #557): its identity, then every object of
/// `tower`'s [`entries`] with its exact format, layout and shape, placed
/// where `residency` keeps the tower -- on the device, or in mapped pinned
/// host memory, staged for each encode (ADR 0075) -- and any object the
/// inventory does not name refused (ADR 0002).
pub fn bind(reader: &Reader, tower: &VisionTower, residency: VisionResidency) -> Result<VisionPlan> {
    check_identity(reader)?;
    let mut binder = Binder::new(reader);
    let mut handles = Vec::new();
    for entry in entries(tower) {
        let Entry::Tensor { name, format, layout, shape } = entry else {
            return Err(fail(format!("{}: the tower holds tensors only", entry.name())));
        };
        let handle = binder.require_tensor(&name, format, layout, &shape)?;
        match residency {
            VisionResidency::Host => binder.place_in_host_mapped(handle)?,
            VisionResidency::Device => binder.materialize_on_device(handle)?,
        }
        handles.push(handle);
    }
    Ok(VisionPlan { plan: binder.finish()?, handles })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inventory::{vision_scope, vision_scope_27b};
    use crate::packer::{pack, PackOutcome};

    /// GitHub #557: a load reads the merger's width off the companion's
    /// directory and binds every object where the residency keeps the tower
    /// -- in mapped host memory under `host`, on the device under `device`;
    /// the main container is no companion, and a tower of other shapes
    /// fails naming the object.
    #[test]
    fn the_companion_binds_where_the_residency_keeps_the_tower() {
        let main = super::super::fixture::build("vision-bind").expect("the main fixture");
        main.tree.write_vision(&main.path).expect("the vision work tree");
        let outcome = pack(&main.tree.vision_pack_options(main.path.clone()), &mut |_| {}).expect("pack the tower");
        assert!(matches!(outcome, PackOutcome::Finished { .. }), "{outcome:?}");
        let companion = Reader::open(&main.tree.vision_artifact_path()).expect("open the companion");
        let tower = fixture_geometry();
        assert_eq!(merger_width(&companion).expect("the merger's width"), tower.out_hidden);

        let host = bind(&companion, &tower, VisionResidency::Host).expect("bind in host memory");
        assert_eq!(host.handles.len(), entries(&tower).len());
        assert!(host.plan.device_objects.is_empty(), "nothing on the device");
        assert_eq!(host.plan.host_mapped_objects.len(), host.handles.len());
        let device = bind(&companion, &tower, VisionResidency::Device).expect("bind on the device");
        assert!(device.plan.host_mapped_objects.is_empty(), "nothing in host memory");
        assert_eq!(device.plan.device_objects.len(), device.handles.len());

        let main_reader = Reader::open(&main.path).expect("open the main container");
        let err = bind(&main_reader, &tower, VisionResidency::Host).err().expect("no companion").to_string();
        assert!(err.contains("not a vision companion"), "{err}");
        assert!(merger_width(&main_reader).is_err(), "the main container has no merger");
        let err = bind(&companion, &geometry(), VisionResidency::Host).err().expect("other shapes").to_string();
        assert!(err.contains("vision/"), "{err}");
    }

    /// What the packer holds the companion to is what a load binds: at
    /// Flash-Next's widths the inventory is `vision_scope(2560)` object for
    /// object (name, format, layout, shape), the 27B tower's 333 objects with
    /// the merger writing 2,560 rows where the 27B's writes 5,120.
    #[test]
    fn the_companion_holds_the_objects_a_load_binds_with_the_merger_2560_wide() {
        let ours = entries(&geometry());
        let bound = vision_scope(2560);
        assert_eq!(ours.len(), 333);
        assert_eq!(ours.len(), bound.len());
        for (entry, bound) in ours.iter().zip(&bound) {
            let Entry::Tensor { name, format, layout, shape } = entry else {
                panic!("{} is a resource", entry.name());
            };
            assert_eq!((name.as_str(), *format, *layout, shape.as_slice()), (bound.name, bound.format, bound.layout, bound.shape));
        }
        let served = vision_scope_27b();
        let differ: Vec<&str> = bound.iter().zip(&served).filter(|(f, s)| f.shape != s.shape).map(|(f, _)| f.name).collect();
        assert_eq!(differ, ["vision/merger/fc2", "vision/merger/fc2_bias"]);
    }
}
