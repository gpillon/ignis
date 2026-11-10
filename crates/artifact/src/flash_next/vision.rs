//! Flash-Next's vision companion container (ADR 0087, flash-next-layout.md
//! §14): the checkpoint's vision tower and its merger, beside the main
//! container, bound only when vision is on.
//!
//! The tower holds the 27B tower's objects under the same names, at
//! Flash-Next's widths, where the merger writes 2,560 columns (the text
//! model's hidden width), not the 27B's 5,120; its formats are its own
//! ([`entries`]): every matrix W8G32, the rest BF16 (GitHub #556, the
//! owner's decision on #560's alarm). Its identity is its own
//! ([`VISION_MODEL_ID`]), and its sidecar pins the one main container it
//! belongs to (`pair.main`, the MTP companion's shape, flash-next-layout.md
//! §13.2).

use std::sync::{Mutex, PoisonError};

use super::{fail, VISION_MODEL_ID};
use crate::binder::{Binder, DeviceEncoding, MaterializationPlan, ObjectHandle, Residence};
use crate::inventory::InventoryEntry;
use crate::nvfp4_27b::{vision_objects, Entry, VisionFormats, VisionTower, VISION_MERGER_OUTPUT};
use crate::{ArtifactIdentity, NumericFormat, Object, Reader, Result, StorageLayout, VisionResidency};

/// The merger's output object, whose rows are the width the tower writes.
const MERGER_OUTPUT: &str = VISION_MERGER_OUTPUT[0];

/// The tower's formats: every matrix W8G32 with binary16 scales, row-split.
pub const FORMATS: VisionFormats =
    VisionFormats::every_matrix((NumericFormat::W8G32F16S, StorageLayout::RowSplitK128V1));

/// The companion's identity: the vision `model_id` and the build's weights
/// id (every matrix W8, the rest BF16, from revision `de4b8e4`).
pub fn identity() -> ArtifactIdentity {
    ArtifactIdentity { model_id: VISION_MODEL_ID.into(), weights_id: "vision-w8-bf16-de4b8e4".into() }
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
/// holds the work tree to, and ([`scope`]) what a load binds. The 27B
/// tower's objects ([`vision_objects`]) in [`FORMATS`]: the patch
/// embedding, the blocks' attention input and output and MLP linears and
/// the 2 x 2 merger's linears W8; the biases, norms and position table BF16.
pub fn entries(tower: &VisionTower) -> Vec<Entry> {
    vision_objects(tower, &FORMATS)
}

/// [`entries`] as the inventory a load binds the tower by and hands the
/// leaf's descriptors (`ignis_core::flash_next_vision`), each on the device
/// as stored. Its names and shapes are `'static`: built once per tower and
/// kept for the process, so a load asking at every switch leaks nothing
/// more (a companion's merger width picks the tower, one per file).
pub fn scope(tower: &VisionTower) -> Vec<InventoryEntry> {
    static BUILT: Mutex<Vec<(VisionTower, &'static [InventoryEntry])>> = Mutex::new(Vec::new());
    let mut built = BUILT.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((_, scope)) = built.iter().find(|(of, _)| of == tower) {
        return scope.to_vec();
    }
    let scope: Vec<InventoryEntry> = entries(tower)
        .into_iter()
        .filter_map(|entry| match entry {
            Entry::Tensor { name, format, layout, shape } => Some(InventoryEntry {
                name: Box::leak(name.into_boxed_str()),
                format,
                layout,
                shape: Box::leak(shape.into_boxed_slice()),
                residence: Residence::Device,
                encoding: DeviceEncoding::Stored,
            }),
            Entry::Resource { .. } => None,
        })
        .collect();
    built.push((tower.clone(), Box::leak(scope.clone().into_boxed_slice())));
    scope
}

/// The companion's sidecar `pair.main` against the opened main container
/// `main`: `Ok` when its `model_id`, `weights_id` and `content_hash` are the
/// main container's, else the first field that differs, both values named.
pub fn check_pair(sidecar: &serde_json::Value, main: &Reader) -> Result<()> {
    super::check_companion_pair(sidecar, main, "vision")
}

/// Refuse a container that is not a vision companion, by its `model_id`
/// (the main container or the MTP companion handed in its place), or one of
/// another encoding of the tower than [`entries`], by its `weights_id` (the
/// grouped tower first shipped, left by an older download).
pub fn check_identity(reader: &Reader) -> Result<()> {
    let ArtifactIdentity { model_id, weights_id } = reader.identity();
    if model_id != VISION_MODEL_ID {
        return Err(super::fail(format!("artifact model_id {model_id} is not a vision companion ({VISION_MODEL_ID})")));
    }
    let ours = identity().weights_id;
    if *weights_id != ours {
        return Err(super::fail(format!(
            "the vision companion's weights_id is {weights_id}, another encoding of the tower than the one this build binds ({ours})"
        )));
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

    /// A load asks for the tower's inventory at every open of the companion,
    /// once per switch request: its names and shapes are built once per
    /// tower, every call handing out the same memory, so a long-running
    /// server's switches leak nothing.
    #[test]
    fn the_scope_is_built_once_per_tower() {
        let (first, again) = (scope(&geometry()), scope(&geometry()));
        assert_eq!(first, again);
        for (built, asked) in first.iter().zip(&again) {
            assert!(std::ptr::eq(built.name, asked.name) && std::ptr::eq(built.shape, asked.shape), "{} built again", built.name);
        }
        let narrow = scope(&VisionTower { out_hidden: 5120, ..geometry() });
        assert_eq!(narrow.len(), first.len());
        assert_ne!(narrow, first, "another merger width is another tower");
        assert!(std::ptr::eq(narrow[0].name, scope(&VisionTower { out_hidden: 5120, ..geometry() })[0].name));
    }

    /// GitHub #556: a companion of another encoding of the tower -- the
    /// grouped one first shipped, weights `vision-w8q4q5q6-bf16-de4b8e4`,
    /// left beside an artifact by an older download -- is refused by its
    /// identity, both weights ids named, before any of its objects is read.
    #[test]
    fn a_companion_of_another_encoding_is_refused_by_its_weights_id() {
        assert_eq!(identity().weights_id, "vision-w8-bf16-de4b8e4");
        let main = super::super::fixture::build("vision-other-encoding").expect("the main fixture");
        main.tree.write_vision(&main.path).expect("the vision work tree");
        let mut options = main.tree.vision_pack_options(main.path.clone());
        options.identity.weights_id = "vision-w8q4q5q6-bf16-de4b8e4".into();
        let outcome = pack(&options, &mut |_| {}).expect("pack the tower");
        assert!(matches!(outcome, PackOutcome::Finished { .. }), "{outcome:?}");
        let companion = Reader::open(&main.tree.vision_artifact_path()).expect("open the companion");
        for err in [
            check_identity(&companion).err().expect("another encoding").to_string(),
            bind(&companion, &fixture_geometry(), VisionResidency::Host).err().expect("another encoding").to_string(),
        ] {
            assert!(err.contains("vision-w8q4q5q6-bf16-de4b8e4") && err.contains("vision-w8-bf16-de4b8e4"), "{err}");
        }
    }

    /// What the packer holds the companion to is what a load binds: at
    /// Flash-Next's widths [`scope`] is [`entries`] object for object (name,
    /// format, layout, shape). The objects are the 27B tower's 333, under the
    /// same names and shapes but for the merger's 2,560 rows where the 27B's
    /// writes 5,120, and every matrix is W8G32 (GitHub #556, the owner's
    /// decision on #560's alarm): the patch embedding, the 27 blocks' four
    /// linears and the merger's two, 111 in all; the biases, norms and
    /// position table stay BF16. Its payload is the 481,445,216 bytes #560's
    /// diagnosis costed for the all-W8 tower.
    #[test]
    fn the_companion_holds_every_matrix_in_w8_and_a_load_binds_what_the_packer_holds() {
        let ours = entries(&geometry());
        let bound = scope(&geometry());
        assert_eq!(ours.len(), 333);
        assert_eq!(ours.len(), bound.len());
        for (entry, bound) in ours.iter().zip(&bound) {
            let Entry::Tensor { name, format, layout, shape } = entry else {
                panic!("{} is a resource", entry.name());
            };
            assert_eq!((name.as_str(), *format, *layout, shape.as_slice()), (bound.name, bound.format, bound.layout, bound.shape));
        }
        let grouped = vision_scope(2560);
        let mut matrices = 0;
        for (ours, grouped) in bound.iter().zip(&grouped) {
            assert_eq!((ours.name, ours.layout, ours.shape), (grouped.name, grouped.layout, grouped.shape));
            if grouped.format == NumericFormat::Bf16 {
                assert_eq!(ours.format, NumericFormat::Bf16, "{}", ours.name);
            } else {
                assert_eq!(ours.format, NumericFormat::W8G32F16S, "{}", ours.name);
                matrices += 1;
            }
        }
        assert_eq!(matrices, 1 + 27 * 4 + 2);
        let served = vision_scope_27b();
        let differ: Vec<&str> = grouped.iter().zip(&served).filter(|(f, s)| f.shape != s.shape).map(|(f, _)| f.name).collect();
        assert_eq!(differ, ["vision/merger/fc2", "vision/merger/fc2_bias"]);
        let payload: u64 = bound
            .iter()
            .map(|e| crate::tensor_encoded_size(e.layout, e.format, e.shape).unwrap_or_else(|err| panic!("{}: {err}", e.name)))
            .sum();
        assert_eq!(payload, 481_445_216);
    }
}
