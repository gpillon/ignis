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

use super::VISION_MODEL_ID;
use crate::nvfp4_27b::{vision_tower, Entry, VisionTower};
use crate::{ArtifactIdentity, Reader, Result};

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

/// The companion's objects in container order, at `tower`.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inventory::vision_scope_27b;

    /// The 27B recipe at Flash-Next's widths: the served 27B tower's 333
    /// objects, name for name and format for format, and every shape the
    /// 27B's but the merger's output, 2,560 wide.
    #[test]
    fn the_companion_holds_the_27b_towers_objects_with_the_merger_2560_wide() {
        let ours = entries(&geometry());
        let served = vision_scope_27b();
        assert_eq!(ours.len(), 333);
        assert_eq!(ours.len(), served.len());
        for (entry, served) in ours.iter().zip(&served) {
            let Entry::Tensor { name, format, layout, shape } = entry else {
                panic!("{} is a resource", entry.name());
            };
            assert_eq!((name.as_str(), *format, *layout), (served.name, served.format, served.layout));
            let expected: &[u64] = match served.name {
                "vision/merger/fc2" => &[2560, 4608],
                "vision/merger/fc2_bias" => &[2560],
                _ => served.shape,
            };
            assert_eq!(shape.as_slice(), expected, "{name}");
        }
    }
}
