//! Flash-Next's vision tower at load (GitHub #557, ADR 0087): its vision
//! companion opened beside the main container and held to it before any
//! byte of the tower is read -- the sidecar's pair pin
//! ([`ignis_artifact::flash_next::vision::check_pair`]), the container's
//! identity, and the width its merger writes, which must be the text
//! model's hidden width ([`EmbeddingWidth::from_inventory`]) -- then bound
//! where the vision residency keeps the tower: in mapped pinned host memory,
//! staged into the prefill scratch for each encode, or on the device (ADR
//! 0075). A start with vision on refuses a missing or mismatched companion
//! naming its file; the server checks it before the load starts.

use std::path::{Path, PathBuf};

use ignis_artifact::flash_next::vision;
use ignis_artifact::packer::{sidecar_path, VISION_ARTIFACT_FILE_NAME};
use ignis_artifact::{vision_scope, InventoryEntry, Reader, VisionResidency};

use crate::vision::EmbeddingWidth;

/// Where a model directory keeps the vision companion.
pub fn companion_path(model_dir: &Path) -> PathBuf {
    model_dir.join(VISION_ARTIFACT_FILE_NAME)
}

/// Why a load with vision on refuses Flash-Next's vision companion: each
/// names the companion's file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompanionRefused {
    /// No companion beside the artifact.
    Missing { path: PathBuf },
    /// One that is not this artifact's: its sidecar unreadable, its pair pin
    /// another main container's, another container in its place, or a
    /// merger writing another width than the model's.
    Mismatched { path: PathBuf, reason: String },
}

impl std::fmt::Display for CompanionRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing { path } => write!(
                f,
                "vision is on, and Qwen3.8-Flash-Next's vision companion {} is not beside its artifact: \
                 `ignis-server model download qwen3.8-flash-next` fetches it, or turn vision off \
                 (--vision-enabled false)",
                path.display()
            ),
            Self::Mismatched { path, reason } => {
                write!(f, "Qwen3.8-Flash-Next's vision companion {} is not this artifact's: {reason}", path.display())
            }
        }
    }
}

impl std::error::Error for CompanionRefused {}

/// The vision companion, opened beside the main container and held to it.
pub struct Companion {
    path: PathBuf,
    reader: Reader,
    width: EmbeddingWidth,
}

impl Companion {
    /// The companion beside the main container at `main_path` (opened as
    /// `main`), for a text model `model_hidden` wide: its sidecar's pair pin
    /// against `main`, its identity, then its merger's width against the
    /// model's, each refused naming the file.
    pub fn open(main_path: &Path, main: &Reader, model_hidden: u64) -> Result<Self, CompanionRefused> {
        let path = companion_path(main_path.parent().unwrap_or_else(|| Path::new(".")));
        if !path.exists() {
            return Err(CompanionRefused::Missing { path });
        }
        let mismatched = |reason: String| CompanionRefused::Mismatched { path: path.clone(), reason };
        let sidecar = sidecar_path(&path);
        let text = std::fs::read(&sidecar).map_err(|e| mismatched(format!("read {}: {e}", sidecar.display())))?;
        let sidecar: serde_json::Value =
            serde_json::from_slice(&text).map_err(|e| mismatched(format!("{}: {e}", sidecar.display())))?;
        vision::check_pair(&sidecar, main).map_err(|e| mismatched(e.to_string()))?;
        let reader = Reader::open(&path).map_err(|e| mismatched(format!("open: {e}")))?;
        vision::check_identity(&reader).map_err(|e| mismatched(e.to_string()))?;
        let rows = vision::merger_width(&reader).map_err(|e| mismatched(e.to_string()))?;
        let width = EmbeddingWidth::from_inventory(&vision_scope(rows), model_hidden).map_err(|e| mismatched(e.to_string()))?;
        Ok(Self { path, reader, width })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn reader(&self) -> &Reader {
        &self.reader
    }

    /// The width the merger writes: the model's hidden width.
    pub fn width(&self) -> EmbeddingWidth {
        self.width
    }

    /// Bind the tower where `residency` keeps it, every object of the
    /// companion consumed.
    pub fn bind(&self, residency: VisionResidency) -> Result<TowerPlan, String> {
        let tower = ignis_artifact::nvfp4_27b::VisionTower { out_hidden: self.width.hidden(), ..vision::geometry() };
        let plan = vision::bind(&self.reader, &tower, residency)
            .map_err(|e| format!("bind the vision companion {}: {e}", self.path.display()))?;
        let host_mapped_bytes = ignis_artifact::vision_host_mapped_bytes(&self.reader, &plan.plan);
        Ok(TowerPlan { entries: vision_scope(self.width.hidden()), plan, host_mapped_bytes })
    }
}

/// The tower's bind: what the reservation plan, the host plan and the load
/// all read.
pub struct TowerPlan {
    /// The tower's objects, in the order [`TowerPlan::plan`] bound them.
    pub entries: Vec<InventoryEntry>,
    pub plan: vision::VisionPlan,
    host_mapped_bytes: u64,
}

impl TowerPlan {
    /// What the tower pins in mapped host memory under the `host` residency:
    /// its spans, with their 4 KiB padding -- the host plan's `vision_tower`
    /// line. 0 on the device.
    pub fn host_mapped_bytes(&self) -> u64 {
        self.host_mapped_bytes
    }

    /// What the tower takes in VRAM under the `device` residency: a weights
    /// line. 0 in host memory.
    pub fn device_bytes(&self) -> u64 {
        self.plan.plan.device_capacity_bytes
    }
}

#[cfg(feature = "cuda")]
pub use device::Tower;

#[cfg(feature = "cuda")]
mod device {
    use std::ffi::CString;

    use ignis_artifact::{materialize, CudaDevice, MaterializedArtifact, VisionResidency};

    use super::{Companion, TowerPlan};
    use crate::model_load::ffi;

    /// The tower placed where its residency keeps it, and its descriptors:
    /// what a Flash-Next load with vision hands the leaf beside the trunk's
    /// (`vision/*`). The model that loads it must be dropped first: its
    /// descriptors point into this tower's memory.
    pub struct Tower {
        _names: Vec<CString>,
        tensors: Vec<ffi::IgnisBoundTensor>,
        plan: TowerPlan,
        artifact: MaterializedArtifact,
        device: CudaDevice,
    }

    impl Tower {
        /// `companion`'s tower, bound and placed under `residency`.
        pub fn load(companion: &Companion, residency: VisionResidency) -> Result<Self, String> {
            let plan = companion.bind(residency)?;
            let mut device = CudaDevice::create(0).map_err(|e| format!("CUDA device: {e}"))?;
            let artifact = materialize(companion.reader(), &plan.plan.plan, &mut device, None)
                .map_err(|e| format!("materialize the vision tower: {e}"))?;
            let (names, tensors) = crate::model_load::vision_tower_tensors(&plan, &artifact)?;
            Ok(Self { _names: names, tensors, plan, artifact, device })
        }

        pub(crate) fn bound_tensors(&self) -> &[ffi::IgnisBoundTensor] {
            &self.tensors
        }

        /// The bind the tower was placed by (a plan's descriptors read it).
        pub fn plan(&self) -> &TowerPlan {
            &self.plan
        }
    }

    impl Drop for Tower {
        fn drop(&mut self) {
            let _ = self.artifact.release_arena(&mut self.device);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ignis_artifact::flash_next::fixture;

    /// GitHub #557: the companion is held to the main container before any
    /// of the tower is read, each refusal naming its file: none beside it,
    /// one pinned to another main container, one whose merger writes another
    /// width than the model's; the one packed for it opens, at its width.
    #[test]
    fn a_companion_is_held_to_its_main_container_and_the_models_width() {
        let tower = vision::fixture_geometry();
        let alone = fixture::build("vision-companion-missing").expect("the main fixture");
        let main = Reader::open(&alone.path).expect("open the main container");
        let err = Companion::open(&alone.path, &main, tower.out_hidden).err().expect("no companion");
        let path = companion_path(alone.path.parent().unwrap());
        assert_eq!(err, CompanionRefused::Missing { path: path.clone() });
        let text = err.to_string();
        assert!(text.contains(&path.display().to_string()) && text.contains("--vision-enabled false"), "{text}");

        let paired = fixture::build("vision-companion-paired").expect("the main fixture");
        let beside = paired.with_vision_companion().expect("its vision companion");
        assert_eq!(beside, companion_path(paired.path.parent().unwrap()));
        let main = Reader::open(&paired.path).expect("open the main container");
        let companion = Companion::open(&paired.path, &main, tower.out_hidden).expect("its own companion");
        assert_eq!((companion.path(), companion.width().hidden()), (beside.as_path(), tower.out_hidden));

        let err = Companion::open(&paired.path, &main, 2560).err().expect("another width").to_string();
        assert!(err.contains("vision/merger/fc2") && err.contains("2560") && err.contains(&tower.out_hidden.to_string()), "{err}");
        // Another container in the main one's place: the pin names the field.
        let other = Reader::open(companion.path()).expect("another container");
        let err = Companion::open(&paired.path, &other, tower.out_hidden).err().expect("pinned elsewhere").to_string();
        assert!(err.contains("pair.main.model_id") && err.contains(VISION_ARTIFACT_FILE_NAME), "{err}");
    }
}
