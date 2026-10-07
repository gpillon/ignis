//! The MTP head on the device (spec flash-next/07 phase D, GitHub #307):
//! the companion container opened beside the main one, held to the main
//! container's identity before any byte of the head is read
//! ([`ignis_artifact::flash_next::mtp::check_pair`]), bound, and every
//! object placed on the device -- its 29 tensors, which the load hands the
//! leaf as `mtp.*` descriptors beside the trunk's, and its 1024 expert
//! projections, which are resident for the life of the load (outside the
//! trunk's expert cache) and reach the leaf as one slot table.

use std::ffi::{c_void, CString};
use std::path::{Path, PathBuf};

use ignis_artifact::flash_next::{mtp, FlashNextGeometry};
use ignis_artifact::packer::{sidecar_path, MTP_ARTIFACT_FILE_NAME};
use ignis_artifact::{materialize, CudaDevice, MaterializedArtifact, Reader};

use crate::model_load::{ffi, mtp_bound_tensors};

/// Where a model directory keeps the head's companion.
pub fn companion_path(model_dir: &Path) -> PathBuf {
    model_dir.join(MTP_ARTIFACT_FILE_NAME)
}

/// The head's companion, bound and on the device. The model that loads it
/// must be dropped first: its descriptors and slots point into this arena.
pub struct MtpHead {
    _names: Vec<CString>,
    tensors: Vec<ffi::IgnisBoundTensor>,
    slots: Vec<ffi::IgnisMoeSlot>,
    plan: mtp::MtpPlan,
    artifact: MaterializedArtifact,
    device: CudaDevice,
}

impl MtpHead {
    /// The companion at `path`, pinned to the opened main container `main`.
    pub fn load(path: &Path, main: &Reader, geometry: &FlashNextGeometry) -> Result<Self, String> {
        let sidecar = sidecar_path(path);
        let text = std::fs::read(&sidecar).map_err(|e| format!("read {}: {e}", sidecar.display()))?;
        let sidecar: serde_json::Value =
            serde_json::from_slice(&text).map_err(|e| format!("parse {}: {e}", sidecar.display()))?;
        mtp::check_pair(&sidecar, main).map_err(|e| e.to_string())?;
        let reader = Reader::open(path).map_err(|e| format!("open {}: {e:?}", path.display()))?;
        let plan = mtp::bind(&reader, geometry).map_err(|e| format!("bind the MTP head: {e}"))?;
        let mut device = CudaDevice::create(0).map_err(|e| format!("CUDA device: {e}"))?;
        let artifact =
            materialize(&reader, &plan.plan, &mut device, None).map_err(|e| format!("materialize the MTP head: {e}"))?;
        let placement = |handle| {
            let view = artifact.device_view(handle).map_err(|e| e.to_string())?;
            Ok((view.bytes, view.base as *const c_void))
        };
        let (names, tensors) = mtp_bound_tensors(&plan, geometry, placement)?;
        let slots = plan
            .experts
            .iter()
            .map(|expert| {
                let view = artifact.device_view(expert.handle).map_err(|e| e.to_string())?;
                Ok(ffi::IgnisMoeSlot { record: view.base as *const c_void, k2: u32::from(expert.k.k2()), reserved: 0 })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self { _names: names, tensors, slots, plan, artifact, device })
    }

    pub(crate) fn bound_tensors(&self) -> &[ffi::IgnisBoundTensor] {
        &self.tensors
    }

    pub(crate) fn slots(&self) -> &[ffi::IgnisMoeSlot] {
        &self.slots
    }

    /// The bind the head was placed by (a plan's descriptors read it).
    pub fn plan(&self) -> &mtp::MtpPlan {
        &self.plan
    }
}

impl Drop for MtpHead {
    fn drop(&mut self) {
        let _ = self.artifact.release_arena(&mut self.device);
    }
}
