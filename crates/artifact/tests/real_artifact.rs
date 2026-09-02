//! Integration test against the real Qwen 3.8-27B `nvfp4full` v2 artifact
//! (the 1,325-object container, ~19 GB). Skips gracefully when the file is
//! not at its machine-local path.
//!
//! Tiers (ADR 0002: the binder must consume *every* object — an unconsumed
//! object is a load failure):
//!   1. Inventory + bind-all (default, cheap — a directory walk, no I/O).
//!   2. Full `CpuDevice` materialization (gated: `IGNIS_TEST_FULL_MATERIALIZE=1`).
//!   3. `CudaDevice` upload (feature `cuda`, gated: `IGNIS_TEST_CUDA=1`; the
//!      RTX 5090 is held by the reference runner — ADR 0006, do not run).

use std::path::Path;

use ignis_artifact::{
    materialize, Binder, CpuDevice, MaterializationPlan, NumericFormat, Object, Reader,
    StorageLayout,
};
#[cfg(feature = "cuda")]
use ignis_artifact::CudaDevice;

/// The fork-local model cache (the artifact the running `ninfer-serve` loads).
const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";

/// Open the real artifact, or `None` (with a skip note) when it is absent.
fn open_or_skip() -> Option<Reader> {
    let path = Path::new(ARTIFACT);
    if !path.exists() {
        eprintln!("skip: {ARTIFACT} does not exist");
        return None;
    }
    Some(
        Reader::open(path)
            .unwrap_or_else(|e| panic!("open {ARTIFACT}: {e}")),
    )
}

/// Bind every object in the directory against its own descriptor, place each
/// (device tensors -> the device arena, resources -> host-retained), and
/// finish. ADR 0002: an unconsumed object is a load failure, so `finish()`
/// must succeed (zero unconsumed, zero unplanned).
fn bind_all(reader: &Reader) -> Result<MaterializationPlan, String> {
    let mut binder = Binder::new(reader);
    for object in reader.objects() {
        match object {
            Object::Tensor(t) => {
                let handle = binder
                    .require_tensor(&t.name, t.format, t.layout, &t.shape)
                    .map_err(|e| e.to_string())?;
                binder
                    .materialize_on_device(handle)
                    .map_err(|e| e.to_string())?;
            }
            Object::Resource(r) => {
                let handle = binder
                    .require_resource(&r.name, r.encoding)
                    .map_err(|e| e.to_string())?;
                binder
                    .retain_on_host(handle)
                    .map_err(|e| e.to_string())?;
            }
        }
    }
    binder.finish().map_err(|e| e.to_string())
}

#[test]
fn real_nvfp4full_inventory_and_bind_all() {
    let reader = match open_or_skip() {
        Some(r) => r,
        None => return,
    };

    // 1,319 tensors + 6 frontend resources = 1,325 objects.
    let objects = reader.objects();
    let tensors = objects.iter().filter(|o| matches!(o, Object::Tensor(_))).count();
    let resources = objects.iter().filter(|o| matches!(o, Object::Resource(_))).count();
    assert_eq!(objects.len(), 1325, "expected 1325 objects");
    assert_eq!(tensors, 1319);
    assert_eq!(resources, 6);

    assert_eq!(reader.identity().model_id, "qwen3.8-27b");

    // The 6 frontend resources carry the tokenizer + chat template (ADR 0002:
    // the engine reads them straight from the container).
    let resource_names: Vec<&str> = objects
        .iter()
        .filter_map(|o| match o {
            Object::Resource(r) => Some(r.name.as_str()),
            _ => None,
        })
        .collect();
    for expected in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"] {
        assert!(
            resource_names.iter().any(|n| n.contains(expected)),
            "expected a resource containing '{expected}'; got {resource_names:?}"
        );
    }

    // Every NVFP4 tensor must use the blockscale layout.
    let nvfp4: Vec<&ignis_artifact::TensorDescriptor> = objects
        .iter()
        .filter_map(|o| match o {
            Object::Tensor(t) if t.format == NumericFormat::Nvfp4 => Some(t),
            _ => None,
        })
        .collect();
    assert!(!nvfp4.is_empty(), "expected NVFP4 tensors");
    for t in &nvfp4 {
        assert_eq!(
            t.layout,
            StorageLayout::BlockScaleK16M128x4V1,
            "NVFP4 tensor {} must use the blockscale layout",
            t.name
        );
    }

    // ADR 0002: bind *every* object (zero unconsumed) and finish.
    let plan = bind_all(&reader).expect("all objects consumed (ADR 0002)");
    assert_eq!(plan.object_count, 1325);
    assert_eq!(plan.device_objects.len(), 1319);
    assert_eq!(plan.host_objects.len(), 6);
    assert!(plan.device_capacity_bytes > 0);
}

/// Full `CpuDevice` materialization of the whole artifact. Gated: it
/// allocates several GB of host RAM and reads the entire file, so it only
/// runs when `IGNIS_TEST_FULL_MATERIALIZE=1` is set.
#[test]
fn real_nvfp4full_full_cpu_materialization() {
    let reader = match open_or_skip() {
        Some(r) => r,
        None => return,
    };
    if std::env::var("IGNIS_TEST_FULL_MATERIALIZE").is_err() {
        eprintln!("skip: set IGNIS_TEST_FULL_MATERIALIZE=1 to run the full CpuDevice materialization");
        return;
    }
    let plan = bind_all(&reader).expect("all objects consumed (ADR 0002)");
    let mut device = CpuDevice::new();
    let artifact = materialize(&reader, &plan, &mut device, None).expect("materialize");
    let stats = artifact.stats();
    assert_eq!(stats.tensor_count, 1319);
    assert_eq!(stats.resource_count, 6);
    assert!(stats.device_capacity_bytes > 0);
    assert!(stats.h2d_bytes > 0);
    assert!(stats.peak_staging_bytes > 0);
}

/// `CudaDevice` upload path (feature `cuda`). The RTX 5090 is held by the
/// reference `ninfer-serve` (ADR 0006), so this skips unless explicitly
/// enabled with a free GPU.
#[cfg(feature = "cuda")]
#[test]
fn real_nvfp4full_cuda_device() {
    if std::env::var("IGNIS_TEST_CUDA").is_err() {
        eprintln!("skip: set IGNIS_TEST_CUDA=1 (GPU must be free) to run the CudaDevice path");
        return;
    }
    let reader = match open_or_skip() {
        Some(r) => r,
        None => return,
    };
    let plan = bind_all(&reader).expect("all objects consumed (ADR 0002)");
    let mut device = CudaDevice::create(0).expect("CUDA driver available");
    let artifact = materialize(&reader, &plan, &mut device, None).expect("materialize");
    let stats = artifact.stats();
    assert!(stats.device_capacity_bytes > 0);
    assert_eq!(stats.tensor_count, 1319);
}