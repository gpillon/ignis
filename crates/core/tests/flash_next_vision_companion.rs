//! Flash-Next's vision companion on this machine (GitHub #557, ADR 0087):
//! the shipped companion beside the real main container is held to it and
//! binds where either residency keeps the tower, with the figures the host
//! plan and the VRAM plan charge for it.
//!
//! Machine-local, as `crates/artifact/tests/flash_next_artifact.rs` is: it
//! skips when `F:/ai/models/Qwen3.8-Flash-Next-ignis/` (or
//! `IGNIS_FLASH_NEXT_DIR`) holds no companion. Seconds: nothing here reads
//! the tower's payload or the main container's.

use std::path::PathBuf;

use ignis_artifact::flash_next::FlashNextGeometry;
use ignis_artifact::packer::ARTIFACT_FILE_NAME;
use ignis_artifact::{Reader, VisionResidency};
use ignis_core::compute::ModelConfig;
use ignis_core::flash_next_vision::{companion_path, Companion};

const MODEL_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";

/// The model directory, when it holds the main container and its vision
/// companion; `None`, with the reason printed, otherwise.
fn model_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(MODEL_DIR), PathBuf::from);
    for path in [dir.join(ARTIFACT_FILE_NAME), companion_path(&dir)] {
        if !path.exists() {
            eprintln!("skip: {} does not exist", path.display());
            return None;
        }
    }
    Some(dir)
}

/// The companion opens against the main container at Flash-Next's hidden
/// width, and binds every one of its 333 objects: under `host` all in mapped
/// host memory, its spans the host plan's `vision_tower` line and no VRAM;
/// under `device` all on the device, a weights line and no host line. Every
/// matrix is W8 (GitHub #556), and the plan the leaf's descriptors read
/// carries the companion's formats: the tower is larger than the 27B's
/// grouped one, 297,000,960 bytes of spans (ADR 0075), though its merger
/// writes half the rows.
#[test]
fn the_shipped_companion_binds_under_either_residency() {
    let Some(dir) = model_dir() else { return };
    let main_path = dir.join(ARTIFACT_FILE_NAME);
    let main = Reader::open(&main_path).expect("open the main container");
    let hidden = ModelConfig::flash_next_from(&FlashNextGeometry::qwen38_flash_next()).hidden;
    let companion = Companion::open(&main_path, &main, hidden).unwrap_or_else(|refused| panic!("{refused}"));
    assert_eq!(companion.width().hidden(), 2560);

    let host = companion.bind(VisionResidency::Host).expect("bind in host memory");
    assert_eq!(host.plan.handles.len(), 333);
    assert_eq!(host.plan.plan.host_mapped_objects.len(), 333);
    assert_eq!(host.device_bytes(), 0, "host: no VRAM");
    let device = companion.bind(VisionResidency::Device).expect("bind on the device");
    assert_eq!(device.plan.plan.device_objects.len(), 333);
    assert_eq!(device.host_mapped_bytes(), 0, "device: no host line");
    let file_bytes = std::fs::metadata(companion.path()).expect("the companion's size").len();
    println!(
        "vision companion {}: {} bytes on disk, {} bytes of host spans (the host plan's vision_tower line), \
         {} bytes on the device",
        companion.path().display(),
        file_bytes,
        host.host_mapped_bytes(),
        device.device_bytes()
    );
    let formats: std::collections::BTreeMap<&str, usize> = host.entries.iter().fold(Default::default(), |mut n, e| {
        *n.entry(e.format.name()).or_default() += 1;
        n
    });
    assert_eq!(formats, [("BF16", 222), ("W8G32_F16S", 111)].into_iter().collect(), "every matrix W8");
    // The companion re-encoded with every matrix W8 (#556, sha256 0110bb79...):
    // its spans, each padded to 4 KiB, and its device arena. The grouped one
    // first shipped had 284,454,912 and 283,180,544.
    assert_eq!(host.host_mapped_bytes(), 482_705_408, "the host plan's vision_tower line");
    assert_eq!(device.device_bytes(), 481_451_264, "the weights line under device");
}

/// A Flash-Next load with vision prepares images with the processor its own
/// container carries (`frontend/preprocessor_config.json`), checked against
/// the model contract's placeholders by its own tokenizer: an item at most
/// 4096 x 4096 pixels, 16,384 merged tokens, and the 27B's lower bound.
#[test]
fn flash_next_carries_its_own_image_processor() {
    let Some(dir) = model_dir() else { return };
    let main = Reader::open(&dir.join(ARTIFACT_FILE_NAME)).expect("open the main container");
    let frontend = ignis_artifact::FrontendSet::from_reader(&main).expect("its frontend");
    let processor = frontend.vision_processor().expect("its preprocessor_config.json");
    let options = processor.options();
    println!("Flash-Next's processor: {options:?}");
    assert_eq!(options.max_item_tokens(), 16_384, "4096 x 4096 pixels");
    assert_eq!(options.min_pixels, 65_536, "the 27B's lower bound");
}
