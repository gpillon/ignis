//! The served media encode's rows, for the tower check against the BF16 tower
//! (GitHub #560, spec #555's seam 2, ADR 0087): Flash-Next's vision companion
//! and, as the bar, the 27B's tower, each encoding the same images the way the
//! server does -- the family's own vision processor on the image's bytes, then
//! the leaf's media encode under the RTX 5090 profile's `host` residency.
//!
//! The images: the vision canary's four, sixteen synthetic ones drawn here
//! (rings, checkers, stripes, gradients with blocks, from 32 x 32 to
//! 1,440 x 1,440, square to 1:17), and every PNG or JPEG in
//! `IGNIS_TOWER_IMAGES` when it is set (the known-answer set's photos, resized
//! by `tools/vision-known-answers/fetch.py`). Every item stays at most
//! [`MAX_ITEM_TOKENS`] merged tokens, so the BF16 tower's reference
//! (`tools/flash-next-vision/tower_check.py reference`, on the CPU) runs in
//! minutes; an extra image past it fails the test by name.
//!
//! Each load checks what it encoded: every item gives its grid's merged
//! tokens, each row finite, the first and last rows apart, and the first item
//! encoded twice gives the same bits. With `IGNIS_TOWER_OUT` set it writes the
//! dump `tower_check.py` reads, `<out>/<family>/`: `manifest.json`
//! (`ignis-vision-tower-dump/1`), and per item `<id>.patches.bf16` (the
//! processor's patch rows, the encode's input) and `<id>.embedding.bf16` (the
//! embedding rows the pool holds, `[tokens][hidden]`), BF16 bits,
//! little-endian.
//!
//! ```text
//! IGNIS_TOWER_IMAGES=H:/fn-vision-560/tower-images IGNIS_TOWER_OUT=H:/fn-vision-560/tower-dump ^
//!   cargo test -p ignis-server --features cuda --test vision_tower_dump_gpu -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Machine-local: Flash-Next in `F:/ai/models/Qwen3.8-Flash-Next-ignis/` with
//! its vision companion (or `IGNIS_FLASH_NEXT_DIR`), ~40 GB of free host RAM
//! for its expert pool; the 27B at `F:\ai\q38\ninfer-models`. One load at a
//! time. Explicit GPU profile (ADR 0006): outside `IGNIS_GPU_PROFILE=1` a
//! missing artifact or GPU is a skip, under it a failure.

#![cfg(feature = "cuda")]

#[path = "support/mod.rs"]
mod support;

use std::path::{Path, PathBuf};

use ignis_artifact::flash_next::FlashNextGeometry;
use ignis_artifact::packer::ARTIFACT_FILE_NAME;
use ignis_artifact::vision::{PreparedMedia, PATCH_FEATURES};
use ignis_artifact::{bind_model_scope_27b_with, materialize, CudaDevice, FrontendSet, Reader};
use ignis_core::compute::ModelConfig;
use ignis_core::flash_next::EngineOptions;
use ignis_core::gpu_profile;
use ignis_core::model_load::{load_qwen38_27b_with_options, model_scope};
use ignis_core::step;
use ignis_core::vision::{vision_item_control, MediaItem, TokenSpan};
use ignis_core::{ExceptionWeights, KvFormat, RopeScaling, Vision, VisionResidency};
use ignis_runtime::{FlashNextLeaf, StepLeaf};
use serde_json::json;

const FLASH_NEXT_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
const ARTIFACT_27B: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
/// The loads' context and vision envelope: room for every item, whole.
const MAX_CONTEXT: u32 = 8192;
/// The largest item the set holds: what the CPU reference runs in minutes.
const MAX_ITEM_TOKENS: u64 = 4096;
const PREFILL_CHUNK: u32 = 1024;

/// One image of the set: its id in the dump, where it came from, its bytes.
struct Image {
    id: String,
    source: String,
    bytes: Vec<u8>,
}

/// One item as the leaf encoded it.
struct Encoded {
    image: Image,
    media: PreparedMedia,
    rows: Vec<u16>,
}

fn png_bytes(width: u32, height: u32, pixels: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().expect("png header");
        writer.write_image_data(pixels).expect("png data");
    }
    out
}

/// A deterministic `width` x `height` picture with structure at more than one
/// scale: `kind` picks the drawing, a low xorshift grain sits under it.
fn synthetic(width: u32, height: u32, kind: &str) -> Vec<u8> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ (u64::from(width) << 32 | u64::from(height));
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    let (cx, cy) = (width as f32 / 2.0, height as f32 / 2.0);
    let scale = width.min(height).max(1) as f32;
    for y in 0..height {
        for x in 0..width {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let grain = (state >> 59) as f32;
            let (fx, fy) = (x as f32, y as f32);
            let rgb: [f32; 3] = match kind {
                "rings" => {
                    let r = ((fx - cx).hypot(fy - cy) / scale * 12.0) as u32;
                    [[220.0, 40.0, 40.0], [30.0, 120.0, 220.0], [240.0, 210.0, 60.0], [250.0, 250.0, 250.0]][r as usize % 4]
                }
                "checker" => {
                    let cell = (scale / 8.0).max(4.0) as u32;
                    if (x / cell + y / cell) % 2 == 0 { [20.0, 20.0, 20.0] } else { [235.0, 235.0, 220.0] }
                }
                "stripes" => {
                    let band = ((fx + 2.0 * fy) / (scale / 6.0)) as u32;
                    [[60.0, 170.0, 80.0], [200.0, 90.0, 200.0], [250.0, 140.0, 30.0]][band as usize % 3]
                }
                _ => {
                    let inside = (x / (width / 5).max(1)) % 2 == 1 && (y / (height / 4).max(1)) % 2 == 0;
                    if inside {
                        [10.0, 40.0, 120.0]
                    } else {
                        [255.0 * fx / width as f32, 255.0 * fy / height as f32, 128.0]
                    }
                }
            };
            pixels.extend(rgb.iter().map(|c| (c + grain).clamp(0.0, 255.0) as u8));
        }
    }
    png_bytes(width, height, &pixels)
}

/// The synthetic part of the set: sizes from below the processor's floor
/// (upscaled to it) to past 2,000 merged tokens, square to 1:17.
const SYNTHETIC: [(u32, u32, &str); 16] = [
    (32, 32, "rings"),
    (64, 64, "checker"),
    (200, 100, "stripes"),
    (100, 300, "blocks"),
    (448, 448, "rings"),
    (512, 288, "checker"),
    (300, 900, "stripes"),
    (1024, 256, "blocks"),
    (256, 1024, "rings"),
    (640, 480, "checker"),
    (800, 800, "stripes"),
    (1280, 720, "blocks"),
    (1440, 1440, "rings"),
    (2048, 512, "checker"),
    (96, 1600, "stripes"),
    (777, 333, "blocks"),
];

/// The set, in dump order: the canaries, the synthetic images, the extra ones.
fn images() -> Vec<Image> {
    let canaries = support::vision_canary::fixture_dir();
    let mut out: Vec<Image> = ["number", "colour", "circles", "text"]
        .iter()
        .map(|name| {
            let path = canaries.join(format!("{name}.png"));
            Image {
                id: format!("canary-{name}"),
                source: format!("crates/server/tests/fixtures/vision_canary/{name}.png"),
                bytes: std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display())),
            }
        })
        .collect();
    out.extend(SYNTHETIC.iter().map(|&(w, h, kind)| Image {
        id: format!("synthetic-{kind}-{w}x{h}"),
        source: format!("vision_tower_dump_gpu.rs synthetic({w}, {h}, \"{kind}\")"),
        bytes: synthetic(w, h, kind),
    }));
    if let Some(dir) = std::env::var_os("IGNIS_TOWER_IMAGES") {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("IGNIS_TOWER_IMAGES {}: {e}", Path::new(&dir).display()))
            .map(|entry| entry.expect("a directory entry").path())
            .filter(|p| {
                matches!(p.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).as_deref(), Some("png" | "jpg" | "jpeg"))
            })
            .collect();
        paths.sort();
        out.extend(paths.into_iter().map(|path| Image {
            id: format!("extra-{}", path.file_stem().expect("a file name").to_string_lossy()),
            source: path.file_name().expect("a file name").to_string_lossy().into_owned(),
            bytes: std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display())),
        }));
    }
    out
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Every image through `frontend`'s vision processor and `encode`, checked
/// item by item, the first item encoded twice.
fn encode_set(
    family: &str,
    frontend: &FrontendSet,
    hidden: usize,
    mut encode: impl FnMut(&MediaItem) -> Vec<u16>,
) -> Vec<Encoded> {
    let processor = frontend.vision_processor().unwrap_or_else(|e| panic!("{family}: vision processor: {e}"));
    let mut out = Vec::new();
    for (index, image) in images().into_iter().enumerate() {
        let media = processor
            .prepare_media(index, &image.bytes)
            .unwrap_or_else(|e| panic!("{family} {}: prepare: {e}", image.id));
        let tokens = media.grid.vision_tokens();
        assert!(
            tokens <= MAX_ITEM_TOKENS,
            "{} is {tokens} merged tokens, past the set's {MAX_ITEM_TOKENS}: resize it",
            image.source
        );
        assert_eq!(media.patches.len(), media.grid.raw_patches() as usize * PATCH_FEATURES);
        let item = MediaItem {
            grid: media.grid,
            token_span: TokenSpan { begin: 0, count: tokens as usize },
            patches: media.patches.clone(),
            content_digest: media.content_digest,
        };
        let rows = encode(&item);
        assert_eq!(rows.len(), tokens as usize * hidden, "{family} {}: one row per merged token", image.id);
        let non_finite = rows.iter().filter(|&&b| b & 0x7f80 == 0x7f80).count();
        assert_eq!(non_finite, 0, "{family} {}: {non_finite} values are not finite", image.id);
        if tokens > 1 {
            assert_ne!(rows[..hidden], rows[rows.len() - hidden..], "{family} {}: the first and last rows", image.id);
        }
        if index == 0 {
            assert_eq!(encode(&item), rows, "{family} {}: a second encode of one item", image.id);
        }
        println!(
            "{family} {}: grid {}x{}x{}, {tokens} tokens",
            image.id, media.grid.t, media.grid.h, media.grid.w
        );
        out.push(Encoded { image, media, rows });
    }
    out
}

fn le_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The dump `tower_check.py` reads, when `IGNIS_TOWER_OUT` names where.
fn write_dump(family: &str, artifact: &Path, hidden: usize, encoded: &[Encoded]) {
    let Some(out) = std::env::var_os("IGNIS_TOWER_OUT") else {
        println!("{family}: IGNIS_TOWER_OUT is not set, no dump written");
        return;
    };
    let dir = PathBuf::from(out).join(family);
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
    let mut items = Vec::new();
    for e in encoded {
        let (patches, embedding) = (format!("{}.patches.bf16", e.image.id), format!("{}.embedding.bf16", e.image.id));
        std::fs::write(dir.join(&patches), le_bytes(&e.media.patches)).expect("write the patches");
        std::fs::write(dir.join(&embedding), le_bytes(&e.rows)).expect("write the embedding");
        items.push(json!({
            "id": e.image.id,
            "image": e.image.source,
            "image_sha256": hex(&e.media.content_digest),
            "source_pixels": [e.media.source_pixels.0, e.media.source_pixels.1],
            "grid": [e.media.grid.t, e.media.grid.h, e.media.grid.w],
            "tokens": e.media.grid.vision_tokens(),
            "patches": patches,
            "embedding": embedding,
        }));
    }
    let manifest = json!({
        "schema": "ignis-vision-tower-dump/1",
        "family": family,
        "artifact": artifact.file_name().expect("a file name").to_string_lossy(),
        "residency": "host",
        "hidden": hidden,
        "patch_features": PATCH_FEATURES,
        "items": items,
    });
    let text = serde_json::to_string_pretty(&manifest).expect("the manifest");
    std::fs::write(dir.join("manifest.json"), text).expect("write the manifest");
    println!("{family}: {} items dumped in {}", encoded.len(), dir.display());
}

#[test]
#[ignore = "GPU profile only: the Flash-Next artifact, its vision companion and the card exclusively"]
fn flash_nexts_companion_encodes_the_set_as_served() {
    let dir = std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(FLASH_NEXT_DIR), PathBuf::from);
    let artifact = dir.join(ARTIFACT_FILE_NAME);
    let companion = ignis_core::flash_next_vision::companion_path(&dir);
    for path in [&artifact, &companion] {
        if !path.exists() && gpu_profile::skip_or_fail(&format!("absent: {}", path.display())) {
            return;
        }
    }
    let frontend = FrontendSet::from_reader(&Reader::open(&artifact).expect("open the artifact")).expect("frontend");
    let hidden = ModelConfig::flash_next_from(&FlashNextGeometry::qwen38_flash_next()).hidden as usize;
    let options = EngineOptions {
        max_context_tokens: MAX_CONTEXT,
        prefill_chunk_tokens: PREFILL_CHUNK,
        vision: Some(Vision::new(MAX_CONTEXT).expect("an envelope").with_residency(VisionResidency::Host)),
        ..EngineOptions::default()
    };
    let leaf = match FlashNextLeaf::open(&artifact, options) {
        Ok(leaf) => leaf,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("open the Flash-Next leaf with vision: {e}"));
            return;
        }
    };
    let model = leaf.load_model().unwrap_or_else(|code| panic!("load the model: leaf code {code}"));
    let encoded = encode_set("qwen3.8-flash-next", &frontend, hidden, |item| {
        let media = leaf.encode_media(&model, item).unwrap_or_else(|code| panic!("encode: leaf code {code}"));
        let rows = media.read_bits(hidden as u64).unwrap_or_else(|e| panic!("read the embedding: {e}"));
        leaf.release_media(&model, media);
        rows
    });
    write_dump("qwen3.8-flash-next", &artifact, hidden, &encoded);
    leaf.release_model(model);
}

#[test]
#[ignore = "GPU profile only: the 27B artifact and the card exclusively"]
fn the_27bs_tower_encodes_the_set_as_served() {
    let artifact = Path::new(ARTIFACT_27B);
    if !artifact.exists() && gpu_profile::skip_or_fail(&format!("absent: {ARTIFACT_27B}")) {
        return;
    }
    let reader = Reader::open(artifact).unwrap_or_else(|e| panic!("open {ARTIFACT_27B}: {e}"));
    let frontend = FrontendSet::from_reader(&reader).expect("frontend");
    let config = ModelConfig::qwen38_27b();
    let hidden = config.hidden as usize;
    let mut device = match CudaDevice::create(0) {
        Ok(d) => d,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("CUDA device unavailable: {e}"));
            return;
        }
    };
    let item_bound = MAX_ITEM_TOKENS as u32;
    let vision = Vision::new(MAX_CONTEXT)
        .expect("an envelope")
        .with_item_max_tokens(item_bound)
        .with_residency(VisionResidency::Host);
    let (plan, handles) = bind_model_scope_27b_with(&reader, model_scope(None, Some(vision), ExceptionWeights::Bf16))
        .unwrap_or_else(|e| panic!("bind: {e}"));
    let mut weights = match materialize(&reader, &plan, &mut device, None) {
        Ok(a) => a,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("materialize: {e}"));
            return;
        }
    };
    let model = load_qwen38_27b_with_options(
        &reader,
        &weights,
        &handles,
        PREFILL_CHUNK,
        MAX_CONTEXT,
        KvFormat::Bf16,
        None,
        Some(vision),
        RopeScaling::NONE,
    )
    .unwrap_or_else(|e| panic!("ignis_model_load: {e}"));
    let encoded = encode_set("qwen3.8-27b", &frontend, hidden, |item| {
        let embedding = step::encode_media(&model, item.grid, &item.patches, &vision_item_control(item.grid))
            .unwrap_or_else(|(rc, e)| panic!("encode {rc}: {e}"));
        embedding.read_bits(hidden as u64).unwrap_or_else(|e| panic!("read the embedding: {e}"))
    });
    write_dump("qwen3.8-27b", artifact, hidden, &encoded);
    drop(model);
    let _ = weights.release_arena(&mut device);
}
