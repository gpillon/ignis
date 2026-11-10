//! ignis-artifact-pack: assemble the Flash-Next `.ninfer` v2 container from
//! the converter's work tree (spec flash-next/01; the work-file contract is
//! `docs/formats/flash-next-layout.md`, the mechanics `ignis_artifact::packer`).
//!
//! Usage:
//!   ignis-artifact-pack --work <work dir> [--out <artifact.ninfer>]
//!                       [--layers N] [--experts N] [--header-mib N]
//!                       [--accept-status STATUS] [--keep-work] [--weights-id ID]
//!                       [--geometry qwen3.8-flash-next|fixture]
//!                       [--family main|mtp --pair-main <main.ninfer>]
//!   ignis-artifact-pack --family flash-next-vision --work <work dir> --pair-main <main.ninfer>
//!                       [--out <artifact.ninfer>] [--geometry qwen3.8-flash-next|fixture]
//!                       [--header-mib N] [--accept-status STATUS] [--keep-work]
//!   ignis-artifact-pack --family qwen-image --work <work dir> [--out <artifact.ninfer>]
//!                       [--header-mib N] [--accept-status STATUS] [--keep-work]
//!   ignis-artifact-pack --family qwen3.8-27b-trellis --work <work dir> [--out <artifact.ninfer>]
//!                       [--header-mib N] [--accept-status STATUS] [--keep-work]
//!   ignis-artifact-pack --family qwen3.8-27b-nvfp4 --work <work dir> --out <artifact.ninfer>
//!                       --weights-id ID [--geometry qwen3.8-27b|fixture]
//!                       [--header-mib N] [--accept-status STATUS] [--keep-work]
//!
//! Run it after the converter ends (`work/converter.json` with status
//! `complete`; until then it waits and deletes nothing). It appends every
//! unit (frontend, global, ngram, then the layers), deleting work files once
//! they are in, and finishes the container after layer N-1 (`--layers`,
//! default the checkpoint's 48). `--out` defaults to
//! `<work>/../qwen3_8_flash_next_trellis_a25-v2.ninfer`. Safe to run again at
//! any point: it resumes where the last run stopped.
//!
//! For test packs only: `--accept-status dry-run` (or `fixture`) packs a
//! dry run's record, `--layers` / `--experts` / `--geometry fixture` match a
//! reduced tree, and `--keep-work` deletes nothing (refused without room for
//! two copies). Exit status: 0 when the container is finished, 3 when it
//! waits, 1 on error.
//!
//! `--family mtp` packs the MTP head's companion container (flash-next-layout.md §13)
//! from the head converter's work tree (its one unit `mtp/`), pinned to the
//! main container `--pair-main`; `--out` then defaults to
//! `<work>/../qwen3_8_flash_next_mtp_3p0-v2.ninfer`, the identity and the
//! 1 MiB header are the companion's.
//!
//! `--family flash-next-vision` packs Flash-Next's vision companion
//! (flash-next-layout.md §14, ADR 0087) from `tools/flash-next-vision`'s work
//! tree (its one unit `vision/`, holding exactly the objects of
//! `ignis_artifact::flash_next::vision::entries`), pinned to the main
//! container `--pair-main`; `--out` then defaults to
//! `<work>/../qwen3_8_flash_next_vision-v2.ninfer`, the identity and the
//! 1 MiB header are the companion's. `--geometry fixture` packs the tool's
//! synthetic tower; `--layers` and `--experts` are refused.
//!
//! `--family qwen-image` packs the image model's container (spec
//! image-model/01) from `tools/image-converter`'s work tree: `frontend/` and
//! its four tensor units, no Flash-Next geometry (`--layers`, `--experts` and
//! `--geometry` are refused); `--out` defaults to
//! `<work>/../ignis_image_2_1_turbo-v2.ninfer`.
//!
//! `--family qwen3.8-27b-trellis` packs the 27B's 16 GB artifact (spec
//! 27b-16gb/01) from `tools/27b-16gb-converter`'s work tree: `frontend/` and
//! the units `text`, `mtp`, `vision`, `dflash2`, no Flash-Next geometry;
//! `--out` defaults to `<work>/../qwen3_8_27b_trellis_sc30-v2.ninfer`.
//!
//! `--family qwen3.8-27b-nvfp4` packs an NVFP4 container of the 27B (GitHub
//! #467), the served one's layout for the base or a fine-tune, from the
//! NVFP4 converter's work tree (`docs/formats/27b-nvfp4-work-tree.md`):
//! `frontend/` and the units `text`, `mtp`, `vision`, `dflash2` holding the
//! 1,325 objects of `ignis_artifact::nvfp4_27b::inventory`, and
//! `converter.json`. `--weights-id` and `--out` are required: the weights id
//! is what tells one build of the layout from another, and the file name is
//! the build's own. `--geometry fixture` packs the converter's reduced
//! synthetic tree. It packs only a whole tree, and refuses one that is not
//! the inventory's before it writes anything.

use std::path::PathBuf;

use ignis_artifact::flash_next::FlashNextGeometry;
use ignis_artifact::nvfp4_27b;
use ignis_artifact::packer::{
    pack, Family, PackOptions, PackOutcome, ARTIFACT_FILE_NAME, MTP_ARTIFACT_FILE_NAME, QWEN_IMAGE_ARTIFACT_FILE_NAME,
    TRELLIS_27B_ARTIFACT_FILE_NAME, VISION_ARTIFACT_FILE_NAME,
};
use ignis_artifact::flash_next::vision;

const USAGE: &str = "usage: ignis-artifact-pack --work <dir> [--out <artifact.ninfer>] [--layers N] \
                     [--experts N] [--header-mib N] [--accept-status STATUS] [--keep-work] \
                     [--weights-id ID] [--geometry qwen3.8-flash-next|qwen3.8-27b|fixture] \
                     [--family main|mtp|flash-next-vision|qwen-image|qwen3.8-27b-trellis|qwen3.8-27b-nvfp4 \
                     --pair-main <main.ninfer>]\n\
                     --family qwen3.8-27b-nvfp4: --work <NVFP4 work tree: frontend/ text/ mtp/ vision/ dflash2/ \
                     converter.json> --out <artifact.ninfer> --weights-id ID (docs/formats/27b-nvfp4-work-tree.md)";

fn main() {
    let mut work = None;
    let mut out = None;
    let mut layers = None;
    let mut experts = None;
    let mut header_mib = None;
    let mut accept_status = Vec::new();
    let mut keep_work = false;
    let mut weights_id = None;
    let mut geometry_name = None;
    let mut family = None;
    let mut pair_main = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--work" => work = args.next().map(PathBuf::from),
            "--out" => out = args.next().map(PathBuf::from),
            "--layers" => layers = Some(positive(args.next(), "--layers")),
            "--experts" => experts = Some(positive(args.next(), "--experts")),
            "--header-mib" => header_mib = Some(positive(args.next(), "--header-mib")),
            "--accept-status" => match args.next() {
                Some(status) => accept_status.push(status),
                None => usage_error("--accept-status takes a status"),
            },
            "--keep-work" => keep_work = true,
            "--weights-id" => weights_id = args.next(),
            // read once --family is known: each family names its own
            "--geometry" => match args.next() {
                Some(name) => geometry_name = Some(name),
                None => usage_error("--geometry takes a geometry"),
            },
            "--family" => family = args.next(),
            "--pair-main" => pair_main = args.next().map(PathBuf::from),
            other => usage_error(&format!("unknown argument: {other}")),
        }
    }
    let Some(work_dir) = work else {
        usage_error("--work is required");
    };
    let family = match (family.as_deref(), pair_main) {
        (None | Some("main"), None) => Family::Main,
        (Some("mtp"), Some(pair_main)) => Family::Mtp { pair_main },
        (Some("mtp"), None) => usage_error("--family mtp needs --pair-main"),
        (Some("flash-next-vision"), Some(pair_main)) => Family::FlashNextVision {
            pair_main,
            tower: match geometry_name.as_deref() {
                None | Some("qwen3.8-flash-next") => vision::geometry(),
                Some("fixture") => vision::fixture_geometry(),
                Some(_) => usage_error("--geometry takes qwen3.8-flash-next or fixture for --family flash-next-vision"),
            },
        },
        (Some("flash-next-vision"), None) => usage_error("--family flash-next-vision needs --pair-main"),
        (Some("qwen-image"), None) => Family::QwenImage21,
        (Some("qwen3.8-27b-trellis"), None) => Family::Qwen27bTrellis,
        (Some("qwen3.8-27b-nvfp4"), None) => Family::Qwen27bNvfp4 {
            geometry: match geometry_name.as_deref() {
                None | Some("qwen3.8-27b") => nvfp4_27b::Geometry::qwen3_8_27b(),
                Some("fixture") => nvfp4_27b::Geometry::fixture(),
                Some(_) => usage_error("--geometry takes qwen3.8-27b or fixture for --family qwen3.8-27b-nvfp4"),
            },
        },
        (None | Some("main") | Some("qwen-image") | Some("qwen3.8-27b-trellis") | Some("qwen3.8-27b-nvfp4"), Some(_)) => {
            usage_error("--pair-main is for --family mtp or flash-next-vision")
        }
        _ => usage_error("--family takes main, mtp, flash-next-vision, qwen-image, qwen3.8-27b-trellis or qwen3.8-27b-nvfp4"),
    };
    let mut geometry = match (&family, geometry_name.as_deref()) {
        (Family::Main | Family::Mtp { .. }, None | Some("qwen3.8-flash-next")) => FlashNextGeometry::qwen38_flash_next(),
        (Family::Main | Family::Mtp { .. }, Some("fixture")) => FlashNextGeometry::fixture(),
        (Family::Main | Family::Mtp { .. }, Some(_)) => usage_error("--geometry takes qwen3.8-flash-next or fixture"),
        (Family::QwenImage21 | Family::Qwen27bTrellis, Some(_)) => not_flash_next(FLASH_NEXT_FLAGS),
        // unused by the other families
        _ => FlashNextGeometry::qwen38_flash_next(),
    };
    if layers.is_some() || experts.is_some() {
        match family {
            Family::Main | Family::Mtp { .. } => {}
            Family::Qwen27bNvfp4 { .. } | Family::FlashNextVision { .. } => not_flash_next("--layers and --experts"),
            _ => not_flash_next(FLASH_NEXT_FLAGS),
        }
    }
    let artifact = match out {
        Some(out) => out,
        None => match work_dir.parent() {
            Some(model_dir) => model_dir.join(match family {
                Family::Main => ARTIFACT_FILE_NAME,
                Family::Mtp { .. } => MTP_ARTIFACT_FILE_NAME,
                Family::FlashNextVision { .. } => VISION_ARTIFACT_FILE_NAME,
                Family::QwenImage21 => QWEN_IMAGE_ARTIFACT_FILE_NAME,
                Family::Qwen27bTrellis => TRELLIS_27B_ARTIFACT_FILE_NAME,
                Family::Qwen27bNvfp4 { .. } => usage_error("--family qwen3.8-27b-nvfp4 needs --out: the build's own file name"),
            }),
            None => usage_error("--work has no parent directory: pass --out"),
        },
    };
    if let Some(layers) = layers {
        geometry.layers = layers as usize;
    }
    if let Some(experts) = experts {
        geometry.experts = experts;
    }

    let mut options = match family {
        Family::Mtp { pair_main } => PackOptions::mtp(work_dir, artifact, geometry, pair_main),
        Family::FlashNextVision { pair_main, tower } => PackOptions::flash_next_vision(work_dir, artifact, pair_main, tower),
        Family::Main => PackOptions::new(work_dir, artifact, geometry),
        Family::QwenImage21 => PackOptions::qwen_image(work_dir, artifact),
        Family::Qwen27bTrellis => PackOptions::trellis_27b(work_dir, artifact),
        Family::Qwen27bNvfp4 { geometry } => match weights_id.as_deref() {
            Some(id) => PackOptions::nvfp4_27b(work_dir, artifact, id, geometry),
            None => usage_error("--family qwen3.8-27b-nvfp4 needs --weights-id: it is what tells one build of the layout from another"),
        },
    };
    if let Some(mib) = header_mib {
        options.header_bytes = match mib.checked_mul(1 << 20) {
            Some(bytes) => bytes,
            None => usage_error("--header-mib is too large"),
        };
    }
    if let Some(id) = weights_id {
        options.identity.weights_id = id;
    }
    options.accept_status.extend(accept_status);
    options.keep_work = keep_work;

    match pack(&options, &mut |line| eprintln!("{line}")) {
        Ok(PackOutcome::Finished {
            file_bytes,
            object_count,
            converter_merged,
        }) => {
            println!(
                "finished {}: {object_count} objects, {file_bytes} bytes{}",
                options.artifact.display(),
                if converter_merged {
                    ""
                } else {
                    " (converter.json not written yet: run again after the converter ends to merge it into the sidecar)"
                }
            );
        }
        Ok(PackOutcome::Waiting {
            appended,
            total,
            next,
        }) => {
            println!("waiting: {appended}/{total} units appended, {next} is not complete yet");
            std::process::exit(3);
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

fn positive(value: Option<String>, flag: &str) -> u64 {
    match value.and_then(|v| v.parse::<u64>().ok()) {
        Some(n) if n > 0 => n,
        _ => usage_error(&format!("{flag} takes a positive integer")),
    }
}

fn usage_error(message: &str) -> ! {
    eprintln!("{message}\n{USAGE}");
    std::process::exit(2);
}

/// Flash-Next's own flags, which a family without its geometry refuses.
const FLASH_NEXT_FLAGS: &str = "--layers, --experts and --geometry";

/// Refuses `flags` for a family that has no Flash-Next geometry.
fn not_flash_next(flags: &str) -> ! {
    usage_error(&format!("{flags} are Flash-Next's, not this --family's"))
}
