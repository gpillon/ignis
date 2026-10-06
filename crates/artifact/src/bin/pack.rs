//! ignis-artifact-pack: assemble the Flash-Next `.ninfer` v2 container from
//! the converter's work tree (spec flash-next/01; the work-file contract is
//! `docs/specs/flash-next/layout.md`, the mechanics `ignis_artifact::packer`).
//!
//! Usage:
//!   ignis-artifact-pack --work <work dir> [--out <artifact.ninfer>]
//!                       [--layers N] [--experts N] [--header-mib N]
//!                       [--accept-status STATUS] [--keep-work] [--weights-id ID]
//!                       [--geometry qwen3.8-flash-next|fixture]
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

use std::path::PathBuf;

use ignis_artifact::flash_next::FlashNextGeometry;
use ignis_artifact::packer::{pack, PackOptions, PackOutcome, ARTIFACT_FILE_NAME};

const USAGE: &str = "usage: ignis-artifact-pack --work <dir> [--out <artifact.ninfer>] [--layers N] \
                     [--experts N] [--header-mib N] [--accept-status STATUS] [--keep-work] \
                     [--weights-id ID] [--geometry qwen3.8-flash-next|fixture]";

fn main() {
    let mut work = None;
    let mut out = None;
    let mut layers = None;
    let mut experts = None;
    let mut header_mib = None;
    let mut accept_status = Vec::new();
    let mut keep_work = false;
    let mut weights_id = None;
    let mut geometry = FlashNextGeometry::qwen38_flash_next();
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
            "--geometry" => {
                geometry = match args.next().as_deref() {
                    Some("qwen3.8-flash-next") => FlashNextGeometry::qwen38_flash_next(),
                    Some("fixture") => FlashNextGeometry::fixture(),
                    _ => usage_error("--geometry takes qwen3.8-flash-next or fixture"),
                }
            }
            other => usage_error(&format!("unknown argument: {other}")),
        }
    }
    let Some(work_dir) = work else {
        usage_error("--work is required");
    };
    let artifact = match out {
        Some(out) => out,
        None => match work_dir.parent() {
            Some(model_dir) => model_dir.join(ARTIFACT_FILE_NAME),
            None => usage_error("--work has no parent directory: pass --out"),
        },
    };
    if let Some(layers) = layers {
        geometry.layers = layers as usize;
    }
    if let Some(experts) = experts {
        geometry.experts = experts;
    }

    let mut options = PackOptions::new(work_dir, artifact, geometry);
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
