//! `vendor-ninfer` — copy, verify and patch-record the vendored reference
//! subtree in the kernel leaf (ADR 0010). See `kernel/vendor/VENDOR.md`.
//!
//! Exit codes: 0 clean, 1 a verification finding, 2 a usage or I/O error.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ignis_vendor::manifest::{Manifest, Patch};
use ignis_vendor::sha256::sha256_hex;
use ignis_vendor::vendor;

const USAGE: &str = "\
vendor-ninfer — the kernel leaf's vendoring tool (ADR 0010)

USAGE:
    vendor-ninfer <COMMAND> [OPTIONS]

COMMANDS:
    verify              Check the vendor tree against the manifest. With
                        --reference, also check the reference checkout still
                        carries the pinned content.
    sync                Copy every manifest file from the reference into the
                        vendor tree. Refuses to copy anything unless the
                        reference matches the pinned hashes.
    repin               Recompute the manifest's reference hashes from the
                        checkout (use after moving the pinned commit).
    record-patch PATH   Record the local edit of one vendored file as a
                        committed diff plus a patched hash.

OPTIONS:
    --manifest PATH     The manifest (default: kernel/vendor/manifest.json).
    --reference PATH    The reference checkout (default: the manifest's
                        default_path, or $IGNIS_NINFER_REFERENCE).
    --force-patched     sync: overwrite files that carry a recorded patch,
                        discarding the patch.
    --reason TEXT       record-patch: why the patch exists (required).
    -h, --help          This text.
";

const DEFAULT_MANIFEST: &str = "kernel/vendor/manifest.json";

struct Options {
    command: String,
    argument: Option<String>,
    manifest: PathBuf,
    reference: Option<PathBuf>,
    force_patched: bool,
    reason: Option<String>,
}

fn parse_options() -> Result<Options, String> {
    let mut arguments = std::env::args().skip(1);
    let mut options = Options {
        command: String::new(),
        argument: None,
        manifest: PathBuf::from(DEFAULT_MANIFEST),
        reference: std::env::var_os("IGNIS_NINFER_REFERENCE").map(PathBuf::from),
        force_patched: false,
        reason: None,
    };

    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "-h" | "--help" => return Err(String::new()),
            "--manifest" => {
                options.manifest = arguments
                    .next()
                    .ok_or("--manifest needs a path")?
                    .into();
            }
            "--reference" => {
                options.reference =
                    Some(arguments.next().ok_or("--reference needs a path")?.into());
            }
            "--reason" => {
                options.reason = Some(arguments.next().ok_or("--reason needs a text")?);
            }
            "--force-patched" => options.force_patched = true,
            other if other.starts_with('-') => return Err(format!("unknown option {other}")),
            other if options.command.is_empty() => options.command = other.to_string(),
            other if options.argument.is_none() => options.argument = Some(other.to_string()),
            other => return Err(format!("unexpected argument {other}")),
        }
    }

    if options.command.is_empty() {
        return Err("no command given".into());
    }
    Ok(options)
}

/// The vendor root is the manifest's directory: the manifest sits at its root,
/// so a moved checkout needs no path rewriting.
fn vendor_root(manifest_path: &Path) -> PathBuf {
    manifest_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn reference_root(options: &Options, manifest: &Manifest) -> PathBuf {
    options
        .reference
        .clone()
        .unwrap_or_else(|| PathBuf::from(&manifest.reference.default_path))
}

fn print_report(label: &str, report: &vendor::Report) {
    if report.is_clean() {
        println!("{label}: {} files, all matching", report.checked);
        return;
    }
    eprintln!(
        "{label}: {} of {} files do not match",
        report.findings.len(),
        report.checked
    );
    for finding in &report.findings {
        eprintln!("  {finding}");
    }
}

fn run_verify(options: &Options, manifest: &Manifest) -> Result<bool, String> {
    let root = vendor_root(&options.manifest);
    let tree = vendor::verify_vendor_tree(&root, manifest)
        .map_err(|error| format!("{}: {error}", root.display()))?;
    print_report("vendor tree", &tree);

    let reference = reference_root(options, manifest);
    if !reference.exists() {
        println!(
            "reference: {} is not on this machine — skipping the source check",
            reference.display()
        );
        return Ok(tree.is_clean());
    }
    let source = vendor::verify_reference(&reference, manifest)
        .map_err(|error| format!("{}: {error}", reference.display()))?;
    print_report(
        &format!("reference @ {}", &manifest.reference.commit[..12.min(manifest.reference.commit.len())]),
        &source,
    );
    Ok(tree.is_clean() && source.is_clean())
}

fn run_sync(options: &Options, manifest: &Manifest) -> Result<bool, String> {
    let root = vendor_root(&options.manifest);
    let reference = reference_root(options, manifest);
    let report = vendor::sync(&reference, &root, manifest, options.force_patched)
        .map_err(|error| error.to_string())?;
    println!("synced {} files from {}", report.copied.len(), reference.display());
    for path in &report.kept_patched {
        println!("  kept (patched, not overwritten): {path}");
    }
    let tree = vendor::verify_vendor_tree(&root, manifest)
        .map_err(|error| format!("{}: {error}", root.display()))?;
    print_report("vendor tree", &tree);
    Ok(tree.is_clean())
}

fn run_repin(options: &Options, manifest: &mut Manifest) -> Result<bool, String> {
    let reference = reference_root(options, manifest);
    let changed = vendor::refresh_reference_hashes(&reference, manifest)
        .map_err(|error| format!("{}: {error}", reference.display()))?;
    manifest
        .store(&options.manifest)
        .map_err(|error| error.to_string())?;
    if changed.is_empty() {
        println!("repin: the manifest already matches {}", reference.display());
    } else {
        println!("repin: {} hashes updated", changed.len());
        for path in &changed {
            println!("  {path}");
        }
        println!("Set reference.commit to the checkout's revision before committing.");
    }
    Ok(true)
}

fn run_record_patch(options: &Options, manifest: &mut Manifest) -> Result<bool, String> {
    let path = options
        .argument
        .clone()
        .ok_or("record-patch needs the vendored file's manifest path")?;
    let reason = options
        .reason
        .clone()
        .ok_or("record-patch needs --reason (a patch without a reason is an accident)")?;

    let root = vendor_root(&options.manifest);
    let reference = reference_root(options, manifest);
    if manifest.file(&path).is_none() {
        return Err(format!("{path} is not in the manifest"));
    }

    let local = root.join(&path);
    let bytes = std::fs::read(&local).map_err(|error| format!("{}: {error}", local.display()))?;
    let patched_sha256 = sha256_hex(&bytes);

    let diff_relative = format!("patches/{path}.diff");
    let diff_path = root.join(&diff_relative);
    if let Some(parent) = diff_path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| format!("{}: {error}", parent.display()))?;
    }
    let diff = vendor::unified_diff(&reference.join(&path), &local, &path)?;
    std::fs::write(&diff_path, diff).map_err(|error| format!("{}: {error}", diff_path.display()))?;

    let entry = manifest.file_mut(&path).expect("checked above");
    if patched_sha256 == entry.sha256 {
        return Err(format!(
            "{path} is byte-identical to the reference — there is no patch to record"
        ));
    }
    entry.patch = Some(Patch {
        diff: diff_relative.clone(),
        sha256: patched_sha256,
        reason,
    });
    manifest
        .store(&options.manifest)
        .map_err(|error| error.to_string())?;
    println!("recorded {path} -> {diff_relative}");
    Ok(true)
}

fn main() -> ExitCode {
    let options = match parse_options() {
        Ok(options) => options,
        Err(message) => {
            if message.is_empty() {
                print!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            eprintln!("vendor-ninfer: {message}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };

    let mut manifest = match Manifest::load(&options.manifest) {
        Ok(manifest) => manifest,
        Err(error) => {
            eprintln!("vendor-ninfer: {error}");
            return ExitCode::from(2);
        }
    };

    let outcome = match options.command.as_str() {
        "verify" => run_verify(&options, &manifest),
        "sync" => run_sync(&options, &manifest),
        "repin" => run_repin(&options, &mut manifest),
        "record-patch" => run_record_patch(&options, &mut manifest),
        other => Err(format!("unknown command {other}")),
    };

    match outcome {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(message) => {
            eprintln!("vendor-ninfer: {message}");
            ExitCode::from(2)
        }
    }
}
