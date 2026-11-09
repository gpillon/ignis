//! Flash-Next's `none` rotary table, byte for byte, across the change that
//! taught its kernels YaRN (ADR 0059, spec rope-scaling/02 AC 9): the
//! unscaled table multiplies cos and sin by exactly 1.0f, an IEEE identity,
//! so a `none` load must give the very logits it gave before the change.
//!
//! - **`none` is the logits recorded before the change.** A real-text prompt
//!   of 2,560 tokens (the converter's `long8192` window 0) in 1,024-token
//!   chunks, so the last chunk runs the indexer's sparse selection: every
//!   row's BF16 logits hash to what the capture recorded.
//! - **A `yarn:4` load moves every row but the first** (AC 10): the same
//!   prompt, every logit finite and every row's bytes but the first's other
//!   than the capture's, so the scaled table reaches the program at all.
//!   The first row attends to itself alone, and a softmax over one key is 1
//!   whatever the table makes of its score, so it must stay the capture's.
//!   The argmax agreement with `none` is printed, not judged.
//! - **Vision on is the capture too** (GitHub #557): the same prompt on a
//!   load with Flash-Next's embedding pool, host and device, a text span
//!   taking the text path -- every row's hash the capture's.
//!
//! The capture is per row: the SHA-256 of the row's BF16 logits and its
//! argmax. It is committed, `tests/fixtures/flash_next/none_logits_2560.txt`,
//! beside a `.provenance.json` naming the card, the driver and the build
//! that recorded it (`hq_kv_fixture_integrity.rs` holds the file to its
//! SHA-256). A bit-exact comparison only means something on that card and
//! driver -- another card's split decisions reorder the sums -- so on any
//! other, or with no capture, the test goes through `skip_or_fail` (ADR
//! 0006) before it loads anything; and since a later kernel change may move
//! these bits legitimately, it is behind the non-default `logits-capture`
//! feature, which the GPU profile never builds:
//!
//! ```text
//! cargo test -p ignis-core --features cuda,logits-capture --test flash_next_rope_logits_gpu -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `IGNIS_RECORD_FLASH_NEXT_LOGITS=1` records the capture and its provenance
//! instead (on the build before a change); without it the test compares.
//! `IGNIS_FLASH_NEXT_LOGITS_CAPTURE` names another capture than the
//! committed one (its provenance beside it, `<name>.provenance.json`).
//!
//! Machine-local: `F:/ai/models/Qwen3.8-Flash-Next-ignis/` (or
//! `IGNIS_FLASH_NEXT_DIR`). Needs ~38 GB of free RAM for the pinned expert
//! pool.

#![cfg(all(feature = "cuda", feature = "logits-capture"))]

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use ignis_core::flash_next::{EngineOptions, FlashNextEngine};
use ignis_core::{gpu_profile, KvFormat, RopeScaling, Vision, VisionResidency};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const MODEL_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
/// Past the indexer's dense threshold (2,051 visible tokens), so the last
/// chunk's rows select blocks.
const PROMPT_TOKENS: usize = 2560;

fn model_dir() -> PathBuf {
    std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(MODEL_DIR), PathBuf::from)
}

/// The committed capture.
fn committed_capture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/flash_next/none_logits_2560.txt")
}

fn capture_path() -> PathBuf {
    std::env::var_os("IGNIS_FLASH_NEXT_LOGITS_CAPTURE").map_or_else(committed_capture, PathBuf::from)
}

/// A capture's provenance: `none_logits_2560.txt` beside
/// `none_logits_2560.provenance.json`.
fn provenance_path(capture: &Path) -> PathBuf {
    capture.with_extension("provenance.json")
}

/// The first [`PROMPT_TOKENS`] tokens of the converter's `long8192` window 0.
fn prompt(dir: &Path) -> Option<Vec<u32>> {
    let path = dir.join("references").join("long8192").join("tokens.u32");
    let Ok(bytes) = fs::read(&path) else {
        gpu_profile::skip_or_fail(&format!("{} does not exist", path.display()));
        return None;
    };
    let tokens: Vec<u32> = bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().expect("four bytes"))).collect();
    Some(tokens[..PROMPT_TOKENS].to_vec())
}

/// One row of the capture: its logits' SHA-256 (hex) and argmax.
#[derive(Debug, Clone, PartialEq)]
struct Row {
    sha256: String,
    argmax: u32,
}

fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn hex(digest: &[u8]) -> String {
    digest.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// Every row of `tokens` teacher-forced through a load of `options`.
fn rows(dir: &Path, options: EngineOptions, tokens: &[u32]) -> Option<Vec<Row>> {
    let mut engine = match FlashNextEngine::load(dir, options) {
        Ok(engine) => engine,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("load the Flash-Next engine: {e}"));
            return None;
        }
    };
    let vocab = engine.vocab();
    let mut out = Vec::with_capacity(tokens.len());
    engine
        .span_logits(tokens, &mut |first, logits| {
            assert_eq!(first, out.len(), "rows arrive in order");
            for row in logits.chunks_exact(vocab) {
                assert!(row.iter().all(|&v| bf16(v).is_finite()), "row {} holds a non-finite logit", out.len());
                let bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
                let sha256 = hex(&Sha256::digest(&bytes));
                let argmax = (0..vocab).max_by(|&a, &b| bf16(row[a]).total_cmp(&bf16(row[b])).then(b.cmp(&a))).expect("a row");
                out.push(Row { sha256, argmax: argmax as u32 });
            }
            Ok(())
        })
        .unwrap_or_else(|e| panic!("span logits: {e}"));
    assert_eq!(out.len(), tokens.len());
    Some(out)
}

/// The card and the driver a capture's bits repeat on.
#[derive(Debug, Clone, PartialEq)]
struct Machine {
    gpu: String,
    driver: String,
}

/// This machine's first card and its driver, as nvidia-smi names them.
fn this_machine() -> Result<Machine, String> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=name,driver_version", "--format=csv,noheader"])
        .output()
        .map_err(|e| format!("run nvidia-smi: {e}"))?;
    if !out.status.success() {
        return Err(format!("nvidia-smi exited with {}", out.status));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text.lines().next().ok_or("nvidia-smi named no card")?;
    let (gpu, driver) = line.split_once(',').ok_or_else(|| format!("nvidia-smi answered {line:?}"))?;
    Ok(Machine { gpu: gpu.trim().to_owned(), driver: driver.trim().to_owned() })
}

/// Why a capture whose provenance is `provenance` cannot be compared on
/// `machine`, or `None` when it can: its bits repeat only on the card and
/// the driver that recorded them.
fn not_comparable_on(provenance: &Value, machine: &Machine) -> Option<String> {
    let recorded = |key: &str| provenance[key].as_str();
    match (recorded("gpu"), recorded("driver")) {
        (Some(gpu), Some(driver)) if gpu == machine.gpu && driver == machine.driver => None,
        (Some(gpu), Some(driver)) => Some(format!(
            "the capture was recorded on {gpu} with driver {driver}, and this is {} with driver {}: \
             its bits repeat only there (IGNIS_RECORD_FLASH_NEXT_LOGITS=1 on the build before a \
             change records one here)",
            machine.gpu, machine.driver
        )),
        _ => Some("the capture's provenance names no card or driver".to_owned()),
    }
}

/// The capture's rows, or why they cannot be read.
fn parse_capture(text: &str) -> Result<Vec<Row>, String> {
    text.lines()
        .enumerate()
        .map(|(i, line)| match line.split(' ').collect::<Vec<_>>()[..] {
            [row, sha256, argmax] if row == i.to_string() => {
                let argmax = argmax.parse().map_err(|e| format!("line {i}: the argmax {argmax:?}: {e}"))?;
                Ok(Row { sha256: sha256.to_owned(), argmax })
            }
            _ => Err(format!("line {i} is not `{i} <sha256> <argmax>`: {line:?}")),
        })
        .collect()
}

/// The capture to compare against, or `None` -- through `skip_or_fail` --
/// when it is missing, unreadable, or was recorded on another card or
/// driver than this one.
fn recorded_capture() -> Option<Vec<Row>> {
    let path = capture_path();
    let read = |path: &Path| fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()));
    let checked = || -> Result<Vec<Row>, String> {
        let text = read(&path)?;
        let sidecar = provenance_path(&path);
        let provenance: Value =
            serde_json::from_str(&read(&sidecar)?).map_err(|e| format!("parse {}: {e}", sidecar.display()))?;
        let machine = this_machine().map_err(|e| format!("which card and driver this is: {e}"))?;
        if let Some(why) = not_comparable_on(&provenance, &machine) {
            return Err(why);
        }
        parse_capture(&text).map_err(|e| format!("{}: {e}", path.display()))
    };
    match checked() {
        Ok(rows) => Some(rows),
        Err(why) => {
            gpu_profile::skip_or_fail(&format!("no capture to compare against: {why}"));
            None
        }
    }
}

/// Record `rows` and their provenance (this card, this driver, this
/// checkout's commit) as the capture at `path`.
fn write_capture(path: &Path, rows: &[Row]) {
    let text: String = rows.iter().enumerate().map(|(i, r)| format!("{i} {} {}\n", r.sha256, r.argmax)).collect();
    let machine = this_machine().unwrap_or_else(|e| panic!("name the card that records the capture: {e}"));
    let build = Command::new("git")
        .args(["-C", env!("CARGO_MANIFEST_DIR"), "rev-parse", "--short=8", "HEAD"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map_or_else(|| "unknown".to_owned(), |out| String::from_utf8_lossy(&out.stdout).trim().to_owned());
    let seconds = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let provenance = json!({
        "build": build,
        "capture_unix_seconds": seconds,
        "captured_by": "crates/core/tests/flash_next_rope_logits_gpu.rs (IGNIS_RECORD_FLASH_NEXT_LOGITS=1)",
        "driver": machine.driver,
        "gpu": machine.gpu,
        "prompt": "references/long8192/tokens.u32 under the artifact's directory, window 0, tokens [0, 2560)",
        "rows": rows.len(),
        "txt_bytes": text.len(),
        "txt_sha256": hex(&Sha256::digest(text.as_bytes())),
    });
    fs::create_dir_all(path.parent().expect("a parent")).expect("the capture's directory");
    fs::write(path, text).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    let sidecar = provenance_path(path);
    let json = serde_json::to_string_pretty(&provenance).expect("the provenance") + "\n";
    fs::write(&sidecar, json).unwrap_or_else(|e| panic!("write {}: {e}", sidecar.display()));
}

/// The load both runs of the comparison make: BF16 KV (the oracle format),
/// 1,024-token chunks, no graphs (a span runs none).
fn options() -> EngineOptions {
    EngineOptions {
        prefill_chunk_tokens: 1024,
        max_context_tokens: 8192,
        kv_format: KvFormat::Bf16,
        capture_graphs: false,
        ..EngineOptions::default()
    }
}

#[test]
#[ignore = "GPU, logits-capture feature: bit-exact on the recording card and driver only"]
fn a_none_load_gives_the_logits_recorded_before_the_change() {
    let dir = model_dir();
    let Some(tokens) = prompt(&dir) else { return };
    let recorded = if std::env::var_os("IGNIS_RECORD_FLASH_NEXT_LOGITS").is_some() {
        None
    } else {
        let Some(recorded) = recorded_capture() else { return };
        Some(recorded)
    };
    let Some(none) = rows(&dir, options(), &tokens) else { return };
    let Some(recorded) = recorded else {
        let path = capture_path();
        write_capture(&path, &none);
        println!("recorded {} rows to {}", none.len(), path.display());
        return;
    };
    assert_eq!(recorded.len(), none.len(), "the capture's rows");
    let differing: Vec<usize> = (0..none.len()).filter(|&i| none[i] != recorded[i]).collect();
    println!("none against the capture: {} of {} rows differ", differing.len(), none.len());
    assert!(differing.is_empty(), "a none load moved rows {:?}", &differing[..differing.len().min(16)]);
}

/// GitHub #557: vision on (its embedding pool, host and device) moves no row
/// of a text span from the capture: a span without media takes the text path.
#[test]
#[ignore = "GPU, logits-capture feature: bit-exact on the recording card and driver only"]
fn a_vision_load_gives_the_none_capture() {
    let dir = model_dir();
    let Some(tokens) = prompt(&dir) else { return };
    let Some(recorded) = recorded_capture() else { return };
    for residency in [VisionResidency::Host, VisionResidency::Device] {
        let vision = Vision::new(8192).expect("an envelope").with_residency(residency);
        let options = EngineOptions { vision: Some(vision), ..options() };
        let Some(rows) = rows(&dir, options, &tokens) else { return };
        assert_eq!(recorded.len(), rows.len(), "the capture's rows");
        let differing: Vec<usize> = (0..rows.len()).filter(|&i| rows[i] != recorded[i]).collect();
        println!("vision ({residency:?}) against the capture: {} of {} rows differ", differing.len(), rows.len());
        assert!(differing.is_empty(), "a vision load ({residency:?}) moved rows {:?}", &differing[..differing.len().min(16)]);
    }
}

#[test]
#[ignore = "GPU, logits-capture feature: against the capture of the recording card and driver"]
fn a_yarn_load_moves_every_row_but_the_first_from_the_none_capture() {
    let dir = model_dir();
    let Some(tokens) = prompt(&dir) else { return };
    let Some(recorded) = recorded_capture() else { return };
    let options = EngineOptions { rope_scaling: RopeScaling::yarn(4.0).expect("yarn:4"), ..options() };
    let Some(yarn) = rows(&dir, options, &tokens) else { return };
    assert_eq!(recorded.len(), yarn.len(), "the capture's rows");
    let unmoved: Vec<usize> = (0..yarn.len()).filter(|&i| yarn[i].sha256 == recorded[i].sha256).collect();
    let agree = (0..yarn.len()).filter(|&i| yarn[i].argmax == recorded[i].argmax).count();
    println!(
        "yarn:4 against the none capture: {} of {} rows unmoved, argmax agrees on {agree}",
        unmoved.len(),
        yarn.len()
    );
    assert_eq!(unmoved, [0], "the rows a yarn:4 load leaves as none's: the first alone, which attends to itself");
}

/// No GPU: a capture is compared only on the card and the driver its
/// provenance names, and the committed one parses whole.
#[test]
fn a_capture_is_compared_only_on_the_card_and_driver_that_recorded_it() {
    let here = Machine { gpu: "NVIDIA GeForce RTX 5090".into(), driver: "610.88".into() };
    let provenance = json!({ "gpu": "NVIDIA GeForce RTX 5090", "driver": "610.88", "build": "67f8d2e5" });
    assert_eq!(not_comparable_on(&provenance, &here), None);
    let other_card = Machine { gpu: "NVIDIA RTX PRO 6000 Blackwell Server Edition".into(), ..here.clone() };
    let why = not_comparable_on(&provenance, &other_card).expect("another card");
    assert!(why.contains("RTX 5090") && why.contains("RTX PRO 6000"), "{why}");
    let other_driver = Machine { driver: "611.02".into(), ..here.clone() };
    assert!(not_comparable_on(&provenance, &other_driver).expect("another driver").contains("611.02"));
    assert!(not_comparable_on(&json!({ "build": "67f8d2e5" }), &here).is_some(), "a provenance naming no card");

    let committed = committed_capture();
    let provenance: Value =
        serde_json::from_str(&fs::read_to_string(provenance_path(&committed)).expect("its provenance")).expect("JSON");
    assert_eq!((provenance["gpu"].as_str(), provenance["driver"].as_str()), (Some(here.gpu.as_str()), Some("610.88")));
    let rows = parse_capture(&fs::read_to_string(&committed).expect("the committed capture")).expect("it parses");
    assert_eq!(rows.len(), PROMPT_TOKENS);
    assert!(parse_capture("0 ab 1\n2 cd 3\n").is_err(), "a row out of order");
    assert!(parse_capture("0 ab\n").is_err(), "a row without its argmax");
}
