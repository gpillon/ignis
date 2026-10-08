//! KV-disk's writes against Flash-Next's n-gram reads on one volume,
//! measured (spec vram-budget/03 AC 25, ADR 0045). No GPU: the real n-gram
//! table of the Flash-Next artifact, uncovered (no hot row, so every row a
//! gather stages is read from the file), and the tier's own IO threads
//! writing a 1 GiB blob as 32 MiB unbuffered windows beside it.
//!
//! On the card a spill the scheduler starts never overlaps a prefill chunk:
//! the arrival that needs the room is the prefill head, and waits for it
//! (`kv_disk_contention_gpu` records that). What a chunk shares with the
//! tier is the volume, through any write the tier has in flight when the
//! chunk's gather starts -- so this measures the gather, the part of a
//! chunk the volume can slow:
//!
//! - **alone**: chunks' gathers with nothing else on the volume, and the
//!   1 GiB written with nothing else on it (the tier's throughput);
//! - **gated**: the same, with the blob being written through threads that
//!   wait on the table's prefill gate (production);
//! - **ungated**: the same with a gate no gather raises -- what the gate
//!   buys.
//!
//! Each chunk is its gather then `CHUNK_COMPUTE` of idle (the device time of
//! a Flash-Next chunk past its gather, measured on the card), during which a
//! gated writer runs. Machine-local (`IGNIS_FLASH_NEXT_DIR`, files beside
//! the artifact's volume under `IGNIS_KV_DISK_TEST_DIR`); prints, and writes
//! the samples to `IGNIS_KV_P2_RAW` as JSON when set.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ignis_artifact::flash_next::{self, FlashNextGeometry};
use ignis_artifact::packer::ARTIFACT_FILE_NAME;
use ignis_artifact::{AlignedBuffer, DirectWriter, Reader};
use ignis_core::compute::ModelConfig;
use ignis_core::ngram_table::{GatherGate, NgramTable, NgramTableOptions};
use ignis_runtime::kv_disk::{Bytes, IoThreads, Job, Ticket, WINDOW_BYTES};

const FLASH_NEXT_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
const CHUNK_TOKENS: u32 = 8192;
const CHUNKS: u32 = 6;
const BLOB_WINDOWS: usize = 32;
/// A Flash-Next 8,192-token chunk's wall time past its gather, on the card
/// (2,835 ms less 1,144 ms, `kv_disk_contention_gpu` 2026-10-08).
const CHUNK_COMPUTE: Duration = Duration::from_millis(1690);

fn tokens(seed: u32) -> Vec<u32> {
    (0..CHUNK_TOKENS).map(|i| 1000 + (i.wrapping_mul(7919) ^ seed.wrapping_mul(104_729)) % 240_000).collect()
}

/// Queue the blob's windows; the tickets, and the buffers they write from.
fn write_blob(io: &IoThreads, file: &Arc<DirectWriter>) -> (Vec<Ticket>, Vec<AlignedBuffer>) {
    let mut buffers: Vec<AlignedBuffer> = (0..BLOB_WINDOWS)
        .map(|w| {
            let mut buffer = AlignedBuffer::new(WINDOW_BYTES as usize).expect("an aligned window");
            buffer.as_mut_slice().fill(w as u8);
            buffer
        })
        .collect();
    let tickets = buffers
        .iter_mut()
        .enumerate()
        .map(|(w, buffer)| {
            let slice = buffer.as_mut_slice();
            io.submit(Job::Write {
                file: Arc::clone(file),
                offset: w as u64 * WINDOW_BYTES,
                bytes: Bytes { ptr: slice.as_mut_ptr(), len: slice.len() },
                crc_len: slice.len(),
                copy_from: None,
            })
        })
        .collect();
    (tickets, buffers)
}

struct Leg {
    name: &'static str,
    gathers_ms: Vec<f64>,
    /// Whether the blob was still being written when each gather ended.
    overlapped: Vec<bool>,
    write_ms: Option<f64>,
}

fn run(table: &NgramTable, dir: &Path, name: &'static str, gate: Option<GatherGate>, first_seed: u32) -> Leg {
    let mut out = vec![0u8; CHUNK_TOKENS as usize * table.token_bytes()];
    let writing = gate.map(|gate| {
        let io = IoThreads::start(gate).expect("the IO threads");
        let file = Arc::new(DirectWriter::create(&dir.join(format!("{name}.kv"))).expect("the blob file"));
        let started = Instant::now();
        let (tickets, buffers) = write_blob(&io, &file);
        // When the last window lands, timed off the model thread's loop.
        let waiting = tickets.clone();
        let finished = std::thread::spawn(move || {
            for ticket in &waiting {
                ticket.wait().expect("a window written");
            }
            started.elapsed()
        });
        (io, file, tickets, buffers, finished)
    });
    let mut gathers_ms = Vec::new();
    let mut overlapped = Vec::new();
    for chunk in 0..CHUNKS {
        let mut context = table.new_context();
        let started = Instant::now();
        table.stage(&mut context, &tokens(first_seed + chunk), &mut out).expect("a gather");
        gathers_ms.push(started.elapsed().as_secs_f64() * 1e3);
        overlapped.push(writing.as_ref().is_some_and(|(_, _, tickets, _, _)| tickets.iter().any(|t| t.poll().is_none())));
        std::thread::sleep(CHUNK_COMPUTE);
    }
    let write_ms = writing.map(|(io, file, tickets, buffers, finished)| {
        let ms = finished.join().expect("the timer").as_secs_f64() * 1e3;
        drop((io, file, tickets, buffers));
        ms
    });
    Leg { name, gathers_ms, overlapped, write_ms }
}

/// The blob alone on the volume: the tier's write throughput.
fn write_alone(dir: &Path) -> f64 {
    let io = IoThreads::start(GatherGate::default()).expect("the IO threads");
    let file = Arc::new(DirectWriter::create(&dir.join("alone.kv")).expect("the blob file"));
    let started = Instant::now();
    let (tickets, buffers) = write_blob(&io, &file);
    for ticket in &tickets {
        ticket.wait().expect("a window written");
    }
    let ms = started.elapsed().as_secs_f64() * 1e3;
    drop((io, file, buffers));
    ms
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

#[test]
#[ignore = "machine-local: the Flash-Next artifact and its volume, ~2 min"]
fn kv_disk_writes_beside_uncovered_ngram_gathers_on_one_volume() {
    let artifact_dir = std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(FLASH_NEXT_DIR), PathBuf::from);
    let path = artifact_dir.join(ARTIFACT_FILE_NAME);
    if !path.exists() {
        eprintln!("skipped: no Flash-Next artifact at {}", path.display());
        return;
    }
    let dir = std::env::var_os("IGNIS_KV_DISK_TEST_DIR")
        .map_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.scratch/kv-disk-gpu"), PathBuf::from)
        .join("volume");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let reader = Reader::open(&path).expect("open the artifact");
    let geometry = FlashNextGeometry::qwen38_flash_next();
    let plan = flash_next::bind(&reader, &geometry).expect("bind the artifact");
    let ngram = ModelConfig::flash_next_from(&geometry).ngram.expect("an n-gram embedding");
    let options = NgramTableOptions { hot_bytes: 0, ..NgramTableOptions::default() };
    let table = NgramTable::from_artifact(&path, &reader, &plan, ngram, options).expect("the n-gram table");
    drop(reader);

    let alone_write_ms = write_alone(&dir);
    let alone = run(&table, &dir, "alone", None, 100);
    let gated = run(&table, &dir, "gated", Some(table.prefill_gate()), 200);
    let ungated = run(&table, &dir, "ungated", Some(GatherGate::default()), 300);

    let gib = (BLOB_WINDOWS as f64 * WINDOW_BYTES as f64) / f64::from(1u32 << 30);
    println!("1 GiB blob written alone in {alone_write_ms:.0} ms: {:.2} GiB/s", gib / (alone_write_ms / 1e3));
    for leg in [&alone, &gated, &ungated] {
        println!(
            "{:>8}: gather median {:.1} ms {:?}, overlapped {:?}, blob written in {}",
            leg.name,
            median(&leg.gathers_ms),
            leg.gathers_ms.iter().map(|g| g.round()).collect::<Vec<_>>(),
            leg.overlapped,
            leg.write_ms.map_or("-".to_string(), |ms| format!("{ms:.0} ms")),
        );
    }
    if let Some(raw) = std::env::var_os("IGNIS_KV_P2_RAW") {
        let leg = |l: &Leg| {
            format!(
                "{{\"name\":\"{}\",\"gathers_ms\":{:?},\"overlapped\":{:?},\"write_ms\":{}}}",
                l.name,
                l.gathers_ms,
                l.overlapped,
                l.write_ms.map_or("null".to_string(), |ms| format!("{ms:.3}"))
            )
        };
        let json = format!(
            "{{\"chunk_tokens\":{CHUNK_TOKENS},\"blob_bytes\":{},\"alone_write_ms\":{alone_write_ms:.3},\"legs\":[{},{},{}]}}\n",
            BLOB_WINDOWS as u64 * WINDOW_BYTES,
            leg(&alone),
            leg(&gated),
            leg(&ungated)
        );
        std::fs::write(&raw, json).expect("write the samples");
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(gated.overlapped.iter().any(|&o| o), "the gated blob was written beside a gather");
}
