//! One-shot capture of a small corpus of real BF16 K/V rows from a real
//! prefill of the 27B artifact, committed as
//! `kernel/tests/fixtures/hq_kv_rows_27b.bin` (+ `.provenance.json`) so the
//! hq-e8-2b codec's op-level oracle (P4-03, GitHub #119) can measure its
//! round-trip error against real activations and derive its tolerance from
//! that measurement, without loading 19 GB of weights on every CTest run.
//!
//! This is a *recording* test, not a regression test: running it overwrites
//! the committed fixture with a fresh capture. Run it only when the fixture
//! needs to be re-captured (the artifact or the prompt below changes) --
//! never as part of a normal green run. What keeps `cargo test` honest about
//! the committed files *not* silently drifting between recordings is the
//! companion cheap, non-GPU
//! `crates/core/tests/hq_kv_fixture_integrity.rs`, which is deliberately
//! left ungated so it runs on every plain `cargo test`.
//!
//! Row readback goes through `Seq::capture_kv_rows_for_test`
//! (`crates/core/src/seq.rs`), which calls the leaf's test-only diagnostic
//! seam `ignis_kv_capture_rows` (`kernel/include/ignis_kv_capture.h`) --
//! neither is part of the production forward-pass ABI (ADR 0009); both
//! exist solely for this file, and both are gated behind this crate's
//! non-default `kv-capture` feature (never on in a production build).
//!
//! Because of that gate, this file is no longer reachable from
//! `scripts/gpu-profile.ps1` (which passes only `--features cuda`) -- on
//! purpose: this is a one-shot recording tool, not something that should
//! re-run on every GPU profile pass. To re-capture the fixture, stop
//! ninfer and run:
//!
//! ```text
//! cargo test -p ignis-core --features cuda,kv-capture -- --ignored \
//!     hq_kv_fixture_capture_gpu --test-threads=1
//! ```

#![cfg(all(feature = "cuda", feature = "kv-capture"))]

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use ignis_artifact::{CudaDevice, FrontendSet, Reader, bind_text_scope_27b, materialize};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::load_qwen38_27b;
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step::prefill_program;
use sha2::{Digest, Sha256};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const ARTIFACT_SHA256_SIDECAR: &str =
    r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer.sha256";
const MODEL_ID: &str = "qwen3.8-27b-nvfp4full-v2";
const MAX_CONTEXT: u32 = 1024;

/// The capture prompt: one fixed, human-readable passage, committed here
/// verbatim so the capture is reproducible byte-for-byte from source alone
/// (no external corpus file to go stale or disappear). Long enough to span
/// several 64-token KV pages once tokenized -- the exact token count is
/// reported by the test (BPE tokenization is not stable across model
/// vocabularies, so it is measured, not assumed).
const PROMPT: &str = "Keeper's Log, Third Year at the Point\n\n\
The lighthouse stood at the far end of the point, where the land narrowed to a spit of grey stone and the sea took over completely. I had come there in early September, when the tourists had gone home and the ferry ran only twice a week, carrying mail, diesel, and whoever was fool enough to want the job. My predecessor left a short note pinned to the inside of the pantry door: the paraffin runs low faster in the northeast wind, mind the fog signal's compressor on wet mornings, and never trust the barometer during a falling glass in October. I read it three times before I understood he meant it as a warning, not a joke.\n\n\
The first winter taught me more about the mechanism than any manual could. The lamp itself was modest by then, an electric bulb behind a rotating Fresnel lens that had been ground somewhere in France a century earlier, its panels beveled to bend light into a single disciplined beam. The old clockwork that once turned the lens by falling weights had been replaced by a small motor, but the housing still smelled faintly of the whale oil it once burned, a smell that never quite left the brass fittings no matter how many coats of paint went over them. I would climb the spiral stair twice a day, once at dusk to light the lamp and once before dawn to check the logbook and the weather glass, and each time the iron treads rang a little differently underfoot depending on how much damp had gotten into the tower overnight.\n\n\
Fog was the real adversary, not storms. A storm announces itself: the barometer falls, the gulls go quiet, the swell builds a day in advance if you know how to read it. Fog arrives without ceremony, rolling in off the bank in a matter of minutes, swallowing the beam whole so that from the rocks below you could no longer tell the light was burning at all. That was what the horn was for, a compressed-air signal that groaned out over the water every thirty seconds, a sound with no beauty in it but enormous purpose. On the worst nights, when the compressor iced up and had to be hand-cranked, I understood why keepers before me had gone a little strange out here, talking to the gulls, keeping elaborate private calendars, arguing with the sea as if it could be reasoned with.\n\n\
Supplies came by boat, weather permitting, and in a bad month permitting meant nothing at all. I learned to stretch a fortnight's stores into six weeks, to ration paraffin and flour and the handful of books I had brought with me, rereading each one until the spines gave out. The isolation was not the hardship people imagine from the mainland; the hardship was the waiting, the long stretches when nothing at all happened and the only variable in the day was which direction the wind chose to blow the rain. I kept the log meticulously anyway, noting wind speed and direction, visibility, any vessel sighted, any bird blown far off its course and found resting, exhausted, on the gallery rail. Entry after entry, most of them identical, until an entry that was not, and those were the ones I remembered.\n\n\
There was a night in November when a trawler lost its bearings in the fog and came in far closer to the point than any vessel had a right to. I heard the engine before I saw the running lights, a low diesel throb underneath the horn's groan, and for a long minute I stood on the gallery convinced I was about to watch a wreck happen a few hundred yards from where I stood, powerless to do anything but keep the light turning and the horn sounding. The engine note changed, sharpened, and the lights swung hard away into the murk. I never learned who was aboard or how close it truly came; the log entry for that night is three lines long and says nothing about how my hands would not stop shaking afterward.\n\n\
By the second spring I had stopped counting the days until relief and started noticing the place instead: the seals that hauled out on the lee rocks every low tide, indifferent to the horn and the light both; the exact week in April when the terns returned and began their furious, repetitive defense of a rock they had claimed the year before and would claim again; the particular green the sea turned in the hour before a storm, a color I never managed to describe adequately in the log no matter how many adjectives I tried. The mechanism I had once found intimidating became, by then, almost meditative to maintain: the daily climb, the cleaning of the lens panels with a soft cloth and a prescribed solvent, the careful winding and checking of the backup mechanism in case the motor failed, the logging of every observation in the same terse, factual hand my predecessor had used, and his predecessor before him, an unbroken chain of keepers each adding their few lines to a record that would outlast every one of us individually.\n\n\
When my relief finally came, on a clear morning in late May with the sea flat calm and the terns already screaming over their rock, I found I did not want to leave nearly as much as I had expected to. I wrote the final entry, packed the two books I had not yet finished, and climbed down the spiral stair one last time, counting the rings of the iron treads out of habit, already missing a sound I would not hear again for a long while.";

/// The four GQA layer ordinals captured (spread across the 16-layer GQA
/// stack, not an absolute backbone layer index -- `Seq`/
/// `ignis_seq_internal.h`'s own 0..15 numbering).
const GQA_LAYER_ORDINALS: [i32; 4] = [0, 5, 10, 15];
/// 0 = K, 1 = V (`kernel/src/seq.cu`'s plane order: `2 * layer + role`).
const ROLES: [i32; 2] = [0, 1];
/// Positions [0, ROWS_PER_BLOCK) are captured per (layer, role, kv_head) --
/// the first four of this GQA layer's 64-token pages, so the addressing
/// exercises three physical page boundaries per block even though the
/// prefill itself spans many more pages than this.
const ROWS_PER_BLOCK: i32 = 256;

const FIXTURE_MAGIC: &[u8; 8] = b"IGNHQKV1";
const FIXTURE_FORMAT_VERSION: u32 = 1;

/// Corruption detection, not cryptography (the provenance file's SHA-256 of
/// the whole `.bin` is what actually authenticates it) -- a plain FNV-1a 64
/// over the row payload bytes only (the header is excluded so re-deriving
/// `total_rows`/etc. never invalidates an otherwise-untouched payload).
fn fnv1a64(data: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET_BASIS;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// BF16 bit pattern -> f32 (widen into the top 16 bits), for the sanity
/// check below only -- mirrors `kernel/vendor/tests/ops/op_tester.h`'s
/// `bf16_to_f32` host convention.
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernel/tests/fixtures")
}

/// Unix seconds -> `YYYY-MM-DDTHH:MM:SSZ`. No `chrono` dependency for one
/// provenance timestamp: the standard proleptic-Gregorian, no-leap-seconds
/// civil-from-days algorithm (Howard Hinnant,
/// http://howardhinnant.github.io/date_algorithms.html#civil_from_days).
fn format_unix_utc(unix_secs: u64) -> String {
    let days = (unix_secs / 86_400) as i64;
    let secs_of_day = unix_secs % 86_400;
    let (hour, minute, second) = (secs_of_day / 3600, (secs_of_day / 60) % 60, secs_of_day % 60);

    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let day = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let month = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = if month <= 2 { yoe as i64 + era * 400 + 1 } else { yoe as i64 + era * 400 };

    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1 -- one-shot fixture capture, GitHub #119"]
fn capture_hq_kv_fixture_from_a_real_prefill() {
    let artifact_path = Path::new(ARTIFACT);
    if !artifact_path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}"))
    {
        return;
    }

    let reader = Reader::open(artifact_path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let full_prompt: Vec<i32> = frontend
        .tokenizer()
        .encode(PROMPT)
        .unwrap_or_else(|e| panic!("tokenize prompt: {e}"))
        .into_iter()
        .map(|id| i32::try_from(id).expect("token id fits i32"))
        .collect();
    assert!(!full_prompt.is_empty());

    let prompt_tokens: Vec<i32> = if full_prompt.len() > MAX_CONTEXT as usize {
        full_prompt[..MAX_CONTEXT as usize].to_vec()
    } else {
        full_prompt.clone()
    };
    let used_token_count = prompt_tokens.len();
    assert!(
        used_token_count >= ROWS_PER_BLOCK as usize,
        "capture prompt ({used_token_count} tokens) must cover at least the {ROWS_PER_BLOCK} \
         captured positions -- extend PROMPT"
    );

    let (plan, handles) = bind_text_scope_27b(&reader).unwrap_or_else(|e| panic!("bind: {e}"));
    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA unavailable: {e}")) {
                return;
            }
            unreachable!();
        }
    };
    let artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize: {e}")) {
                return;
            }
            unreachable!();
        }
    };

    let model = load_qwen38_27b(&reader, &artifact, &handles, MAX_CONTEXT, MAX_CONTEXT, ignis_core::KvFormat::Bf16)
        .unwrap_or_else(|e| panic!("load model: {e}"));
    let cfg = ModelConfig::qwen38_27b();
    let pool = SeqPool::create(
        &cfg,
        &SeqPoolBudget {
            kv_format: ignis_core::KvFormat::Bf16,
            kv_page_group_count: MAX_CONTEXT / 64,
            max_context_tokens: MAX_CONTEXT,
            slot_count: 1,
        },
    )
    .unwrap_or_else(|e| panic!("seq pool: {e}"));
    let mut sequence = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc seq: {e}"));
    prefill_program(&model, &pool, &mut sequence, &prompt_tokens, 0, None)
        .unwrap_or_else(|e| panic!("prefill: {e}"));

    let head_dim = cfg.head_dim as i32;
    let kv_heads = cfg.num_kv_heads as i32;

    let mut payload: Vec<u16> = Vec::with_capacity(
        GQA_LAYER_ORDINALS.len()
            * ROLES.len()
            * kv_heads as usize
            * ROWS_PER_BLOCK as usize
            * head_dim as usize,
    );
    let mut total_rows: usize = 0;
    let mut nonzero_rows: usize = 0;
    let mut layer0_k_rows: Vec<Vec<u16>> = Vec::new();
    let mut layer0_k_abs_mean: Vec<f64> = Vec::new();

    for &layer in &GQA_LAYER_ORDINALS {
        for &role in &ROLES {
            for kv_head in 0..kv_heads {
                let rows = sequence
                    .capture_kv_rows_for_test(layer, role, kv_head, 0, ROWS_PER_BLOCK, head_dim)
                    .unwrap_or_else(|e| {
                        panic!("capture layer={layer} role={role} kv_head={kv_head}: {e}")
                    });
                assert_eq!(rows.len(), (ROWS_PER_BLOCK as usize) * (head_dim as usize));
                total_rows += ROWS_PER_BLOCK as usize;
                if rows.iter().any(|&v| v != 0) {
                    nonzero_rows += ROWS_PER_BLOCK as usize;
                }
                if layer == GQA_LAYER_ORDINALS[0] && role == 0 {
                    let abs_mean = rows.iter().map(|&v| f64::from(bf16_to_f32(v).abs())).sum::<f64>()
                        / rows.len() as f64;
                    layer0_k_abs_mean.push(abs_mean);
                    layer0_k_rows.push(rows.clone());
                }
                payload.extend_from_slice(&rows);
            }
        }
    }

    assert!(nonzero_rows > 0, "captured rows must not be all zero");
    let heads_differ = layer0_k_rows.windows(2).any(|pair| pair[0] != pair[1]);
    assert!(heads_differ, "different kv_heads captured bit-identical rows");

    let mut row_bytes = Vec::with_capacity(payload.len() * 2);
    for value in &payload {
        row_bytes.extend_from_slice(&value.to_le_bytes());
    }
    let payload_checksum = fnv1a64(&row_bytes);

    // Header layout (all little-endian): magic[8], format_version u32,
    // head_dim u32, kv_heads u32, role_count u32, layer_count u32,
    // layer_ordinals[layer_count] u32, first_position u32, rows_per_block
    // u32, total_rows u32, checksum_fnv1a64 u64 -- then the row payload,
    // ordered `for layer in layer_ordinals { for role in [K, V] { for
    // kv_head in 0..kv_heads { for position in
    // first_position..first_position+rows_per_block { head_dim x bf16 bit
    // pattern, u16 LE } } } }`. Mirrored in the provenance JSON below.
    let mut header = Vec::new();
    header.extend_from_slice(FIXTURE_MAGIC);
    header.extend_from_slice(&FIXTURE_FORMAT_VERSION.to_le_bytes());
    header.extend_from_slice(&(head_dim as u32).to_le_bytes());
    header.extend_from_slice(&(kv_heads as u32).to_le_bytes());
    header.extend_from_slice(&(ROLES.len() as u32).to_le_bytes());
    header.extend_from_slice(&(GQA_LAYER_ORDINALS.len() as u32).to_le_bytes());
    for &layer in &GQA_LAYER_ORDINALS {
        header.extend_from_slice(&(layer as u32).to_le_bytes());
    }
    header.extend_from_slice(&0u32.to_le_bytes()); // first_position
    header.extend_from_slice(&(ROWS_PER_BLOCK as u32).to_le_bytes());
    header.extend_from_slice(&(total_rows as u32).to_le_bytes());
    header.extend_from_slice(&payload_checksum.to_le_bytes());

    let mut bin_bytes = header;
    bin_bytes.extend_from_slice(&row_bytes);

    let dir = fixtures_dir();
    fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
    let bin_path = dir.join("hq_kv_rows_27b.bin");
    fs::write(&bin_path, &bin_bytes).unwrap_or_else(|e| panic!("write {}: {e}", bin_path.display()));

    let bin_sha256 = {
        let mut hasher = Sha256::new();
        hasher.update(&bin_bytes);
        format!("{:x}", hasher.finalize())
    };
    let artifact_sha256 = fs::read_to_string(ARTIFACT_SHA256_SIDECAR)
        .unwrap_or_else(|e| panic!("read {ARTIFACT_SHA256_SIDECAR}: {e}"))
        .split_whitespace()
        .next()
        .unwrap_or_else(|| panic!("{ARTIFACT_SHA256_SIDECAR} has no digest"))
        .to_string();
    let capture_date_utc = format_unix_utc(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_secs(),
    );

    let provenance = serde_json::json!({
        "artifact_path": ARTIFACT,
        "artifact_sha256": artifact_sha256,
        "model_id": MODEL_ID,
        "prompt": PROMPT,
        "prompt_token_count_full": full_prompt.len(),
        "prompt_token_count_used": used_token_count,
        "max_context_tokens": MAX_CONTEXT,
        "gqa_layer_ordinals": GQA_LAYER_ORDINALS,
        "kv_heads": kv_heads,
        "roles": {"0": "K", "1": "V"},
        "first_position": 0,
        "rows_per_block": ROWS_PER_BLOCK,
        "total_rows": total_rows,
        "head_dim": head_dim,
        "row_order": "for layer in gqa_layer_ordinals { for role in [K, V] { for kv_head in 0..kv_heads { for position in first_position..first_position+rows_per_block { head_dim x bf16-bit-pattern u16 LE } } } }",
        "fixture_format": {
            "magic": "IGNHQKV1",
            "format_version": FIXTURE_FORMAT_VERSION,
            "checksum": "FNV-1a 64 over the row payload bytes only, little-endian u64 in the header (corruption detection, not cryptography)"
        },
        "capture_date_utc": capture_date_utc,
        "bin_sha256": bin_sha256,
        "bin_bytes": bin_bytes.len(),
        "captured_by": "crates/core/tests/hq_kv_fixture_capture_gpu.rs",
    });

    let provenance_path = dir.join("hq_kv_rows_27b.provenance.json");
    fs::write(
        &provenance_path,
        serde_json::to_string_pretty(&provenance).expect("serialize provenance") + "\n",
    )
    .unwrap_or_else(|e| panic!("write {}: {e}", provenance_path.display()));

    eprintln!(
        "captured {total_rows} rows ({} bytes) -> {}",
        bin_bytes.len(),
        bin_path.display()
    );
    eprintln!(
        "sanity: nonzero_rows={nonzero_rows}/{total_rows}, layer0 K per-head |mean| = {:?}",
        layer0_k_abs_mean
    );
}
