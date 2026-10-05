//! Flash-Next's MoE ops on the real artifact (spec flash-next/02, Acceptance 2 and 4,
//! GitHub #300), machine-local: `F:/ai/models/Qwen3.8-Flash-Next-ignis/`, written by the
//! converter and the packer (spec 01).
//!
//! - **Full-shape decode.** Expert projections of every K class at three depths, read from the
//!   container through the Flash-Next binder's expert index, decoded by the kernel and checksummed
//!   exactly as the converter checksummed exllamav3's `reconstruct` of the same tensors
//!   (`references/trellis_checksums.json`; the checksum is `record.py`'s `checksum_u16`). Equal
//!   checksums mean the decode is bit-exact at full shape.
//! - **The whole block on recorded activations.** For layers 2, 24 and 46, 64 real tokens of the
//!   quantized stream (`references/moe_block/LNN/`: the block's input, its output from the
//!   checkpoint's `Qwen4ExpTextSparseMoeBlock` with the decoded weights, the router's selection
//!   and the shared expert's part). The kernels run router -> routed experts (prefill route, and
//!   the decode route on the first three tokens) -> shared expert -> combine against the recorded
//!   output. With the recorded selection every token is held to the bound; with the kernel's own
//!   router the selection differences are counted and reported separately, and only tokens whose
//!   ten experts agree are held to it.
//!
//! The bound, per token: relative L2 error <= 1.5e-2. The reference runs the decoded weights
//! rounded to BF16 (2^-9 relative per weight) and BF16 activations between its stages; the
//! kernels keep the exact fp16 weights and fp32 intermediates, so the two differ by the
//! reference's own roundings, ~1e-3 per stage over four to five stages. The measured figures are
//! printed.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1` a missing artifact
//! or GPU is a skip, under it a hard failure (`ignis_core::gpu_profile`).

#![cfg(feature = "cuda")]

use std::collections::BTreeSet;
use std::os::raw::c_void;
use std::path::{Path, PathBuf};

use ignis_artifact::flash_next::{self, FlashNextGeometry, FlashNextPlan, Projection};
use ignis_artifact::packer::ARTIFACT_FILE_NAME;
use ignis_artifact::{CudaDevice, Device, DeviceBuffer, Reader};
use ignis_core::gpu_profile;
use ignis_core::moe::{self, MoeSlot};

const MODEL_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
const BLOCK_REL_L2: f64 = 1.5e-2;
const H: usize = moe::HIDDEN as usize;
const K: usize = moe::TOP_K as usize;

struct Artifact {
    reader: Reader,
    plan: FlashNextPlan,
}

/// The artifact and the reference files a test needs, or `None` after the profile's verdict.
fn artifact(needs: &[&str]) -> Option<Artifact> {
    let dir = Path::new(MODEL_DIR);
    let path = dir.join(ARTIFACT_FILE_NAME);
    for need in std::iter::once(path.clone()).chain(needs.iter().map(|n| dir.join(n))) {
        if !need.exists() {
            gpu_profile::skip_or_fail(&format!("Flash-Next artifact input {} is absent", need.display()));
            return None;
        }
    }
    let reader = Reader::open(&path).unwrap_or_else(|e| panic!("open {}: {e:?}", path.display()));
    let plan = flash_next::bind(&reader, &FlashNextGeometry::qwen38_flash_next()).expect("bind the Flash-Next artifact");
    Some(Artifact { reader, plan })
}

fn device() -> Option<CudaDevice> {
    match CudaDevice::create(0) {
        Ok(d) => Some(d),
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("no CUDA device: {e}"));
            None
        }
    }
}

fn upload(dev: &mut CudaDevice, bytes: &[u8]) -> DeviceBuffer {
    let buf = dev.allocate(bytes.len() as u64).expect("device allocation");
    dev.copy_h2d(&buf, 0, bytes).expect("upload");
    dev.synchronize().expect("upload sync");
    buf
}

fn zeroed(dev: &mut CudaDevice, bytes: usize) -> DeviceBuffer {
    upload(dev, &vec![0u8; bytes])
}

fn download(dev: &mut CudaDevice, buf: &DeviceBuffer) -> Vec<u8> {
    let mut out = vec![0u8; buf.len() as usize];
    dev.copy_d2h(buf, 0, &mut out).expect("download");
    dev.synchronize().expect("download sync");
    out
}

fn ptr(buf: &DeviceBuffer) -> *mut c_void {
    buf.base_ptr() as *mut c_void
}

fn ran(result: Result<(), String>, what: &str) {
    if let Err(e) = result.and_then(|_| moe::stream_sync(std::ptr::null_mut())) {
        panic!("{what}: {e}");
    }
}

fn lowbias32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846C_A68B);
    x ^= x >> 16;
    x
}

/// record.py's checksum_u16 over row-major fp16 bits.
fn checksum_u16(bytes: &[u8]) -> u64 {
    bytes
        .chunks(2)
        .enumerate()
        .fold(0u64, |s, (i, c)| {
            s.wrapping_add(u16::from_le_bytes([c[0], c[1]]) as u64 * (lowbias32(i as u32) | 1) as u64)
        })
}

fn shape(projection: Projection) -> (u32, u32) {
    match projection {
        Projection::GateUp => (moe::HIDDEN, 2 * moe::INTERMEDIATE),
        Projection::Down => (moe::INTERMEDIATE, moe::HIDDEN),
    }
}

fn record<'a>(a: &'a Artifact, layer: usize, expert: usize, projection: Projection) -> (&'a [u8], u32) {
    let rec = a.plan.experts.get(layer, expert, projection).expect("expert index entry");
    let span = a.reader.payload(&flash_next::expert_name(layer, expert as u64, projection)).expect("expert record");
    (span.data, rec.k.k2() as u32)
}

#[test]
#[ignore = "GPU + machine-local artifact: run under scripts/gpu-profile.ps1"]
fn full_shape_decode_matches_the_converters_reconstruct_checksums() {
    let Some(a) = artifact(&["references/trellis_checksums.json"]) else { return };
    let Some(mut dev) = device() else { return };
    let text = std::fs::read_to_string(Path::new(MODEL_DIR).join("references/trellis_checksums.json")).unwrap();
    let entries: Vec<serde_json::Value> = match serde_json::from_str::<serde_json::Value>(&text).unwrap() {
        serde_json::Value::Array(v) => v,
        other => other["entries"].as_array().expect("a list of checksum entries").clone(),
    };
    assert!(!entries.is_empty(), "no checksum entries recorded");
    let mut classes = BTreeSet::new();
    for e in &entries {
        let layer = e["layer"].as_u64().unwrap() as usize;
        let expert = e["expert"].as_u64().unwrap() as usize;
        let projection = match e["proj"].as_str().unwrap() {
            "gu" => Projection::GateUp,
            "dn" => Projection::Down,
            other => panic!("unknown projection {other}"),
        };
        let k2 = e["k2"].as_u64().unwrap() as u32;
        let want = u64::from_str_radix(e["checksum"].as_str().unwrap().trim_start_matches("0x"), 16).unwrap();
        let (bytes, stored_k2) = record(&a, layer, expert, projection);
        assert_eq!(stored_k2, k2, "L{layer} expert {expert} {projection:?}: K class differs from the recording");
        let (inp, out) = shape(projection);
        let trellis_bytes = (inp as usize) * (out as usize) * (k2 as usize) / 16;
        let words = upload(&mut dev, &bytes[..trellis_bytes]);
        let w = dev.allocate(inp as u64 * out as u64 * 2).unwrap();
        ran(
            unsafe { moe::trellis_reconstruct(ptr(&words), k2, inp, out, ptr(&w), std::ptr::null_mut()) },
            "ignis_moe_trellis_reconstruct",
        );
        let got = checksum_u16(&download(&mut dev, &w));
        assert_eq!(got, want, "L{layer} expert {expert} {projection:?} k2 {k2}: decode differs from reconstruct");
        classes.insert((format!("{projection:?}"), k2));
        dev.deallocate(w).unwrap();
        dev.deallocate(words).unwrap();
    }
    println!("full-shape decode: {} projections bit-exact, K classes {classes:?}", entries.len());
}

fn read_vec<T: Copy>(path: &Path, from: fn(&[u8]) -> T, width: usize) -> Vec<T> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes.chunks(width).map(from).collect()
}

fn f32s(path: &Path) -> Vec<f32> {
    read_vec(path, |c| f32::from_le_bytes(c.try_into().unwrap()), 4)
}

fn rel_l2(got: &[f64], want: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for (g, w) in got.iter().zip(want) {
        num += (g - *w as f64).powi(2);
        den += (*w as f64).powi(2);
    }
    (num / den.max(1e-300)).sqrt()
}

fn bf16_bytes_to_f64(bytes: &[u8]) -> Vec<f64> {
    bytes.chunks(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) as f64).collect()
}

/// The MoE block on one layer's recorded tokens, with the given selection; returns the BF16
/// output as f64 and the shared expert's fp32 output.
#[allow(clippy::too_many_arguments)]
fn run_block(
    dev: &mut CudaDevice,
    a: &Artifact,
    layer: usize,
    x: &DeviceBuffer,
    tokens: usize,
    ids: &[i32],
    weights: &[f32],
    decode_route: bool,
) -> (Vec<f64>, Vec<f32>) {
    // Slots for every expert the selection uses, one device copy of each record.
    let mut slots = vec![MoeSlot::ABSENT; moe::EXPERTS as usize * 2];
    let mut held = Vec::new();
    for e in ids.iter().copied().collect::<BTreeSet<i32>>() {
        for projection in Projection::ALL {
            let (bytes, k2) = record(a, layer, e as usize, projection);
            let buf = upload(dev, bytes);
            slots[e as usize * 2 + projection.code() as usize] = MoeSlot { record: buf.base_ptr() as *const c_void, k2, reserved: 0 };
            held.push(buf);
        }
    }
    let slot_bytes: Vec<u8> = slots
        .iter()
        .flat_map(|s| {
            let mut b = (s.record as u64).to_le_bytes().to_vec();
            b.extend_from_slice(&s.k2.to_le_bytes());
            b.extend_from_slice(&s.reserved.to_le_bytes());
            b
        })
        .collect();
    let d_slots = upload(dev, &slot_bytes);
    let d_ids = upload(dev, &ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    let d_w = upload(dev, &weights.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    let max_tokens = tokens as u32;
    let workspace = zeroed(dev, moe::workspace_bytes(max_tokens) as usize);
    let acc = zeroed(dev, tokens * H * 8);
    let null = std::ptr::null_mut();
    let routed = if decode_route {
        unsafe {
            moe::experts_decode(ptr(x), max_tokens, ptr(&d_ids) as *const i32, ptr(&d_w) as *const f32,
                                ptr(&d_slots) as *const MoeSlot, ptr(&workspace), ptr(&acc) as *mut i64, null)
        }
    } else {
        unsafe {
            moe::experts_prefill(ptr(x), max_tokens, ptr(&d_ids) as *const i32, ptr(&d_w) as *const f32,
                                 ptr(&d_slots) as *const MoeSlot, ptr(&workspace), max_tokens, ptr(&acc) as *mut i64, null)
        }
    };
    ran(routed, "routed experts");
    let name = |s: &str| format!("layers.{layer}.mlp.{s}");
    let gate = upload(dev, a.reader.payload(&name("shared_expert.gate_proj.weight")).unwrap().data);
    let up = upload(dev, a.reader.payload(&name("shared_expert.up_proj.weight")).unwrap().data);
    let down = upload(dev, a.reader.payload(&name("shared_expert.down_proj.weight")).unwrap().data);
    let w_gate = upload(dev, a.reader.payload(&name("shared_expert_gate.weight")).unwrap().data);
    let h = zeroed(dev, tokens * moe::INTERMEDIATE as usize * 2);
    let shared = zeroed(dev, tokens * H * 4);
    let out = zeroed(dev, tokens * H * 2);
    ran(
        unsafe { moe::shared_expert(ptr(&gate), ptr(&up), ptr(&down), ptr(x), max_tokens, ptr(&h), ptr(&shared) as *mut f32, null) },
        "shared expert",
    );
    let shared_out: Vec<f32> =
        download(dev, &shared).chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    ran(
        unsafe { moe::combine(ptr(&acc) as *mut i64, ptr(&shared) as *const f32, ptr(x), ptr(&w_gate), max_tokens, ptr(&out), null) },
        "combine",
    );
    let result = bf16_bytes_to_f64(&download(dev, &out));
    for buf in held.into_iter().chain([d_slots, d_ids, d_w, workspace, acc, gate, up, down, w_gate, h, shared, out]) {
        dev.deallocate(buf).unwrap();
    }
    (result, shared_out)
}

#[test]
#[ignore = "GPU + machine-local artifact: run under scripts/gpu-profile.ps1"]
fn whole_block_matches_recorded_activations_at_three_depths() {
    let layers = [2usize, 24, 46];
    let needs: Vec<String> = layers.iter().map(|l| format!("references/moe_block/L{l:02}/x.bf16")).collect();
    let needs: Vec<&str> = needs.iter().map(String::as_str).collect();
    let Some(a) = artifact(&needs) else { return };
    let Some(mut dev) = device() else { return };
    for layer in layers {
        let dir: PathBuf = Path::new(MODEL_DIR).join(format!("references/moe_block/L{layer:02}"));
        let x_bytes = std::fs::read(dir.join("x.bf16")).unwrap();
        let tokens = x_bytes.len() / (2 * H);
        let y = f32s(&dir.join("y.f32"));
        let rec_ids: Vec<i32> = read_vec(&dir.join("ids.i32"), |c| i32::from_le_bytes(c.try_into().unwrap()), 4);
        let rec_w = f32s(&dir.join("weights.f32"));
        let rec_shared = f32s(&dir.join("shared.f32"));
        assert_eq!((y.len(), rec_ids.len()), (tokens * H, tokens * K));
        let x = upload(&mut dev, &x_bytes);

        // Recorded selection, prefill route: every token held to the bound.
        let (out, shared) = run_block(&mut dev, &a, layer, &x, tokens, &rec_ids, &rec_w, false);
        let mut worst = 0.0f64;
        for t in 0..tokens {
            let e = rel_l2(&out[t * H..(t + 1) * H], &y[t * H..(t + 1) * H]);
            worst = worst.max(e);
            assert!(e <= BLOCK_REL_L2, "L{layer} token {t}: relative L2 {e:.3e} over {BLOCK_REL_L2:.1e}");
        }
        let shared64: Vec<f64> = shared.iter().map(|&v| v as f64).collect();
        let shared_err = rel_l2(&shared64, &rec_shared);

        // The decode route on the first three tokens, same selection.
        let x3 = upload(&mut dev, &x_bytes[..3 * 2 * H]);
        let (out3, _) = run_block(&mut dev, &a, layer, &x3, 3, &rec_ids[..3 * K], &rec_w[..3 * K], true);
        let mut worst3 = 0.0f64;
        for t in 0..3 {
            let e = rel_l2(&out3[t * H..(t + 1) * H], &y[t * H..(t + 1) * H]);
            worst3 = worst3.max(e);
            assert!(e <= BLOCK_REL_L2, "L{layer} decode token {t}: relative L2 {e:.3e} over {BLOCK_REL_L2:.1e}");
        }

        // The kernel's own router: selection differences counted, agreeing tokens held to the bound.
        let router_w = upload(&mut dev, a.reader.payload(&format!("layers.{layer}.mlp.gate.weight")).unwrap().data);
        let ids = zeroed(&mut dev, tokens * K * 4);
        let weights = zeroed(&mut dev, tokens * K * 4);
        let logits = zeroed(&mut dev, tokens * moe::EXPERTS as usize * 4);
        ran(
            unsafe {
                moe::router(ptr(&x), tokens as u32, ptr(&router_w), ptr(&ids) as *mut i32, ptr(&weights) as *mut f32,
                            ptr(&logits) as *mut f32, std::ptr::null_mut())
            },
            "router",
        );
        let own_ids: Vec<i32> = download(&mut dev, &ids).chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
        let own_w: Vec<f32> = download(&mut dev, &weights).chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let (own_out, _) = run_block(&mut dev, &a, layer, &x, tokens, &own_ids, &own_w, false);
        let mut differing = 0;
        let mut worst_own = 0.0f64;
        for t in 0..tokens {
            let ours: BTreeSet<i32> = own_ids[t * K..(t + 1) * K].iter().copied().collect();
            let theirs: BTreeSet<i32> = rec_ids[t * K..(t + 1) * K].iter().copied().collect();
            if ours != theirs {
                differing += 1;
                continue;
            }
            let e = rel_l2(&own_out[t * H..(t + 1) * H], &y[t * H..(t + 1) * H]);
            worst_own = worst_own.max(e);
            assert!(e <= BLOCK_REL_L2, "L{layer} token {t} (own router): relative L2 {e:.3e} over {BLOCK_REL_L2:.1e}");
        }
        println!(
            "L{layer}: {tokens} tokens; recorded selection worst relative L2 {worst:.3e} (prefill), {worst3:.3e} \
             (decode, 3 tokens); shared expert {shared_err:.3e}; own router: {differing} token(s) select other \
             experts, the rest worst {worst_own:.3e}"
        );
        for buf in [x, x3, router_w, ids, weights, logits] {
            dev.deallocate(buf).unwrap();
        }
    }
}
