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
//!   and the gated shared expert alone). The kernels run router -> routed experts (prefill route,
//!   and the decode route on the first three tokens) -> shared expert -> combine against the
//!   recorded output, and the shared expert, gated by sigmoid(x . w_gate) as the checkpoint
//!   gates it, against the recorded `shared.f32`. With the recorded selection every token is held
//!   to the bound; with the kernel's own router the selection differences are counted and
//!   reported separately, and only tokens whose ten experts agree are held to it.
//!
//! The bound, per token: relative L2 error <= 1.5e-2. The reference runs the decoded weights
//! rounded to BF16 (2^-9 relative per weight) and BF16 activations between its stages; the
//! kernels keep the exact fp16 weights and fp32 intermediates, so the two differ by the
//! reference's own roundings, ~1e-3 per stage over four to five stages. The measured figures are
//! printed; the bound is to be confirmed by the first run on the artifact.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1` a missing artifact,
//! reference file or GPU, or a kernel error, is a skip; under it each is a hard failure
//! (`ignis_core::gpu_profile`). Until spec 01's conversion has written the artifact and its
//! references, these two tests therefore fail under the profile.

#![cfg(feature = "cuda")]

use std::collections::BTreeSet;
use std::os::raw::c_void;
use std::path::{Path, PathBuf};

use ignis_artifact::flash_next::{self, FlashNextGeometry, FlashNextPlan, Projection};
use ignis_artifact::packer::ARTIFACT_FILE_NAME;
use ignis_artifact::{CudaDevice, Device, DeviceBuffer, Reader};
use ignis_core::gpu_profile;
use ignis_core::moe::{self, MoeSlot, MoeWorkspace};

const MODEL_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
const BLOCK_REL_L2: f64 = 1.5e-2;
const LAYERS: [usize; 3] = [2, 24, 46];
const BLOCK_FILES: [&str; 6] = ["x.bf16", "y.f32", "ids.i32", "weights.f32", "shared.f32", "manifest.json"];
const H: usize = moe::HIDDEN as usize;
const K: usize = moe::TOP_K as usize;

struct Artifact {
    reader: Reader,
    plan: FlashNextPlan,
}

/// The artifact, with every reference file a test needs present, or `None` after the profile's
/// verdict.
fn artifact(needs: &[String]) -> Option<Artifact> {
    let dir = Path::new(MODEL_DIR);
    let path = dir.join(ARTIFACT_FILE_NAME);
    for need in std::iter::once(path.clone()).chain(needs.iter().map(|n| dir.join(n))) {
        if !need.exists() {
            gpu_profile::skip_or_fail(&format!("Flash-Next artifact input {} is absent", need.display()));
            return None;
        }
    }
    let reader = match Reader::open(&path) {
        Ok(r) => r,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("open {}: {e:?}", path.display()));
            return None;
        }
    };
    match flash_next::bind(&reader, &FlashNextGeometry::qwen38_flash_next()) {
        Ok(plan) => Some(Artifact { reader, plan }),
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("bind the Flash-Next artifact: {e:?}"));
            None
        }
    }
}

/// The device, prepared for the MoE ops, or `None` after the profile's verdict.
fn device() -> Option<CudaDevice> {
    let dev = match CudaDevice::create(0) {
        Ok(d) => d,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("no CUDA device: {e}"));
            return None;
        }
    };
    if let Err(e) = moe::prepare() {
        gpu_profile::skip_or_fail(&format!("ignis_moe_prepare: {e}"));
        return None;
    }
    Some(dev)
}

fn upload(dev: &mut CudaDevice, bytes: &[u8]) -> Option<DeviceBuffer> {
    let buf = dev.allocate(bytes.len() as u64);
    match buf.and_then(|b| dev.copy_h2d(&b, 0, bytes).and_then(|_| dev.synchronize()).map(|_| b)) {
        Ok(b) => Some(b),
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("device upload: {e:?}"));
            None
        }
    }
}

fn zeroed(dev: &mut CudaDevice, bytes: usize) -> Option<DeviceBuffer> {
    upload(dev, &vec![0u8; bytes])
}

fn download(dev: &mut CudaDevice, buf: &DeviceBuffer) -> Option<Vec<u8>> {
    let mut out = vec![0u8; buf.len() as usize];
    match dev.copy_d2h(buf, 0, &mut out).and_then(|_| dev.synchronize()) {
        Ok(()) => Some(out),
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("device download: {e:?}"));
            None
        }
    }
}

fn ptr(buf: &DeviceBuffer) -> *mut c_void {
    buf.base_ptr() as *mut c_void
}

/// A kernel call and the stream it ran on, under the profile's rule for a kernel error.
fn ran(result: Result<(), String>, what: &str) -> bool {
    match result.and_then(|_| moe::stream_sync(std::ptr::null_mut())) {
        Ok(()) => true,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("{what}: {e}"));
            false
        }
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
    bytes.chunks(2).enumerate().fold(0u64, |s, (i, c)| {
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
    let Some(a) = artifact(&["references/trellis_checksums.json".to_string()]) else { return };
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
        let Some(words) = upload(&mut dev, &bytes[..trellis_bytes]) else { return };
        let Some(w) = zeroed(&mut dev, inp as usize * out as usize * 2) else { return };
        let call = unsafe { moe::trellis_reconstruct(ptr(&words), k2, inp, out, ptr(&w), std::ptr::null_mut()) };
        if !ran(call, "ignis_moe_trellis_reconstruct") {
            return;
        }
        let Some(decoded) = download(&mut dev, &w) else { return };
        assert_eq!(checksum_u16(&decoded), want, "L{layer} expert {expert} {projection:?} k2 {k2}: decode differs from reconstruct");
        classes.insert((format!("{projection:?}"), k2));
        dev.deallocate(w).unwrap();
        dev.deallocate(words).unwrap();
    }
    println!("full-shape decode: {} projections bit-exact, K classes {classes:?}", entries.len());
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn i32s(bytes: &[u8]) -> Vec<i32> {
    bytes.chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn bf16s(bytes: &[u8]) -> Vec<f64> {
    bytes.chunks(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) as f64).collect()
}

fn rel_l2(got: &[f64], want: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for (g, w) in got.iter().zip(want) {
        num += (g - *w as f64).powi(2);
        den += (*w as f64).powi(2);
    }
    (num / den.max(1e-300)).sqrt()
}

struct BlockOut {
    out: Vec<f64>,
    /// The shared expert's output, gated by sigmoid(x . w_gate) in f64 as the checkpoint gates it.
    gated_shared: Vec<f64>,
}

/// The MoE block on one layer's recorded tokens with the given selection.
#[allow(clippy::too_many_arguments)]
fn run_block(
    dev: &mut CudaDevice,
    a: &Artifact,
    layer: usize,
    x_bytes: &[u8],
    tokens: usize,
    ids: &[i32],
    weights: &[f32],
    decode_route: bool,
) -> Option<BlockOut> {
    // Slots for every expert the selection uses, one device copy of each record.
    let mut slots = vec![MoeSlot::ABSENT; moe::EXPERTS as usize * 2];
    let mut held = Vec::new();
    for e in ids.iter().copied().collect::<BTreeSet<i32>>() {
        for projection in Projection::ALL {
            let (bytes, k2) = record(a, layer, e as usize, projection);
            let buf = upload(dev, bytes)?;
            slots[e as usize * 2 + projection.code() as usize] = MoeSlot { record: ptr(&buf), k2, reserved: 0 };
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
    let x = upload(dev, x_bytes)?;
    let d_slots = upload(dev, &slot_bytes)?;
    let d_ids = upload(dev, &ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
    let d_w = upload(dev, &weights.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
    let n = tokens as u32;
    let decode_tokens = if decode_route { n } else { 1 };
    let workspace = zeroed(dev, moe::workspace_bytes(decode_tokens, n) as usize)?;
    let acc = zeroed(dev, tokens * H * 8)?;
    let ws = MoeWorkspace { base: ptr(&workspace), decode_tokens, prefill_tokens: n };
    let null = std::ptr::null_mut();
    if !ran(unsafe { moe::workspace_init(&ws, ptr(&acc) as *mut i64, null) }, "workspace init") {
        return None;
    }
    let routed = unsafe {
        let (x, ids, w, slots, acc) = (ptr(&x), ptr(&d_ids) as *const i32, ptr(&d_w) as *const f32, ptr(&d_slots) as *const MoeSlot, ptr(&acc) as *mut i64);
        if decode_route {
            moe::experts_decode(x, n, ids, w, slots, &ws, acc, null)
        } else {
            moe::experts_prefill(x, n, ids, w, slots, &ws, acc, null)
        }
    };
    if !ran(routed, "routed experts") {
        return None;
    }
    let name = |s: &str| format!("layers.{layer}.mlp.{s}");
    let payload = |s: &str| a.reader.payload(&name(s)).expect("shared expert tensor").data;
    let gate = upload(dev, payload("shared_expert.gate_proj.weight"))?;
    let up = upload(dev, payload("shared_expert.up_proj.weight"))?;
    let down = upload(dev, payload("shared_expert.down_proj.weight"))?;
    let w_gate_bytes = payload("shared_expert_gate.weight");
    let w_gate = upload(dev, w_gate_bytes)?;
    let h = zeroed(dev, tokens * moe::INTERMEDIATE as usize * 2)?;
    let shared = zeroed(dev, tokens * H * 4)?;
    let out = zeroed(dev, tokens * H * 2)?;
    let call = unsafe { moe::shared_expert(ptr(&gate), ptr(&up), ptr(&down), ptr(&x), n, ptr(&h), ptr(&shared) as *mut f32, null) };
    if !ran(call, "shared expert") {
        return None;
    }
    let shared_out = f32s(&download(dev, &shared)?);
    let call = unsafe { moe::combine(ptr(&acc) as *mut i64, ptr(&shared) as *const f32, ptr(&x), ptr(&w_gate), n, ptr(&out), null) };
    if !ran(call, "combine") {
        return None;
    }
    let result = bf16s(&download(dev, &out)?);
    // The checkpoint's gate: sigmoid(x . w_gate) over the BF16 shared_expert_gate [1, 2560].
    let xv = bf16s(x_bytes);
    let wv = bf16s(w_gate_bytes);
    let mut gated_shared = vec![0.0; tokens * H];
    for t in 0..tokens {
        let dot: f64 = (0..H).map(|k| xv[t * H + k] * wv[k]).sum();
        let g = 1.0 / (1.0 + (-dot).exp());
        for k in 0..H {
            gated_shared[t * H + k] = g * shared_out[t * H + k] as f64;
        }
    }
    for buf in held.into_iter().chain([x, d_slots, d_ids, d_w, workspace, acc, gate, up, down, w_gate, h, shared, out]) {
        dev.deallocate(buf).unwrap();
    }
    Some(BlockOut { out: result, gated_shared })
}

#[test]
#[ignore = "GPU + machine-local artifact: run under scripts/gpu-profile.ps1"]
fn whole_block_matches_recorded_activations_at_three_depths() {
    let needs: Vec<String> = LAYERS
        .iter()
        .flat_map(|l| BLOCK_FILES.iter().map(move |f| format!("references/moe_block/L{l:02}/{f}")))
        .collect();
    let Some(a) = artifact(&needs) else { return };
    let Some(mut dev) = device() else { return };
    for layer in LAYERS {
        let dir: PathBuf = Path::new(MODEL_DIR).join(format!("references/moe_block/L{layer:02}"));
        let read = |f: &str| std::fs::read(dir.join(f)).unwrap_or_else(|e| panic!("{}: {e}", dir.join(f).display()));
        let x_bytes = read("x.bf16");
        let tokens = x_bytes.len() / (2 * H);
        let y = f32s(&read("y.f32"));
        let rec_ids = i32s(&read("ids.i32"));
        let rec_w = f32s(&read("weights.f32"));
        let rec_shared = f32s(&read("shared.f32"));
        assert_eq!((y.len(), rec_ids.len(), rec_shared.len()), (tokens * H, tokens * K, tokens * H));

        // Recorded selection, prefill route: every token, and its gated shared expert, held to the bound.
        let Some(b) = run_block(&mut dev, &a, layer, &x_bytes, tokens, &rec_ids, &rec_w, false) else { return };
        let mut worst = 0.0f64;
        let mut worst_shared = 0.0f64;
        for t in 0..tokens {
            let e = rel_l2(&b.out[t * H..(t + 1) * H], &y[t * H..(t + 1) * H]);
            worst = worst.max(e);
            assert!(e <= BLOCK_REL_L2, "L{layer} token {t}: relative L2 {e:.3e} over {BLOCK_REL_L2:.1e}");
            let s = rel_l2(&b.gated_shared[t * H..(t + 1) * H], &rec_shared[t * H..(t + 1) * H]);
            worst_shared = worst_shared.max(s);
            assert!(s <= BLOCK_REL_L2, "L{layer} token {t}: gated shared expert relative L2 {s:.3e} over {BLOCK_REL_L2:.1e}");
        }

        // The decode route on the first three tokens, same selection.
        let Some(b3) = run_block(&mut dev, &a, layer, &x_bytes[..3 * 2 * H], 3, &rec_ids[..3 * K], &rec_w[..3 * K], true) else {
            return;
        };
        let mut worst3 = 0.0f64;
        for t in 0..3 {
            let e = rel_l2(&b3.out[t * H..(t + 1) * H], &y[t * H..(t + 1) * H]);
            worst3 = worst3.max(e);
            assert!(e <= BLOCK_REL_L2, "L{layer} decode token {t}: relative L2 {e:.3e} over {BLOCK_REL_L2:.1e}");
        }

        // The kernel's own router: selection differences counted, agreeing tokens held to the bound.
        let Some(x) = upload(&mut dev, &x_bytes) else { return };
        let Some(router_w) = upload(&mut dev, a.reader.payload(&format!("layers.{layer}.mlp.gate.weight")).unwrap().data) else { return };
        let Some(ids) = zeroed(&mut dev, tokens * K * 4) else { return };
        let Some(weights) = zeroed(&mut dev, tokens * K * 4) else { return };
        let Some(logits) = zeroed(&mut dev, tokens * moe::EXPERTS as usize * 4) else { return };
        let call = unsafe {
            moe::router(ptr(&x), tokens as u32, ptr(&router_w), ptr(&ids) as *mut i32, ptr(&weights) as *mut f32,
                        ptr(&logits) as *mut f32, std::ptr::null_mut())
        };
        if !ran(call, "router") {
            return;
        }
        let Some(own_ids) = download(&mut dev, &ids).map(|b| i32s(&b)) else { return };
        let Some(own_w) = download(&mut dev, &weights).map(|b| f32s(&b)) else { return };
        let Some(own) = run_block(&mut dev, &a, layer, &x_bytes, tokens, &own_ids, &own_w, false) else { return };
        let mut differing = 0;
        let mut worst_own = 0.0f64;
        for t in 0..tokens {
            let ours: BTreeSet<i32> = own_ids[t * K..(t + 1) * K].iter().copied().collect();
            let theirs: BTreeSet<i32> = rec_ids[t * K..(t + 1) * K].iter().copied().collect();
            if ours != theirs {
                differing += 1;
                continue;
            }
            let e = rel_l2(&own.out[t * H..(t + 1) * H], &y[t * H..(t + 1) * H]);
            worst_own = worst_own.max(e);
            assert!(e <= BLOCK_REL_L2, "L{layer} token {t} (own router): relative L2 {e:.3e} over {BLOCK_REL_L2:.1e}");
        }
        println!(
            "L{layer}: {tokens} tokens; recorded selection worst relative L2 {worst:.3e} (prefill), {worst3:.3e} \
             (decode, 3 tokens); gated shared expert {worst_shared:.3e}; own router: {differing} token(s) select \
             other experts, the rest worst {worst_own:.3e}"
        );
        for buf in [x, router_w, ids, weights, logits] {
            dev.deallocate(buf).unwrap();
        }
    }
}
