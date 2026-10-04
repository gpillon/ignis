//! Flash-Next's MoE ops driven through the Rust bindings (spec flash-next/02, GitHub #300):
//! the trellis decode bit for bit against exllamav3's `reconstruct`, and the router's top-10
//! sets against the checkpoint's transformers router, on the fixtures the kernel leaf's own
//! CTests use (`kernel/tests/fixtures/flash_next/`, recorded by its `record.py`). The CTests
//! check the ops in depth; this checks that the bindings carry every argument to the right
//! place.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1` a missing GPU or a
//! kernel error is a skip, under it a hard failure (`ignis_core::gpu_profile`). Run via
//! `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

use std::collections::{BTreeSet, HashMap};
use std::os::raw::c_void;
use std::path::PathBuf;

use ignis_artifact::{CudaDevice, Device, DeviceBuffer};
use ignis_core::gpu_profile;
use ignis_core::moe;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../kernel/tests/fixtures/flash_next")
}

fn take<'a>(bytes: &'a [u8], at: &mut usize, n: usize) -> &'a [u8] {
    let s = &bytes[*at..*at + n];
    *at += n;
    s
}

fn le_u32(b: &[u8]) -> u32 {
    u32::from_le_bytes(b.try_into().unwrap())
}

/// The IGNFX001 container (record.py's Writer): name -> raw payload.
fn read_fixture(name: &str) -> HashMap<String, Vec<u8>> {
    let bytes = std::fs::read(fixture_dir().join(name)).unwrap_or_else(|e| panic!("fixture {name}: {e}"));
    assert_eq!(&bytes[..8], b"IGNFX001", "{name} is not an IGNFX001 fixture");
    let mut at = 8;
    let mut out = HashMap::new();
    while at < bytes.len() {
        let name_len = le_u32(take(&bytes, &mut at, 4)) as usize;
        let key = String::from_utf8(take(&bytes, &mut at, name_len).to_vec()).unwrap();
        let _dtype = le_u32(take(&bytes, &mut at, 4));
        let ndim = le_u32(take(&bytes, &mut at, 4)) as usize;
        take(&bytes, &mut at, 8 * ndim);
        let n = u64::from_le_bytes(take(&bytes, &mut at, 8).try_into().unwrap()) as usize;
        out.insert(key, take(&bytes, &mut at, n).to_vec());
    }
    out
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

fn download(dev: &mut CudaDevice, buf: &DeviceBuffer) -> Vec<u8> {
    let mut out = vec![0u8; buf.len() as usize];
    dev.copy_d2h(buf, 0, &mut out).expect("download");
    dev.synchronize().expect("download sync");
    out
}

fn ptr(buf: &DeviceBuffer) -> *mut c_void {
    buf.base_ptr() as *mut c_void
}

/// Runs a binding, under the profile's rule for a kernel error.
fn ran(result: Result<(), String>, what: &str) -> bool {
    match result.and_then(|_| moe::stream_sync(std::ptr::null_mut())) {
        Ok(()) => true,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("{what}: {e}"));
            false
        }
    }
}

#[test]
#[ignore = "GPU: run under scripts/gpu-profile.ps1"]
fn trellis_reconstruct_through_the_bindings_is_exllamav3s_bit_for_bit() {
    let Some(mut dev) = device() else { return };
    for (name, k2) in [("k2", 4u32), ("k2p5", 5), ("k3", 6), ("k4", 8)] {
        let fx = read_fixture(&format!("trellis_{name}.bin"));
        let words = upload(&mut dev, &fx["trellis"]);
        let oracle = &fx["reconstruct"];
        let out = dev.allocate(oracle.len() as u64).expect("device allocation");
        let call = unsafe { moe::trellis_reconstruct(ptr(&words), k2, 256, 128, ptr(&out), std::ptr::null_mut()) };
        if !ran(call, "ignis_moe_trellis_reconstruct") {
            return;
        }
        assert_eq!(&download(&mut dev, &out), oracle, "K class {name}: decode differs from reconstruct");
        dev.deallocate(out).unwrap();
        dev.deallocate(words).unwrap();
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

/// record.py's hash_uniform, rounded to BF16 (nearest even) as torch's `.to(bfloat16)`.
fn hash_bf16(stream: u32, i: u64, amplitude: f32) -> u16 {
    let h = lowbias32((i as u32).wrapping_mul(0x9E37_79B9).wrapping_add(stream.wrapping_mul(0x85EB_CA6B)));
    let v = ((h >> 8) as f32 * (1.0f32 / 8_388_608.0) - 1.0) * amplitude;
    let b = v.to_bits();
    ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}

#[test]
#[ignore = "GPU: run under scripts/gpu-profile.ps1"]
fn router_through_the_bindings_selects_the_checkpoints_experts() {
    let Some(mut dev) = device() else { return };
    let fx = read_fixture("router_ref.bin");
    let geo: Vec<i32> = fx["geometry"].chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
    let amp: Vec<f32> = fx["amplitudes"].chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let (tokens, hidden, experts) = (geo[0] as usize, geo[1] as usize, geo[2] as usize);
    assert_eq!((hidden, experts), (moe::HIDDEN as usize, moe::EXPERTS as usize));
    let to_bytes = |v: Vec<u16>| v.into_iter().flat_map(|h| h.to_le_bytes()).collect::<Vec<u8>>();
    let x = to_bytes((0..tokens * hidden).map(|i| hash_bf16(geo[4] as u32, i as u64, amp[0])).collect());
    let w = to_bytes((0..experts * hidden).map(|i| hash_bf16(geo[5] as u32, i as u64, amp[1])).collect());
    let (dx, dw) = (upload(&mut dev, &x), upload(&mut dev, &w));
    let k = moe::TOP_K as usize;
    let ids = dev.allocate((tokens * k * 4) as u64).unwrap();
    let weights = dev.allocate((tokens * k * 4) as u64).unwrap();
    let logits = dev.allocate((tokens * experts * 4) as u64).unwrap();
    let call = unsafe {
        moe::router(
            ptr(&dx),
            tokens as u32,
            ptr(&dw),
            ptr(&ids) as *mut i32,
            ptr(&weights) as *mut f32,
            ptr(&logits) as *mut f32,
            std::ptr::null_mut(),
        )
    };
    if !ran(call, "ignis_moe_router") {
        return;
    }
    let got: Vec<i32> = download(&mut dev, &ids).chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
    let want: Vec<i32> = fx["ids"].chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
    // Measured by the CTest on these inputs: every token's set equals the checkpoint router's.
    for t in 0..tokens {
        let a: BTreeSet<i32> = got[t * k..(t + 1) * k].iter().copied().collect();
        let b: BTreeSet<i32> = want[t * k..(t + 1) * k].iter().copied().collect();
        assert_eq!(a, b, "token {t}: top-10 set differs from the checkpoint router's");
    }
    for buf in [ids, weights, logits, dx, dw] {
        dev.deallocate(buf).unwrap();
    }
}
