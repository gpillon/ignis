# 01 — binder + materializer (device materialization)

GitHub: #4

Port the reference's per-model **binder** (`binder.h`), **materializer**
(`materializer.h`), and **typed binding** (`typed_binding.h`) into
`ignis-artifact`, on top of the generic reader:

- The binder consumes **every** object at bind time — an unconsumed object
  is a load failure (ADR 0002). It owns the per-model semantic layer the
  generic reader deliberately leaves out (format spec §6.1).
- The materializer places NVFP4 / BF16 tensors into VRAM at the geometry /
  alignment the reader computed (rowsplit / blockscale / rowscale layouts).
- The typed binding exposes the materialized tensors to the engine
  (`ignis-core`).
- Load `qwen3_8_27b_nvfp4full-v2.ninfer` (1,325 objects: 281 NVFP4 + 9 BF16
  + 6 frontend).

## Acceptance

- Load succeeds; the full object manifest is consumed (zero unconsumed
  objects, ADR 0002).
- NVFP4 / BF16 tensors materialize into VRAM at the reader-computed
  geometry.
- The typed binding is consumable by `ignis-core`.

## Status (done)

The binder / materializer / typed-binding / device layer is ported into
`crates/artifact` as idiomatic Rust, mirroring the reference structure with a
small public surface + checked arithmetic + `Result<ArtifactError>`:

- **Binder** (`src/binder.rs`) — consumes *every* object; `finish()` fails on
  any unconsumed **or** unplanned object (the ADR 0002 invariant on both
  success and failure paths). Produces a `MaterializationPlan` with
  alignment-aware device offsets + host-retained resources.
- **Materializer** (`src/materializer.rs`) — places the plan on a `Device`:
  4096-aligned direct-I/O reads (page-aligned staging, the reader's
  `read_direct` path), H2D copies at the reader-computed geometry, streaming
  staging (peak = largest single object, not the sum), and host-retained
  resources. Emits `MaterializationStats`.
- **Device** (`src/device.rs`) — the `Device` trait + `CpuDevice`
  (a `Vec<u8>` mock, always available, the ADR 0006 stand-in) + `CudaDevice`
  (behind the `cuda` cargo feature, links the kernel leaf's flat C device
  surface).
- **Typed binding** (`src/binding.rs`) — NVFP4 → block-scale planes
  (code / scale + offset, weight divisor), BF16 → typed view, resources →
  host `&[u8]`. Kept minimal: frontend extraction is ticket 02.
- **Kernel device surface** (new, flat C, ADR 0001) —
  `kernel/include/ignis_device.h` + `kernel/src/device.cu`: one device handle
  owning a non-blocking load stream + a blocking-sync event (the reference
  `DeviceContext` pattern, so `synchronize()` sleeps instead of spinning),
  `cudaMalloc` chunk, async H2D/D2H on the load stream, `cudaMemGetInfo`.
  Flat C: explicit pointers + sizes, `int32` return codes, no C++ types
  across the boundary. Picked up automatically by the kernel CMake glob.
- **Fixture** (`src/fixture.rs`) — Rust port of the reference
  `artifact_fixture.h`: a tiny synthetic `.ninfer` v2 container written to a
  tempdir (all four tensor layouts + a raw resource), shared by the unit and
  integration tests.
- **Tests** — `cargo test -p ignis-artifact` (default features, CPU) is
  green: 29 unit tests (binder ADR 0002, materializer geometry / stats /
  host-retained, binding typed accessors, CpuDevice round-trip + rejection,
  fixture round-trip, geometry) + the real-artifact integration test
  (`tests/real_artifact.rs`) which, when the 19 GB artifact is present,
  **binds all 1,325 objects (zero unconsumed)** and finishes the plan; full
  `CpuDevice` materialization is gated behind `IGNIS_TEST_FULL_MATERIALIZE=1`
  (heavy: multi-GB host alloc + a whole-file read); a `CudaDevice` test is
  behind the `cuda` feature and skips unless `IGNIS_TEST_CUDA=1` + a free GPU.
  The default build is pure Rust (no kernel linking) — `crates/artifact/
  build.rs` only emits link directives when the `cuda` feature is on.

## Pending (revisit later)

- **GPU verification of `CudaDevice` + real-artifact VRAM materialization.**
  The `cuda` feature compiles (`cargo check -p ignis-artifact --features
  cuda`) and the device surface builds (`kernel/build-a` scratch), but it is
  not *run*: the RTX 5090 is held by the reference `ninfer-serve` (ADR 0006,
  ~30/32 GB, 98% util). Running `CudaDevice` / a real VRAM upload requires a
  free GPU; gate it with `IGNIS_TEST_CUDA=1` (and `IGNIS_TEST_FULL_MATERIALIZE=1`
  for the CPU full-materialization) once the reference runner is stopped.
- **`IGNIS_TEST_FULL_MATERIALIZE=1` run.** The gated full `CpuDevice`
  materialization of the 19 GB artifact has not been executed here (it
  allocates several GB of host RAM and reads the whole file); it is wired and
  compiles, but should be run once on a machine with enough free RAM to
  confirm the streaming path at scale.
- **Real-artifact full `CpuDevice` materialization test is not in the
  default gate** (by design — it is heavy). It runs only when the env var is
  set, so the default `cargo test` stays fast.
