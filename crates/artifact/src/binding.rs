//! The typed binding: the typed view `ignis-core` consumes from a
//! materialized artifact (the frontend extraction arrives with
//! artifact-02).
//!
//! Per-name typed accessors over the materialized artifact: NVFP4 tensors
//! expose their block-scale planes (code plane, scale plane + offset,
//! weight divisor) as typed handles; BF16 contiguous tensors expose a
//! row-major typed view; resources expose their raw host bytes.

use crate::binder::ObjectHandle;
use crate::materializer::{MaterializedArtifact, TensorView};
use crate::{
    block_scale_geometry, fail, Result, Reader, StorageLayout, NumericFormat,
};

/// A typed NVFP4 block-scale tensor: code plane, scale plane (with its
/// offset), and the trailing FP32 weight divisor (the kernel reads these
/// on-device).
pub struct Nvfp4View {
    /// The object's directory name.
    pub name: String,
    pub rows: u64,
    pub cols: u64,
    /// Code plane (packed NVFP4, 2 values per byte).
    pub code: *const u8,
    pub code_bytes: u64,
    /// Scale plane (E4M3FN, swizzled).
    pub scale: *const u8,
    pub scale_bytes: u64,
    /// Offset of the scale plane relative to `code`.
    pub scale_plane_offset: u64,
    /// The trailing little-endian FP32 weight divisor (4 bytes).
    pub weight_divisor: *const f32,
    /// Offset of the weight divisor relative to `code`.
    pub weight_divisor_offset: u64,
}

/// A typed contiguous BF16 tensor (row-major, 2-byte elements).
pub struct Bf16View {
    /// The object's directory name.
    pub name: String,
    pub rows: u64,
    pub cols: u64,
    /// Pointer to the data on the device.
    pub data: *const u8,
    /// Exact payload length in bytes.
    pub bytes: u64,
}

/// A typed binding over a materialized artifact, addressed by the
/// directory object name.
pub struct Binding<'a> {
    reader: &'a Reader,
    artifact: &'a MaterializedArtifact,
}

impl<'a> Binding<'a> {
    /// Bind a materialized artifact (the reader it was opened from).
    pub fn new(reader: &'a Reader, artifact: &'a MaterializedArtifact) -> Self {
        Self { reader, artifact }
    }

    /// A typed view of a device-placed tensor by name (error if the name
    /// is unknown or the object was not placed on the device).
    pub fn tensor(&self, name: &str) -> Result<&TensorView> {
        let index = self
            .reader
            .index_of(name)
            .ok_or_else(|| fail(format!("unknown artifact object: {name}")))?;
        self.artifact.device_view(ObjectHandle { index })
    }

    /// A typed NVFP4 block-scale handle by name (code plane, scale plane +
    /// offset, weight divisor).
    pub fn nvfp4(&self, name: &str) -> Result<Nvfp4View> {
        let view = self.tensor(name)?;
        if view.format != NumericFormat::Nvfp4
            || view.layout != StorageLayout::BlockScaleK16M128x4V1
        {
            return Err(fail(format!("object {name} is not an NVFP4 block-scale tensor")));
        }
        if view.shape.len() != 2 {
            return Err(fail(format!("object {name} is not rank-two")));
        }
        let g = block_scale_geometry(view.format, &view.shape)?;
        Ok(Nvfp4View {
            name: view.name.clone(),
            rows: view.shape[0],
            cols: view.shape[1],
            code: view.base,
            code_bytes: g.code_plane_bytes,
            scale: view
                .scale_plane()
                .ok_or_else(|| fail("NVFP4 tensor is missing scale-plane geometry"))?,
            scale_bytes: g.scale_plane_bytes,
            scale_plane_offset: g.scale_plane_offset,
            weight_divisor: view
                .weight_divisor()
                .ok_or_else(|| fail("NVFP4 tensor is missing weight-divisor geometry"))?,
            weight_divisor_offset: g.weight_divisor_offset,
        })
    }

    /// A typed contiguous BF16 view by name (row-major, 2-byte elements).
    pub fn bf16(&self, name: &str) -> Result<Bf16View> {
        let view = self.tensor(name)?;
        if view.format != NumericFormat::Bf16
            || view.layout != StorageLayout::ContiguousLeV1
        {
            return Err(fail(format!("object {name} is not a contiguous BF16 tensor")));
        }
        if view.shape.len() != 2 || view.shape[0] == 0 || view.shape[1] == 0 {
            return Err(fail(format!("object {name} is not a positive rank-two shape")));
        }
        Ok(Bf16View {
            name: view.name.clone(),
            rows: view.shape[0],
            cols: view.shape[1],
            data: view.base,
            bytes: view.bytes,
        })
    }

    /// The host-retained resource bytes by name (e.g.
    /// `frontend/tokenizer.json` — exactly the raw bytes the reference
    /// `take_resource_bytes` hands to the frontend; extraction is
    /// artifact-02).
    pub fn resource_bytes(&self, name: &str) -> Result<&[u8]> {
        let index = self
            .reader
            .index_of(name)
            .ok_or_else(|| fail(format!("unknown artifact object: {name}")))?;
        self.artifact.resource_bytes(ObjectHandle { index })
    }

    /// The underlying artifact (arena + stats access).
    pub fn artifact(&self) -> &MaterializedArtifact {
        self.artifact
    }
}

// ---------------------------------------------------------------------------
// Tests (CPU mock only — no GPU, ADR 0006)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{CpuDevice, Device};
    use crate::fixture;
    use crate::{Binder, NumericFormat, Reader, ResourceEncoding, StorageLayout};

    /// A 2-object fixture: one BF16 contiguous tensor + one NVFP4
    /// block-scale tensor (tiny sizes) + a raw resource.
    fn small_fixture() -> (Vec<fixture::FixtureObject>, Vec<u8>) {
        let objects = vec![
            fixture::FixtureObject::Resource {
                name: "frontend/tokenizer.json",
                encoding: "raw-bytes-v1",
                offset: 0,
                bytes: 3,
            },
            fixture::FixtureObject::Tensor {
                name: "w/bf16",
                shape: vec![2, 4],
                format: "BF16",
                layout: "contiguous-le-v1",
                offset: 256,
                bytes: 16,
            },
            fixture::FixtureObject::Tensor {
                name: "w/nvfp4",
                shape: vec![128, 64],
                format: "NVFP4",
                layout: "blockscale-k16-m128x4-v1",
                offset: 512,
                bytes: 4612,
            },
        ];
        let mut payload = vec![0u8; 512 + 4612];
        payload[..3].copy_from_slice(&[1u8, 1, 1]);
        payload[256..256 + 16].fill(2);
        payload[512..512 + 4612].fill(3);
        (objects, payload)
    }

    #[test]
    fn typed_binding_consumes_cpu_artifact() {
        let (objects, payload) = small_fixture();
        let fixture = fixture::write_fixture(&objects, &payload, "binding").expect("fixture");
        let reader = Reader::open(&fixture.path).expect("open fixture");

        // Bind all three objects (ADR 0002: consume everything).
        let mut binder = Binder::new(&reader);
        let resource = binder
            .require_resource("frontend/tokenizer.json", ResourceEncoding::RawBytesV1)
            .expect("resource");
        binder.retain_on_host(resource).expect("retain");
        let bf16 = binder
            .require_tensor(
                "w/bf16",
                NumericFormat::Bf16,
                StorageLayout::ContiguousLeV1,
                &[2, 4],
            )
            .expect("bf16");
        binder.materialize_on_device(bf16).expect("place bf16");
        let nvfp4 = binder
            .require_tensor(
                "w/nvfp4",
                NumericFormat::Nvfp4,
                StorageLayout::BlockScaleK16M128x4V1,
                &[128, 64],
            )
            .expect("nvfp4");
        binder.materialize_on_device(nvfp4).expect("place nvfp4");
        let plan = binder.finish().expect("plan");

        let mut device = CpuDevice::new();
        let artifact =
            crate::materialize(&reader, &plan, &mut device, None).expect("materialize");
        let binding = Binding::new(&reader, &artifact);

        // NVFP4 planes: code + scale (with offset) + weight divisor.
        let nv = binding.nvfp4("w/nvfp4").expect("nvfp4 view");
        assert_eq!(nv.rows, 128);
        assert_eq!(nv.cols, 64);
        assert_eq!(nv.code_bytes, 128 * 64 / 2);
        assert_eq!(nv.scale_plane_offset, 4096); // align_up(4096, 256)
        assert_eq!(nv.scale_bytes, 128 * 64 / 16);
        assert_eq!(nv.weight_divisor_offset, 4096 + nv.scale_bytes);
        // The plane pointers are laid out inside the arena.
        assert_eq!(
            nv.scale as usize,
            nv.code as usize + nv.scale_plane_offset as usize
        );
        assert_eq!(
            nv.weight_divisor as usize,
            nv.code as usize + nv.weight_divisor_offset as usize
        );

        // BF16 view: row-major typed access.
        let bf = binding.bf16("w/bf16").expect("bf16 view");
        assert_eq!(bf.rows, 2);
        assert_eq!(bf.cols, 4);
        assert_eq!(bf.bytes, 16);

        // Verify the bytes through the CPU mock (d2h).
        let arena = artifact.arena().expect("arena");
        let mut back = vec![0u8; 16];
        device
            .copy_d2h(
                arena,
                artifact
                    .device_view(bf16)
                    .expect("bf16 view")
                    .offset_in_arena,
                &mut back,
            )
            .expect("d2h");
        assert!(back.iter().all(|&b| b == 2), "bf16 payload differs");

        // Resource bytes round-trip.
        let resource = binding.resource_bytes("frontend/tokenizer.json").expect("resource");
        assert_eq!(resource, &[1u8, 1, 1]);

        // Wrong-kind accessors error cleanly.
        assert!(binding.nvfp4("w/bf16").is_err());
        assert!(binding.bf16("w/nvfp4").is_err());
        assert!(binding.resource_bytes("w/bf16").is_err());
        assert!(binding.tensor("w/missing").is_err());
        drop(reader);
    }
}