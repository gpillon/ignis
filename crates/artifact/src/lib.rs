//! ignis-artifact: NInfer `.ninfer` v2 container reader.
//!
//! Rust port of the reference stack's `src/artifact/` reader + storage-layout
//! modules (ADR 0002): 16-byte binary prefix, closed JSON object directory,
//! checked payload geometry, memory-mapped payload spans, and 4096-aligned
//! direct I/O.
//!
//! The generic reader is deliberately minimal: it validates framing, the
//! closed JSON schema, and payload geometry, then exposes object descriptors
//! and payload spans. It knows nothing about model semantics — per-model
//! binders own that layer (ticket 03 and later).
//!
//! Format reference: `docs/maintainer/artifact-container.md` in the reference
//! tree (v2 framing), mirrored by this module.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use serde_json::Value;

pub mod binding;
pub mod binder;
pub mod checksum;
pub mod device;
pub mod fixture;
pub mod frontend;
pub mod inventory;
pub mod materializer;
pub mod normalize;

/// FFI declarations for the kernel leaf's device surface (feature `cuda`
/// only — the default build is pure Rust).
#[cfg(feature = "cuda")]
mod ffi;

pub use binding::{Binding, Bf16View, Nvfp4View};
pub use binder::{
    Binder, DevicePlacement, HostPlacement, MaterializationPlan, ObjectHandle,
};
pub use checksum::{
    verify, ChecksumReport, GraftedSource, Nvfp4Record, ObjectCheck, Outcome, Sidecar,
};
pub use device::{CpuDevice, Device, DeviceBuffer};
#[cfg(feature = "cuda")]
pub use device::CudaDevice;
pub use frontend::{
    ChatMessage, ChatTemplate, ContentPart, FRONTEND_RESOURCES, FrontendSet, MessageContent,
    Role, ToolCall, Tokenizer,
};
pub use materializer::{materialize, MaterializationStats, MaterializedArtifact, TensorView};
pub use normalize::{dequant_w8_endpoints, normalize_tensor, NormalizedTensor, W8Endpoints};
pub use inventory::{InventoryEntry, OUT_OF_SCOPE_TEXT_NAMES, text_scope_27b};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// A string-carrying error, mirroring the reference `ArtifactError`
/// (`std::runtime_error`-shaped: message is the payload).
#[derive(Debug)]
pub struct ArtifactError(String);

impl std::fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl ArtifactError {
    /// Construct an error with a preformatted message (the error's
    /// payload) — the public half of the crate's private `fail` helper,
    /// for callers outside the crate (the server's loader path,
    /// server-03, refuses a failed checksum report with one).
    pub fn new(message: impl std::fmt::Display) -> Self {
        Self(message.to_string())
    }
}

impl std::error::Error for ArtifactError {}

type Result<T> = std::result::Result<T, ArtifactError>;

fn fail(message: impl std::fmt::Display) -> ArtifactError {
    ArtifactError::new(message)
}

// ---------------------------------------------------------------------------
// Checked arithmetic (u64, mirroring the reference `checked_*` helpers)
// ---------------------------------------------------------------------------

fn checked_add(a: u64, b: u64, label: &str) -> Result<u64> {
    if b > u64::MAX - a {
        return Err(fail(format!("{label} overflows u64")));
    }
    Ok(a + b)
}

fn checked_mul(a: u64, b: u64, label: &str) -> Result<u64> {
    if a != 0 && b > u64::MAX / a {
        return Err(fail(format!("{label} overflows u64")));
    }
    Ok(a * b)
}

fn align_up(value: u64, alignment: u64, label: &str) -> Result<u64> {
    let biased = checked_add(value, alignment - 1, label)?;
    Ok(biased / alignment * alignment)
}

// ---------------------------------------------------------------------------
// Container framing constants
// ---------------------------------------------------------------------------

const PREFIX_BYTES: u64 = 16;

/// The 4096-byte alignment required for direct I/O (file offset, buffer
/// length, and buffer address).
pub const DIRECT_IO_ALIGNMENT: u64 = 4096;

/// Payload alignment: the payload section starts on a 4096-byte boundary.
pub const PAYLOAD_ALIGNMENT: u64 = 4096;

/// File alignment for tensor payloads.
pub const TENSOR_ALIGNMENT: u64 = 256;

/// K-dimension alignment for row-split layouts.
const K_ALIGNMENT: u64 = 128;

const MAGIC: [u8; 8] = *b"NINFER\x00\x02";
const V1_MAGIC: [u8; 8] = *b"NINFER\x00\x01";

// ---------------------------------------------------------------------------
// Registered identities (closed registries — exact-match parsing only)
// ---------------------------------------------------------------------------

/// Persistent numeric formats (the closed v2 registry).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum NumericFormat {
    Bf16,
    Fp32,
    I32,
    Q4G64F16S,
    Q5G64F16S,
    Q6G64F16S,
    W8G32F16S,
    Nvfp4,
    Fp8E4M3FnRowBf16S,
}

impl NumericFormat {
    pub fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::Fp32 => "FP32",
            Self::I32 => "I32",
            Self::Q4G64F16S => "Q4G64_F16S",
            Self::Q5G64F16S => "Q5G64_F16S",
            Self::Q6G64F16S => "Q6G64_F16S",
            Self::W8G32F16S => "W8G32_F16S",
            Self::Nvfp4 => "NVFP4",
            Self::Fp8E4M3FnRowBf16S => "FP8_E4M3FN_ROW_BF16S",
        }
    }

    fn parse(name: &str) -> Result<Self> {
        Ok(match name {
            "BF16" => Self::Bf16,
            "FP32" => Self::Fp32,
            "I32" => Self::I32,
            "Q4G64_F16S" => Self::Q4G64F16S,
            "Q5G64_F16S" => Self::Q5G64F16S,
            "Q6G64_F16S" => Self::Q6G64F16S,
            "W8G32_F16S" => Self::W8G32F16S,
            "NVFP4" => Self::Nvfp4,
            "FP8_E4M3FN_ROW_BF16S" => Self::Fp8E4M3FnRowBf16S,
            other => return Err(fail(format!("unknown tensor format: {other}"))),
        })
    }
}

/// Persistent tensor storage layouts (the closed v2 registry).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageLayout {
    ContiguousLeV1,
    RowSplitK128V1,
    BlockScaleK16M128x4V1,
    RowScaleV1,
}

impl StorageLayout {
    pub fn name(self) -> &'static str {
        match self {
            Self::ContiguousLeV1 => "contiguous-le-v1",
            Self::RowSplitK128V1 => "row-split-k128-v1",
            Self::BlockScaleK16M128x4V1 => "blockscale-k16-m128x4-v1",
            Self::RowScaleV1 => "row-scale-v1",
        }
    }

    fn parse(name: &str) -> Result<Self> {
        Ok(match name {
            "contiguous-le-v1" => Self::ContiguousLeV1,
            "row-split-k128-v1" => Self::RowSplitK128V1,
            "blockscale-k16-m128x4-v1" => Self::BlockScaleK16M128x4V1,
            "row-scale-v1" => Self::RowScaleV1,
            other => return Err(fail(format!("unknown tensor layout: {other}"))),
        })
    }
}

/// Required-resource encodings (the closed v2 registry).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceEncoding {
    RawBytesV1,
}

impl ResourceEncoding {
    pub fn name(self) -> &'static str {
        match self {
            Self::RawBytesV1 => "raw-bytes-v1",
        }
    }

    fn parse(name: &str) -> Result<Self> {
        Ok(match name {
            "raw-bytes-v1" => Self::RawBytesV1,
            other => return Err(fail(format!("unknown resource encoding: {other}"))),
        })
    }
}

/// File alignment for a tensor layout (all tensor layouts align to 256 B).
pub fn tensor_alignment(_layout: StorageLayout) -> u64 {
    TENSOR_ALIGNMENT
}

/// File alignment for a resource encoding (raw bytes align to 1 B).
pub fn resource_alignment(_encoding: ResourceEncoding) -> u64 {
    1
}

// ---------------------------------------------------------------------------
// Payload geometry (port of the reference `storage_layouts.cpp`)
// ---------------------------------------------------------------------------

/// Geometry of a `row-split-k128-v1` tensor payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowSplitGeometry {
    pub rows: u64,
    pub columns: u64,
    pub padded_columns: u64,
    pub group_size: u64,
    pub groups_per_row: u64,
    pub low_bytes_per_group: u64,
    pub high_bytes_per_group: u64,
    pub low_plane_bytes: u64,
    pub high_plane_bytes: u64,
    pub scale_plane_bytes: u64,
    pub high_plane_offset: u64,
    pub scale_plane_offset: u64,
    pub encoded_bytes: u64,
}

/// Geometry of a `blockscale-k16-m128x4-v1` (NVFP4) tensor payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockScaleGeometry {
    pub rows: u64,
    pub columns: u64,
    pub groups_per_row: u64,
    pub k_tiles: u64,
    pub code_plane_bytes: u64,
    pub scale_plane_offset: u64,
    pub scale_plane_bytes: u64,
    pub weight_divisor_offset: u64,
    pub encoded_bytes: u64,
}

/// Geometry of a `row-scale-v1` (FP8 E4M3FN) tensor payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowScaleGeometry {
    pub rows: u64,
    pub columns: u64,
    pub code_plane_bytes: u64,
    pub scale_plane_offset: u64,
    pub scale_plane_bytes: u64,
    pub encoded_bytes: u64,
}

/// (group_size, base_bytes_per_group, high_bytes_per_group) per quantized
/// format for `row-split-k128-v1`.
fn quant_geometry(format: NumericFormat) -> Result<(u64, u64, u64)> {
    Ok(match format {
        NumericFormat::Q4G64F16S => (64, 32, 0),
        NumericFormat::Q5G64F16S => (64, 32, 8),
        NumericFormat::Q6G64F16S => (64, 32, 16),
        NumericFormat::W8G32F16S => (32, 32, 0),
        _ => {
            return Err(fail("row-split-k128-v1 requires a grouped quantized format"));
        }
    })
}

/// Direct-word size in bytes for `contiguous-le-v1` formats.
fn direct_word_bytes(format: NumericFormat) -> Result<u64> {
    Ok(match format {
        NumericFormat::Bf16 => 2,
        NumericFormat::Fp32 | NumericFormat::I32 => 4,
        _ => return Err(fail("contiguous-le-v1 requires BF16, FP32, or I32")),
    })
}

fn require_rank2(label: &str, shape: &[u64]) -> Result<()> {
    if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
        return Err(fail(format!(
            "{label} requires a positive rank-two shape"
        )));
    }
    Ok(())
}

/// Encoded payload size of a tensor with the given (layout, format, shape).
///
/// Mirrors the reference `tensor_encoded_size`; the layout's exact encoded-size
/// rule must equal the stored `bytes` for every object (enforced at read time).
pub fn tensor_encoded_size(
    layout: StorageLayout,
    format: NumericFormat,
    shape: &[u64],
) -> Result<u64> {
    match layout {
        StorageLayout::ContiguousLeV1 => {
            if shape.len() > 16 {
                return Err(fail("contiguous-le-v1 supports rank 0 through 16"));
            }
            let mut elements = 1u64;
            for &dim in shape {
                if dim == 0 {
                    return Err(fail("tensor shape dimensions must be positive"));
                }
                elements = checked_mul(elements, dim, "tensor element count")?;
            }
            let word_bytes = direct_word_bytes(format)?;
            checked_mul(elements, word_bytes, "tensor encoded size")
        }
        StorageLayout::RowSplitK128V1 => {
            require_rank2("row-split-k128-v1", shape)?;
            Ok(row_split_geometry(format, shape)?.encoded_bytes)
        }
        StorageLayout::BlockScaleK16M128x4V1 => {
            Ok(block_scale_geometry(format, shape)?.encoded_bytes)
        }
        StorageLayout::RowScaleV1 => {
            Ok(row_scale_geometry(format, shape)?.encoded_bytes)
        }
    }
}

/// Geometry of a `row-split-k128-v1` payload for a (format, shape) tensor.
pub fn row_split_geometry(format: NumericFormat, shape: &[u64]) -> Result<RowSplitGeometry> {
    require_rank2("row-split-k128-v1", shape)?;
    let (group_size, low_bpg, high_bpg) = quant_geometry(format)?;
    let (rows, columns) = (shape[0], shape[1]);
    let padded_columns = align_up(columns, K_ALIGNMENT, "padded K")?;
    let groups_per_row = padded_columns / group_size;
    let groups = checked_mul(rows, groups_per_row, "physical group count")?;
    let low_plane_bytes = checked_mul(groups, low_bpg, "base plane bytes")?;
    let high_plane_bytes = checked_mul(groups, high_bpg, "high plane bytes")?;
    let scale_plane_bytes = checked_mul(groups, 2, "scale plane bytes")?;
    let high_plane_offset = align_up(low_plane_bytes, TENSOR_ALIGNMENT, "high plane offset")?;
    let aligned_high =
        align_up(high_plane_bytes, TENSOR_ALIGNMENT, "scale plane alignment")?;
    let scale_plane_offset = checked_add(
        high_plane_offset,
        aligned_high,
        "scale plane offset",
    )?;
    let encoded_bytes =
        checked_add(scale_plane_offset, scale_plane_bytes, "tensor encoded size")?;
    Ok(RowSplitGeometry {
        rows,
        columns,
        padded_columns,
        group_size,
        groups_per_row,
        low_bytes_per_group: low_bpg,
        high_bytes_per_group: high_bpg,
        low_plane_bytes,
        high_plane_bytes,
        scale_plane_bytes,
        high_plane_offset,
        scale_plane_offset,
        encoded_bytes,
    })
}

/// Geometry of a `blockscale-k16-m128x4-v1` (NVFP4) payload for a shape.
///
/// Requires NVFP4, a positive rank-two shape, N divisible by 128 and K
/// divisible by 64. The payload is a packed E2M1 code plane, a swizzled
/// E4M3FN scale plane, and a trailing little-endian FP32 weight divisor.
pub fn block_scale_geometry(format: NumericFormat, shape: &[u64]) -> Result<BlockScaleGeometry> {
    if format != NumericFormat::Nvfp4 {
        return Err(fail("blockscale-k16-m128x4-v1 requires NVFP4"));
    }
    require_rank2("blockscale-k16-m128x4-v1", shape)?;
    let (rows, columns) = (shape[0], shape[1]);
    if rows % 128 != 0 || columns % 64 != 0 {
        return Err(fail(
            "blockscale-k16-m128x4-v1 requires N divisible by 128 and K divisible by 64",
        ));
    }
    let groups_per_row = columns / 16;
    let k_tiles = columns / 64;
    let elements = checked_mul(rows, columns, "NVFP4 element count")?;
    let code_plane_bytes = elements / 2;
    let scale_plane_offset =
        align_up(code_plane_bytes, TENSOR_ALIGNMENT, "NVFP4 scale plane offset")?;
    let scale_plane_bytes = elements / 16;
    let weight_divisor_offset = checked_add(
        scale_plane_offset,
        scale_plane_bytes,
        "NVFP4 weight divisor offset",
    )?;
    let encoded_bytes = checked_add(weight_divisor_offset, 4, "NVFP4 tensor encoded size")?;
    Ok(BlockScaleGeometry {
        rows,
        columns,
        groups_per_row,
        k_tiles,
        code_plane_bytes,
        scale_plane_offset,
        scale_plane_bytes,
        weight_divisor_offset,
        encoded_bytes,
    })
}

/// Geometry of a `row-scale-v1` (FP8 E4M3FN) payload for a shape.
///
/// Requires the FP8_E4M3FN_ROW_BF16S format and a positive rank-two shape.
pub fn row_scale_geometry(format: NumericFormat, shape: &[u64]) -> Result<RowScaleGeometry> {
    if format != NumericFormat::Fp8E4M3FnRowBf16S {
        return Err(fail("row-scale-v1 requires FP8_E4M3FN_ROW_BF16S"));
    }
    require_rank2("row-scale-v1", shape)?;
    let (rows, columns) = (shape[0], shape[1]);
    let code_plane_bytes = checked_mul(rows, columns, "FP8 element count")?;
    let scale_plane_offset =
        align_up(code_plane_bytes, TENSOR_ALIGNMENT, "FP8 scale plane offset")?;
    let scale_plane_bytes = checked_mul(rows, 2, "FP8 scale plane bytes")?;
    let encoded_bytes =
        checked_add(scale_plane_offset, scale_plane_bytes, "FP8 tensor encoded size")?;
    Ok(RowScaleGeometry {
        rows,
        columns,
        code_plane_bytes,
        scale_plane_offset,
        scale_plane_bytes,
        encoded_bytes,
    })
}

// ---------------------------------------------------------------------------
// Object descriptors
// ---------------------------------------------------------------------------

/// A tensor object directory entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDescriptor {
    pub name: String,
    pub shape: Vec<u64>,
    pub format: NumericFormat,
    pub layout: StorageLayout,
    /// Byte offset relative to the payload section start.
    pub offset: u64,
    /// Exact stored payload length.
    pub bytes: u64,
}

/// A required-resource object directory entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceDescriptor {
    pub name: String,
    pub encoding: ResourceEncoding,
    /// Byte offset relative to the payload section start.
    pub offset: u64,
    /// Exact stored payload length.
    pub bytes: u64,
}

/// One object in the container directory (tensor or required resource).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Object {
    Tensor(TensorDescriptor),
    Resource(ResourceDescriptor),
}

impl Object {
    pub fn kind(&self) -> &'static str {
        match self {
            Object::Tensor(_) => "tensor",
            Object::Resource(_) => "resource",
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Object::Tensor(d) => &d.name,
            Object::Resource(d) => &d.name,
        }
    }

    pub fn offset(&self) -> u64 {
        match self {
            Object::Tensor(d) => d.offset,
            Object::Resource(d) => d.offset,
        }
    }

    pub fn bytes(&self) -> u64 {
        match self {
            Object::Tensor(d) => d.bytes,
            Object::Resource(d) => d.bytes,
        }
    }

    /// File alignment for this object's payload (256 B tensors, 1 B resources).
    pub fn alignment(&self) -> u64 {
        match self {
            Object::Tensor(d) => tensor_alignment(d.layout),
            Object::Resource(d) => resource_alignment(d.encoding),
        }
    }
}

/// Exact hierarchical identity: (model_id, weights_id).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactIdentity {
    pub model_id: String,
    pub weights_id: String,
}

// ---------------------------------------------------------------------------
// Closed-JSON validation helpers
// ---------------------------------------------------------------------------

fn require_members(value: &Value, members: &[&str], label: &str) -> Result<()> {
    let obj = value
        .as_object()
        .ok_or_else(|| fail(format!("{label} must be a JSON object")))?;
    if obj.len() != members.len() {
        return Err(fail(format!("{label} has missing or extra members")));
    }
    for member in members {
        if !obj.contains_key(*member) {
            return Err(fail(format!("{label} has missing or extra members")));
        }
    }
    Ok(())
}

fn require_string<'a>(value: &'a Value, label: &str) -> Result<&'a str> {
    match value.as_str() {
        Some(s) if !s.is_empty() => Ok(s),
        _ => Err(fail(format!("{label} must be a nonempty string"))),
    }
}

fn require_unsigned(value: &Value, label: &str, positive: bool) -> Result<u64> {
    match value.as_u64() {
        Some(n) if !positive || n != 0 => Ok(n),
        _ => Err(fail(format!("{label} must be an integer"))),
    }
}

fn parse_tensor(value: &Value) -> Result<Object> {
    const MEMBERS: [&str; 7] =
        ["name", "kind", "shape", "format", "layout", "offset", "bytes"];
    require_members(value, &MEMBERS, "tensor entry")?;

    let name = require_string(value.get("name").unwrap(), "tensor name")?;
    let format = NumericFormat::parse(require_string(
        value.get("format").unwrap(),
        "tensor format",
    )?)?;
    let layout = StorageLayout::parse(require_string(
        value.get("layout").unwrap(),
        "tensor layout",
    )?)?;
    let offset = require_unsigned(value.get("offset").unwrap(), "tensor offset", false)?;
    let stored_bytes = require_unsigned(value.get("bytes").unwrap(), "tensor bytes", true)?;

    let raw_shape = value
        .get("shape")
        .and_then(|v| v.as_array())
        .ok_or_else(|| fail("tensor shape must be an array"))?;
    let mut shape = Vec::with_capacity(raw_shape.len());
    for dim in raw_shape {
        shape.push(require_unsigned(dim, "shape dimension", true)?);
    }

    let encoded_bytes = tensor_encoded_size(layout, format, &shape)?;
    if stored_bytes != encoded_bytes {
        return Err(fail(format!(
            "tensor {name} stores {stored_bytes} bytes; layout requires {encoded_bytes}"
        )));
    }

    Ok(Object::Tensor(TensorDescriptor {
        name: name.to_owned(),
        shape,
        format,
        layout,
        offset,
        bytes: stored_bytes,
    }))
}

fn parse_resource(value: &Value) -> Result<Object> {
    const MEMBERS: [&str; 5] =
        ["name", "kind", "encoding", "offset", "bytes"];
    require_members(value, &MEMBERS, "resource entry")?;

    let name = require_string(value.get("name").unwrap(), "resource name")?;
    let encoding = ResourceEncoding::parse(require_string(
        value.get("encoding").unwrap(),
        "resource encoding",
    )?)?;
    let offset = require_unsigned(value.get("offset").unwrap(), "resource offset", false)?;
    let bytes = require_unsigned(value.get("bytes").unwrap(), "resource bytes", true)?;

    Ok(Object::Resource(ResourceDescriptor {
        name: name.to_owned(),
        encoding,
        offset,
        bytes,
    }))
}

fn parse_object(value: &Value) -> Result<Object> {
    let obj = value
        .as_object()
        .ok_or_else(|| fail("each object entry must be a JSON object"))?;
    let kind_value = obj
        .get("kind")
        .ok_or_else(|| fail("object kind must be 'tensor' or 'resource'"))?;
    let kind = kind_value
        .as_str()
        .ok_or_else(|| fail("object kind must be 'tensor' or 'resource'"))?;
    match kind {
        "tensor" => parse_tensor(value),
        "resource" => parse_resource(value),
        _ => Err(fail("object kind must be 'tensor' or 'resource'")),
    }
}

// ---------------------------------------------------------------------------
// Direct I/O (platform backends)
// ---------------------------------------------------------------------------

/// A file handle opened for aligned direct I/O (bypasses the page cache),
/// mirroring the reference `MappedFile` direct-handle half.
#[cfg(windows)]
mod direct_io {
    use std::ffi::c_void;
    use std::os::windows::ffi::OsStrExt;

    use windows_sys::Win32::Foundation::{
        CloseHandle, GENERIC_READ, GetLastError, INVALID_HANDLE_VALUE,
    };
    use windows_sys::Win32::Storage::FileSystem::{
        CreateFileW, FILE_ATTRIBUTE_NORMAL, FILE_BEGIN, FILE_FLAG_NO_BUFFERING,
        FILE_FLAG_SEQUENTIAL_SCAN, FILE_SHARE_READ, OPEN_EXISTING, ReadFile, SetFilePointerEx,
    };

    pub struct DirectFile {
        handle: *mut c_void,
    }

    const CHUNK_BYTES: u32 = 1 << 30;
    const ERROR_END_OF_FILE: u32 = 138;

    impl DirectFile {
        pub fn open(path: &std::path::Path) -> std::io::Result<Self> {
            let mut wide: Vec<u16> = path.as_os_str().encode_wide().collect();
            wide.push(0);
            let handle = unsafe {
                CreateFileW(
                    wide.as_ptr(),
                    GENERIC_READ,
                    FILE_SHARE_READ,
                    std::ptr::null(),
                    OPEN_EXISTING,
                    FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN,
                    std::ptr::null_mut(),
                )
            };
            if handle == INVALID_HANDLE_VALUE {
                return Err(std::io::Error::last_os_error());
            }
            Ok(Self { handle })
        }

        pub fn read(&self, offset: u64, destination: &mut [u8]) -> std::io::Result<usize> {
            // Position the NO_BUFFERING handle at the (4096-aligned) offset, then
            // read synchronously in 1 GiB chunks.
            let positioned =
                unsafe { SetFilePointerEx(self.handle, offset as i64, std::ptr::null_mut(), FILE_BEGIN) };
            if positioned == 0 {
                return Err(std::io::Error::last_os_error());
            }
            let mut total = 0usize;
            while total < destination.len() {
                let amount = std::cmp::min(
                    CHUNK_BYTES,
                    (destination.len() - total) as u32,
                );
                let mut read = 0u32;
                let ok = unsafe {
                    ReadFile(
                        self.handle,
                        destination.as_mut_ptr().add(total),
                        amount,
                        &mut read,
                        std::ptr::null_mut(),
                    )
                };
                if ok == 0 {
                    let code = unsafe { GetLastError() };
                    // Short read at EOF (or 0 bytes): stop, not an error.
                    if code == ERROR_END_OF_FILE || read == 0 {
                        break;
                    }
                    return Err(std::io::Error::from_raw_os_error(code as i32));
                }
                total += read as usize;
                if read < amount {
                    break;
                }
            }
            Ok(total)
        }
    }

    impl Drop for DirectFile {
        fn drop(&mut self) {
            unsafe {
                CloseHandle(self.handle);
            }
        }
    }
}

#[cfg(unix)]
mod direct_io {
    use libc::{close, open, pread, O_CLOEXEC, O_DIRECT, O_RDONLY};
    use std::os::unix::ffi::OsStrExt;

    pub struct DirectFile {
        fd: std::os::unix::io::RawFd,
    }

    impl DirectFile {
        pub fn open(path: &std::path::Path) -> std::io::Result<Self> {
            let bytes = path.as_os_str().as_bytes();
            let ptr = bytes.as_ptr() as *const libc::c_char;
            let fd = unsafe { open(ptr, O_RDONLY | O_CLOEXEC | O_DIRECT) };
            if fd < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(Self { fd })
        }

        pub fn read(&self, offset: u64, destination: &mut [u8]) -> std::io::Result<usize> {
            let mut total = 0usize;
            while total < destination.len() {
                let amount = destination.len() - total;
                let n = unsafe {
                    pread(
                        self.fd,
                        destination.as_mut_ptr().add(total) as *mut std::ffi::c_void,
                        amount,
                        (offset + total as u64) as libc::off_t,
                    )
                };
                if n < 0 {
                    let err = std::io::Error::last_os_error();
                    if err.raw_os_error() == Some(libc::EINTR) {
                        continue;
                    }
                    return Err(err);
                }
                if n == 0 {
                    break;
                }
                total += n as usize;
                if (n as usize) < amount {
                    break;
                }
            }
            Ok(total)
        }
    }

    impl Drop for DirectFile {
        fn drop(&mut self) {
            unsafe {
                close(self.fd);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// A memory-mapped `.ninfer` v2 artifact with its validated object directory.
///
/// Construct with [`Reader::open`]; the mapping and the direct-I/O handle are
/// owned and released on drop.
pub struct Reader {
    /// Kept alive for the reader's lifetime so the mapping's backing handle
    /// stays valid; the mapping itself is owned by `mmap`.
    #[allow(dead_code)]
    file: File,
    mmap: Mmap,
    direct: direct_io::DirectFile,
    identity: ArtifactIdentity,
    objects: Vec<Object>,
    index: HashMap<String, usize>,
    payload_start: u64,
    file_bytes: u64,
}

/// A validated payload span, resolved to an absolute file offset and a
/// mapping-backed byte slice.
pub struct PayloadSpan<'a> {
    /// Absolute file offset of the payload start.
    pub absolute_offset: u64,
    /// The payload bytes (borrowed from the reader's mapping).
    pub data: &'a [u8],
}

impl std::fmt::Debug for Reader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reader")
            .field("identity", &self.identity)
            .field("object_count", &self.objects.len())
            .field("payload_start", &self.payload_start)
            .field("file_bytes", &self.file_bytes)
            .finish()
    }
}

impl Reader {
    /// Open, map, and validate a `.ninfer` v2 artifact.
    ///
    /// Rejects v1 artifacts (with a migration hint), invalid framing, schema
    /// violations, and any geometry/ordering/alignment/bounds inconsistency.
    pub fn open(path: &Path) -> Result<Reader> {
        let file = File::open(path)
            .map_err(|e| fail(format!("open {}: {e}", path.display())))?;
        let file_bytes = file
            .metadata()
            .map_err(|e| fail(format!("stat {}: {e}", path.display())))?
            .len();

        if file_bytes < PREFIX_BYTES {
            return Err(fail("artifact is shorter than the v2 prefix"));
        }

        // --- memory-map the whole file (payload spans come from the map) ---
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| fail(format!("mmap {}: {e}", path.display())))?;

        // --- prefix: magic + little-endian json_bytes (read from the map) ---
        let prefix = &mmap[..PREFIX_BYTES as usize];
        if prefix[..8] == V1_MAGIC {
            return Err(fail(
                "NInfer artifact v1 is no longer supported; migrate it with: \
                 python3 -m tools.artifact.migrate_v1_to_v2 <artifact>",
            ));
        }
        if prefix[..8] != MAGIC {
            return Err(fail("artifact magic is not NInfer v2"));
        }

        let json_bytes = u64::from_le_bytes(prefix[8..16].try_into().unwrap());
        if json_bytes == 0 {
            return Err(fail("json_bytes must be positive"));
        }
        let metadata_end = checked_add(PREFIX_BYTES, json_bytes, "JSON range")?;
        let payload_start = align_up(metadata_end, PAYLOAD_ALIGNMENT, "payload offset")?;
        if metadata_end > file_bytes || payload_start > file_bytes {
            return Err(fail("declared JSON or payload start extends beyond the file"));
        }

        // --- parse + validate the closed JSON directory ---------------------
        let json_range = PREFIX_BYTES as usize..metadata_end as usize;
        let directory: Value =
            serde_json::from_slice(&mmap[json_range.clone()])
                .map_err(|e| fail(format!("invalid JSON directory: {e}")))?;

        require_members(&directory, &["identity", "objects"], "directory root")?;
        let raw_identity = &directory["identity"];
        require_members(raw_identity, &["model_id", "weights_id"], "artifact identity")?;
        let identity = ArtifactIdentity {
            model_id: require_string(&raw_identity["model_id"], "model_id")?.to_owned(),
            weights_id: require_string(&raw_identity["weights_id"], "weights_id")?.to_owned(),
        };

        let raw_objects = directory
            .get("objects")
            .and_then(|v| v.as_array())
            .ok_or_else(|| fail("objects must be a nonempty array"))?;
        if raw_objects.is_empty() {
            return Err(fail("objects must be a nonempty array"));
        }

        // --- validate ordering, alignment, bounds, and duplicate names -----
        let payload_bytes = file_bytes - payload_start;
        let mut cursor: u64 = 0;
        let mut objects = Vec::with_capacity(raw_objects.len());
        let mut index: HashMap<String, usize> = HashMap::with_capacity(raw_objects.len());

        for raw_object in raw_objects {
            let object = parse_object(raw_object)?;
            let name = object.name().to_owned();
            let offset = object.offset();
            let bytes = object.bytes();
            let alignment = object.alignment();

            if offset < cursor {
                return Err(fail(format!("object {name} overlaps or is out of order")));
            }
            if offset % alignment != 0 {
                return Err(fail(format!(
                    "object {name} is not {alignment}-byte aligned"
                )));
            }
            let end = checked_add(offset, bytes, "object payload range")?;
            if end > payload_bytes {
                return Err(fail(format!("object {name} extends beyond the file")));
            }
            if index.insert(name.clone(), objects.len()).is_some() {
                return Err(fail(format!("duplicate object name: {name}")));
            }
            objects.push(object);
            cursor = end;
        }

        // --- direct-I/O handle (bypasses the page cache for tensor reads) --
        let direct =
            direct_io::DirectFile::open(path).map_err(|e| fail(format!("open direct handle for {}: {e}", path.display())))?;

        Ok(Reader {
            file,
            mmap,
            direct,
            identity,
            objects,
            index,
            payload_start,
            file_bytes,
        })
    }

    /// The exact hierarchical (model_id, weights_id) identity.
    pub fn identity(&self) -> &ArtifactIdentity {
        &self.identity
    }

    /// All directory objects, in physical-offset order.
    pub fn objects(&self) -> &[Object] {
        &self.objects
    }

    /// Look up an object by exact name.
    pub fn find(&self, name: &str) -> Option<&Object> {
        self.index.get(name).map(|&i| &self.objects[i])
    }

    /// The index of `name` in the object directory (its position in
    /// [`Reader::objects()`]), or None if unknown.
    ///
    /// This is what the binder's `ObjectHandle`s carry: a handle names a
    /// directory slot, not an object.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.index.get(name).copied()
    }

    /// Total file size in bytes.
    pub fn file_bytes(&self) -> u64 {
        self.file_bytes
    }

    /// File offset where the payload section starts (4096-aligned).
    pub fn payload_offset(&self) -> u64 {
        self.payload_start
    }

    /// The whole file as a byte view (mapping-backed; pages fault on access).
    pub fn mapped_bytes(&self) -> &[u8] {
        &self.mmap[..]
    }

    /// Validate + resolve an object's payload span in the mapping.
    pub fn payload_at(&self, object: &Object) -> Result<PayloadSpan<'_>> {
        let absolute =
            checked_add(self.payload_start, object.offset(), "absolute payload offset")?;
        let end = checked_add(absolute, object.bytes(), "absolute payload range")?;
        if end > self.file_bytes {
            return Err(fail("object payload extends beyond the file"));
        }
        // Safety: the mapping outlives the returned span (both borrow from
        // self); the [absolute, end) range was bounds-checked against the
        // file (mapping) size.
        let base = self.mapped_bytes().as_ptr();
        let data = unsafe {
            std::slice::from_raw_parts(base.add(absolute as usize), object.bytes() as usize)
        };
        Ok(PayloadSpan {
            absolute_offset: absolute,
            data,
        })
    }

    /// Validate + resolve a named object's payload span in the mapping.
    pub fn payload(&self, name: &str) -> Result<PayloadSpan<'_>> {
        let object = self
            .find(name)
            .ok_or_else(|| fail(format!("unknown artifact object: {name}")))?;
        self.payload_at(object)
    }

    /// Read `destination.len()` bytes at an absolute file offset via direct
    /// I/O (bypasses the page cache).
    ///
    /// The offset, the buffer length, and the buffer address must all be
    /// aligned to [`DIRECT_IO_ALIGNMENT`] (4096). Returns the bytes read; a
    /// short read at end-of-file is not an error.
    pub fn read_direct(&self, absolute_offset: u64, destination: &mut [u8]) -> Result<usize> {
        const ALIGN: usize = DIRECT_IO_ALIGNMENT as usize;
        if !absolute_offset.is_multiple_of(DIRECT_IO_ALIGNMENT)
            || !destination.len().is_multiple_of(ALIGN)
            || !(destination.as_mut_ptr() as usize).is_multiple_of(ALIGN)
        {
            return Err(fail("direct artifact read is not 4096-byte aligned"));
        }
        self.direct
            .read(absolute_offset, destination)
            .map_err(|e| fail(format!("direct artifact read: {e}")))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- synthetic fixture builder -----------------------------------------

    /// Assemble a complete, minimal v2 artifact from a JSON directory string
    /// and a payload buffer (offsets in the directory are payload-relative).
    fn build_artifact(directory_json: &str, payload: &[u8]) -> Vec<u8> {
        let json_len = directory_json.len() as u64;
        let metadata_end = PREFIX_BYTES + json_len;
        let payload_start = metadata_end.div_ceil(PAYLOAD_ALIGNMENT)
            * PAYLOAD_ALIGNMENT;
        let padding = (payload_start - metadata_end) as usize;
        let mut file = Vec::new();
        file.extend_from_slice(&MAGIC);
        file.extend_from_slice(&json_len.to_le_bytes());
        file.extend_from_slice(directory_json.as_bytes());
        file.extend(std::iter::repeat_n(0u8, padding));
        file.extend_from_slice(payload);
        file
    }

    const ID: (&str, &str) = ("test-model", "test-weights");

    /// Wrap an `objects` inner string (comma-separated JSON object literals,
    /// single braces) in the closed `identity` + `objects` envelope.
    ///
    /// `objects_inner` is a *value* substituted for `{2}`, so its braces are
    /// emitted verbatim — no re-escaping needed.
    fn dir_with(objects_inner: &str) -> String {
        format!(
            r#"{{"identity":{{"model_id":"{0}","weights_id":"{1}"}},
                 "objects":[{2}]}}"#,
            ID.0, ID.1, objects_inner
        )
    }

    /// Write `bytes` to a unique temp file, run `f` on the path, and clean up.
    fn with_artifact_file(name: &str, bytes: &[u8], f: impl FnOnce(&Path)) {
        let path = std::env::temp_dir().join(format!(
            "ignis-artifact-{name}-{}.ninfer",
            std::process::id()
        ));
        std::fs::write(&path, bytes).unwrap();
        f(&path);
        let _ = std::fs::remove_file(&path);
    }

    /// A well-formed 2-object artifact: a 64-byte BF16 tensor at offset 0 and
    /// a 4096-byte raw resource at offset 64 (large enough to exercise an
    /// aligned 4096-byte direct read).
    fn valid_two_object_file() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let objects = r#"{ "name":"t/x","kind":"tensor","shape":[4,8],"format":"BF16",
                           "layout":"contiguous-le-v1","offset":0,"bytes":64},
                         { "name":"frontend/r","kind":"resource","encoding":"raw-bytes-v1",
                           "offset":64,"bytes":4096 }"#;
        let directory = dir_with(objects);
        let mut payload = vec![0xA5u8; 64];
        payload.extend(vec![0x5Au8; 4096]);
        let file = build_artifact(&directory, &payload);
        (file, vec![0xA5u8; 64], vec![0x5Au8; 4096])
    }

    // -- geometry ------------------------------------------------------------

    #[test]
    fn block_scale_geometry_nvfp4() {
        let g = block_scale_geometry(NumericFormat::Nvfp4, &[128, 64]).unwrap();
        assert_eq!(g.code_plane_bytes, 128 * 64 / 2);
        assert_eq!(g.scale_plane_offset, 4096);
        assert_eq!(g.scale_plane_bytes, 128 * 64 / 16);
        assert_eq!(g.weight_divisor_offset, g.scale_plane_offset + g.scale_plane_bytes);
        assert_eq!(g.encoded_bytes, g.weight_divisor_offset + 4);
    }

    #[test]
    fn row_split_geometry_q4() {
        // Q4G64_F16S: group 64, base 32, high 0. Shape [128,128]:
        // padded K=128, groups_per_row=2, groups=256, low=8192, high=0,
        // scale=512, high_plane_offset=8192 (align 256), scale_offset=8192,
        // encoded=8704.
        let g = row_split_geometry(NumericFormat::Q4G64F16S, &[128, 128]).unwrap();
        assert_eq!(g.padded_columns, 128);
        assert_eq!(g.groups_per_row, 2);
        assert_eq!(g.low_plane_bytes, 8192);
        assert_eq!(g.high_plane_bytes, 0);
        assert_eq!(g.scale_plane_bytes, 512);
        assert_eq!(g.high_plane_offset, 8192);
        assert_eq!(g.scale_plane_offset, 8192);
        assert_eq!(g.encoded_bytes, 8704);
    }

    #[test]
    fn row_scale_geometry_fp8() {
        // FP8 row-scale [256,512]: code=131072, scale_offset=131072 (align
        // 256), scale=512, encoded=131584.
        let g = row_scale_geometry(NumericFormat::Fp8E4M3FnRowBf16S, &[256, 512]).unwrap();
        assert_eq!(g.code_plane_bytes, 256 * 512);
        assert_eq!(g.scale_plane_offset, 131072);
        assert_eq!(g.scale_plane_bytes, 512);
        assert_eq!(g.encoded_bytes, 131584);
    }

    #[test]
    fn contiguous_encoded_size() {
        assert_eq!(
            tensor_encoded_size(
                StorageLayout::ContiguousLeV1,
                NumericFormat::Bf16,
                &[2, 3]
            )
            .unwrap(),
            12
        );
        assert_eq!(
            tensor_encoded_size(
                StorageLayout::ContiguousLeV1,
                NumericFormat::Fp32,
                &[]
            )
            .unwrap(),
            4
        );
    }

    #[test]
    fn geometry_rejects_bad_combos() {
        assert!(block_scale_geometry(NumericFormat::Bf16, &[128, 64]).is_err());
        assert!(block_scale_geometry(NumericFormat::Nvfp4, &[64, 64]).is_err());
        assert!(row_scale_geometry(NumericFormat::Bf16, &[256, 512]).is_err());
        assert!(row_split_geometry(NumericFormat::Bf16, &[128, 128]).is_err());
    }

    // -- reader round trip ---------------------------------------------------

    #[test]
    fn synthetic_round_trip() {
        let (file, tensor_bytes, resource_bytes) = valid_two_object_file();
        with_artifact_file("round-trip", &file, |path| {
            let reader = Reader::open(path).expect("open synthetic artifact");
            assert_eq!(reader.identity().model_id, ID.0);
            assert_eq!(reader.identity().weights_id, ID.1);
            assert_eq!(reader.objects().len(), 2);
            assert_eq!(reader.file_bytes() as usize, file.len());
            assert!(reader.payload_offset().is_multiple_of(PAYLOAD_ALIGNMENT));

            let tensor = reader
                .find("t/x")
                .expect("tensor present")
                .clone();
            assert!(matches!(tensor, Object::Tensor(_)));
            let t = match &tensor {
                Object::Tensor(t) => t,
                _ => panic!(),
            };
            assert_eq!(t.shape, vec![4, 8]);
            assert_eq!(t.offset, 0);
            assert_eq!(t.bytes, 64);

            let span = reader.payload_at(&tensor).expect("tensor span");
            assert_eq!(span.data, tensor_bytes.as_slice());

            let resource = reader
                .find("frontend/r")
                .expect("resource present")
                .clone();
            let rs = reader.payload_at(&resource).expect("resource span");
            assert_eq!(rs.data, resource_bytes.as_slice());

            // read_direct sanity: an aligned 4096-byte read at the payload
            // start equals the mapping view.
            let payload_offset = reader.payload_offset();
            let mut buf = vec![0u8; 8192];
            let ptr = buf.as_mut_ptr() as usize;
            let aligned_base = ptr.div_ceil(4096) * 4096;
            let off = aligned_base - ptr;
            let dst = &mut buf[off..off + 4096];
            let n = reader.read_direct(payload_offset, dst).expect("direct read");
            assert_eq!(n, 4096);
            assert_eq!(
                dst,
                &reader.mapped_bytes()[payload_offset as usize..(payload_offset + 4096) as usize]
            );
        });
    }

    // -- reader rejection paths ---------------------------------------------

    #[test]
    fn bad_magic_rejected() {
        let mut file = valid_two_object_file().0;
        file[0] = b'X';
        with_artifact_file("bad-magic", &file, |path| {
            let err = Reader::open(path).unwrap_err().to_string();
            assert!(err.contains("magic"), "{err}");
        });
    }

    #[test]
    fn v1_magic_rejected_with_migration_hint() {
        let mut file = valid_two_object_file().0;
        file[7] = 1; // v1 magic
        with_artifact_file("v1-magic", &file, |path| {
            let err = Reader::open(path).unwrap_err().to_string();
            assert!(err.contains("v1"), "{err}");
            assert!(err.contains("migrate"), "{err}");
        });
    }

    #[test]
    fn short_file_rejected() {
        with_artifact_file("short", &MAGIC, |path| {
            let err = Reader::open(path).unwrap_err().to_string();
            assert!(err.contains("shorter than the v2 prefix"), "{err}");
        });
    }

    #[test]
    fn json_range_beyond_file_rejected() {
        // Claim a 10MB JSON directory but only provide a tiny file.
        let directory = dir_with(
            r#"{"name":"t/x","kind":"tensor","shape":[4,8],"format":"BF16",
                 "layout":"contiguous-le-v1","offset":0,"bytes":64}"#,
        );
        let payload = vec![0xA5u8; 64];
        let mut file = build_artifact(&directory, &payload);
        // Corrupt json_bytes (bytes 8..16) to a huge value.
        file[8..16].copy_from_slice(&10u64.pow(6).to_le_bytes());
        with_artifact_file("json-beyond", &file, |path| {
            let err = Reader::open(path).unwrap_err().to_string();
            assert!(err.contains("extends beyond the file"), "{err}");
        });
    }

    #[test]
    fn missing_member_rejected() {
        let directory =
            dir_with(r#"{"name":"t/x","kind":"tensor","shape":[4,8],"format":"BF16",
                           "layout":"contiguous-le-v1","offset":0}"#); // missing "bytes"
        let file = build_artifact(&directory, &[0u8; 64]);
        with_artifact_file("missing-member", &file, |path| {
            let err = Reader::open(path).unwrap_err().to_string();
            assert!(err.contains("missing or extra members"), "{err}");
        });
    }

    #[test]
    fn extra_member_rejected() {
        let directory =
            dir_with(r#"{"name":"t/x","kind":"tensor","shape":[4,8],"format":"BF16",
                           "layout":"contiguous-le-v1","offset":0,"bytes":64,"extra":1}"#);
        let file = build_artifact(&directory, &[0u8; 64]);
        with_artifact_file("extra-member", &file, |path| {
            let err = Reader::open(path).unwrap_err().to_string();
            assert!(err.contains("missing or extra members"), "{err}");
        });
    }

    #[test]
    fn duplicate_name_rejected() {
        let directory =
            dir_with(r#"{"name":"t/x","kind":"tensor","shape":[4,8],"format":"BF16",
                           "layout":"contiguous-le-v1","offset":0,"bytes":64},
                        {"name":"t/x","kind":"resource","encoding":"raw-bytes-v1",
                          "offset":256,"bytes":4}"#);
        let mut payload = vec![0u8; 64];
        payload.resize(260, 0);
        let file = build_artifact(&directory, &payload);
        with_artifact_file("dup-name", &file, |path| {
            let err = Reader::open(path).unwrap_err().to_string();
            assert!(err.contains("duplicate object name"), "{err}");
        });
    }

    #[test]
    fn out_of_order_rejected() {
        // Second object starts before the first object's end (cursor=64).
        let directory =
            dir_with(r#"{"name":"t/x","kind":"tensor","shape":[4,8],"format":"BF16",
                           "layout":"contiguous-le-v1","offset":0,"bytes":64},
                        {"name":"frontend/r","kind":"resource","encoding":"raw-bytes-v1",
                          "offset":32,"bytes":4}"#);
        let payload = vec![0u8; 64];
        let file = build_artifact(&directory, &payload);
        with_artifact_file("out-of-order", &file, |path| {
            let err = Reader::open(path).unwrap_err().to_string();
            assert!(err.contains("overlaps or is out of order"), "{err}");
        });
    }

    #[test]
    fn misaligned_tensor_rejected() {
        // Tensor at offset 256 (not 256-aligned? 256 is aligned; use 128).
        // Use offset 128 for the second object: 128 % 256 != 0.
        let directory =
            dir_with(r#"{"name":"t/x","kind":"tensor","shape":[4,8],"format":"BF16",
                           "layout":"contiguous-le-v1","offset":0,"bytes":64},
                        {"name":"t/y","kind":"tensor","shape":[4,8],"format":"BF16",
                          "layout":"contiguous-le-v1","offset":128,"bytes":64}"#);
        let payload = vec![0u8; 256];
        let file = build_artifact(&directory, &payload);
        with_artifact_file("misaligned", &file, |path| {
            let err = Reader::open(path).unwrap_err().to_string();
            assert!(err.contains("aligned"), "{err}");
        });
    }

    #[test]
    fn encoded_size_mismatch_rejected() {
        // bytes=64 but [4,8] BF16 contiguous encodes to 64 — use a wrong
        // value: bytes=60.
        let directory =
            dir_with(r#"{"name":"t/x","kind":"tensor","shape":[4,8],"format":"BF16",
                           "layout":"contiguous-le-v1","offset":0,"bytes":60}"#);
        let payload = vec![0u8; 64];
        let file = build_artifact(&directory, &payload);
        with_artifact_file("size-mismatch", &file, |path| {
            let err = Reader::open(path).unwrap_err().to_string();
            assert!(err.contains("stores 60 bytes; layout requires 64"), "{err}");
        });
    }

    #[test]
    fn unknown_format_rejected() {
        let directory =
            dir_with(r#"{"name":"t/x","kind":"tensor","shape":[4,8],"format":"FROB",
                           "layout":"contiguous-le-v1","offset":0,"bytes":64}"#);
        let payload = vec![0u8; 64];
        let file = build_artifact(&directory, &payload);
        with_artifact_file("unknown-format", &file, |path| {
            let err = Reader::open(path).unwrap_err().to_string();
            assert!(err.contains("unknown tensor format"), "{err}");
        });
    }
}