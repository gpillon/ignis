//! Test fixture: write a small synthetic `.ninfer` v2 container to a
//! tempdir (Rust port of the reference `tests/artifact_fixture.h`).
//!
//! The fixture hand-crafts the v2 framing (16-byte binary prefix, closed
//! JSON directory, payload aligned to 4096) so integration tests can build
//! a container with arbitrary object layouts without a real artifact.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Value, json};

/// The v2 container magic (mirrors the reader's `MAGIC`).
pub const MAGIC_V2: [u8; 8] = *b"NINFER\x00\x02";

/// The 16-byte binary prefix: 8-byte magic + little-endian JSON length.
const PREFIX_BYTES: u64 = 16;

/// The payload section starts on a 4096 boundary.
const PAYLOAD_ALIGNMENT: u64 = 4096;

/// The fixture's identity (non-empty strings, as the closed schema
/// requires).
pub const FIXTURE_IDENTITY: (&str, &str) = ("fixture-model", "fixture-weights");

/// One object in the fixture directory (tensor or resource).
///
/// Offsets are payload-relative and must be ascending with the alignment
/// the reader enforces (256 B for tensors, 1 B for resources).
pub enum FixtureObject {
    Tensor {
        name: &'static str,
        shape: Vec<u64>,
        format: &'static str,
        layout: &'static str,
        offset: u64,
        bytes: u64,
    },
    Resource {
        name: &'static str,
        encoding: &'static str,
        offset: u64,
        bytes: u64,
    },
}

impl FixtureObject {
    /// Payload-relative offset (shared by both variants).
    pub fn offset(&self) -> u64 {
        match self {
            Self::Tensor { offset, .. } => *offset,
            Self::Resource { offset, .. } => *offset,
        }
    }

    /// Payload-relative length.
    pub fn bytes(&self) -> u64 {
        match self {
            Self::Tensor { bytes, .. } => *bytes,
            Self::Resource { bytes, .. } => *bytes,
        }
    }
}

/// Build the closed JSON directory from the object list (identity +
/// objects with exactly the members the v2 reader accepts).
pub fn directory(objects: &[FixtureObject]) -> String {
    let object_values: Vec<Value> = objects
        .iter()
        .map(|o| match o {
            FixtureObject::Tensor {
                name,
                shape,
                format,
                layout,
                offset,
                bytes,
            } => {
                json!({
                    "name": name,
                    "kind": "tensor",
                    "shape": shape,
                    "format": format,
                    "layout": layout,
                    "offset": offset,
                    "bytes": bytes,
                })
            }
            FixtureObject::Resource {
                name,
                encoding,
                offset,
                bytes,
            } => {
                json!({
                    "name": name,
                    "kind": "resource",
                    "encoding": encoding,
                    "offset": offset,
                    "bytes": bytes,
                })
            }
        })
        .collect();
    json!({
        "identity": {
            "model_id": FIXTURE_IDENTITY.0,
            "weights_id": FIXTURE_IDENTITY.1,
        },
        "objects": object_values,
    })
    .to_string()
}

/// Assemble a complete v2 artifact file (prefix + JSON + zero padding +
/// `payload`).
///
/// `payload` must cover every object's `[offset, offset + bytes)` range.
pub fn build_file(objects: &[FixtureObject], payload: &[u8]) -> Vec<u8> {
    let json = directory(objects);
    let json_len = json.len() as u64;
    let metadata_end = PREFIX_BYTES + json_len;
    let payload_start = metadata_end.div_ceil(PAYLOAD_ALIGNMENT) * PAYLOAD_ALIGNMENT;
    let mut file = Vec::with_capacity(payload_start as usize + payload.len());
    file.extend_from_slice(&MAGIC_V2);
    file.extend_from_slice(&json_len.to_le_bytes());
    file.extend_from_slice(json.as_bytes());
    file.extend(std::iter::repeat_n(0u8, (payload_start - metadata_end) as usize));
    file.extend_from_slice(payload);
    file
}

static SEQ: AtomicU64 = AtomicU64::new(0);

/// A temporary fixture artifact (removed on drop).
pub struct TempArtifact {
    /// The fixture file path.
    pub path: PathBuf,
}

impl TempArtifact {
    /// Write the fixture to a unique tempdir path.
    pub fn write(objects: &[FixtureObject], payload: &[u8], tag: &str) -> std::io::Result<Self> {
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "ignis-fixture-{tag}-{}-{seq}.ninfer",
            std::process::id()
        ));
        std::fs::write(&path, build_file(objects, payload))?;
        Ok(Self { path })
    }
}

impl Drop for TempArtifact {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Write a fixture artifact to a unique tempdir path (a convenience wrapper
/// over [`TempArtifact::write`] for the test call sites).
pub fn write_fixture(
    objects: &[FixtureObject],
    payload: &[u8],
    tag: &str,
) -> std::io::Result<TempArtifact> {
    TempArtifact::write(objects, payload, tag)
}

// ---------------------------------------------------------------------------
// The shared "all layouts" table: all four tensor layouts + a raw resource,
// tiny sizes (used by the unit and integration tests).
// ---------------------------------------------------------------------------

/// All four tensor layouts + a raw resource. Payload offsets are 256-
/// aligned and ascending (the reader's invariants); each object's region is
/// filled with its index + 1 (see [`all_layout_payload`]).
pub fn all_layout_objects() -> Vec<FixtureObject> {
    vec![
        FixtureObject::Resource {
            name: "frontend/tokenizer.json",
            encoding: "raw-bytes-v1",
            offset: 0,
            bytes: 4096,
        },
        FixtureObject::Tensor {
            name: "w/bf16",
            shape: vec![4, 8],
            format: "BF16",
            layout: "contiguous-le-v1",
            offset: 4096,
            bytes: 64,
        },
        FixtureObject::Tensor {
            name: "w/nvfp4",
            shape: vec![128, 64],
            format: "NVFP4",
            layout: "blockscale-k16-m128x4-v1",
            offset: 4352,
            bytes: 4612,
        },
        FixtureObject::Tensor {
            name: "w/q4",
            shape: vec![128, 128],
            format: "Q4G64_F16S",
            layout: "row-split-k128-v1",
            offset: 9216,
            bytes: 8704,
        },
        FixtureObject::Tensor {
            name: "w/fp8",
            shape: vec![128, 64],
            format: "FP8_E4M3FN_ROW_BF16S",
            layout: "row-scale-v1",
            offset: 17920,
            bytes: 8448,
        },
    ]
}

/// The payload for [`all_layout_objects`] (each object's region filled
/// with its index + 1; 26368 bytes total).
pub fn all_layout_payload() -> Vec<u8> {
    let mut p = vec![0u8; 26368];
    fill(&mut p, 0, 4096, 0x01);
    fill(&mut p, 4096, 64, 0x02);
    fill(&mut p, 4352, 4612, 0x03);
    fill(&mut p, 9216, 8704, 0x04);
    fill(&mut p, 17920, 8448, 0x05);
    p
}

fn fill(p: &mut [u8], offset: usize, len: usize, byte: u8) {
    p[offset..offset + len].fill(byte);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Reader;

    #[test]
    fn fixture_round_trips_through_reader() {
        let objects = all_layout_objects();
        let payload = all_layout_payload();
        let artifact = TempArtifact::write(&objects, &payload, "round-trip").expect("fixture");
        let reader = Reader::open(&artifact.path).expect("open fixture artifact");

        assert_eq!(reader.identity().model_id, FIXTURE_IDENTITY.0);
        assert_eq!(reader.objects().len(), 5);
        assert!(reader.payload_offset() % PAYLOAD_ALIGNMENT == 0);
        // The payload spans are intact (mapping-backed).
        for object in reader.objects() {
            let span = reader.payload_at(object).expect("span");
            assert_eq!(span.data.len(), object.bytes() as usize);
        }
        // The reader sees the fixture objects at the declared offsets.
        let nvfp4 = reader.find("w/nvfp4").expect("nvfp4 present");
        assert_eq!(nvfp4.offset(), 4352);
        assert_eq!(nvfp4.bytes(), 4612);
    }
}