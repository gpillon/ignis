//! The streaming `.ninfer` v2 writer the packer and the Flash-Next fixture
//! write through (spec flash-next/01: reader and writer share one
//! implementation). `crate::fixture::build_file` still assembles the small
//! in-memory containers of the older reader tests.
//!
//! The v2 framing puts the JSON directory *before* the payload, and the
//! directory of an artifact assembled layer by layer is not known until the
//! last layer is in. The writer therefore reserves the header region up
//! front (`header_bytes`, a multiple of 4096: the payload starts exactly
//! there), appends payloads behind it, and at [`ContainerWriter::finish`]
//! writes the prefix and the directory into the reservation, padded with
//! spaces (JSON allows trailing whitespace, and `json_bytes` covers the
//! whole reservation).
//!
//! Every object is checked against the reader's own rules before a byte of
//! it is written: a unique nonempty name, a tensor's stored size equal to its
//! layout's encoded size, the layout's alignment. An object can be written
//! in several pieces ([`ContainerWriter::begin_tensor`], then
//! [`ContainerWriter::write`] per piece, then [`ContainerWriter::end_object`])
//! so a 29 GB table can come from shards. The writer's [`WriterState`]
//! (identity, reservation, payload cursor, directory so far, the open
//! object's progress) is what a resumable caller persists:
//! [`ContainerWriter::resume`] cuts the file back to the state's cursor,
//! dropping whatever was written after the state was taken.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use serde_json::{Value, json};

use crate::{
    align_up, checked_add, fail, parse_object, require_members, require_string,
    resource_alignment, tensor_alignment, tensor_encoded_size, ArtifactIdentity, NumericFormat,
    Object, ResourceDescriptor, ResourceEncoding, Result, StorageLayout, TensorDescriptor, MAGIC,
    PAYLOAD_ALIGNMENT, PREFIX_BYTES,
};

/// Copy buffer for appended payloads (the n-gram table is ~29 GB: stream it).
const COPY_BYTES: usize = 8 << 20;

/// An object whose payload is being written: its final descriptor and the
/// bytes of it written so far.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenObject {
    pub object: Object,
    pub written: u64,
}

/// What a resumable writer persists: everything needed to reopen the file
/// and keep appending.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriterState {
    pub identity: ArtifactIdentity,
    /// The reserved header region; the payload starts at this file offset.
    pub header_bytes: u64,
    /// Payload-relative end of the bytes written so far.
    pub cursor: u64,
    /// The finished objects, in physical order.
    pub objects: Vec<Object>,
    /// The object being written, if any.
    pub open: Option<OpenObject>,
}

impl WriterState {
    /// The state as JSON: the closed directory plus the writer's fields.
    pub fn to_json(&self) -> String {
        self.to_value().to_string()
    }

    /// The state as a JSON value (for a caller that embeds it in its own
    /// record).
    pub fn to_value(&self) -> Value {
        let open = match &self.open {
            Some(open) => json!({"object": object_value(&open.object), "written": open.written}),
            None => Value::Null,
        };
        json!({
            "header_bytes": self.header_bytes,
            "cursor": self.cursor,
            "directory": directory_value(&self.identity, &self.objects),
            "open": open,
        })
    }

    /// Parse a state written by [`WriterState::to_json`]. Every object is
    /// re-validated with the reader's own entry parser.
    pub fn from_json(text: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(text)
            .map_err(|e| fail(format!("invalid writer state JSON: {e}")))?;
        Self::from_value(&value)
    }

    /// Parse a state value written by [`WriterState::to_value`].
    pub fn from_value(value: &Value) -> Result<Self> {
        require_members(value, &["header_bytes", "cursor", "directory", "open"], "writer state")?;
        let header_bytes = value["header_bytes"]
            .as_u64()
            .ok_or_else(|| fail("writer state header_bytes must be an integer"))?;
        let cursor = value["cursor"]
            .as_u64()
            .ok_or_else(|| fail("writer state cursor must be an integer"))?;
        let (identity, objects) = parse_directory_value(&value["directory"])?;
        let open = match &value["open"] {
            Value::Null => None,
            raw => {
                require_members(raw, &["object", "written"], "writer state open object")?;
                Some(OpenObject {
                    object: parse_object(&raw["object"])?,
                    written: raw["written"]
                        .as_u64()
                        .ok_or_else(|| fail("writer state open.written must be an integer"))?,
                })
            }
        };
        Ok(Self {
            identity,
            header_bytes,
            cursor,
            objects,
            open,
        })
    }

    /// The exact length of the directory JSON the finished objects produce.
    pub fn directory_bytes(&self) -> u64 {
        directory_json(&self.identity, &self.objects).len() as u64
    }

    /// Where the bytes written so far end: the open object's progress, or
    /// the end of the last finished object.
    fn written_end(&self) -> u64 {
        match &self.open {
            Some(open) => open.object.offset() + open.written,
            None => self.objects.last().map_or(0, |o| o.offset() + o.bytes()),
        }
    }
}

/// A container being written (see the module doc).
pub struct ContainerWriter {
    file: File,
    state: WriterState,
    names: HashSet<String>,
}

impl ContainerWriter {
    /// Create (truncating) `path` with `header_bytes` reserved for the prefix
    /// and the directory.
    pub fn create(path: &Path, identity: ArtifactIdentity, header_bytes: u64) -> Result<Self> {
        check_header_bytes(header_bytes)?;
        if identity.model_id.is_empty() || identity.weights_id.is_empty() {
            return Err(fail("artifact identity strings must be nonempty"));
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(|e| fail(format!("create {}: {e}", path.display())))?;
        let mut writer = Self {
            file,
            state: WriterState {
                identity,
                header_bytes,
                cursor: 0,
                objects: Vec::new(),
                open: None,
            },
            names: HashSet::new(),
        };
        writer.set_len(header_bytes)?;
        Ok(writer)
    }

    /// Reopen `path` at a persisted state: the file is cut back to the
    /// state's cursor (anything written after the state was taken is
    /// dropped and will be written again).
    pub fn resume(path: &Path, state: WriterState) -> Result<Self> {
        check_header_bytes(state.header_bytes)?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| fail(format!("open {}: {e}", path.display())))?;
        if state.written_end() != state.cursor {
            return Err(fail(format!(
                "writer state cursor {} is not where its objects end ({})",
                state.cursor,
                state.written_end()
            )));
        }
        let end = checked_add(state.header_bytes, state.cursor, "writer state end")?;
        let len = file
            .metadata()
            .map_err(|e| fail(format!("stat {}: {e}", path.display())))?
            .len();
        if len < end {
            return Err(fail(format!(
                "{} is {len} bytes, shorter than its saved state ({end}): the state is not this file's",
                path.display()
            )));
        }
        let mut names = HashSet::with_capacity(state.objects.len() + 1);
        for name in state
            .objects
            .iter()
            .chain(state.open.as_ref().map(|o| &o.object))
            .map(Object::name)
        {
            if !names.insert(name.to_owned()) {
                return Err(fail(format!("duplicate object name in writer state: {name}")));
            }
        }
        let mut writer = Self { file, state, names };
        writer.set_len(end)?;
        writer
            .file
            .seek(SeekFrom::Start(end))
            .map_err(|e| fail(format!("seek container: {e}")))?;
        Ok(writer)
    }

    /// The state to persist (durable after [`ContainerWriter::sync`]).
    pub fn state(&self) -> &WriterState {
        &self.state
    }

    /// Append a tensor of `shape` whose stored bytes are read from `source`
    /// (exactly the layout's encoded size: a short or a long source fails).
    pub fn append_tensor(
        &mut self,
        name: &str,
        format: NumericFormat,
        layout: StorageLayout,
        shape: &[u64],
        source: &mut dyn Read,
    ) -> Result<()> {
        self.begin_tensor(name, format, layout, shape)?;
        self.write(source)?;
        self.end_object()
    }

    /// Append a resource of `bytes` bytes read from `source`.
    pub fn append_resource(
        &mut self,
        name: &str,
        encoding: ResourceEncoding,
        bytes: u64,
        source: &mut dyn Read,
    ) -> Result<()> {
        self.begin_resource(name, encoding, bytes)?;
        self.write(source)?;
        self.end_object()
    }

    /// Open a tensor whose payload the following [`ContainerWriter::write`]
    /// calls supply.
    pub fn begin_tensor(
        &mut self,
        name: &str,
        format: NumericFormat,
        layout: StorageLayout,
        shape: &[u64],
    ) -> Result<()> {
        let bytes = tensor_encoded_size(layout, format, shape)
            .map_err(|e| fail(format!("tensor {name}: {e}")))?;
        let offset = self.claim(name, tensor_alignment(layout), bytes)?;
        self.open(
            Object::Tensor(TensorDescriptor {
                name: name.to_owned(),
                shape: shape.to_vec(),
                format,
                layout,
                offset,
                bytes,
            }),
        )
    }

    /// Open a resource of `bytes` bytes.
    pub fn begin_resource(&mut self, name: &str, encoding: ResourceEncoding, bytes: u64) -> Result<()> {
        if bytes == 0 {
            return Err(fail(format!("resource {name} is empty")));
        }
        let offset = self.claim(name, resource_alignment(encoding), bytes)?;
        self.open(Object::Resource(ResourceDescriptor {
            name: name.to_owned(),
            encoding,
            offset,
            bytes,
        }))
    }

    /// Append everything `source` holds to the open object. Returns the
    /// bytes written; fails when the object would grow past its stored size.
    pub fn write(&mut self, source: &mut dyn Read) -> Result<u64> {
        let open = self
            .state
            .open
            .as_ref()
            .ok_or_else(|| fail("no object is open for writing"))?;
        let name = open.object.name().to_owned();
        let bytes = open.object.bytes();
        let mut written = open.written;
        let mut buffer = vec![0u8; COPY_BYTES];
        let mut total = 0u64;
        loop {
            // One byte past the object's end is read on purpose: it is how a
            // long source is told apart from an exact one.
            let want = buffer.len().min((bytes - written) as usize + 1);
            let got = match source.read(&mut buffer[..want]) {
                Ok(got) => got,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(fail(format!("read payload of {name}: {e}"))),
            };
            if got == 0 {
                break;
            }
            if written + got as u64 > bytes {
                // An object that cannot be exact is dropped, as if never
                // begun: a later end_object cannot accept it.
                self.abandon_open()?;
                return Err(fail(format!(
                    "payload of {name} is longer than the {bytes} bytes its layout stores"
                )));
            }
            self.file
                .write_all(&buffer[..got])
                .map_err(|e| fail(format!("write payload of {name}: {e}")))?;
            written += got as u64;
            total += got as u64;
            if let Some(open) = self.state.open.as_mut() {
                open.written = written;
            }
            self.state.cursor += got as u64;
        }
        Ok(total)
    }

    /// Close the open object: it must hold exactly its stored size (a short
    /// one is dropped, as if never begun).
    pub fn end_object(&mut self) -> Result<()> {
        let open = self
            .state
            .open
            .as_ref()
            .ok_or_else(|| fail("no object is open"))?;
        if open.written != open.object.bytes() {
            let name = open.object.name().to_owned();
            let (written, bytes) = (open.written, open.object.bytes());
            self.abandon_open()?;
            return Err(fail(format!(
                "payload of {name} ended after {written} bytes; its layout stores {bytes}"
            )));
        }
        let open = self.state.open.take().expect("checked above");
        self.state.objects.push(open.object);
        Ok(())
    }

    /// Drop the open object and put the cursor back where the finished
    /// objects end, keeping the writer usable.
    fn abandon_open(&mut self) -> Result<()> {
        if let Some(open) = self.state.open.take() {
            self.names.remove(open.object.name());
        }
        self.state.cursor = self.state.written_end();
        self.file
            .seek(SeekFrom::Start(self.state.header_bytes + self.state.cursor))
            .map(|_| ())
            .map_err(|e| fail(format!("seek container: {e}")))
    }

    /// Flush what was written to disk (call before persisting the state).
    pub fn sync(&mut self) -> Result<()> {
        self.file
            .sync_all()
            .map_err(|e| fail(format!("sync container: {e}")))
    }

    /// Write the prefix and the directory into the reserved header and sync.
    /// Returns the file size. Fails, writing nothing, when an object is still
    /// open or the directory does not fit the reservation.
    pub fn finish(mut self) -> Result<u64> {
        if let Some(open) = &self.state.open {
            return Err(fail(format!("object {} is still open", open.object.name())));
        }
        if self.state.objects.is_empty() {
            return Err(fail("a container needs at least one object"));
        }
        let directory = directory_json(&self.state.identity, &self.state.objects);
        let json_bytes = self.state.header_bytes - PREFIX_BYTES;
        if directory.len() as u64 > json_bytes {
            return Err(fail(format!(
                "the directory is {} bytes, the header reserves {json_bytes}",
                directory.len()
            )));
        }
        let mut header = Vec::with_capacity(self.state.header_bytes as usize);
        header.extend_from_slice(&MAGIC);
        header.extend_from_slice(&json_bytes.to_le_bytes());
        header.extend_from_slice(directory.as_bytes());
        header.resize(self.state.header_bytes as usize, b' ');
        self.file
            .seek(SeekFrom::Start(0))
            .and_then(|_| self.file.write_all(&header))
            .map_err(|e| fail(format!("write container header: {e}")))?;
        // A refused write may have left bytes past the last object.
        let file_bytes = checked_add(self.state.header_bytes, self.state.cursor, "container size")?;
        self.set_len(file_bytes)?;
        self.sync()?;
        Ok(file_bytes)
    }

    /// Validate a new object's name and place it after the cursor.
    fn claim(&self, name: &str, alignment: u64, bytes: u64) -> Result<u64> {
        if let Some(open) = &self.state.open {
            return Err(fail(format!("object {} is still open", open.object.name())));
        }
        if name.is_empty() {
            return Err(fail("object names must be nonempty"));
        }
        if self.names.contains(name) {
            return Err(fail(format!("duplicate object name: {name}")));
        }
        let offset = align_up(self.state.cursor, alignment, "object offset")?;
        checked_add(offset, bytes, "object payload range")?;
        Ok(offset)
    }

    /// Write the alignment padding (zeros: the file may hold stale bytes
    /// there from a write a resume dropped) and make `object` the open one.
    fn open(&mut self, object: Object) -> Result<()> {
        let padding_start = checked_add(self.state.header_bytes, self.state.cursor, "padding offset")?;
        let padding = object.offset() - self.state.cursor;
        self.file
            .seek(SeekFrom::Start(padding_start))
            .and_then(|_| self.file.write_all(&vec![0u8; padding as usize]))
            .map_err(|e| fail(format!("write padding before {}: {e}", object.name())))?;
        self.names.insert(object.name().to_owned());
        self.state.cursor = object.offset();
        self.state.open = Some(OpenObject { object, written: 0 });
        Ok(())
    }

    fn set_len(&mut self, len: u64) -> Result<()> {
        self.file
            .set_len(len)
            .map_err(|e| fail(format!("resize container to {len}: {e}")))
    }
}

/// The header reservation must hold the prefix and start the payload on a
/// 4096 boundary.
fn check_header_bytes(header_bytes: u64) -> Result<()> {
    if header_bytes <= PREFIX_BYTES || !header_bytes.is_multiple_of(PAYLOAD_ALIGNMENT) {
        return Err(fail(format!(
            "header reservation {header_bytes} must be a positive multiple of {PAYLOAD_ALIGNMENT}"
        )));
    }
    Ok(())
}

/// The bytes `object` adds to the directory JSON (its entry and a comma),
/// for a caller that bounds the directory before writing: with `u64::MAX`
/// offset and size it is an upper bound for any placement.
pub fn entry_json_bytes(object: &Object) -> u64 {
    object_value(object).to_string().len() as u64 + 1
}

/// The closed directory JSON (`identity` + `objects`, exactly the members
/// the reader accepts).
pub fn directory_json(identity: &ArtifactIdentity, objects: &[Object]) -> String {
    directory_value(identity, objects).to_string()
}

fn object_value(object: &Object) -> Value {
    match object {
        Object::Tensor(t) => json!({
            "name": t.name,
            "kind": "tensor",
            "shape": t.shape,
            "format": t.format.name(),
            "layout": t.layout.name(),
            "offset": t.offset,
            "bytes": t.bytes,
        }),
        Object::Resource(r) => json!({
            "name": r.name,
            "kind": "resource",
            "encoding": r.encoding.name(),
            "offset": r.offset,
            "bytes": r.bytes,
        }),
    }
}

fn directory_value(identity: &ArtifactIdentity, objects: &[Object]) -> Value {
    json!({
        "identity": {
            "model_id": identity.model_id,
            "weights_id": identity.weights_id,
        },
        "objects": objects.iter().map(object_value).collect::<Vec<_>>(),
    })
}

fn parse_directory_value(value: &Value) -> Result<(ArtifactIdentity, Vec<Object>)> {
    require_members(value, &["identity", "objects"], "directory root")?;
    let raw_identity = &value["identity"];
    require_members(raw_identity, &["model_id", "weights_id"], "artifact identity")?;
    let identity = ArtifactIdentity {
        model_id: require_string(&raw_identity["model_id"], "model_id")?.to_owned(),
        weights_id: require_string(&raw_identity["weights_id"], "weights_id")?.to_owned(),
    };
    let objects = value["objects"]
        .as_array()
        .ok_or_else(|| fail("objects must be an array"))?
        .iter()
        .map(parse_object)
        .collect::<Result<Vec<_>>>()?;
    Ok((identity, objects))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Reader;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static SEQ: AtomicU64 = AtomicU64::new(0);

    /// A temp path removed on drop.
    struct TempPath(PathBuf);

    impl TempPath {
        fn new(tag: &str) -> Self {
            let seq = SEQ.fetch_add(1, Ordering::Relaxed);
            Self(std::env::temp_dir().join(format!(
                "ignis-writer-{tag}-{}-{seq}.ninfer",
                std::process::id()
            )))
        }
    }

    impl Drop for TempPath {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn identity() -> ArtifactIdentity {
        ArtifactIdentity {
            model_id: "writer-model".into(),
            weights_id: "writer-weights".into(),
        }
    }

    #[test]
    fn a_written_container_reads_back_object_for_object() {
        let path = TempPath::new("round-trip");
        let mut writer = ContainerWriter::create(&path.0, identity(), 4096).unwrap();
        // A 3-byte resource first, so the tensor after it needs alignment
        // padding up to 256.
        writer
            .append_resource("frontend/r", ResourceEncoding::RawBytesV1, 3, &mut &b"abc"[..])
            .unwrap();
        let tensor = vec![0x5Au8; 64];
        writer
            .append_tensor(
                "t/x",
                NumericFormat::Bf16,
                StorageLayout::ContiguousLeV1,
                &[4, 8],
                &mut tensor.as_slice(),
            )
            .unwrap();
        let file_bytes = writer.finish().unwrap();

        let reader = Reader::open(&path.0).expect("the reader opens what the writer wrote");
        assert_eq!(reader.identity(), &identity());
        assert_eq!(reader.file_bytes(), file_bytes);
        assert_eq!(reader.payload_offset(), 4096, "the payload starts at the reservation");
        assert_eq!(reader.objects().len(), 2);
        assert_eq!(reader.payload("frontend/r").unwrap().data, b"abc");
        assert_eq!(reader.find("t/x").unwrap().offset(), 256);
        assert_eq!(reader.payload("t/x").unwrap().data, tensor.as_slice());
        // 4096 header + 256 (resource + padding) + 64 tensor bytes.
        assert_eq!(file_bytes, 4096 + 256 + 64);
    }

    #[test]
    fn an_object_written_in_pieces_reads_back_whole() {
        let path = TempPath::new("pieces");
        let mut writer = ContainerWriter::create(&path.0, identity(), 4096).unwrap();
        writer
            .begin_tensor("t/x", NumericFormat::Bf16, StorageLayout::ContiguousLeV1, &[4, 8])
            .unwrap();
        assert_eq!(writer.write(&mut &[1u8; 40][..]).unwrap(), 40);
        assert_eq!(writer.write(&mut &[2u8; 24][..]).unwrap(), 24);
        writer.end_object().unwrap();
        writer.finish().unwrap();

        let reader = Reader::open(&path.0).unwrap();
        let data = reader.payload("t/x").unwrap().data;
        assert_eq!(&data[..40], &[1u8; 40]);
        assert_eq!(&data[40..], &[2u8; 24]);
    }

    #[test]
    fn a_tensor_whose_source_is_not_its_encoded_size_is_refused() {
        let path = TempPath::new("size");
        let mut writer = ContainerWriter::create(&path.0, identity(), 4096).unwrap();
        let short = vec![0u8; 63];
        let err = writer
            .append_tensor("t/x", NumericFormat::Bf16, StorageLayout::ContiguousLeV1, &[4, 8], &mut short.as_slice())
            .unwrap_err()
            .to_string();
        assert!(err.contains("ended after 63 bytes; its layout stores 64"), "{err}");
        let long = vec![0u8; 65];
        let err = writer
            .append_tensor("t/y", NumericFormat::Bf16, StorageLayout::ContiguousLeV1, &[4, 8], &mut long.as_slice())
            .unwrap_err()
            .to_string();
        assert!(err.contains("longer than the 64 bytes"), "{err}");
        // Either way the object is dropped: nothing is left open, both names
        // are free again and nothing of them stays in the file.
        let err = writer.end_object().unwrap_err().to_string();
        assert!(err.contains("no object is open"), "{err}");
        writer
            .append_resource("t/y", ResourceEncoding::RawBytesV1, 1, &mut &b"y"[..])
            .unwrap();
        writer
            .append_resource("t/x", ResourceEncoding::RawBytesV1, 2, &mut &b"ok"[..])
            .unwrap();
        let file_bytes = writer.finish().unwrap();
        let reader = Reader::open(&path.0).unwrap();
        assert_eq!(reader.objects().len(), 2);
        assert_eq!(reader.payload("t/y").unwrap().data, b"y");
        assert_eq!(reader.payload("t/x").unwrap().data, b"ok");
        assert_eq!(file_bytes, 4096 + 3);
        assert_eq!(reader.file_bytes(), file_bytes);
    }

    #[test]
    fn a_source_that_hands_over_its_bytes_in_two_reads_cannot_overfill_an_object() {
        // Exactly the stored size in the first read, one byte more in the
        // second: the second read must still be refused.
        struct TwoReads(u8);
        impl Read for TwoReads {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                self.0 += 1;
                match self.0 {
                    1 => {
                        buf[..64].fill(7);
                        Ok(64)
                    }
                    2 => Err(std::io::Error::from(std::io::ErrorKind::Interrupted)),
                    3 => {
                        buf[0] = 8;
                        Ok(1)
                    }
                    _ => Ok(0),
                }
            }
        }
        let path = TempPath::new("two-reads");
        let mut writer = ContainerWriter::create(&path.0, identity(), 4096).unwrap();
        writer
            .begin_tensor("t/x", NumericFormat::Bf16, StorageLayout::ContiguousLeV1, &[4, 8])
            .unwrap();
        let err = writer.write(&mut TwoReads(0)).unwrap_err().to_string();
        assert!(err.contains("longer than the 64 bytes"), "an interrupted read is retried: {err}");
        assert!(writer.end_object().is_err(), "the over-long object is gone");
    }

    #[test]
    fn a_state_with_a_bad_header_reservation_is_not_resumed() {
        let path = TempPath::new("bad-header");
        let mut writer = ContainerWriter::create(&path.0, identity(), 4096).unwrap();
        writer
            .append_resource("r", ResourceEncoding::RawBytesV1, 1, &mut &b"r"[..])
            .unwrap();
        let mut state = writer.state().clone();
        drop(writer);
        state.header_bytes = 4000;
        let err = ContainerWriter::resume(&path.0, state).err().expect("refused").to_string();
        assert!(err.contains("must be a positive multiple of 4096"), "{err}");
    }

    #[test]
    fn a_duplicate_name_is_refused() {
        let path = TempPath::new("dup");
        let mut writer = ContainerWriter::create(&path.0, identity(), 4096).unwrap();
        writer
            .append_resource("r", ResourceEncoding::RawBytesV1, 1, &mut &b"a"[..])
            .unwrap();
        let err = writer
            .append_resource("r", ResourceEncoding::RawBytesV1, 1, &mut &b"b"[..])
            .unwrap_err()
            .to_string();
        assert!(err.contains("duplicate object name: r"), "{err}");
    }

    #[test]
    fn finish_refuses_a_directory_larger_than_the_reservation() {
        let path = TempPath::new("overflow");
        let mut writer = ContainerWriter::create(&path.0, identity(), 4096).unwrap();
        // ~100 bytes of JSON per resource: 64 of them overflow 4080 bytes.
        for i in 0..64 {
            writer
                .append_resource(&format!("frontend/resource-{i:03}"), ResourceEncoding::RawBytesV1, 1, &mut &b"x"[..])
                .unwrap();
        }
        let err = writer.finish().unwrap_err().to_string();
        assert!(err.contains("the header reserves 4080"), "{err}");
    }

    #[test]
    fn a_resumed_writer_drops_what_was_written_after_its_state() {
        let path = TempPath::new("resume");
        let mut writer = ContainerWriter::create(&path.0, identity(), 4096).unwrap();
        writer
            .append_resource("kept", ResourceEncoding::RawBytesV1, 4, &mut &b"keep"[..])
            .unwrap();
        // Half of a two-piece object, then a state.
        writer.begin_resource("pieces", ResourceEncoding::RawBytesV1, 6).unwrap();
        writer.write(&mut &b"abc"[..]).unwrap();
        writer.sync().unwrap();
        let saved = WriterState::from_json(&writer.state().to_json()).unwrap();
        assert_eq!(&saved, writer.state(), "the state survives its JSON");
        // Written but never recorded (a crash before the next state).
        writer.write(&mut &b"XYZ"[..]).unwrap();
        writer.end_object().unwrap();
        writer
            .append_resource("lost", ResourceEncoding::RawBytesV1, 4, &mut &b"lost"[..])
            .unwrap();
        drop(writer);

        let mut writer = ContainerWriter::resume(&path.0, saved).unwrap();
        writer.write(&mut &b"def"[..]).unwrap();
        writer.end_object().unwrap();
        writer
            .append_resource("again", ResourceEncoding::RawBytesV1, 5, &mut &b"again"[..])
            .unwrap();
        writer.finish().unwrap();

        let reader = Reader::open(&path.0).unwrap();
        let names: Vec<&str> = reader.objects().iter().map(Object::name).collect();
        assert_eq!(names, ["kept", "pieces", "again"]);
        assert_eq!(reader.payload("pieces").unwrap().data, b"abcdef");
        assert_eq!(reader.payload("again").unwrap().data, b"again");
        assert_eq!(reader.file_bytes(), 4096 + 4 + 6 + 5);
    }
}
