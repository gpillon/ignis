//! Our own GGUF reader (spec speech/01 §1, Q60): the header, the metadata,
//! the tensor table and the tensor data, memory-mapped.
//!
//! The format is the file the kernels read (ADR 0068): GGUF v2
//! and v3, little-endian, every metadata value type, `general.alignment`
//! (32 when absent). Nothing here interprets the model; `model.rs` does.

use std::fmt;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

/// A ggml tensor type the executor knows by its GGUF code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorType {
    F32,
    F16,
    Q8_0,
    Q4K,
    Q6K,
    Bf16,
    /// Any other ggml type: readable in the table, refused by the executor.
    Other(u32),
}

impl TensorType {
    pub fn from_code(code: u32) -> TensorType {
        match code {
            0 => TensorType::F32,
            1 => TensorType::F16,
            8 => TensorType::Q8_0,
            12 => TensorType::Q4K,
            14 => TensorType::Q6K,
            30 => TensorType::Bf16,
            c => TensorType::Other(c),
        }
    }

    /// (elements per block, bytes per block), `None` for an unknown type.
    pub fn block(self) -> Option<(usize, usize)> {
        match self {
            TensorType::F32 => Some((1, 4)),
            TensorType::F16 | TensorType::Bf16 => Some((1, 2)),
            TensorType::Q8_0 => Some((32, 34)),
            TensorType::Q4K => Some((256, 144)),
            TensorType::Q6K => Some((256, 210)),
            TensorType::Other(_) => None,
        }
    }

    pub fn name(self) -> String {
        match self {
            TensorType::F32 => "F32".into(),
            TensorType::F16 => "F16".into(),
            TensorType::Q8_0 => "Q8_0".into(),
            TensorType::Q4K => "Q4_K".into(),
            TensorType::Q6K => "Q6_K".into(),
            TensorType::Bf16 => "BF16".into(),
            TensorType::Other(c) => format!("ggml type {c}"),
        }
    }
}

/// One metadata value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    /// The value as an unsigned integer, when it is a non-negative integer.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v as u64),
            Value::U16(v) => Some(v as u64),
            Value::U32(v) => Some(v as u64),
            Value::U64(v) => Some(v),
            Value::I8(v) if v >= 0 => Some(v as u64),
            Value::I16(v) if v >= 0 => Some(v as u64),
            Value::I32(v) if v >= 0 => Some(v as u64),
            Value::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F32(v) => Some(v as f64),
            Value::F64(v) => Some(v),
            _ => self.as_u64().map(|v| v as f64),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
}

/// One entry of the tensor table.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// ggml order: `dims[0]` is the contiguous (row) length.
    pub dims: Vec<u64>,
    pub ty: TensorType,
    /// Offset from the start of the data section.
    pub offset: u64,
    /// Bytes of tensor data.
    pub size: u64,
}

impl TensorInfo {
    pub fn elements(&self) -> u64 {
        self.dims.iter().product()
    }
}

#[derive(Debug)]
pub enum GgufError {
    Io(PathBuf, std::io::Error),
    /// The file is not a well-formed GGUF: what is wrong and where.
    Format(PathBuf, String),
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GgufError::Io(p, e) => write!(f, "{}: {e}", p.display()),
            GgufError::Format(p, m) => write!(f, "{}: not a valid GGUF: {m}", p.display()),
        }
    }
}

impl std::error::Error for GgufError {}

/// An open GGUF file: its metadata and tensor table, and the mapped bytes.
pub struct Gguf {
    path: PathBuf,
    map: Mmap,
    pub version: u32,
    pub alignment: u64,
    metadata: Vec<(String, Value)>,
    tensors: Vec<TensorInfo>,
    data_offset: usize,
}

/// A bounds-checked little-endian cursor over the mapped header.
struct Cursor<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.b.len());
        match end {
            Some(e) => {
                let s = &self.b[self.pos..e];
                self.pos = e;
                Ok(s)
            }
            None => Err(format!("truncated at byte {} (wanted {n} more)", self.pos)),
        }
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn len(&mut self, what: &str) -> Result<usize, String> {
        let n = self.u64()?;
        // No count can exceed the bytes left: refuses a corrupt length before allocating for it.
        if n > (self.b.len() - self.pos) as u64 {
            return Err(format!("{what} count {n} at byte {} exceeds the file", self.pos));
        }
        Ok(n as usize)
    }
    fn string(&mut self) -> Result<String, String> {
        let n = self.len("string length")?;
        let at = self.pos;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| format!("string at byte {at} is not UTF-8"))
    }
    fn value(&mut self, ty: u32, depth: u32) -> Result<Value, String> {
        Ok(match ty {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.u8()? as i8),
            2 => Value::U16(self.u16()?),
            3 => Value::I16(self.u16()? as i16),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => match self.u8()? {
                0 => Value::Bool(false),
                1 => Value::Bool(true),
                b => return Err(format!("bool value {b} at byte {}", self.pos - 1)),
            },
            8 => Value::String(self.string()?),
            9 => {
                if depth > 0 {
                    return Err("nested metadata array".into());
                }
                let et = self.u32()?;
                let n = self.len("array")?;
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.value(et, depth + 1)?);
                }
                Value::Array(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            t => return Err(format!("metadata value type {t} at byte {}", self.pos)),
        })
    }
}

impl Gguf {
    /// Maps `path` and parses its header and tensor table.
    pub fn open(path: &Path) -> Result<Gguf, GgufError> {
        let file = File::open(path).map_err(|e| GgufError::Io(path.into(), e))?;
        // SAFETY: the file is opened read-only and the mapping is never written. A file
        // truncated by another process while mapped is outside what the executor guards.
        let map = unsafe { Mmap::map(&file) }.map_err(|e| GgufError::Io(path.into(), e))?;
        Self::parse(path, map)
    }

    fn parse(path: &Path, map: Mmap) -> Result<Gguf, GgufError> {
        let bad = |m: String| GgufError::Format(path.into(), m);
        let mut c = Cursor { b: &map, pos: 0 };
        if c.take(4).map_err(&bad)? != b"GGUF" {
            return Err(bad("no GGUF magic".into()));
        }
        let version = c.u32().map_err(&bad)?;
        if version != 2 && version != 3 {
            return Err(bad(format!("version {version} (2 and 3 are read)")));
        }
        let n_tensors = c.len("tensor").map_err(&bad)?;
        let n_kv = c.len("metadata").map_err(&bad)?;
        let mut metadata = Vec::with_capacity(n_kv);
        for _ in 0..n_kv {
            let k = c.string().map_err(&bad)?;
            let t = c.u32().map_err(&bad)?;
            let v = c.value(t, 0).map_err(|m| bad(format!("{k}: {m}")))?;
            metadata.push((k, v));
        }
        let alignment = match metadata.iter().find(|(k, _)| k == "general.alignment") {
            Some((_, v)) => v.as_u64().filter(|a| a.is_power_of_two()).ok_or_else(|| bad(format!("general.alignment {v:?}")))?,
            None => 32,
        };
        let mut tensors = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = c.string().map_err(&bad)?;
            let nd = c.u32().map_err(&bad)?;
            if nd == 0 || nd > 4 {
                return Err(bad(format!("{name}: {nd} dimensions")));
            }
            let mut dims = Vec::with_capacity(nd as usize);
            for _ in 0..nd {
                dims.push(c.u64().map_err(&bad)?);
            }
            let ty = TensorType::from_code(c.u32().map_err(&bad)?);
            let offset = c.u64().map_err(&bad)?;
            let size = tensor_size(ty, &dims).map_err(|m| bad(format!("{name}: {m}")))?;
            if offset % alignment != 0 {
                return Err(bad(format!("{name}: offset {offset} not aligned to {alignment}")));
            }
            tensors.push(TensorInfo { name, dims, ty, offset, size });
        }
        let data_offset = (c.pos as u64).div_ceil(alignment) * alignment;
        let data_len = (map.len() as u64).saturating_sub(data_offset);
        for t in &tensors {
            if t.offset.checked_add(t.size).is_none_or(|end| end > data_len) {
                return Err(bad(format!("{}: data [{}, +{}) past the end of the file", t.name, t.offset, t.size)));
            }
        }
        Ok(Gguf { path: path.into(), map, version, alignment, metadata, tensors, data_offset: data_offset as usize })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn metadata(&self) -> &[(String, Value)] {
        &self.metadata
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// The tensor's bytes in the mapped file.
    pub fn data(&self, t: &TensorInfo) -> &[u8] {
        let start = self.data_offset + t.offset as usize;
        &self.map[start..start + t.size as usize]
    }

    /// The whole tensor data section: a tensor's `offset` indexes it.
    pub fn data_section(&self) -> &[u8] {
        &self.map[self.data_offset.min(self.map.len())..]
    }
}

fn tensor_size(ty: TensorType, dims: &[u64]) -> Result<u64, String> {
    let (be, bb) = ty.block().ok_or_else(|| format!("unknown {}", ty.name()))?;
    if dims[0] % be as u64 != 0 {
        return Err(format!("row length {} not a multiple of the {} block ({be})", dims[0], ty.name()));
    }
    let mut n = 1u64;
    for &d in dims {
        n = n.checked_mul(d).ok_or("element count overflows")?;
    }
    (n / be as u64).checked_mul(bb as u64).ok_or_else(|| "byte size overflows".into())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Writes a GGUF v3 file: metadata `kv` (key, type code, encoded value) and tensors
    /// (name, dims, type code, bytes), aligned to `align`.
    pub(crate) fn write_gguf(path: &Path, kv: &[(&str, u32, Vec<u8>)], tensors: &[(&str, Vec<u64>, u32, Vec<u8>)], align: u64) {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        b.extend_from_slice(&(kv.len() as u64).to_le_bytes());
        for (k, t, v) in kv {
            b.extend_from_slice(&enc_str(k));
            b.extend_from_slice(&t.to_le_bytes());
            b.extend_from_slice(v);
        }
        let mut off = 0u64;
        let mut offs = Vec::new();
        for (name, dims, ty, data) in tensors {
            b.extend_from_slice(&enc_str(name));
            b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                b.extend_from_slice(&d.to_le_bytes());
            }
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(&off.to_le_bytes());
            offs.push(off);
            off = (off + data.len() as u64).div_ceil(align) * align;
        }
        while b.len() as u64 % align != 0 {
            b.push(0);
        }
        let base = b.len() as u64;
        for ((_, _, _, data), o) in tensors.iter().zip(offs) {
            b.resize((base + o) as usize, 0);
            b.extend_from_slice(data);
        }
        std::fs::write(path, b).unwrap();
    }

    pub(crate) fn enc_str(s: &str) -> Vec<u8> {
        let mut v = (s.len() as u64).to_le_bytes().to_vec();
        v.extend_from_slice(s.as_bytes());
        v
    }

    pub(crate) fn temp_path(tag: &str) -> PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        std::env::temp_dir().join(format!("ignis-speech-{tag}-{}-{}.gguf", std::process::id(), N.fetch_add(1, Ordering::Relaxed)))
    }

    #[test]
    fn reads_every_value_type_and_the_tensor_table() {
        let path = temp_path("types");
        let mut arr = 8u32.to_le_bytes().to_vec();
        arr.extend_from_slice(&2u64.to_le_bytes());
        arr.extend_from_slice(&enc_str("a b"));
        arr.extend_from_slice(&enc_str("<|im_end|>"));
        let f32s: Vec<u8> = [1.5f32, -2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let q80 = vec![7u8; 34 * 2];
        write_gguf(
            &path,
            &[
                ("u8", 0, vec![200]),
                ("i8", 1, vec![0xff]),
                ("u16", 2, 500u16.to_le_bytes().to_vec()),
                ("i16", 3, (-3i16).to_le_bytes().to_vec()),
                ("u32", 4, 70000u32.to_le_bytes().to_vec()),
                ("i32", 5, (-70000i32).to_le_bytes().to_vec()),
                ("f32", 6, 0.25f32.to_le_bytes().to_vec()),
                ("bool", 7, vec![1]),
                ("str", 8, enc_str("qwen3vl")),
                ("arr", 9, arr),
                ("u64", 10, (1u64 << 40).to_le_bytes().to_vec()),
                ("i64", 11, (-5i64).to_le_bytes().to_vec()),
                ("f64", 12, 1e6f64.to_le_bytes().to_vec()),
                ("general.alignment", 4, 64u32.to_le_bytes().to_vec()),
            ],
            &[("w.f32", vec![2], 0, f32s.clone()), ("w.q8", vec![64], 8, q80.clone())],
            64,
        );
        let g = Gguf::open(&path).unwrap();
        assert_eq!(g.version, 3);
        assert_eq!(g.alignment, 64);
        assert_eq!(g.get("u8"), Some(&Value::U8(200)));
        assert_eq!(g.get("i8"), Some(&Value::I8(-1)));
        assert_eq!(g.get("u16").and_then(Value::as_u64), Some(500));
        assert_eq!(g.get("i16"), Some(&Value::I16(-3)));
        assert_eq!(g.get("u32").and_then(Value::as_u64), Some(70000));
        assert_eq!(g.get("i32"), Some(&Value::I32(-70000)));
        assert_eq!(g.get("f32").and_then(Value::as_f64), Some(0.25));
        assert_eq!(g.get("bool"), Some(&Value::Bool(true)));
        assert_eq!(g.get("str").and_then(Value::as_str), Some("qwen3vl"));
        let a = g.get("arr").and_then(Value::as_array).unwrap();
        assert_eq!(a, &[Value::String("a b".into()), Value::String("<|im_end|>".into())]);
        assert_eq!(g.get("u64").and_then(Value::as_u64), Some(1 << 40));
        assert_eq!(g.get("i64").and_then(Value::as_u64), None);
        assert_eq!(g.get("f64").and_then(Value::as_f64), Some(1e6));
        let t = g.tensor("w.f32").unwrap();
        assert_eq!((t.ty, t.size), (TensorType::F32, 8));
        assert_eq!(g.data(t), &f32s[..]);
        let t = g.tensor("w.q8").unwrap();
        assert_eq!((t.ty, t.size, t.offset), (TensorType::Q8_0, 68, 64));
        assert_eq!(g.data(t), &q80[..]);
        drop(g);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn refuses_what_is_not_a_well_formed_gguf() {
        let path = temp_path("bad");
        std::fs::write(&path, b"GGML\x03\0\0\0").unwrap();
        assert!(matches!(Gguf::open(&path), Err(GgufError::Format(_, m)) if m.contains("magic")));
        // A tensor whose data runs past the end of the file.
        write_gguf(&path, &[], &[("t", vec![64], 0, vec![0u8; 16])], 32);
        let mut b = std::fs::read(&path).unwrap();
        b.truncate(b.len() - 8);
        std::fs::write(&path, b).unwrap();
        assert!(matches!(Gguf::open(&path), Err(GgufError::Format(_, m)) if m.contains("past the end")));
        // A row that is not a whole number of blocks.
        write_gguf(&path, &[], &[("q", vec![40], 8, vec![0u8; 68])], 32);
        assert!(matches!(Gguf::open(&path), Err(GgufError::Format(_, m)) if m.contains("multiple")));
        // A length larger than the file.
        let mut b = b"GGUF".to_vec();
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(&u64::MAX.to_le_bytes());
        std::fs::write(&path, b).unwrap();
        assert!(matches!(Gguf::open(&path), Err(GgufError::Format(_, m)) if m.contains("exceeds")));
        std::fs::remove_file(&path).ok();
    }
}
