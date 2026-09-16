//! JPEG to RGB8 whose pixels are exactly the reference's (FFmpeg 8.1.2 as
//! built for the reference: x86-64 with assembly enabled).
//!
//! This is our own implementation, not vendored code: no FFmpeg source is
//! copied (ADR 0010 — it carries no provenance claim). It reproduces the
//! arithmetic of the reference's decode path, checked against fixtures
//! recorded from the reference (`tools/vision-fixtures`). The Huffman tables
//! below are the ITU T.81 Annex K example tables. The behaviour reproduced,
//! by the reference component that defines it:
//!
//! - `libavcodec/mjpegdec.c`: marker walk, DQT/DHT/SOF0-2/SOS/DRI, scan
//!   unescaping with restart intervals, baseline and progressive (spectral
//!   selection and successive approximation) coefficient decoding with
//!   FFmpeg's exact dequantization and integer wrapping. The DC predictor
//!   starts at `4 << 8`, which is how FFmpeg level-shifts by 128.
//! - `libavcodec/x86/simple_idct10_template.asm` `simple_idct8_put` (SSE2,
//!   chosen by `ff_idctdsp_init_x86` on x86-64): FFmpeg's 8-bit simple IDCT
//!   with the assembly's own edges — the row pass saturates to `i16`
//!   (`packssdw`) where the C wraps, keeps the C's DC-only shortcut, and the
//!   column pass adds its `+32` bias with 16-bit wraparound.
//! - `libswscale/x86/yuv_2_rgb.asm` `yuv_420_rgb24_ssse3` (the unscaled
//!   `SWS_POINT` conversion swscale picks for `yuv420p`/`yuv422p` to
//!   `rgb24` when the height is even): 16-bit fixed-point arithmetic with
//!   coefficients from `ff_yuv2rgb_c_init_tables` (BT.601, full range for
//!   JFIF's `yuvj*`), chroma shared by each 2-pixel (and for 4:2:0, 2-row)
//!   block.
//! - `media/decode/decode.cpp` `exif_orientation` and its rotation.
//!
//! Exact: 8-bit YCbCr 4:2:0 and 4:2:2 with an even height, baseline or
//! progressive. Other layouts (4:4:4, grayscale, odd heights) reach swscale's
//! generic scaler in the reference; they are decoded here with the same
//! coefficients and point-sampled chroma, which is not guaranteed to be
//! bit-identical (GitHub #176's report). Arithmetic coding, lossless,
//! hierarchical, 12-bit and CMYK JPEGs are refused as unsupported.

use super::decode::DecodeFailure;
use super::{InvalidMedia, Rgb8Image};

/// `ff_zigzag_direct`: raster index of the i-th coefficient in scan order.
const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21,
    28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61,
    54, 47, 55, 62, 63,
];

fn invalid(why: impl Into<String>) -> DecodeFailure {
    DecodeFailure::Invalid(InvalidMedia::Undecodable(why.into()))
}

fn unsupported(why: impl Into<String>) -> DecodeFailure {
    DecodeFailure::Invalid(InvalidMedia::Unsupported(why.into()))
}

// ---------------------------------------------------------------------------
// Bits and Huffman tables
// ---------------------------------------------------------------------------

/// MSB-first reader over an unescaped scan interval. Reads past the end
/// yield zeros, as FFmpeg's padded buffer does; [`Bits::left`] goes
/// negative so the MCU loop can refuse an overread.
struct Bits {
    data: Vec<u8>,
    pos: usize,
}

impl Bits {
    fn bit(&mut self) -> u32 {
        let byte = self.data.get(self.pos >> 3).copied().unwrap_or(0);
        let bit = (byte >> (7 - (self.pos & 7))) & 1;
        self.pos += 1;
        bit as u32
    }

    fn bits(&mut self, n: u32) -> u32 {
        (0..n).fold(0, |acc, _| (acc << 1) | self.bit())
    }

    /// `get_xbits`: `n` bits as a JPEG magnitude category value.
    fn signed(&mut self, n: u32) -> i32 {
        if n == 0 {
            return 0;
        }
        let v = self.bits(n) as i32;
        if v < 1 << (n - 1) { v - (1 << n) + 1 } else { v }
    }

    fn left(&self) -> isize {
        (self.data.len() * 8) as isize - self.pos as isize
    }
}

/// A canonical JPEG Huffman table.
#[derive(Clone)]
struct Huffman {
    maxcode: [i32; 17],
    mincode: [i32; 17],
    valptr: [usize; 17],
    values: Vec<u8>,
}

impl Huffman {
    fn new(counts: &[u8; 16], values: Vec<u8>) -> Result<Self, DecodeFailure> {
        let mut table = Huffman { maxcode: [-1; 17], mincode: [0; 17], valptr: [0; 17], values };
        let (mut code, mut k) = (0i32, 0usize);
        for len in 1..=16 {
            let n = counts[len - 1] as i32;
            table.valptr[len] = k;
            table.mincode[len] = code;
            code += n;
            k += n as usize;
            if n > 0 {
                table.maxcode[len] = code - 1;
            }
            if code > 1 << len {
                return Err(invalid("huffman table has more codes than lengths allow"));
            }
            code <<= 1;
        }
        Ok(table)
    }

    fn decode(&self, bits: &mut Bits) -> Result<u8, DecodeFailure> {
        let mut code = 0i32;
        for len in 1..=16 {
            code = (code << 1) | bits.bit() as i32;
            if code <= self.maxcode[len] {
                return Ok(self.values[self.valptr[len] + (code - self.mincode[len]) as usize]);
            }
        }
        Err(invalid("bad huffman code"))
    }
}

// ---------------------------------------------------------------------------
// IDCT (simple_idct8_put, SSE2 semantics)
// ---------------------------------------------------------------------------

const W1: i32 = 22725;
const W2: i32 = 21407;
const W3: i32 = 19266;
const W4: i32 = 16383;
const W5: i32 = 12873;
const W6: i32 = 8867;
const W7: i32 = 4520;

fn sat16(x: i32) -> i16 {
    x.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

/// One 1-D pass over `v` (8 values): FFmpeg's even/odd butterflies in
/// wrapping 32-bit arithmetic, `(a ± b) >> shift` saturated to `i16`.
fn idct_1d(v: [i32; 8], round: i32, shift: u32) -> [i16; 8] {
    let a = |w: i32, x: i32| w.wrapping_mul(x);
    let base = a(W4, v[0]).wrapping_add(round);
    let a0 = base.wrapping_add(a(W2, v[2])).wrapping_add(a(W4, v[4])).wrapping_add(a(W6, v[6]));
    let a1 = base.wrapping_add(a(W6, v[2])).wrapping_sub(a(W4, v[4])).wrapping_sub(a(W2, v[6]));
    let a2 = base.wrapping_sub(a(W6, v[2])).wrapping_sub(a(W4, v[4])).wrapping_add(a(W2, v[6]));
    let a3 = base.wrapping_sub(a(W2, v[2])).wrapping_add(a(W4, v[4])).wrapping_sub(a(W6, v[6]));
    let b0 = a(W1, v[1]).wrapping_add(a(W3, v[3])).wrapping_add(a(W5, v[5])).wrapping_add(a(W7, v[7]));
    let b1 = a(W3, v[1]).wrapping_sub(a(W7, v[3])).wrapping_sub(a(W1, v[5])).wrapping_sub(a(W5, v[7]));
    let b2 = a(W5, v[1]).wrapping_sub(a(W1, v[3])).wrapping_add(a(W7, v[5])).wrapping_add(a(W3, v[7]));
    let b3 = a(W7, v[1]).wrapping_sub(a(W5, v[3])).wrapping_add(a(W3, v[5])).wrapping_sub(a(W1, v[7]));
    let out = |x: i32| sat16(x >> shift);
    [
        out(a0.wrapping_add(b0)),
        out(a1.wrapping_add(b1)),
        out(a2.wrapping_add(b2)),
        out(a3.wrapping_add(b3)),
        out(a3.wrapping_sub(b3)),
        out(a2.wrapping_sub(b2)),
        out(a1.wrapping_sub(b1)),
        out(a0.wrapping_sub(b0)),
    ]
}

/// IDCT `block` (raster order) and write the clamped 8x8 pixels at
/// `plane[offset..]` with `stride`.
fn idct_put(block: &[i16; 64], plane: &mut [u8], offset: usize, stride: usize) {
    let mut rows = [0i16; 64];
    for r in 0..8 {
        let row = &block[r * 8..r * 8 + 8];
        let out = if row[1..].iter().all(|&c| c == 0) {
            // The DC-only shortcut the assembly copies from the C: a shift,
            // not a multiply by W4 = 16383.
            [(row[0] as i32).wrapping_shl(3) as i16; 8]
        } else {
            idct_1d(std::array::from_fn(|i| row[i] as i32), 1 << 10, 11)
        };
        rows[r * 8..r * 8 + 8].copy_from_slice(&out);
    }
    for c in 0..8 {
        let mut col: [i32; 8] = std::array::from_fn(|r| rows[r * 8 + c] as i32);
        col[0] = (col[0] as i16).wrapping_add(32) as i32;
        for (r, value) in idct_1d(col, 0, 20).into_iter().enumerate() {
            plane[offset + r * stride + c] = value.clamp(0, 255) as u8;
        }
    }
}

// ---------------------------------------------------------------------------
// Decoder state
// ---------------------------------------------------------------------------

struct Component {
    id: u8,
    h: usize,
    v: usize,
    quant: usize,
}

struct Frame {
    width: usize,
    height: usize,
    progressive: bool,
    components: Vec<Component>,
    h_max: usize,
    v_max: usize,
    planes: Vec<Vec<u8>>,
    strides: Vec<usize>,
    /// Progressive only: coefficient blocks per component, and the highest
    /// coded index per block.
    blocks: Vec<Vec<[i16; 64]>>,
    last_nnz: Vec<Vec<u8>>,
    block_stride: Vec<usize>,
    /// Chroma subsampling shifts of the output pixel format.
    chroma_shift: (usize, usize),
}

impl Frame {
    /// Visible size of plane `c` (chroma planes are subsampled).
    fn plane_size(&self, c: usize) -> (usize, usize) {
        if c == 1 || c == 2 {
            (self.width.div_ceil(1 << self.chroma_shift.0), self.height.div_ceil(1 << self.chroma_shift.1))
        } else {
            (self.width, self.height)
        }
    }
}

struct Scan {
    /// (component index, DC table, AC table) per scan component.
    components: Vec<(usize, usize, usize)>,
    ss: usize,
    se: usize,
    ah: u32,
    al: u32,
}

struct Decoder {
    quant: [[u16; 64]; 4],
    dc: [Option<Huffman>; 4],
    ac: [Option<Huffman>; 4],
    restart_interval: usize,
    frame: Option<Frame>,
    scans: usize,
}

// `jpegtabs.h`: the standard tables FFmpeg installs before any DHT
// (`init_default_huffman_tables`), so a JPEG without DHT still decodes.
const DC_LUMINANCE_COUNTS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
const DC_CHROMINANCE_COUNTS: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
const DC_VALUES: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const AC_LUMINANCE_COUNTS: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d];
const AC_LUMINANCE_VALUES: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14,
    0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09,
    0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a,
    0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65,
    0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88,
    0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9,
    0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca,
    0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
    0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
];
const AC_CHROMINANCE_COUNTS: [u8; 16] = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];
const AC_CHROMINANCE_VALUES: [u8; 162] = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13, 0x22, 0x32,
    0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0, 0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16,
    0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64,
    0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86,
    0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8,
    0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
    0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
];

fn be16(bytes: &[u8], at: usize) -> Result<usize, DecodeFailure> {
    bytes
        .get(at..at + 2)
        .map(|b| (b[0] as usize) << 8 | b[1] as usize)
        .ok_or_else(|| invalid("truncated marker segment"))
}

/// `mjpeg_parse_len`: the segment payload after the length field.
fn segment(bytes: &[u8], at: usize) -> Result<&[u8], DecodeFailure> {
    let len = be16(bytes, at)?;
    if len < 2 || bytes.len() < at + len {
        return Err(invalid("invalid marker segment length"));
    }
    Ok(&bytes[at + 2..at + len])
}

/// `ff_mjpeg_find_marker`: the next marker code at or after `*pos`, with
/// `*pos` left just past it.
fn find_marker(bytes: &[u8], pos: &mut usize) -> Option<u8> {
    let mut i = *pos;
    while let Some(ff) = bytes[i..].iter().position(|&b| b == 0xff) {
        i += ff + 1;
        while i < bytes.len() {
            let val = bytes[i];
            i += 1;
            if val != 0xff {
                if val >= 0xc0 && val <= 0xfe {
                    *pos = i;
                    return Some(val);
                }
                break;
            }
        }
    }
    *pos = bytes.len();
    None
}

/// `ff_mjpeg_unescape_sos`: the entropy-coded bytes from `start` to the
/// next marker with `FF 00` unstuffed, and where scanning resumes — after a
/// restart marker, or at any other marker.
fn unescape(bytes: &[u8], start: usize) -> (Vec<u8>, usize) {
    let mut out = Vec::new();
    let (mut src, mut ptr) = (start, start);
    while let Some(ff) = bytes[ptr..].iter().position(|&b| b == 0xff) {
        ptr += ff + 1;
        if ptr < bytes.len() {
            out.extend_from_slice(&bytes[src..ptr - 1]);
            let mut x = bytes[ptr];
            ptr += 1;
            while x == 0xff && ptr < bytes.len() {
                x = bytes[ptr];
                ptr += 1;
            }
            src = ptr;
            match x {
                0 => out.push(0xff),
                0xd0..=0xd7 => return (out, ptr),
                _ => return (out, ptr - 2),
            }
        }
    }
    out.extend_from_slice(&bytes[src..]);
    (out, bytes.len())
}

impl Decoder {
    fn dqt(&mut self, mut data: &[u8]) -> Result<(), DecodeFailure> {
        while data.len() >= 65 {
            let (pr, index) = ((data[0] >> 4) as usize, (data[0] & 0x0f) as usize);
            if pr > 1 || index >= 4 || data.len() < 1 + 64 * (1 + pr) {
                return Err(invalid("dqt: invalid table"));
            }
            for i in 0..64 {
                self.quant[index][i] = if pr == 1 {
                    (data[1 + 2 * i] as u16) << 8 | data[2 + 2 * i] as u16
                } else {
                    data[1 + i] as u16
                };
            }
            data = &data[1 + 64 * (1 + pr)..];
        }
        Ok(())
    }

    fn dht(&mut self, mut data: &[u8]) -> Result<(), DecodeFailure> {
        while !data.is_empty() {
            if data.len() < 17 {
                return Err(invalid("dht: truncated"));
            }
            let (class, index) = ((data[0] >> 4) as usize, (data[0] & 0x0f) as usize);
            if class >= 2 || index >= 4 {
                return Err(invalid("dht: invalid table"));
            }
            let counts: [u8; 16] = data[1..17].try_into().unwrap();
            let n: usize = counts.iter().map(|&c| c as usize).sum();
            if data.len() < 17 + n || n > 256 {
                return Err(invalid("dht: truncated"));
            }
            let table = Huffman::new(&counts, data[17..17 + n].to_vec())?;
            if class == 0 { self.dc[index] = Some(table) } else { self.ac[index] = Some(table) }
            data = &data[17 + n..];
        }
        Ok(())
    }

    fn sof(&mut self, data: &[u8], progressive: bool, max_pixels: u64) -> Result<(), DecodeFailure> {
        if data.len() < 6 {
            return Err(invalid("sof: truncated"));
        }
        if data[0] != 8 {
            return Err(unsupported(format!("{}-bit JPEG", data[0])));
        }
        let height = (data[1] as usize) << 8 | data[2] as usize;
        let width = (data[3] as usize) << 8 | data[4] as usize;
        let count = data[5] as usize;
        if width == 0 || height == 0 {
            return Err(invalid("sof: zero image dimension"));
        }
        if data.len() != 6 + 3 * count || count == 0 {
            return Err(invalid("sof: component count mismatch"));
        }
        if count != 1 && count != 3 {
            return Err(unsupported(format!("{count}-component JPEG")));
        }
        let pixels = width as u64 * height as u64;
        if pixels > max_pixels {
            return Err(DecodeFailure::Pixels(pixels));
        }
        let components: Vec<Component> = (0..count)
            .map(|i| {
                let c = &data[6 + 3 * i..9 + 3 * i];
                Component { id: c[0], h: (c[1] >> 4) as usize, v: (c[1] & 0x0f) as usize, quant: c[2] as usize }
            })
            .collect();
        if components.iter().any(|c| c.h == 0 || c.v == 0 || c.h > 4 || c.v > 4 || c.quant >= 4) {
            return Err(invalid("sof: invalid sampling factor or quant table"));
        }
        let h_max = components.iter().map(|c| c.h).max().unwrap();
        let v_max = components.iter().map(|c| c.v).max().unwrap();
        let (mb_w, mb_h) = (width.div_ceil(h_max * 8), height.div_ceil(v_max * 8));
        let chroma_shift = if count == 3 {
            let (h, v) = (h_max / components[1].h, v_max / components[1].v);
            (h.trailing_zeros() as usize, v.trailing_zeros() as usize)
        } else {
            (0, 0)
        };
        let strides: Vec<usize> = components.iter().map(|c| mb_w * c.h * 8).collect();
        let planes = components.iter().zip(&strides).map(|(c, &s)| vec![0u8; s * mb_h * c.v * 8]).collect();
        let (blocks, last_nnz) = if progressive {
            (
                components.iter().map(|c| vec![[0i16; 64]; mb_w * mb_h * c.h * c.v]).collect(),
                components.iter().map(|c| vec![0u8; mb_w * mb_h * c.h * c.v]).collect(),
            )
        } else {
            (Vec::new(), Vec::new())
        };
        let block_stride = components.iter().map(|c| mb_w * c.h).collect();
        self.frame = Some(Frame {
            width,
            height,
            progressive,
            components,
            h_max,
            v_max,
            planes,
            strides,
            blocks,
            last_nnz,
            block_stride,
            chroma_shift,
        });
        self.scans = 0;
        Ok(())
    }

    fn sos_header(&self, data: &[u8]) -> Result<Scan, DecodeFailure> {
        let frame = self.frame.as_ref().ok_or_else(|| invalid("sos before sof"))?;
        let count = *data.first().ok_or_else(|| invalid("sos: truncated"))? as usize;
        if count == 0 || count > 4 || data.len() != 4 + 2 * count {
            return Err(invalid("sos: length mismatch"));
        }
        let mut components = Vec::with_capacity(count);
        for i in 0..count {
            let (id, tables) = (data[1 + 2 * i], data[2 + 2 * i]);
            let index = frame
                .components
                .iter()
                .position(|c| c.id == id)
                .ok_or_else(|| invalid("sos: unknown component"))?;
            let (dc, ac) = ((tables >> 4) as usize, (tables & 0x0f) as usize);
            if dc >= 4 || ac >= 4 || self.dc[dc].is_none() {
                return Err(invalid("sos: missing huffman table"));
            }
            components.push((index, dc, ac));
        }
        let at = 1 + 2 * count;
        Ok(Scan { components, ss: data[at] as usize, se: data[at + 1] as usize, ah: (data[at + 2] >> 4) as u32, al: (data[at + 2] & 0x0f) as u32 })
    }

    /// Decode one scan starting at `start` (just past the SOS segment);
    /// returns where marker scanning resumes.
    fn scan(&mut self, scan: &Scan, bytes: &[u8], start: usize) -> Result<usize, DecodeFailure> {
        let frame = self.frame.as_mut().unwrap();
        let progressive_ac = frame.progressive && scan.ss != 0;
        if !frame.progressive && scan.components.iter().any(|&(_, _, ac)| self.ac[ac].is_none()) {
            return Err(invalid("sos: missing huffman table"));
        }
        if frame.progressive && self.ac[scan.components[0].2].is_none() && scan.ss != 0 {
            return Err(invalid("sos: missing huffman table"));
        }
        // Scan geometry: interleaved scans walk MCUs; a single-component
        // scan walks that component's blocks one at a time.
        let interleaved = scan.components.len() > 1;
        let (mb_w, mb_h) = if interleaved {
            (frame.width.div_ceil(frame.h_max * 8), frame.height.div_ceil(frame.v_max * 8))
        } else {
            let c = &frame.components[scan.components[0].0];
            (frame.width.div_ceil(frame.h_max / c.h * 8), frame.height.div_ceil(frame.v_max / c.v * 8))
        };
        let sampling: Vec<(usize, usize)> = scan
            .components
            .iter()
            .map(|&(c, _, _)| if interleaved { (frame.components[c].h, frame.components[c].v) } else { (1, 1) })
            .collect();

        let mut gpos = start;
        let mut bits = Bits { data: Vec::new(), pos: 0 };
        let mut restart_count: isize = -1;
        let mut last_dc = vec![0i32; scan.components.len()];
        let mut eobrun = 0usize;

        if progressive_ac {
            let &(c, _, ac) = &scan.components[0];
            if scan.se < scan.ss || scan.se > 63 {
                return Err(invalid("progressive scan: invalid spectral selection"));
            }
            let table = self.ac[ac].clone().unwrap();
            let quant = self.quant[frame.components[c].quant];
            for mb_y in 0..mb_h {
                for mb_x in 0..mb_w {
                    if should_restart(self.restart_interval, &mut restart_count) {
                        (bits.data, gpos) = unescape(bytes, gpos);
                        bits.pos = 0;
                        eobrun = 0;
                    }
                    let index = mb_y * frame.block_stride[c] + mb_x;
                    let block = &mut frame.blocks[c][index];
                    let nnz = &mut frame.last_nnz[c][index];
                    if scan.ah != 0 {
                        refine_ac(&mut bits, &table, block, nnz, &quant, scan, &mut eobrun)?;
                    } else {
                        progressive_ac_block(&mut bits, &table, block, nnz, &quant, scan, &mut eobrun)?;
                    }
                    if bits.left() < 0 {
                        return Err(invalid("overread in progressive scan"));
                    }
                }
            }
            self.scans += 1;
            return Ok(gpos);
        }

        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                if should_restart(self.restart_interval, &mut restart_count) {
                    (bits.data, gpos) = unescape(bytes, gpos);
                    bits.pos = 0;
                    last_dc.fill(4 << 8);
                }
                if bits.left() < 0 {
                    return Err(invalid("overread in scan"));
                }
                for (i, &(c, dc, ac)) in scan.components.iter().enumerate() {
                    let (h, v) = sampling[i];
                    let quant = self.quant[frame.components[c].quant];
                    let (plane_w, plane_h) = frame.plane_size(c);
                    for y in 0..v {
                        for x in 0..h {
                            let (bx, by) = (h * mb_x + x, v * mb_y + y);
                            if frame.progressive {
                                let index = frame.block_stride[c] * by + bx;
                                let target = &mut frame.blocks[c][index];
                                if scan.ah != 0 {
                                    let bit = bits.bit() as i32;
                                    target[0] = target[0].wrapping_add(((bit * quant[0] as i32) << scan.al) as i16);
                                } else {
                                    let value = decode_dc(&mut bits, self.dc[dc].as_ref().unwrap())?;
                                    *target = [0; 64];
                                    let val = (value as u32)
                                        .wrapping_mul((quant[0] as u32) << scan.al)
                                        .wrapping_add(last_dc[i] as u32);
                                    last_dc[i] = val as i32;
                                    target[0] = val as i16;
                                }
                                continue;
                            }
                            let mut block = [0i16; 64];
                            let value = decode_dc(&mut bits, self.dc[dc].as_ref().unwrap())?;
                            let val = value.wrapping_mul(quant[0] as i32).wrapping_add(last_dc[i]);
                            last_dc[i] = val;
                            block[0] = val.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                            baseline_ac(&mut bits, self.ac[ac].as_ref().unwrap(), &mut block, &quant)?;
                            if 8 * bx < plane_w && 8 * by < plane_h {
                                let stride = frame.strides[c];
                                idct_put(&block, &mut frame.planes[c], by * 8 * stride + bx * 8, stride);
                            }
                        }
                    }
                }
            }
        }
        self.scans += 1;
        Ok(gpos)
    }

    /// `mjpeg_idct_scan_progressive_ac`, at EOI.
    fn finish_progressive(&mut self) {
        let frame = self.frame.as_mut().unwrap();
        for c in 0..frame.components.len() {
            let comp = &frame.components[c];
            let (h, v) = (frame.h_max / comp.h, frame.v_max / comp.v);
            let (mb_w, mb_h) = (frame.width.div_ceil(h * 8), frame.height.div_ceil(v * 8));
            let stride = frame.strides[c];
            for mb_y in 0..mb_h {
                for mb_x in 0..mb_w {
                    let block = frame.blocks[c][mb_y * frame.block_stride[c] + mb_x];
                    idct_put(&block, &mut frame.planes[c], mb_y * 8 * stride + mb_x * 8, stride);
                }
            }
        }
    }
}

/// `ff_mjpeg_should_restart`.
fn should_restart(interval: usize, count: &mut isize) -> bool {
    if interval > 0 {
        let restart = *count <= 0;
        if restart {
            *count = interval as isize;
        }
        *count -= 1;
        restart
    } else if *count < 0 {
        *count = 0;
        true
    } else {
        false
    }
}

/// `mjpeg_decode_dc`.
fn decode_dc(bits: &mut Bits, table: &Huffman) -> Result<i32, DecodeFailure> {
    let size = table.decode(bits)?;
    if size > 16 {
        return Err(invalid("bad dc code"));
    }
    Ok(bits.signed(size as u32))
}

/// The AC half of `decode_block`: FFmpeg's AC symbols are `value + 16`
/// (EOB is `16 * 256`), so a symbol advances the index by run + 1.
fn baseline_ac(bits: &mut Bits, table: &Huffman, block: &mut [i16; 64], quant: &[u16; 64]) -> Result<(), DecodeFailure> {
    let mut i = 0usize;
    loop {
        let value = table.decode(bits)?;
        let code = if value == 0 { 16 * 256 } else { value as usize + 16 };
        i += code >> 4;
        let size = (code & 0xf) as u32;
        if size != 0 {
            let level = bits.signed(size);
            if i > 63 {
                return Err(invalid("ac coefficient index out of range"));
            }
            block[ZIGZAG[i]] = level.wrapping_mul(quant[i] as i32) as i16;
        }
        if i >= 63 {
            return Ok(());
        }
    }
}

/// `decode_block_progressive`.
fn progressive_ac_block(
    bits: &mut Bits,
    table: &Huffman,
    block: &mut [i16; 64],
    last_nnz: &mut u8,
    quant: &[u16; 64],
    scan: &Scan,
    eobrun: &mut usize,
) -> Result<(), DecodeFailure> {
    if *eobrun > 0 {
        *eobrun -= 1;
        return Ok(());
    }
    let (se, al) = (scan.se, scan.al);
    let mut i = scan.ss;
    loop {
        let code = table.decode(bits)? as usize;
        let run = code >> 4;
        let size = (code & 0xf) as u32;
        if size != 0 {
            i += run;
            let level = bits.signed(size) as u32;
            if i >= se {
                if i == se {
                    block[ZIGZAG[se]] = level.wrapping_mul((quant[se] as u32) << al) as i16;
                    break;
                }
                return Err(invalid("progressive ac index out of range"));
            }
            block[ZIGZAG[i]] = level.wrapping_mul((quant[i] as u32) << al) as i16;
        } else if run == 0xf {
            i += 15;
            if i >= se {
                return Err(invalid("progressive ac zero run overflow"));
            }
        } else {
            let mut val = 1usize << run;
            if run != 0 {
                val += bits.bits(run as u32) as usize;
            }
            *eobrun = val - 1;
            break;
        }
        i += 1;
    }
    if i > *last_nnz as usize {
        *last_nnz = i as u8;
    }
    Ok(())
}

/// `decode_block_refinement`.
fn refine_ac(
    bits: &mut Bits,
    table: &Huffman,
    block: &mut [i16; 64],
    last_nnz: &mut u8,
    quant: &[u16; 64],
    scan: &Scan,
    eobrun: &mut usize,
) -> Result<(), DecodeFailure> {
    let (se, al) = (scan.se, scan.al);
    let last = se.min(*last_nnz as usize);
    let mut i = scan.ss;
    let refine = |bits: &mut Bits, block: &mut [i16; 64], i: usize| {
        let j = ZIGZAG[i];
        let sign = (block[j] >> 15) as i32;
        let bit = bits.bit() as i32;
        block[j] = block[j].wrapping_add((bit * ((quant[i] as i32 ^ sign) - sign) << al) as i16);
    };
    // ZERO_RUN: refine the nonzero coefficients passed while skipping `run`
    // zero ones; returns the index of the zero coefficient that ends it.
    let zero_run = |bits: &mut Bits, block: &mut [i16; 64], i: &mut usize, mut run: isize| -> Result<(), DecodeFailure> {
        loop {
            if *i > last {
                *i += run as usize;
                if *i > se {
                    return Err(invalid("refinement index out of range"));
                }
                return Ok(());
            }
            if block[ZIGZAG[*i]] != 0 {
                refine(bits, block, *i);
            } else {
                if run == 0 {
                    return Ok(());
                }
                run -= 1;
            }
            *i += 1;
        }
    };
    if *eobrun > 0 {
        *eobrun -= 1;
    } else {
        loop {
            let code = table.decode(bits)? as usize;
            let run = (code >> 4) as isize;
            if code & 0xf != 0 {
                let val = bits.bit() as i32 - 1;
                zero_run(bits, block, &mut i, run)?;
                let j = ZIGZAG[i];
                block[j] = ((((quant[i] as i32) << al) ^ val) - val) as i16;
                if i == se {
                    if i > *last_nnz as usize {
                        *last_nnz = i as u8;
                    }
                    return Ok(());
                }
            } else if run == 0xf {
                zero_run(bits, block, &mut i, run)?;
            } else {
                let mut val = 1usize << run;
                if run != 0 {
                    val += bits.bits(run as u32) as usize;
                }
                *eobrun = val - 1;
                break;
            }
            i += 1;
        }
        if i > *last_nnz as usize {
            *last_nnz = i as u8;
        }
    }
    while i <= last {
        if block[ZIGZAG[i]] != 0 {
            refine(bits, block, i);
        }
        i += 1;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// YCbCr to RGB (yuv_420_rgb24_ssse3 semantics)
// ---------------------------------------------------------------------------

/// `roundToInt16`.
fn round_to_i16(f: i64) -> i16 {
    let r = ((f + (1 << 15)) >> 16) as i32;
    if r < -0x7fff {
        i16::MIN
    } else {
        r.min(0x7fff) as i16
    }
}

/// The SIMD coefficients `ff_yuv2rgb_c_init_tables` stores for BT.601
/// (`ff_yuv2rgb_coeffs[SWS_CS_DEFAULT]`), default brightness, contrast and
/// saturation: `[y, vr, ub, vg, ug, y_offset]`.
fn yuv2rgb_coefficients(full_range: bool) -> [i16; 6] {
    let inv = [104_597i64, 132_201, 25_675, 53_279];
    let (mut crv, mut cbu, mut cgu, mut cgv) = (inv[0], inv[1], -inv[2], -inv[3]);
    let (mut cy, mut oy) = (1i64 << 16, 0i64);
    if full_range {
        crv = crv * 224 / 255;
        cbu = cbu * 224 / 255;
        cgu = cgu * 224 / 255;
        cgv = cgv * 224 / 255;
    } else {
        cy = cy * 255 / 219;
        oy = 16 << 16;
    }
    let (contrast, saturation) = (1i64 << 16, 1i64 << 16);
    cy = (cy * contrast) >> 16;
    crv = (crv * contrast * saturation) >> 32;
    cbu = (cbu * contrast * saturation) >> 32;
    cgu = (cgu * contrast * saturation) >> 32;
    cgv = (cgv * contrast * saturation) >> 32;
    [
        round_to_i16(cy * (1 << 13)),
        round_to_i16(crv * (1 << 13)),
        round_to_i16(cbu * (1 << 13)),
        round_to_i16(cgv * (1 << 13)),
        round_to_i16(cgu * (1 << 13)),
        round_to_i16(oy * (1 << 3)),
    ]
}

fn pmulhw(a: i16, b: i16) -> i16 {
    ((a as i32 * b as i32) >> 16) as i16
}

/// One pixel of `yuv_420_rgb24_ssse3`.
fn yuv_to_rgb(y: u8, u: u8, v: u8, k: &[i16; 6]) -> [u8; 3] {
    let [yc, vrc, ubc, vgc, ugc, yoff] = *k;
    let u = ((u as i16) << 3).saturating_sub(0x400);
    let v = ((v as i16) << 3).saturating_sub(0x400);
    let luma = pmulhw(((y as i16) << 3).wrapping_sub(yoff), yc);
    let g = pmulhw(u, ugc).saturating_add(pmulhw(v, vgc)).saturating_add(luma);
    let b = pmulhw(u, ubc).saturating_add(luma);
    let r = pmulhw(v, vrc).saturating_add(luma);
    [r.clamp(0, 255) as u8, g.clamp(0, 255) as u8, b.clamp(0, 255) as u8]
}

fn to_rgb(frame: &Frame) -> Rgb8Image {
    let (w, h) = (frame.width, frame.height);
    // JFIF YCbCr decodes to `yuvj*`, which swscale treats as full range.
    let k = yuv2rgb_coefficients(true);
    let mut rgb = Vec::with_capacity(w * h * 3);
    if frame.components.len() == 1 {
        for y in 0..h {
            let row = &frame.planes[0][y * frame.strides[0]..];
            rgb.extend(row[..w].iter().flat_map(|&l| [l, l, l]));
        }
    } else {
        let (hs, vs) = frame.chroma_shift;
        for y in 0..h {
            let (row_y, row_c) = (y * frame.strides[0], (y >> vs) * frame.strides[1]);
            for x in 0..w {
                let luma = frame.planes[0][row_y + x];
                let (cb, cr) = (frame.planes[1][row_c + (x >> hs)], frame.planes[2][row_c + (x >> hs)]);
                rgb.extend_from_slice(&yuv_to_rgb(luma, cb, cr, &k));
            }
        }
    }
    Rgb8Image { width: w as u32, height: h as u32, rgb }
}

// ---------------------------------------------------------------------------
// EXIF orientation (decode.cpp)
// ---------------------------------------------------------------------------

/// `exif_orientation`: the first APP1 Exif IFD0 orientation tag of a JPEG,
/// 1 when absent or malformed.
fn exif_orientation(bytes: &[u8]) -> u16 {
    let bytes = &bytes[..bytes.len().min(1 << 20)];
    if bytes.len() < 4 || bytes[0] != 0xff || bytes[1] != 0xd8 {
        return 1;
    }
    let be16 = |at: usize| (bytes[at] as u16) << 8 | bytes[at + 1] as u16;
    let mut marker = 2;
    while marker + 4 <= bytes.len() {
        if bytes[marker] != 0xff {
            break;
        }
        let kind = bytes[marker + 1];
        if kind == 0xda || kind == 0xd9 {
            break;
        }
        let length = be16(marker + 2) as usize;
        if length < 2 || marker + 2 + length > bytes.len() {
            break;
        }
        let (payload, payload_size) = (marker + 4, length - 2);
        if kind == 0xe1 && payload_size >= 14 && &bytes[payload..payload + 6] == b"Exif\0\0" {
            let tiff = payload + 6;
            let little = &bytes[tiff..tiff + 2] == b"II";
            if !little && &bytes[tiff..tiff + 2] != b"MM" {
                return 1;
            }
            let u16_at = |at: usize| -> u16 {
                match bytes.get(at..at + 2) {
                    Some(b) if little => u16::from_le_bytes([b[0], b[1]]),
                    Some(b) => u16::from_be_bytes([b[0], b[1]]),
                    None => 0,
                }
            };
            let u32_at = |at: usize| -> u32 {
                match bytes.get(at..at + 4) {
                    Some(b) if little => u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                    Some(b) => u32::from_be_bytes([b[0], b[1], b[2], b[3]]),
                    None => 0,
                }
            };
            if u16_at(tiff + 2) != 42 {
                return 1;
            }
            let ifd = tiff + u32_at(tiff + 4) as usize;
            if ifd + 2 > bytes.len() {
                return 1;
            }
            for i in 0..u16_at(ifd) as usize {
                let entry = ifd + 2 + i * 12;
                if entry + 12 > bytes.len() {
                    return 1;
                }
                if u16_at(entry) == 0x0112 && u16_at(entry + 2) == 3 && u32_at(entry + 4) == 1 {
                    let orientation = u16_at(entry + 8);
                    return if (1..=8).contains(&orientation) { orientation } else { 1 };
                }
            }
            return 1;
        }
        marker += 2 + length;
    }
    1
}

/// `Decoder::rgb`'s orientation remap.
fn orient(image: Rgb8Image, orientation: u16) -> Rgb8Image {
    if orientation == 1 {
        return image;
    }
    let (w, h) = (image.width as usize, image.height as usize);
    let swap = orientation >= 5;
    let (rw, rh) = if swap { (h, w) } else { (w, h) };
    let mut rgb = vec![0u8; image.rgb.len()];
    for y in 0..h {
        for x in 0..w {
            let (rx, ry) = match orientation {
                2 => (w - 1 - x, y),
                3 => (w - 1 - x, h - 1 - y),
                4 => (x, h - 1 - y),
                5 => (y, x),
                6 => (h - 1 - y, x),
                7 => (h - 1 - y, w - 1 - x),
                _ => (y, w - 1 - x),
            };
            let (src, dst) = ((y * w + x) * 3, (ry * rw + rx) * 3);
            rgb[dst..dst + 3].copy_from_slice(&image.rgb[src..src + 3]);
        }
    }
    Rgb8Image { width: rw as u32, height: rh as u32, rgb }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Decode a JPEG to RGB8 with its EXIF orientation applied. `max_pixels`
/// bounds the frame size before anything is allocated.
pub fn decode(bytes: &[u8], max_pixels: u64) -> Result<Rgb8Image, DecodeFailure> {
    let table = |counts: &[u8; 16], values: &[u8]| Huffman::new(counts, values.to_vec()).ok();
    let mut decoder = Decoder {
        quant: [[0; 64]; 4],
        dc: [table(&DC_LUMINANCE_COUNTS, &DC_VALUES), table(&DC_CHROMINANCE_COUNTS, &DC_VALUES), None, None],
        ac: [
            table(&AC_LUMINANCE_COUNTS, &AC_LUMINANCE_VALUES),
            table(&AC_CHROMINANCE_COUNTS, &AC_CHROMINANCE_VALUES),
            None,
            None,
        ],
        restart_interval: 0,
        frame: None,
        scans: 0,
    };
    let mut pos = 0;
    loop {
        let Some(marker) = find_marker(bytes, &mut pos) else {
            // FFmpeg emulates a missing EOI once a scan was decoded.
            if decoder.frame.is_some() && decoder.scans > 0 {
                break;
            }
            return Err(invalid("image contains no decoded frame"));
        };
        match marker {
            0xd8 => decoder.restart_interval = 0,
            0xdb => decoder.dqt(segment(bytes, pos)?)?,
            0xc4 => decoder.dht(segment(bytes, pos)?)?,
            0xc0 | 0xc1 => decoder.sof(segment(bytes, pos)?, false, max_pixels)?,
            0xc2 => decoder.sof(segment(bytes, pos)?, true, max_pixels)?,
            0xc3 | 0xc5..=0xc7 | 0xc9..=0xcb | 0xcd..=0xcf | 0xf7 => {
                return Err(unsupported("lossless, hierarchical or arithmetic-coded JPEG"));
            }
            0xdd => {
                if be16(bytes, pos)? != 4 {
                    return Err(invalid("dri: invalid length"));
                }
                decoder.restart_interval = be16(bytes, pos + 2)?;
            }
            0xda => {
                let header = segment(bytes, pos)?;
                let scan = decoder.sos_header(header)?;
                pos = decoder.scan(&scan, bytes, pos + 2 + header.len())?;
                continue;
            }
            0xd9 => {
                if decoder.frame.is_some() {
                    break;
                }
            }
            _ => {}
        }
    }
    let frame = decoder.frame.as_ref().unwrap();
    if frame.progressive && decoder.scans > 0 {
        decoder.finish_progressive();
    }
    let rgb = to_rgb(decoder.frame.as_ref().unwrap());
    Ok(orient(rgb, exif_orientation(bytes)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_dc_only_block_takes_the_shift_shortcut() {
        // DC 1600: the row pass gives 1600 << 3 = 12800 in every cell
        // (W4 * 1600 >> 11 would give 12799), then the column pass gives
        // (16383 * (12800 + 32) + ...) >> 20 = 200 for every pixel.
        let mut block = [0i16; 64];
        block[0] = 1600;
        let mut plane = [0u8; 64];
        idct_put(&block, &mut plane, 0, 8);
        let expected = ((W4 as i64 * (12800 + 32)) >> 20) as u8;
        assert_eq!(expected, 200);
        assert!(plane.iter().all(|&p| p == expected), "{plane:?}");
    }

    #[test]
    fn the_row_pass_saturates_where_the_c_would_wrap() {
        // (16383 * 32767 + 1024 + 22725 * 32767) >> 11 = 625709: the C
        // stores it wrapped into an i16, packssdw saturates.
        let out = idct_1d([32767, 32767, 0, 0, 0, 0, 0, 0], 1 << 10, 11);
        assert_eq!(out[0], i16::MAX);
    }

    #[test]
    fn full_range_bt601_coefficients_match_swscale() {
        // Worked from ff_yuv2rgb_c_init_tables by hand.
        assert_eq!(yuv2rgb_coefficients(true), [8192, 11485, 14516, -5850, -2819, 0]);
    }

    #[test]
    fn neutral_chroma_is_gray_and_extremes_clamp() {
        let k = yuv2rgb_coefficients(true);
        assert_eq!(yuv_to_rgb(128, 128, 128, &k), [128, 128, 128]);
        assert_eq!(yuv_to_rgb(0, 128, 128, &k), [0, 0, 0]);
        assert_eq!(yuv_to_rgb(255, 128, 128, &k), [255, 255, 255]);
        assert_eq!(yuv_to_rgb(255, 128, 255, &k)[0], 255);
    }

    #[test]
    fn unescape_unstuffs_and_stops_at_markers() {
        let bytes = [0xda, 0x12, 0xff, 0x00, 0x34, 0xff, 0xd0, 0x56, 0xff, 0xd9];
        assert_eq!(unescape(&bytes, 1), (vec![0x12, 0xff, 0x34], 7));
        assert_eq!(unescape(&bytes, 7), (vec![0x56], 8));
    }

    #[test]
    fn huffman_codes_are_canonical() {
        // Two 2-bit codes (00, 01) and one 3-bit code (100).
        let mut counts = [0u8; 16];
        counts[1] = 2;
        counts[2] = 1;
        let table = Huffman::new(&counts, vec![7, 8, 9]).unwrap();
        let mut bits = Bits { data: vec![0b0001_1000], pos: 0 };
        assert_eq!(table.decode(&mut bits).unwrap(), 7);
        assert_eq!(table.decode(&mut bits).unwrap(), 8);
        assert_eq!(table.decode(&mut bits).unwrap(), 9);
    }

    #[test]
    fn exif_orientation_reads_ifd0_and_defaults_to_one() {
        let mut jpeg = vec![0xff, 0xd8, 0xff, 0xe1];
        let mut app1 = b"Exif\0\0II*\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x06\0\0\0".to_vec();
        jpeg.extend_from_slice(&((app1.len() + 2) as u16).to_be_bytes());
        jpeg.append(&mut app1);
        assert_eq!(exif_orientation(&jpeg), 6);
        assert_eq!(exif_orientation(&[0xff, 0xd8, 0xff, 0xd9]), 1);
    }

    #[test]
    fn orientation_six_rotates_clockwise() {
        // 2x1 image [A B] shown rotated 90 CW becomes 1x2 [A; B].
        let image = Rgb8Image { width: 2, height: 1, rgb: vec![1, 1, 1, 2, 2, 2] };
        let out = orient(image, 6);
        assert_eq!((out.width, out.height), (1, 2));
        assert_eq!(out.rgb, vec![1, 1, 1, 2, 2, 2]);
    }
}
