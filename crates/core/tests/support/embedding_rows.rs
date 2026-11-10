//! The rows Flash-Next's embedding writes for text tokens, read off the
//! artifact: the columns a test places over an image's placeholders where a
//! text's own embedding would stand (`FlashNextEngine::place_media`). Included
//! by path from the GPU tests that place them (GitHub #557's placed columns,
//! #564's drift experiment).

// Each including test takes what it needs.
#![allow(dead_code)]

use ignis_artifact::{NumericFormat, Object, Reader};

fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// `v` rounded to BF16, to nearest even: CUDA's `__float2bfloat16`.
fn to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    if v.is_nan() {
        return 0x7FC0;
    }
    ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
}

/// An FP8 E4M3FN code's value (bias 7, subnormals below exponent 1, no
/// infinities; 0x7F and 0xFF are NaN).
fn e4m3(code: u8) -> f32 {
    let sign = if code & 0x80 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((code >> 3) & 0x0F);
    let mantissa = f32::from(code & 0x07);
    if code & 0x7F == 0x7F {
        return f32::NAN;
    }
    sign * if exponent == 0 { mantissa / 8.0 * 2f32.powi(-6) } else { (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7) }
}

/// The embedding rows of `tokens` as the leaf's embedding writes them into
/// every stream (`fn_embed`): an FP8 row-scale table decoded as
/// bf16(e4m3(code) * scale), a BF16 table copied. BF16 bits, `[tokens][hidden]`:
/// the columns a placeholder of each token's place takes. The table is read
/// through `reader`'s mapping, only the rows asked for.
pub fn embedding_rows(reader: &Reader, tokens: &[u32]) -> Vec<u16> {
    let name = "embed_tokens.weight";
    let Some(Object::Tensor(tensor)) = reader.find(name) else { panic!("the artifact has no tensor {name}") };
    let (rows, hidden) = (tensor.shape[0] as usize, tensor.shape[1] as usize);
    let data = reader.payload(name).unwrap_or_else(|e| panic!("{name}'s payload: {e:?}")).data;
    let mut out = Vec::with_capacity(tokens.len() * hidden);
    for &token in tokens {
        let token = token as usize;
        assert!(token < rows, "token {token} past the table's {rows} rows");
        match tensor.format {
            NumericFormat::Bf16 => {
                let row = &data[token * hidden * 2..(token + 1) * hidden * 2];
                out.extend(row.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])));
            }
            NumericFormat::Fp8E4M3FnRowBf16S => {
                let geometry = ignis_artifact::row_scale_geometry(tensor.format, &tensor.shape).expect("row-scale geometry");
                let at = geometry.scale_plane_offset as usize + token * 2;
                let scale = bf16(u16::from_le_bytes([data[at], data[at + 1]]));
                out.extend(data[token * hidden..(token + 1) * hidden].iter().map(|&code| to_bf16(e4m3(code) * scale)));
            }
            format => panic!("{name} is {format:?}, which this test does not decode"),
        }
    }
    out
}
