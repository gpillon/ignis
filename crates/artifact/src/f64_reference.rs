//! CPU `f64` layer oracle for the Qwen 3.8-27B text backbone (P1-20).
//!
//! This module deliberately reads the artifact's *stored* weights rather than
//! the device materialization or a kernel-side decode path.  It is therefore
//! an independent, slow correctness oracle for P1-21/P1-22, not a CPU
//! implementation intended for serving.

use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};

use crate::{
    block_scale_geometry, fail, row_split_geometry, NumericFormat, Object, Reader, Result,
    StorageLayout, TensorDescriptor,
};

pub const HIDDEN: usize = 5120;
pub const GQA_HEADS: usize = 24;
pub const GQA_KV_HEADS: usize = 4;
pub const GQA_HEAD_DIM: usize = 256;
pub const GDN_QK_HEADS: usize = 16;
pub const GDN_VALUE_HEADS: usize = 48;
pub const GDN_HEAD_DIM: usize = 128;
pub const MAX_TOKENS: usize = 4;
const EPS: f64 = 1.0e-6;

/// The two decoder-layer kinds in the text backbone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Gqa,
    Gdn,
}

/// A residual stream and positions fed to one layer oracle.
///
/// Values are token-major: `[token][hidden]`.  The position is the scalar
/// text position used for all three MRoPE axes.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerInput {
    pub residual: Vec<f64>,
    pub positions: Vec<i32>,
}

impl LayerInput {
    pub fn new(residual: Vec<f64>, positions: Vec<i32>) -> Result<Self> {
        if positions.is_empty() || positions.len() > MAX_TOKENS {
            return Err(fail("layer reference requires 1..=4 tokens"));
        }
        if residual.len() != positions.len() * HIDDEN {
            return Err(fail(
                "layer reference residual is not token-major [T, 5120]",
            ));
        }
        if residual.iter().any(|v| !v.is_finite()) {
            return Err(fail("layer reference residual contains a non-finite value"));
        }
        Ok(Self {
            residual,
            positions,
        })
    }
    pub fn tokens(&self) -> usize {
        self.positions.len()
    }
}

/// The externally useful result: the layer's residual stream after attention
/// or GDN *and* its MLP tail.  It is token-major `[T, 5120]`.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerOutput {
    pub residual: Vec<f64>,
}

/// An on-disk hand-off fixture for P1-21/P1-22.
///
/// The format is intentionally a small binary envelope, rather than JSON:
/// f64 output vectors are large and JSON would add both substantial size and
/// an accidental decimal-rounding boundary.  All integer fields are LE.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerFixture {
    pub gqa_layer: u32,
    pub gdn_layer: u32,
    pub gqa_input: LayerInput,
    pub gqa_output: LayerOutput,
    pub gdn_input: LayerInput,
    pub gdn_output: LayerOutput,
}

impl LayerFixture {
    const MAGIC: [u8; 8] = *b"IGLREF01";

    pub fn write_to(&self, path: &Path) -> Result<()> {
        let mut out =
            File::create(path).map_err(|e| fail(format!("create {}: {e}", path.display())))?;
        out.write_all(&Self::MAGIC)
            .map_err(|e| fail(format!("write {}: {e}", path.display())))?;
        out.write_all(&self.gqa_layer.to_le_bytes())
            .map_err(io_fail)?;
        out.write_all(&self.gdn_layer.to_le_bytes())
            .map_err(io_fail)?;
        write_case(&mut out, &self.gqa_input, &self.gqa_output)?;
        write_case(&mut out, &self.gdn_input, &self.gdn_output)
    }

    pub fn read_from(path: &Path) -> Result<Self> {
        let mut input =
            File::open(path).map_err(|e| fail(format!("open {}: {e}", path.display())))?;
        let mut magic = [0; 8];
        input.read_exact(&mut magic).map_err(io_fail)?;
        if magic != Self::MAGIC {
            return Err(fail(
                "not an ignis f64 layer-reference fixture (expected IGLREF01)",
            ));
        }
        let gqa_layer = read_u32(&mut input)?;
        let gdn_layer = read_u32(&mut input)?;
        let (gqa_input, gqa_output) = read_case(&mut input)?;
        let (gdn_input, gdn_output) = read_case(&mut input)?;
        let mut trailing = [0; 1];
        if input.read(&mut trailing).map_err(io_fail)? != 0 {
            return Err(fail("layer-reference fixture has trailing bytes"));
        }
        Ok(Self {
            gqa_layer,
            gdn_layer,
            gqa_input,
            gqa_output,
            gdn_input,
            gdn_output,
        })
    }
}

fn io_fail(error: std::io::Error) -> crate::ArtifactError {
    fail(error)
}
fn write_case(out: &mut File, input: &LayerInput, output: &LayerOutput) -> Result<()> {
    if output.residual.len() != input.residual.len() {
        return Err(fail("layer fixture input/output token counts differ"));
    }
    out.write_all(&(input.tokens() as u32).to_le_bytes())
        .map_err(io_fail)?;
    for p in &input.positions {
        out.write_all(&p.to_le_bytes()).map_err(io_fail)?;
    }
    for value in input.residual.iter().chain(output.residual.iter()) {
        out.write_all(&value.to_le_bytes()).map_err(io_fail)?;
    }
    Ok(())
}
fn read_u32(input: &mut File) -> Result<u32> {
    let mut bytes = [0; 4];
    input.read_exact(&mut bytes).map_err(io_fail)?;
    Ok(u32::from_le_bytes(bytes))
}
fn read_case(input: &mut File) -> Result<(LayerInput, LayerOutput)> {
    let tokens = read_u32(input)? as usize;
    if tokens == 0 || tokens > MAX_TOKENS {
        return Err(fail("fixture token count is outside 1..=4"));
    }
    let mut positions = Vec::with_capacity(tokens);
    for _ in 0..tokens {
        let mut b = [0; 4];
        input.read_exact(&mut b).map_err(io_fail)?;
        positions.push(i32::from_le_bytes(b));
    }
    let mut numbers = vec![0f64; tokens * HIDDEN * 2];
    for value in &mut numbers {
        let mut b = [0; 8];
        input.read_exact(&mut b).map_err(io_fail)?;
        *value = f64::from_le_bytes(b);
    }
    let output = numbers.split_off(tokens * HIDDEN);
    Ok((
        LayerInput::new(numbers, positions)?,
        LayerOutput { residual: output },
    ))
}

/// The public seam for the oracle: evaluate exactly one layer from an opened
/// artifact and a supplied residual stream.  `layer` uses the artifact's
/// zero-based numbering; GQA layers are 3, 7, ..., 63.
pub fn evaluate_layer(reader: &Reader, layer: usize, input: &LayerInput) -> Result<LayerOutput> {
    let kind = if (layer + 1) % 4 == 0 {
        LayerKind::Gqa
    } else {
        LayerKind::Gdn
    };
    let prefix = format!("text/layers/{layer}/");
    let common = CommonWeights::load(reader, &prefix)?;
    match kind {
        LayerKind::Gqa => evaluate_gqa(reader, &prefix, common, input),
        LayerKind::Gdn => evaluate_gdn(reader, &prefix, common, input),
    }
}

struct Matrix<'a> {
    rows: usize,
    cols: usize,
    format: NumericFormat,
    layout: StorageLayout,
    payload: &'a [u8],
}
impl<'a> Matrix<'a> {
    fn load(reader: &'a Reader, name: &str, rows: usize, cols: usize) -> Result<Self> {
        let Object::Tensor(TensorDescriptor {
            format,
            layout,
            shape,
            ..
        }) = reader
            .find(name)
            .ok_or_else(|| fail(format!("missing layer weight: {name}")))?
        else {
            return Err(fail(format!("layer weight is not a tensor: {name}")));
        };
        if shape.as_slice() != [rows as u64, cols as u64] {
            return Err(fail(format!("unexpected shape for {name}: {shape:?}")));
        }
        let payload = reader.payload(name)?.data;
        Ok(Self {
            rows,
            cols,
            format: *format,
            layout: *layout,
            payload,
        })
    }
    fn load_vector(reader: &'a Reader, name: &str, len: usize) -> Result<Self> {
        let Object::Tensor(TensorDescriptor {
            format,
            layout,
            shape,
            ..
        }) = reader
            .find(name)
            .ok_or_else(|| fail(format!("missing layer weight: {name}")))?
        else {
            return Err(fail(format!("layer weight is not a tensor: {name}")));
        };
        if shape.as_slice() != [len as u64] {
            return Err(fail(format!("unexpected shape for {name}: {shape:?}")));
        }
        Ok(Self {
            rows: 1,
            cols: len,
            format: *format,
            layout: *layout,
            payload: reader.payload(name)?.data,
        })
    }
    fn at(&self, row: usize, col: usize) -> Result<f64> {
        if row >= self.rows || col >= self.cols {
            return Err(fail("host matrix coordinate is out of bounds"));
        }
        match (self.format, self.layout) {
            (NumericFormat::Bf16, StorageLayout::ContiguousLeV1) => {
                Ok(bf16(self.word(row * self.cols + col)))
            }
            (NumericFormat::Fp32, StorageLayout::ContiguousLeV1) => {
                let i = (row * self.cols + col) * 4;
                Ok(f32::from_le_bytes(self.payload[i..i + 4].try_into().unwrap()) as f64)
            }
            (NumericFormat::Nvfp4, StorageLayout::BlockScaleK16M128x4V1) => self.nvfp4(row, col),
            (NumericFormat::W8G32F16S, StorageLayout::RowSplitK128V1) => self.w8(row, col),
            _ => Err(fail(
                "f64 layer oracle supports BF16, FP32, NVFP4, and W8G32 host matrices only",
            )),
        }
    }
    fn word(&self, index: usize) -> u16 {
        u16::from_le_bytes(self.payload[index * 2..index * 2 + 2].try_into().unwrap())
    }
    fn nvfp4(&self, row: usize, col: usize) -> Result<f64> {
        let shape = [self.rows as u64, self.cols as u64];
        let g = block_scale_geometry(NumericFormat::Nvfp4, &shape)?;
        let code = self.payload[row * self.cols / 2 + col / 2];
        let nibble = if col & 1 == 0 { code & 15 } else { code >> 4 };
        let group = col / 16;
        let tiles = self.cols / 64;
        let inner = row % 128;
        let scale = (row / 128 * tiles + group / 4) * 512
            + (inner % 32) * 16
            + (inner / 32) * 4
            + group % 4;
        let divisor_at = g.weight_divisor_offset as usize;
        let divisor =
            f32::from_le_bytes(self.payload[divisor_at..divisor_at + 4].try_into().unwrap()) as f64;
        if !divisor.is_finite() || divisor <= 0.0 {
            return Err(fail("NVFP4 weight divisor is not positive and finite"));
        }
        Ok(e2m1(nibble) * e4m3(self.payload[g.scale_plane_offset as usize + scale]) / divisor)
    }
    fn w8(&self, row: usize, col: usize) -> Result<f64> {
        let g = row_split_geometry(
            NumericFormat::W8G32F16S,
            &[self.rows as u64, self.cols as u64],
        )?;
        let group = row * g.groups_per_row as usize + col / 32;
        let code = self.payload[group * 32 + col % 32] as i8 as f64;
        let at = g.scale_plane_offset as usize + group * 2;
        Ok(code
            * f16(u16::from_le_bytes(
                self.payload[at..at + 2].try_into().unwrap(),
            )))
    }
    fn product(&self, input: &[f64]) -> Result<Vec<f64>> {
        // Below this much work a thread costs more than the rows it saves,
        // and the oracle calls this for small projections too.
        const SERIAL_BELOW_ELEMENTS: usize = 1 << 16;
        let threads = if self.rows.saturating_mul(self.cols) < SERIAL_BELOW_ELEMENTS {
            1
        } else {
            std::thread::available_parallelism().map_or(1, |n| n.get())
        };
        self.product_with_threads(input, threads)
    }
    /// One matrix-vector product, its output rows spread over `threads`.
    ///
    /// Every row is summed by the same code in the same order regardless of
    /// the split, so the schedule cannot move a bit of the result (GitHub
    /// #135) -- an oracle carrying a measured tolerance may not depend on
    /// the core count it happened to run on.
    fn product_with_threads(&self, input: &[f64], threads: usize) -> Result<Vec<f64>> {
        if input.len() != self.cols {
            return Err(fail(
                "host matrix product input width differs from weight K",
            ));
        }
        let mut out = vec![0.; self.rows];
        match (self.format, self.layout) {
            (NumericFormat::Bf16, StorageLayout::ContiguousLeV1) => {
                fill_rows(&mut out, threads, |row| {
                    let words = &self.payload[row * self.cols * 2..(row + 1) * self.cols * 2];
                    words
                        .chunks_exact(2)
                        .zip(input)
                        .map(|(word, x)| bf16(u16::from_le_bytes(word.try_into().unwrap())) * x)
                        .sum()
                });
            }
            (NumericFormat::Nvfp4, StorageLayout::BlockScaleK16M128x4V1) => {
                let shape = [self.rows as u64, self.cols as u64];
                let g = block_scale_geometry(NumericFormat::Nvfp4, &shape)?;
                let divisor_at = g.weight_divisor_offset as usize;
                let divisor = f32::from_le_bytes(
                    self.payload[divisor_at..divisor_at + 4].try_into().unwrap(),
                ) as f64;
                if !divisor.is_finite() || divisor <= 0.0 {
                    return Err(fail("NVFP4 weight divisor is not positive and finite"));
                }
                let tiles = self.cols / 64;
                let scales = &self.payload[g.scale_plane_offset as usize
                    ..g.scale_plane_offset as usize + g.scale_plane_bytes as usize];
                fill_rows(&mut out, threads, |row| {
                    let codes = &self.payload[row * self.cols / 2..(row + 1) * self.cols / 2];
                    let inner = row % 128;
                    let row_tile = row / 128;
                    let mut total = 0.;
                    for group in 0..self.cols / 16 {
                        let scale_at = (row_tile * tiles + group / 4) * 512
                            + (inner % 32) * 16
                            + (inner / 32) * 4
                            + group % 4;
                        let scale = e4m3(scales[scale_at]) / divisor;
                        for within in 0..16 {
                            let column = group * 16 + within;
                            let byte = codes[column / 2];
                            let code = if column & 1 == 0 {
                                byte & 15
                            } else {
                                byte >> 4
                            };
                            total += e2m1(code) * scale * input[column];
                        }
                    }
                    total
                });
            }
            (NumericFormat::W8G32F16S, StorageLayout::RowSplitK128V1) => {
                let g = row_split_geometry(
                    NumericFormat::W8G32F16S,
                    &[self.rows as u64, self.cols as u64],
                )?;
                let scales = &self.payload[g.scale_plane_offset as usize..];
                fill_rows(&mut out, threads, |row| {
                    let mut total = 0.;
                    for group in 0..self.cols / 32 {
                        let scale = f16(u16::from_le_bytes(
                            scales[(row * g.groups_per_row as usize + group) * 2
                                ..(row * g.groups_per_row as usize + group) * 2 + 2]
                                .try_into()
                                .unwrap(),
                        ));
                        let code_start = (row * g.groups_per_row as usize + group) * 32;
                        for within in 0..32 {
                            total += self.payload[code_start + within] as i8 as f64
                                * scale
                                * input[group * 32 + within];
                        }
                    }
                    total
                });
            }
            _ => {
                return Err(fail(
                    "f64 layer oracle supports BF16, FP32, NVFP4, and W8G32 host matrices only",
                ))
            }
        }
        Ok(out)
    }
}

/// Fill every element of `out` with `row_value(index)`, spreading the work
/// over at most `threads` scoped threads.
///
/// Rows are independent, so this only chooses *who* computes a row, never
/// *how*: `row_value` sees the same index and runs the same arithmetic in
/// the same order whatever the split (GitHub #135). A matrix narrower than
/// the thread count simply uses fewer threads, and an empty one spawns none.
fn fill_rows<F>(out: &mut [f64], threads: usize, row_value: F)
where
    F: Fn(usize) -> f64 + Sync,
{
    let rows = out.len();
    if rows == 0 {
        return;
    }
    let threads = threads.clamp(1, rows);
    if threads == 1 {
        for (row, target) in out.iter_mut().enumerate() {
            *target = row_value(row);
        }
        return;
    }
    let chunk = rows.div_ceil(threads);
    let row_value = &row_value;
    std::thread::scope(|scope| {
        for (index, slice) in out.chunks_mut(chunk).enumerate() {
            scope.spawn(move || {
                let base = index * chunk;
                for (offset, target) in slice.iter_mut().enumerate() {
                    *target = row_value(base + offset);
                }
            });
        }
    });
}

struct CommonWeights<'a> {
    input_norm: Matrix<'a>,
    post_norm: Matrix<'a>,
    gate_up: Matrix<'a>,
    down: Matrix<'a>,
}
impl<'a> CommonWeights<'a> {
    fn load(reader: &'a Reader, prefix: &str) -> Result<Self> {
        Ok(Self {
            input_norm: Matrix::load_vector(reader, &format!("{prefix}input_norm"), HIDDEN)?,
            post_norm: Matrix::load_vector(
                reader,
                &format!("{prefix}post_attention_norm"),
                HIDDEN,
            )?,
            gate_up: Matrix::load(reader, &format!("{prefix}mlp/gate_up"), 34816, HIDDEN)?,
            down: Matrix::load(reader, &format!("{prefix}mlp/down"), HIDDEN, 17408)?,
        })
    }
}

fn vector(weight: &Matrix<'_>) -> Result<Vec<f64>> {
    if weight.rows != 1 {
        return Err(fail("expected a vector weight"));
    }
    (0..weight.cols).map(|i| weight.at(0, i)).collect()
}
fn norm(input: &[f64], weight: &[f64], unit_offset: bool) -> Result<Vec<f64>> {
    if input.len() != weight.len() {
        return Err(fail("RMSNorm input and weight widths differ"));
    }
    let inv = 1.0 / ((input.iter().map(|x| x * x).sum::<f64>() / input.len() as f64) + EPS).sqrt();
    Ok(input
        .iter()
        .zip(weight)
        .map(|(x, w)| x * inv * (if unit_offset { 1.0 + w } else { *w }))
        .collect())
}
fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
fn mlp(common: &CommonWeights<'_>, residual: &mut [f64]) -> Result<()> {
    let normalized = norm(residual, &vector(&common.post_norm)?, true)?;
    let gates = common.gate_up.product(&normalized)?;
    let mut fused = vec![0.; 17408];
    for i in 0..17408 {
        fused[i] = silu(gates[i]) * gates[17408 + i];
    }
    let down = common.down.product(&fused)?;
    for (x, add) in residual.iter_mut().zip(down) {
        *x += add;
    }
    Ok(())
}

fn evaluate_gqa(
    reader: &Reader,
    prefix: &str,
    common: CommonWeights<'_>,
    input: &LayerInput,
) -> Result<LayerOutput> {
    let projection = Matrix::load(
        reader,
        &format!("{prefix}attention/query_key_gate_value"),
        14336,
        HIDDEN,
    )?;
    let q_norm = Matrix::load_vector(
        reader,
        &format!("{prefix}attention/query_norm"),
        GQA_HEAD_DIM,
    )?;
    let k_norm = Matrix::load_vector(reader, &format!("{prefix}attention/key_norm"), GQA_HEAD_DIM)?;
    let output = Matrix::load(
        reader,
        &format!("{prefix}attention/output"),
        HIDDEN,
        GQA_HEADS * GQA_HEAD_DIM,
    )?;
    let input_norm = vector(&common.input_norm)?;
    let q_weight = vector(&q_norm)?;
    let k_weight = vector(&k_norm)?;
    let t = input.tokens();
    let mut qs = vec![vec![0.; GQA_HEADS * GQA_HEAD_DIM]; t];
    let mut ks = vec![vec![0.; GQA_KV_HEADS * GQA_HEAD_DIM]; t];
    let mut vs = ks.clone();
    let mut gates = qs.clone();
    for token in 0..t {
        let h = norm(
            &input.residual[token * HIDDEN..(token + 1) * HIDDEN],
            &input_norm,
            true,
        )?;
        let p = projection.product(&h)?;
        qs[token].copy_from_slice(&p[..6144]);
        ks[token].copy_from_slice(&p[6144..7168]);
        gates[token].copy_from_slice(&p[7168..13312]);
        vs[token].copy_from_slice(&p[13312..]);
        for head in 0..GQA_HEADS {
            let start = head * GQA_HEAD_DIM;
            let prepared = rope(
                &norm(&qs[token][start..start + GQA_HEAD_DIM], &q_weight, true)?,
                input.positions[token],
            );
            qs[token][start..start + GQA_HEAD_DIM].copy_from_slice(&prepared);
        }
        for head in 0..GQA_KV_HEADS {
            let start = head * GQA_HEAD_DIM;
            let prepared = rope(
                &norm(&ks[token][start..start + GQA_HEAD_DIM], &k_weight, true)?,
                input.positions[token],
            );
            ks[token][start..start + GQA_HEAD_DIM].copy_from_slice(&prepared);
        }
    }
    let mut final_residual = input.residual.clone();
    for token in 0..t {
        let mut attention = vec![0.; GQA_HEADS * GQA_HEAD_DIM];
        for head in 0..GQA_HEADS {
            let kv = head / (GQA_HEADS / GQA_KV_HEADS);
            let q = &qs[token][head * GQA_HEAD_DIM..(head + 1) * GQA_HEAD_DIM];
            let mut scores = Vec::with_capacity(token + 1);
            for previous in 0..=token {
                let k = &ks[previous][kv * GQA_HEAD_DIM..(kv + 1) * GQA_HEAD_DIM];
                scores.push(q.iter().zip(k).map(|(a, b)| a * b).sum::<f64>() / 16.0);
            }
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let denom = scores.iter().map(|s| (s - max).exp()).sum::<f64>();
            let out = &mut attention[head * GQA_HEAD_DIM..(head + 1) * GQA_HEAD_DIM];
            for (previous, score) in scores.iter().enumerate() {
                let scale = (*score - max).exp() / denom;
                let v = &vs[previous][kv * GQA_HEAD_DIM..(kv + 1) * GQA_HEAD_DIM];
                for (target, value) in out.iter_mut().zip(v) {
                    *target += scale * value;
                }
            }
            for (value, gate) in out
                .iter_mut()
                .zip(&gates[token][head * GQA_HEAD_DIM..(head + 1) * GQA_HEAD_DIM])
            {
                *value *= sigmoid(*gate);
            }
        }
        let add = output.product(&attention)?;
        let residual = &mut final_residual[token * HIDDEN..(token + 1) * HIDDEN];
        for (x, a) in residual.iter_mut().zip(add) {
            *x += a;
        }
        mlp(&common, residual)?;
    }
    Ok(LayerOutput {
        residual: final_residual,
    })
}

fn rope(values: &[f64], position: i32) -> Vec<f64> {
    let mut out = values.to_vec();
    for pair in 0..32 {
        let angle = position as f64 * 10_000_000f64.powf(-(2.0 * pair as f64 / 64.0));
        let (s, c) = angle.sin_cos();
        let a = values[pair];
        let b = values[pair + 32];
        out[pair] = a * c - b * s;
        out[pair + 32] = b * c + a * s;
    }
    out
}

fn evaluate_gdn(
    reader: &Reader,
    prefix: &str,
    common: CommonWeights<'_>,
    input: &LayerInput,
) -> Result<LayerOutput> {
    let a_log = Matrix::load_vector(reader, &format!("{prefix}gdn/a_log"), GDN_VALUE_HEADS)?;
    let dt_bias = Matrix::load_vector(reader, &format!("{prefix}gdn/dt_bias"), GDN_VALUE_HEADS)?;
    let convolution = Matrix::load(reader, &format!("{prefix}gdn/convolution"), 4, 10240)?;
    let ab = Matrix::load(reader, &format!("{prefix}gdn/a_b_projection"), 96, HIDDEN)?;
    let projection = Matrix::load(
        reader,
        &format!("{prefix}gdn/query_key_value_z"),
        16384,
        HIDDEN,
    )?;
    let gdn_norm = Matrix::load_vector(reader, &format!("{prefix}gdn/norm"), GDN_HEAD_DIM)?;
    let output = Matrix::load(reader, &format!("{prefix}gdn/output"), HIDDEN, 6144)?;
    let input_norm = vector(&common.input_norm)?;
    let a_log = vector(&a_log)?;
    let dt_bias = vector(&dt_bias)?;
    let gdn_norm = vector(&gdn_norm)?;
    let t = input.tokens();
    let mut projected = vec![vec![0.; 10240]; t];
    let mut z = vec![vec![0.; 6144]; t];
    let mut controls = vec![vec![0.; 96]; t];
    for token in 0..t {
        let h = norm(
            &input.residual[token * HIDDEN..(token + 1) * HIDDEN],
            &input_norm,
            true,
        )?;
        let p = projection.product(&h)?;
        projected[token].copy_from_slice(&p[..10240]);
        z[token].copy_from_slice(&p[10240..]);
        controls[token] = ab.product(&h)?;
    }
    let mut qkv = vec![vec![0.; 10240]; t];
    for token in 0..t {
        for channel in 0..10240 {
            let mut value = 0.;
            for tap in 0..4 {
                if token + tap >= 3 {
                    value += convolution.at(tap, channel)? * projected[token + tap - 3][channel];
                }
            }
            qkv[token][channel] = silu(value);
        }
    }
    let mut state = vec![0.; GDN_VALUE_HEADS * GDN_HEAD_DIM * GDN_HEAD_DIM];
    let mut recurrent = vec![vec![0.; 6144]; t];
    for token in 0..t {
        for head in 0..GDN_VALUE_HEADS {
            let qh = head / 3;
            let q = &qkv[token][qh * GDN_HEAD_DIM..(qh + 1) * GDN_HEAD_DIM];
            let k = &qkv[token][2048 + qh * GDN_HEAD_DIM..2048 + (qh + 1) * GDN_HEAD_DIM];
            let v = &qkv[token][4096 + head * GDN_HEAD_DIM..4096 + (head + 1) * GDN_HEAD_DIM];
            let q = unit(q);
            let k = unit(k);
            let alpha =
                (-a_log[head].exp() * softplus(controls[token][head] + dt_bias[head])).exp();
            let beta = sigmoid(controls[token][48 + head]);
            let base = head * GDN_HEAD_DIM * GDN_HEAD_DIM;
            let mut delta = [0.; GDN_HEAD_DIM];
            for row in 0..GDN_HEAD_DIM {
                let dot = (0..GDN_HEAD_DIM)
                    .map(|col| state[base + row * GDN_HEAD_DIM + col] * k[col])
                    .sum::<f64>();
                delta[row] = beta * (v[row] - alpha * dot);
            }
            for row in 0..GDN_HEAD_DIM {
                for col in 0..GDN_HEAD_DIM {
                    state[base + row * GDN_HEAD_DIM + col] =
                        alpha * state[base + row * GDN_HEAD_DIM + col] + delta[row] * k[col];
                }
                let dot = (0..GDN_HEAD_DIM)
                    .map(|col| state[base + row * GDN_HEAD_DIM + col] * q[col])
                    .sum::<f64>();
                recurrent[token][head * GDN_HEAD_DIM + row] = dot / (GDN_HEAD_DIM as f64).sqrt();
            }
        }
    }
    let mut final_residual = input.residual.clone();
    for token in 0..t {
        let mut gated = vec![0.; 6144];
        for head in 0..GDN_VALUE_HEADS {
            let range = head * GDN_HEAD_DIM..(head + 1) * GDN_HEAD_DIM;
            let normalized = norm(&recurrent[token][range.clone()], &gdn_norm, false)?;
            for d in 0..GDN_HEAD_DIM {
                gated[range.start + d] = normalized[d] * silu(z[token][range.start + d]);
            }
        }
        let add = output.product(&gated)?;
        let residual = &mut final_residual[token * HIDDEN..(token + 1) * HIDDEN];
        for (x, a) in residual.iter_mut().zip(add) {
            *x += a;
        }
        mlp(&common, residual)?;
    }
    Ok(LayerOutput {
        residual: final_residual,
    })
}

fn unit(values: &[f64]) -> Vec<f64> {
    let inv = 1.0 / (values.iter().map(|v| v * v).sum::<f64>() + EPS).sqrt();
    values.iter().map(|v| v * inv).collect()
}
fn softplus(value: f64) -> f64 {
    if value > 20. {
        value
    } else {
        (1.0 + value.exp()).ln()
    }
}
fn bf16(bits: u16) -> f64 {
    f32::from_bits((bits as u32) << 16) as f64
}
fn f16(bits: u16) -> f64 {
    // exact IEEE binary16 expansion
    let sign = if bits & 0x8000 == 0 { 1. } else { -1. };
    let exponent = (bits >> 10) & 31;
    let fraction = (bits & 1023) as f64;
    if exponent == 0 {
        sign * fraction * 2f64.powi(-24)
    } else if exponent == 31 {
        if fraction == 0. {
            sign * f64::INFINITY
        } else {
            f64::NAN
        }
    } else {
        sign * (1. + fraction / 1024.) * 2f64.powi(exponent as i32 - 15)
    }
}
fn e2m1(code: u8) -> f64 {
    const VALUES: [f64; 8] = [0., 0.5, 1., 1.5, 2., 3., 4., 6.];
    let value = VALUES[(code & 7) as usize];
    if code & 8 == 0 {
        value
    } else {
        -value
    }
}
fn e4m3(code: u8) -> f64 {
    let sign = if code & 128 == 0 { 1. } else { -1. };
    let exponent = (code >> 3) & 15;
    let fraction = (code & 7) as f64;
    if exponent == 0 {
        sign * fraction * 2f64.powi(-9)
    } else if exponent == 15 && fraction == 7. {
        f64::NAN
    } else {
        sign * (1. + fraction / 8.)
            * 2f64.powi(if exponent == 15 {
                8
            } else {
                exponent as i32 - 7
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn host_float_decoders_preserve_known_values() {
        assert_eq!(bf16(0x3fc0), 1.5);
        assert_eq!(f16(0x3e00), 1.5);
        assert_eq!(e2m1(0x7), 6.);
        assert_eq!(e2m1(0xf), -6.);
        assert_eq!(e4m3(0x38), 1.);
        assert_eq!(e4m3(0x52), 10.);
    }
    #[test]
    fn fixture_round_trip_preserves_f64_bits_and_positions() {
        let input = LayerInput::new(vec![0.; HIDDEN], vec![9]).unwrap();
        let output = LayerOutput {
            residual: vec![f64::from_bits(0x3ff0000000000001); HIDDEN],
        };
        let fixture = LayerFixture {
            gqa_layer: 3,
            gdn_layer: 4,
            gqa_input: input.clone(),
            gqa_output: output.clone(),
            gdn_input: input,
            gdn_output: output,
        };
        let path = std::env::temp_dir().join("ignis-layer-ref-round-trip.bin");
        fixture.write_to(&path).unwrap();
        assert_eq!(LayerFixture::read_from(&path).unwrap(), fixture);
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn gdn_unit_vector_is_normalized_with_epsilon() {
        let v = unit(&[3., 4.]);
        assert!((v[0] * v[0] + v[1] * v[1] - 1.).abs() < 1e-6);
    }

    /// Every thread split must agree with the serial one bit for bit.
    fn assert_schedule_is_bit_identical(what: &str, matrix: &Matrix<'_>) {
        let input: Vec<f64> = (0..matrix.cols).map(|i| (i as f64 * 0.37).sin()).collect();
        let serial = matrix
            .product_with_threads(&input, 1)
            .unwrap_or_else(|e| panic!("{what} serial product: {e}"));
        assert!(
            serial.iter().any(|value| *value != 0.),
            "{what}: degenerate payload proves nothing"
        );
        assert!(
            serial.iter().all(|value| value.is_finite()),
            "{what}: payload decodes to a non-finite value"
        );
        // Coprime with the row counts, so the last chunk is short, plus one
        // count far above the rows (every thread but the first gets nothing).
        for threads in [2, 3, 7, 64, 4096] {
            let parallel = matrix
                .product_with_threads(&input, threads)
                .unwrap_or_else(|e| panic!("{what} over {threads} threads: {e}"));
            assert_eq!(parallel.len(), serial.len(), "{what}: row count");
            for (row, (got, want)) in parallel.iter().zip(&serial).enumerate() {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{what}: row {row} differs over {threads} threads"
                );
            }
        }
    }

    /// GitHub #135: the oracle spreads its output rows over threads. Each row
    /// is still summed by the same code in the same order, so the schedule
    /// must not move a single bit -- an oracle whose value depends on the
    /// core count it happened to run on could not carry a measured tolerance.
    ///
    /// All three decode arms are covered. `layer_reference_real` also pins
    /// this against the committed fixtures, which were recorded before the
    /// rows were split, but it is `#[ignore]`d and needs the real artifact,
    /// so on its own it would leave the default CPU gate proving nothing
    /// about two of the three arms.
    #[test]
    fn the_row_schedule_does_not_change_a_single_bit_of_any_decode_arm() {
        // BF16: a flat little-endian plane, two bytes per element.
        const BF16_ROWS: usize = 131;
        const BF16_COLS: usize = 64;
        let bf16_payload: Vec<u8> = (0..BF16_ROWS * BF16_COLS * 2)
            .map(|i| ((i * 37 + i / 11) % 251) as u8)
            .collect();
        assert_schedule_is_bit_identical(
            "bf16",
            &Matrix {
                rows: BF16_ROWS,
                cols: BF16_COLS,
                format: NumericFormat::Bf16,
                layout: StorageLayout::ContiguousLeV1,
                payload: &bf16_payload,
            },
        );

        // NVFP4: a code plane, an E4M3 scale plane, then the FP32 divisor.
        // Rows must divide 128 and columns 64 for the geometry to exist.
        const NVFP4_ROWS: usize = 256;
        const NVFP4_COLS: usize = 64;
        let g = block_scale_geometry(
            NumericFormat::Nvfp4,
            &[NVFP4_ROWS as u64, NVFP4_COLS as u64],
        )
        .expect("nvfp4 geometry");
        let mut nvfp4_payload = vec![0u8; g.encoded_bytes as usize];
        for (i, byte) in nvfp4_payload[..g.code_plane_bytes as usize]
            .iter_mut()
            .enumerate()
        {
            *byte = ((i * 53 + i / 7) % 256) as u8;
        }
        // E4M3 codes near 1.0: an arbitrary byte here can decode to NaN,
        // which would make the comparison vacuous rather than wrong.
        let scale_at = g.scale_plane_offset as usize;
        for (i, byte) in nvfp4_payload[scale_at..scale_at + g.scale_plane_bytes as usize]
            .iter_mut()
            .enumerate()
        {
            *byte = 0x34 + (i % 9) as u8;
        }
        let divisor_at = g.weight_divisor_offset as usize;
        nvfp4_payload[divisor_at..divisor_at + 4].copy_from_slice(&1.5f32.to_le_bytes());
        assert_schedule_is_bit_identical(
            "nvfp4",
            &Matrix {
                rows: NVFP4_ROWS,
                cols: NVFP4_COLS,
                format: NumericFormat::Nvfp4,
                layout: StorageLayout::BlockScaleK16M128x4V1,
                payload: &nvfp4_payload,
            },
        );

        // W8G32: an i8 code plane and an F16 per-group scale plane.
        const W8_ROWS: usize = 67;
        const W8_COLS: usize = 128;
        let g = row_split_geometry(
            NumericFormat::W8G32F16S,
            &[W8_ROWS as u64, W8_COLS as u64],
        )
        .expect("w8 geometry");
        let mut w8_payload = vec![0u8; g.encoded_bytes as usize];
        for (i, byte) in w8_payload[..g.low_plane_bytes as usize]
            .iter_mut()
            .enumerate()
        {
            *byte = ((i * 29 + i / 5) % 256) as u8;
        }
        // F16 scales around 1.0 (0x3C00), for the same reason as above.
        let scale_at = g.scale_plane_offset as usize;
        for (group, pair) in w8_payload[scale_at..scale_at + g.scale_plane_bytes as usize]
            .chunks_exact_mut(2)
            .enumerate()
        {
            pair.copy_from_slice(&(0x3C00u16 + (group % 97) as u16).to_le_bytes());
        }
        assert_schedule_is_bit_identical(
            "w8g32",
            &Matrix {
                rows: W8_ROWS,
                cols: W8_COLS,
                format: NumericFormat::W8G32F16S,
                layout: StorageLayout::RowSplitK128V1,
                payload: &w8_payload,
            },
        );
    }

    /// A thread count above the row count must not panic on an empty chunk,
    /// and a zero-row matrix must not spawn anything.
    #[test]
    fn a_product_wider_than_its_row_count_is_still_well_formed() {
        let matrix = Matrix {
            rows: 0,
            cols: 4,
            format: NumericFormat::Bf16,
            layout: StorageLayout::ContiguousLeV1,
            payload: &[],
        };
        assert!(matrix
            .product_with_threads(&[1., 2., 3., 4.], 8)
            .expect("empty product")
            .is_empty());
    }
}
