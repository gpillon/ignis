# F64 layer-reference fixture

P1-20's CPU oracle exposes `ignis_artifact::f64_reference`. It evaluates a
single text decoder layer from the artifact's stored host bytes, with a
token-major `LayerInput` (`T` in `1..=4`, then `T * 5120` f64 residual values)
and scalar text positions. It returns the complete post-MLP residual stream.
P1-21 and P1-22 consume the same `LayerFixture` envelope rather than redoing
host decoding.

`LayerFixture` is binary so expected values cross no decimal serialization
boundary:

1. `IGLREF01` (8 ASCII bytes)
2. little-endian `u32` GQA layer index, then `u32` GDN layer index
3. a GQA case and a GDN case, each as: `u32 T`; `T` little-endian `i32`
   positions; `T * 5120` little-endian IEEE-754 f64 input words; then the
   same number of f64 output words.

The recorded coverage set is deliberately mixed-quant:

- GQA layer 27: NVFP4 attention input and output arms;
- GQA layer 3: BF16 attention input and output arms;
- GDN layer 4: the BF16 `gdn/output` arm (while its fused input remains
  NVFP4).

The main fixture selects one GQA case and the GDN case; a second fixture is
recorded for the other GQA arm. Inputs use two positions so the second output
is sensitive to the first token's causal-attention or GDN recurrent state.
