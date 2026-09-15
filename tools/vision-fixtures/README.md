# Vision processor fixtures (GitHub #176)

One-off tooling that records what the reference processor (ninfer
`targets/qwen3_6/impl/frontend/processor.cpp`) prepares for a small image set,
so `crates/artifact/tests/vision_processor.rs` can hold ignis to it exactly:
token ids, token types, positions, `rope_delta`, grids, token spans, content
digest, the SHA-256 of the packed BF16 patch rows, and the rewrite-checkpoint
frontier. Each fixture also records the reference's decoded RGB8 digest, so a
patch mismatch can be attributed to decode or to resize/pack.

Nothing here needs the GPU.

| File | What |
|------|------|
| `make_images.py` | Generates the committed image set (`crates/artifact/tests/fixtures/vision/images`). Pillow + numpy. |
| `cases.json` | The cases: messages, image files, thinking options. |
| `record.cpp` | The recorder: links the reference's `build-ninja` static libraries and runs its frontend on each case. |
| `build.ps1` | Builds `record.exe` with MSVC against `F:\ai\q38\ninfer\build-ninja`. |
| `crates/artifact/examples/dump_frontend.rs` | Writes an artifact's six frontend resources to a directory for the recorder. |

## Re-record

```powershell
$Artifact = "F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer"
$Frontend = "$env:TEMP\ignis-vision-frontend"
cargo run -p ignis-artifact --example dump_frontend -- $Artifact $Frontend
python tools/vision-fixtures/make_images.py crates/artifact/tests/fixtures/vision/images
powershell -NoProfile -ExecutionPolicy Bypass -File tools/vision-fixtures/build.ps1
& "$env:TEMP\ignis-vision-recorder\record.exe" $Frontend tools/vision-fixtures/cases.json `
  crates/artifact/tests/fixtures/vision/images crates/artifact/tests/fixtures/vision/expected
cargo test -p ignis-artifact --test vision_processor
```

`make_images.py` output depends on the Pillow version (recorded with 11.1.0);
re-record after regenerating images, since the fixtures pin the image bytes.

## Notes

- A multi-turn conversation is not in the set: ignis renders history
  assistant turns with an empty `<think></think>` block where the reference
  does not, a text-path divergence independent of vision (GitHub #182).
- The JPEG cases cover the path the processor ports exactly (8-bit YCbCr
  4:2:0/4:2:2, even height, baseline/progressive/restart intervals). Other
  JPEG layouts and lossy WebP decode best-effort; see `crates/artifact/src/vision/jpeg.rs`
  and `decode.rs`.
