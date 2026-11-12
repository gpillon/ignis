# Testing structured output

## The grammar (`crates/grammar`, GitHub #585, spec #287, ADR 0089)

`ignis-grammar` is CPU-only and runs in the default suite. What a test holds
is what a caller sees: a token checked, a run accepted up to its first miss,
the permitted set where a matcher stands, a compile or its named refusal,
never how the automaton stores its states.

- **Unit tests** (`cargo test -p ignis-grammar --lib`, about 5 s), on a toy
  byte-level vocabulary: the automaton against hand-written documents; the
  permitted set against its reference (`Matcher::mask_reference`, the check
  run on every column) on every state of random documents and on the rare
  string states (inside a `\u` escape, after an unanchored pattern matched); a
  completion from every state of random walks and of walks that nest as deep
  as the stack allows (no dead end); every refusal with the keyword or limit it
  names (`schema::tests::refusals`, which `every_refusal_code_is_pinned` holds
  to the source); every keyword of the strict subset compiling; what a schema
  without `strict` enforces and names as inert; the grammar cache (a schema
  sent again is not compiled again, memory bounded).
- **The real tokenizer** (`cargo test -p ignis-grammar --test real_tokenizer`,
  about 5 s): machine-local, skips when the 27B artifact is not at
  `F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer`. The byte table
  equals the tokenizer's decode; on each of the corpus's 19 cases
  (`crates/grammar/tests/corpus/cases.json`) the permitted set equals the
  reference on at least 40 states (1,953 in all), every document's next token
  is in it, valid documents are accepted token by token and violations
  refused.
- **`token_bytes_table`** itself: `cargo test -p ignis-artifact --lib
  frontend::tests::token_bytes_table`.

## The bench and the independent validator

The timings #585 bounds, and the cross-check of every verdict against
`jsonschema` (Draft 2020-12, format checking, numbers as exact decimals). It
needs the 27B artifact; it takes about a minute.

```sh
cargo build --release -p ignis-grammar --example grammar_bench
target/x86_64-pc-windows-msvc/release/examples/grammar_bench.exe \
    F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer \
    crates/grammar/tests/corpus/cases.json .scratch/grammar-bench
python -m venv .scratch/grammar-venv
.scratch/grammar-venv/Scripts/pip install -r tools/grammar-check/requirements.txt
.scratch/grammar-venv/Scripts/python -I tools/grammar-check/validate.py .scratch/grammar-bench/crosscheck.jsonl
```

(On Linux the venv's tools are under `bin/`, and the bench under
`target/release/examples/`.) The bench prints each figure beside its bound
(`bounds (#585)`, every line `met`), `reference states N, mismatches 0` and
`failures: 0`; `summary.json` has the numbers. The validator exits non-zero on
any disagreement: a document the grammar accepted that the standard refuses,
or a corpus document it refused that the standard accepts, except the
`outside_subset` ones (keys out of schema order, valid but never generated).
The measured figures are in
[the finding](../../findings/2026-11-12-structured-output-grammar-measured-on-the-27b.md).
