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

## The served path (`json_object` GitHub #587, `json_schema` #588)

- **Over the mock engine** (`cargo test -p ignis-server --test
  structured_output_http`, about 1 s): the `/v1` surface with a test
  vocabulary and a scripted model (`MockCompute::script`). The script misses
  on purpose -- a preamble, a fence, a remark after the object, an early stop,
  only whitespace -- and each answer comes back a valid object, streamed and
  whole alike; the reasoning stays unchecked with thinking on, and after a
  thinking budget's forced close the answer is held; cuts by `max_tokens` and
  by the deadline; the refusals (Flash-Next, a forced tool call). Against a
  strict schema the script writes key names of its own, an enum in the wrong
  case, numbers out of range, a missing key, an extra key, a string its
  pattern refuses and a date in words: each answer is checked by hand and by
  the independent validator below, and keys come in the schema's order (the
  schema reaches the grammar as sent, `serde_json::value::RawValue`). Every
  refusal the spec names is a 400 with the refusal's code, `param`
  `response_format` and the keyword or limit in the message, and the mock
  saw no prefill; a non-strict `minLength` is served and listed inert by the
  generated reference; a schema sent three times, its whitespace changed, is
  compiled once. The mock refuses a repair made after a later call, so a
  scheduler that checks a draw late fails here first. Its repair ranks a
  script's later tokens before the vocabulary in order, and the test
  vocabulary puts the tokens that close a value first.
- **The independent validator in the served tests**: the schema answers go
  through `tools/grammar-check/validate.py` with the Python of
  `IGNIS_GRAMMAR_CHECK_PYTHON`, else the venv at `.scratch/grammar-venv`
  (made as above). Where neither exists the validation is skipped with a
  line saying so, and the hand checks still run.
- **The generated reference** (`docs/user/reference/json-schema.md`,
  `user_docs::json_schema`): its lists come from the grammar crate's, and the
  page compiles each inert keyword (refused under `strict`, inert without)
  and a schema past each limit, so generating it fails if the grammar
  disagrees. `cargo test -p ignis-server --test user_docs_reference` holds
  the committed page to it.
- **The adapter** (`cargo test -p ignis-runtime --test runtime_compute
  a_checked_lane`): a checked lane reports its waiting draw, the EOS as a stop.
- **On the GPU** (27B, speculation off, greedy): the served answers -- three
  `json_object` prompts and every case of the corpus, each schema compiled
  strict and not described in the prompt -- equal the per-token masked argmax
  of the prefill route's readouts, up to a near-tie both routes call one.
  Ignored outside the GPU profile:

  ```sh
  cargo test -p ignis-server --features cuda --test structured_output_gpu -- --ignored --nocapture
  ```
