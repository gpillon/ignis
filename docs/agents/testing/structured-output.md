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
- **The validator** (`ignis_grammar::validate`, #608, ADR 0094): what every
  tool call's arguments are checked against, and the oracle of the grammar's
  property tests, so it shares nothing with the automaton but the JSON parser
  and the pattern translation. `cargo test -p ignis-grammar --lib validate`
  (where a violation is found, exact decimals, a `$ref` cycle, the formats'
  RFC edges) and `cargo test -p ignis-grammar --test tool_arguments` (about 1
  s): a corpus of tool arguments (`tests/corpus/tool_arguments.json`), each
  invalid one labelled by hand with its kinds, held to the labels and, through
  `tools/grammar-check/validate.py`, to `jsonschema` (`ours valid, theirs
  valid: N` and `ours invalid, theirs invalid: M`, 0 disagreements), the
  Python found as below and skipped or failed on CI the same way.
- **Where a value ends** (`ignis_grammar::call`, #610, ADR 0094): the rule
  the call parser and a held tool's grammar share. `cargo test -p
  ignis-grammar --lib call` pins it on bytes (a `</parameter>` line followed
  by text, openers of a written or undeclared parameter, a `</function>` not
  followed by `</tool_call>`, a forgotten closer, a close with a required
  parameter missing, the residue, whitespace and newlines between the tags,
  every prefix of a step pending) and holds a line's answer stable as bytes
  arrive. `cargo test -p ignis-server --lib toolcall` holds the parser's
  first-occurrence cuts to a verbatim copy of the pre-#610 parser on 20,000
  generated tag soups, fed in random chunks, checked or not, ending at the
  first call or not, and reads a held call (`ToolCallScanner::holding`, set
  through `ToolSchemas::holding` since #611) whole and split at every byte.

## The bench and the independent validator

The timings #585 bounds, and the cross-check of every verdict against
`jsonschema` (Draft 2020-12, format checking, numbers as exact decimals). It
needs the 27B artifact; it takes about a minute.

```sh
cargo build --release -p ignis-grammar --example grammar_bench
target/x86_64-pc-windows-msvc/release/examples/grammar_bench.exe \
    F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer \
    crates/grammar/tests/corpus/cases.json .scratch/grammar-bench
make grammar-venv
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
  by the deadline; the refusals (Flash-Next, tools beside a split
  `<tool_call>`). Tools beside the schema (#592, the test vocabulary's
  `<tool_call>` one control token): a turn that opens a call is released and
  its call comes back as written, with no repair, no miss on the series and
  its later rounds unchecked, also with thinking on and inside a verified run
  that keeps every draft; the next turn is held again; an opener inside the
  answer is a miss; a repair's permitted set holds the opener until the
  answer starts and never after; a forced `tool_choice` forces the call.
  `openai_http_fields strict_on_a_function_tool` pins `strict` on a function
  tool inert where no grammar holds the answers (its template has none), on
  chat completions and the Responses API; on the 27B both hold it (#611,
  #613, below). Against a
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
- **The call check over the mock engine** (`cargo test -p ignis-server
  --test tool_call_checks_http`, about 1 s, #608): a script that writes a
  call of every kind (valid, each violation, an undeclared tool, a block that
  does not parse) is counted under its outcome and kinds on chat completions
  (whole, with a stop sequence, streamed) and on the Responses API (HTTP,
  WebSocket); each response equals the same request's on a server with
  metrics off, which checks nothing (whole: its bytes; streamed: what the
  stream assembles to); a call the first token writes is counted once, not
  again by the first-token probe; under `parallel_tool_calls: false` (#609)
  only the blocks up to the call that ends the turn are counted, a dropped
  one before it included (#640);
  strict tools outside the subset are counted as they arrive, and refused on
  chat completions (#611) and on the Responses API (#613); a value cut at a
  `</parameter>` line, whose text after it the parser dropped and the
  confirmed close would keep, is counted split (#610), and a plain value, the
  residue and a close with its required parameter missing are not; a call
  holding a key an open object does not declare, in a tool `held` would hold,
  is counted once under `ignis_tool_call_undeclared_parameters_total`, valid
  or not, on chat completions and on the Responses API (`strict: false`), and
  not one under `additionalProperties: true`, in a tool `held` leaves free (a
  nested map, a `oneOf`), in a closed object, or on Flash-Next (#641).
- **The independent validator in the served tests**: the schema answers go
  through `tools/grammar-check/validate.py` with the Python of
  `IGNIS_GRAMMAR_CHECK_PYTHON`, else the venv at `.scratch/grammar-venv`
  (`make grammar-venv`). Where neither exists the validation is skipped with
  a line saying so, and the hand checks still run; on CI (`CI` set) it fails
  instead. CI's test jobs install the requirements on every shard and set
  `IGNIS_GRAMMAR_CHECK_PYTHON` (`release.yml`, held by
  `ci_pipeline::every_test_shard_installs_the_grammar_validator_before_the_suite`).
- **The generated reference** (`docs/user/reference/json-schema.md`,
  `user_docs::json_schema`): its lists come from the grammar crate's, and the
  page compiles each inert keyword (refused under `strict`, inert without)
  and a schema past each limit, so generating it fails if the grammar
  disagrees. `cargo test -p ignis-server --test user_docs_reference` holds
  the committed page to it.
- **The adapter** (`cargo test -p ignis-runtime --test runtime_compute
  a_checked_lane`): a checked lane reports its pending token, the EOS as a stop;
  through `decode_step_cut` (#590) its verified run reaches the host by its
  job, the EOS as a stop, and is committed as far as the host kept it.
- **On the GPU** (27B, greedy): the served answers -- three `json_object`
  prompts and every case of the corpus, each schema compiled strict and not
  described in the prompt. With DFlash2 off each equals the per-token masked
  argmax of the prefill route's readouts, up to a near-tie both prefill
  routes call one. With DFlash2 on each equals the answer with it off, or
  parts from it at such a near-tie, and equals exactly the per-token masked
  argmax of the verify route's own scores: the answer teacher-forced through
  verify rounds at width 1 (`step::decode_program_verify_forced`), each
  column read (`dflash2_readout::verify_logits`). A verify column's scores
  are the same bit for bit whichever round it falls in, but they part from
  both prefill routes' by more than a near-tie, so a speculative answer is
  never held to the prefill routes past the point where it parts from the
  answer with speculation off. Ignored outside the GPU profile:

  ```sh
  cargo test -p ignis-server --features cuda --test structured_output_gpu -- --ignored --nocapture
  ```

## Speculation stays on (GitHub #590)

A verify round's run is checked before it is committed: the host's cut keeps
it up to its first draft the grammar does not pass, and that draft waits to be
repaired from the verify pass's scores, as a plain round's draw is.

- **Over the mock engine** (`cargo test -p ignis-core --test
  structured_verify_cut`, `cargo test -p ignis-server --test
  structured_output_http with_speculation`): `MockCompute::with_runs` commits
  runs as verify rounds do and shows each checked lane's run to the host
  (`MockCompute::cut_calls`). A miss in the middle of a run is cut there and
  repaired, the tokens before it kept; an answer is the same with runs and one
  token a round; an unconstrained neighbour keeps its runs.
- **The leaf's cut arithmetic** (`ignis_cut_host_kept_run`, host-only, in
  `kernel/tests/test_reasoning_redirect.cpp`):
  `ctest --test-dir kernel/build -R ignis_kernel_reasoning_redirect_test`.
- **On the GPU**, beside the served test above (`verify_cut_gpu`): three
  DFlash2 lanes, one held to a strict schema whose keys the model does not
  guess (under `json_object` its own object passes draft after draft, and no
  run is cut inside). The free lanes verify and commit what they do beside a
  free lane, every round replays its verify graph, each repair after a cut
  is the masked argmax of the verify column that drew it, and a run was cut
  inside. Its second test (#691) runs the same at the extended shape, a
  DFlash2 load at 15 drafts under `IGNIS_ROUND_SHAPE=extended` (16 columns a
  lane, the column read at that stride), then holds a repair to its own
  round's scores: a cut at one shape, a round of the other over the free lanes,
  and the repair after it refused, both ways. The two tests load the 27B each,
  so they run one at a time:

  ```sh
  cargo test -p ignis-core --features cuda --test verify_cut_gpu -- --ignored --nocapture --test-threads=1
  ```

  A cut round's shape under the shape rule, which reads what the target
  accepted and not what the cut kept, is part 4 of `dflash2_round_shape_gpu`.

## Held tools (GitHub #611, spec #582, ADR 0094)

- **The grammar** (`cargo test -p ignis-grammar --lib tool_calls`, about 6
  s): free text passes, the opener (token or bytes) arms, the name is held,
  a free tool's body runs to its first `</tool_call>`, a held call takes its
  parameters in any order and once, closes only with every required one
  written, holds an enum, a pattern and a format on one line, misses at
  spike S1's two points (spec #582), bounds the whitespace between tags (the byte past
  the bound a miss) and refuses what it cannot hold by name. No state is a
  dead end (from every state of generated texts, over grammars whose
  whitespace is bounded at two or three bytes: a way out of its value and
  of each after it, and of every value its call may still open under every
  set of parameters written), and the mask equals the token-by-token
  reference on every state of them. Under a named `tool_choice` only the
  named tool can be called (#615); the deepest typed value held at the
  widest `model.grammar_max_depth` reads back as its JSON. A
  normalized tool (#614) is held, its objects closed and its optional
  parameters omissible and left out; one whose schema does not normalize,
  or that the dialect cannot hold, is left free, never refused, a strict
  tool beside it held; strict and normalized compile apart. Machine-local, on the 27B's tokenizer (`--test tool_calls_real_tokenizer`,
  a few seconds, skips without the artifact): the tags are one added token
  each, read as they decode; no ordinary token holds the opener, so the first
  draw keeps no scores; masks equal the reference on held calls as the
  tokenizer splits them; S1's miss points on the real ids.
- **The grammar against the parser** (`cargo test -p ignis-server --lib
  toolcall`): what the grammar accepts from dialect pieces, read back in
  random chunks, delivers every held call, each valid under #608's validator
  (a free tool's call may be dropped as today); every valid call in the
  template's form is accepted, but for the residue and S1's miss points,
  carved out by name.
- **The scheduler** (`cargo test -p ignis-core --test held_tool_calls`): the
  prefill's draw opening a call, with no scores kept; a verified run holding
  `</think>`, the opener and a miss cut at the miss; free text never a miss;
  a stop inside a held call a miss; an empty permitted set ending the
  request in an error, unrepaired. Beside a forced literal (#615), thinking
  off and on: its tokens fed to the grammar and never a miss, the model's
  after it held; a literal token the grammar refuses not forced; a named
  choice's call held to its name where the literal yields or the model
  opened the call itself. Beside a
  structured answer: the call the opener starts held, and the next one, a
  verified run cut at a miss after the opener.
- **Over the mock engine** (`cargo test -p ignis-server --test
  held_tools_http`, about 1 s): the scripts that miss on purpose come out
  valid, each with its repairs counted under `type="tool"`; each refusal by
  code and `param`; the schema's key order; a free tool beside a strict one;
  no strict tool, no grammar; streaming; a cap; the reasoning free. On the
  Responses API (#613), over HTTP whole and streamed and over the WebSocket
  mode, whose frame `tools` are read again from its text: the same scripts
  come out the same valid calls in the schema's order, repairs counted, a
  streamed call in one arguments delta; each refusal names the flat tool
  (`tools[1].parameters`), or the chat path for a tool sent in the chat
  shape, as a 400 and as the socket's `error` event; a tool with
  `strict: false` is held to nothing and echoed as sent. A tool sent
  without `strict` (#614) is normalized: an invented parameter and a nested
  object's own key repaired away, the optional parameters left out, the
  arguments valid under the schema as sent, echoed `strict: true`; one whose
  length bounds are left out held on the rest, a string past its bound served
  and counted `constraint`, on the Responses API and at `held` on chat
  completions (#641); one whose
  schema does not normalize, and every one on Flash-Next, served free and
  echoed `strict: false`; each tool's echo, in its `function` for the chat
  shape, and none held under `tool_choice: "none"`. On chat completions a
  tool without `strict` stays free. Beside the fields that shape
  a turn (#615): a `"required"` and a named forced call missing after its
  opening, valid, the forced tokens no miss; beside `response_format` a held
  call, or an answer valid under the schema, both counted under
  `type="json_schema"`; with thinking on a call written in the
  reasoning neither held nor parsed; under `parallel_tool_calls: false` a
  value holding `</tool_call>` delivered whole, the turn ended at the
  confirmed close. On the Responses API, over HTTP whole and streamed and
  over the socket, `text.format`, `"required"` and `parallel_tool_calls:
  false` the same. Under the operator's switch at `held` (#616,
  `Server::with_tool_arguments`): a chat tool without `strict` and one with
  `strict: false` held as the Responses default holds one without, the same
  misses; one whose schema does not normalize free on both surfaces; a
  strict tool's refusals unchanged; on the Responses API a `strict: false`
  tool held whole, streamed and over the socket and echoed `strict: true`,
  each tool's echo; Flash-Next unchanged. The byte-identity of requests with
  no strict tool, and of every request at the default `as-sent`, is the
  existing suites'.
- **One call per turn over the mock engine** (`cargo test -p ignis-server
  --test parallel_tool_calls_http`, #609): `parallel_tool_calls: false` ends
  the turn at its first call on chat completions (whole and streamed) and
  the Responses API (HTTP and socket), on both families, a call closed in the
  reasoning ending nothing, and a closed block that does not parse ending
  nothing either (#640); `true` or absent serves every call.
  `openai_http_tool_choice` covers the forced call beside the grammar.
- **On the GPU**, the served close-out (#617): `held_tools_gpu` serves a
  corpus of agent requests (`support/held_tools_corpus.rs`: chat completions
  and the Responses API, strict tools and the Responses default, forced calls
  with thinking off and on, `response_format` beside the tools, and cases
  built to make the model miss) on the 27B with DFlash2 on, each beside its
  free twin, every tool sent `strict: false`: every call to a declared tool,
  a named choice's to the named one, every held call valid, each miss
  repaired and counted, the verify rounds drafting and accepting. It prints
  both deliveries with the twin's faults and what the held call wrote there,
  the cost on held traffic beside free, and the one-call-per-turn
  conversations; each test goes through every case before it fails. The two
  tests load the 27B under one card lock; run them one at a time:

  ```sh
  cargo test -p ignis-server --features cuda --test held_tools_gpu -- --ignored --nocapture --test-threads=1
  ```

  `flash_next_parallel_tool_calls_gpu` holds the same conversations to one
  call per turn on Flash-Next. It needs the Flash-Next artifact
  (`IGNIS_FLASH_NEXT_DIR` names another directory) and about 40 GiB of host
  RAM, and runs under the GPU profile:

  ```sh
  cargo test -p ignis-server --features cuda --test flash_next_parallel_tool_calls_gpu -- --ignored --nocapture
  ```
