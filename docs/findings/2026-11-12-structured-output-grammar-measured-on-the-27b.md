# Structured output's grammar on the 27B: a check costs 0.2 µs and a repair's permitted set 34 µs, exact to a per-token reference and an independent validator, and the model must be told the schema

- Kind: experiment
- Status: current
- Observed: 2026-11-12
- Last verified: 2026-11-12
- Scope: crates/grammar (ignis-grammar), the 27B's tokenizer, the #287 spike's
  check-first emulation on the 27B
- Topic: structured-output
- Related: https://github.com/gpillon/ignis/issues/287,
  https://github.com/gpillon/ignis/issues/585,
  [ADR 0089](../adr/0089-a-grammar-checks-each-sampled-token-and-repairs-a-miss-and-the-model-is-told-the-schema.md)
- Superseded by: none
- Summary: On the 27B's 248,320-token vocabulary a grammar check takes 0.2 µs (1.0 µs p99) and a first-visit permitted set 34 µs (0.71 ms p99); misses fall from 4.2 to 0.04 per 100 tokens when the prompt describes the schema.

## Question

Can ignis hold an answer to a JSON schema by checking each sampled token
against its own grammar, and repairing only a miss, at a cost that disappears
next to a 27B decode round, without the grammar ever disagreeing with the
token-by-token truth or with the JSON Schema standard? And how often does the
real model miss?

## Evidence

**The grammar** is `crates/grammar` (`ignis-grammar`, #585), from the #287
spike: a pushdown automaton over bytes for `json_object` and JSON schemas
(OpenAI's strict subset), our own code; `regex-automata` builds the DFAs of
`pattern` and `format`. The vocabulary is every logits column decoded to its
exact bytes (`Tokenizer::token_bytes_table`), plus a byte trie over it.

**Corpus**: `crates/grammar/tests/corpus/cases.json`, 19 cases of our own
content: 2 `json_object` and 17 strict schemas shaped like an agent SDK's
(nested objects, arrays of objects, enums, null unions, `anyOf`, `$defs` and
root recursion, bounded numbers, `multipleOf`, all nine formats, patterns),
each with valid documents, violations and a natural prompt.

**Part A, the grammar on the CPU** (#585, release build, Intel Core
i9-10900K, Windows 11, 2026-11-12):

```text
cargo build --release -p ignis-grammar --example grammar_bench
target/x86_64-pc-windows-msvc/release/examples/grammar_bench.exe \
    F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer \
    crates/grammar/tests/corpus/cases.json OUT_DIR
python -m venv VENV && VENV/Scripts/pip install -r tools/grammar-check/requirements.txt
VENV/Scripts/python -I tools/grammar-check/validate.py OUT_DIR/crosscheck.jsonl
```

The bench replays each valid document as the tokenizer encodes it, compact
and pretty-printed, timing the check on every token and the permitted set on
every state (on a fresh grammar, so a first visit is a real one), holds the
set to the reference (the check run on all 248,320 columns) on sampled states
and to the document's next token on every state, replays the violations, and
generates random documents of each grammar's language. Its output (the log,
`summary.json`, the documents and the validator's report) is kept in
`.scratch/287/585-bench/`.

| operation | n | p50 | p99 | max |
|---|---|---|---|---|
| check, corpus tokens | 4,345 | 0.2 µs | 1.0 µs | 4.9 µs |
| check, random-document tokens | 91,070 | 0.1 µs | 0.9 µs | 49.2 µs |
| permitted set, first visit of a state | 3,722 | 33.7 µs | 714 µs | 1.36 ms |
| permitted set, cached | 4,968 | 0.8 µs | 3.7 µs | 56.0 µs |
| compile, the corpus's schemas (first) | 19 | | | 2.14 ms |
| compile, an SDK-sized schema (99 properties, 9 `$defs`, 8 KB) | 1 | | | 1.97 ms |
| compile, a schema at OpenAI's limits (5,000 properties, 1,000 enum values) | 1 | | | 26.5 ms |

- **Exactness.** 3,823 states equal the per-token reference, 0 mismatches;
  every document's next token is in its permitted set; every violation is
  refused. The machine-local test
  (`cargo test -p ignis-grammar --test real_tokenizer`) holds another 1,953
  states, at least 69 per case, plus the rare string states (inside a `\u`
  escape, after an unanchored pattern has matched).
- **The independent validator.** `jsonschema` 4.26.0, Draft 2020-12, format
  checking on, numbers parsed as exact decimals: 1,058 documents, 0
  disagreements. 60 valid corpus documents accepted, 64 violations refused, 12
  valid documents outside the generated subset (keys out of schema order)
  refused by design, and 922 random documents of the grammars' languages valid:
  the 19 corpus grammars, the two large schemas and four schemas compiled
  without `strict`. Four random `invoice` documents change verdict when the
  validator uses floats, which is why it uses decimals.
- **No dead end.** The unit tests take a completion from every state of random
  walks and from walks that nest as deep as the automaton's stack allows; one
  always exists (`cargo test -p ignis-grammar --lib`).
- **Memory.** The vocabulary takes 13.2 MiB per tokenizer, built in about
  160 ms (the byte table 93 ms, the trie and the nine free-string tables
  64 ms). A
  grammar with its caches takes 0.2-4.5 MiB on the corpus.
- **The slow first visits** are the strings an `email` or a `hostname` holds:
  their 63-character label counters make 290-536 DFA states, and a state's
  first visit walks most of the vocabulary's ASCII words.

**Part B, the check-first decode on the 27B** (the #287 spike, 2026-11-12;
the spike's emulation harness was not kept once #287 shipped, and
[the served measurements](2026-11-13-structured-output-served-on-the-27b.md)
take its place; raw results in `.scratch/287/spike/partB/`):
nvfp4full-v2, KV hq-e8-2b, thinking off, greedy, `max_tokens` 512. Each
position's full logits come from the prefill route's readout; the
unconstrained argmax is checked and a miss repaired with the argmax over the
permitted set, so the answer is exactly what constrained greedy decoding
writes.

| | schema absent from the prompt | schema described in the prompt |
|---|---|---|
| runs / tokens | 18 / 2,718 | 17 / 2,547 |
| misses per 100 tokens | 4.78 | 0.51 |
| misses after the first token, per 100 | 4.16 | 0.04 |
| conforming | 17/18 (one runaway cut at 512) | 17/17 |
| unconstrained output conforming | 2/18 | 16/17 |
| misses by kind | key 72, value start 23, literal 14, structural 13, string 4, number 4 | value start 12, string 1 |
| repair's permitted set, p50 / max | 28 / 107 µs | 3 / 33 µs |
| estimated decode against free, DFlash2 at 4.41 tokens a round (mean / worst) | 0.925 / 0.860 | 0.991 / 0.983 |

## Finding

- **Observed: the check is free and the repair cheap.** A check is 0.2 µs, a
  repair's permitted set 34 µs on a state's first visit and under 1 µs after;
  in the served runs a repair took 26 µs typically and 107 µs at most, about
  0.2% of a 12.8 ms round. The #585 bounds hold: check p99 1.0 µs (bound
  10 µs), first-visit permitted set p99 0.71 ms (bound 1 ms), an SDK-sized
  compile 2.0 ms (bound 50 ms).
- **Observed: the grammar is exact.** The permitted set equals the per-token
  reference on 5,776 states of the real vocabulary, and an independent
  validator in exact decimals agrees with every verdict.
- **Observed: whether the model sees the schema decides the cost.** Without
  it, the model writes its own key names and the grammar puts the schema's in,
  in odd token splits (`s|tep|s`), about 4 misses per 100 tokens; values stay
  right (43 of 44). Described at the end of the prompt, misses after the first
  token fall to 0.04 per 100; the remaining first-token misses are the opening
  fence, which the prefill's own draw repairs before any drafting.
- **Inferred: under DFlash2**, each miss after the first token cuts a verified
  run; from the miss positions, schema absent costs an estimated 8-18% of
  decode speed, schema described 1-2%. Not a served measurement.

## Implications

- Check-first with repair only on a miss is the design (ADR 0089), and the
  server describes the schema by default.
- The first-visit p99 is 0.71 ms against a 1 ms bound, set by `email` and
  `hostname`; a schema with many such fields can approach it. Three fixes, none
  needed yet: build those strings' tables at compile time, drop the label
  counters, or give the walk a fast path for ASCII words.
- A schema at OpenAI's limits compiles in 27 ms; a request that sends one pays
  that once, then the cache (`GrammarCache`) serves it.

## Limits and unknowns

- One CPU (an i9-10900K), one model (the 27B; the 16 GB artifacts share its
  tokenizer), greedy only, thinking off in Part B.
- Part B's speed is estimated from miss positions; the served cost is measured
  at the end of #287's work.
- The generated documents are a subset of the valid ones (keys in schema
  order, canonical literals, integers as digits, bounded numbers in plain
  decimal, whitespace runs of at most 48 bytes, nesting bounded by the stack);
  `hostname`'s 253-character total and `email`'s lengths are not enforced.

## Follow-ups

- The leaf's repair and the check-first decode on the 27B: #586 and the
  tickets after it under #287.
- A loop inside a constrained answer: #584.
