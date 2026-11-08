# Qwen3-ASR on the CPU: speed, FLEURS, and the owner's dictations with and without context

- Kind: experiment
- Status: current
- Observed: 2026-10-24
- Last verified: 2026-10-24
- Scope: speech / Qwen3-ASR 0.6B and 1.7B (bf16, Q8_0, Q4_K_M) on llama.cpp b11840, CPU only, on the owner's
  i9-10900K (Windows) and a 4-core podman container: time per clip, threads, memory, FLEURS WER, the owner's eleven
  dictations with and without context, language ID
- Topic: speech
- Related: [ADR 0067](../adr/0067-the-speech-model-waits-on-the-host-and-borrows-free-kv-pages.md),
  [ADR 0068](../adr/0068-the-speech-models-cpu-executor-is-ours-in-rust.md),
  [spec speech](../specs/speech/spec.md), [spec speech/01](../specs/speech/01-cpu-executor.md),
  [Rust quantized matmul matches ggml on the CPU](2026-10-24-rust-quantized-matmul-matches-ggml.md)
- Superseded by: none
- Summary: Warm at 10 threads on an i9-10900K every Qwen3-ASR variant runs 2-10x real time (1.7B Q4_K_M 4.8x), so the CPU works only through incremental transcription, and on the owner's Italian 1.7B Q4_K_M has 3.7% WER, 2.4% with context.

## Question

Can Qwen3-ASR transcribe dictation fast enough on a CPU alone, so that ignis can serve it on any card class, and at
what quality? Which variant is the smallest that holds on the owner's own Italian, with its coding jargon? Does
context in the system turn help, and at what latency? What does the model's language label do on code-switched
Italian?

## Evidence

Two runs, 2026-10-23 to 2026-10-24, both CPU only (`CUDA_VISIBLE_DEVICES=""`), never at the same time.

**Setup.**
- **Host:** i9-10900K, 10 cores / 20 threads, AVX2 and FMA, no AVX-512, no BF16; 4x16 GB DDR4-3200 dual channel;
  Windows 11.
- **Container:** podman on the WSL machine, `--cpuset-cpus=0-3`, 4 threads.
- **Engine:** llama.cpp release b11840 (`36e36e63a`), official CPU binaries. On Windows the build is Clang 20.1.8
  and loads `ggml-cpu-haswell.dll`; in the container it is GNU 11.4.
- **Weights:** ggml-org's bf16 and Q8_0 GGUFs, and a Q4_K_M quantized from ggml-org's bf16 with b11840's
  `llama-quantize` (ggml-org ships no 4-bit). The audio encoder is Q8_0 beside Q8_0 and Q4_K_M decoders. Every hash
  is in the raw reports.
- **Reference:** Transformers 5.20.2 on torch 2.14.1+cpu, FP32, with `Qwen/Qwen3-ASR-0.6B-hf` @`7f1569a4` and
  `-1.7B-hf` @`bcd2b5b7`.

**Run 1, speed and FLEURS** (`llama-mtmd-cli`, one process per clip):
- clips: 20 Italian and 20 English FLEURS test clips of 5-25 s, plus one 60.8 s Italian clip made of three joined;
- 3 runs per clip, interleaved, median per clip;
- warm time = prompt eval + eval, which matches the log's time span within ~1%;
- x real time is pooled over the 40 short clips. "10 s" comes from a line fitted through them, and "60 s" is the
  60.8 s clip's median.

| host, threads | variant | x real time | 10 s clip | 60 s clip (encode / prefill / decode) | decode ms/token | resident MB (`-lm none`) | FLEURS WER it / en |
|---|---|---|---|---|---|---|---|
| i9, 10 | 0.6B Q4_K_M | 8.7x | 1.2 s | 7.1 s (1.7 / 1.7 / 3.8) | 18.2 | 1,371 | 6.3 / 7.2 |
| i9, 10 | 0.6B Q8_0 | 6.7x | 1.5 s | 8.7 s (1.6 / 2.4 / 4.7) | 24.1 | 1,673 | 6.1 / 6.6 |
| i9, 10 | 1.7B Q4_K_M | 4.8x | 2.1 s | 12.3 s (2.1 / 3.2 / 7.0) | 35.1 | 2,274 | 3.5 / 5.3 |
| i9, 10 | 1.7B Q8_0 | 3.1x | 3.3 s | 18.0 s (2.1 / 6.0 / 10.0) | 54.8 | 3,112 | 3.5 / 4.9 |
| i9, 10 | 0.6B bf16 | 5.4x | 1.9 s | 10.4 s | 39.2 | | 6.1 / 6.8 |
| i9, 10 | 1.7B bf16 | 2.1x | 5.0 s | 26.3 s | 110.8 | | 3.9 / 4.9 |
| 4-core, 4 | 0.6B Q4_K_M | 6.2x | 1.6 s | 10.5 s | 17.0 | | 6.3 / 7.2 |
| 4-core, 4 | 0.6B Q8_0 | 5.0x | 2.0 s | 12.5 s | 23.6 | | 5.5 / 6.6 |
| 4-core, 4 | 1.7B Q4_K_M | 3.0x | 3.5 s | 20.5 s | 42.6 | | 3.9 / 5.1 |
| 4-core, 4 | 1.7B Q8_0 | 2.2x | 4.7 s | 27.3 s | 63.3 | | 3.9 / 4.9 |

- **Threads:** 10 (the physical cores) is fastest in every configuration. 20 is 8-23% slower, all of it in decode;
  4 is 30-45% slower.
- **Transformers FP32** on the same 10 clips is 2.8x (0.6B) and 2.9x (1.7B) slower than llama.cpp Q8_0. The 1.7B
  runs at 1.1x real time and peaks at 9.2 GB.
- **1.7B decode** reads ~2.1 GB a token in 55 ms (Q8_0) and ~1.3 GB in 35 ms (Q4_K_M). Both are ~37-38 GB/s, the
  practical ceiling of dual-channel DDR4-3200. 0.6B decode (18-24 ms a token) is bound by per-token overhead.
- **Memory:** KV is 448 MiB at a 4,096-token context; a 20-minute clip would need ~1.7 GB. With the default mmap,
  Q4_K_M's working set (2,752 MB for 1.7B) counts both the mapped file and the repacked private copy.
- **FLEURS:** greedy output was identical across the 3 runs for every clip and configuration. Language ID errors: 0.
  The paper gives it 2.41 / 4.99 and en 3.35 / 4.39. The gap is the 20-clip sample and the normalisation.

**Run 2, the owner's dictations** (`llama-server`, warm, 10 threads, greedy, one run per clip):
- **Clips:** d01-d11, recorded by the owner at 48 kHz stereo and resampled to 16 kHz mono with soxr. They run from
  10.3 to 67.0 s, ~364 s in all. d01-d10 are Italian with coding jargon (574 reference words); d11 is English (46).
- **References** are the written target form (digits, file names, identifiers). WER is strict: `_ . / - :` become
  spaces, and there is no number or identifier normalisation.
- **Arms:**
  - **A**: an empty system turn, the official no-context prompt (15 text tokens, as the official processor counts);
  - **B**: a 1,496-character extract of the README as the system turn (420 tokens, as official), frozen by sha256
    before any reference was opened;
  - **B cached**: B with the prefix's KV reused from the previous request (`cache_n` 414 on every run).
- **Rendering:** the official layout was rebuilt from the server's log and checked on all 140 runs.

| variant | arm | Italian WER (errors) | English WER | 10 s dictation | 67 s dictation |
|---|---|---|---|---|---|
| 0.6B Q4_K_M | A / B / B cached | 12.7% (73) / 11.1% (64) / 12.0% (69) | 8.7% / 2.2% / 2.2% | 1.14 / 2.15 / 1.29 s | 8.24 / 10.38 / 9.21 s |
| 0.6B Q8_0 | A / B / B cached | 9.1% (52) / 9.2% (53) / 9.4% (54) | 2.2% / 2.2% / 2.2% | 1.90 / 3.26 / 1.68 s | 11.18 / 12.73 / 11.76 s |
| 1.7B Q4_K_M | A / B / B cached | 3.7% (21) / 2.4% (14) / 2.1% (12) | 0% / 0% / 0% | 2.65 / 4.39 / 2.41 s | 16.43 / 19.65 / 16.68 s |
| 1.7B Q8_0 | A / B / B cached | 3.3% (19) / 2.6% (15) / 2.6% (15) | 0% / 0% / 0% | 3.36 / 6.35 / 3.62 s | 24.13 / 29.21 / 24.37 s |

- **Context, cost:** an uncached context adds a median of 1.1 s (0.6B Q4_K_M) to 3.3 s (1.7B Q8_0) per request.
  Cached, it adds 0.1-0.4 s, and 0.09 s on 1.7B Q4_K_M.
- **Context, effect:** on 1.7B, B beats A on 5 of 10 Italian clips and loses none (Q4_K_M), or wins 3 and loses 1
  (Q8_0). Its gains are the words only the context could supply ("OpenCode", "Playground", "Flash-Next", "routes",
  "prefix reuse"). No distinctive context word was inserted where it was not spoken.
- **Positive control:** before the matrix, d06 and d05 with their own reference as context (1.7B Q8_0) went from 4
  errors to 1, fixing "gillon" → "gpillon" and "prefisso reuse" → "prefix reuse".
- **Language ID:** 1.7B labelled 4 of 44 runs wrongly, and none of them changed the text. d02 (Italian, git jargon)
  came out `English`. d03 and d11 came out `None` with a full, correct transcript. Forcing Italian on d03 took it
  from 2.7% to 10.8% ("route" → "root" three times). 0.6B made no LID error.
- **Residual errors on 1.7B with context** are identifiers: "hq-e8-2b" → "HQ-8-2B", "Qwen3" → "Qwen", "gpillon" →
  "gillon".
- **Probable ad-libs:** four errors appear in every arm and variant, the reference-as-context control included.
  They are probably what the owner actually said, about 0.7 pt of the pooled Italian WER.

**Raw data** stays on `H:\speech-study\` (nothing binary enters the repo):
- `results\`: `llamacpp_win.csv` (1,722 runs), `llamacpp_container4.csv` (492), `hf_reference.csv` (66),
  `summary_all.csv`, `mem_check*.csv`, `dictation_runs.csv` (140), `dictation_summary.csv`, `dictation_diffs.md`,
  the rendered prompts;
- `logs\`, `scripts\`, `data\clips\`;
- `dictations\` (the owner's recordings and references, private).

## Finding

**Observed:**
- Every variant transcribes at 2-10x real time on the i9 at 10 threads. The fastest is 0.6B Q4_K_M at 8.7x; 1.7B
  Q4_K_M runs at 4.8x.
- Whole-clip, a 60 s dictation waits 7.1-18.0 s after it ends.
- Q4_K_M matches Q8_0 on WER (within ~0.6 pt on FLEURS, within the noise on the dictations) and is 1.3x (0.6B) to
  1.55x (1.7B) faster.
- On the owner's Italian, 1.7B makes about a third of 0.6B's errors. 0.6B Q4_K_M fails a 10% bar (12.7%); 0.6B Q8_0
  passes it (9.1%).
- Context lowers 1.7B's Italian WER by 0.7-1.6 pt, the size of the noise floor but in the same direction across
  both quantisations and both B passes. A cached prefix makes it nearly free.
- The official Transformers path is ~3x slower than ggml on this CPU.

**Inferred:**
- On a CPU, dictation is practical only if the work is done while the user speaks: encoding and prefilling each
  window as it fills leaves only the last window and the decode after the user stops.
- A real budget 4-core with slower RAM will be slower than the container's 3.0x for 1.7B Q4_K_M, and can fall
  under 2x. 0.6B Q8_0 is the fallback that holds the 10% bar.
- 1.7B decode competes for memory bandwidth with anything else on the host.

**The go/no-go.** The coordinator *proposed* Q34's bands before the numbers existed, and the owner *confirmed* them
after seeing them. This is not a pre-registration. The bands:
- at least 10x real time: go;
- 2-10x: go only through incremental Realtime transcription;
- below 2x: no-go;
- the variant: the smallest whose WER on the owner's dictations is at most 10%.

The result: every variant is in 2-10x, so the CPU is a go only through incremental transcription. The smallest
variant within 10% is 0.6B Q8_0 (9.1%, 6.7x).

## Implications

- ADR 0067: the CPU executor is the speech model's baseline, and Realtime's incremental path is on phase 1's
  critical path.
- ADR 0068: the CPU path needs quantized SIMD kernels, since FP32 is ~3x slower, and Q4_K_M with a repacked layout
  is worth keeping. This run compared llama.cpp only with Transformers FP32, so it shows that such kernels are
  needed, not that they must be ggml's; [Rust quantized matmul matches ggml on the
  CPU](2026-10-24-rust-quantized-matmul-matches-ggml.md) measured ours.
- The catalog (Q49): `auto` is 1.7B Q4_K_M; 1.7B Q8_0 is also listed; 0.6B Q8_0 is the light fallback. 0.6B
  Q4_K_M is left out.
- The context (Q50) is prefilled once and reused, since uncached it costs 2-3 s on 1.7B.
- The language label (Q51) is never trusted to force anything, and `None` with text is not silence.
- The oracle (Q52) compares no free-running transcript, because a cached prefix alone changes text on 3-5 of 11
  clips. Its WER gate allows 1 pt.

## Limits and unknowns

- **The FLEURS runs used a non-official prompt.** `llama-mtmd-cli` cannot send a system turn, so they had none (13
  text tokens against the official 15). The dictation runs used the official one. The difference was not measured.
- **llama.cpp tokenises the last partial audio chunk differently from the official processor:** d01 (10.34 s) gets
  143 audio tokens against the official 135, d05 325 against 324. ignis's engine follows the official chunking
  (spec speech/01), so its numbers may differ slightly.
- **The 4-core container is optimistic.** Its vCPUs are not pinned, and it shares the host's 20 MB L3 and full
  dual-channel bandwidth.
- **Every dictation timing is a single run**; FLEURS times are medians of 3.
- **The noise floor is about 1 pt:** the cached pass renders the identical prompt and still changes the text on
  3-5 of 11 clips.
- **Small sample:** one speaker, one microphone, one room; 574 Italian words and 46 English. "60 s" in the dictation
  table is d10, 67 s.
- **Load:** another agent's GPU test process (~0.6 core) and cargo ran during run 1. 1.7B Q8_0 at 10 threads
  started with the CPU counter at 34.5%, so it may be a few % pessimistic.
- **Not measured:**
  - 9 threads, `speech.threads`' default on this host; the 10-thread and 4-thread figures bound it;
  - the effect of a CPU transcription on the GPU's text decode (Q38's check);
  - contention with Flash-Next's offloaded expert stream, which also uses host memory bandwidth;
  - the incremental path's latency and WER;
  - any GPU;
  - a Transformers run on the dictations.
- **The Q4_K_M files were quantized from ggml-org's bf16**, not from the official weights. The shipped files will be
  converted by ignis (Q47).

## Follow-ups

- Spec speech/01 AC 23: ignis's own executor, timed on the i9 and the container.
- Spec speech/03 AC 13: commit to completed within 2 s for a 60 s dictation on the i9.
- Spec speech/02 AC 16: text decode on the GPU during a CPU transcription (Q38).
