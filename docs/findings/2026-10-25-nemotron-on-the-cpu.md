# Nemotron on the CPU: its ASR loses Italian to Qwen3-ASR, its diarizer tells Italian speakers apart at a price

- Kind: experiment
- Status: current
- Observed: 2026-10-25
- Last verified: 2026-10-25
- Scope: speech / NVIDIA Nemotron 3.5 ASR Streaming 0.6B and Nemotron 3 Diarization on the CPU (NeMo-Speech.cpp
  Q8_0 nightly `caa3f47`, Transformers FP32), on the owner's i9-10900K (Windows, WSL) and a 4-core podman container:
  WER on the owner's eleven dictations and the FLEURS clips of the Qwen3-ASR study, x real time, latency to the
  final text, DER on synthetic Italian conversations, speaker attribution per word, CPU share
- Topic: speech
- Related: [ADR 0067](../adr/0067-the-speech-model-waits-on-the-host-and-borrows-free-kv-pages.md),
  [ADR 0068](../adr/0068-the-speech-models-cpu-executor-is-ours-in-rust.md),
  [ADR 0069](../adr/0069-the-speakers-come-from-nemotron-diarization-beside-qwen3-asr.md),
  [spec speech/07](../specs/speech/07-transcribe.md), [spec speech/08](../specs/speech/08-diarization.md),
  [Qwen3-ASR on the CPU](2026-10-24-qwen3-asr-on-cpu.md)
- Superseded by: none
- Summary: Nemotron's streaming ASR loses the owner's Italian to Qwen3-ASR (best WER 13.2% against 3.2-3.4%), while Nemotron 3 Diarization tells Italian speakers apart at DER 3.6-4.0%, at half of a 4-core machine at its low geometry.

## Question

The Playground's Transcribe view (spec speech/07) wants speaker labels, and Qwen3-ASR gives none: no diarization, no
speaker tokens, no word timestamps. Two NVIDIA open models could supply them: Nemotron 3 Diarization beside
Qwen3-ASR, or Nemotron 3.5 ASR with the diarizer, which NVIDIA pairs for speaker tags per word. On the CPU alone
(ADR 0067: the speech model never stays in VRAM), and on the owner's Italian:
- Does Nemotron's ASR transcribe the owner's dictations as well as Qwen3-ASR?
- Does the diarizer tell Italian speakers apart, although Italian is not named in its training data?
- What does each cost, on the i9 and on a 4-core machine?

Qwen3-ASR was not re-run: its numbers are the ones already measured, through the same scorer.

## Evidence

One spike, 2026-10-25, CPU only (`CUDA_VISIBLE_DEVICES=` everywhere, CPU-only builds and wheels). The report, every
script, raw output, RTTM and log are in `H:\speech-study\nemotron-spike\` (`REPORT.md`, section 5 lists the paths).

**The models** (cards and configs saved in `cards\`, HF `nvidia/nemotron-3.5-asr-streaming-0.6b` @`ea30d66d`,
`nvidia/Nemotron-3-Diarization` @`f667ed73`):
- **ASR:** cache-aware FastConformer, 24 layers (d 1024), RNNT decoder, 600M parameters, a 128-dim one-hot language
  prompt beside the encoder output. Chunks of 80, 160, 320, 560 and 1120 ms; `target_lang` such as `it-IT`, or
  `auto`. Its training text is "normalized to have spoken forms". No context biasing in the model.
- **Diarizer:** 100M parameters, a 31-layer RoPE Transformer (d 512) with a Sortformer head, 8 speakers in arrival
  order, an arrival-order speaker cache (AOSC) and a FIFO. Its geometries, in 80 ms frames: low (1.04 s) SPKCACHE 264
  / FIFO 264 / CHUNK 9 / RC 4 / UPDATE 222; very low (0.64 s) chunk 6 / rc 2; ultra low (0.32 s) chunk 3 / rc 1;
  offline (30.4 s) 264 / 40 / 340 / 40 / 300. Italian is not named in its training data, which does hold multilingual
  sets whose languages are not listed.
- Both under OpenMDW-1.1, F32 safetensors (ASR 2.55 GB, diarizer 397 MB) and Q8_0 GGUFs.

**Setup.**
- **Machines:** i9-10900K (10 cores / 20 threads, AVX2, no AVX-512), Windows 11; the podman WSL machine (20 CPUs,
  unpinned); a 4-core container (`--cpuset-cpus=0-3`), the Qwen3-ASR study's.
- **Runtimes:** NeMo-Speech.cpp Q8_0 (Apache-2.0, ggml-based; official nightly CPU archives for Windows and Linux,
  plus a Linux build of `caa3f47` whose only change lets the diarizer take a thread count: the Windows nightly's
  standalone diarizer is pinned at 4 threads). Transformers 5.20.2 on torch 2.14.1+cpu, FP32, after the cards'
  streaming code. NeMo itself was not installed (Linux only; Transformers covers FP32).
- **Threads:** ASR 9 on the i9 (also 10 and 20), 3 in the container (also 4). Diarizer 4 on Windows (pinned), 9 under
  WSL, 3 in the container.
- **Timing:** every timed run waited for total CPU load at or under 15% (`\Processor(_Total)`, 3 s); all 155 started
  at 4.6-14.7%. Each ran twice; the repeats agree within 1-2%. Quality runs were not timed.
- **Scorer:** the Qwen3-ASR study's `dict_score.py` and `analyze.py`, unchanged. Re-scoring Qwen's llama.cpp
  hypotheses through them reproduced that finding's 3.39% (dictations) and 4.36% (FLEURS 40) exactly.

**Data.**
- The owner's eleven dictations (620 reference words, 574 Italian, 46 English) and the Qwen3-ASR study's 40 FLEURS
  clips plus its 60.8 s Italian clip.
- **Twenty synthetic Italian conversations** from VoxPopuli `it` test and validation (`build_conversations.py`, seed
  20261025): 76 distinct speakers, each used once; 2-6 speakers each; 144-312 s each, 4,627 s in all. Turn changes:
  a pause of 0.3-1.5 s (70%), no pause (18%), an overlap of 0.2-1.0 s (12%). In every conversation one speaker returns
  after a 61-111 s absence. Speech levelled to −25 dBFS; reference RTTM from a frame-energy VAD, overlap kept.

**ASR WER, strict** (Nemotron with no context; "no-form" drops the error spans whose reference holds a digit or whose
hypothesis spells a symbol, an approximate rule):

| engine, arm | 11 dictations (IT / EN) | no-form | FLEURS 40 (it / en) |
|---|---|---|---|
| Nemotron, stream 160 ms, it-IT | 19.84 (20.38 / 13.04) | 6.29 | 8.40 (5.49 / 11.44) |
| Nemotron, stream 560 ms, it-IT | 15.16 (15.33 / 13.04) | 5.00 | 6.95 (4.67 / 9.32) |
| **Nemotron, stream 1120 ms, it-IT** | **13.23 (13.24 / 13.04)** | **4.68** | **7.37 (4.47 / 10.38)** |
| Nemotron, stream 1120 ms, auto | 13.55 (13.59 / 13.04) | 5.00 | 7.78 (4.88 / 10.81) |
| Nemotron, offline pass, it-IT | 16.45 (16.55 / 15.22) | 5.97 | 7.78 (5.49 / 10.17) |
| Qwen `qwen3-asr-1.7b` executor, batch, no context (spec 01) | 3.23 | n/a | 4.05 |
| same, Realtime manual commit (spec 03) / `server_vad` items (spec 05) | 4.03 / 4.68 | | |
| Qwen llama.cpp 1.7B Q4_K_M, arm A, empty system turn | 3.39 (3.66 / 0.00) | 2.74 | 4.36 (3.46 / 5.30) |

- NeMo-Speech.cpp Q8_0 and Transformers FP32 agree within ±1.1 pt at every matching chunk: the quantization costs
  nothing.
- After the form spans, what remains is English jargon inside Italian: "branch" → "brench", "KV cache" → "kukesh",
  "VRAM" → "wram", "api.rs" → "ap rs".
- In `auto`, the language tag follows a final punctuation mark, and streamed text often has none: 25-32 of 52 clips
  carried no tag when streamed.

**ASR speed** (NeMo-Speech.cpp, flat out; the share is CPU-seconds per audio second over the physical cores):

| chunk | i9, 9 threads | share of 10 cores | final text after the last audio (d10 / median / max) | container, 3 threads | after the last audio |
|---|---|---|---|---|---|
| 160 ms | 3.54-3.58x | 24.5% | 176 / 176 / 193 ms | 1.35x | 455 / 453 / 463 ms |
| 560 ms | 8.14-8.22x | 10.7% | 190 / 196 / 252 ms | 3.16x | 489 / 493 / 651 ms |
| 1120 ms | 10.92-10.95x | 8.1% | 266 / 272 / 303 ms | 4.70x | 639 / 632 / 666 ms |

Qwen3-ASR for comparison (spec 01, spec 03): the executor's batch path 4.1-4.7x on the i9 and 2.9-3.0x on the
container; through the Realtime route a 67 s dictation goes from commit to `completed` in 2.4-4.4 s on the i9 and
5.8-6.1 s on the container.

**DER on the synthetic conversations** (NeMo-Speech.cpp Q8_0, collar 0.25 s, overlap scored):

| geometry (latency) | DER (miss / FA / confusion) | speaker count right | returning speaker keeps its label | turns labelled right |
|---|---|---|---|---|
| low (1.04 s) | 3.97 (1.60 / 0.38 / 1.99) | 15 of 20 | 19 of 20 | 97.7% |
| the runtime's default, 264/80/13/1/40 (1.12 s) | 3.85 (1.77 / 0.35 / 1.74) | 15 of 20 | 19 of 20 | 97.7% |
| chunk 25, 264/264/25/4/222 (2.32 s) | 3.60 (1.37 / 0.37 / 1.86) | 14 of 20 | 19 of 20 | 98.3% |
| offline (30.4 s) | 3.56 (0.78 / 0.39 / 2.40) | 16 of 20 | 20 of 20 | 97.1% |

- At collar 0 the DER is 10.3-10.7%; the extra ≈6.6 pt is false alarm from the runtime's padding against a tight
  energy-VAD reference, not a model error.
- Counting errors are almost all **one extra speaker**. In 4 of the 5 low-geometry cases the extra label holds 0.9-5.1 s
  of speech; one (14.5 s) is a real split of one speaker.
- The worst conversations are three same-gender speakers: two female voices with a change and no pause were merged
  (15.1% DER).
- On 5 conversations Transformers FP32 gives the same confusion as Q8_0 (0.92 against 0.67 at low, 0.38 against 0.38
  offline): neither the quantization nor the runtime's relabelling moves who is who.

**Diarizer cost** (the first 90 s of one conversation):

| geometry | GFLOP per step (estimate) | Windows nightly, 4 threads | WSL, 9 threads | container, 3 threads |
|---|---|---|---|---|
| low | 124 | 1.44x, 28.7% of 10 cores | 2.45x, 35.8% | 1.26x, 54% of 4 cores |
| very low | 123 | 0.98x: falls behind | 1.67x | 0.86x: falls behind |
| ultra low | 122 | not run | 0.85x: falls behind | not run |
| chunk 25 | 129 | 3.83x, 10.4% | 6.46x, 13.9% | 3.32x |
| offline | 163 | 27.4x | 47.6x | 25.1x |
| Transformers FP32, low, 9 threads, Windows | | 3.07x (245 ms a step) | | |

- Each step re-encodes the speaker cache, the FIFO and the chunk, about 540 frames through all 31 layers. That is why
  a longer chunk is cheaper per audio second and costs no DER (chunk 25 against low).
- On this model PyTorch FP32 beats ggml Q8_0 (508 against 419 GFLOP/s at 9 threads): each step is one large GEMM.
- The diarizer costs about 11x the Nemotron ASR per audio second (173 against ~16 GFLOP, estimates).

**Nemotron ASR with the diarizer, speaker per word** (NeMo-Speech.cpp `transcribe --stream --diarize`, ASR at
1120 ms it-IT, four conversations of 2, 3, 4 and 6 speakers, 857 s):
- low geometry: **98.1%** of 1,810 scored words get the right speaker (conv08 94.4%, the others 99.2-100%), and all
  10 words in overlaps; **1.77x** real time on the i9 at 9 threads (50% of the cores). In the container: 0.87-0.88x,
  falling behind (wall time only: its CPU time was lost to a script error).
- the runtime's default geometry: 94.1% (conv08 79.4%), 2.84x on the i9.
- This is not the card's integration guide's pipeline (`masked_asr`, one ASR stream per speaker), which
  NeMo-Speech.cpp does not have.

## Finding

- **Nemotron 3.5 ASR is not a replacement for Qwen3-ASR on the owner's Italian.** Its best strict WER on the eleven
  dictations is 13.2% against 3.2-3.4% for Qwen 1.7B with no context. Two things cause it, and neither is the
  streaming:
  - the spoken-form training text: "ventisette B", "slash";
  - the English jargon of a developer's Italian.
  Without the form spans it is still 4.7% against 2.7%. On FLEURS Italian it is 4.5% against 3.5%.
- **What it does better is latency and cost.** It is truly streaming: the final text comes 0.18-0.30 s after the last
  audio on the i9, at 8% of the cores at 1120 ms. Qwen3-ASR's Realtime commit takes 2.4-4.4 s.
- **Nemotron 3 Diarization tells Italian speakers apart** on clean synthetic conversations, although Italian is not
  named in its training data:
  - DER 3.6-4.0% at every geometry, about 98% of turns right;
  - the returning speaker keeps their label in 19-20 of 20.
  Its two failure modes are an extra ghost speaker of a few seconds in 20-30% of conversations, and a merge of two
  same-gender voices that change with no pause.
- **The diarizer's cost is the risk, and the geometry is the lever.**
  - At the card's low geometry it takes about half of a 4-core machine (1.26x real time on 3 threads) and a third of
    the i9. Beside Qwen3-ASR's live partials (0.77 of real time on the i9) the sum passes the machine.
  - A 2.32 s chunk costs 2.6x less at no DER cost.
  - The offline geometry costs 25-47x less than real time.

## Implications

- ADR 0069: Qwen3-ASR stays the speech model, and the speakers come from Nemotron 3 Diarization beside it, ported to
  our own Rust. Nemotron's ASR is rejected on the Italian WER.
- Spec speech/08: the live default is a long-chunk geometry (about 2.3 s), files take the offline geometry, and every
  geometry parameter is a config field. A rule against the ghost speaker is a config field (0 = off). The port's
  weight format is chosen by measurement, since FP32 beat Q8_0 here.
- The diarizer's work ranks below the speech model's windows and commits and above its partials, so on a slow CPU
  the partials thin out first.

## Limits and unknowns

- **Synthetic conversations are easier than real meetings:**
  - every speaker comes from a different recording session;
  - the speech is clean, read parliament;
  - overlap is rare.
  These DERs are not comparable to the card's CALLHOME 10.29%. The AMI real-meeting check was skipped (hours of
  audio at 1.4-2.5x real time).
- VoxPopuli is in the Nemotron ASR's training list, so the conversations' WER is not a quality number.
- **The comparison is not like-for-like:**
  - Qwen's 4.1-4.7x is its batch path, while Nemotron's figures are streaming;
  - Qwen's latency goes through the Realtime route, Nemotron's comes from a bench trace with no route;
  - Qwen's arm B, with context, is not comparable to Nemotron with none.
- **Not measured:**
  - Nemotron's RNNT word boosting (`--speech-context`), which might recover jargon;
  - an Italian inverse text normaliser;
  - option (iii), a per-utterance speaker embedding such as CAM++;
  - the cut at a speaker change through Qwen3-ASR, end to end.
- One i9 and one 4-core container that shares its L3 and memory: a budget 4-core with slower RAM is worse.

## Follow-ups

- Spec speech/08 ports the diarizer and measures it through the route: DER on the same twenty conversations, its CPU
  share beside Qwen3-ASR on the i9 and on the 4-core container, and a small AMI set, reported.
