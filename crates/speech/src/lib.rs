//! ignis-speech: the speech model's CPU executor, ours in Rust (ADR 0068,
//! spec `docs/specs/speech/01-cpu-executor.md`).
//!
//! Qwen3-ASR on the CPU: our GGUF reader, the activation quantizers and the
//! Q4_K / Q6_K / Q8_0 / F16 / F32 matmul kernels (a scalar reference
//! everywhere, AVX2 kernels picked at run time on x86-64), our interleaved
//! Q4_K layout and thread pool, the AuT encoder and Qwen3 decoder graphs, the
//! log-mel front end, and the incremental API the server drives:
//!
//! ```no_run
//! # use ignis_speech::*;
//! let files = ModelFiles { decoder: "Qwen3-ASR-1.7B-Q4_K_M.gguf".into(), encoder: "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf".into() };
//! let exec = Executor::new(Settings::default(), files, Features::detect());
//! assert_eq!(exec.availability(), Availability::Available);
//! let mut s = exec.open("", None)?;          // loads at first use, prefills the prompt's head
//! s.push(&[0.0f32; 16_000])?;                 // 16 kHz mono samples
//! let p = s.partial()?;                       // stable text + provisional tail
//! let t = s.commit()?;                        // the transcript and its language
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Beside it, [`vad`]: `server_vad`'s voice activity detection, which finds the
//! turns in a Realtime session's audio (spec speech/05); and [`diarizer`]:
//! Nemotron 3 Diarization, who spoke when in that audio (spec speech/08).
//!
//! Provenance: every kernel and op is our own (ADR 0051's first tier). The
//! GGUF block layouts and their quantization rules are the file format the
//! kernels read, not copied code; nothing from ggml or llama.cpp is in the
//! tree or the build. Two modules are adapted ports, each with its provenance
//! in its header and in `NOTICE`: [`vad`], Silero VAD's graph with its
//! weights built in (MIT, spec speech/05), and [`diarizer`], Nemotron 3
//! Diarization's graph and streaming state from the checkpoint's Transformers
//! code (Apache-2.0, spec speech/08), its weights out of the tree.

#[cfg(target_arch = "x86_64")]
pub mod avx2;
pub mod cpu;
pub mod decoder;
pub mod diarizer;
pub mod encoder;
pub mod executor;
pub mod gguf;
pub mod math;
pub mod matmul;
pub mod mel;
pub mod mem;
pub mod model;
pub mod ops;
pub mod pool;
pub mod quant;
pub mod session;
pub mod text;
pub mod vad;

#[cfg(test)]
pub(crate) mod testutil;

pub use cpu::{avx2_dispatches, Features, Isa, NO_AVX2};
pub use executor::{Availability, ExecError, Executor, Loaded, ModelFiles, SessionTuning, Settings, Threads};
pub use model::{LoadError, LoadReport, Model};
pub use session::{Commit, Partial, Session, SessionError, Stats, Transcript};
pub use text::ContextCopy;
