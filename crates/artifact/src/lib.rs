//! ignis-artifact: `.ninfer` container reader.
//!
//! v1 scope: load the `qwen3_8_27b_nvfp4full-v2.ninfer` artifact, bind every
//! object (unconsumed object = load failure, ADR 0002), and expose tensors +
//! frontend objects (tokenizer / chat template) to `ignis-core`.
//!
//! Ported from the reference stack's `src/artifact/*` module
//! (see `.scratch/kernel-port/spec.md`, issue 02).