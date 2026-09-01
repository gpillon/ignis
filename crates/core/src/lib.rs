//! ignis-core: the engine's scheduling and state layer.
//!
//! v1 scope (docs/design/ignis-v1.md §2):
//! - scheduler: 1 global prefill lane + N decode lanes, full admission state
//!   machine (ADR 0004)
//! - paged KV cache in VRAM + block tables
//! - GDN state management (resumable at checkpoint/frontier boundaries only)
//! - request state machine (admit → prefill → decode → done / evict)
//! - flat C ABI bindings to the kernel leaf (ADR 0001)