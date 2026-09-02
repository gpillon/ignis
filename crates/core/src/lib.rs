//! ignis-core: the engine's scheduling and state layer.
//!
//! v1 scope (`docs/design/ignis-v1.md` §2):
//! - scheduler: 1 global prefill lane + N decode lanes, full admission state
//!   machine (ADR 0004)
//! - paged KV cache in VRAM + block tables
//! - GDN state management (resumable at checkpoint/frontier boundaries only)
//! - request state machine (admit → prefill → decode → done / evict)
//! - flat C ABI bindings to the kernel leaf (ADR 0001, `ffi.rs`)
//!
//! The **public contract** (what `ignis-server` and tests code against) is
//! [`types`] + [`scheduler`]: the [`Scheduler`] trait is driven by the
//! server, and the [`Compute`] trait is the only GPU-coupled seam (mocked
//! in CPU tests, ADR 0006). The engine's concrete modules (paged KV, GDN
//! state, request state machine, scheduler, KV-RAM host tier, prefix
//! reuse) are implemented on top of this contract.

pub mod admission;
pub mod concrete;
pub mod ffi;
pub mod gdn;
pub mod host;
pub mod kv;
pub mod mock;
pub mod prefix;
pub mod request;
pub mod scheduler;
pub mod types;

pub use admission::{
    ActiveAdmissionSnapshot, AdmissionError, AdmissionProtection, AdmissionResources,
    ProtectionPhase, RetainedLaneCandidate,
};
pub use concrete::{ConcreteScheduler, SchedulerConfig};
pub use host::{HostEntry, HostError, HostTier, Tier};
pub use mock::MockCompute;
pub use prefix::{PrefixCache, PrefixClaim, PrefixEntry, PrefixId};
pub use request::{Request, admit_candidates, basic_admission};
pub use scheduler::{Compute, DecodeJob, PrefillJob, Scheduler};
pub use types::{
    BackfillClass, ComputeError, DecodeParams, EngineMode, LaneId, N_DECODE_LANES, RequestClass,
    RequestId, RequestInput, RequestState, SchedEvent, SubmitError, TokenId,
};
