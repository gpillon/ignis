//! ignis-core: the engine's scheduling and state layer.
//!
//! v1 scope (`docs/design/ignis-v1.md` §2):
//! - scheduler: 1 global prefill lane + N decode lanes, full admission state
//!   machine (ADR 0004)
//! - paged KV cache in VRAM + block tables
//! - GDN state management (resumable at checkpoint/frontier boundaries only)
//! - request state machine (admit → prefill → decode → done / evict)
//! - the model topology config (`compute::ModelConfig`) the forward pass
//!   will be parameterized by, once the vendored compute adapter lands
//!   (GitHub #39 deleted the superseded flat-C-ABI forward; the
//!   replacement is tracked at `.scratch/ROADMAP.md`, P1-24 / #60)
//! - the model-load step ABI call (`model_load`, feature `cuda`, GitHub
//!   #53 / ADR 0009): builds the bound-tensor + topology descriptors the
//!   kernel leaf's `ignis_model_load` consumes
//! - the degenerate step ABI call (`step`, feature `cuda`, GitHub #54 /
//!   ADR 0009): embedding -> final norm -> output head -> argmax with every
//!   decoder layer skipped (test-only until P1-21/P1-22 add the layer
//!   bodies)
//!
//! The **public contract** (what `ignis-server` and tests code against) is
//! [`types`] + [`scheduler`]: the [`Scheduler`] trait is driven by the
//! server, and the [`Compute`] trait is the only GPU-coupled seam (mocked
//! in CPU tests, ADR 0006). The engine's concrete modules (paged KV, GDN
//! state, request state machine, scheduler, KV-RAM host tier, prefix
//! reuse) are implemented on top of this contract.

pub mod admission;
pub mod compute;
pub mod concrete;
pub mod gdn;
pub mod gpu_profile;
pub mod host;
pub mod kv;
pub mod mock;
#[cfg(feature = "cuda")]
pub mod model_load;
pub mod prefix;
pub mod request;
pub mod scheduler;
#[cfg(feature = "cuda")]
pub mod step;
pub mod types;

pub use admission::{
    ActiveAdmissionSnapshot, AdmissionError, AdmissionProtection, AdmissionResources,
    ProtectionPhase, RetainedLaneCandidate,
};
pub use compute::{LayerKind, ModelConfig};
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
