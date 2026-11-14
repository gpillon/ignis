//! The KV pools the VRAM-plan GPU tests pin (GitHub #607), as the bytes a
//! load's plan charges beside them, and the pool those bytes leave.
//!
//! A plan's KV pool takes the whole pages its budget leaves once every other
//! line is placed. All but two of those lines are closed forms of the load's
//! shape on this build: the artifact, the workspaces, the lanes and rounds,
//! the residual window and the measured residual, the pool's own block
//! tables. The other two are the card's: the CUDA context, which the process
//! measures when it makes it (GitHub #604), and differs by card, driver, OS
//! and process, and the trellis workspace, which the kernel sizes for the
//! load's widest call by the card's multiprocessors. A test pins the shape's
//! bytes exactly and reads the card's two lines off the plan, so its pool is
//! exact on any card and no longer moves a page when the context does; on
//! the RTX 5090 the pin holds the trellis workspace too, to the byte.
//! Included via `#[path]` by the binaries that pin one; the `kv_pool_pins`
//! test binary checks the arithmetic on the CPU.

// Each binary uses a subset of it.
#![allow(dead_code)]

use ignis_core::kv_format::KV_PAGE_TOKENS;
use ignis_core::{KvFormat, KvGeometry};

/// One KV page of the 27B under hq-e8-2b.
pub fn page_bytes() -> u64 {
    KvFormat::HqE8_2b.page_bytes(KvGeometry::qwen38_27b())
}

/// The `name` attribute of the plan `lines` (the `ignis.runtime.vram_plan`
/// attributes).
fn read(lines: &serde_json::Value, name: &str) -> u64 {
    lines[name].as_u64().unwrap_or_else(|| panic!("the plan's {name}: {lines}"))
}

/// The lines of a plan the card decides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CardLines {
    /// `cuda_context_bytes`: the context the process measured, or
    /// `CUDA_CONTEXT_BYTES` where it could not.
    pub cuda_context: u64,
    /// `trellis_workspace_bytes`: the load's widest trellis call, split by
    /// the card's multiprocessors.
    pub trellis_workspace: u64,
}

impl CardLines {
    /// The card's lines of the plan `lines`.
    pub fn of(lines: &serde_json::Value) -> CardLines {
        CardLines { cuda_context: read(lines, "cuda_context_bytes"), trellis_workspace: read(lines, "trellis_workspace_bytes") }
    }

    fn bytes(self) -> u64 {
        self.cuda_context + self.trellis_workspace
    }
}

/// The bytes the plan `lines` charges beside its KV pool's pages and the
/// card's lines: what a [`PinnedLoad`] holds, to the byte.
pub fn shape_bytes(lines: &serde_json::Value) -> u64 {
    (read(lines, "kv_pool_pages") * page_bytes())
        .checked_add(CardLines::of(lines).bytes())
        .and_then(|beside| read(lines, "total_bytes").checked_sub(beside))
        .unwrap_or_else(|| panic!("the plan's total holds its pool and the card's lines: {lines}"))
}

/// Whether the card is the one the pins' trellis workspaces were read on: an
/// RTX 5090 at its own multiprocessor count (no `IGNIS_TRELLIS_SM_CAP`).
fn on_the_rtx_5090() -> bool {
    #[cfg(feature = "cuda")]
    {
        std::env::var_os("IGNIS_TRELLIS_SM_CAP").is_none_or(|cap| cap.is_empty())
            && ignis_artifact::CudaDevice::identity(0).is_ok_and(|(name, _)| name == "NVIDIA GeForce RTX 5090")
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Holds the plan `lines` to `pinned`, printing what it reads: its budget,
/// the bytes it charges beside the pool and the card's lines to the byte, its
/// pool the whole pages those leave beside the card's own lines, and on the
/// RTX 5090 its trellis workspace.
pub fn assert_pinned(lines: &serde_json::Value, pinned: PinnedLoad, what: &str) {
    let card = CardLines::of(lines);
    let pool = read(lines, "kv_pool_tokens");
    let rtx_5090 = on_the_rtx_5090();
    println!(
        "{what}: {} bytes beside the pool and the card's lines, a CUDA context of {} bytes and a trellis workspace \
         of {} ({}): {} tokens expected, {pool} planned",
        shape_bytes(lines),
        card.cuda_context,
        card.trellis_workspace,
        if rtx_5090 { "the RTX 5090's, pinned" } else { "this card's" },
        pinned.pool_tokens(card)
    );
    assert_eq!(read(lines, "budget_bytes"), pinned.budget, "{what}: the budget");
    assert_eq!(shape_bytes(lines), pinned.shape_bytes, "{what}: the bytes the plan charges beside the pool and the card's lines");
    if rtx_5090 {
        assert_eq!(card.trellis_workspace, pinned.rtx_5090_trellis_workspace, "{what}: the RTX 5090's trellis workspace");
    }
    assert_eq!(pool, pinned.pool_tokens(card), "{what}: the pool they leave beside the card's lines");
}

/// A load as its pin holds it: its budget, the bytes its plan charges beside
/// the KV pool's pages and the card's lines, and its trellis workspace on the
/// RTX 5090.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PinnedLoad {
    pub budget: u64,
    pub shape_bytes: u64,
    pub rtx_5090_trellis_workspace: u64,
}

impl PinnedLoad {
    /// The pages of the pool this load plans beside `card`.
    pub fn pool_pages(self, card: CardLines) -> u64 {
        self.budget.saturating_sub(self.shape_bytes + card.bytes()) / page_bytes()
    }

    /// [`PinnedLoad::pool_pages`] in tokens.
    pub fn pool_tokens(self, card: CardLines) -> u64 {
        self.pool_pages(card) * u64::from(KV_PAGE_TOKENS)
    }

    /// The card's lines of this load on the RTX 5090 beside a context of
    /// `cuda_context` bytes.
    pub fn on_the_rtx_5090(self, cuda_context: u64) -> CardLines {
        CardLines { cuda_context, trellis_workspace: self.rtx_5090_trellis_workspace }
    }
}

/// `rtx-24gb` in the 24 GiB budget spec #474 measured the 24 GB class in, on
/// the routes an 8.x card runs unset, its solo width at the 2,048-token chunk
/// (GitHub #508): read off main's plan on 2026-11-14.
pub const RTX_24GB_AT_THE_CHUNK: PinnedLoad =
    PinnedLoad { budget: 25_769_803_776, shape_bytes: 14_748_563_076, rtx_5090_trellis_workspace: 64_252_160 };

/// `rtx-16gb` in 16 GiB less 1 GiB, on the routes an 8.x card runs unset, its
/// solo width at the 1,024-token chunk (GitHub #489): read off main's plan on
/// 2026-11-14.
pub const RTX_16GB_AT_THE_CHUNK: PinnedLoad =
    PinnedLoad { budget: 16_106_127_360, shape_bytes: 12_063_478_444, rtx_5090_trellis_workspace: 32_134_400 };

/// The 16 GB artifact's shape under `rtx50-16gb` (262,144 tokens on two lanes,
/// the 1,024-token chunk), on its own card's routes: read off main's plans on
/// 2026-11-14, the same in 16 GiB and in 16 GiB less 1 GiB.
pub const RTX50_16GB_SHAPE: u64 = 11_997_942_444;

/// That shape's trellis workspace on the RTX 5090: `mlp/down` at the chunk.
pub const RTX50_16GB_RTX_5090_TRELLIS_WORKSPACE: u64 = 36_225_280;

/// That shape in the 16 GiB budget of a 16 GB card (#339 A1).
pub const RTX50_16GB: PinnedLoad = PinnedLoad {
    budget: 17_179_869_184,
    shape_bytes: RTX50_16GB_SHAPE,
    rtx_5090_trellis_workspace: RTX50_16GB_RTX_5090_TRELLIS_WORKSPACE,
};

/// That shape less 1 GiB of simulated desktop (#340): nothing adapts.
pub const RTX50_16GB_LESS_1_GIB: PinnedLoad = PinnedLoad { budget: 17_179_869_184 - (1 << 30), ..RTX50_16GB };

/// That shape less 3 GiB, adapted to 65,536 tokens on two lanes (#340): every
/// line is the 262,144-token load's but the pool's block tables, which hold
/// 65,536 tokens on each lane, 24,576 bytes fewer, as the plan read on the
/// RTX 5090 on 2026-11-14.
pub const RTX50_16GB_LESS_3_GIB: PinnedLoad =
    PinnedLoad { budget: 17_179_869_184 - (3 << 30), shape_bytes: RTX50_16GB_SHAPE - 24_576, ..RTX50_16GB };

/// A plan refused below its floor, as its pin holds it: its budget, the bytes
/// it needs (`VramPlanError::BelowMinimum`'s `needed_bytes`) beside the card's
/// lines, its pool included, and its trellis workspace on the RTX 5090.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PinnedRefusal {
    pub budget: u64,
    pub needed_bytes: u64,
    pub rtx_5090_trellis_workspace: u64,
}

impl PinnedRefusal {
    /// The whole KV pages this plan misses its budget by beside `card`; 0
    /// where it fits.
    pub fn pages_short(self, card: CardLines) -> u64 {
        (self.needed_bytes + card.bytes()).saturating_sub(self.budget).div_ceil(page_bytes())
    }

    /// The card's lines of this plan on the RTX 5090 beside a context of
    /// `cuda_context` bytes.
    pub fn on_the_rtx_5090(self, cuda_context: u64) -> CardLines {
        CardLines { cuda_context, trellis_workspace: self.rtx_5090_trellis_workspace }
    }
}

/// The ladder's 131,072-token step less 3 GiB: one sequence and the spare
/// reservation beside it (`vram.adapt_spare_tokens`, 65,536), 3,072 pages,
/// read off main's refusal on 2026-11-14.
pub const RTX50_16GB_LESS_3_GIB_131072_SPARE: PinnedRefusal = PinnedRefusal {
    budget: 17_179_869_184 - (3 << 30),
    needed_bytes: 13_809_865_388,
    rtx_5090_trellis_workspace: RTX50_16GB_RTX_5090_TRELLIS_WORKSPACE,
};
