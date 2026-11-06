//! Hardware profiles (spec config-v2/02 §`--profile`): a named bundle of
//! field defaults for one kind of card, slotted between the config file and
//! the hardcoded defaults — a smarter default, never an override.
//!
//! The built-in profiles are written below in the config file's own
//! `profiles:` shape and read by the same [`super::file::read_document`] a
//! file's section is, so a profile an operator defines for a card the
//! maintainers do not own resolves exactly as a built-in one does: there is
//! one code path, and a built-in profile is only the default contents of it.

use std::collections::BTreeMap;

use super::arch::{BuiltArchitectures, ComputeCapability, GpuGeneration};
use super::file::read_document;
use super::source::Layer;
use super::ConfigError;

/// The profile used when nothing names one: [`AUTO_PROFILE`], which picks the
/// built-in profile by the card's total memory ([`CardClass::profile`]). An
/// unset `--profile` is this one, not "no profile" — for that, name
/// [`NONE_PROFILE`].
pub const DEFAULT_PROFILE: &str = AUTO_PROFILE;

/// `--profile auto` / `IGNIS_PROFILE=auto` (ADR 0063, spec 27b-16gb/04):
/// the built-in profile of the card's [`CardClass`], read from NVML's total
/// memory at the start. Not a name [`lookup`] resolves: `gather` chooses
/// the real profile first and names that one, so `GET /v1/config`, `make
/// config` and the Playground show `rtx5090`, never `auto`. It picks the
/// class's profile by NAME, through the same [`lookup`] a named profile goes
/// through: a config file's `profiles:` that redefines `rtx-16gb`,
/// `rtx-24gb`, `rtx5090` or `rtxpro6000` replaces what `auto` picks for that
/// class.
pub const AUTO_PROFILE: &str = "auto";

/// What the start could read of the card `--profile auto` chooses by. Injected
/// into config resolution, like the environment: resolution runs in contexts
/// with no card (CI on Linux without NVML, the CPU tests), so the module never
/// reads one itself. The `config` and `model` verbs read it exactly as a
/// start does (`config generate|print|patch`, `model list|download`), except
/// that the `model` verbs, which need no card, take [`Card::Unreadable`] for
/// [`Card::Absent`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Card {
    /// The card the start read ([`CardReading`]).
    Read(CardReading),
    /// A build with no CUDA (the mock engine, CI): there is no card to read,
    /// and `auto` resolves as `none` would — the hardcoded defaults.
    Absent,
    /// A CUDA build that could not read the total, and why (NVML's error):
    /// `auto` refuses, naming the reason and `--profile`.
    Unreadable(String),
}

/// What the start read of a card (GitHub #457): its total memory, which
/// `--profile auto` chooses by, and what it is, which decides whether this
/// build's kernels run on it at all ([`super::arch`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CardReading {
    /// The card's total memory in bytes, as NVML reports it.
    pub total: u64,
    /// Its name, as NVML reports it ("NVIDIA GeForce RTX 5090").
    pub name: String,
    /// Its compute capability, as the CUDA runtime reports it.
    pub capability: ComputeCapability,
    /// The architectures this build's kernels were compiled for.
    pub built_for: BuiltArchitectures,
}

impl CardReading {
    /// The card's class, by its total ([`CardClass::of_total`]); `None` for a
    /// total of no class.
    pub fn class(&self) -> Option<CardClass> {
        CardClass::of_total(self.total)
    }

    /// The card's GPU generation, by its compute capability
    /// ([`GpuGeneration::of`]); `None` for a capability of none.
    pub fn generation(&self) -> Option<GpuGeneration> {
        GpuGeneration::of(self.capability)
    }
}

impl Card {
    /// The card's total memory when it was read.
    pub fn total(&self) -> Option<u64> {
        self.reading().map(|reading| reading.total)
    }

    /// What was read of the card, when it was.
    pub fn reading(&self) -> Option<&CardReading> {
        match self {
            Self::Read(reading) => Some(reading),
            Self::Absent | Self::Unreadable(_) => None,
        }
    }

    /// A Blackwell card (compute capability 12.0) of `total` bytes under the
    /// default build, which its kernels run on: what the CPU tests and the
    /// callers with no card to read inject.
    pub fn blackwell(total: u64) -> Card {
        Card::Read(CardReading {
            total,
            name: "NVIDIA RTX (Blackwell)".to_owned(),
            capability: ComputeCapability::new(12, 0),
            built_for: BuiltArchitectures::default_build(),
        })
    }

    /// The refusal of a start on this card (GitHub #457): a card the build's
    /// kernels do not run on, named with its compute capability and the
    /// capabilities they do run on. Nothing to refuse on a card they run on,
    /// or with no card read (the mock build; NVML that cannot be read is
    /// `--profile auto`'s to refuse).
    pub fn refusal(&self) -> Option<String> {
        self.reading().and_then(|card| super::arch::uncovered(&card.name, card.capability, &card.built_for))
    }
}

/// The total memory of an RTX 5090 as NVML reports it (32,607 MiB): what the
/// CPU test helpers inject so that every test resolves `rtx5090` as before.
pub const RTX_5090_TOTAL_BYTES: u64 = 32_607 << 20;

/// The profile `--profile auto` chooses for `card`: its class's built-in
/// profile (`Some(name)`), no profile at all on a build with no card
/// (`None`), or a refusal naming the total and `--profile`.
pub fn auto(card: Card) -> Result<Option<&'static str>, ConfigError> {
    match card {
        Card::Absent => Ok(None),
        Card::Unreadable(why) => Err(ConfigError(format!(
            "`--profile auto` could not read the card's total memory ({why}): name a profile with `--profile` or IGNIS_PROFILE ({}, or `none`)",
            built_in_names()
        ))),
        Card::Read(CardReading { total, .. }) => CardClass::of_total(total).map(|class| Some(class.profile())).ok_or_else(|| {
            ConfigError(format!(
                "`--profile auto`: a card of {:.1} GiB is no card class ({}): name a profile with `--profile` or IGNIS_PROFILE ({}, or `none`)",
                total as f64 / (1u64 << 30) as f64,
                class_bounds(),
                built_in_names()
            ))
        }),
    }
}

/// The boundaries of every class, spelled from the `CARD_CLASS_*` constants:
/// "16 GB below 20 GiB, 24 GB from 20 to under 28 GiB, 32 GB from 28 to 40
/// GiB, 96 GB from 88 to 104 GiB".
fn class_bounds() -> String {
    CardClass::ALL.into_iter().map(|class| format!("{class} {}", class.totals())).collect::<Vec<_>>().join(", ")
}

/// The sentence `--help` gives `--profile auto`: each class's profile by its
/// own name, built from [`CardClass::ALL`] and [`CardClass::profile`].
pub fn classes_text() -> String {
    CardClass::ALL.into_iter().map(|class| format!("{class}: {}", class.profile())).collect::<Vec<_>>().join(", ")
}

fn built_in_names() -> String {
    built_in().keys().cloned().collect::<Vec<_>>().join(", ")
}

/// `--profile none` / `IGNIS_PROFILE=none`: skip the profile layer
/// entirely, hardcoded defaults only. Not a name [`lookup`] resolves — a
/// config file's `profiles:` may not define one by this name either (nor one
/// called [`AUTO_PROFILE`]), since it would never be reachable (`gather`
/// checks for both first): `read_document` refuses it.
pub const NONE_PROFILE: &str = "none";

/// The built-in profiles, as a config file's `profiles:` section.
///
/// `rtx5090` is the owner's own card, measured end to end on it (2026-10-10):
/// the 27B's leg pushes context and turns vision on, found by raising
/// `model.max_context` under `--vision-enabled` until the VRAM plan refused,
/// then trading `vram.headroom_bytes` down from the 1536 MiB hardcoded
/// default against `spec.backend dflash2` staying on — the owner's own
/// choice over a `vram.headroom_bytes` trade that would have reached ~864K
/// tokens with speculation off instead. Re-measured the same way on
/// 2026-10-21 (GitHub #335), once the prompt planes were gone and the token
/// embedding had moved to host memory: 917,504 holds, 950,272 is refused by
/// 59 MB (finding 2026-10-21-prefill-attention-without-prompt-planes). The
/// 27B prefills 2,048 tokens a chunk (GitHub #341): with vision on the
/// encoder sizes the workspace arena, so the plan is the same byte for byte
/// as at 1,024, and against 1,024 it measured +1.9% / +2.2% cold prefill at
/// 46K / 105K, the same at 8K / 14K, a lower 4K KLD and decode unchanged; a
/// decoding lane's longest pause while a prompt prefills grows ~65% (finding
/// 2026-10-21-the-27b-prefill-chunk-at-2048). Flash-Next's leg holds its decode
/// lanes at their own default (3), in its own section since the 27B takes the
/// field too and keeps its 8 (GitHub #336), and raises `model.max_context` to
/// 512K + 32K + 32K tokens, the floor the owner asked for on both the expert
/// cache (>= 15 GiB; it measured ~16.6 GiB) and the context. Past the
/// checkpoint's trained 262,144 positions the 27B's text rotary table needs
/// YaRN, which its leg does not spell out: `model.rope_scaling` defaults to
/// `auto`, the factor `max_context / 262,144` (3.5 at 917,504). Flash-Next
/// resolves the same way over its 589,824: 2.25 (ADR 0057, ADR 0059).
/// `vram.headroom_bytes`'s 1536 MiB hardcoded
/// default (finding 2026-10-09-vram-headroom-wddm-paging) still holds for
/// Flash-Next, which never needed to trade it down. Flash-Next also pins its
/// calibration-hottest 2 GiB of experts in the expert cache
/// (`vram.pinned_expert_bytes`, the owner's call, ADR 0055): measured on the
/// card (finding 2026-10-18-pinned-hot-set-measured-on-the-5090), that gives
/// back 2.149 GB of the host pool's RAM for 0.1-0.2% of one-lane and
/// 0.2-1.1% of three-lane decode (two sessions), greedy text unchanged; `4G`
/// would give back 4.296 GB for 4.3% and 5.9%. The field is
/// Flash-Next-only: the 27B passes it over. Flash-Next prefills 8,192 tokens a
/// chunk (GitHub #489), as `make` always started it: each chunk streams its
/// experts once (finding 2026-10-07-flash-next-prefill-chunk-and-decode-share),
/// and the wider scratch costs the expert cache 1.03 GiB against 1,024
/// (1,100,985,088 bytes, planned on the card on 2026-11-06), so from the ~16.6
/// GiB measured at 1,024 it keeps ~15.6, above the 15 GiB asked for.
///
/// `rtxpro6000` is the RTX PRO 6000 Blackwell series (Workstation, Max-Q,
/// Server Edition: 96 GB each, sm_120, the 5090's architecture). Derived
/// first (2026-10-17), then measured on a Server Edition under Linux; the
/// numbers are in the finding 2026-10-18-rtxpro6000-profile-measured. The
/// measurement kept the derived choices and added one. Both families run at
/// 1,048,576 tokens of context, the 27B's whole hq-e8-2b attention envelope
/// (the plan holds it; no request that long was run): `auto` rope scaling
/// makes both models' tables YaRN at 4 over the 262,144 trained positions
/// (ADR 0057, ADR 0059). The 27B leaves its headroom at the hardcoded
/// default, since nothing has to be traded for vision any more, and drafts
/// with DFlash2 at 7 tokens, which beat 4 in every cell measured but one,
/// and prefills 2,048 tokens a chunk, as on the 5090 (GitHub #341).
/// Flash-Next runs 8 decode lanes, where its aggregate peaks: its expert
/// cache holds nearly every expert, so a lane costs its state and its share
/// of the KV pool. It drafts with its MTP head, 3 tokens a lane, which the
/// 5090 leaves off because decode there is bound by the experts it copies
/// over PCIe (finding 2026-10-07-flash-next-mtp-speculation-is-pcie-bound);
/// here MTP pays at one to four active lanes and with simulated agents at
/// four and eight. The row budget stays the default 8, where every wider
/// one lost on this card (finding 2026-10-18-verify-rows-on-the-rtx-pro-6000),
/// so 3 drafts at one or two lanes, 1 at three or four, none from five on. Its
/// n-gram hot rows are `auto`, the measured addition: the whole table in RAM
/// keeps the disk out of every decode round, and `auto` takes it wherever
/// the host plan leaves room, never refusing a start the 1 GiB default would
/// make. The VRAM plan sizes the rest, the KV pool among it.
///
/// `rtx-16gb` is the 16 GB class (ADR 0063, ADR 0064; named `rtx50-16gb` until
/// GitHub #476, a name [`canonical`] still accepts): the RTX 5060 Ti 16 GB,
/// 5070 Ti and 5080, the RTX 50 cards `--profile auto` finds below 20 GiB.
/// Derived, not measured: no 16 GB card exists in the project. It serves the 27B
/// (the 16 GB catalog entry) at the 262,144 positions it was trained on, with
/// DFlash2 at 7 tokens, as the other profiles, and vision on, explicit
/// (ADR 0063 as amended by ADR 0075): at a 4,096-token envelope, an item bound
/// of about 2,048 x 2,048 pixels, and residency `host`, written out so the
/// profile table shows it. The tower sits in pinned host RAM and the pool is
/// host-mapped, so a load with no image in flight keeps the text-only KV pool
/// (7,956 pages at 1,024 and at 4,096, measured on the 5090 on a 16 GB budget).
/// `vram.qwen38.headroom_bytes` is the 5090's 1200M
/// until a real card says otherwise, and `prefill_chunk` stays at its 1,024: the
/// prefill scratch is sized by the chunk, and 2,048 would cost ~0.4 GB of
/// KV pool. The profile names no model: a start with no artifact resolves the
/// default id `qwen3.8-27b` through the catalog's class-aware alias to the 16 GB
/// entry. The 27B decodes two lanes, where eight would leave no room for its
/// context, and leaves the drafter's head to `auto`, the full head (ADR 0063).
/// Every one of these is a default the load adapts when the free memory it
/// finds cannot hold it: the head, then the context down to 65,536, then one
/// lane (ADR 0063 §3). On a 16 GB RTX 40 card the routes it runs unset since
/// spec #503 add the dense prefill's 89 MB buffer and take less prefill
/// scratch: on the 5090 at a 16 GB budget less 1 GiB (2026-11-05) they left the
/// KV pool 6,656 tokens smaller, 386,048 against 392,704, and the context
/// loaded with nothing adapted. The 27B's solo width (GitHub #489) is pinned
/// to the chunk, the owner's call (2026-11-06): the KV pool before a wider
/// chunk. On that load, with an RTX 4080 as the card read, `auto` would keep
/// 2,048 (4,096 leaves less than the spare reservation past the context), its
/// pool 352,768 tokens against 386,048. Flash-Next's `prefill_chunk` 2,048 is
/// a placeholder no card of the class has measured (GitHub #526).
///
/// `rtx-24gb` is the 24 GB class (GitHub #480, spec #474): the RTX 3090 and
/// RTX 4090, the cards `--profile auto` finds from 20 GiB up to 28 GiB. It
/// takes the RTX 5090 profile's 27B values: `vram.qwen38.headroom_bytes`
/// 1200M, a 2,048-token prefill chunk, DFlash2 at 7 tokens, vision on with
/// residency `host` at the default envelope, and the decode lanes (8) and the
/// drafter's head at their defaults. Like `rtx-16gb` it names no model: no
/// built-in entry states the 24 GB class, the 32 GB ones stay above it and the
/// 16 GB ones fit it, so a start with no artifact resolves the default id
/// `qwen3.8-27b` through the catalog's class-aware alias to the 16 GB entry.
/// Its context is measured (2026-11-04) by the simulation `rtx-16gb`'s numbers
/// came from: the 16 GB entry on the RTX 5090 under a 24 GiB budget
/// (`--vram-budget-bytes 25769803776`), with the FP16 prompt band
/// (`--model-prefill-attention fp16`, the band a card below compute capability
/// 12.0 runs), hq-e8-2b KV and the 5090 profile's 27B values. 1,048,576 tokens
/// load with nothing adapted (`adapted: ""`), 8 decode lanes and a KV pool of
/// 18,735 pages (1,199,040 tokens); 1,048,640 is refused at the start by the
/// 27B's hq-e8-2b attention envelope, so the context is capped by that
/// envelope, as on `rtxpro6000`, not by memory. `auto` rope scaling makes the
/// 27B's table YaRN at 4 over its 262,144 trained positions (ADR 0057). The
/// pool's 150,464 tokens past the context (~1.39 GB) are what a real card's
/// desktop may take before the load adapts, as it adapts every value here
/// that the free memory it finds cannot hold (ADR 0063 §3); the closing runs
/// on an RTX 3090 and an RTX 4090 confirm the context (spec #474). The 27B
/// also keeps [`RTX_24GB_RETAINED_DEVICE_SLOTS`] retained slots on the device
/// (GitHub #508), written in the text below by that constant's name, with
/// prompt reuse on; with them the same load keeps 17,946 pages (1,148,544
/// tokens), 99,968 past the context (~0.92 GB). On the routes a card below
/// compute capability 12.0 runs unset since spec #503 (the dense prefill and
/// its 89 MB buffer, the integer band and our GDN prefill, named on the 5090:
/// `--model-prefill-linear int8 --model-prefill-attention int8` and
/// `IGNIS_GDN_PREFILL=ours`, 2026-11-05) it still loads 1,048,576 tokens on 8
/// lanes with nothing adapted, the slots beside them, and keeps 17,809 pages
/// (1,139,776 tokens), 91,200 past the context (~0.84 GB). The 27B's solo
/// width (GitHub #489) is pinned to the 2,048-token chunk, the owner's call
/// (2026-11-06); on that simulation, with an RTX 3090 as the card read, `auto`
/// would keep 4,096 for 13,312 of those tokens (8,192 does not fit), which
/// #516's session can weigh on a real card. Flash-Next's `prefill_chunk` 4,096
/// is a placeholder no card of the class has measured (GitHub #526). No other
/// card ships a profile; a config file's `profiles:` defines one for any other
/// card.
const BUILT_IN: &str = r#"{
    "rtx5090": {
        "vram": {
            "headroom_bytes": "1536M",
            "pinned_expert_bytes": "2G",
            "qwen38": { "headroom_bytes": "1200M" }
        },
        "model": {
            "qwen38": { "max_context": 917504, "prefill_chunk": 2048 },
            "qwen38flashnext": { "max_context": 589824, "prefill_chunk": 8192 }
        },
        "vision": { "enabled": true, "residency": "host" },
        "spec": {
            "qwen38": { "backend": "dflash2", "draft_tokens": 7 },
            "qwen38flashnext": { "decode_lanes": 3 }
        }
    },
    "rtxpro6000": {
        "model": {
            "qwen38": { "max_context": 1048576, "prefill_chunk": 2048 },
            "qwen38flashnext": { "max_context": 1048576, "prefill_chunk": 8192 }
        },
        "vision": { "enabled": true, "residency": "device" },
        "spec": {
            "qwen38": { "backend": "dflash2", "draft_tokens": 7 },
            "qwen38flashnext": { "backend": "mtp", "draft_tokens": 3, "decode_lanes": 8 }
        },
        "ngram": { "hot_bytes": "auto" }
    },
    "rtx-16gb": {
        "vram": {
            "qwen38": { "headroom_bytes": "1200M" }
        },
        "model": {
            "qwen38": { "max_context": 262144, "prefill_chunk_solo": 1024 },
            "qwen38flashnext": { "prefill_chunk": 2048 }
        },
        "vision": { "enabled": true, "max_tokens": 4096, "residency": "host" },
        "spec": {
            "qwen38": { "backend": "dflash2", "draft_tokens": 7, "decode_lanes": 2, "draft_head": "auto" }
        }
    },
    "rtx-24gb": {
        "vram": {
            "qwen38": { "headroom_bytes": "1200M" }
        },
        "model": {
            "qwen38": { "max_context": 1048576, "prefill_chunk": 2048, "prefill_chunk_solo": 2048 },
            "qwen38flashnext": { "prefill_chunk": 4096 }
        },
        "reuse": {
            "qwen38": { "retained_device": RTX_24GB_RETAINED_DEVICE_SLOTS }
        },
        "vision": { "enabled": true, "residency": "host" },
        "spec": {
            "qwen38": { "backend": "dflash2", "draft_tokens": 7 }
        }
    }
}"#;

/// A **card class** (ADR 0063, ADR 0064, `GLOSSARY.md`): the cards one
/// built-in profile serves, named by their memory, and the smallest a
/// catalog entry loads on. Ordered by size, so `entry > card` reads "the
/// entry needs a larger card than this one".
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CardClass {
    Gb16,
    Gb24,
    Gb32,
    Gb96,
}

/// The total memories each [`CardClass`] covers (spec 27b-16gb/04, spec #474),
/// as NVML reports a card's total: below 20 GiB is the 16 GB class (a 16 GB
/// card reports ~15.9 GiB), from 20 GiB up to, but not including, 28 GiB the
/// 24 GB one (the RTX 3090 and RTX 4090, 24 GiB), 28-40 GiB the 32 GB one
/// (the 5090 ~31.8 GiB), 88-104 GiB the 96 GB one (the RTX PRO 6000 ~95.6
/// GiB). Any other total is no class: `--profile auto` refuses it, naming
/// `--profile`.
pub const CARD_CLASS_16GB_BELOW_BYTES: u64 = 20 << 30;
pub const CARD_CLASS_24GB_BYTES: std::ops::Range<u64> = CARD_CLASS_16GB_BELOW_BYTES..*CARD_CLASS_32GB_BYTES.start();
pub const CARD_CLASS_32GB_BYTES: std::ops::RangeInclusive<u64> = (28 << 30)..=(40 << 30);
pub const CARD_CLASS_96GB_BYTES: std::ops::RangeInclusive<u64> = (88 << 30)..=(104 << 30);

impl CardClass {
    /// Every class, smallest first.
    pub const ALL: [CardClass; 4] = [Self::Gb16, Self::Gb24, Self::Gb32, Self::Gb96];

    /// The class a card whose total memory is `total_bytes` belongs to, by
    /// the boundaries above; `None` for a total between or past them.
    pub fn of_total(total_bytes: u64) -> Option<Self> {
        if total_bytes < CARD_CLASS_16GB_BELOW_BYTES {
            Some(Self::Gb16)
        } else if CARD_CLASS_24GB_BYTES.contains(&total_bytes) {
            Some(Self::Gb24)
        } else if CARD_CLASS_32GB_BYTES.contains(&total_bytes) {
            Some(Self::Gb32)
        } else if CARD_CLASS_96GB_BYTES.contains(&total_bytes) {
            Some(Self::Gb96)
        } else {
            None
        }
    }

    /// The total memories the class covers, spelled from the `CARD_CLASS_*`
    /// constants: "below 20 GiB", "from 20 to under 28 GiB", "from 28 to 40
    /// GiB", "from 88 to 104 GiB".
    pub fn totals(self) -> String {
        let gib = |bytes: u64| bytes >> 30;
        match self {
            Self::Gb16 => format!("below {} GiB", gib(CARD_CLASS_16GB_BELOW_BYTES)),
            Self::Gb24 => format!("from {} to under {} GiB", gib(CARD_CLASS_24GB_BYTES.start), gib(CARD_CLASS_24GB_BYTES.end)),
            Self::Gb32 => format!("from {} to {} GiB", gib(*CARD_CLASS_32GB_BYTES.start()), gib(*CARD_CLASS_32GB_BYTES.end())),
            Self::Gb96 => format!("from {} to {} GiB", gib(*CARD_CLASS_96GB_BYTES.start()), gib(*CARD_CLASS_96GB_BYTES.end())),
        }
    }

    /// The class's size in GB, as a catalog entry spells it (`card_class: 16`).
    pub fn gb(self) -> u32 {
        match self {
            Self::Gb16 => 16,
            Self::Gb24 => 24,
            Self::Gb32 => 32,
            Self::Gb96 => 96,
        }
    }

    /// The class a catalog entry's `card_class` names; `None` for any other
    /// number.
    pub fn from_gb(gb: u32) -> Option<Self> {
        Self::ALL.into_iter().find(|class| class.gb() == gb)
    }

    /// The built-in profile `--profile auto` picks for a card of this class.
    pub fn profile(self) -> &'static str {
        match self {
            Self::Gb16 => "rtx-16gb",
            Self::Gb24 => "rtx-24gb",
            Self::Gb32 => "rtx5090",
            Self::Gb96 => "rtxpro6000",
        }
    }
}

impl std::fmt::Display for CardClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} GB", self.gb())
    }
}

/// The 16 GB profile's name before GitHub #476: still accepted everywhere a
/// profile is named, permanently and without a warning.
pub const LEGACY_16GB_PROFILE: &str = "rtx50-16gb";

/// The name a profile is known by: [`LEGACY_16GB_PROFILE`] is `rtx-16gb`,
/// every other name is itself. `--profile`, `IGNIS_PROFILE`, a file's
/// `profile:` and the keys of its `profiles:` all go through it.
pub fn canonical(name: &str) -> &str {
    if name == LEGACY_16GB_PROFILE {
        CardClass::Gb16.profile()
    } else {
        name
    }
}

/// The device retained slots `rtx-24gb` gives the 27B (GitHub #508, spec
/// #503): its prompt checkpoints and prefix images are then copied device to
/// device, off the PCIe path a host slot's capture and claim take. A slot is a
/// lane's state and its hq-e8-2b residual window, ~232 MB at the profile's
/// shape, and the KV pool pays for it: the profile's context is capped by the
/// 27B's attention envelope, not by memory, and #474's 24 GiB load left 150,464
/// tokens (~1.39 GB) of pool past it. Measured on the RTX 5090 (2026-11-05,
/// `trellis_16g_gpu`'s 24 GiB test, #474's flags): with two slots the 16 GB
/// entry still loads at 1,048,576 tokens on 8 lanes with nothing adapted, the
/// slots a 393,760,768-byte line, the pool 17,946 pages (1,148,544 tokens),
/// 99,968 past the context. Were the context to drop, the value is 1. The
/// 16 GB profiles keep none: their context is set by memory.
pub const RTX_24GB_RETAINED_DEVICE_SLOTS: u32 = 2;

/// Every built-in profile, by name: [`BUILT_IN`], its constants' names
/// replaced by their values.
pub fn built_in() -> BTreeMap<String, Layer> {
    let profiles = BUILT_IN.replace("RTX_24GB_RETAINED_DEVICE_SLOTS", &RTX_24GB_RETAINED_DEVICE_SLOTS.to_string());
    let value = serde_json::from_str(&format!(r#"{{"profiles": {profiles}}}"#)).expect("the built-in profiles are JSON");
    read_document(&value, "the built-in profiles").expect("the built-in profiles name only declared fields").profiles
}

/// The profile called `name`: the config file's own first (an operator may
/// redefine a built-in one for their card), then the built-in ones.
pub fn lookup(name: &str, from_file: &BTreeMap<String, Layer>) -> Result<Layer, ConfigError> {
    let name = canonical(name);
    if let Some(layer) = from_file.get(name) {
        return Ok(layer.clone());
    }
    let built_in = built_in();
    built_in.get(name).cloned().ok_or_else(|| {
        let mut known: Vec<&str> = built_in.keys().map(String::as_str).collect();
        known.extend(from_file.keys().map(String::as_str));
        known.sort_unstable();
        known.dedup();
        ConfigError(format!("no profile `{name}` (the profiles are {})", known.join(", ")))
    })
}

/// The cards the CPU tests inject (spec #474), as the start reads them:
/// by total memory, name and compute capability, under the release's build
/// (`86;120a`), whose kernels run on every GPU generation. The integration
/// tests have their own copy, `tests/support/card.rs`.
#[cfg(test)]
pub(crate) mod testing {
    use super::{CardClass, CardReading, RTX_5090_TOTAL_BYTES};
    use crate::config::arch::{BuiltArchitectures, ComputeCapability};

    /// A card of `total` bytes named `name`, of compute capability
    /// `major.minor`.
    pub(crate) fn reading(total: u64, name: &str, major: u32, minor: u32) -> CardReading {
        CardReading {
            total,
            name: name.to_owned(),
            capability: ComputeCapability::new(major, minor),
            built_for: BuiltArchitectures::parse("86;120a").expect("the release's list parses"),
        }
    }

    /// A Blackwell card (compute capability 12.0) of `class`: the card the
    /// class rule's tests are written for, whose generation every entry fits.
    pub(crate) fn on(class: CardClass) -> CardReading {
        reading(u64::from(class.gb()) << 30, "NVIDIA RTX (Blackwell)", 12, 0)
    }

    /// The cards spec #474 is written for, by total memory and compute
    /// capability: an RTX 3090 (Ampere) and an RTX 4090 (Ada), both 24 GiB.
    pub(crate) fn rtx_3090() -> CardReading {
        reading(24 << 30, "NVIDIA GeForce RTX 3090", 8, 6)
    }

    pub(crate) fn rtx_4090() -> CardReading {
        reading(24 << 30, "NVIDIA GeForce RTX 4090", 8, 9)
    }

    /// An RTX 5090, at the total NVML reports for it.
    pub(crate) fn rtx_5090() -> CardReading {
        reading(RTX_5090_TOTAL_BYTES, "NVIDIA GeForce RTX 5090", 12, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::schema::field;

    #[test]
    fn the_default_profile_carries_the_rtx5090s_measured_numbers() {
        use ignis_core::compute::ModelFamily;

        let profiles = built_in();
        let rtx5090 = &profiles["rtx5090"];
        let headroom = field("vram", "headroom_bytes").unwrap();
        // Flash-Next never traded headroom down: the hardcoded default still
        // holds for it, and as the group's general (unscoped) value.
        assert_eq!((headroom.default)(), serde_json::Value::String("1536M".into()));
        assert_eq!(rtx5090.get(headroom, None).unwrap().raw, "1536M");
        assert!(rtx5090.get(headroom, Some(ModelFamily::FlashNext)).is_none(), "no override; it resolves to the general value");
        // The 27B traded it down to fit vision plus its own context.
        assert_eq!(rtx5090.get(headroom, Some(ModelFamily::Qwen38_27b)).unwrap().raw, "1200M");

        let max_context = field("model", "max_context").unwrap();
        assert_eq!(rtx5090.get(max_context, Some(ModelFamily::Qwen38_27b)).unwrap().raw, "917504");
        assert_eq!(rtx5090.get(max_context, Some(ModelFamily::FlashNext)).unwrap().raw, "589824");

        let rope = field("model", "rope_scaling").unwrap();
        assert!(rtx5090.get(rope, None).is_none(), "unset: `auto` follows the 27B's own context");

        let vision = field("vision", "enabled").unwrap();
        assert_eq!(rtx5090.get(vision, None).unwrap().raw, "true");
        let residency = field("vision", "residency").unwrap();
        assert_eq!(rtx5090.get(residency, None).unwrap().raw, "host", "ADR 0075: the 5090 stages the tower");

        let backend = field("spec", "backend").unwrap();
        assert_eq!(rtx5090.get(backend, Some(ModelFamily::Qwen38_27b)).unwrap().raw, "dflash2");
        assert!(rtx5090.get(backend, Some(ModelFamily::FlashNext)).is_none(), "Flash-Next keeps speculation off here");

        // GitHub #336: both models take the lanes, so the profile scopes
        // Flash-Next's 3 to its own section; the 27B keeps its own 8.
        let lanes = field("spec", "decode_lanes").unwrap();
        assert_eq!(rtx5090.get(lanes, Some(ModelFamily::FlashNext)).unwrap().raw, "3");
        assert!(rtx5090.get(lanes, None).is_none() && rtx5090.get(lanes, Some(ModelFamily::Qwen38_27b)).is_none());

        // Flash-Next pins its calibration-hottest 2 GiB of experts (the
        // owner's call, ADR 0055): the field's own default stays 0, and the
        // field is Flash-Next-only, so the 27B passes it over.
        let pinned = field("vram", "pinned_expert_bytes").unwrap();
        assert_eq!((pinned.default)(), serde_json::Value::from(0));
        assert_eq!(rtx5090.get(pinned, None).unwrap().raw, "2G");
        assert!(profiles["rtxpro6000"].get(pinned, None).is_none(), "a device-only load keeps no expert in RAM");
    }

    /// The 27B started under `rtx5090` takes the profile's Flash-Next-only
    /// pin as a default it passes over, and Flash-Next takes 2 GiB.
    #[test]
    fn under_the_rtx5090_profile_only_flash_next_pins_experts() {
        use super::super::source::{resolve_settings, Fit, Sources};
        use ignis_core::compute::ModelFamily;

        let sources = Sources { profile: built_in()["rtx5090"].clone(), ..Sources::default() };
        let qwen = resolve_settings(&sources, Some(ModelFamily::Qwen38_27b), Fit::Start).unwrap();
        assert_eq!(qwen.settings.vram.pinned_expert_bytes, 0);
        assert!(qwen.dropped.is_empty(), "passed over, not dropped: {:?}", qwen.dropped);
        let flash = resolve_settings(&sources, Some(ModelFamily::FlashNext), Fit::Start).unwrap();
        assert_eq!(flash.settings.vram.pinned_expert_bytes, 2 << 30);
    }

    /// GitHub #336: the decode lanes are both models' field, so a profile
    /// value written for one model is scoped to it -- the 27B started under
    /// either built-in profile keeps its own 8, not Flash-Next's 3.
    #[test]
    fn under_the_built_in_profiles_each_model_keeps_its_own_decode_lanes() {
        use super::super::source::{resolve_settings, Fit, Sources};
        use ignis_core::compute::ModelFamily;

        for (profile, flash_next_lanes) in [("rtx5090", 3), ("rtxpro6000", 8)] {
            let sources = Sources { profile: built_in()[profile].clone(), ..Sources::default() };
            let qwen = resolve_settings(&sources, Some(ModelFamily::Qwen38_27b), Fit::Start).unwrap();
            assert_eq!(qwen.settings.spec.decode_lanes, None, "{profile}: the 27B's default");
            let flash = resolve_settings(&sources, Some(ModelFamily::FlashNext), Fit::Start).unwrap();
            assert_eq!(flash.settings.spec.decode_lanes, Some(flash_next_lanes), "{profile}");
            let general = resolve_settings(&sources, None, Fit::Start).unwrap();
            assert_eq!(general.settings.spec.decode_lanes, None, "{profile}: nothing general");
        }
    }

    /// GitHub #341: the 27B prefills 2,048 tokens a chunk under the built-in
    /// profiles that set one, free there because vision's encoder sizes the
    /// arena; a start without a profile keeps the hardcoded width. GitHub
    /// #489: Flash-Next prefills 8,192 under the two that give it a section,
    /// as `make` always started it, since each of its chunks streams its
    /// experts once.
    #[test]
    fn under_the_built_in_profiles_the_27b_prefills_2048_tokens_a_chunk() {
        use super::super::source::{resolve_settings, Fit, Sources};
        use crate::config::DEFAULT_PREFILL_CHUNK;
        use ignis_core::compute::ModelFamily;

        let sources = Sources { profile: built_in()["rtx5090"].clone(), ..Sources::default() };
        let qwen = resolve_settings(&sources, Some(ModelFamily::Qwen38_27b), Fit::Start).unwrap();
        assert_eq!(qwen.settings.model.prefill_chunk, 2048);
        let flash = resolve_settings(&sources, Some(ModelFamily::FlashNext), Fit::Start).unwrap();
        assert_eq!(flash.settings.model.prefill_chunk, 8192);
        let bare = resolve_settings(&Sources::default(), Some(ModelFamily::Qwen38_27b), Fit::Start).unwrap();
        assert_eq!(bare.settings.model.prefill_chunk, DEFAULT_PREFILL_CHUNK);
        let pro = Sources { profile: built_in()["rtxpro6000"].clone(), ..Sources::default() };
        assert_eq!(resolve_settings(&pro, Some(ModelFamily::Qwen38_27b), Fit::Start).unwrap().settings.model.prefill_chunk, 2048);
        assert_eq!(resolve_settings(&pro, Some(ModelFamily::FlashNext), Fit::Start).unwrap().settings.model.prefill_chunk, 8192);
        let rtx24 = Sources { profile: built_in()["rtx-24gb"].clone(), ..Sources::default() };
        assert_eq!(resolve_settings(&rtx24, Some(ModelFamily::Qwen38_27b), Fit::Start).unwrap().settings.model.prefill_chunk, 2048, "the 5090's");
        let small = Sources { profile: built_in()["rtx-16gb"].clone(), ..Sources::default() };
        // GitHub #489: Flash-Next's chunk on the 24 and 16 GB classes, values
        // no card of theirs has measured yet (GitHub #526).
        for (profile, chunk) in [(&rtx24, 4096), (&small, 2048)] {
            let flash = resolve_settings(profile, Some(ModelFamily::FlashNext), Fit::Start).unwrap();
            assert_eq!(flash.settings.model.prefill_chunk, chunk);
        }
    }

    /// `auto` rope scaling under each profile, as a load of each model
    /// resolves it: each model's factor is its own context over the trained
    /// position envelope (ADR 0057), Flash-Next's too (ADR 0059).
    #[test]
    fn the_profiles_rope_table_follows_each_models_context() {
        use ignis_core::compute::ModelFamily;

        let table = |profile: &str, family| {
            let args = ["--profile".to_owned(), profile.to_owned()];
            let crate::config::ConfigOutcome::Config(config) = crate::config::resolve(&args, |_| None).unwrap() else {
                panic!("a config")
            };
            let fitted = config.for_family(family).unwrap();
            assert!(fitted.settings().model.rope_scaling.is_auto());
            fitted.rope_scaling
        };
        assert_eq!(table("rtx5090", ModelFamily::Qwen38_27b).factor(), 3.5, "917,504 / 262,144");
        assert_eq!(table("rtx5090", ModelFamily::FlashNext).factor(), 2.25, "589,824 / 262,144");
        assert_eq!(table("rtxpro6000", ModelFamily::Qwen38_27b).factor(), 4.0);
        assert_eq!(table("rtxpro6000", ModelFamily::FlashNext).factor(), 4.0);
        assert_eq!(table("rtx-24gb", ModelFamily::Qwen38_27b).factor(), 4.0, "1,048,576 / 262,144");
    }

    /// The RTX PRO 6000 Blackwell series (Workstation, Max-Q, Server Edition:
    /// 96 GB each, sm_120), measured on a Server Edition (2026-10-18).
    #[test]
    fn the_rtxpro6000_profile_gives_both_models_their_whole_context_and_flash_next_every_lane() {
        use ignis_core::compute::ModelFamily;

        let profiles = built_in();
        let pro = &profiles["rtxpro6000"];
        let max_context = field("model", "max_context").unwrap();
        // The 27B's whole hq-e8-2b envelope, YaRN at 4 under `auto`; so is
        // Flash-Next's (ADR 0057, ADR 0059).
        assert_eq!(pro.get(max_context, Some(ModelFamily::Qwen38_27b)).unwrap().raw, "1048576");
        assert_eq!(pro.get(max_context, Some(ModelFamily::FlashNext)).unwrap().raw, "1048576");
        let rope = field("model", "rope_scaling").unwrap();
        assert!(pro.get(rope, None).is_none(), "unset: `auto` makes the 27B's 1,048,576 tokens YaRN at 4");

        let lanes = field("spec", "decode_lanes").unwrap();
        assert_eq!(pro.get(lanes, Some(ModelFamily::FlashNext)).unwrap().raw, "8", "every lane: on 96 GB a lane costs its state");
        assert!(pro.get(lanes, None).is_none(), "Flash-Next's own: the 27B's are its default");

        // The 5090's choices the card's memory does not change.
        let vision = field("vision", "enabled").unwrap();
        assert_eq!(pro.get(vision, None).unwrap().raw, "true");
        let residency = field("vision", "residency").unwrap();
        assert_eq!(pro.get(residency, None).unwrap().raw, "device", "ADR 0075: 96 GB keeps the tower resident");
        let backend = field("spec", "backend").unwrap();
        assert_eq!(pro.get(backend, Some(ModelFamily::Qwen38_27b)).unwrap().raw, "dflash2");
        let draft = field("spec", "draft_tokens").unwrap();
        assert_eq!(pro.get(draft, Some(ModelFamily::Qwen38_27b)).unwrap().raw, "7");

        // Flash-Next drafts with its MTP head, 3 tokens a lane, cut by the
        // default row budget's 8 rows (GitHub #329 item 5: measured on this
        // card, every wider budget loses to 8): on 96 GB no expert crosses
        // PCIe, the cost that made MTP lose on the 5090.
        assert_eq!(pro.get(backend, Some(ModelFamily::FlashNext)).unwrap().raw, "mtp");
        assert_eq!(pro.get(draft, Some(ModelFamily::FlashNext)).unwrap().raw, "3");
        let rows = field("spec", "draft_rows").unwrap();
        assert!(pro.get(rows, None).is_none(), "unset: the default row budget, 8 of 32");

        // No headroom traded down: the hardcoded default, for both families.
        let headroom = field("vram", "headroom_bytes").unwrap();
        assert!(pro.get(headroom, None).is_none());
        assert!(pro.get(headroom, Some(ModelFamily::Qwen38_27b)).is_none());

        // The whole n-gram table when the host plan leaves room for it, so no
        // decode round reads the disk; Flash-Next-only, the 27B passes it over.
        let hot = field("ngram", "hot_bytes").unwrap();
        assert_eq!(pro.get(hot, None).unwrap().raw, "auto");
        assert!(profiles["rtx5090"].get(hot, None).is_none(), "the 5090 keeps the 1 GiB default");
    }

    /// GitHub #480: `rtx-24gb` is the RTX 5090 profile's 27B values at the
    /// context measured for the 24 GB class, and nothing else: no Flash-Next
    /// section, no general value, the lanes and the drafter's head unset.
    #[test]
    fn the_rtx_24gb_profile_takes_the_5090s_27b_values_at_its_measured_context() {
        use ignis_core::compute::ModelFamily::{FlashNext, Qwen38_27b};

        let profiles = built_in();
        let rtx24 = &profiles["rtx-24gb"];
        let value = |group: &str, name: &str, family| rtx24.get(field(group, name).unwrap(), family).map(|candidate| candidate.raw.clone());
        assert_eq!(value("model", "max_context", Some(Qwen38_27b)).as_deref(), Some("1048576"), "measured: the hq-e8-2b envelope");
        assert_eq!(value("vram", "headroom_bytes", Some(Qwen38_27b)).as_deref(), Some("1200M"));
        assert_eq!(value("model", "prefill_chunk", Some(Qwen38_27b)).as_deref(), Some("2048"));
        assert_eq!(value("spec", "backend", Some(Qwen38_27b)).as_deref(), Some("dflash2"));
        assert_eq!(value("spec", "draft_tokens", Some(Qwen38_27b)).as_deref(), Some("7"));
        assert_eq!(value("vision", "enabled", None).as_deref(), Some("true"));
        assert_eq!(value("vision", "residency", None).as_deref(), Some("host"));
        // GitHub #508: one or two device retained slots, the 27B's alone.
        assert!((1..=2).contains(&RTX_24GB_RETAINED_DEVICE_SLOTS), "one or two slots (spec #503)");
        assert_eq!(value("reuse", "retained_device", Some(Qwen38_27b)), Some(RTX_24GB_RETAINED_DEVICE_SLOTS.to_string()));
        assert_eq!(value("reuse", "retained_device", None), None, "no general value");
        for (name, layer) in &profiles {
            if name != "rtx-24gb" {
                assert!(layer.entries().all(|(meta, _, _)| meta.group != "reuse"), "{name}: no reuse value");
            }
        }
        for (group, name) in [("vision", "max_tokens"), ("spec", "decode_lanes"), ("spec", "draft_head"), ("vram", "pinned_expert_bytes"), ("ngram", "hot_bytes"), ("model", "rope_scaling")] {
            assert_eq!(value(group, name, None), None, "{group}.{name}: left to its default");
            assert_eq!(value(group, name, Some(Qwen38_27b)), None, "{group}.{name}: the 27B's default");
        }
        assert_eq!(value("vram", "headroom_bytes", None), None, "the general headroom stays the default");
        // GitHub #489: Flash-Next's section holds its prefill chunk alone.
        let flash: Vec<_> = rtx24.entries().filter(|(_, family, _)| *family == Some(FlashNext)).map(|(meta, _, _)| meta.name).collect();
        assert_eq!(flash, ["prefill_chunk"], "Flash-Next's section");
        // The values the 5090's profile gives the 27B, but its context.
        let rtx5090 = &profiles["rtx5090"];
        for (group, name) in [("vram", "headroom_bytes"), ("model", "prefill_chunk"), ("spec", "backend"), ("spec", "draft_tokens"), ("spec", "decode_lanes"), ("spec", "draft_head")] {
            let meta = field(group, name).unwrap();
            let raw = |layer: &Layer| layer.get(meta, Some(Qwen38_27b)).map(|candidate| candidate.raw.clone());
            assert_eq!(raw(rtx24), raw(rtx5090), "{group}.{name}");
        }
        for (group, name) in [("vision", "enabled"), ("vision", "residency"), ("vision", "max_tokens")] {
            let meta = field(group, name).unwrap();
            let raw = |layer: &Layer| layer.get(meta, None).map(|candidate| candidate.raw.clone());
            assert_eq!(raw(rtx24), raw(rtx5090), "{group}.{name}");
        }
    }

    /// The 27B started under `rtxpro6000` takes the profile's Flash-Next-only
    /// n-gram budget as a default it passes over, not a value to refuse.
    #[test]
    fn the_27b_starts_under_the_rtxpro6000_profile() {
        use super::super::source::{resolve_settings, Fit, Sources};
        use ignis_core::compute::ModelFamily;

        let sources = Sources { profile: built_in()["rtxpro6000"].clone(), ..Sources::default() };
        let qwen = resolve_settings(&sources, Some(ModelFamily::Qwen38_27b), Fit::Start).unwrap();
        assert_eq!(qwen.settings.ngram.hot_bytes, None);
        let flash = resolve_settings(&sources, Some(ModelFamily::FlashNext), Fit::Start).unwrap();
        assert_eq!(flash.settings.ngram.hot_bytes.map(|b| b.to_string()).as_deref(), Some("auto"));
    }

    /// Spec 27b-16gb/04 and spec #474: the class boundaries by NVML's total,
    /// at the totals the cards report and either side of every boundary.
    #[test]
    fn a_cards_total_memory_names_its_class() {
        const GIB: f64 = (1u64 << 30) as f64;
        let class = |gib: f64| CardClass::of_total((gib * GIB) as u64);
        assert_eq!(class(15.9), Some(CardClass::Gb16), "a 16 GB card");
        assert_eq!(class(11.9), Some(CardClass::Gb16), "a 12 GB card is the 16 GB class's");
        assert_eq!(class(31.8), Some(CardClass::Gb32), "the 5090");
        assert_eq!(class(95.5), Some(CardClass::Gb96), "the RTX PRO 6000");
        assert_eq!(CardClass::of_total(CARD_CLASS_16GB_BELOW_BYTES - 1), Some(CardClass::Gb16));
        assert_eq!(CardClass::of_total(CARD_CLASS_16GB_BELOW_BYTES), Some(CardClass::Gb24), "20 GiB: the 24 GB class");
        assert_eq!(class(24.0), Some(CardClass::Gb24), "an RTX 3090 or RTX 4090");
        assert_eq!(CardClass::of_total(*CARD_CLASS_32GB_BYTES.start() - 1), Some(CardClass::Gb24), "just below 28 GiB");
        assert_eq!(CardClass::of_total(*CARD_CLASS_32GB_BYTES.start()), Some(CardClass::Gb32));
        assert_eq!(CardClass::of_total(*CARD_CLASS_32GB_BYTES.end()), Some(CardClass::Gb32));
        assert_eq!(CardClass::of_total(*CARD_CLASS_32GB_BYTES.end() + 1), None);
        assert_eq!(class(48.0), None, "a 48 GB card");
        assert_eq!(CardClass::of_total(*CARD_CLASS_96GB_BYTES.start()), Some(CardClass::Gb96));
        assert_eq!(CardClass::of_total(*CARD_CLASS_96GB_BYTES.end() + 1), None, "past 104 GiB");
        assert_eq!(class(141.0), None, "a 141 GB card");
    }

    #[test]
    fn a_card_class_is_named_by_its_gb_and_orders_by_size() {
        for class in CardClass::ALL {
            assert_eq!(CardClass::from_gb(class.gb()), Some(class));
        }
        assert_eq!(CardClass::from_gb(24), Some(CardClass::Gb24));
        assert_eq!(CardClass::from_gb(48), None);
        assert!(CardClass::Gb16 < CardClass::Gb24 && CardClass::Gb24 < CardClass::Gb32 && CardClass::Gb32 < CardClass::Gb96);
        assert_eq!(CardClass::Gb24.to_string(), "24 GB");
        assert_eq!(CardClass::Gb32.to_string(), "32 GB");
        assert_eq!(CardClass::Gb24.profile(), "rtx-24gb");
        assert_eq!(CardClass::Gb32.profile(), "rtx5090", "the 5090's class keeps its profile");
        assert_eq!(CardClass::Gb96.profile(), "rtxpro6000");
    }

    #[test]
    fn the_old_16gb_name_is_the_same_profile() {
        assert_eq!(canonical("rtx50-16gb"), "rtx-16gb");
        assert_eq!(canonical("rtx5090"), "rtx5090");
        assert_eq!(CardClass::Gb16.profile(), "rtx-16gb");
        assert!(built_in().contains_key("rtx-16gb") && !built_in().contains_key("rtx50-16gb"));
        assert_eq!(lookup("rtx50-16gb", &BTreeMap::new()).unwrap(), built_in()["rtx-16gb"]);
    }

    #[test]
    fn a_file_profile_shadows_a_built_in_one_and_an_unknown_name_lists_the_known() {
        let headroom = field("vram", "headroom_bytes").unwrap();
        let mut mine = Layer::default();
        mine.set(headroom, None, super::super::source::Candidate { raw: "2G".into(), spelling: "x".into() }).unwrap();
        let from_file = BTreeMap::from([("rtx5090".to_owned(), mine.clone()), ("a4000".to_owned(), mine.clone())]);
        assert_eq!(lookup("rtx5090", &from_file).unwrap(), mine);
        assert_eq!(lookup("rtx5090", &BTreeMap::new()).unwrap(), built_in()["rtx5090"]);
        let err = lookup("no-such-card", &from_file).unwrap_err().0;
        assert!(err.contains("`no-such-card`") && err.contains("a4000, rtx-16gb, rtx-24gb, rtx5090"), "{err}");
    }
}
