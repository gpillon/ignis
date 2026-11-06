//! What this build's kernels run on (GitHub #457): a card's compute
//! capability against the CUDA architectures the kernel leaf was compiled for
//! (`kernel/CMakeLists.txt`), decided here and nowhere else.
//!
//! The start reads both without creating a CUDA context (`main`'s probe): the
//! capability from the CUDA runtime, the architectures from the leaf, which
//! the build tells them. Asking the loader instead (`cudaFuncGetAttributes`
//! on a leaf kernel) would create the context, and the VRAM plan reads the
//! free memory before the context exists (GitHub #210).

use std::fmt;

use ignis_core::compute::ModelFamily;
use ignis_core::{ExceptionWeights, PrefillAttention, PrefillLinear};

use super::profile::CardReading;
use super::source::{Origin, Source};

/// A card's compute capability, as the CUDA runtime reports it: `8.6`, `12.0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComputeCapability {
    pub major: u32,
    pub minor: u32,
}

impl ComputeCapability {
    pub const fn new(major: u32, minor: u32) -> Self {
        ComputeCapability { major, minor }
    }
}

impl fmt::Display for ComputeCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// A card's **GPU generation** (`GLOSSARY.md`, spec #474): Ampere, Ada or
/// Blackwell, ordered by age, so `entry > card` reads "the entry needs a
/// newer card than this one". It follows from the compute capability the
/// start reads ([`GpuGeneration::of`]); a catalog entry states the oldest one
/// it loads on (`generation`, ADR 0079).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuGeneration {
    Ampere,
    Ada,
    Blackwell,
}

impl GpuGeneration {
    /// Every generation, oldest first.
    pub const ALL: [GpuGeneration; 3] = [Self::Ampere, Self::Ada, Self::Blackwell];

    /// The generation of a card of `capability`: 8.6 and 8.7 Ampere, 8.9
    /// Ada, every 12.x Blackwell. `None` for any other, which no catalog
    /// entry is refused on by generation.
    pub fn of(capability: ComputeCapability) -> Option<Self> {
        match (capability.major, capability.minor) {
            (8, 6 | 7) => Some(Self::Ampere),
            (8, 9) => Some(Self::Ada),
            (12, _) => Some(Self::Blackwell),
            _ => None,
        }
    }

    /// As a catalog entry's `generation` spells it: `ampere`, `ada`,
    /// `blackwell`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ampere => "ampere",
            Self::Ada => "ada",
            Self::Blackwell => "blackwell",
        }
    }

    /// The generation a catalog entry's `generation` names, matched exactly;
    /// `None` for any other text.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|generation| generation.as_str() == name)
    }

    /// The cards of the generation, as a refusal names them: the one
    /// wording every refusal on a card takes, by generation or by compute
    /// capability ([`BuiltArchitectures::describe`]).
    pub fn cards(self) -> &'static str {
        match self {
            Self::Ampere => "RTX 30",
            Self::Ada => "RTX 40",
            Self::Blackwell => "RTX 50, RTX PRO",
        }
    }

    /// The compute capability of the generation's RTX cards ([`Self::cards`]):
    /// 8.6, 8.9, 12.0.
    pub const fn capability(self) -> ComputeCapability {
        match self {
            Self::Ampere => ComputeCapability::new(8, 6),
            Self::Ada => ComputeCapability::new(8, 9),
            Self::Blackwell => ComputeCapability::new(12, 0),
        }
    }
}

impl fmt::Display for GpuGeneration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Ampere => "Ampere",
            Self::Ada => "Ada",
            Self::Blackwell => "Blackwell",
        })
    }
}

/// One architecture the leaf was compiled for, as CMake names it: `86`,
/// machine code for 8.6, which CUDA runs on every 8.x from 8.6 up; or `120a`,
/// arch-specific code, which runs on 12.0 alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Architecture {
    pub capability: ComputeCapability,
    /// The `a` suffix: the arch-specific features, on that capability only.
    pub specific: bool,
}

impl Architecture {
    /// Whether this architecture's machine code runs on a card of
    /// `capability`: an arch-specific one on its own capability alone, a
    /// plain one on its major from its minor up (CUDA's rule for machine
    /// code). The PTX a plain entry also carries is not counted: a card that
    /// would run the leaf only by compiling it at load is not one the build
    /// was made for.
    pub fn runs_on(self, capability: ComputeCapability) -> bool {
        if self.specific {
            capability == self.capability
        } else {
            capability.major == self.capability.major && capability.minor >= self.capability.minor
        }
    }
}

impl fmt::Display for Architecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}{}", self.capability.major, self.capability.minor, if self.specific { "a" } else { "" })
    }
}

/// The architectures the kernel leaf was compiled for, as the build names
/// them (`120a`, `86;120a`): what `ignis_build_cuda_architectures` returns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuiltArchitectures(Vec<Architecture>);

/// The architectures of the default build (`kernel/CMakeLists.txt`), for the
/// CPU tests and the callers that inject a card rather than read one.
pub const DEFAULT_BUILT_ARCHITECTURES: &str = "120a";

impl BuiltArchitectures {
    /// The build's list, `;`-separated: each entry a major then a one-digit
    /// minor, `a` for an arch-specific one (`86`, `89`, `120a`).
    pub fn parse(list: &str) -> Result<Self, String> {
        let entries = list
            .split(';')
            .map(str::trim)
            .filter(|entry| !entry.is_empty())
            .map(|entry| {
                let (digits, specific) = match entry.strip_suffix('a') {
                    Some(digits) => (digits, true),
                    None => (entry, false),
                };
                let invalid = || format!("`{entry}` in the build's CUDA architectures `{list}` is not an architecture such as 86 or 120a");
                if digits.len() < 2 || !digits.bytes().all(|b| b.is_ascii_digit()) {
                    return Err(invalid());
                }
                let (major, minor) = digits.split_at(digits.len() - 1);
                let capability = ComputeCapability::new(major.parse().map_err(|_| invalid())?, minor.parse().map_err(|_| invalid())?);
                Ok(Architecture { capability, specific })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if entries.is_empty() {
            return Err(format!("the build's CUDA architectures `{list}` name none"));
        }
        Ok(BuiltArchitectures(entries))
    }

    /// The default build's list ([`DEFAULT_BUILT_ARCHITECTURES`]).
    pub fn default_build() -> Self {
        Self::parse(DEFAULT_BUILT_ARCHITECTURES).expect("the default build's list parses")
    }

    /// The entries, in the build's order.
    pub fn entries(&self) -> &[Architecture] {
        &self.0
    }

    /// Whether the leaf's kernels run on a card of `capability`: some entry's
    /// machine code does ([`Architecture::runs_on`]).
    pub fn cover(&self, capability: ComputeCapability) -> bool {
        self.0.iter().any(|arch| arch.runs_on(capability))
    }

    /// The build's arch-specific Blackwell entries (`120a`): the only
    /// architectures the leaf's sm_120a set compiles for (`kernel/CMakeLists.txt`,
    /// GitHub #456) -- the NVFP4 W4A4 prefill GEMMs, the block-scaled FP8 MMA,
    /// Flash-Next's decode clusters and staged kernel.
    fn sm120a(&self) -> impl Iterator<Item = &Architecture> {
        self.0.iter().filter(|arch| arch.specific && arch.capability.major == 12)
    }

    /// Whether the sm_120a set's kernels run on a card of `capability`.
    pub fn sm120a_kernels_run_on(&self, capability: ComputeCapability) -> bool {
        self.sm120a().any(|arch| arch.runs_on(capability))
    }

    /// The capabilities the kernels run on, with the cards that have them
    /// ([`GpuGeneration::cards`]): "12.0 (RTX 50, RTX PRO)", "8.6 and up
    /// within 8.x (RTX 30, RTX 40) and 12.0 (…)".
    pub fn describe(&self) -> String {
        let described: Vec<String> = self
            .0
            .iter()
            .map(|arch| {
                let span = if arch.specific {
                    arch.capability.to_string()
                } else {
                    format!("{} and up within {}.x", arch.capability, arch.capability.major)
                };
                let cards: Vec<&str> =
                    GpuGeneration::ALL.iter().filter(|generation| arch.runs_on(generation.capability())).map(|generation| generation.cards()).collect();
                if cards.is_empty() { span } else { format!("{span} ({})", cards.join(", ")) }
            })
            .collect();
        described.join(" and ")
    }
}

impl fmt::Display for BuiltArchitectures {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<String> = self.0.iter().map(Architecture::to_string).collect();
        f.write_str(&names.join(";"))
    }
}

/// The guide's account of the cards ignis runs on, where a refusal on a card
/// sends the reader.
const WHAT_IT_RUNS_ON: &str = "https://gpillon.github.io/ignis/start/#what-it-runs-on";

/// The refusal of a start on a card of `capability` named `name` that the
/// build's kernels do not run on; `None` when they do. A start on such a card
/// could only fail at its first kernel launch, with CUDA's own "no kernel
/// image is available", so it is refused before anything loads, whatever the
/// profile.
pub fn uncovered(name: &str, capability: ComputeCapability, built_for: &BuiltArchitectures) -> Option<String> {
    (!built_for.cover(capability)).then(|| {
        format!(
            "this build's kernels run on compute capability {}, built for {built_for}; the card is {name} ({capability}), which they were not built for: see {WHAT_IT_RUNS_ON}",
            built_for.describe()
        )
    })
}

// ── the routes a card runs (GitHub #456) ─────────────────────────────────────
//
// A build for an older card than Blackwell (`86;120a`) compiles the routes
// with no counterpart there for its `120a` entry alone: on another card their
// launch fails with CUDA's "no kernel image is available". The host never
// picks one there: a field left to its default takes the other route where
// there is one, and a load that has none is refused before it starts, by
// name.

/// A route whose kernels exist in the build's sm_120a set alone
/// ([`BuiltArchitectures::sm120a_kernels_run_on`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sm120aRoute {
    /// The 27B NVFP4 artifact's prefill: the W4A4 GEMMs on the FP4 tensor
    /// cores, with TMA.
    Nvfp4Prefill,
    /// The image model: its GEMMs and its attention on the block-scaled FP8
    /// tensor cores.
    ImageModel,
    /// Flash-Next's staged expert decode, `IGNIS_FN_MOE_STAGED=1`: its
    /// `mbarrier.try_wait` wait.
    FlashNextStagedDecode,
}

impl Sm120aRoute {
    fn what(self) -> &'static str {
        match self {
            Self::Nvfp4Prefill => "the 27B's NVFP4 artifact, whose prefill runs the W4A4 GEMMs on the FP4 tensor cores,",
            Self::ImageModel => "the image model, whose GEMMs and attention run on the block-scaled FP8 tensor cores,",
            Self::FlashNextStagedDecode => "Flash-Next's staged expert decode (IGNIS_FN_MOE_STAGED=1)",
        }
    }

    fn instead(self) -> &'static str {
        match self {
            Self::Nvfp4Prefill => ": on this card the 27B loads from its trellis artifact (`--model-id qwen3.8-27b-16g`)",
            Self::ImageModel => "",
            Self::FlashNextStagedDecode => ": unset IGNIS_FN_MOE_STAGED, and Flash-Next decodes on its register kernel",
        }
    }
}

/// Whether `card` runs the sm_120a set's kernels: with no card read (the
/// mock build, a caller that injects none) nothing is refused or rerouted.
fn runs_sm120a(card: Option<&CardReading>) -> bool {
    card.is_none_or(|card| card.built_for.sm120a_kernels_run_on(card.capability))
}

/// The refusal of `route` on `card`, naming both; `None` on a card that runs it.
pub fn refuse(card: Option<&CardReading>, route: Sm120aRoute) -> Option<String> {
    let card = card?;
    if card.built_for.sm120a_kernels_run_on(card.capability) {
        return None;
    }
    let set: Vec<String> = card.built_for.sm120a().map(Architecture::to_string).collect();
    let blackwell = GpuGeneration::Blackwell;
    Some(format!(
        "{} needs kernels this build compiles for {} alone, which run on compute capability {} ({}); the card is {} ({}){}: see {WHAT_IT_RUNS_ON}",
        route.what(),
        set.join(";"),
        blackwell.capability(),
        blackwell.cards(),
        card.name,
        card.capability,
        route.instead()
    ))
}

/// The load of `family` on `card` refused when it needs a route the card does
/// not run (GitHub #456): the image model; the 27B when its artifact holds
/// an NVFP4 body (`nvfp4_body`, asked only on such a card); Flash-Next with
/// its staged decode asked for (`staged`). Its decode clusters are no load
/// option: only a test's workspace picks them, and the leaf refuses them on
/// a card that runs no cluster.
pub fn load_refusal(
    card: Option<&CardReading>,
    family: ModelFamily,
    nvfp4_body: impl FnOnce() -> bool,
    staged: bool,
) -> Option<String> {
    if runs_sm120a(card) {
        return None;
    }
    let route = match family {
        ModelFamily::QwenImage21 => Some(Sm120aRoute::ImageModel),
        ModelFamily::Qwen38_27b if nvfp4_body() => Some(Sm120aRoute::Nvfp4Prefill),
        ModelFamily::FlashNext if staged => Some(Sm120aRoute::FlashNextStagedDecode),
        _ => None,
    };
    route.and_then(|route| refuse(card, route))
}

/// A prefill route spec #503 adds, which this build carries or not
/// ([`prefill_route_built`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillRoute {
    /// The dense prefill: each trellis matrix decoded once per chunk into one
    /// INT8 buffer and multiplied on the integer MMA (`--model-prefill-linear
    /// int8`).
    DensePrefill,
    /// The integer band of the prompt attention (`--model-prefill-attention
    /// int8`, the 8-bit band of the 8.x cards).
    IntegerBand,
}

impl PrefillRoute {
    fn what(self) -> &'static str {
        match self {
            Self::DensePrefill => "the dense prefill",
            Self::IntegerBand => "the integer band",
        }
    }
}

/// Whether this build carries `route`: the one place that says so (spec
/// #503). A route no kernel runs yet is refused when the operator names it
/// and falls back when a default or a profile gives it; the ticket that
/// builds a route flips its line here, with its test.
pub const fn prefill_route_built(route: PrefillRoute) -> bool {
    match route {
        // GitHub #509: the leaf's dense prefill (kernel/include/ignis_dense_prefill.h).
        PrefillRoute::DensePrefill => true,
        // GitHub #510: the integer band's kernel (kernel/src/hq_prompt_int8.cu).
        PrefillRoute::IntegerBand => true,
    }
}

/// Who gave a prefill route's value (spec #503's fallback rule, GitHub
/// #456): what decides whether the card replaces it, keeps it or refuses it.
/// Only a field left to its hardcoded default takes the card's own route; a
/// profile's value is kept wherever the card and the build run it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GivenBy {
    /// No one: the field's hardcoded default, which an 8.x card replaces with
    /// its own route (GitHub #515).
    Default,
    /// A hardware profile, or the card's own decision already made
    /// ([`Source::Card`]): kept where the card and the build run it, the
    /// fallback where they do not.
    Profile,
    /// The operator, from a flag, the environment, a file or a live patch:
    /// kept, or refused by its spelling where the card or the build cannot
    /// run it.
    Operator,
}

impl GivenBy {
    /// The giver of a value whose origin is `origin` (`None`: the hardcoded
    /// default, as [`super::source::Resolution::origin`] reports it).
    pub fn of(origin: Option<&Origin>) -> Self {
        match origin.map(|origin| origin.source) {
            None | Some(Source::Default) => Self::Default,
            Some(Source::Profile | Source::Card) => Self::Profile,
            Some(Source::Patch | Source::Flag | Source::Env | Source::File) => Self::Operator,
        }
    }
}

/// Whether `card` is read and runs no sm_120a kernel: an 8.x card, whose
/// 8-bit prompt attention band is the integer one.
fn is_8x(card: Option<&CardReading>) -> bool {
    card.is_some_and(|card| !runs_sm120a(Some(card)))
}

/// The 27B's prompt attention band on `card` (`--model-prefill-attention`,
/// GitHub #456, #505, #510, #515): `resolved` where the card and the build
/// run it. The 8-bit band, spelled `8bit` or `fp8`, is the integer band on an
/// 8.x card, so the default there is the integer band, and today's
/// block-scaled FP8 band elsewhere; `int8` names the integer band on every
/// card, and a profile gives it on an 8.x card alone. `fp16` is kept from
/// every source. A band the build cannot run is the FP16 band on an 8.x card
/// (the FP8 band elsewhere) when a default or a profile `given` it, and a
/// refusal naming `spelled` when the operator named it.
pub fn prefill_attention(
    resolved: PrefillAttention,
    given: GivenBy,
    card: Option<&CardReading>,
    spelled: &str,
) -> Result<PrefillAttention, String> {
    prefill_attention_with(resolved, given, card, spelled, prefill_route_built)
}

/// [`prefill_attention`] with the build's routes given, for the tests that
/// flip one.
pub fn prefill_attention_with(
    resolved: PrefillAttention,
    given: GivenBy,
    card: Option<&CardReading>,
    spelled: &str,
    built: impl Fn(PrefillRoute) -> bool,
) -> Result<PrefillAttention, String> {
    let on_8x = is_8x(card);
    let band = if on_8x && resolved == PrefillAttention::Fp8 { PrefillAttention::Int8 } else { resolved };
    if band != PrefillAttention::Int8 {
        return Ok(band);
    }
    let named = given == GivenBy::Operator;
    if !on_8x && !named {
        // No default or profile on another card selects the integer band.
        return Ok(PrefillAttention::Fp8);
    }
    if built(PrefillRoute::IntegerBand) {
        return Ok(PrefillAttention::Int8);
    }
    if !named {
        return Ok(PrefillAttention::Fp16);
    }
    let on = card.map(|card| format!("; the card is {} ({})", card.name, card.capability)).unwrap_or_default();
    let value = if resolved == PrefillAttention::Int8 { "int8" } else { "fp8" };
    let meaning = if resolved == PrefillAttention::Int8 {
        "names the integer band".to_owned()
    } else {
        "names the 8-bit band, which on compute capability 8.x is the integer band".to_owned()
    };
    Err(format!(
        "`{spelled} {value}` {meaning}, and this build does not carry {} yet{on}: name `fp16`, or leave it unset and it is `fp16` on this card",
        PrefillRoute::IntegerBand.what()
    ))
}

/// The 27B's trellis prefill route on `card` (`--model-prefill-linear`,
/// spec #503, GitHub #505, #515): `resolved` where the build runs it. Left to
/// its default, the field is the dense prefill (`int8`) on an 8.x card and
/// the fused prefill (`fp16`) elsewhere; a profile's or a named `fp16` is
/// kept on every card; a profile's `int8` runs on an 8.x card alone, a named
/// one on every card. One the build does not carry is the fused prefill when
/// a default or a profile `given` it, and a refusal naming `spelled` when the
/// operator named it.
pub fn prefill_linear(
    resolved: PrefillLinear,
    given: GivenBy,
    card: Option<&CardReading>,
    spelled: &str,
) -> Result<PrefillLinear, String> {
    prefill_linear_with(resolved, given, card, spelled, prefill_route_built)
}

/// [`prefill_linear`] with the build's routes given, for the tests that flip
/// one.
pub fn prefill_linear_with(
    resolved: PrefillLinear,
    given: GivenBy,
    card: Option<&CardReading>,
    spelled: &str,
    built: impl Fn(PrefillRoute) -> bool,
) -> Result<PrefillLinear, String> {
    // Left to its default on an 8.x card: the dense prefill (GitHub #515).
    let wanted = if given == GivenBy::Default && is_8x(card) { PrefillLinear::Int8 } else { resolved };
    if wanted != PrefillLinear::Int8 {
        return Ok(wanted);
    }
    let named = given == GivenBy::Operator;
    if !built(PrefillRoute::DensePrefill) {
        return if named {
            Err(format!(
                "`{spelled} int8` names {}, which this build does not carry yet: name `fp16`, or leave it unset and it is `fp16`",
                PrefillRoute::DensePrefill.what()
            ))
        } else {
            Ok(PrefillLinear::Fp16)
        };
    }
    // Built: no default or profile on a Blackwell card selects it.
    Ok(if named || is_8x(card) { PrefillLinear::Int8 } else { PrefillLinear::Fp16 })
}

/// The refusal of the 27B's dense prefill on an artifact whose body is
/// NVFP4 (`nvfp4_body`, asked only for `int8`), which has no trellis matrix
/// to decode (GitHub #509; the review of spec #503), naming the option as
/// `spelled` gave it. Only a named `int8` reaches it: no default or profile
/// selects the dense prefill on an RTX 50 or RTX PRO card, and on an 8.x
/// card [`load_refusal`] refuses the NVFP4 artifact first. The leaf refuses
/// the same load too, by its option's number.
pub fn prefill_linear_refusal(
    prefill_linear: PrefillLinear,
    nvfp4_body: impl FnOnce() -> bool,
    spelled: &str,
) -> Option<String> {
    (prefill_linear == PrefillLinear::Int8 && nvfp4_body()).then(|| {
        format!(
            "`{spelled} int8` names the dense prefill, which decodes the trellis body's matrices, and this artifact's \
             body is NVFP4: name `fp16` or leave it unset, or load the 27B's trellis artifact (`--model-id qwen3.8-27b-16g`)"
        )
    })
}

/// Whether a load of `family` on a card of `generation` publishes its
/// generation opener's page by the prompt checkpoint's capture, as a
/// pages-only link (ADR 0029 as amended 2026-10-07 and 2026-11-05, GitHub
/// #306, #508): the one decision, which [`crate::runtime::EngineShape`] asks
/// for the load's family and the card the start read.
///
/// On, a reusable prompt is cut once, at the opener; off, twice, at the
/// opener's page floor too, where its imaged prefix is published. Flash-Next
/// rides on every card: its cut is a traversal that re-streams its experts.
/// The 27B rides on Ampere and Ada, where a traversal of its trellis body
/// decodes every matrix again and the cut costs clearly more than the ~19 ms
/// it costs on the RTX 5090. On Blackwell, and with no card read, it keeps
/// both cuts and the imaged prefix at the floor, its splits unchanged since
/// #306. The image model and the diarizer capture nothing. A request whose
/// image spans its opener's page floor lends the pages at its capture on any
/// load (GitHub #249, `CaptureLink::ImageSpansFloor`).
pub fn opener_page_rides_capture(family: ModelFamily, generation: Option<GpuGeneration>) -> bool {
    match family {
        ModelFamily::FlashNext => true,
        ModelFamily::Qwen38_27b => matches!(generation, Some(GpuGeneration::Ampere | GpuGeneration::Ada)),
        ModelFamily::QwenImage21 | ModelFamily::Nemotron3Diarization => false,
    }
}

/// The solo widths an `auto` load of `family` on a card of `generation`
/// tries, widest first (GitHub #489, ADR 0018 and ADR 0063 §3 as amended):
/// the one decision, which [`crate::runtime::EngineShape::solo_widths`] asks.
/// The load keeps the first its VRAM plan holds with nothing else adapted,
/// otherwise the prefill chunk.
///
/// 8,192 then 4,096 for the 27B on Ampere and Ada, where the dense prefill
/// decodes each trellis matrix once per chunk, so a wider chunk spreads that
/// decode over more tokens. None on Blackwell: a wider chunk buys the 27B at
/// most 4% of cold TTFT there (finding 2026-10-07-27b-prefill-chunk-width),
/// and the `rtx5090` profile's margin does not hold 4,096 (finding
/// 2026-10-21-the-27b-prefill-chunk-at-2048). None for Flash-Next on any
/// card: every chunk pays a near-constant expert stream, so it wants its
/// widest chunk with lanes decoding as well (finding
/// 2026-10-07-flash-next-prefill-chunk-and-decode-share). None with no card
/// read.
pub fn solo_ladder(family: ModelFamily, generation: Option<GpuGeneration>) -> &'static [u32] {
    match (family, generation) {
        (ModelFamily::Qwen38_27b, Some(GpuGeneration::Ampere | GpuGeneration::Ada)) => &[8192, 4096],
        _ => &[],
    }
}

/// The 27B's BF16 exception tensors on `card` (`--model-exception-weights`,
/// GitHub #456): held in FP8, they prefill on the W8A8 GEMM, the sm_120a
/// set's block-scaled FP8 MMA, so on a card that does not run it `fp8` is
/// refused, naming `spelled`; `bf16`, the default, runs everywhere.
pub fn exception_weights(
    resolved: ExceptionWeights,
    card: Option<&CardReading>,
    spelled: &str,
) -> Result<ExceptionWeights, String> {
    match card {
        Some(card) if resolved.is_fp8() && !runs_sm120a(Some(card)) => Err(format!(
            "`{spelled} fp8` prefills the exception tensors on the block-scaled FP8 tensor cores, which compute capability 12.0 has and the card, {} ({}), does not: leave it `bf16`",
            card.name, card.capability
        )),
        _ => Ok(resolved),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cc(major: u32, minor: u32) -> ComputeCapability {
        ComputeCapability::new(major, minor)
    }

    /// Spec #474: the GPU generation follows from the compute capability the
    /// start reads — 8.6 and 8.7 Ampere, 8.9 Ada, every 12.x Blackwell — and
    /// a capability of none of them has none, which refuses nothing. The
    /// three are ordered by age, as a catalog entry's `generation` reads them.
    #[test]
    fn the_gpu_generation_follows_from_the_compute_capability() {
        use GpuGeneration::*;
        for (capability, generation) in [(cc(8, 6), Ampere), (cc(8, 7), Ampere), (cc(8, 9), Ada), (cc(12, 0), Blackwell), (cc(12, 1), Blackwell)] {
            assert_eq!(GpuGeneration::of(capability), Some(generation), "{capability}");
        }
        for capability in [cc(7, 5), cc(8, 0), cc(8, 8), cc(9, 0), cc(10, 0), cc(11, 0)] {
            assert_eq!(GpuGeneration::of(capability), None, "{capability}");
        }
        assert!(Ampere < Ada && Ada < Blackwell);
        assert_eq!(GpuGeneration::ALL, [Ampere, Ada, Blackwell]);
        for (generation, name, said, cards, capability) in
            [(Ampere, "ampere", "Ampere", "RTX 30", cc(8, 6)), (Ada, "ada", "Ada", "RTX 40", cc(8, 9)), (Blackwell, "blackwell", "Blackwell", "RTX 50, RTX PRO", cc(12, 0))]
        {
            assert_eq!((generation.as_str(), generation.to_string(), generation.cards(), generation.capability()), (name, said.to_owned(), cards, capability));
            assert_eq!(GpuGeneration::of(capability), Some(generation));
            assert_eq!(GpuGeneration::from_name(name), Some(generation));
        }
        for other in ["Blackwell", "turing", "", " ada", "12"] {
            assert_eq!(GpuGeneration::from_name(other), None, "{other:?}");
        }
    }

    #[test]
    fn the_list_parses_as_cmake_names_it_and_refuses_anything_else() {
        let list = BuiltArchitectures::parse("86;120a").unwrap();
        assert_eq!(list.entries(), [Architecture { capability: cc(8, 6), specific: false }, Architecture { capability: cc(12, 0), specific: true }]);
        assert_eq!(list.to_string(), "86;120a");
        assert_eq!(BuiltArchitectures::default_build().to_string(), "120a");
        for bad in ["", ";", "native", "all-major", "8", "sm_86", "120f", "86-real"] {
            assert!(BuiltArchitectures::parse(bad).is_err(), "{bad}");
        }
    }

    /// The default build (`120a`) runs on 12.0 alone: not on the RTX 40 and
    /// 30, not on the DGX Spark's 12.1, not on any other capability.
    #[test]
    fn the_default_build_covers_12_0_alone() {
        let built = BuiltArchitectures::default_build();
        assert!(built.cover(cc(12, 0)));
        for other in [cc(8, 6), cc(8, 9), cc(9, 0), cc(10, 0), cc(12, 1), cc(7, 5)] {
            assert!(!built.cover(other), "{other}");
        }
    }

    /// A plain entry's machine code runs on its major from its minor up: an
    /// `86;120a` build covers 8.6 and 8.9, an `89;120a` one not 8.6.
    #[test]
    fn a_plain_entry_covers_its_major_from_its_minor_up() {
        let ampere = BuiltArchitectures::parse("86;120a").unwrap();
        for covered in [cc(8, 6), cc(8, 7), cc(8, 9), cc(12, 0)] {
            assert!(ampere.cover(covered), "{covered}");
        }
        for other in [cc(8, 0), cc(9, 0), cc(12, 1)] {
            assert!(!ampere.cover(other), "{other}");
        }
        let ada = BuiltArchitectures::parse("89;120a").unwrap();
        assert!(ada.cover(cc(8, 9)) && !ada.cover(cc(8, 6)));
    }

    #[test]
    fn the_refusal_names_the_card_its_capability_and_the_supported_set() {
        let built = BuiltArchitectures::default_build();
        let refusal = uncovered("NVIDIA GeForce RTX 4080", cc(8, 9), &built).unwrap();
        assert!(refusal.contains("NVIDIA GeForce RTX 4080 (8.9)"), "{refusal}");
        assert!(refusal.contains("compute capability 12.0 (RTX 50, RTX PRO), built for 120a;"), "{refusal}");
        assert!(refusal.contains("built for 120a;"), "{refusal}");
        assert!(refusal.ends_with("see https://gpillon.github.io/ignis/start/#what-it-runs-on"), "{refusal}");
        assert!(!refusal.contains("#456"), "{refusal}");
        assert_eq!(uncovered("NVIDIA GeForce RTX 5090", cc(12, 0), &built), None);
        let ampere = BuiltArchitectures::parse("86;120a").unwrap();
        assert_eq!(uncovered("NVIDIA GeForce RTX 3090", cc(8, 6), &ampere), None);
        let turing = uncovered("NVIDIA GeForce RTX 2080 Ti", cc(7, 5), &ampere).unwrap();
        assert!(turing.contains("8.6 and up within 8.x (RTX 30, RTX 40) and 12.0 (RTX 50, RTX PRO), built for 86;120a;"), "{turing}");
    }

    fn card(name: &str, capability: ComputeCapability, built_for: &str) -> CardReading {
        CardReading { total: 24 << 30, name: name.to_owned(), capability, built_for: BuiltArchitectures::parse(built_for).unwrap() }
    }

    fn ampere() -> CardReading {
        card("NVIDIA GeForce RTX 3090", cc(8, 6), "86;120a")
    }

    fn blackwell() -> CardReading {
        card("NVIDIA GeForce RTX 5090", cc(12, 0), "86;120a")
    }

    /// The sm_120a set is the build's arch-specific Blackwell entries: an
    /// `86;120a` build runs it on 12.0 alone, never on 8.x.
    #[test]
    fn the_sm120a_set_runs_on_12_0_alone() {
        let built = BuiltArchitectures::parse("86;120a").unwrap();
        assert!(built.sm120a_kernels_run_on(cc(12, 0)));
        for other in [cc(8, 6), cc(8, 9), cc(9, 0), cc(12, 1)] {
            assert!(!built.sm120a_kernels_run_on(other), "{other}");
        }
    }

    /// On an Ampere card the prompt attention's band left to its default is
    /// the integer one (GitHub #515; it was the FP16 band until then), a
    /// profile's or a named `fp16` stays, and a named `fp8` runs the integer
    /// band (a build without it refuses it by its spelling, which
    /// `the_prefill_attention_without_the_integer_band_built` holds); a
    /// Blackwell card and no card at all keep what resolved.
    #[test]
    fn the_prefill_attention_follows_the_card() {
        use GivenBy::*;
        let (fp8, fp16) = (PrefillAttention::Fp8, PrefillAttention::Fp16);
        let flag = "--model-prefill-attention";
        assert_eq!(prefill_attention(fp8, Default, Some(&ampere()), flag), Ok(PrefillAttention::Int8));
        for given in [Profile, Operator] {
            assert_eq!(prefill_attention(fp16, given, Some(&ampere()), flag), Ok(fp16), "{given:?}");
            assert_eq!(prefill_attention(fp8, given, Some(&ampere()), flag), Ok(PrefillAttention::Int8), "{given:?}");
        }
        for given in [Default, Profile, Operator] {
            assert_eq!(prefill_attention(fp8, given, Some(&blackwell()), flag), Ok(fp8));
            assert_eq!(prefill_attention(fp8, given, None, flag), Ok(fp8));
        }
    }

    /// Who gave a value, by its origin: the hardcoded default (no origin),
    /// the profile or the card's decision, the operator from any source.
    #[test]
    fn the_giver_of_a_value_is_its_origins_source() {
        let origin = |source| Origin { source, spelling: String::new(), at_default: false, scoped: false };
        assert_eq!(GivenBy::of(None), GivenBy::Default);
        assert_eq!(GivenBy::of(Some(&origin(Source::Default))), GivenBy::Default);
        for source in [Source::Profile, Source::Card] {
            assert_eq!(GivenBy::of(Some(&origin(source))), GivenBy::Profile, "{source:?}");
        }
        for source in [Source::Patch, Source::Flag, Source::Env, Source::File] {
            assert_eq!(GivenBy::of(Some(&origin(source))), GivenBy::Operator, "{source:?}");
        }
    }

    /// FP8 exception tensors need the W8A8 GEMM: refused on an Ampere card,
    /// by the spelling that named them; BF16 runs everywhere.
    #[test]
    fn fp8_exception_weights_are_refused_on_a_card_without_the_w8a8_gemm() {
        let (fp8, bf16) = (ExceptionWeights::Fp8, ExceptionWeights::Bf16);
        let flag = "--model-exception-weights";
        let refused = exception_weights(fp8, Some(&ampere()), flag).unwrap_err();
        assert!(refused.contains("`--model-exception-weights fp8`") && refused.contains("(8.6)"), "{refused}");
        assert_eq!(exception_weights(bf16, Some(&ampere()), flag), Ok(bf16));
        assert_eq!(exception_weights(fp8, Some(&blackwell()), flag), Ok(fp8));
        assert_eq!(exception_weights(fp8, None, flag), Ok(fp8));
    }

    /// The loads with no route on an Ampere card are refused before they
    /// start, naming what needs the sm_120a set and the card; the 27B's
    /// trellis artifact, Flash-Next on its register kernel and everything on
    /// a Blackwell card or with no card read go ahead, and the artifact is not
    /// asked about its body there.
    #[test]
    fn a_load_with_no_route_on_the_card_is_refused_by_name() {
        let never = || -> bool { panic!("the body is asked only on a card that runs no sm_120a kernel") };
        let ampere = ampere();
        let image = load_refusal(Some(&ampere), ModelFamily::QwenImage21, never, false).unwrap();
        assert!(image.starts_with("the image model") && image.contains("NVIDIA GeForce RTX 3090 (8.6)") && image.contains("120a alone"), "{image}");
        assert!(image.contains("which run on compute capability 12.0 (RTX 50, RTX PRO); the card is"), "{image}");
        let nvfp4 = load_refusal(Some(&ampere), ModelFamily::Qwen38_27b, || true, false).unwrap();
        assert!(nvfp4.contains("W4A4") && nvfp4.contains("--model-id qwen3.8-27b-16g"), "{nvfp4}");
        assert_eq!(load_refusal(Some(&ampere), ModelFamily::Qwen38_27b, || false, false), None, "the trellis artifact loads");
        let staged = load_refusal(Some(&ampere), ModelFamily::FlashNext, || false, true).unwrap();
        assert!(staged.contains("IGNIS_FN_MOE_STAGED"), "{staged}");
        assert_eq!(load_refusal(Some(&ampere), ModelFamily::FlashNext, || false, false), None, "its register kernel is portable");
        for family in [ModelFamily::QwenImage21, ModelFamily::Qwen38_27b, ModelFamily::FlashNext] {
            assert_eq!(load_refusal(Some(&blackwell()), family, never, true), None, "{family:?}");
            assert_eq!(load_refusal(None, family, never, true), None, "{family:?}");
        }
    }

    /// Spec #503 (GitHub #505, #509, #510, #515): the build carries both
    /// routes ([`prefill_route_built`]), the dense prefill and the integer
    /// band, and an 8.x card runs the dense prefill when the field is left to
    /// its default; a profile's or a named `fp16` keeps the fused prefill
    /// there, and Blackwell runs the dense prefill only when named.
    #[test]
    fn this_build_carries_both_routes_and_an_8x_card_runs_the_dense_prefill_by_default() {
        use GivenBy::*;
        assert!(prefill_route_built(PrefillRoute::DensePrefill));
        assert!(prefill_route_built(PrefillRoute::IntegerBand));
        let flag = "--model-prefill-linear";
        let (a, b) = (ampere(), blackwell());
        assert_eq!(prefill_linear(PrefillLinear::Fp16, Default, Some(&a), flag), Ok(PrefillLinear::Int8), "8.x: unset");
        assert_eq!(prefill_linear(PrefillLinear::Fp16, Profile, Some(&a), flag), Ok(PrefillLinear::Fp16), "8.x: a profile's fp16");
        assert_eq!(prefill_linear(PrefillLinear::Fp16, Operator, Some(&a), flag), Ok(PrefillLinear::Fp16), "8.x: fp16 named");
        assert_eq!(prefill_linear(PrefillLinear::Fp16, Default, Some(&b), flag), Ok(PrefillLinear::Fp16), "Blackwell: unset");
        assert_eq!(prefill_linear(PrefillLinear::Int8, Profile, Some(&a), flag), Ok(PrefillLinear::Int8), "8.x: a profile's");
        assert_eq!(prefill_linear(PrefillLinear::Int8, Operator, Some(&a), flag), Ok(PrefillLinear::Int8), "8.x: named");
        assert_eq!(prefill_linear(PrefillLinear::Int8, Operator, Some(&b), flag), Ok(PrefillLinear::Int8), "Blackwell: named");
        assert_eq!(prefill_linear(PrefillLinear::Int8, Profile, Some(&b), flag), Ok(PrefillLinear::Fp16), "Blackwell: a profile's falls back");
        assert_eq!(prefill_linear(PrefillLinear::Fp16, Default, None, flag), Ok(PrefillLinear::Fp16), "no card read: unset");
    }

    /// The prompt attention band by generation, with the integer band not
    /// built: the 8-bit band (`fp8`, `8bit`) is the integer one on 8.x and
    /// falls back to `fp16` from a default or a profile, is refused when
    /// named; `int8` named is refused on every card, from a profile it falls
    /// back; `fp16` runs everywhere; Blackwell keeps its FP8 band.
    #[test]
    fn the_prefill_attention_without_the_integer_band_built() {
        use GivenBy::*;
        let none = |_: PrefillRoute| false;
        let flag = "--model-prefill-attention";
        for card in [ampere(), card("NVIDIA GeForce RTX 4090", cc(8, 9), "86;120a")] {
            let c = Some(&card);
            for given in [Default, Profile] {
                assert_eq!(prefill_attention_with(PrefillAttention::Fp8, given, c, flag, none), Ok(PrefillAttention::Fp16));
                assert_eq!(prefill_attention_with(PrefillAttention::Int8, given, c, flag, none), Ok(PrefillAttention::Fp16));
            }
            assert_eq!(prefill_attention_with(PrefillAttention::Fp16, Operator, c, flag, none), Ok(PrefillAttention::Fp16));
            let fp8 = prefill_attention_with(PrefillAttention::Fp8, Operator, c, flag, none).unwrap_err();
            assert!(fp8.contains("`--model-prefill-attention fp8`") && fp8.contains("integer band") && fp8.contains("not carry"), "{fp8}");
            let int8 = prefill_attention_with(PrefillAttention::Int8, Operator, c, flag, none).unwrap_err();
            assert!(int8.contains("`--model-prefill-attention int8`") && int8.contains(&card.name), "{int8}");
        }
        let b = blackwell();
        let c = Some(&b);
        assert_eq!(prefill_attention_with(PrefillAttention::Fp8, Operator, c, flag, none), Ok(PrefillAttention::Fp8));
        assert_eq!(prefill_attention_with(PrefillAttention::Fp8, Default, c, flag, none), Ok(PrefillAttention::Fp8));
        assert_eq!(prefill_attention_with(PrefillAttention::Int8, Profile, c, flag, none), Ok(PrefillAttention::Fp8), "no profile selects it there");
        assert!(prefill_attention_with(PrefillAttention::Int8, Operator, c, flag, none).unwrap_err().contains("int8"));
        assert_eq!(prefill_attention_with(PrefillAttention::Fp8, Operator, None, flag, none), Ok(PrefillAttention::Fp8));
    }

    /// The same decision with the integer band built, the line #510 flips: on
    /// 8.x it runs for every spelling of the 8-bit band named, and for a
    /// profile's `int8`, and for a default's or a profile's `fp8` (the
    /// default since #515); on Blackwell it runs only when named.
    #[test]
    fn the_prefill_attention_with_the_integer_band_built() {
        use GivenBy::*;
        let built = |route: PrefillRoute| route == PrefillRoute::IntegerBand;
        let (a, b) = (ampere(), blackwell());
        for given in [Default, Profile, Operator] {
            assert_eq!(prefill_attention_with(PrefillAttention::Fp8, given, Some(&a), "", built), Ok(PrefillAttention::Int8));
            assert_eq!(prefill_attention(PrefillAttention::Fp8, given, Some(&a), ""), Ok(PrefillAttention::Int8), "the build's own");
            assert_eq!(prefill_attention_with(PrefillAttention::Fp8, given, Some(&b), "", built), Ok(PrefillAttention::Fp8));
        }
        for given in [Profile, Operator] {
            assert_eq!(prefill_attention_with(PrefillAttention::Int8, given, Some(&a), "", built), Ok(PrefillAttention::Int8));
            assert_eq!(prefill_attention_with(PrefillAttention::Fp16, given, Some(&a), "", built), Ok(PrefillAttention::Fp16));
        }
        assert_eq!(prefill_attention_with(PrefillAttention::Int8, Operator, Some(&b), "", built), Ok(PrefillAttention::Int8));
        assert_eq!(prefill_attention_with(PrefillAttention::Int8, Profile, Some(&b), "", built), Ok(PrefillAttention::Fp8));
    }

    /// The dense prefill on the NVFP4 artifact is refused naming the option
    /// as it was spelled; on the trellis artifact, and for `fp16`, nothing is
    /// refused, and the artifact is asked about its body for `int8` alone.
    #[test]
    fn the_dense_prefill_on_an_nvfp4_body_is_refused_naming_the_option() {
        let never = || -> bool { panic!("the body is asked only for int8") };
        for spelled in ["--model-prefill-linear", "IGNIS_MODEL_PREFILL_LINEAR"] {
            let refusal = prefill_linear_refusal(PrefillLinear::Int8, || true, spelled).unwrap();
            assert!(refusal.starts_with(&format!("`{spelled} int8` names the dense prefill")), "{refusal}");
            assert!(refusal.contains("NVFP4") && refusal.contains("`fp16`") && refusal.contains("qwen3.8-27b-16g"), "{refusal}");
        }
        assert_eq!(prefill_linear_refusal(PrefillLinear::Int8, || false, "--model-prefill-linear"), None, "the trellis artifact");
        assert_eq!(prefill_linear_refusal(PrefillLinear::Fp16, never, "--model-prefill-linear"), None);
    }

    /// GitHub #508 (spec #503): the 27B's generation-opener page rides the
    /// prompt checkpoint's capture on Ampere and Ada, and not on Blackwell nor
    /// with no card read; Flash-Next's rides it on every card, as since #306;
    /// the image model and the diarizer capture nothing.
    #[test]
    fn the_opener_page_rides_the_capture_by_family_and_generation() {
        use GpuGeneration::*;
        let qwen = |generation| opener_page_rides_capture(ModelFamily::Qwen38_27b, generation);
        assert!(qwen(Some(Ampere)), "the 27B on Ampere: one cut, at the opener");
        assert!(qwen(Some(Ada)), "the 27B on Ada: one cut, at the opener");
        assert!(!qwen(Some(Blackwell)), "the 27B on Blackwell keeps its two cuts");
        assert!(!qwen(None), "no card read: the 27B keeps its two cuts");
        for generation in [Some(Ampere), Some(Ada), Some(Blackwell), None] {
            assert!(opener_page_rides_capture(ModelFamily::FlashNext, generation), "Flash-Next on {generation:?}");
            for family in [ModelFamily::QwenImage21, ModelFamily::Nemotron3Diarization] {
                assert!(!opener_page_rides_capture(family, generation), "{family:?} on {generation:?}");
            }
        }
    }

    /// GitHub #489: `auto`'s solo ladder is 8,192 then 4,096 for the 27B on
    /// Ampere and Ada, and empty everywhere else: Blackwell, no card read,
    /// Flash-Next on any card, the models with no prefill chunk.
    #[test]
    fn the_solo_ladder_is_the_27bs_on_ampere_and_ada_only() {
        use GpuGeneration::*;
        assert_eq!(solo_ladder(ModelFamily::Qwen38_27b, Some(Ampere)), [8192, 4096]);
        assert_eq!(solo_ladder(ModelFamily::Qwen38_27b, Some(Ada)), [8192, 4096]);
        assert!(solo_ladder(ModelFamily::Qwen38_27b, Some(Blackwell)).is_empty());
        assert!(solo_ladder(ModelFamily::Qwen38_27b, None).is_empty());
        for generation in [Some(Ampere), Some(Ada), Some(Blackwell), None] {
            for family in [ModelFamily::FlashNext, ModelFamily::QwenImage21, ModelFamily::Nemotron3Diarization] {
                assert!(solo_ladder(family, generation).is_empty(), "{family:?} on {generation:?}");
            }
        }
    }

    /// The trellis prefill route: `fp16` given by a profile or named runs
    /// everywhere; `int8` named is refused while the dense prefill is not
    /// built and a default or profile one falls back; built, it runs on 8.x
    /// from the default, a profile or the operator, and on Blackwell only
    /// when named.
    #[test]
    fn the_prefill_linear_follows_the_build_and_the_generation() {
        use GivenBy::*;
        let flag = "--model-prefill-linear";
        let none = |_: PrefillRoute| false;
        let built = |route: PrefillRoute| route == PrefillRoute::DensePrefill;
        let (a, b) = (ampere(), blackwell());
        for card in [Some(&a), Some(&b), None] {
            for given in [Profile, Operator] {
                assert_eq!(prefill_linear_with(PrefillLinear::Fp16, given, card, flag, none), Ok(PrefillLinear::Fp16));
                assert_eq!(prefill_linear_with(PrefillLinear::Fp16, given, card, flag, built), Ok(PrefillLinear::Fp16));
            }
            assert_eq!(prefill_linear_with(PrefillLinear::Int8, Profile, card, flag, none), Ok(PrefillLinear::Fp16));
            let refused = prefill_linear_with(PrefillLinear::Int8, Operator, card, flag, none).unwrap_err();
            assert!(refused.contains("`--model-prefill-linear int8`") && refused.contains("dense prefill") && refused.contains("not carry"), "{refused}");
        }
        assert_eq!(prefill_linear_with(PrefillLinear::Int8, Profile, Some(&a), flag, built), Ok(PrefillLinear::Int8));
        assert_eq!(prefill_linear_with(PrefillLinear::Fp16, Default, Some(&a), flag, built), Ok(PrefillLinear::Int8), "unset on 8.x");
        assert_eq!(prefill_linear_with(PrefillLinear::Fp16, Default, Some(&a), flag, none), Ok(PrefillLinear::Fp16), "not built: fused");
        assert_eq!(prefill_linear_with(PrefillLinear::Fp16, Default, Some(&b), flag, built), Ok(PrefillLinear::Fp16), "unset on Blackwell");
        assert_eq!(prefill_linear_with(PrefillLinear::Int8, Operator, Some(&a), flag, built), Ok(PrefillLinear::Int8));
        assert_eq!(prefill_linear_with(PrefillLinear::Int8, Operator, Some(&b), flag, built), Ok(PrefillLinear::Int8));
        assert_eq!(prefill_linear_with(PrefillLinear::Int8, Profile, Some(&b), flag, built), Ok(PrefillLinear::Fp16), "no profile selects it there");
        assert_eq!(prefill_linear(PrefillLinear::Fp16, Operator, Some(&a), flag), Ok(PrefillLinear::Fp16));
    }
}
