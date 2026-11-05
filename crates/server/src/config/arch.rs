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
use ignis_core::{ExceptionWeights, PrefillAttention};

use super::profile::CardReading;

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

/// The 27B's prompt attention band on `card` (`--model-prefill-attention`,
/// GitHub #456): `resolved` on a card that runs the 8-bit band's
/// block-scaled E4M3 MMA; elsewhere the FP16 band, when the operator left the
/// field to its default or a profile, and a refusal naming `spelled` when
/// they named the 8-bit band (`named`).
pub fn prefill_attention(
    resolved: PrefillAttention,
    named: bool,
    card: Option<&CardReading>,
    spelled: &str,
) -> Result<PrefillAttention, String> {
    if resolved != PrefillAttention::Fp8 || runs_sm120a(card) {
        return Ok(resolved);
    }
    if !named {
        return Ok(PrefillAttention::Fp16);
    }
    let card = card.expect("a card that runs no sm_120a kernel was read");
    Err(format!(
        "`{spelled} fp8` runs the prompt attention's band on the block-scaled FP8 tensor cores, which compute capability 12.0 has and the card, {} ({}), does not: name `fp16`, or leave it unset and it is `fp16` on this card",
        card.name, card.capability
    ))
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
    /// the FP16 one, a named `fp16` stays, and a named `fp8` is refused by its
    /// spelling; a Blackwell card and no card at all keep what resolved.
    #[test]
    fn the_prefill_attention_follows_the_card() {
        let (fp8, fp16) = (PrefillAttention::Fp8, PrefillAttention::Fp16);
        let flag = "--model-prefill-attention";
        assert_eq!(prefill_attention(fp8, false, Some(&ampere()), flag), Ok(fp16));
        assert_eq!(prefill_attention(fp16, true, Some(&ampere()), flag), Ok(fp16));
        let refused = prefill_attention(fp8, true, Some(&ampere()), "IGNIS_MODEL_PREFILL_ATTENTION").unwrap_err();
        assert!(refused.contains("`IGNIS_MODEL_PREFILL_ATTENTION fp8`") && refused.contains("NVIDIA GeForce RTX 3090 (8.6)"), "{refused}");
        assert!(refused.contains("`fp16`"), "{refused}");
        for named in [false, true] {
            assert_eq!(prefill_attention(fp8, named, Some(&blackwell()), flag), Ok(fp8));
            assert_eq!(prefill_attention(fp8, named, None, flag), Ok(fp8));
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
}
