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

    /// The capabilities the kernels run on, with the cards that have them:
    /// "12.0 (RTX 50, RTX PRO 6000 Blackwell)", "8.6 and up within 8.x (RTX
    /// 30, RTX 40) and 12.0 (…)".
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
                    KNOWN_SERIES.iter().filter(|(capability, _)| arch.runs_on(*capability)).map(|(_, series)| *series).collect();
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

/// The card series a refusal names beside a capability, by architecture.
const KNOWN_SERIES: [(ComputeCapability, &str); 3] = [
    (ComputeCapability::new(8, 6), "RTX 30"),
    (ComputeCapability::new(8, 9), "RTX 40"),
    (ComputeCapability::new(12, 0), "RTX 50, RTX PRO 6000 Blackwell"),
];

/// The refusal of a start on a card of `capability` named `name` that the
/// build's kernels do not run on; `None` when they do. A start on such a card
/// could only fail at its first kernel launch, with CUDA's own "no kernel
/// image is available", so it is refused before anything loads, whatever the
/// profile.
pub fn uncovered(name: &str, capability: ComputeCapability, built_for: &BuiltArchitectures) -> Option<String> {
    (!built_for.cover(capability)).then(|| {
        format!(
            "this build's kernels run on compute capability {}, built for {built_for}; the card is {name} ({capability}), which they were not built for: see GitHub #456",
            built_for.describe()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cc(major: u32, minor: u32) -> ComputeCapability {
        ComputeCapability::new(major, minor)
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
    /// 30, not on the DGX Spark's 12.1, not on a datacenter Hopper.
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
        assert!(refusal.contains("compute capability 12.0 (RTX 50, RTX PRO 6000 Blackwell)"), "{refusal}");
        assert!(refusal.contains("built for 120a;") && refusal.contains("#456"), "{refusal}");
        assert_eq!(uncovered("NVIDIA GeForce RTX 5090", cc(12, 0), &built), None);
        let ampere = BuiltArchitectures::parse("86;120a").unwrap();
        assert_eq!(uncovered("NVIDIA GeForce RTX 3090", cc(8, 6), &ampere), None);
        let turing = uncovered("NVIDIA GeForce RTX 2080 Ti", cc(7, 5), &ampere).unwrap();
        assert!(turing.contains("8.6 and up within 8.x (RTX 30, RTX 40) and 12.0 (RTX 50"), "{turing}");
    }
}
