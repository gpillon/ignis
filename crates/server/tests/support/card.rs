//! The cards the integration tests inject (spec #474), as the start reads
//! them: by total memory, name and compute capability, under the release's
//! build (`86;120a`), whose kernels run on every GPU generation. Included
//! via `#[path]` by the binaries that inject one; the unit tests' copy is
//! `config::profile::testing`.

// Each binary uses a subset of it.
#![allow(dead_code)]

use ignis_server::config::arch::{BuiltArchitectures, ComputeCapability};
use ignis_server::config::profile::{CardReading, RTX_5090_TOTAL_BYTES};

/// A card of `total` bytes named `name`, of compute capability `major.minor`.
pub fn reading(total: u64, name: &str, major: u32, minor: u32) -> CardReading {
    CardReading {
        total,
        name: name.to_owned(),
        capability: ComputeCapability::new(major, minor),
        built_for: BuiltArchitectures::parse("86;120a").expect("the release's list parses"),
    }
}

/// A card of `gib` GiB named `name`, of compute capability `major.minor`.
pub fn card(gib: u64, name: &str, major: u32, minor: u32) -> CardReading {
    reading(gib << 30, name, major, minor)
}

/// The cards spec #474 is written for: 24 GiB each, Ampere and Ada.
pub fn rtx_3090() -> CardReading {
    card(24, "NVIDIA GeForce RTX 3090", 8, 6)
}

pub fn rtx_4090() -> CardReading {
    card(24, "NVIDIA GeForce RTX 4090", 8, 9)
}

/// An RTX 5090, at the total NVML reports for it.
pub fn rtx_5090() -> CardReading {
    reading(RTX_5090_TOTAL_BYTES, "NVIDIA GeForce RTX 5090", 12, 0)
}
