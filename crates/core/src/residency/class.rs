//! The unit residency moves and the eight slot classes it falls into.
//!
//! The unit is one **expert projection**: an expert's fused gate/up plane or
//! its down plane. Its byte size is fixed by its shape and its expert's K
//! (spec flash-next/01's layout: tiles and channel scales in one 4 KiB-aligned
//! range), so two shapes × four K make eight **K classes**, and every slot of
//! a class's pool is interchangeable.

/// Which plane of an expert a projection is. Gate/up sorts before down,
/// as the artifact orders them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Projection {
    /// The fused gate/up plane, hidden → 2 × intermediate.
    GateUp,
    /// The down plane, intermediate → hidden.
    Down,
}

impl Projection {
    pub const ALL: [Projection; 2] = [Projection::GateUp, Projection::Down];

    pub fn as_str(self) -> &'static str {
        match self {
            Projection::GateUp => "gate_up",
            Projection::Down => "down",
        }
    }
}

/// An expert projection's trellis bit width (ADR 0044).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KBits {
    K2,
    /// The mul1 codebook's half step.
    K2_5,
    K3,
    K4,
}

impl KBits {
    pub const ALL: [KBits; 4] = [KBits::K2, KBits::K2_5, KBits::K3, KBits::K4];

    /// Bits per weight, in half bits (4, 5, 6, 8), so arithmetic on K stays
    /// in integers.
    pub fn half_bits(self) -> u32 {
        match self {
            KBits::K2 => 4,
            KBits::K2_5 => 5,
            KBits::K3 => 6,
            KBits::K4 => 8,
        }
    }

    /// The K whose [`KBits::half_bits`] is `half_bits` — the artifact's
    /// `k2` field (layout.md §3) — if it is one of the four.
    pub fn from_half_bits(half_bits: u32) -> Option<KBits> {
        KBits::ALL.into_iter().find(|k| k.half_bits() == half_bits)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            KBits::K2 => "k2",
            KBits::K2_5 => "k2_5",
            KBits::K3 => "k3",
            KBits::K4 => "k4",
        }
    }
}

/// One of the eight slot classes: a projection shape at one K.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KClass {
    pub projection: Projection,
    pub k: KBits,
}

impl KClass {
    pub const COUNT: usize = 8;

    /// Gate/up at K = 2, 2.5, 3, 4, then down at the same four: the order of
    /// every per-class array in this module.
    pub const ALL: [KClass; Self::COUNT] = [
        KClass::new(Projection::GateUp, KBits::K2),
        KClass::new(Projection::GateUp, KBits::K2_5),
        KClass::new(Projection::GateUp, KBits::K3),
        KClass::new(Projection::GateUp, KBits::K4),
        KClass::new(Projection::Down, KBits::K2),
        KClass::new(Projection::Down, KBits::K2_5),
        KClass::new(Projection::Down, KBits::K3),
        KClass::new(Projection::Down, KBits::K4),
    ];

    pub const fn new(projection: Projection, k: KBits) -> Self {
        Self { projection, k }
    }

    /// The class's position in [`KClass::ALL`].
    pub fn index(self) -> usize {
        let shape = match self.projection {
            Projection::GateUp => 0,
            Projection::Down => 4,
        };
        let k = match self.k {
            KBits::K2 => 0,
            KBits::K2_5 => 1,
            KBits::K3 => 2,
            KBits::K4 => 3,
        };
        shape + k
    }

    /// The `class` label value: `gate_up_k2` … `down_k4`, eight fixed
    /// spellings (ADR 0017's bounded cardinality).
    pub fn as_str(self) -> &'static str {
        const NAMES: [&str; KClass::COUNT] = [
            "gate_up_k2",
            "gate_up_k2_5",
            "gate_up_k3",
            "gate_up_k4",
            "down_k2",
            "down_k2_5",
            "down_k3",
            "down_k4",
        ];
        NAMES[self.index()]
    }
}

/// One expert projection: (layer, expert, plane). The derived order —
/// layer, then expert, then gate/up before down — is the **canonical key
/// order** every tie-break and every reported sequence in this module uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProjectionId {
    pub layer: u16,
    pub expert: u16,
    pub projection: Projection,
}

impl ProjectionId {
    pub const fn new(layer: u16, expert: u16, projection: Projection) -> Self {
        Self {
            layer,
            expert,
            projection,
        }
    }
}

/// A K map that does not cover the model's experts exactly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogMismatch {
    pub expected: usize,
    pub got: usize,
}

impl std::fmt::Display for CatalogMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the K map names {} experts, the model has {}",
            self.got, self.expected
        )
    }
}

impl std::error::Error for CatalogMismatch {}

/// Every routed expert's K for each plane, and the byte size of one slot of
/// each class: what residency knows about the artifact's experts. Both come
/// from the artifact (its expert index and sidecar); this module never
/// derives a size from a formula of its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertCatalog {
    layers: u16,
    experts: u16,
    /// `(gate/up K, down K)` per expert, layer-major.
    k_map: Vec<(KBits, KBits)>,
    slot_bytes: [u64; KClass::COUNT],
}

impl ExpertCatalog {
    /// `k_map` holds `layers × experts` entries, layer-major; `slot_bytes`
    /// is indexed like [`KClass::ALL`].
    pub fn new(
        layers: u16,
        experts: u16,
        k_map: Vec<(KBits, KBits)>,
        slot_bytes: [u64; KClass::COUNT],
    ) -> Result<Self, CatalogMismatch> {
        let expected = usize::from(layers) * usize::from(experts);
        if k_map.len() != expected {
            return Err(CatalogMismatch {
                expected,
                got: k_map.len(),
            });
        }
        Ok(Self {
            layers,
            experts,
            k_map,
            slot_bytes,
        })
    }

    pub fn layers(&self) -> u16 {
        self.layers
    }

    pub fn experts(&self) -> u16 {
        self.experts
    }

    /// Whether the catalog has this projection's layer and expert.
    pub fn contains(&self, id: ProjectionId) -> bool {
        id.layer < self.layers && id.expert < self.experts
    }

    /// The projection's class. Panics outside the catalog: callers check
    /// [`ExpertCatalog::contains`] at their boundary.
    pub fn class_of(&self, id: ProjectionId) -> KClass {
        let (gate_up, down) =
            self.k_map[usize::from(id.layer) * usize::from(self.experts) + usize::from(id.expert)];
        match id.projection {
            Projection::GateUp => KClass::new(Projection::GateUp, gate_up),
            Projection::Down => KClass::new(Projection::Down, down),
        }
    }

    /// One slot of `class`, in bytes.
    pub fn slot_bytes(&self, class: KClass) -> u64 {
        self.slot_bytes[class.index()]
    }

    /// What copying this projection moves.
    pub fn bytes(&self, id: ProjectionId) -> u64 {
        self.slot_bytes(self.class_of(id))
    }

    /// How many projections each class holds, indexed like [`KClass::ALL`].
    pub fn class_counts(&self) -> [u64; KClass::COUNT] {
        let mut counts = [0u64; KClass::COUNT];
        for &(gate_up, down) in &self.k_map {
            counts[KClass::new(Projection::GateUp, gate_up).index()] += 1;
            counts[KClass::new(Projection::Down, down).index()] += 1;
        }
        counts
    }

    /// Every projection's bytes: the pinned host expert pool's size.
    pub fn total_bytes(&self) -> u64 {
        self.class_counts()
            .iter()
            .zip(self.slot_bytes)
            .map(|(count, bytes)| count * bytes)
            .sum()
    }

    /// The bytes of every projection of one layer: the most a prefill chunk
    /// can touch there.
    pub fn layer_bytes(&self, layer: u16) -> u64 {
        let experts = usize::from(self.experts);
        self.k_map[usize::from(layer) * experts..(usize::from(layer) + 1) * experts]
            .iter()
            .map(|&(gate_up, down)| {
                self.slot_bytes(KClass::new(Projection::GateUp, gate_up))
                    + self.slot_bytes(KClass::new(Projection::Down, down))
            })
            .sum()
    }
}
