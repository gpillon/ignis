//! The GPU residency's oracle (spec flash-next/03, GitHub #301, acceptance 4):
//! a routing trace and, for every step, what the CPU policy model does with
//! it, written as the text fixture the kernel leaf's residency CTest replays
//! (`kernel/tests/fixtures/residency/trace_v1.txt`). The GPU's hits, misses,
//! evictions and prefetches must equal these, step by step.
//!
//! This test regenerates the fixture from the model and compares it with the
//! committed file byte for byte, so the GPU is always held to what the model
//! does today. After a deliberate change to the policy, rewrite it with
//! `IGNIS_WRITE_RESIDENCY_FIXTURE=1 cargo test -p ignis-core --test expert_residency_fixture`.
//!
//! The trace covers decode rounds of one to three lanes with a lookahead,
//! prefill chunks with a lookahead (the rank-interleaved first-occurrence
//! order over tokens), a prefetch budget that bites, a warm start, steps the
//! model refuses (a class that cannot hold its selection), and forwards
//! restarted at a layer (the same layer twice in a row, in decode and in
//! prefill).
//!
//! Format, whitespace-separated, one record per line:
//! ```text
//! ignis-residency-fixture 2
//! layers <L> experts <E> top_k <10>
//! record_bytes <8 values, KClass order>
//! capacity <8 values>
//! width <W> prefill_width <a prefill step's W> budget <bytes, 18446744073709551615 = none>
//! k2 <L * E * 2 values, key order>
//! warm <n> <keys, hottest first>
//! steps <n>
//! step <layer> <phase 0 decode|1 prefill> <tokens> <tokens * top_k ids> <rows> <stride> <rows * stride ids, -1 = none>
//! expect <status 0 | 1 + class refused>
//! hits <n> <keys>
//! prefetch_hits <n> <keys>
//! misses <n> <key admission>...      (admission 0 slot, 1 staging)
//! evictions <n> <keys>
//! prefetches <n> <key admission>...
//! dropped <n> <keys>
//! bytes <bytes moved>
//! ```
//! A key is `(layer * experts + expert) * 2 + projection` (gate/up 0, down 1):
//! the canonical order. Every list is in key order.

use std::fmt::Write as _;

use ignis_core::residency::{
    Admission, ExpertCatalog, KBits, KClass, LayerStep, PolicyConfig, Projection, ProjectionId,
    ResidencyModel, StepError,
};

const LAYERS: u16 = 4;
const EXPERTS: u16 = 64;
const TOP_K: usize = 10;
const WIDTH: usize = 4;
/// Narrower than [`WIDTH`], so the GPU is held to a prefill step's own width.
const PREFILL_WIDTH: usize = 3;
/// Small records keep the GPU test's pools tiny: gate/up 1-4 pages, down 1-2.
const RECORD_BYTES: [u64; 8] = [4096, 8192, 12288, 16384, 4096, 4096, 8192, 8192];
/// Tight enough to evict, and for the down K4 class tight enough that a
/// three-lane round can be refused.
const CAPACITY: [u32; 8] = [40, 36, 36, 30, 44, 34, 34, 8];
const BUDGET: u64 = 20_480;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    /// `count` distinct experts, skewed toward the low ids as real routing is
    /// toward its hot experts.
    fn experts(&mut self, count: usize) -> Vec<u16> {
        let mut out: Vec<u16> = Vec::with_capacity(count);
        while out.len() < count {
            let e = if self.below(3) == 0 {
                self.below(u64::from(EXPERTS))
            } else {
                self.below(u64::from(EXPERTS) / 4)
            } as u16;
            if !out.contains(&e) {
                out.push(e);
            }
        }
        out
    }
}

fn key(layer: u16, expert: u16, projection: Projection) -> u32 {
    (u32::from(layer) * u32::from(EXPERTS) + u32::from(expert)) * 2
        + u32::from(projection == Projection::Down)
}

fn key_of(id: ProjectionId) -> u32 {
    key(id.layer, id.expert, id.projection)
}

fn k_of_class(index: u64) -> KBits {
    KBits::ALL[index as usize]
}

struct Step {
    layer: u16,
    prefill: bool,
    /// `[tokens][TOP_K]`, the router's selection.
    ids: Vec<Vec<u16>>,
    /// `[rows][WIDTH]`: per lane (decode) or per token (prefill), the next
    /// layer's lookahead ranking; empty at the last layer.
    lookahead: Vec<Vec<i32>>,
}

/// Rounds of decode (one to three lanes) and, every fifth round, a prefill
/// chunk, each round walking the layers in order. The lookahead of layer L
/// is mostly what layer L+1 then selects, with a wrong guess mixed in.
fn trace(rng: &mut Rng, rounds: usize) -> Vec<Step> {
    let mut steps = Vec::new();
    for round in 0..rounds {
        let prefill = round % 5 == 4;
        let rows = if prefill { 6 + rng.below(7) as usize } else { 1 + rng.below(3) as usize };
        let per_layer: Vec<Vec<Vec<u16>>> = (0..LAYERS)
            .map(|_| (0..rows).map(|_| rng.experts(TOP_K)).collect())
            .collect();
        for layer in 0..LAYERS {
            let lookahead = if layer + 1 < LAYERS {
                (0..rows)
                    .map(|row| {
                        let next = &per_layer[usize::from(layer) + 1][row];
                        (0..WIDTH)
                            .map(|r| {
                                if rng.below(4) == 0 {
                                    i32::from(rng.below(u64::from(EXPERTS)) as u16)
                                } else if rng.below(16) == 0 {
                                    -1
                                } else {
                                    i32::from(next[r])
                                }
                            })
                            .collect()
                    })
                    .collect()
            } else {
                Vec::new()
            };
            steps.push(Step {
                layer,
                prefill,
                ids: per_layer[usize::from(layer)].clone(),
                lookahead: lookahead.clone(),
            });
            // A forward restarted at this layer: the same layer twice in a row, the second with
            // other tokens. The first step's staging is released before the second classifies.
            let restart = (round == 24 || round == 34) && layer == 1 || round == 12 && layer == 2;
            if restart {
                steps.push(Step {
                    layer,
                    prefill,
                    ids: (0..rows).map(|_| rng.experts(TOP_K)).collect(),
                    lookahead,
                });
            }
        }
    }
    steps
}

fn catalog(rng: &mut Rng) -> ExpertCatalog {
    let map = (0..usize::from(LAYERS) * usize::from(EXPERTS))
        .map(|_| (k_of_class(rng.below(4)), k_of_class(rng.below(4))))
        .collect();
    ExpertCatalog::new(LAYERS, EXPERTS, map, RECORD_BYTES).expect("catalog")
}

fn list(out: &mut String, name: &str, keys: &[u32]) {
    let _ = write!(out, "{name} {}", keys.len());
    for k in keys {
        let _ = write!(out, " {k}");
    }
    out.push('\n');
}

fn admitted(out: &mut String, name: &str, entries: &[(ProjectionId, Admission)]) {
    let _ = write!(out, "{name} {}", entries.len());
    for (id, admission) in entries {
        let _ = write!(out, " {} {}", key_of(*id), u32::from(*admission == Admission::Staging));
    }
    out.push('\n');
}

fn fixture_text() -> String {
    let mut rng = Rng(301);
    let catalog = catalog(&mut rng);
    let mut model = ResidencyModel::new(
        catalog.clone(),
        PolicyConfig {
            capacity: CAPACITY,
            prefetch_width: WIDTH,
            prefill_prefetch_width: PREFILL_WIDTH,
            prefetch_budget_bytes: Some(BUDGET),
        },
    );
    // A warm start from a made-up calibration ranking: the low experts of
    // every layer, hottest first.
    let warm: Vec<ProjectionId> = (0..12u16)
        .flat_map(|e| (0..LAYERS).map(move |l| (l, e)))
        .flat_map(|(l, e)| Projection::ALL.map(|p| ProjectionId::new(l, e, p)))
        .collect();
    model.warm_start(&warm);
    let steps = trace(&mut rng, 40);

    let mut out = String::new();
    let _ = writeln!(out, "ignis-residency-fixture 2");
    let _ = writeln!(out, "layers {LAYERS} experts {EXPERTS} top_k {TOP_K}");
    let line = |name: &str, values: Vec<String>| format!("{name} {}\n", values.join(" "));
    out.push_str(&line("record_bytes", RECORD_BYTES.iter().map(u64::to_string).collect()));
    out.push_str(&line("capacity", CAPACITY.iter().map(u32::to_string).collect()));
    let _ = writeln!(out, "width {WIDTH} prefill_width {PREFILL_WIDTH} budget {BUDGET}");
    let mut k2 = Vec::new();
    for layer in 0..LAYERS {
        for expert in 0..EXPERTS {
            for projection in Projection::ALL {
                let class = catalog.class_of(ProjectionId::new(layer, expert, projection));
                k2.push(class.k.half_bits().to_string());
            }
        }
    }
    out.push_str(&line("k2", k2));
    let warm_keys: Vec<u32> = warm.iter().map(|&id| key_of(id)).collect();
    let _ = write!(out, "warm {}", warm_keys.len());
    for k in &warm_keys {
        let _ = write!(out, " {k}");
    }
    out.push('\n');
    let _ = writeln!(out, "steps {}", steps.len());

    let mut refused = 0;
    for s in &steps {
        let _ = write!(out, "step {} {} {}", s.layer, u32::from(s.prefill), s.ids.len());
        for row in &s.ids {
            for e in row {
                let _ = write!(out, " {e}");
            }
        }
        let stride = if s.lookahead.is_empty() { 0 } else { WIDTH };
        let _ = write!(out, " {} {stride}", s.lookahead.len());
        for row in &s.lookahead {
            for e in row {
                let _ = write!(out, " {e}");
            }
        }
        out.push('\n');

        let selected: Vec<u16> = s.ids.iter().flatten().copied().collect();
        // The model takes a lane's ranking without its holes, as the GPU's
        // walk skips them.
        let rows: Vec<Vec<u16>> = s
            .lookahead
            .iter()
            .map(|row| row.iter().filter(|&&e| e >= 0).map(|&e| e as u16).collect())
            .collect();
        let lanes: Vec<&[u16]> = rows.iter().map(Vec::as_slice).collect();
        let step = if s.prefill {
            LayerStep::prefill(s.layer, &selected)
        } else {
            LayerStep::decode(s.layer, &selected)
        };
        match model.step(&step.lookahead(&lanes)) {
            Ok(o) => {
                let _ = writeln!(out, "expect 0");
                list(&mut out, "hits", &o.hits.iter().map(|&id| key_of(id)).collect::<Vec<_>>());
                list(
                    &mut out,
                    "prefetch_hits",
                    &o.prefetch_hits.iter().map(|&id| key_of(id)).collect::<Vec<_>>(),
                );
                admitted(&mut out, "misses", &o.misses);
                list(&mut out, "evictions", &o.evictions.iter().map(|&id| key_of(id)).collect::<Vec<_>>());
                admitted(&mut out, "prefetches", &o.prefetches);
                list(
                    &mut out,
                    "dropped",
                    &o.prefetch_dropped.iter().map(|&id| key_of(id)).collect::<Vec<_>>(),
                );
                let _ = writeln!(out, "bytes {}", o.bytes_moved);
            }
            Err(StepError::NoEvictableSlot { class, .. }) => {
                refused += 1;
                let _ = writeln!(out, "expect {}", 1 + class.index());
            }
            Err(e) => panic!("{e}"),
        }
    }
    assert!(refused > 0, "the trace must exercise a refused step");
    out
}

fn fixture_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernel/tests/fixtures/residency/trace_v1.txt")
}

#[test]
fn the_committed_gpu_fixture_is_what_the_policy_model_produces() {
    let text = fixture_text();
    let path = fixture_path();
    if std::env::var_os("IGNIS_WRITE_RESIDENCY_FIXTURE").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &text).unwrap();
    }
    let committed = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e} (write it with IGNIS_WRITE_RESIDENCY_FIXTURE=1)", path.display()));
    assert!(
        committed == text,
        "{} is not what the policy model produces now; if the policy changed on purpose, \
         rewrite it with IGNIS_WRITE_RESIDENCY_FIXTURE=1",
        path.display()
    );
}

#[test]
fn the_fixture_exercises_every_path_the_gpu_must_match() {
    let text = fixture_text();
    let count = |prefix: &str| {
        text.lines()
            .filter(|l| l.starts_with(prefix))
            .filter(|l| l.split_whitespace().nth(1).is_some_and(|n| n != "0"))
            .count()
    };
    assert!(count("evictions") > 10, "evictions");
    assert!(count("prefetch_hits") > 10, "prefetch hits");
    assert!(count("dropped") > 10, "a budget that bites");
    // Staging admissions appear in prefill misses or prefetches.
    assert!(
        text.lines()
            .filter(|l| l.starts_with("misses") || l.starts_with("prefetches"))
            .any(|l| {
                let v: Vec<&str> = l.split_whitespace().skip(2).collect();
                v.chunks(2).any(|p| p.get(1) == Some(&"1"))
            }),
        "staging"
    );
    let _ = KClass::ALL;
}
