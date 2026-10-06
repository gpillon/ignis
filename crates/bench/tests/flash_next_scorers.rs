//! Flash-Next's acceptance scorers (`ignis_bench::flash_next`, spec
//! flash-next/04 acceptance 4-6 and 8), CPU only:
//!
//! - the KLD and McNemar ports against the converter's own Python
//!   (`tests/fixtures/flash_next_kld/`, recorded by its `record.py` from
//!   `tools/flash-next-converter/scoring.py`);
//! - a whole pass through the [`SpanLogits`] seam on a mock engine over a
//!   small reference set written here, its figures checked against the
//!   exact KL over the full vocabulary, computed independently;
//! - the MMLU questions found by token identity, the G1 run's row mapping,
//!   and the seam's refusals.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ignis_bench::flash_next::kld::{self, DenseRow, bf16_to_f32};
use ignis_bench::flash_next::mmlu::{self, MmluSet};
use ignis_bench::flash_next::references::{ConverterRecord, ReferenceSet, TOP, TopRow};
use ignis_bench::flash_next::{RowSink, SpanLogits, g1, run_acceptance, run_set};
use ignis_bench::oracle::{Fixture, FixturePrompt};
use serde::Deserialize;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join("flash_next_kld")
}

#[derive(Deserialize)]
struct Case {
    name: String,
    ref_ids: Vec<i32>,
    ref_lp: Vec<f32>,
    kl_f32: f64,
    kl_f64: f64,
    cand_argmax: u32,
    ref_argmax: u32,
}

#[derive(Deserialize)]
struct McNemarCase {
    lost: usize,
    gained: usize,
    p: f64,
}

#[derive(Deserialize)]
struct Recorded {
    vocab: usize,
    top: usize,
    cases: Vec<Case>,
    mcnemar: Vec<McNemarCase>,
}

fn recorded() -> (Recorded, Vec<u16>) {
    let dir = fixture_dir();
    let cases: Recorded = serde_json::from_str(&std::fs::read_to_string(dir.join("cases.json")).unwrap()).unwrap();
    let bytes = std::fs::read(dir.join("cand_bf16.bin")).unwrap();
    let rows: Vec<u16> = bytes.as_chunks::<2>().0.iter().map(|&b| u16::from_le_bytes(b)).collect();
    assert_eq!(rows.len(), cases.cases.len() * cases.vocab, "cand_bf16.bin holds one row per case");
    (cases, rows)
}

#[test]
fn the_kld_port_is_the_converters_top64_scorer_on_an_engine_row() {
    let (recorded, rows) = recorded();
    assert_eq!(recorded.top, TOP);
    for (case, row) in recorded.cases.iter().zip(rows.chunks_exact(recorded.vocab)) {
        let row = DenseRow::new(row).unwrap();
        let reference = TopRow { ids: &case.ref_ids, lp: &case.ref_lp };
        let score = kld::score_dense(reference, &row).unwrap();
        // The same function on the same inputs in float64: equal to rounding.
        assert!(
            (score.kl - case.kl_f64).abs() <= 1e-10 + 1e-12 * case.kl_f64.abs(),
            "{}: {} against the converter's {} (float64)",
            case.name,
            score.kl,
            case.kl_f64
        );
        // The converter's own fp32 figure: the stored log-probs are fp32
        // values of logits up to |64| (ulp 7.6e-6), so the tail mass and the
        // figure move by a few 1e-6 at most.
        assert!(
            (score.kl - case.kl_f32).abs() <= 2e-5 + 1e-6 * case.kl_f32.abs(),
            "{}: {} against the converter's {} (fp32)",
            case.name,
            score.kl,
            case.kl_f32
        );
        assert_eq!(row.argmax(), case.cand_argmax, "{}: argmax, lowest id on ties", case.name);
        assert_eq!(score.top1, case.cand_argmax == case.ref_argmax, "{}: top-1", case.name);
    }
    assert!(recorded.cases.iter().any(|c| c.name == "moderate-tied-max"), "the tie case is recorded");
}

#[test]
fn mcnemar_is_the_converters_exact_binomial_test() {
    let (recorded, _) = recorded();
    for case in &recorded.mcnemar {
        let p = mmlu::mcnemar_p(case.lost, case.gained);
        assert!((p - case.p).abs() <= 1e-12 * case.p.max(1e-300), "({}, {}): {p} against {}", case.lost, case.gained, case.p);
    }
    let paired = mmlu::paired(&[true, true, false, false, true], &[true, false, true, false, false]);
    assert_eq!((paired.lost, paired.gained), (2, 1));
}

#[test]
fn a_row_with_a_nan_or_an_infinity_is_refused() {
    let mut row = vec![0x3f80u16; 8];
    row[5] = 0x7fc0;
    assert!(DenseRow::new(&row).unwrap_err().contains("logit 5"));
    row[5] = 0x7f80;
    assert!(DenseRow::new(&row).is_err());
}

#[test]
fn a_top64_row_settles_an_answer_only_when_no_missing_letter_could_tie() {
    let ids: Vec<i32> = (100..164).collect();
    let mut lp: Vec<f32> = (0..64).map(|k| -1.0 - k as f32 * 0.1).collect();
    let letters = [100u32, 163, 7, 101];
    // Letter 0 is the most probable present letter, letter 2 is missing.
    assert_eq!(mmlu::answer_top64(TopRow { ids: &ids, lp: &lp }, &letters, 4), Some(0));
    // Only the 64th entry's letter is present and a letter is missing: it could tie.
    assert_eq!(mmlu::answer_top64(TopRow { ids: &ids, lp: &lp }, &[163, 7], 2), None);
    // No option letter is among the 64.
    assert_eq!(mmlu::answer_top64(TopRow { ids: &ids, lp: &lp }, &[7, 8], 2), None);
    // Equal log-probs: the first letter wins.
    lp[1] = lp[0];
    assert_eq!(mmlu::answer_top64(TopRow { ids: &ids, lp: &lp }, &[101, 100], 2), Some(0));
}

// ── a small world: a reference model, a quantized one, an engine ─────────

const VOCAB: usize = 300;
const LETTERS: [u32; 4] = [10, 11, 12, 13];

fn mix(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^ (h >> 33)
}

fn unit(h: u64) -> f64 {
    (mix(h) >> 11) as f64 / (1u64 << 53) as f64
}

fn bf16(x: f64) -> u16 {
    let bits = (x as f32).to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// A model's BF16 logits after `prefix`: 64 live tokens (the four letters
/// and 60 hashed others) with spread logits, every other token 60 below,
/// so the reference's mass outside its top-64 is under 1e-12 and the
/// top-64 KL is the full-vocabulary KL. `noise` (with its own salt) is the
/// engine's or the quantized stream's distance from the reference.
fn model(prefix: &[u32], noise: f64, salt: u64) -> Vec<u16> {
    let h = prefix.iter().fold(0x9e3779b97f4a7c15u64, |h, &t| mix(h ^ t as u64));
    let mut live: Vec<usize> = LETTERS.iter().map(|&l| l as usize).collect();
    let mut k = 0u64;
    while live.len() < 64 {
        let id = (mix(h ^ k.wrapping_mul(0x51)) % VOCAB as u64) as usize;
        if !live.contains(&id) {
            live.push(id);
        }
        k += 1;
    }
    let mut logits = vec![-40.0f64; VOCAB];
    for (j, &id) in live.iter().enumerate() {
        let base = 4.0 * unit(h ^ (j as u64) << 20) + 2.0 * unit(h ^ 7);
        logits[id] = base + noise * (unit(h ^ salt ^ (id as u64) << 32) - 0.5);
    }
    logits.into_iter().map(bf16).collect()
}

const REF_SALT: u64 = 0;
const Q_SALT: u64 = 0xabc;
const ENGINE_SALT: u64 = 0x5eed;

fn log_softmax(row: &[u16]) -> Vec<f64> {
    let x: Vec<f64> = row.iter().map(|&v| bf16_to_f32(v) as f64).collect();
    let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lse = max + x.iter().map(|v| (v - max).exp()).sum::<f64>().ln();
    x.iter().map(|v| v - lse).collect()
}

fn top64(lp: &[f64]) -> (Vec<i32>, Vec<f32>) {
    let mut order: Vec<usize> = (0..lp.len()).collect();
    order.sort_by(|&a, &b| lp[b].partial_cmp(&lp[a]).unwrap().then(a.cmp(&b)));
    order.truncate(TOP);
    (order.iter().map(|&i| i as i32).collect(), order.iter().map(|&i| lp[i] as f32).collect())
}

fn argmax(lp: &[f64]) -> usize {
    (0..lp.len()).fold(0, |best, i| if lp[i] > lp[best] { i } else { best })
}

/// The engine: the reference model plus noise (`salt` picks which: the
/// engine's own, or the quantized stream's).
struct MockEngine {
    noise: f64,
    salt: u64,
    rows_per_chunk: usize,
    calls: usize,
}

impl SpanLogits for MockEngine {
    fn vocab(&self) -> usize {
        VOCAB
    }

    fn span_logits(&mut self, tokens: &[u32], sink: &mut RowSink<'_>) -> Result<(), String> {
        self.calls += 1;
        for first in (0..tokens.len()).step_by(self.rows_per_chunk) {
            let rows: Vec<u16> = (first..(first + self.rows_per_chunk).min(tokens.len()))
                .flat_map(|i| model(&tokens[..=i], self.noise, self.salt))
                .collect();
            sink(first, &rows)?;
        }
        Ok(())
    }
}

struct World {
    dir: PathBuf,
    windows: Vec<(String, String, Vec<u32>, usize)>,
}

fn write_words<T: Copy>(path: &Path, values: &[T], encode: fn(T) -> [u8; 4]) {
    std::fs::write(path, values.iter().flat_map(|&v| encode(v)).collect::<Vec<u8>>()).unwrap();
}

/// Writes a reference set the way the converter's `RefWriter` does: the
/// BF16 stream from the reference model, the quantized stream from its own
/// noise, every stored position of every window.
fn write_set(name: &str, windows: &[(&str, &str, Vec<u32>, usize)]) -> World {
    let dir = temp_dir(name);
    write_set_at(&dir, windows);
    World {
        dir,
        windows: windows.iter().map(|(k, s, ids, v)| (k.to_string(), s.to_string(), ids.clone(), *v)).collect(),
    }
}

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ignis-flash-next-scorers-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_set_at(dir: &Path, windows: &[(&str, &str, Vec<u32>, usize)]) {
    std::fs::create_dir_all(dir).unwrap();
    let (mut tokens, mut manifest) = (Vec::new(), Vec::new());
    let (mut r_ids, mut r_lp, mut r_lse, mut q_ids, mut q_lp, mut q_lse, mut q_arg) =
        (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    let mut first = 0usize;
    for (index, (kind, source, ids, valid)) in windows.iter().enumerate() {
        manifest.push(serde_json::json!({"index": index, "kind": kind, "source": source, "length": ids.len(),
                                         "valid": valid, "first_position": first}));
        tokens.extend_from_slice(ids);
        for p in 0..*valid {
            let lr = log_softmax(&model(&ids[..=p], 0.0, REF_SALT));
            let lq = log_softmax(&model(&ids[..=p], 0.6, Q_SALT));
            let (a, b) = top64(&lr);
            r_ids.extend(a);
            r_lp.extend(b);
            r_lse.push(0.0f32);
            let (a, b) = top64(&lq);
            q_ids.extend(a);
            q_lp.extend(b);
            q_lse.push(0.0f32);
            q_arg.push(argmax(&lq) as i32);
        }
        first += valid;
    }
    let manifest = serde_json::json!({"windows": manifest, "positions": first, "top": TOP});
    std::fs::write(dir.join("manifest.json"), manifest.to_string()).unwrap();
    write_words(&dir.join("tokens.u32"), &tokens, u32::to_le_bytes);
    write_words(&dir.join("bf16_top64_ids.i32"), &r_ids, i32::to_le_bytes);
    write_words(&dir.join("bf16_top64_lp.f32"), &r_lp, f32::to_le_bytes);
    write_words(&dir.join("bf16_lse.f32"), &r_lse, f32::to_le_bytes);
    write_words(&dir.join("q_top64_ids.i32"), &q_ids, i32::to_le_bytes);
    write_words(&dir.join("q_top64_lp.f32"), &q_lp, f32::to_le_bytes);
    write_words(&dir.join("q_lse.f32"), &q_lse, f32::to_le_bytes);
    write_words(&dir.join("q_argmax.i32"), &q_arg, i32::to_le_bytes);
}

impl Drop for World {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn ids(seed: u64, n: usize) -> Vec<u32> {
    (0..n).map(|i| 14 + (mix(seed ^ i as u64) % (VOCAB as u64 - 14)) as u32).collect()
}

const MARKS: [(usize, u8, u8); 4] = [(3, 1, 4), (6, 0, 3), (8, 3, 4), (11, 2, 4)];

/// The 2048-token set in small: two domains, a window with padding past
/// `valid`, and the questions' window.
fn test_windows() -> Vec<(&'static str, &'static str, Vec<u32>, usize)> {
    vec![
        ("code", "windows.json+long_windows.json", ids(1, 12), 12),
        ("prose", "windows.json+long_windows.json", ids(2, 12), 12),
        ("code", "windows.json+long_windows.json", ids(3, 12), 9),
        ("mmlu", "chunks.json", ids(4, 12), 12),
    ]
}

fn world(name: &str) -> World {
    write_set(name, &test_windows())
}

/// The corpus file the questions come from: the held-out chunk carrying
/// the marks (one of them on the last position, which the converter never
/// scores), plus a calibration chunk with the same tokens that must not be
/// taken for it.
fn write_chunks(world: &World) -> PathBuf {
    write_chunks_at(&world.dir, &world.windows[3].2, world.windows[3].3, &MARKS)
}

fn write_chunks_at(dir: &Path, tokens: &[u32], valid: usize, marks: &[(usize, u8, u8)]) -> PathBuf {
    let marks: Vec<serde_json::Value> =
        marks.iter().map(|&(p, g, o)| serde_json::json!([p, g, o, "law"])).collect();
    let chunks = serde_json::json!({
        "chunks": [
            {"ids": tokens, "kind": "mmlu", "test": false, "valid": valid, "mmlu": [[2, 0, 4, "decoy"]]},
            {"ids": ids(9, 12), "kind": "en", "test": true, "valid": 12, "mmlu": []},
            {"ids": tokens, "kind": "mmlu", "test": true, "valid": valid, "mmlu": marks},
        ],
        "letter_ids": LETTERS,
    });
    let path = dir.join("chunks.json");
    std::fs::write(&path, chunks.to_string()).unwrap();
    path
}

#[test]
fn a_pass_scores_every_domain_against_the_full_vocabulary_kl() {
    let world = world("pass");
    let set = ReferenceSet::read(&world.dir).unwrap();
    let mut engine = MockEngine { noise: 0.8, salt: ENGINE_SALT, rows_per_chunk: 5, calls: 0 };
    let run = run_set(&mut engine, &set, None).unwrap();
    assert_eq!(engine.calls, 4, "one prefill per window");

    // Independently: the exact KL over all 300 columns, at every position
    // with a next token, pooled per kind.
    let mut want: BTreeMap<&str, (u64, f64, u64)> = BTreeMap::new();
    for (kind, _, ids, valid) in &world.windows {
        for p in 0..valid - 1 {
            let lr = log_softmax(&model(&ids[..=p], 0.0, REF_SALT));
            let le = log_softmax(&model(&ids[..=p], 0.8, ENGINE_SALT));
            let kl: f64 = lr.iter().zip(&le).map(|(r, e)| r.exp() * (r - e)).sum();
            let entry = want.entry(kind.as_str()).or_default();
            entry.0 += 1;
            entry.1 += kl;
            entry.2 += (argmax(&lr) == argmax(&le)) as u64;
        }
    }
    assert_eq!(run.kld.domains.keys().map(String::as_str).collect::<Vec<_>>(), want.keys().copied().collect::<Vec<_>>());
    for (kind, &(n, kl, top1)) in &want {
        let got = run.kld.domains[*kind];
        assert_eq!(got.positions, n, "{kind}: positions with a next token");
        let mean = kl / n as f64;
        assert!(mean > 1e-3, "{kind}: the engine is measurably off the reference ({mean})");
        // The stored log-probs are fp32: their rounding (about 1e-7 of mass)
        // reaches the tail term, a few 1e-7 nats.
        assert!((got.kld_top64 - mean).abs() <= 5e-6, "{kind}: {} against {mean}", got.kld_top64);
        assert_eq!(got.top1, top1 as f64 / n as f64, "{kind}: top-1");
    }
    // The same windows through the quantized stream's stored rows: every
    // BF16 top-64 id is live in it too, so its KL is known everywhere.
    for row in 0..set.positions() {
        assert!(kld::score_quantized(&set, row).0.is_some());
    }
}

#[test]
fn a_domain_passes_within_its_quantization_only_figure_times_1_1_or_plus_0_01() {
    assert_eq!(kld::kld_limit(0.05), 0.060000000000000005);
    assert_eq!(kld::kld_limit(0.2), 0.22000000000000003);
    let figures = |pairs: &[(&str, f64)]| -> BTreeMap<String, ignis_bench::flash_next::references::DomainFigures> {
        pairs
            .iter()
            .map(|&(d, q)| (d.to_string(), serde_json::from_value(serde_json::json!({"kld": q * 1.2, "kld_top64": q, "top1": 0.9})).unwrap()))
            .collect()
    };
    let engine = |pairs: &[(&str, f64)]| kld::KldReport {
        domains: pairs
            .iter()
            .map(|&(d, e)| (d.to_string(), kld::DomainKld { positions: 10, kld_top64: e, top1: 0.9 }))
            .collect(),
    };
    let verdicts = kld::judge(&engine(&[("code", 0.0599), ("prose", 0.2201)]), &figures(&[("code", 0.05), ("prose", 0.2)])).unwrap();
    assert_eq!(verdicts.iter().map(|v| v.pass).collect::<Vec<_>>(), [true, false]);
    assert!(kld::judge(&engine(&[("code", 0.05)]), &figures(&[("code", 0.05), ("prose", 0.2)])).is_err());
}

#[test]
fn the_mmlu_questions_are_found_by_token_identity_and_answered_at_their_rows() {
    let world = world("mmlu");
    let set = ReferenceSet::read(&world.dir).unwrap();
    let questions = MmluSet::from_chunks(&write_chunks(&world), &set).unwrap();
    // The decoy's mark is not taken; the mark on the last position is left out.
    assert_eq!(questions.questions.iter().map(|q| (q.position, q.gold)).collect::<Vec<_>>(), [(3, 1), (6, 0), (8, 3)]);
    assert!(questions.questions.iter().all(|q| q.window == 3));

    let mut engine = MockEngine { noise: 0.8, salt: ENGINE_SALT, rows_per_chunk: 4, calls: 0 };
    let run = run_set(&mut engine, &set, Some(&questions)).unwrap();
    let (_, _, tokens, _) = &world.windows[3];
    let pick = |noise, salt, q: &mmlu::Question| -> u8 {
        let lp = log_softmax(&model(&tokens[..=q.position], noise, salt));
        (0..q.options as usize).fold(0usize, |b, k| if lp[LETTERS[k] as usize] > lp[LETTERS[b] as usize] { k } else { b }) as u8
    };
    let want: Vec<u8> = questions.questions.iter().map(|q| pick(0.8, ENGINE_SALT, q)).collect();
    assert_eq!(run.answers, want);
    let bf16 = questions.stored_answers(&set, false).unwrap();
    let quantized = questions.stored_answers(&set, true).unwrap();
    assert_eq!(bf16, questions.questions.iter().map(|q| pick(0.0, REF_SALT, q)).collect::<Vec<_>>());
    assert_eq!(quantized, questions.questions.iter().map(|q| pick(0.6, Q_SALT, q)).collect::<Vec<_>>());

    let report = mmlu::judge(&questions, &run.answers, &bf16, &quantized).unwrap();
    let ok = |answers: &[u8]| questions.correct(answers).iter().filter(|&&b| b).count();
    assert_eq!(report.n, 3);
    assert_eq!(report.accuracy, ok(&run.answers) as f64 / 3.0);
    assert_eq!(report.bf16_accuracy, ok(&bf16) as f64 / 3.0);
    assert_eq!(report.pass, report.accuracy >= mmlu::MMLU_FLOOR);
}

#[test]
fn a_window_unknown_to_the_corpus_file_is_refused() {
    let world = world("unknown");
    let set = ReferenceSet::read(&world.dir).unwrap();
    let path = world.dir.join("chunks.json");
    std::fs::write(&path, serde_json::json!({"chunks": [], "letter_ids": LETTERS}).to_string()).unwrap();
    assert!(MmluSet::from_chunks(&path, &set).unwrap_err().contains("window 3"));
}

/// A G1 fixture as the converter records it: each canary's expected
/// column is the quantized model's argmax after the prompt and the
/// canary's first `i` tokens.
fn g1_fixture() -> Fixture {
    let canary = |id: &str, prompt: Vec<u32>, tokens: Vec<u32>| {
        let expected: Vec<u32> = (0..tokens.len())
            .map(|i| argmax(&log_softmax(&model(&[&prompt[..], &tokens[..i]].concat(), 0.6, Q_SALT))) as u32)
            .collect();
        FixturePrompt {
            id: id.into(),
            token_ids: tokens,
            prompt_token_ids: Some(prompt),
            expected_argmax: Some(expected),
            ..Default::default()
        }
    };
    Fixture {
        model: "qwen3.8-flash-next-ignis".into(),
        max_tokens: 32,
        prompts: vec![canary("a", ids(20, 7), ids(21, 9)), canary("b", ids(22, 3), ids(23, 14))],
        render: None,
        reference: Some("quantized".into()),
    }
}

#[test]
fn g1_scores_each_canary_row_against_the_prefix_its_expected_argmax_was_recorded_after() {
    let fixture = g1_fixture();
    let mut engine = MockEngine { noise: 1.5, salt: ENGINE_SALT, rows_per_chunk: 4, calls: 0 };
    let run = g1::run(&mut engine, &fixture).unwrap();

    let mut agree_total = 0;
    for (prompt, result) in fixture.prompts.iter().zip(&run.results) {
        let p = prompt.prompt_token_ids.as_ref().unwrap();
        let expected = prompt.expected_argmax.as_ref().unwrap();
        let mismatched: Vec<usize> = (0..prompt.token_ids.len())
            .filter(|&i| {
                let le = log_softmax(&model(&[&p[..], &prompt.token_ids[..i]].concat(), 1.5, ENGINE_SALT));
                argmax(&le) as u32 != expected[i]
            })
            .collect();
        assert_eq!(result.compared, prompt.token_ids.len());
        assert_eq!(result.mismatches.iter().map(|m| m.position).collect::<Vec<_>>(), mismatched, "{}", prompt.id);
        agree_total += result.agree;
    }
    assert!(run.results.iter().any(|r| !r.mismatches.is_empty()), "the engine's noise flips some argmax");
    assert_eq!(run.overall, agree_total as f64 / 23.0);
    assert_eq!(run.pass, run.overall >= 0.95);
}

/// An engine that breaks the seam's contract in one way.
struct Broken {
    skip_row: bool,
    extra_row: bool,
}

impl SpanLogits for Broken {
    fn vocab(&self) -> usize {
        VOCAB
    }

    fn span_logits(&mut self, tokens: &[u32], sink: &mut RowSink<'_>) -> Result<(), String> {
        let rows = tokens.len() + self.extra_row as usize;
        for i in 0..rows {
            if self.skip_row && i == 2 {
                continue;
            }
            sink(i, &model(&tokens[..=i.min(tokens.len() - 1)], 0.0, REF_SALT))?;
        }
        Ok(())
    }
}

#[test]
fn an_engine_that_skips_or_adds_rows_is_refused() {
    let world = world("broken");
    let set = ReferenceSet::read(&world.dir).unwrap();
    let skipped = run_set(&mut Broken { skip_row: true, extra_row: false }, &set, None).unwrap_err();
    assert!(skipped.contains("expected row 2"), "{skipped}");
    let extra = run_set(&mut Broken { skip_row: false, extra_row: true }, &set, None).unwrap_err();
    assert!(extra.contains("rows up to"), "{extra}");
}

#[test]
fn a_set_whose_files_disagree_with_its_manifest_is_refused() {
    let world = world("truncated");
    let path = world.dir.join("q_top64_lp.f32");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.truncate(bytes.len() - 4);
    std::fs::write(&path, bytes).unwrap();
    assert!(ReferenceSet::read(&world.dir).unwrap_err().contains("q_top64_lp.f32"));
}

/// The quantized model's own full-vocabulary KL from the reference model,
/// pooled per kind: what the converter records for these windows.
fn quantized_figures(windows: &[(&str, &str, Vec<u32>, usize)]) -> serde_json::Value {
    let mut sums: BTreeMap<&str, (f64, f64, f64)> = BTreeMap::new();
    for (kind, _, ids, valid) in windows {
        for p in 0..valid - 1 {
            let lr = log_softmax(&model(&ids[..=p], 0.0, REF_SALT));
            let lq = log_softmax(&model(&ids[..=p], 0.6, Q_SALT));
            let entry = sums.entry(*kind).or_default();
            entry.0 += 1.0;
            entry.1 += lr.iter().zip(&lq).map(|(r, q)| r.exp() * (r - q)).sum::<f64>();
            entry.2 += (argmax(&lr) == argmax(&lq)) as u64 as f64;
        }
    }
    sums.iter()
        .map(|(k, (n, kl, top1))| (k.to_string(), serde_json::json!({"kld": kl / n, "kld_top64": kl / n, "top1": top1 / n})))
        .collect::<serde_json::Map<_, _>>()
        .into()
}

#[test]
fn the_quantized_model_through_the_seam_meets_its_own_figures_and_a_noisy_engine_fails() {
    let root = temp_dir("acceptance");
    let test = test_windows();
    let long = vec![("code", "long_windows.json[3]", ids(5, 40), 40), ("prose", "long_windows.json[5]", ids(6, 40), 40)];
    write_set_at(&root.join("test2048"), &test);
    write_set_at(&root.join("long8192"), &long);
    g1_fixture().write(&root.join("g1_flash_next.json")).unwrap();
    // Gold letters the quantized model picks, so its proxy is 100%.
    let tokens = &test[3].2;
    let marks: Vec<(usize, u8, u8)> = MARKS
        .iter()
        .map(|&(p, _, o)| {
            let lq = log_softmax(&model(&tokens[..=p], 0.6, Q_SALT));
            let gold = (0..o as usize).fold(0usize, |b, k| if lq[LETTERS[k] as usize] > lq[LETTERS[b] as usize] { k } else { b });
            (p, gold as u8, o)
        })
        .collect();
    let chunks = write_chunks_at(&root, tokens, test[3].3, &marks);
    let record: ConverterRecord = serde_json::from_value(serde_json::json!({
        "status": "complete",
        "kld": {"quantized": quantized_figures(&test), "fp8_only": {}},
        "kld_long8192": {"q": quantized_figures(&long), "f8": {}, "head_fp8_only": {}},
        "mmlu": {"n": 3, "bf16": null, "quantized": null, "mcnemar": {}},
    }))
    .unwrap();

    let mut quantized = MockEngine { noise: 0.6, salt: Q_SALT, rows_per_chunk: 7, calls: 0 };
    let acceptance = run_acceptance(&mut quantized, &root, &record, &chunks).unwrap();
    for v in acceptance.kld_2048.iter().chain(&acceptance.kld_8192) {
        assert!((v.engine - v.quantized).abs() <= 5e-6, "{}: {} against its own {}", v.domain, v.engine, v.quantized);
        assert!(v.pass);
    }
    assert_eq!(acceptance.kld_2048.iter().map(|v| v.domain.as_str()).collect::<Vec<_>>(), ["code", "mmlu", "prose"]);
    assert_eq!(acceptance.kld_8192.iter().map(|v| v.domain.as_str()).collect::<Vec<_>>(), ["code", "prose"]);
    assert_eq!(acceptance.g1.overall, 1.0);
    assert_eq!((acceptance.mmlu.vs_quantized.lost, acceptance.mmlu.vs_quantized.gained), (0, 0));
    assert_eq!(acceptance.mmlu.accuracy, 1.0);
    assert!(acceptance.pass(), "{}", acceptance.render());

    let mut noisy = MockEngine { noise: 4.0, salt: ENGINE_SALT, rows_per_chunk: 7, calls: 0 };
    let acceptance = run_acceptance(&mut noisy, &root, &record, &chunks).unwrap();
    assert!(!acceptance.pass());
    assert!(acceptance.kld_2048.iter().all(|v| !v.pass), "{}", acceptance.render());
    let text = acceptance.render();
    let mismatches: usize = acceptance.g1.results.iter().map(|r| r.mismatches.len()).sum();
    assert!(mismatches > 0);
    assert_eq!(text.matches("    position ").count(), mismatches, "every G1 mismatch is listed:\n{text}");
    assert!(text.contains("FAIL"));
    let _ = std::fs::remove_dir_all(&root);
}

/// Acceptance 8 at an artifact whose own quantized reference misses the
/// floor and which the owner accepted so (2026-10-06): the engine is held to
/// that reference, and only at the artifact the decision names.
#[test]
fn an_accepted_floor_miss_holds_the_engine_to_the_quantized_reference() {
    let report = |quantized_correct: usize, lost: usize, gained: usize| {
        let vs_quantized = mmlu::Paired { lost, gained, p: mmlu::mcnemar_p(lost, gained) };
        mmlu::MmluReport {
            n: 281,
            accuracy: (quantized_correct + gained - lost) as f64 / 281.0,
            bf16_accuracy: 207.0 / 281.0,
            quantized_accuracy: quantized_correct as f64 / 281.0,
            vs_bf16: mmlu::Paired { lost: 17, gained: 7, p: mmlu::mcnemar_p(17, 7) },
            vs_quantized,
            pass: (quantized_correct + gained - lost) as f64 / 281.0 >= mmlu::MMLU_FLOOR,
            significantly_below_bf16: false,
            significantly_below_quantized: lost > gained && vs_quantized.p <= 0.05,
        }
    };
    let accepted = Some(mmlu::AcceptedFloorMiss { quantized_correct: 197 });

    // This artifact (197 of 281 = 70.11%), the engine even with it.
    let even = report(197, 5, 5);
    assert_eq!(even.quantized_correct(), 197);
    assert!(!even.verdict(None), "without the decision the floor stands");
    assert!(even.verdict(accepted));
    // The engine losing answers the reference has: an engine regression.
    let below = report(197, 20, 3);
    assert!(below.significantly_below_quantized);
    assert!(!below.verdict(accepted));
    // Another artifact below the floor is not the one the owner accepted.
    assert!(!report(190, 2, 2).verdict(accepted));
    // An artifact at the floor is held to it.
    let at_floor = report(205, 10, 0);
    assert!(!at_floor.floor_miss_accepted(accepted));
    assert_eq!(at_floor.verdict(accepted), at_floor.pass);
}
