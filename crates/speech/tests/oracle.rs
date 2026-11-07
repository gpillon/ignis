//! The `--ignored` CPU legs on the catalog files (spec speech/01 §5, testing decisions): they
//! read the study disk (`IGNIS_SPEECH_STUDY`, default `H:/speech-study`) and fail without it.
//! Run them alone on the machine, one at a time:
//!
//! ```text
//! cargo test --release -p ignis-speech --test oracle -- --ignored --test-threads=1
//! ```

mod common;

use std::collections::HashMap;

use common::*;
use ignis_speech::cpu::{select, Features, Isa};
use ignis_speech::decoder::{self, Cache, Logits};
use ignis_speech::encoder::encode_window;
use ignis_speech::gguf::{Gguf, TensorType};
use ignis_speech::matmul::{matmul_tiled, quantize, Tiles, Weight};
use ignis_speech::mel::{audio_tokens, frames_of, normalise, Mel, N_MELS};
use ignis_speech::mem::{AlignedBuf, Blob};
use ignis_speech::model::{check_family, Model};
use ignis_speech::pool::Pool;
use ignis_speech::quant::{dequant_act_f64, dequant_row_f64, WType};
use ignis_speech::text::Prompt;
use ignis_speech::{Executor, Settings, Threads};
use serde_json::Value;

fn avx2() -> Isa {
    select(Features::detect()).expect("the oracle legs run the served kernels: this CPU lacks AVX2")
}

fn threads() -> usize {
    Threads::Auto.resolve().0
}

/// The whole clip's normalised log-mel, rows of 128, with the clip's own maximum (the batch path).
fn clip_mel(pool: &Pool, isa: Isa, audio: &[f32]) -> (Vec<f32>, usize) {
    let n = frames_of(audio.len());
    let mut x = audio.to_vec();
    x.resize(x.len().max(8000), 0.0);
    let mut log = Vec::new();
    Mel::new().log_frames(pool, isa, &x, 0, n, true, &mut log);
    let max = log.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    (log.iter().map(|&v| normalise(v, max)).collect(), n)
}

type Taps = HashMap<String, Vec<f32>>;

/// The clip's projected audio, window by window, with every tap appended in order.
fn encode_clip(m: &Model, pool: &Pool, isa: Isa, mel: &[f32], frames: usize, taps: &mut Taps) -> Vec<f32> {
    let mut out = Vec::new();
    for w in 0..frames.div_ceil(800) {
        let n = (frames - w * 800).min(800);
        let mut tap = |name: &str, rows: &[f32]| taps.entry(name.to_string()).or_default().extend_from_slice(rows);
        out.extend(encode_window(&m.encoder, pool, isa, &mel[w * 800 * N_MELS..], n, Some(&mut tap)));
    }
    out
}

/// One teacher-forced pass over `ids`, the projected audio at the audio placeholders.
fn teacher_forced(m: &Model, pool: &Pool, isa: Isa, ids: &[u32], audio: &[f32], want: Logits, taps: &mut Taps) -> Vec<f32> {
    let dec = &m.decoder;
    let h = dec.hidden;
    let mut x = Vec::with_capacity(ids.len() * h);
    let mut a = 0;
    for &id in ids {
        if id == m.text.audio_pad {
            x.extend_from_slice(&audio[a * h..(a + 1) * h]);
            a += 1;
        } else {
            x.extend(decoder::embed(dec, &[id]));
        }
    }
    assert_eq!(a * h, audio.len(), "audio placeholders and audio tokens differ");
    let mut cache = Cache::new(dec, ids.len());
    let mut tap = |name: &str, rows: &[f32]| taps.entry(name.to_string()).or_default().extend_from_slice(rows);
    decoder::forward(dec, pool, isa, &mut cache, x, want, Some(&mut tap))
}

fn rows(t: &[f32], width: usize, idx: &[usize]) -> Vec<f32> {
    idx.iter().flat_map(|&r| t[r * width..(r + 1) * width].iter().copied()).collect()
}

fn idx(v: &Value) -> Vec<usize> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect()
}

/// Every tapped tensor's error against the fixture, and the names over their bound.
fn compare(fx: &Layers, taps: &Taps, bounds: &Value) -> (Vec<(String, f64, f64)>, Vec<String>) {
    let (enc_rows, dec_rows) = (idx(&fx.meta["enc_rows"]), idx(&fx.meta["dec_rows"]));
    let mut report = Vec::new();
    let mut over = Vec::new();
    let mut names: Vec<&String> = fx.tensors.keys().filter(|k| *k != "mel").collect();
    names.sort();
    for name in names {
        let (shape, reference) = &fx.tensors[name];
        let width = shape[1];
        let ours = taps.get(name).unwrap_or_else(|| panic!("no tap {name}"));
        let sel = if name.starts_with("enc.") || name == "proj" { &enc_rows } else { &dec_rows };
        let e = row_rel_l2(&rows(ours, width, sel), reference, width);
        let b = bounds[name].as_f64().unwrap_or_else(|| panic!("no bound for {name}"));
        if e > b {
            over.push(name.clone());
        }
        report.push((name.clone(), e, b));
    }
    (report, over)
}

/// Acceptance 12: per catalog variant, the mel, every encoder layer, the projector and every
/// decoder layer within `thresholds.json` of Transformers FP32, teacher-forced on the same audio
/// tokens; a perturbed layer fails.
#[test]
#[ignore]
fn every_layer_is_within_its_bound_of_transformers() {
    let isa = avx2();
    let pool = Pool::new(threads());
    let audio = read_wav(&fixtures().join("it_it-1703.wav"));
    let th = json(&fixtures().join("thresholds.json"));
    for v in &VARIANTS {
        let f = v.files();
        let mut m = Model::load(&f.decoder, &f.encoder, false).unwrap();
        let fx = layers(v.size);
        let bounds = &th["variants"][v.id]["bound"];
        let (mel, frames) = clip_mel(&pool, isa, &audio);
        let mut taps = Taps::new();
        let proj = encode_clip(&m, &pool, isa, &mel, frames, &mut taps);
        let ids = ids(&fx.meta["input_ids"]);
        let logits = teacher_forced(&m, &pool, isa, &ids, &proj, Logits::All, &mut taps);
        let (report, over) = compare(&fx, &taps, bounds);
        for (n, e, b) in &report {
            println!("{} {n}: {e:.3e} (bound {b:.3e})", v.id);
        }
        assert!(over.is_empty(), "{}: over their bound: {over:?}", v.id);
        // Teacher-forced top-1 on the fixture clip's transcript.
        let pl = fx.meta["prompt_len"].as_u64().unwrap() as usize;
        let top1 = ids_of_argmax(&logits, m.decoder.vocab);
        let agree = (pl - 1..ids.len() - 1).filter(|&p| top1[p] == ids[p + 1]).count() as f64 / (ids.len() - pl) as f64;
        println!("{}: top-1 on the fixture clip {agree:.3}", v.id);

        // Negative controls: encoder layer 3 without its attention output bias, and decoder
        // layers 10 and 11 with their down projections swapped (an off-by-one load), each fail
        // at that layer. (Swapping a layer's Q and K norms stayed inside the decoder bounds:
        // Q4_K's own error is 12-17% per row of the residual, which the bounds must admit.)
        m.encoder.layers[3].o.b = None;
        let (a, b) = m.decoder.layers.split_at_mut(11);
        std::mem::swap(&mut a[10].down, &mut b[0].down);
        let mut bad = Taps::new();
        let proj = encode_clip(&m, &pool, isa, &mel, frames, &mut bad);
        teacher_forced(&m, &pool, isa, &ids, &proj, Logits::Last, &mut bad);
        bad.insert("dec.norm".into(), taps["dec.norm"].clone());
        let (_, over) = compare(&fx, &bad, bounds);
        assert!(over.contains(&"enc.layer.3".to_string()), "{}: a dropped bias passed: {over:?}", v.id);
        assert!(!over.contains(&"enc.layer.2".to_string()), "{}: the layer before the perturbation failed", v.id);
        assert!(over.iter().any(|n| n == "dec.layer.10"), "{}: swapped down projections passed: {over:?}", v.id);
    }
}

fn ids_of_argmax(logits: &[f32], vocab: usize) -> Vec<u32> {
    logits.chunks(vocab).map(decoder::argmax).collect()
}

/// Acceptance 13: teacher-forced top-1 agreement with Transformers at least 95% per variant, over
/// the finding's 40 FLEURS clips.
#[test]
#[ignore]
fn teacher_forced_top1_agrees_with_transformers() {
    let isa = avx2();
    let pool = Pool::new(threads());
    let clips = fleurs_clips();
    for v in &VARIANTS {
        let f = v.files();
        let m = Model::load(&f.decoder, &f.encoder, false).unwrap();
        let reference = json(&fixtures().join(v.size).join("fleurs.json"));
        let prompt = Prompt::new(&m.text, "", 512, None);
        let (mut agree, mut total) = (0usize, 0usize);
        for c in reference["clips"].as_array().unwrap() {
            let id = c["id"].as_str().unwrap();
            let audio = read_wav(&clips[id].0);
            let (mel, frames) = clip_mel(&pool, isa, &audio);
            let proj = encode_clip(&m, &pool, isa, &mel, frames, &mut Taps::new());
            let out = ids(&c["output"]);
            let n_audio = audio_tokens(frames);
            let mut seq = prompt.head.clone();
            seq.extend(std::iter::repeat_n(m.text.audio_pad, n_audio));
            seq.extend(&prompt.tail);
            seq.extend(&out);
            let logits = teacher_forced(&m, &pool, isa, &seq, &proj, Logits::Tail(out.len() + 1), &mut Taps::new());
            let top1 = ids_of_argmax(&logits, m.decoder.vocab);
            agree += out.iter().zip(&top1).filter(|(a, b)| a == b).count();
            total += out.len();
        }
        let rate = agree as f64 / total as f64;
        println!("{}: teacher-forced top-1 {agree}/{total} = {rate:.4}", v.id);
        assert!(rate >= 0.95, "{}: top-1 agreement {rate:.4} under the G1 floor's 0.95", v.id);
    }
}

/// The finding's 40 FLEURS clips on the study disk: id → (path, reference).
fn fleurs_clips() -> HashMap<String, (std::path::PathBuf, String)> {
    let s = study();
    let text = std::fs::read_to_string(s.join("data").join("clips").join("manifest.csv")).unwrap();
    let mut out = HashMap::new();
    for line in text.lines().skip(1) {
        // clip_id,lang,wav,duration_s,reference (the reference may hold commas, unquoted or quoted).
        let mut parts = line.splitn(5, ',');
        let (id, lang, wav, _, r) = (parts.next().unwrap(), parts.next().unwrap(), parts.next().unwrap(), parts.next(), parts.next().unwrap_or(""));
        if lang != "it_it" && lang != "en_us" {
            continue;
        }
        let r = r.trim().trim_matches('"').replace("\"\"", "\"");
        out.insert(id.to_string(), (s.join(wav.replace('\\', "/")), r));
    }
    out
}

fn dictations() -> Vec<(String, std::path::PathBuf, String)> {
    let d = study().join("dictations");
    (1..=11)
        .map(|i| {
            let id = format!("d{i:02}");
            let r = std::fs::read_to_string(d.join(format!("{id}.txt"))).unwrap();
            (id.clone(), d.join("16k").join(format!("{id}.wav")), r)
        })
        .collect()
}

/// Acceptance 11: on every FLEURS clip and every dictation the audio-token count is the
/// official processor's (d01: 135).
#[test]
#[ignore]
fn audio_token_counts_are_the_official_processors() {
    let clips = fleurs_clips();
    for size in ["1.7b", "0.6b"] {
        for c in json(&fixtures().join(size).join("fleurs.json"))["clips"].as_array().unwrap() {
            let audio = read_wav(&clips[c["id"].as_str().unwrap()].0);
            assert_eq!(audio_tokens(frames_of(audio.len())) as u64, c["n_audio_tokens"].as_u64().unwrap(), "{}", c["id"]);
        }
        let private = json(&study().join("results").join("oracle").join(size).join("dictations.json"));
        let by_id: HashMap<&str, u64> =
            private["dictations"].as_array().unwrap().iter().map(|d| (d["id"].as_str().unwrap(), d["n_audio_tokens"].as_u64().unwrap())).collect();
        for (id, path, _) in dictations() {
            let n = audio_tokens(frames_of(read_wav(&path).len())) as u64;
            assert_eq!(n, by_id[id.as_str()], "{id}");
            if id == "d01" {
                assert_eq!(n, 135);
            }
        }
    }
}

/// Acceptance 17: the context is the system turn and its text tokens are the official
/// processor's (15 with none, 420 with the README context); the same context twice prefills no
/// context token the second time, and a new context replaces the cache's one entry.
#[test]
#[ignore]
fn the_context_tokens_and_the_prefix_cache() {
    let readme = std::fs::read_to_string(study().join("dictations").join("context_readme.txt")).unwrap();
    for v in &VARIANTS {
        let f = v.files();
        let m = Model::load(&f.decoder, &f.encoder, false).unwrap();
        let none = Prompt::new(&m.text, "", 512, None);
        let official = ids(&json(&fixtures().join(v.size).join("fleurs.json"))["no_context_text_tokens"]);
        assert_eq!([&none.head[..], &none.tail[..]].concat(), official, "{}: no context", v.id);
        assert_eq!(none.text_tokens(), 15);
        let ctx = Prompt::new(&m.text, &readme, 512, None);
        let private = json(&study().join("results").join("oracle").join(v.size).join("dictations.json"));
        assert_eq!([&ctx.head[..], &ctx.tail[..]].concat(), ids(&private["readme_context_text_tokens"]), "{}: README context", v.id);
        assert_eq!(ctx.text_tokens(), 420);
        // Over the cap, the last tokens are kept.
        let capped = Prompt::new(&m.text, &readme, 100, None);
        assert_eq!(capped.context_tokens, 100);
        assert_eq!(capped.head[capped.head.len() - 6 - 100..capped.head.len() - 6], ctx.head[ctx.head.len() - 6 - 100..ctx.head.len() - 6]);
        drop(m);

        let exec = Executor::new(Settings { threads: Threads::Count(threads()), ..Settings::default() }, f, Features::detect());
        let prefilled = |c: &str| exec.open(c, None).unwrap().stats().context_prefilled;
        assert_eq!(prefilled(&readme), ctx.context_tokens);
        assert_eq!(prefilled(&readme), 0, "the same context prefilled again");
        assert!(prefilled("Vocabulary: ignis, Qwen3-ASR, GGUF.") > 0);
        assert_eq!(prefilled(&readme), ctx.context_tokens, "the new context did not replace the entry");
    }
}

fn transcribe(exec: &Executor, audio: &[f32], realtime: bool) -> (ignis_speech::Transcript, Vec<String>, ignis_speech::Stats) {
    let mut s = exec.open("", None).unwrap();
    let mut stables = Vec::new();
    if realtime {
        for chunk in audio.chunks(1600) {
            s.push(chunk).unwrap();
            if s.partial_due() {
                stables.push(s.partial().unwrap().stable);
            }
        }
    } else {
        // The batch path as the server drives it: 8 s at a time (spec 02's worker).
        for chunk in audio.chunks(128_000) {
            s.push(chunk).unwrap();
        }
    }
    let stats = s.stats().clone();
    (s.commit().unwrap(), stables, stats)
}

struct Pools {
    fleurs: Vec<(Vec<String>, Vec<String>)>,
    dictations: Vec<(Vec<String>, Vec<String>)>,
}

fn run_wer(v: &Variant, realtime: bool) -> Pools {
    run_wer_with(v, realtime, Settings::default().rollback_tokens)
}

fn run_wer_with(v: &Variant, realtime: bool, rollback_tokens: usize) -> Pools {
    let settings = Settings { threads: Threads::Count(threads()), rollback_tokens, ..Settings::default() };
    let exec = Executor::new(settings, v.files(), Features::detect());
    let clips = fleurs_clips();
    let mut ids: Vec<&String> = clips.keys().collect();
    ids.sort();
    let fleurs = ids
        .iter()
        .map(|id| {
            let (path, r) = &clips[*id];
            let (t, _, _) = transcribe(&exec, &read_wav(path), realtime);
            (norm_fleurs(r), norm_fleurs(&t.text))
        })
        .collect();
    let dictations = dictations()
        .iter()
        .map(|(_, path, r)| {
            let (t, _, _) = transcribe(&exec, &read_wav(path), realtime);
            (norm_dictation(r), norm_dictation(&t.text))
        })
        .collect();
    Pools { fleurs, dictations }
}

/// Transformers FP32's pooled WER for a size, from the recorded transcripts.
fn reference_wer(size: &str) -> (f64, f64) {
    let clips = fleurs_clips();
    let f: Vec<_> = json(&fixtures().join(size).join("fleurs.json"))["clips"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| (norm_fleurs(&clips[c["id"].as_str().unwrap()].1), norm_fleurs(c["transcript"].as_str().unwrap())))
        .collect();
    let private = json(&study().join("results").join("oracle").join(size).join("dictations.json"));
    let by_id: HashMap<&str, &str> =
        private["dictations"].as_array().unwrap().iter().map(|d| (d["id"].as_str().unwrap(), d["transcript"].as_str().unwrap())).collect();
    let d: Vec<_> = dictations().iter().map(|(id, _, r)| (norm_dictation(r), norm_dictation(by_id[id.as_str()]))).collect();
    (pooled_wer(&f), pooled_wer(&d))
}

/// llama.cpp b11840's pooled WER on the same clips, from the finding's runs on the i9
/// (`results/llamacpp_win.csv`, 10 threads, first run; the dictations' arm A, the official
/// empty system turn, in `results/dictation_runs.csv`).
fn llama_cpp_wer(v: &Variant) -> (f64, f64) {
    let r = study().join("results");
    let (model, quant) = match v.id {
        "qwen3-asr-1.7b" => ("1.7B", "Q4_K_M"),
        "qwen3-asr-1.7b-q8" => ("1.7B", "Q8_0"),
        _ => ("0.6B", "Q8_0"),
    };
    let clips = fleurs_clips();
    let f: Vec<_> = csv_rows(&r.join("llamacpp_win.csv"))
        .into_iter()
        .filter(|x| x["model"] == model && x["quant"] == quant && x["mmproj_quant"] == "Q8_0" && x["threads"] == "10" && x["run"] == "1")
        .filter(|x| clips.contains_key(&x["clip_id"]))
        .map(|x| (norm_fleurs(&clips[&x["clip_id"]].1), norm_fleurs(&x["hypothesis"])))
        .collect();
    assert_eq!(f.len(), 40, "{}: llama.cpp FLEURS runs", v.id);
    let refs: HashMap<String, String> = dictations().into_iter().map(|(id, _, r)| (id, r)).collect();
    let variant = format!("{model}-{quant}");
    let d: Vec<_> = csv_rows(&r.join("dictation_runs.csv"))
        .into_iter()
        .filter(|x| x["plan"] == "matrix" && x["arm"] == "A" && x["forced_language"].is_empty() && x["variant"] == variant)
        .map(|x| (norm_dictation(&refs[&x["clip"]]), norm_dictation(&x["hypothesis"])))
        .collect();
    assert_eq!(d.len(), 11, "{}: llama.cpp dictation runs", v.id);
    (pooled_wer(&f), pooled_wer(&d))
}

/// Acceptance 14: through the batch path, pooled WER on FLEURS 40 and on the 11 dictations not
/// worse than Transformers FP32's by more than 1 pt, per variant. No transcript is compared.
/// Acceptance 24's WER half: and within 1 pt of llama.cpp b11840's on the same clips.
#[test]
#[ignore]
fn batch_wer_is_within_a_point_of_transformers_and_llama_cpp() {
    for v in &VARIANTS {
        let (rf, rd) = reference_wer(v.size);
        let (lf, ld) = llama_cpp_wer(v);
        let p = run_wer(v, false);
        let (f, d) = (pooled_wer(&p.fleurs), pooled_wer(&p.dictations));
        println!(
            "{} batch: FLEURS {:.2}% (Transformers {:.2}%, llama.cpp {:.2}%), dictations {:.2}% (Transformers {:.2}%, llama.cpp {:.2}%)",
            v.id, 100.0 * f, 100.0 * rf, 100.0 * lf, 100.0 * d, 100.0 * rd, 100.0 * ld
        );
        assert!(f <= rf + 0.01 && d <= rd + 0.01, "{}: batch WER over Transformers' + 1 pt", v.id);
        assert!((f - lf).abs() <= 0.01 && (d - ld).abs() <= 0.01, "{}: batch WER more than 1 pt from llama.cpp's", v.id);
    }
}

/// Acceptance 15: through the incremental path at real-time pace (100 ms pushes, a partial every
/// `partial_interval_ms`), pooled WER not worse than the batch path's by more than 1 pt at the
/// default `rollback_tokens` (10, Q75), for the default variant and the 0.6B. The 1.7B Q8 is an
/// explicit choice documented at +1.29 pt on the dictations (Q75): it is held to 1.5 pt, so a
/// regression past the measurement still fails.
#[test]
#[ignore]
fn incremental_wer_is_within_a_point_of_batch() {
    assert_eq!(Settings::default().rollback_tokens, 10, "AC 15 is stated at the default tail of 10 (Q75)");
    let mut over = Vec::new();
    for v in &VARIANTS {
        let b = run_wer(v, false);
        let (bf, bd) = (pooled_wer(&b.fleurs), pooled_wer(&b.dictations));
        let r = run_wer(v, true);
        let (rf, rd) = (pooled_wer(&r.fleurs), pooled_wer(&r.dictations));
        println!(
            "{} incremental: FLEURS {:.2}% (batch {:.2}%), dictations {:.2}% (batch {:.2}%)",
            v.id, 100.0 * rf, 100.0 * bf, 100.0 * rd, 100.0 * bd
        );
        let allowed = if v.id == "qwen3-asr-1.7b-q8" { 0.015 } else { 0.01 };
        if rf > bf + allowed || rd > bd + allowed {
            over.push(v.id);
        }
    }
    assert!(over.is_empty(), "incremental WER over the batch path's + 1 pt (1.5 for 1.7B Q8): {over:?}");
}

/// Acceptance 18: fed each dictation at real-time pace, a session encodes each window once,
/// prefills the audio prefix once, reports a partial every `partial_interval_ms`, and its
/// completed transcript begins with every stable text it reported.
#[test]
#[ignore]
fn a_realtime_session_encodes_once_and_keeps_its_stable_text() {
    let v = &VARIANTS[0];
    let exec = Executor::new(Settings { threads: Threads::Count(threads()), ..Settings::default() }, v.files(), Features::detect());
    for (id, path, _) in dictations() {
        let audio = read_wav(&path);
        let (t, stables, stats) = transcribe(&exec, &audio, true);
        let frames = frames_of(audio.len());
        // Every window completed while audio arrived was encoded into the prefix once; commit
        // encodes the last one.
        assert_eq!(stats.windows, (audio.len().saturating_sub(200) / 160 + 1) / 800, "{id}: windows encoded while pushing");
        assert_eq!(stats.audio_prefilled, stats.windows * 104, "{id}: audio prefilled while pushing");
        assert_eq!(t.audio_tokens, audio_tokens(frames));
        let due = audio.len() / (2 * 16_000);
        assert_eq!(stats.partials, due, "{id}: a partial every 2 s of audio");
        let mut last = String::new();
        for s in &stables {
            assert!(s.starts_with(&last), "{id}: a stable text was revised");
            assert!(t.text.starts_with(s.as_str()), "{id}: the transcript does not begin with the stable text {s:?}");
            last = s.clone();
        }
        println!("{id}: {} partials, {} windows, stable {} chars of {}", stats.partials, stats.windows, last.len(), t.text.len());
    }
}

/// Acceptance 2 and 22's check: our reader reads every catalog file's header, metadata and
/// tensor table and maps its tensor data; our Q8_0 files' tensor data is ggml-org's, byte for
/// byte, except the converter-computed sinusoid table, whose difference is reported.
#[test]
#[ignore]
fn our_reader_reads_every_catalog_file_and_our_q8_0_is_ggml_orgs() {
    let s = study();
    let conv = s.join("models").join("ignis-conv");
    for v in &VARIANTS {
        let f = v.files();
        let (dec, enc) = (Gguf::open(&f.decoder).unwrap(), Gguf::open(&f.encoder).unwrap());
        check_family(&dec, &enc).unwrap();
        for g in [&dec, &enc] {
            let mut sum = 0u64;
            for t in g.tensors() {
                let d = g.data(t);
                assert_eq!(d.len() as u64, t.size);
                sum = sum.wrapping_add(d[0] as u64 + d[d.len() - 1] as u64);
            }
            println!("{}: v{}, {} metadata, {} tensors, {} MB of data", g.path().display(), g.version, g.metadata().len(), g.tensors().len(), g.data_section().len() >> 20);
            assert!(sum > 0);
        }
    }
    for size in ["1.7B", "0.6B"] {
        for name in [format!("Qwen3-ASR-{size}-Q8_0.gguf"), format!("mmproj-Qwen3-ASR-{size}-Q8_0.gguf")] {
            let ours = Gguf::open(&conv.join(&name)).unwrap();
            let theirs = Gguf::open(&s.join("models").join(format!("qwen3-asr-{}-gguf", size.to_lowercase())).join(&name)).unwrap();
            assert_eq!(ours.tensors().len(), theirs.tensors().len(), "{name}");
            for t in ours.tensors() {
                let o = theirs.tensor(&t.name).unwrap_or_else(|| panic!("{name}: ggml-org has no {}", t.name));
                assert_eq!((t.ty, &t.dims), (o.ty, &o.dims), "{name}: {}", t.name);
                if ours.data(t) == theirs.data(o) {
                    continue;
                }
                // Only the sinusoid table may differ: the converter computes it in f32 (it is not in
                // the checkpoint), and at the table's late positions an angle near 1,500 rad has
                // an f32 ulp of 1.2e-4. The 13 positions a chunk uses must agree.
                assert_eq!(t.name, "a.position_embd.weight", "{name}: {} differs from ggml-org's", t.name);
                let f = |b: &[u8]| b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>();
                let (a, b) = (f(ours.data(t)), f(theirs.data(o)));
                let diff = |r: std::ops::Range<usize>| a[r.clone()].iter().zip(&b[r]).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
                let d = t.dims[0] as usize;
                let (used, all) = (diff(0..13 * d), diff(0..a.len()));
                println!("{name}: {}: the 13 used positions differ by {used:.3e}, the whole table by {all:.3e}", t.name);
                assert!(used <= 1e-6, "{name}: the positions a chunk uses differ by {used}");
            }
        }
    }
}

/// Acceptance 4's `--ignored` leg: every kernel and tile shape within 1e-5 of the fp64 reference
/// on the catalog files' own tensors (their first 64 rows), M = 1, 7, 104 and 512.
/// The tile shapes are `avx2`'s, a module only an x86_64 build has (GitHub #496).
#[cfg(target_arch = "x86_64")]
#[test]
#[ignore]
fn kernels_match_fp64_on_the_catalog_tensors() {
    let isa = avx2();
    let pool = Pool::new(threads());
    let conv = study().join("models").join("ignis-conv");
    let picks = [
        ("Qwen3-ASR-1.7B-Q4_K_M.gguf", "blk.0.ffn_gate.weight"),
        ("Qwen3-ASR-1.7B-Q4_K_M.gguf", "blk.0.ffn_down.weight"),
        ("Qwen3-ASR-1.7B-Q4_K_M.gguf", "output.weight"),
        ("Qwen3-ASR-1.7B-Q8_0.gguf", "blk.0.attn_q.weight"),
        ("Qwen3-ASR-0.6B-Q8_0.gguf", "blk.27.ffn_down.weight"),
        ("mmproj-Qwen3-ASR-1.7B-Q8_0.gguf", "a.blk.0.ffn_up.weight"),
        ("mmproj-Qwen3-ASR-0.6B-Q8_0.gguf", "mm.a.mlp.2.weight"),
        ("mmproj-Qwen3-ASR-1.7B-Q8_0.gguf", "a.conv2d.2.weight"),
        ("mmproj-Qwen3-ASR-1.7B-Q8_0.gguf", "a.conv_out.weight"),
    ];
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        ((rng >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
    };
    for (file, tensor) in picks {
        let g = Gguf::open(&conv.join(file)).unwrap();
        let t = g.tensor(tensor).unwrap_or_else(|| panic!("{file}: no {tensor}")).clone();
        let n_all = *t.dims.last().unwrap() as usize;
        let k = (t.elements() / n_all as u64) as usize;
        let n = 64.min(n_all);
        let ty = match t.ty {
            TensorType::Q4K => WType::Q4K,
            TensorType::Q6K => WType::Q6K,
            TensorType::Q8_0 => WType::Q80,
            TensorType::F16 => WType::F16,
            other => panic!("{tensor}: {other:?}"),
        };
        let bytes = &g.data(&t)[..ty.row_bytes(k) * n];
        let mut buf = AlignedBuf::zeroed(bytes.len());
        buf.as_mut_slice().copy_from_slice(bytes);
        let mut weights = vec![Weight::new(tensor, ty, n, k, Blob::owned(buf))];
        if ty == WType::Q4K {
            weights.push(Weight::repack_q4k(tensor, n, k, bytes));
        }
        for w in &weights {
            let mut wd = vec![0f64; n * k];
            for c in 0..n {
                dequant_row_f64(w.ty, w.bytes(), c, k, &mut wd[c * k..(c + 1) * k]);
            }
            for m in [1usize, 7, 104, 512] {
                let x: Vec<f32> = (0..m * k).map(|_| next()).collect();
                let at = w.ty.act();
                let rb = at.row_bytes(k);
                let mut act = AlignedBuf::zeroed(m * rb);
                if at == ignis_speech::quant::AType::F32 {
                    act.as_mut_slice().copy_from_slice(&x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
                } else {
                    quantize(&pool, Isa::scalar(), at, &x, m, k, act.as_mut_slice());
                }
                let mut ad = vec![0f64; k];
                let mut reference = vec![(0f64, 0f64); m * n];
                for r in 0..m {
                    dequant_act_f64(at, act.as_slice(), r, k, &mut ad);
                    for c in 0..n {
                        let wr = &wd[c * k..(c + 1) * k];
                        reference[r * n + c] = (wr.iter().zip(&ad).map(|(a, b)| a * b).sum(), wr.iter().zip(&ad).map(|(a, b)| (a * b).abs()).sum());
                    }
                }
                let shapes: &[(usize, usize)] = if w.ty == WType::Q4Kx8 { &ignis_speech::avx2::TILES_X8 } else { &ignis_speech::avx2::TILES };
                let d = Tiles::default_for(w.ty);
                for &s in shapes.iter().filter(|s| m > 1 || s.0 == 1) {
                    let tiles = if m == 1 { Tiles { gemv: s, gemm: d.gemm } } else { Tiles { gemv: d.gemv, gemm: s } };
                    let mut out = vec![f32::NAN; m * n];
                    matmul_tiled(&pool, isa, w, &x, m, &mut out, tiles);
                    let worst = out.iter().zip(&reference).map(|(&y, &(r, s))| (y as f64 - r).abs() / s.max(1e-300)).fold(0f64, f64::max);
                    assert!(worst <= 1e-5, "{tensor} {:?} m={m} {tiles:?}: {worst:.3e}", w.ty);
                }
            }
            println!("{file} {tensor} {:?}: every tile within 1e-5", w.ty);
        }
    }
}
