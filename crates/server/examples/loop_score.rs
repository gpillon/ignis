//! Score texts with the loop guard's own detector (`loop_guard.rs`), for the
//! parity check against the offline Python numbers (CPU only).
//!
//! ```text
//! cargo run -p ignis-server --example loop_score -- IN_JSONL OUT_JSONL [WINDOW [normalize|raw]]
//!     [--repeat-min N] [--repeat-max N] [--repeat-copies N] [--repeat-varying N] [--all]
//! ```
//!
//! `IN_JSONL` holds one `{"id", "text"}` object per line (other keys are
//! carried through). Each text is pushed into [`Normalized`] one character
//! at a time, the [`Watch`] looking after each — the guard on a stream of
//! one-character tokens — and every score it takes is written out. Per text,
//! one line of `OUT_JSONL`:
//!
//! - `online`: the worst score the guard sees, `[ratio, at, start]` (raw
//!   characters pushed when it was taken, raw character its window starts at);
//! - `checks`: every score, `[at, scored length, ratio, window hash, window
//!   start]` (the raw character its window starts at);
//! - `raw_ratios`: every score's raw-gate ratio (the last window of raw
//!   characters), in the order of `checks`;
//! - `fire`: the first score that is a loop with the default raw gate,
//!   `[at, start, ratio, raw ratio]`, and `fire_ungated` the first with no
//!   gate; `cut`: where a rewind at `fire` takes the reasoning back to (floor
//!   0), with `clean`, the clean score it went back to, `[at, start]`;
//! - `offline`: the whole text's scored form, windows every `check_every`
//!   scored characters as the offline research scorer takes them (the
//!   parity of `docs/findings/2026-10-14-reasoning-loop-guard.md`), the worst
//!   as `[ratio, end]`; with `scored_len` and `scored_hash` of that whole
//!   form;
//! - `repeat`: the first repeat the repeat signal finds alone, from floor 0,
//!   `[at, period, copies, first copy's start, second copy's start]`
//!   (`--repeat-*` set its limits, the defaults otherwise);
//! - `guard`: what the guard itself does first, both signals watching
//!   ([`Watch::check`]), a loop whose cut keeps too little held as the guard
//!   holds it: `{signal, at, start, cut, ratio, period, copies}`, `ratio`
//!   null for a repeat and `period` and `copies` for a loop score;
//! - `guard_all`, with `--all`: every action the guard takes in `log` mode,
//!   each one starting the watch again from where it was found, in order.
//!
//! The hash is FNV-1a 64 of the UTF-8 bytes, so another implementation can
//! check it has the same text without the text being written out.

use std::io::{BufRead, BufWriter, Write};

use ignis_server::loop_guard::{compress_ratio, LoopGuardParams, Normalized, Score, Watch};
use serde_json::{json, Value};

fn fire(score: &Score) -> Value {
    json!([score.end, score.start, score.ratio, score.raw_ratio])
}

fn fnv1a(text: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

fn main() {
    let mut args = Vec::new();
    let mut params = LoopGuardParams::default();
    let mut all = false;
    let mut given = std::env::args();
    while let Some(arg) = given.next() {
        if arg == "--all" {
            all = true;
            continue;
        }
        let limit = match arg.as_str() {
            "--repeat-min" => &mut params.repeat_min_chars,
            "--repeat-max" => &mut params.repeat_max_chars,
            "--repeat-copies" => &mut params.repeat_copies,
            "--repeat-varying" => &mut params.repeat_varying,
            _ => {
                args.push(arg);
                continue;
            }
        };
        *limit = given.next().and_then(|value| value.parse().ok()).unwrap_or_else(|| panic!("{arg}: a count"));
    }
    if !(3..=5).contains(&args.len()) {
        eprintln!("usage: loop_score IN_JSONL OUT_JSONL [WINDOW [normalize|raw]] [--repeat-min N] [--repeat-max N] [--repeat-copies N] [--repeat-varying N] [--all]");
        std::process::exit(2);
    }
    if let Some(window) = args.get(3) {
        params.window_chars = window.parse().unwrap_or_else(|e| panic!("window: {e}"));
    }
    if let Some(mode) = args.get(4) {
        params.normalize = match mode.as_str() {
            "normalize" => true,
            "raw" => false,
            other => panic!("`{other}`: normalize or raw"),
        };
    }
    let window = params.window_chars as usize;
    let input = std::io::BufReader::new(std::fs::File::open(&args[1]).unwrap_or_else(|e| panic!("{}: {e}", args[1])));
    let mut out = BufWriter::new(std::fs::File::create(&args[2]).unwrap_or_else(|e| panic!("{}: {e}", args[2])));
    let mut texts = 0;
    for line in input.lines() {
        let line = line.expect("read");
        if line.trim().is_empty() {
            continue;
        }
        let mut record: Value = serde_json::from_str(&line).expect("a JSON object per line");
        let text = record["text"].as_str().expect("a text").to_owned();

        let ungated = LoopGuardParams { raw_gate_permille: 0, ..params.clone() };
        let mut scored = Normalized::new(params.normalize);
        let mut watch = Watch::from(0, &params);
        let mut checks = Vec::new();
        let mut scores: Vec<Score> = Vec::new();
        let mut online: Option<(f64, usize, usize)> = None;
        let (mut first, mut first_ungated, mut cut) = (Value::Null, Value::Null, Value::Null);
        let mut guard = Watch::from(0, &params);
        let (mut actions, mut repeat, mut sentences) = (Vec::new(), Value::Null, 0);
        for c in text.chars() {
            scored.push(c);
            if repeat.is_null() && scored.sentence_count() > sentences {
                sentences = scored.sentence_count();
                if let Some(r) = scored.repeat(0, &params) {
                    repeat = json!([r.end, r.period, r.copies, r.first, r.second]);
                }
            }
            if all || actions.is_empty() {
                if let Some(found) = guard.check(&scored, &params) {
                    let at = guard.cut(scored.raw(), &found, &params);
                    if at < params.min_kept_chars as usize {
                        guard.hold();
                    } else {
                        let (score, r) = (found.score(), found.repeat());
                        actions.push(json!({
                            "signal": found.signal(), "at": found.end(), "start": found.start(), "cut": at,
                            "ratio": score.map(|s| s.ratio), "period": r.map(|r| r.period), "copies": r.map(|r| r.copies),
                        }));
                        guard = Watch::from(scored.raw_len(), &params);
                    }
                }
            }
            if let Some(score) = watch.score(&scored, &params) {
                let len = scored.text().len();
                let hash = fnv1a(&scored.text()[len - window..].iter().collect::<String>());
                checks.push(json!([scored.raw_len(), len, score.ratio, hash, score.start]));
                if online.is_none_or(|(worst, _, _)| score.ratio < worst) {
                    online = Some((score.ratio, scored.raw_len(), score.start));
                }
                if first_ungated.is_null() && score.is_loop(&ungated) {
                    first_ungated = fire(&score);
                }
                if first.is_null() && score.is_loop(&params) {
                    first = fire(&score);
                    let clean = scores.iter().rev().find(|s| s.is_clean(&params));
                    cut = json!({
                        "at": watch.cut_point(scored.raw(), &params),
                        "clean": clean.map_or(Value::Null, |s| json!([s.end, s.start])),
                    });
                }
                scores.push(score);
            }
        }

        let whole: Vec<char> = scored.text().to_vec();
        let mut offline: Option<(f64, usize)> = None;
        for end in (window..=whole.len()).step_by(params.check_every_chars as usize) {
            let ratio = compress_ratio(&whole[end - window..end].iter().collect::<String>());
            if offline.is_none_or(|(worst, _)| ratio < worst) {
                offline = Some((ratio, end));
            }
        }

        let object = record.as_object_mut().expect("an object");
        object.remove("text");
        object.insert("raw_len".into(), json!(scored.raw_len()));
        object.insert("scored_len".into(), json!(whole.len()));
        object.insert("scored_hash".into(), json!(fnv1a(&whole.iter().collect::<String>())));
        object.insert("online".into(), online.map_or(Value::Null, |(r, at, start)| json!([r, at, start])));
        object.insert("offline".into(), offline.map_or(Value::Null, |(r, end)| json!([r, end])));
        object.insert("checks".into(), Value::Array(checks));
        object.insert("raw_ratios".into(), scores.iter().map(|s| s.raw_ratio).collect());
        object.insert("fire".into(), first);
        object.insert("fire_ungated".into(), first_ungated);
        object.insert("cut".into(), cut);
        object.insert("repeat".into(), repeat);
        object.insert("guard".into(), actions.first().cloned().unwrap_or(Value::Null));
        if all {
            object.insert("guard_all".into(), Value::Array(actions));
        }
        writeln!(out, "{record}").expect("write");
        texts += 1;
    }
    out.flush().expect("flush");
    eprintln!(
        "{texts} texts scored, window {window}, {}, repeat min {} max {} copies {} varying {}",
        if params.normalize { "normalized" } else { "raw" },
        params.repeat_min_chars,
        params.repeat_max_chars,
        params.repeat_copies,
        params.repeat_varying
    );
}
