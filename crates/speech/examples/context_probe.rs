//! What the speech model writes under a context on audio with little speech in it, and what the
//! context-copy guard makes of it (#362, #442): the tool behind
//! docs/findings/2026-10-31-asr-context-near-copy.md.
//!
//! It transcribes the items a Realtime session cuts from a long 16 kHz mono WAV (`--wav`): the
//! turns `server_vad` finds at `--silence` ms (1500, the Transcribe view's default), or with
//! `--rttm` the reference turns, a speaker's joined while no one else speaks between, as a
//! session with speakers cuts them; each item is fed after the server's 500 ms of leading
//! silence. Or every WAV file of `--dir`, as `POST /v1/audio/transcriptions` takes a clip.
//! Each item runs under each context of `--contexts` (`{"name": {"prompt": "...", "keywords":
//! [...], "language": "Italian"}}`, joined as the server joins them) with the guard off, so each
//! row of `--out` (JSON lines, appended, rows already there skipped) is the model's own text.
//! Items run with a partial every 2 s of audio, as a live session refreshes, or in one push
//! with `--batch`; `--max-seconds` keeps the shorter items, where copies fall.
//!
//! `--rescore ROWS` runs no model: it applies the guard (`copies_context`) of this build to the
//! rows' text, at `speech.context_copy_words`' default and `speech.context_copy_edits` at each
//! of 0 to 3, and prints what each empties per context, and every row some edits empty that the
//! exact rule keeps (`--list`: every row emptied). With `--control a,b` it applies those contexts
//! instead to the rows' utterances, an item's text counted once, less the copies of the contexts
//! they were recorded under: a same-language control of what each setting takes of real speech.
//!
//! usage: context_probe DECODER ENCODER --threads N --contexts CTX.json [--only a,b] --out ROWS.jsonl
//!                      (--wav FILE [--rttm FILE] [--silence MS] | --dir DIR) [--batch] [--max-seconds S]
//!        context_probe --rescore ROWS.jsonl... --contexts CTX.json [--control a,b] [--list]

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::io::Write;

use ignis_speech::text::{context_of, copies_context};
use ignis_speech::vad::{leading_silence, TurnEvent, TurnSettings, Turns, Vad};
use ignis_speech::{ContextCopy, Executor, Features, ModelFiles, SessionTuning, Settings, Threads};
use serde_json::{json, Value};

const RATE: f64 = 16_000.0;
const LEAD_MS: u32 = 500;
const PADDING_S: f64 = 0.3;

/// An item: its name, where it starts and ends in the recording, its samples.
type Item = (String, f64, f64, Vec<f32>);

fn read_wav(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    assert!(&b[0..4] == b"RIFF" && &b[8..12] == b"WAVE", "{path}: not a WAV file");
    let mut i = 12;
    let mut data = None;
    while i + 8 <= b.len() {
        let len = u32::from_le_bytes(b[i + 4..i + 8].try_into().unwrap()) as usize;
        let body = &b[i + 8..(i + 8 + len).min(b.len())];
        if &b[i..i + 4] == b"fmt " {
            let ok = u16::from_le_bytes([body[0], body[1]]) == 1
                && u16::from_le_bytes([body[2], body[3]]) == 1
                && u32::from_le_bytes(body[4..8].try_into().unwrap()) == 16_000
                && u16::from_le_bytes([body[14], body[15]]) == 16;
            assert!(ok, "{path}: need 16 kHz mono 16-bit PCM");
        } else if &b[i..i + 4] == b"data" {
            data = Some(body);
        }
        i += 8 + len + (len & 1);
    }
    data.expect("no data chunk").as_chunks::<2>().0.iter().map(|s| i16::from_le_bytes(*s) as f32 / 32768.0).collect()
}

/// A context of `--contexts`, joined as the server joins it.
fn joined(spec: &Value) -> String {
    let keywords: Vec<String> = spec["keywords"].as_array().map_or(Vec::new(), |k| k.iter().filter_map(Value::as_str).map(str::to_string).collect());
    context_of(spec["prompt"].as_str(), &keywords)
}

fn lead(samples: &[f32]) -> Vec<f32> {
    let mut x = leading_silence(LEAD_MS);
    x.extend_from_slice(samples);
    x
}

/// The items `server_vad` gives a session: a turn from its prefix padding to the end of the
/// silence that ends it.
fn vad_items(audio: &[f32], silence_ms: u32) -> Vec<Item> {
    let settings = TurnSettings { threshold: 0.5, prefix_padding_ms: 300, silence_duration_ms: silence_ms, idle_timeout_ms: None };
    let mut turns = Turns::new(Vad::new(), settings, 0);
    let (mut out, mut current, mut events) = (Vec::new(), None::<(u64, Vec<f32>)>, Vec::new());
    for chunk in audio.chunks(1600) {
        turns.push(chunk, &mut events);
        for event in events.drain(..) {
            match event {
                TurnEvent::Started { start } => current = Some((start, Vec::new())),
                TurnEvent::Audio(s) => current.as_mut().expect("a turn").1.extend(s),
                TurnEvent::Stopped { end } => {
                    let (start, x) = current.take().expect("a turn");
                    out.push((format!("item-{:.2}", start as f64 / RATE), start as f64 / RATE, end as f64 / RATE, lead(&x)));
                }
                TurnEvent::Pause { .. } | TurnEvent::TimedOut { .. } => {}
            }
        }
    }
    out
}

/// The items a session with speakers cuts, from reference turns (RTTM): a speaker's turns
/// joined while no other speaker's starts between and the silence between is under
/// `silence_ms`, from 300 ms before to `silence_ms` after, or to the next turn's start.
fn rttm_items(audio: &[f32], rttm: &str, silence_ms: u32) -> Vec<Item> {
    let text = std::fs::read_to_string(rttm).unwrap_or_else(|e| panic!("{rttm}: {e}"));
    let mut turns: Vec<(f64, f64, String)> = text
        .lines()
        .filter_map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            let (start, length) = (f.get(3)?.parse::<f64>().ok()?, f.get(4)?.parse::<f64>().ok()?);
            Some((start, start + length, f.get(7)?.to_string()))
        })
        .collect();
    turns.sort_by(|a, b| a.0.total_cmp(&b.0));
    let gap = f64::from(silence_ms) / 1000.0;
    let mut joined: Vec<(f64, f64, String)> = Vec::new();
    for t in turns {
        match joined.last_mut() {
            Some(last) if last.2 == t.2 && t.0 - last.1 < gap => last.1 = last.1.max(t.1),
            _ => joined.push(t),
        }
    }
    let length = audio.len() as f64 / RATE;
    let mut out = Vec::new();
    for (i, (a, b, _)) in joined.iter().enumerate() {
        let next = joined.get(i + 1).map_or(length, |t| t.0);
        let (start, end) = ((a - PADDING_S).max(0.0), (b + gap).min(next.max(*b)).min(length));
        if start < end {
            out.push((format!("item-{start:.2}"), start, end, lead(&audio[(start * RATE) as usize..(end * RATE) as usize])));
        }
    }
    out
}

fn arg(args: &mut Vec<String>, name: &str) -> Option<String> {
    let i = args.iter().position(|a| a == name)?;
    args.remove(i);
    Some(args.remove(i))
}

fn flag(args: &mut Vec<String>, name: &str) -> bool {
    args.iter().position(|a| a == name).map(|i| args.remove(i)).is_some()
}

fn rows(paths: &[String]) -> Vec<Value> {
    paths.iter().flat_map(|p| std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{p}: {e}")).lines().map(|l| serde_json::from_str(l).unwrap()).collect::<Vec<Value>>()).collect()
}

/// The edits `--rescore` tries: every value `speech.context_copy_edits` takes.
const EDITS: [usize; 4] = [0, 1, 2, 3];

/// The fewest edits of [`EDITS`] at which the guard of this build empties `text` under `context`,
/// at the default run.
fn emptied_at(context: &str, text: &str) -> Option<usize> {
    EDITS.into_iter().find(|&edits| copies_context(context, text, ContextCopy { edits, ..ContextCopy::DEFAULT }))
}

/// What each value of [`EDITS`] empties of `at`: a transcript emptied at one edit is at more too.
fn counts(at: &[Option<usize>]) -> String {
    EDITS.iter().map(|&e| format!("{} at {e}", at.iter().filter(|a| a.is_some_and(|a| a <= e)).count())).collect::<Vec<_>>().join(", ")
}

/// What the guard of this build empties of recorded rows at each value of [`EDITS`], under each
/// row's own context; with `control`, under each of those contexts instead, of the utterances (an
/// item's text, once) that copy nothing of the contexts they were recorded under.
fn rescore(paths: &[String], contexts: &serde_json::Map<String, Value>, control: Option<Vec<String>>, list: bool) {
    let rows = rows(paths);
    let text = |row: &Value| row["text"].as_str().unwrap().to_string();
    let Some(control) = control else {
        let mut tally: BTreeMap<String, Vec<Option<usize>>> = BTreeMap::new();
        for row in &rows {
            let name = row["context"].as_str().unwrap();
            let at = emptied_at(&joined(&contexts[name]), &text(row));
            if let Some(edits) = at.filter(|&edits| edits > 0 || list) {
                println!("  {name}: {} at {edits} edits: {:?}", row["input"].as_str().unwrap(), text(row));
            }
            tally.entry(name.to_string()).or_default().push(at);
        }
        for (name, at) in tally {
            println!("{name}: {} rows, emptied {}", at.len(), counts(&at));
        }
        return;
    };
    let key = |row: &Value| (row["input"].as_str().unwrap().to_string(), text(row));
    let copies: HashSet<(String, String)> = rows.iter().filter(|row| emptied_at(&joined(&contexts[row["context"].as_str().unwrap()]), &text(row)).is_some()).map(key).collect();
    let said: BTreeSet<(String, String)> = rows.iter().map(key).filter(|k| !k.1.trim().is_empty() && !copies.contains(k)).collect();
    println!("{} utterances; {} copies of the context they were recorded under left out", said.len(), copies.len());
    let mut all = Vec::new();
    for name in &control {
        let context = joined(&contexts[name.as_str()]);
        let at: Vec<Option<usize>> = said.iter().map(|(input, text)| (input, text, emptied_at(&context, text))).inspect(|(input, text, at)| {
            if let Some(edits) = at {
                println!("  {name}: {input} at {edits} edits: {text:?}");
            }
        }).map(|(_, _, at)| at).collect();
        println!("{name}: {} utterances, emptied {}", at.len(), counts(&at));
        all.extend(at);
    }
    println!("all: {} utterances under a context, emptied {}", all.len(), counts(&all));
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let contexts: Value = serde_json::from_str(&std::fs::read_to_string(arg(&mut args, "--contexts").expect("--contexts FILE"))?)?;
    let contexts = contexts.as_object().expect("an object of contexts");
    if flag(&mut args, "--rescore") {
        let control = arg(&mut args, "--control").map(|s| s.split(',').map(str::to_string).collect());
        let list = flag(&mut args, "--list");
        rescore(&args, contexts, control, list);
        return Ok(());
    }
    let threads: usize = arg(&mut args, "--threads").expect("--threads N").parse()?;
    let only: Option<Vec<String>> = arg(&mut args, "--only").map(|s| s.split(',').map(str::to_string).collect());
    let out_path = arg(&mut args, "--out").expect("--out FILE");
    let silence: u32 = arg(&mut args, "--silence").map_or(1500, |v| v.parse().unwrap());
    let max_seconds: f64 = arg(&mut args, "--max-seconds").map_or(f64::MAX, |v| v.parse().unwrap());
    let batch = flag(&mut args, "--batch");
    let mut items: Vec<Item> = Vec::new();
    if let Some(wav) = arg(&mut args, "--wav") {
        let audio = read_wav(&wav);
        items = match arg(&mut args, "--rttm") {
            Some(rttm) => rttm_items(&audio, &rttm, silence),
            None => vad_items(&audio, silence),
        };
    }
    if let Some(dir) = arg(&mut args, "--dir") {
        let mut paths: Vec<_> = std::fs::read_dir(&dir)?.filter_map(|e| Some(e.ok()?.path())).filter(|p| p.extension().is_some_and(|x| x == "wav")).collect();
        paths.sort();
        for p in paths {
            items.push((p.file_stem().unwrap().to_string_lossy().into_owned(), 0.0, 0.0, read_wav(p.to_str().unwrap())));
        }
    }
    items.retain(|item| item.3.len() as f64 / RATE <= max_seconds);
    assert_eq!(args.len(), 2, "DECODER ENCODER");
    let files = ModelFiles { decoder: args[0].clone().into(), encoder: args[1].clone().into() };
    let settings = Settings { threads: Threads::Count(threads), ..Settings::default() };
    let copy = settings.context_copy;
    let off = SessionTuning { context_copy: ContextCopy { run: 0, ..copy }, ..settings.tuning() };
    let exec = Executor::new(settings, files, Features::detect());
    exec.load()?;
    let done: HashSet<(String, String)> = std::fs::read_to_string(&out_path)
        .map(|s| s.lines().filter_map(|l| serde_json::from_str::<Value>(l).ok()).map(|r| (r["context"].as_str().unwrap().to_string(), r["input"].as_str().unwrap().to_string())).collect())
        .unwrap_or_default();
    let mut out = std::fs::OpenOptions::new().create(true).append(true).open(&out_path)?;
    eprintln!("{} items", items.len());
    for (name, spec) in contexts {
        if only.as_ref().is_some_and(|o| !o.contains(name)) {
            continue;
        }
        let context = joined(spec);
        let started = std::time::Instant::now();
        let mut n = 0;
        for (input, start, end, x) in items.iter().filter(|item| !done.contains(&(name.clone(), item.0.clone()))) {
            let mut s = exec.open_tuned(&context, spec["language"].as_str(), off)?;
            if batch {
                s.push(x)?;
            } else {
                for chunk in x.chunks(1600) {
                    s.push(chunk)?;
                    if s.partial_due() {
                        s.partial()?;
                    }
                }
            }
            let t = s.commit()?;
            let copied = copies_context(&context, &t.text, copy);
            let seconds = (x.len() as f64 / RATE * 100.0).round() / 100.0;
            writeln!(out, "{}", json!({ "context": name, "input": input, "start": start, "end": end, "seconds": seconds, "text": t.text, "language": t.language, "copied": copied }))?;
            n += 1;
            println!("{name:16} {input:16} {seconds:6.2} s copied={} {:?}", u8::from(copied), t.text.chars().take(100).collect::<String>());
        }
        eprintln!("{name}: {n} items in {:.1} s", started.elapsed().as_secs_f64());
    }
    Ok(())
}
