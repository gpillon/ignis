//! The executor and the session on a tiny synthetic GGUF pair, in the default suite (spec
//! speech/01 testing decisions): load at first use and unload, pinned weights, the Q4_K repack
//! as the only copy, the prefix cache, the incremental session against a whole-clip run,
//! partials and their stable text, a commit a unit of compute at a time, and the KV cache's
//! text room.

mod common;

use common::synthetic;
use ignis_speech::session::DECODE_UNIT;
use ignis_speech::text::Prompt;
use ignis_speech::{Availability, Executor, Features, Model, ModelFiles, SessionTuning, Settings, Threads};

fn settings() -> Settings {
    Settings { threads: Threads::Count(2), max_clip_seconds: 60, ..Settings::default() }
}

fn exec(files: &ModelFiles, s: Settings) -> Executor {
    let e = Executor::new(s, files.clone(), Features::detect());
    if e.availability() != Availability::Available {
        panic!("this CPU lacks AVX2: the executor cannot serve here");
    }
    e
}

/// Acceptance 21 and 7: nothing is mapped before the first signal; the Q4_K weights are repacked
/// at load and their file bytes are not counted among the mapped ones; after an unload the
/// process holds nothing of the model (on Windows a mapped file cannot be deleted: deleting it
/// shows the mapping is gone).
#[test]
fn the_model_loads_at_first_use_and_unloads_whole() {
    let files = synthetic::pair("load");
    let e = exec(&files, settings());
    assert!(!e.is_loaded(), "loaded before the first signal");
    let r = e.load().unwrap().expect("the first load reports");
    assert!(e.is_loaded());
    assert_eq!(r.repacked_bytes, synthetic::q4k_bytes(), "the Q4_K weights are repacked");
    let file_bytes = std::fs::metadata(&files.decoder).unwrap().len() as usize + std::fs::metadata(&files.encoder).unwrap().len() as usize;
    assert!(r.mapped_bytes + r.repacked_bytes < file_bytes, "the repacked tensors are not also mapped");
    assert!(!r.pinned);
    assert!(e.load().unwrap().is_none(), "a second load is a no-op");
    e.unload();
    assert!(!e.is_loaded());
    synthetic::remove(&files).expect("the files are still mapped after the unload");
}

/// Acceptance 21: an unload (the server's worker asks for one after `speech.idle_unload`, spec
/// 02) gives the files back once every session ended, and a later session loads the model again.
#[test]
fn a_session_after_an_unload_loads_the_model_again() {
    let files = synthetic::pair("idle");
    let e = exec(&files, settings());
    let mut s = e.open("", None).unwrap();
    s.push(&synthetic::audio(1.0)).unwrap();
    s.commit().unwrap();
    e.unload();
    assert!(!e.is_loaded());
    // Every session ended: the files are unmapped.
    std::fs::rename(&files.decoder, files.decoder.with_extension("moved")).expect("the decoder is still mapped");
    std::fs::rename(files.decoder.with_extension("moved"), &files.decoder).unwrap();
    let mut s = e.open("", None).unwrap();
    assert!(e.is_loaded());
    s.push(&synthetic::audio(0.5)).unwrap();
    s.commit().unwrap();
}

/// Acceptance 21: pinned weights are a private copy; the files are not held while it serves.
#[test]
fn pinned_weights_hold_no_mapping_of_the_files() {
    let files = synthetic::pair("pinned");
    let mapped = exec(&files, settings());
    let mut s = mapped.open("", None).unwrap();
    s.push(&synthetic::audio(2.0)).unwrap();
    let expected = s.commit().unwrap();
    drop(mapped);
    let e = exec(&files, Settings { pinned_weights: true, ..settings() });
    let r = e.load().unwrap().unwrap();
    assert!(r.pinned);
    // The copy holds every tensor but the repacked Q4_K ones, whose only copy is the repack.
    assert_eq!(r.repacked_bytes, synthetic::q4k_bytes());
    let file_bytes = std::fs::metadata(&files.decoder).unwrap().len() as usize + std::fs::metadata(&files.encoder).unwrap().len() as usize;
    assert!(r.pinned_bytes > 0 && r.pinned_bytes + r.repacked_bytes <= file_bytes, "{} + {} over {file_bytes}", r.pinned_bytes, r.repacked_bytes);
    let moved = files.decoder.with_extension("moved");
    std::fs::rename(&files.decoder, &moved).expect("pinned weights still map the decoder");
    let mut s = e.open("", None).unwrap();
    s.push(&synthetic::audio(2.0)).unwrap();
    assert_eq!(s.commit().unwrap(), expected, "the pinned copy transcribes as the mapping does");
    std::fs::rename(&moved, &files.decoder).unwrap();
}

/// Acceptance 17's cache: the same context twice prefills it once; a new one replaces the entry.
#[test]
fn the_prefix_cache_holds_one_context() {
    let files = synthetic::pair("prefix");
    let e = exec(&files, settings());
    let prefilled = |c: &str| e.open(c, None).unwrap().stats().context_prefilled;
    let a = prefilled("first context, a few words");
    assert!(a > 0);
    assert_eq!(prefilled("first context, a few words"), 0);
    assert!(prefilled("another one") > 0);
    assert_eq!(prefilled("first context, a few words"), a, "the new context did not replace the entry");
    // Over the cap the context keeps its last tokens.
    let capped = Executor::new(Settings { context_max_tokens: 4, ..settings() }, files.clone(), Features::detect());
    assert_eq!(capped.open("a context longer than four tokens", None).unwrap().stats().context_prefilled, 4);
}

/// The incremental session against a whole-clip run: pushed 100 ms at a time or all at once, a
/// clip gives the same transcript, each window encoded once into the audio prefix. 40 s outgrow
/// the KV cache's first allocation, so it grows on the way.
#[test]
fn pushing_in_pieces_gives_the_whole_clip_transcript() {
    let files = synthetic::pair("pieces");
    let e = exec(&files, settings());
    let audio = synthetic::audio(40.0);
    let mut whole = e.open("", None).unwrap();
    whole.push(&audio).unwrap();
    let ws = whole.stats().clone();
    let w = whole.commit().unwrap();
    let mut s = e.open("", None).unwrap();
    for c in audio.chunks(1600) {
        s.push(c).unwrap();
    }
    let ps = s.stats().clone();
    let p = s.commit().unwrap();
    assert_eq!(p, w);
    // 40 s is 4,000 frames: four whole windows while pushing (the fifth needs the last 200
    // samples' frame), the fifth at commit.
    assert_eq!((ws.windows, ps.windows), (4, 4));
    assert_eq!((ws.audio_prefilled, ps.audio_prefilled), (416, 416));
    assert_eq!(w.audio_tokens, 520);
    assert_eq!(w.tokens.len(), 1, "the synthetic decoder emits one token");
}

/// Acceptance 18's mechanics: a partial every `partial_interval_ms` of audio, the cache rolled
/// back after each (the transcript is the whole-clip one), and with no rollback tokens the
/// decoded token joins the stable text, which the transcript begins with.
#[test]
fn partials_roll_back_and_the_transcript_keeps_the_stable_text() {
    let files = synthetic::pair("partials");
    let audio = synthetic::audio(9.3);
    let e = exec(&files, settings());
    let mut whole = e.open("", None).unwrap();
    whole.push(&audio).unwrap();
    let w = whole.commit().unwrap();
    for rollback in [5, 0] {
        let e = exec(&files, Settings { rollback_tokens: rollback, unfixed_partials: 0, ..settings() });
        let mut s = e.open("", None).unwrap();
        let mut partials = Vec::new();
        for c in audio.chunks(1600) {
            s.push(c).unwrap();
            if s.partial_due() {
                partials.push(s.partial().unwrap());
            }
        }
        assert_eq!(partials.len(), 4, "a partial every 2 s of 9.3 s");
        assert_eq!(s.stats().partials, 4);
        let t = s.commit().unwrap();
        for p in &partials {
            assert!(t.text.starts_with(&p.stable), "the transcript does not begin with {:?}", p.stable);
            if rollback > 0 {
                // One decoded token, all of it provisional: nothing is stable.
                assert_eq!(p.stable, "");
            }
        }
        if rollback > 0 {
            assert_eq!(t, w, "partials changed the transcript: the cache was not rolled back");
        }
    }
}

/// A stop that says so from its call after the first `points`: a partial stopped after passing
/// that many of the points it checks.
fn stop_after(points: usize) -> impl Fn() -> bool {
    let polls = std::cell::Cell::new(0);
    move || {
        polls.set(polls.get() + 1);
        polls.get() > points
    }
}

/// Q79: a partial stopped at any of the points it checks leaves the session as it was before it
/// began: the cache rolled back to the audio prefix, so the commit's transcript is the one with
/// no partial attempted, and the partial not counted (the count decides which partials fix no
/// text, `unfixed_partials`). The synthetic decoder emits one token, so its decode checks once
/// before that token's forward pass.
#[test]
fn a_stopped_partial_leaves_the_session_as_it_was() {
    let files = synthetic::pair("stopped");
    let audio = synthetic::audio(9.3);
    // A 20 s cache leaves room for about ten partials not rolled back (below).
    let e = exec(&files, Settings { rollback_tokens: 0, unfixed_partials: 0, max_clip_seconds: 20, ..settings() });
    let session = || {
        let mut s = e.open("", None).unwrap();
        s.push(&audio).unwrap();
        s
    };
    let untried = session().commit().unwrap();
    let mut points = 0;
    while points < 64 {
        let mut s = session();
        let before = s.stats().clone();
        if s.partial_unless(&stop_after(points)).unwrap().is_some() {
            assert_eq!(s.stats().partials, 1, "a partial nothing stopped counts");
            break;
        }
        assert_eq!(*s.stats(), before, "stopped after {points} checks: the partial counted");
        assert_eq!(s.commit().unwrap(), untried, "stopped after {points} checks: the cache was not rolled back");
        let mut s = session();
        assert!(s.partial_unless(&stop_after(points)).unwrap().is_none());
        s.partial().unwrap();
        assert_eq!(s.stats().partials, 1, "stopped after {points} checks: the next partial is the first");
        assert_eq!(s.commit().unwrap(), untried, "stopped after {points} checks, then a partial: the transcript changed");
        points += 1;
    }
    // Before the encode, before the window's prefill, before the decode's prefill and after it,
    // and before the one token's forward pass.
    assert_eq!(points, 5, "the points a partial checks");
    // The synthetic decoder's token does not depend on what the cache holds, so one commit
    // cannot tell a cache left grown. A hundred partials stopped and not rolled back would fill
    // it about ten times over (each adds the incomplete window and the prompt's tail): the
    // session takes them all.
    let mut s = session();
    for i in 0..100 {
        assert!(s.partial_unless(&stop_after(i % points)).unwrap().is_none());
    }
    assert_eq!(s.stats().partials, 0);
    assert_eq!(s.commit().unwrap(), untried, "a hundred stopped partials changed the transcript");
}

/// Q79: a partial's decode checks its stop before every token's forward pass, so a commit waits
/// at most one decode step for the refresh in flight. On the pair whose head has no EOS the
/// decode runs to the KV cache's end; a partial stopped at its first token, or at its last,
/// leaves the session as it was, as one stopped before the decode does.
#[test]
fn a_partial_stops_at_every_step_of_its_decode() {
    let files = synthetic::endless_pair("yields");
    let audio = synthetic::audio(9.3);
    let e = exec(&files, Settings { rollback_tokens: 0, unfixed_partials: 0, max_clip_seconds: 20, ..settings() });
    let session = || {
        let mut s = e.open("", None).unwrap();
        s.push(&audio).unwrap();
        s
    };
    let untried = session().commit().unwrap();
    assert!(untried.tokens.len() > 2 * DECODE_UNIT, "the decode runs long: {} tokens", untried.tokens.len());
    let polls = std::cell::Cell::new(0);
    let counted = || {
        polls.set(polls.get() + 1);
        false
    };
    assert!(session().partial_unless(&counted).unwrap().is_some());
    let checks = polls.get();
    // Four before the decode's first step, then one a token: as many checks as tokens, where a
    // check every 16 steps made a sixteenth of them.
    assert!(checks >= untried.tokens.len(), "a decode of {} tokens checked {checks} times", untried.tokens.len());
    for points in [4, checks - 1] {
        let mut s = session();
        let before = s.stats().clone();
        assert!(s.partial_unless(&stop_after(points)).unwrap().is_none(), "not stopped after {points} checks");
        assert_eq!(*s.stats(), before, "stopped after {points} checks: the partial counted");
        assert_eq!(s.commit().unwrap(), untried, "stopped after {points} checks: the session changed");
    }
}

/// The server passes `speech.context_max_tokens`, `speech.rollback_tokens` and
/// `speech.unfixed_partials` per session: a session holds the tuning it opened with, not the
/// executor's settings, so a live change of the config holds from the next session. (The
/// synthetic decoder's one token decodes to no text, so the tuning is read back rather than seen
/// in the stable text; `partials_roll_back_and_the_transcript_keeps_the_stable_text` shows the
/// rollback itself.)
#[test]
fn a_session_holds_the_tuning_it_opened_with() {
    let files = synthetic::pair("tuning");
    let e = exec(&files, Settings { rollback_tokens: 10, unfixed_partials: 2, context_max_tokens: 512, ..settings() });
    let tuned = SessionTuning { context_max_tokens: 64, rollback_tokens: 3, unfixed_partials: 0, context_copy_words: 4, context_copy_edits: 1 };
    assert_eq!(e.open_tuned("", None, tuned).unwrap().tuning(), tuned);
    assert_eq!(e.open("", None).unwrap().tuning(), SessionTuning { context_max_tokens: 512, rollback_tokens: 10, unfixed_partials: 2, context_copy_words: 3, context_copy_edits: 2 });
}

/// Acceptance 19 (Q44): a commit is a unit of work a step, so the server's one worker runs other
/// sessions' units between two: the last window with the audio's last frames, then the decode's
/// prefill, then 16 decode steps a step. On the pair whose head has no EOS the decode runs to the
/// KV cache's end; stepped, the commit gives the transcript a whole one does.
#[test]
fn a_commit_is_a_unit_of_work_a_step() {
    let files = synthetic::endless_pair("steps");
    let e = exec(&files, Settings { max_clip_seconds: 20, ..settings() });
    let session = || {
        let mut s = e.open("", None).unwrap();
        s.push(&synthetic::audio(9.3)).unwrap();
        s
    };
    let whole = session().commit().unwrap();
    let decoded = whole.tokens.len();
    assert!(decoded > 4 * DECODE_UNIT, "the decode runs long: {decoded} tokens");
    let mut commit = session().into_commit();
    let mut steps = 0;
    let stepped = loop {
        steps += 1;
        if let Some(t) = commit.step().unwrap() {
            break t;
        }
    };
    assert_eq!(stepped, whole);
    // The window, the prefill, then one step per 16 forward passes: the cache's end comes with
    // the last token taken, which is not forwarded.
    assert_eq!(steps, 2 + (decoded - 1) / DECODE_UNIT + 1);
    assert!(matches!(commit.step(), Err(ignis_speech::SessionError::Committed)));
}

/// Acceptance 19: the decode's prefill (the prompt's tail and the stable text, which grows with
/// a long dictation's refreshes) is a window's worth of tokens a step, not the whole text at
/// once. A refresh with no rollback on the pair whose head has no EOS fixes the whole text room
/// as stable; the commit then prefills it over several steps and gives the transcript a whole
/// commit does.
#[test]
fn a_commits_prefill_is_a_window_of_tokens_a_step() {
    let files = synthetic::endless_pair("prefill");
    let e = exec(&files, Settings { rollback_tokens: 0, unfixed_partials: 0, max_clip_seconds: 20, ..settings() });
    let session = || {
        let mut s = e.open("", None).unwrap();
        s.push(&synthetic::audio(9.3)).unwrap();
        s.partial().unwrap();
        s
    };
    let stable = session().stats().stable_tokens;
    assert!(stable > 104, "the refresh fixed {stable} tokens");
    let whole = session().commit().unwrap();
    let mut commit = session().into_commit();
    let mut steps = 0;
    let stepped = loop {
        steps += 1;
        if let Some(t) = commit.step().unwrap() {
            break t;
        }
    };
    assert_eq!(stepped, whole);
    let tail = Prompt::new(&Model::load(&files.decoder, &files.encoder, false).unwrap().text, "", 512, None).tail.len();
    let decoded = whole.tokens.len() - stable;
    // The last window, the prefill 104 tokens a step, then the decode 16 forward passes a step.
    assert_eq!(steps, 1 + (tail + stable).div_ceil(104) + (decoded - 1) / DECODE_UNIT + 1);
}

/// The KV cache keeps a text room after the longest clip's audio and the stable text never
/// outgrows it, so no clip within `max_clip_seconds` fails for want of cache: on the pair whose
/// head has no EOS a partial halfway through a 20 s clip decodes to the cache's end, and with no
/// rollback all of it would turn stable, more than the room left once the rest of the audio is
/// in. The commit still has the room its prefill needs.
#[test]
fn stable_text_never_outgrows_the_text_room() {
    let files = synthetic::endless_pair("room");
    let e = exec(&files, Settings { rollback_tokens: 0, unfixed_partials: 0, max_clip_seconds: 20, ..settings() });
    let audio = synthetic::audio(19.0);
    let mut s = e.open("", None).unwrap();
    s.push(&audio[..audio.len() / 2]).unwrap();
    let p = s.partial().unwrap();
    s.push(&audio[audio.len() / 2..]).unwrap();
    let t = s.commit().expect("a 19 s clip within a 20 s limit commits");
    assert!(t.text.starts_with(&p.stable));
}

/// Spec speech/08: a session cut at a sample is the session pushed only the audio before it, the
/// same way: a refresh and the transcript are that session's. The windows past the cut leave the
/// audio prefix and the one holding it is encoded again; the stable text, which may hold words
/// past the cut, goes. A session that keeps its audio is cut back over whole windows; one that
/// does not reaches only the audio it still keeps, and says so past it.
#[test]
fn a_cut_session_is_the_session_of_the_audio_before_the_cut() {
    // The pair whose decode runs to the KV cache's end: how far it gets reads the audio prefix's
    // length, and with no rollback all of it is stable.
    let files = synthetic::endless_pair("cut");
    let e = exec(&files, Settings { rollback_tokens: 0, unfixed_partials: 0, max_clip_seconds: 40, ..settings() });
    let audio = synthetic::audio(30.0);
    let session = |upto: usize, keep: bool| {
        let mut s = e.open("", None).unwrap();
        if keep {
            s.keep_audio();
        }
        for c in audio[..upto].chunks(1600) {
            s.push(c).unwrap();
        }
        s
    };
    // 12.3 s is inside the second window, two windows behind the three 30 s encoded; 28 s is
    // past them, in the audio not yet a window.
    // A commit stepped to its end, and its steps: a stable text left behind would be prefilled,
    // a window of tokens a step, where the session of the audio before the cut decodes.
    let stepped = |s: ignis_speech::Session| {
        let mut commit = s.into_commit();
        let mut steps = 0;
        loop {
            steps += 1;
            if let Some(t) = commit.step().unwrap() {
                return (t, steps);
            }
        }
    };
    let refreshed_and_cut = |at: usize, keep: bool| {
        let mut s = session(audio.len(), keep);
        s.partial().unwrap();
        assert!(s.stats().stable_tokens > 104, "a refresh fixed more than a window of tokens");
        s.cut(at).unwrap();
        assert_eq!((s.seconds(), s.stats().stable_tokens), (at as f64 / 16_000.0, 0));
        s
    };
    for (at, keep) in [(196_800, true), (448_000, false), (448_000, true)] {
        assert_eq!(stepped(refreshed_and_cut(at, keep)), stepped(session(at, keep)), "a cut at {at}");
        let (mut cut, mut reference) = (refreshed_and_cut(at, keep), session(at, keep));
        assert_eq!(cut.partial().unwrap(), reference.partial().unwrap(), "a refresh after a cut at {at}");
        assert_eq!(cut.stats().stable_tokens, reference.stats().stable_tokens, "a refresh after a cut at {at}");
    }
    let mut unkept = session(audio.len(), false);
    assert!(matches!(unkept.cut(196_800), Err(ignis_speech::SessionError::CutTooFar { at: 196_800 })));
    // A cut at the end drops the stable text only.
    let mut whole = session(audio.len(), false);
    whole.partial().unwrap();
    whole.cut(audio.len()).unwrap();
    assert_eq!(stepped(whole), stepped(session(audio.len(), false)));
}

/// Acceptance 16: no language is forced unless asked, and a forced one is the official
/// assistant prefill, `language <Name><asr_text>` after the assistant turn's head; the model's
/// output is then all text, labelled with the forced language.
#[test]
fn a_forced_language_is_the_official_prefill() {
    let files = synthetic::pair("forced");
    let model = Model::load(&files.decoder, &files.encoder, false).unwrap();
    let text = &model.text;
    let auto = Prompt::new(text, "", 512, None);
    let forced = Prompt::new(text, "", 512, Some("Italian"));
    assert_eq!(forced.head, auto.head, "the prefix cache's entry does not depend on the language");
    assert!(text.decode(&auto.tail).ends_with("assistant\n"), "{:?}", text.decode(&auto.tail));
    assert!(text.decode(&forced.tail).ends_with("assistant\nlanguage Italian<asr_text>"), "{:?}", text.decode(&forced.tail));
    assert_eq!(forced.tail.last(), Some(&text.asr_text), "`<asr_text>` is the marker token, not its spelling");
    drop(model);
    let e = exec(&files, settings());
    let transcribe = |language: Option<&str>| {
        let mut s = e.open("", language).unwrap();
        s.push(&synthetic::audio(2.0)).unwrap();
        s.commit().unwrap()
    };
    assert_eq!(transcribe(Some("Italian")).language.as_deref(), Some("Italian"));
    // The synthetic decoder's one token is no `language X<asr_text>` header: auto-detection
    // labels nothing.
    assert_eq!(transcribe(None).language, None);
}

/// #362: a transcript is checked for a copy of the context the model read, which past the cap
/// is only the context's last `max_context` tokens.
#[test]
fn the_prompt_keeps_the_context_the_model_reads() {
    let files = synthetic::pair("context-read");
    let model = Model::load(&files.decoder, &files.encoder, false).unwrap();
    let text = &model.text;
    let context = "Topic: the release plan.\nalpha, beta, gamma, delta";
    assert_eq!(Prompt::new(text, context, 512, None).context, context);
    let cut = Prompt::new(text, context, 4, None);
    assert_eq!(cut.context_tokens, 4);
    assert!(!cut.context.is_empty() && cut.context.len() < context.len() && context.ends_with(&cut.context), "{:?}", cut.context);
}

/// Acceptance 20: `auto` is the physical cores less one, an explicit count is used as given.
#[test]
fn threads_resolve_auto_and_counts() {
    let usable = num_cpus::get_physical().min(num_cpus::get());
    assert_eq!(Threads::Auto.resolve(), ((usable - 1).max(1), "auto"));
    assert_eq!(Threads::Count(3).resolve(), (3, "configured"));
    assert_eq!(Threads::Count(0).resolve(), (1, "configured"));
}

/// Spec speech/08: `diarizer.threads`' `auto` is a third of the physical cores, at least 2 (3 on
/// the i9's 10, where conv01 ran 2.4, 3.3 and 3.5 times faster than real time on 2, 3 and 4), and
/// a count is used as given.
#[test]
fn diarizer_threads_resolve_auto_and_counts() {
    assert_eq!([2, 4, 6, 10, 16].map(Threads::diarizer_auto), [2, 2, 2, 3, 5]);
    let usable = num_cpus::get_physical().min(num_cpus::get());
    assert_eq!(Threads::Auto.resolve_diarizer(), (Threads::diarizer_auto(usable), "auto"));
    assert_eq!(Threads::Count(4).resolve_diarizer(), (4, "configured"));
    assert_eq!(Threads::Count(0).resolve_diarizer(), (1, "configured"));
}
