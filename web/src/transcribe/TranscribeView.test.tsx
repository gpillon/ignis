import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { SECURE_CONTEXT, TURNED_OFF } from "../speech/availability.ts";
import { ContextCard } from "./Context.tsx";
import { ANSWER_AFTER_MS, Deck, deckHint, HEADPHONES, startBlocked, startPress, Strip, stripStatus, timeLeft } from "./Deck.tsx";
import { IDLE, type ListeningState } from "./listening.ts";
import { AskBar } from "./Ask.tsx";
import { type Editing, type Marked, NotesPanel, notesButton, recapButton, rewriteButton } from "./Notes.tsx";
import type { AnalystState } from "./analyst.ts";
import type { AskerState } from "./ask.ts";
import { SuggestionsPanel, suggestionText } from "./Suggestions.tsx";
import { suggest } from "./suggestions.ts";
import type { NoteMark } from "./marks.ts";
import type { NotesWriterState } from "./notesWriter.ts";
import { Heading, highlighted, nextCheckIn, RewritePanel, Sheet, speakerColor } from "./Sheet.tsx";
import { startList, titleOf } from "./sessions.ts";
import { EMPTY_CONTEXT } from "./context.ts";
import { Call, fileStem, listMatch, SearchField, Settings, TranscribeView, transcriptSummary } from "./TranscribeView.tsx";
import { addContextImage } from "./contextImages.ts";
import { StaticPictures } from "../image/pictures.tsx";
import { hitsIn } from "./search.ts";
import { addUtterance, createTranscript, fixUtterances, sessionsWithSpeakers, splitUtterance, type Transcript } from "./transcript.ts";
import type { TranscribeActions, TranscribeState } from "./useTranscribe.ts";

// The Transcribe view drawn in its states (spec speech/07): the sheet, the
// deck and its strip, the list's rows.

const none = () => {};
const actions: TranscribeActions = {
  select: none,
  newTranscript: none,
  remove: none,
  rename: none,
  start: none,
  pause: none,
  resume: none,
  stop: none,
  editContext: none,
  changeContext: none,
  applyContext: none,
  cancelContext: none,
  copyContext: none,
  saveDefaultContext: none,
  setSpeakers: none,
  setHoldMic: none,
  setSilence: none,
  renameSpeaker: none,
  moveUtterance: none,
  mergeSpeakers: none,
  splitUtterance: none,
  editText: none,
  setNoteKind: none,
  writeNotes: none,
  rewriteNotes: none,
  writeRecap: none,
  stopNotes: none,
  setNotesLive: none,
  editNote: none,
  addNote: none,
  removeNote: none,
  acceptSuggestion: none,
  dismissSuggestion: none,
  dropSuggestedTerm: none,
  suggestNow: none,
  stopSuggesting: none,
  setSuggestionsOn: none,
  undoFixes: none,
  deleteUtterance: none,
  undoDelete: none,
  holdUndo: none,
  rewriteUtterance: none,
  acceptRewrite: none,
  closeRewrite: none,
  askAbout: none,
  stopAsking: none,
  forgetAnswer: none,
  askFrom: none,
  clearMissed: none,
  addImage: async () => {},
  captureFrame: async () => null,
};

const at = (state: Partial<ListeningState>): ListeningState => ({ ...IDLE, ...state });

function room(): Transcript {
  let t = createTranscript(1, { source: "microphone", at: new Date(2026, 9, 25, 15, 42) });
  t = addUtterance(t, { stream: "mic-1", startMs: 4_000, endMs: 8_000, text: "Apriamo. Io ho finito ieri.", voice: { kind: "speaker", index: 0 } });
  return t;
}

const sheet = (t: Transcript, listening: ListeningState = IDLE, here = false) =>
  renderToStaticMarkup(
    <Sheet transcript={t} live={here ? listening.live : []} phase={listening.phase} call={listening.source === "call"} here={here} actions={actions} deleted={null} hits={[]} current={0} rewrite={null} canRewrite readTo={null} seek={0} autoCheck asking={null} />,
  );

describe("the sheet", () => {
  it("hangs each utterance on the spine: its mm:ss, its speaker as a menu in its hue (a move, a new speaker), its text", () => {
    const html = sheet(room());
    expect(html).toContain(">00:04<");
    expect(html).toMatch(/<select aria-label="Speaker of the utterance at 00:04"[^>]*style="color:var\(--speaker-1\)"/);
    expect(html).toContain('<option value="1" selected="">Speaker 1</option>');
    expect(html).toContain('<option value="new">New speaker</option>');
    expect(html).toContain("Apriamo. Io ho finito ieri.");
  });

  it("is headed, above it, by the title, the start and the exports (Q129)", () => {
    const html = renderToStaticMarkup(
      <Heading transcript={room()} onRename={none}>
        <button type="button">Copy</button>
      </Heading>,
    );
    expect(html).toContain('value="25 Oct 15:42 · Microphone"');
    expect(html).toContain("Started 25 Oct 2026, 15:42");
    expect(html).toContain(">Copy</button>");
  });

  it("marks the words a fix put in, and offers to undo it (Q133)", () => {
    const html = sheet(fixUtterances(room(), "Apriamo", "Apriremo", [1]));
    expect(html).toContain("decoration-dotted");
    expect(html).toMatch(/<mark[^>]*>Apriremo<\/mark>/);
    expect(html).toContain('title="Heard as: Apriamo. Io ho finito ieri."');
    expect(html).toContain(">Undo fix<");
    expect(sheet(room())).not.toContain("Undo fix");
  });

  it("marks a fixed word only where it stands as a word of its own, never inside a longer one", () => {
    const html = renderToStaticMarkup(<p>{highlighted("SQL e PostgreSQL, poi SQL.", ["SQL"])}</p>);
    expect(html.match(/<mark/g)).toHaveLength(2);
    expect(html).toContain("PostgreSQL,");
  });

  it("offers Undo for a few seconds after an utterance is deleted (Q137)", () => {
    const t = room();
    const html = renderToStaticMarkup(<Sheet transcript={t} live={[]} phase="idle" call={false} here actions={actions} deleted={t.utterances[0]} hits={[]} current={0} rewrite={null} canRewrite readTo={null} seek={0} autoCheck asking={null} />);
    expect(html).toMatch(/Utterance at <span[^>]*>00:04<\/span> deleted\./);
    expect(html).toMatch(/<button type="button" autofocus=""[^>]*>Undo<\/button>/);
  });

  it("lights the search's hits, the one it stands on in the ember (Q136)", () => {
    let t = room();
    t = addUtterance(t, { stream: "mic-1", startMs: 9_000, endMs: 9_500, text: "Perché apriamo?", voice: null });
    const hits = hitsIn(t, "apriamo");
    expect(hits).toHaveLength(2);
    const html = renderToStaticMarkup(<Sheet transcript={t} live={[]} phase="idle" call={false} here actions={actions} deleted={null} hits={hits} current={1} rewrite={null} canRewrite readTo={null} seek={0} autoCheck asking={null} />);
    expect(html).toMatch(/<mark data-hit="other"[^>]*>Apriamo<\/mark>/);
    expect(html).toMatch(/<mark data-hit="current" class="bg-ember[^"]*">apriamo<\/mark>/);
    expect(html).toContain('data-utterance="2"');
  });

  it("shows a rewrite's proposal under its utterance, the words it adds and drops, with Accept and Discard (Q138)", () => {
    const t = room();
    const proposal = { transcript: 1, utterance: 1, writing: false, proposal: "Apriamo. Io ho chiuso ieri.", failed: null };
    const html = renderToStaticMarkup(<Sheet transcript={t} live={[]} phase="idle" call={false} here actions={actions} deleted={null} hits={[]} current={0} rewrite={proposal} canRewrite readTo={null} seek={0} autoCheck asking={null} />);
    expect(html).toMatch(/<ins[^>]*>chiuso<\/ins>/);
    expect(html).toMatch(/<del[^>]*>finito<\/del>/);
    expect(html).toContain(">Accept<");
    expect(html).toContain(">Discard<");
    const writing = renderToStaticMarkup(<RewritePanel utterance={t.utterances[0]} rewrite={{ ...proposal, writing: true, proposal: null }} actions={actions} />);
    expect(writing).toContain("spine-live");
    expect(renderToStaticMarkup(<RewritePanel utterance={t.utterances[0]} rewrite={{ ...proposal, proposal: t.utterances[0].text }} actions={actions} />)).toContain(
      "The model would leave it as it is.",
    );
  });

  it("draws where the text model's look stopped reading, and when the next may come (Q140)", () => {
    let t = room();
    t = addUtterance(t, { stream: "mic-1", startMs: 9_000, endMs: 9_500, text: "Poi?", voice: null });
    const sheetAt = (checked: { untilMs: number; nextAt: number | null } | null) =>
      renderToStaticMarkup(<Sheet transcript={t} live={[]} phase="idle" call={false} here actions={actions} deleted={null} hits={[]} current={0} rewrite={null} canRewrite readTo={checked} seek={0} autoCheck asking={null} />);
    const html = sheetAt({ untilMs: 8_000, nextAt: null });
    expect(html.indexOf("Checked to here")).toBeGreaterThan(html.indexOf("Apriamo."));
    expect(html.indexOf("Checked to here")).toBeLessThan(html.indexOf("Poi?"));
    expect(html).toContain('role="separator" aria-label="Checked by the text model up to here"');
    // All read: the line at the foot, after the last utterance.
    const all = sheetAt({ untilMs: 9_500, nextAt: null });
    expect(all.indexOf("Checked to here")).toBeGreaterThan(all.indexOf("Poi?"));
    expect(sheetAt(null)).not.toContain("Checked to here");
    expect(nextCheckIn(1_000, 0, 1)).toBe("next check once more is said");
    expect(nextCheckIn(41_000, 0)).toBe("next check in about 45 s");
    expect(nextCheckIn(90_000, 0)).toBe("next check in about 2 min");
    expect(nextCheckIn(3_000, 0)).toBe("next check any moment");
  });

  it("says, rewriting an utterance the look has not read, that the automatic check has not reached it, the proposal shown or not", () => {
    const t = room();
    const writing = { transcript: 1, utterance: 1, writing: true, proposal: null, failed: null };
    const note = "The automatic check has not read this utterance yet";
    expect(renderToStaticMarkup(<RewritePanel utterance={t.utterances[0]} rewrite={writing} unchecked actions={actions} />)).toContain(note);
    expect(renderToStaticMarkup(<RewritePanel utterance={t.utterances[0]} rewrite={{ ...writing, writing: false, proposal: "Apriremo." }} unchecked actions={actions} />)).toContain(note);
    expect(renderToStaticMarkup(<RewritePanel utterance={t.utterances[0]} rewrite={writing} actions={actions} />)).not.toContain(note);
  });

  it("marks a split's guessed start with ~ (Q98)", () => {
    const t = room();
    const html = sheet(splitUtterance(t, t.utterances[0].id, 9, "new"));
    expect(html).toContain(">~00:05<");
  });

  it("shows the utterances under way at the foot, the tail light, the spine burning while it listens (Q82)", () => {
    const listening = at({ phase: "listening", transcript: 1, live: [{ key: "mic-1:item_2", startMs: 9_000, stable: "Bene", tail: " avanti", lane: "mic" }] });
    const html = sheet(room(), listening, true);
    expect(html).toContain(">00:09<");
    expect(html).toMatch(/<span class="text-ink">Bene<\/span><span class="text-ash"> avanti<\/span>/);
    expect(html).toContain("spine-live");
    // Another Transcript's listening shows nothing here.
    expect(sheet(room(), listening, false)).not.toContain("avanti");
  });

  it("names a call's utterance under way by its side before its speaker is known", () => {
    const listening = at({ phase: "listening", source: "call", transcript: 1, live: [{ key: "mic-1:a", startMs: 0, stable: "", tail: "Pronto", lane: "mic" }] });
    expect(sheet(createTranscript(1), listening, true)).toMatch(/text-ash">Me<\/span>/);
  });

  it("invites a start while nothing is heard", () => {
    expect(sheet(createTranscript(1))).toContain("Nothing heard yet.");
  });
});

describe("the speakers' hues", () => {
  it("are Me in ink, Others in ash, and six for the voices", () => {
    expect(speakerColor({ id: 3, name: "Me", role: "me" })).toBe("var(--ink)");
    expect(speakerColor({ id: 4, name: "Others", role: "others" })).toBe("var(--ash)");
    expect(speakerColor({ id: 7, name: "Speaker 5", role: "voice" })).toBe("var(--speaker-1)");
  });
});

const state = (patch: Partial<TranscribeState> = {}): TranscribeState => {
  const list = startList();
  return { list, active: list.transcripts[0], draft: null, deleted: null, speakers: true, speakersUnavailable: null, holdMic: true, notesLive: false, textBlocked: null, contextTokens: 512, suggestionsOn: true, missedFrom: null, storeError: null, ...patch };
};

describe("the deck", () => {
  const deck = (s: TranscribeState, listening: ListeningState = IDLE) =>
    renderToStaticMarkup(<Deck state={s} listening={listening} actions={actions} speech={{ available: true }} secure here />);

  it("offers Start, or Continue on a Transcript heard in, with what it listens to beside it and the three sources in its menu", () => {
    const html = deck(state());
    expect(html).toContain(">Start</button>");
    // The three sources are its menu's, opened by the button: the microphone until this browser picks another.
    expect(html).toMatch(/aria-haspopup="menu" aria-expanded="false" aria-label="Source: Microphone"/);
    expect(deck(state({ active: room() }))).toContain(">Continue</button>");
  });

  it("keeps the settings to the panel: no context line, no Speakers, no mic held", () => {
    const html = deck(state({ active: { ...room(), context: { languages: { mode: "free", codes: [] }, topic: "Supporto IT", terms: [], other: "", appliedAt: 1 } } }));
    expect(html).not.toContain("Told");
    expect(html).not.toContain("Supporto IT");
    expect(html).not.toContain('role="switch"');
    const call = deck(state(), at({ phase: "listening", source: "call", transcript: 1, held: true, levels: { mic: [0], shared: [0.6] } }));
    expect(call).not.toContain("Hold my mic");
    // The strip still marks the mic held.
    expect(call).toMatch(/Mic<span class="ml-1 text-\[#ff8a4c\]">held<\/span>/);
  });

  it("is the kiln strip while a listening runs", () => {
    const html = deck(state(), at({ phase: "listening", transcript: 1, clockMs: 61_000, levels: { mic: [0, 0.5] } }));
    expect(html).toContain(">Pause<");
    expect(html).toContain(">Stop<");
    expect(html).toContain(">01:01<");
    expect(html).not.toContain(">Start</button>");
  });
});

describe("the hint line", () => {
  it("says what the listening said, else why Start cannot, else headphones before a call", () => {
    const notice = { text: "x", tone: "fault" as const };
    expect(deckHint({ notice, idle: true, blocked: "y", source: "call" })).toBe(notice);
    expect(deckHint({ notice: null, idle: true, blocked: TURNED_OFF, source: "call" })).toEqual({ text: TURNED_OFF, tone: "ash" });
    expect(deckHint({ notice: null, idle: true, blocked: null, source: "call" })).toEqual({ text: HEADPHONES, tone: "ash" });
    expect(deckHint({ notice: null, idle: true, blocked: null, source: "microphone" })).toBeNull();
    expect(deckHint({ notice: null, idle: false, blocked: null, source: "call" })).toBeNull();
  });

  it("blocks a microphone or a call outside a secure context, a file never for that; and each on an unavailable speech model", () => {
    expect(startBlocked("microphone", { secure: false })).toBe(SECURE_CONTEXT);
    expect(startBlocked("file", { secure: false })).toBeNull();
    expect(startBlocked("file", { secure: true, speech: { available: false, reason: "the speech model is turned off (`speech.model` is `off`)" } })).toBe(TURNED_OFF);
  });
});

describe("the strip", () => {
  const strip = (listening: ListeningState) => renderToStaticMarkup(<Strip listening={listening} here actions={actions} />);

  it("shows a file's progress and the time left in place of the levels", () => {
    const html = strip(at({ phase: "listening", file: { name: "talk.mp3", fraction: 0.42, leftMs: 95_000 } }));
    expect(html).toContain('role="progressbar"');
    expect(html).toContain('aria-valuenow="42"');
    expect(html).toContain("about 2 min left");
    expect(html).toContain("Transcribing talk.mp3");
  });

  it("offers Resume while paused, Stop now while finishing, and Cancel while a busy server is waited on", () => {
    expect(strip(at({ phase: "paused" }))).toContain(">Resume<");
    expect(strip(at({ phase: "finishing" }))).toContain(">Stop now<");
    expect(strip(at({ phase: "waiting" }))).toContain(">Cancel<");
    expect(stripStatus(at({ phase: "finishing" }))).toBe("Finishing the last words");
    expect(timeLeft(40_000)).toBe("about 40 s left");
  });
});

describe("the context (Q128)", () => {
  const told = (t: Transcript): Transcript => ({
    ...t,
    context: { languages: { mode: "free", codes: [] }, topic: "Supporto IT", terms: ["PostgreSQL", "VPN"], other: "", appliedAt: new Date(2026, 9, 27, 15, 44).getTime() },
  });
  const card = (t: Transcript, draft: TranscribeState["draft"] = null, live = false) =>
    renderToStaticMarkup(<ContextCard transcript={t} draft={draft} maxTokens={512} live={live} actions={actions} />);

  it("reads, applied, as a record with when it was applied and what it takes of the server's 512 tokens, and offers Edit", () => {
    const html = card(told(room()));
    expect(html).toContain(">Supporto IT<");
    expect(html).toContain(">PostgreSQL<");
    expect(html).toMatch(/Applied at 15:44 · about \d+ of 512 tokens/);
    expect(renderToStaticMarkup(<ContextCard transcript={told(room())} draft={null} maxTokens={1024} live={false} actions={actions} />)).toMatch(/of 1024 tokens/);
    expect(html).toContain(">Edit<");
    expect(html).not.toContain(">Apply<");
  });

  it("is a draft once edited, applied by Apply, from the next utterance while listening", () => {
    const t = told(room());
    const html = card(t, t.context!, true);
    expect(html).toContain('value="Supporto IT"');
    expect(html).toContain('aria-label="Remove PostgreSQL"');
    expect(html).toContain(">Apply<");
    expect(html).toContain("Holds from the next utterance.");
    expect(card(t, t.context!, false)).toContain("A Start applies it too.");
  });

  it("detects the languages by default and forces one on request, saying what each does (Q130)", () => {
    const t = told(room());
    const detect = card(t, { ...t.context!, languages: { mode: "free", codes: ["it", "en"] } });
    expect(detect).toMatch(/type="radio"[^>]*checked=""[^>]*\/><span[^>]*>Detect</);
    expect(detect).toContain('aria-label="Remove English"');
    expect(detect).toContain("The model detects each utterance");
    const force = card(t, { ...t.context!, languages: { mode: "force", codes: ["it"] } });
    expect(force).toContain("Every utterance is transcribed as Italian");
    expect(card({ ...t, context: { ...t.context!, languages: { mode: "force", codes: ["it"] } } })).toContain("Italian (forced)");
  });

  it("offers Add when nothing is told", () => {
    expect(card(room())).toContain(">Add<");
    expect(card(told(room()))).toContain(">Edit<");
  });

  // The owner's word: a default saved from the card, and a question at the first Start, even with the default.
  const saved = { ...told(room()).context!, appliedAt: null };
  const fresh = () => startList().transcripts[0];
  const asked = (s: TranscribeState, withDefault: typeof saved | null = null) =>
    renderToStaticMarkup(<Deck state={s} listening={IDLE} actions={actions} secure here saved={withDefault} checking />);

  it("shows the default on a transcript never started, saves one as the default, or says it is, and clears it", () => {
    const html = renderToStaticMarkup(<ContextCard transcript={fresh()} draft={null} maxTokens={512} live={false} saved={saved} actions={actions} />);
    expect(html).toContain(">Supporto IT<");
    expect(html).toMatch(/Not applied yet · about [1-9]\d* of 512 tokens/);
    expect(html).toContain(">Your default<");
    expect(card(told(room()))).toContain(">Save as default<");
    expect(card(room())).not.toContain("default");
    expect(renderToStaticMarkup(<ContextCard transcript={fresh()} draft={EMPTY_CONTEXT} maxTokens={512} live={false} saved={saved} actions={actions} />)).toContain(">Clear the default<");
    // A transcript once started tells its own alone.
    expect(renderToStaticMarkup(<ContextCard transcript={room()} draft={null} maxTokens={512} live={false} saved={saved} actions={actions} />)).toContain(">Add<");
  });

  it("asks at the first Start what the call is told, over a Start anyway, the card framed meanwhile", () => {
    const other = { ...told(room()), id: 9 };
    const s = state({ list: { transcripts: [fresh(), other], activeId: 1 }, active: fresh() });
    const html = asked(s);
    expect(html).toContain("Nothing told: the model guesses the languages");
    expect(html).toContain(`Use the context of ${titleOf(other)}`);
    expect(html).toContain(">Edit the context<");
    expect(html).toContain(">Start anyway</button>");
    // Not raised yet, or for a transcript once started: Start alone.
    expect(renderToStaticMarkup(<Deck state={s} listening={IDLE} actions={actions} secure here />)).not.toContain("Nothing told");
    expect(asked(state({ active: room() }))).toContain(">Continue</button>");
    expect(renderToStaticMarkup(<ContextCard transcript={fresh()} draft={null} maxTokens={512} live={false} flagged actions={actions} />)).toContain("outline-ember");
  });

  it("asks at the first press, starts at the next, and takes the double click that asked for no answer", () => {
    expect(startPress({ asks: false, checking: false, checkedAt: 0, now: 1 })).toBe("start");
    expect(startPress({ asks: true, checking: false, checkedAt: 0, now: 1 })).toBe("check");
    expect(startPress({ asks: true, checking: true, checkedAt: 1_000, now: 1_000 + ANSWER_AFTER_MS - 1 })).toBeNull();
    expect(startPress({ asks: true, checking: true, checkedAt: 1_000, now: 1_000 + ANSWER_AFTER_MS })).toBe("start");
  });

  it("asks with the default too, and not once a draft is open, which the Start applies", () => {
    const withDefault = asked(state(), saved);
    expect(withDefault).toContain("Your default context: Supporto IT · 2 terms. Is it this call&#x27;s?");
    expect(withDefault).toContain(">Start anyway</button>");
    // The default is the context offered: no other's to take that tells the same.
    const same = asked(state({ list: { transcripts: [fresh(), { ...told(room()), id: 9 }], activeId: 1 } }), saved);
    expect(same).not.toContain("Use the context of");
    const drafting = asked(state({ draft: { ...saved, topic: "Rilascio" } }), saved);
    expect(drafting).toContain("Being edited in the Call tab: Rilascio · 2 terms. Start applies it.");
    expect(drafting).toContain(">Start</button>");
  });
});

describe("the Settings tab", () => {
  const settings = (patch: Partial<TranscribeState> = {}) => renderToStaticMarkup(<Settings state={state(patch)} actions={actions} />);

  it("holds the Speakers toggle, on by default, read at the next Start (Q112)", () => {
    expect(settings()).toMatch(/role="switch" aria-checked="true"[^>]*>.*Speakers<\/span><span[^>]*>The diarizer tells the voices apart/);
  });

  it("strikes the Speakers toggle with the diarizer's reason", () => {
    const html = settings({ speakers: false, speakersUnavailable: "Speakers are turned off on this server (`diarizer.model = off`)." });
    expect(html).toContain("line-through");
    expect(html).toContain('aria-disabled="true"');
    expect(html).toContain("Speakers are turned off on this server (<code");
  });

  it("offers to hold the mic on a call (Q121), on unless turned off", () => {
    expect(settings()).toMatch(/role="switch" aria-checked="true"[^>]*>.*Hold my mic while they speak/);
    expect(settings({ holdMic: false })).toContain("With headphones, nothing comes back");
  });
});

describe("the suggestions (Q132)", () => {
  const idle: AnalystState = { looking: null, nextAt: {}, failed: null };
  const panel = (t: Transcript, looking: AnalystState = idle, patch: Partial<TranscribeState> = {}) =>
    renderToStaticMarkup(<SuggestionsPanel transcript={t} state={state(patch)} actions={actions} looking={looking} />);

  it("wait as cards with what they propose, why, their verb, Always and Dismiss", () => {
    let t = suggest(room(), { kind: "languages", languages: { mode: "force", codes: ["it"] }, why: "2 short utterances came out as Thai and Hindi, among 9 in Italian." }, 1);
    t = suggest(t, { kind: "terms", terms: ["PostgreSQL", "VPN"] }, 2);
    const html = panel(t);
    expect(html).toContain(">Force Italian<");
    expect(html).toContain("came out as Thai and Hindi");
    expect(html).toContain('aria-label="Not PostgreSQL"');
    expect(html).toContain(">Add them<");
    expect(html.match(/>Always</g)).toHaveLength(2);
    expect(html.match(/>Dismiss</g)).toHaveLength(2);
  });

  it("of a fix says where, and that it can be undone", () => {
    const fixed = suggest(room(), { kind: "fix", from: "Apriamo", to: "Apriremo", utterances: [1] }, 1);
    const html = panel(fixed);
    expect(html).toContain("“Apriamo” is Apriremo");
    expect(html).toContain("Misheard at 00:04.");
    expect(html).toContain(">Fix<");
  });

  it("say what each proposes", () => {
    expect(suggestionText({ id: 1, status: "open", kind: "languages", languages: { mode: "free", codes: ["it", "en"] }, why: "" })).toEqual({ headline: "Detect Italian and English", verb: "Apply" });
    expect(suggestionText({ id: 1, status: "open", kind: "title", title: "27 Oct 15:42 · Assistenza" })).toEqual({ headline: "Rename to “27 Oct 15:42 · Assistenza”", verb: "Rename" });
    expect(suggestionText({ id: 1, status: "open", kind: "topic", topic: "IT" }).verb).toBe("Use");
  });

  it("say when the text model is reading, when they are off, and why a look failed; Suggest now waits for a text model", () => {
    const t = room();
    expect(panel(t, { looking: t.id, nextAt: {}, failed: null })).toContain("The text model is reading the transcript.");
    expect(panel(t, { looking: t.id, nextAt: {}, failed: null })).toContain(">Stop<");
    expect(panel(t, idle, { suggestionsOn: false })).toContain("Suggestions while listening are off in Settings.");
    expect(panel(t, { looking: null, nextAt: {}, failed: { transcript: t.id, message: "ignis answered 500" } })).toContain("No suggestions: ignis answered 500");
    expect(panel(t, idle, { textBlocked: "No text model serves: the notes and the suggestions need one. Load it in the Server view." })).toMatch(/disabled=""[^>]*>Suggest now</);
  });
});

describe("asking about the conversation, in its flow (Q139, Q141)", () => {
  const quiet: AskerState = { transcript: null, question: "", untilMs: 0, text: "", writing: false, failed: null };
  const bar = (t: Transcript, asking = quiet, patch: Partial<TranscribeState> = {}) =>
    renderToStaticMarkup(<AskBar transcript={t} state={state(patch)} actions={actions} asking={asking} />);
  /** Two utterances, 4-8 s and 12-15 s. */
  const two = () => addUtterance(room(), { stream: "mic-1", startMs: 12_000, endMs: 15_000, text: "Poi tocca a te.", voice: { kind: "speaker", index: 0 } });
  const sheetAsking = (t: Transcript, asking: AskerState | null = null) =>
    renderToStaticMarkup(
      <Sheet transcript={t} live={[]} phase="idle" call={false} here actions={actions} deleted={null} hits={[]} current={0} rewrite={null} canRewrite readTo={null} seek={0} autoCheck asking={asking} />,
    );

  it("reads What did I miss? from where the user left or picked, its moment on the button, and tints that stretch of the spine", () => {
    const t = two();
    expect(bar(t, quiet, { missedFrom: { kind: "picked", startMs: 12_000 } })).toMatch(/What did I miss\?<span[^>]*>since 00:12<\/span>/);
    expect(bar(t, quiet, { missedFrom: { kind: "away", seenUntilMs: 8_000 } })).toMatch(/What did I miss\?<span[^>]*>since 00:12<\/span>/);
    const tinted = renderToStaticMarkup(
      <Sheet transcript={t} live={[]} phase="idle" call={false} here actions={actions} deleted={null} hits={[]} current={0} rewrite={null} canRewrite readTo={null} seek={0} autoCheck asking={null} readingFrom={1} canAsk />,
    );
    // Each time picks its utterance; from the one read from on, the time and the spine in the ember.
    expect(tinted).toMatch(/aria-label="Ask what was said from 00:04"[^>]*class="[^"]*text-ash/);
    expect(tinted).toMatch(/aria-label="Ask what was said from 00:12"[^>]*class="[^"]*text-ember /);
    expect(tinted.match(/border-ember\/45/g)).toHaveLength(1);
    expect(sheetAsking(t)).not.toContain("border-ember/45");
    // With no text model to ask, a time is a time.
    expect(sheetAsking(t)).not.toContain("Ask what was said from");
  });

  it("gives the field, Ask and the quick questions one height", () => {
    const html = bar(room());
    expect(html).toMatch(/aria-label="Question about the conversation"[^>]*class="h-8 /);
    expect(html).toMatch(/type="submit"[^>]*class="cut h-8 /);
    expect(html.match(/class="flex h-8 shrink-0/g)).toHaveLength(3);
  });

  it("offers the quick questions and the field under the sheet, once something is heard", () => {
    const html = bar(room());
    expect(html).toContain('aria-label="Ask about the conversation"');
    expect(html).toMatch(/What did I miss\?<span[^>]*>last 5 min<\/span>/);
    expect(html).toContain(">Decisions<");
    expect(html).toContain(">Who does what<");
    expect(html).toContain(">Ask<");
    expect(bar(createTranscript(4))).toBe("");
  });

  it("stands each answer on the spine after the last utterance it read, the oldest first, its moments as links", () => {
    const t = {
      ...two(),
      asked: [
        { id: 3, question: "E alla fine?", answer: "Tocca a te [00:12].", at: 3, untilMs: 15_000 },
        { id: 2, question: "Chi ha finito?", answer: "Speaker 1 [00:04].", at: 2, untilMs: 8_000 },
        { id: 1, question: "Prima?", answer: "Niente.", at: 1, untilMs: 8_000 },
      ],
    };
    const html = sheetAsking(t);
    const [apriamo, prima, chi, poi, fine] = ["Apriamo.", "Prima?", "Chi ha finito?", "Poi tocca a te.", "E alla fine?"].map((s) => html.indexOf(s));
    expect(apriamo).toBeLessThan(prima);
    expect(prima).toBeLessThan(chi);
    expect(chi).toBeLessThan(poi);
    expect(poi).toBeLessThan(fine);
    // The moment it read up to, in the time column.
    expect(html).toMatch(/title="It read the conversation up to 00:08"[^>]*>00:08</);
    expect(html).toContain('href="#t=00:04"');
    expect(html).toContain('aria-label="Forget: Chi ha finito?"');
    // One kept before answers had a place stands at the foot.
    const old = sheetAsking({ ...two(), asked: [{ id: 1, question: "Prima?", answer: "Niente.", at: 1 }] });
    expect(old.indexOf("Prima?")).toBeGreaterThan(old.indexOf("Poi tocca a te."));
  });

  it("streams an answer at its place under the heat, the bar's button a Stop, and says why none came", () => {
    const t = two();
    const asking: AskerState = { transcript: t.id, question: "Chi?", untilMs: 15_000, text: "Speaker 1 [00:04]", writing: true, failed: null };
    const html = sheetAsking(t, asking);
    expect(html).toMatch(/aria-label="Answer to: Chi\?" aria-busy="true"/);
    expect(html).toContain("spine-live");
    expect(html.indexOf("Chi?")).toBeGreaterThan(html.indexOf("Poi tocca a te."));
    expect(sheetAsking(t, { ...asking, text: "" })).toContain("Reading the conversation up to 00:15.");
    expect(bar(t, asking)).toContain(">Stop<");
    expect(bar(t, { ...quiet, transcript: t.id, failed: "ignis answered 500" })).toContain("No answer: ignis answered 500");
    // No text model: the reason in place of the quick questions, nothing to press.
    const blocked = bar(t, quiet, { textBlocked: "No text model serves: the notes and the suggestions need one. Load it in the Server view." });
    expect(blocked).toContain("No text model serves");
    expect(blocked).not.toContain("Decisions");
    expect(blocked).toMatch(/disabled=""[^>]*>Ask</);
    // One question at a time for the page: another Transcript's being answered, this one waits and says so.
    const elsewhere = bar(t, { ...asking, transcript: 99 });
    expect(elsewhere).toContain("A question about another transcript is being answered");
    expect(elsewhere).toMatch(/disabled=""[^>]*>Ask</);
    expect(elsewhere).toMatch(/disabled=""[^>]*>Decisions</);
  });
});

describe("the notes", () => {
  const idle: NotesWriterState = { writing: null, form: "running", text: "", failed: null };
  const written = (t: Transcript): Transcript => ({
    ...t,
    notes: {
      kind: "work",
      lines: [{ id: "N1", section: "Now", text: "Si apre.", author: "model" }],
      removed: [],
      nextId: 2,
      writtenAs: "work",
      covered: 1,
      writtenAt: new Date(2026, 9, 27, 16, 5).getTime(),
      fromMs: null,
    },
  });
  const panel = (t: Transcript, writer: NotesWriterState = idle, patch: Partial<TranscribeState> = {}, marked?: Marked, editing?: Editing) =>
    renderToStaticMarkup(<NotesPanel transcript={t} state={state(patch)} actions={actions} writer={writer} marked={marked} editing={editing} />);
  /** A line as the notes draw it: its bullet's author and its mark, then its text, a click away from editing. */
  const drawnLine = (text: string, author = "model", mark = "") =>
    `<li data-author="${author}"${mark ? ` data-mark="${mark}"` : ""}><span tabindex="0" title="${author === "user" ? "Your line: updates leave it as it is. Click to edit it" : "Edit the line"}" class="note-text">${text}</span>`;

  it("offers the four kinds, Write notes, and says nothing goes to the model before a press (Q124)", () => {
    const html = panel(room());
    expect(html).toMatch(/role="radio" aria-checked="true"[^>]*><span[^>]*>General<\/span>/);
    expect(html).toContain("Game session");
    expect(html).toContain(">Write notes<");
    expect(html).toContain("Nothing goes to it until you press Write notes or Write the recap, or turn on Keep updating.");
    expect(html).toMatch(/role="switch" aria-checked="false"[^>]*>.*Keep updating while listening/);
    expect(html).toContain("About once a minute while something new is said, and the recap when the listening ends.");
  });

  it("shows the notes in the kind's live sections, when they were updated, and offers to update them", () => {
    const t = addUtterance(written(room()), { stream: "mic-1", startMs: 9_000, endMs: 9_500, text: "Poi?", voice: null });
    const html = panel(t);
    expect(html).toContain(`<h2>Now</h2><ul>${drawnLine("Si apre.")}`);
    expect(html).toContain("<h2>Commitments heard</h2><p>—</p>");
    expect(html).toContain("Updated at 16:05. 1 more since.");
    expect(html).toContain(">Update notes<");
    expect(html).toMatch(/role="radio" aria-checked="true"[^>]*><span[^>]*>Work call<\/span>/);
    // Kept up while listening, the line says only when.
    const live = panel(t, idle, { notesLive: true });
    expect(live).toContain("Updated at 16:05.");
    expect(live).not.toContain("more since");
  });

  it("updates the notes in place under the heat, the button a Stop", () => {
    const t = written(room());
    const html = panel(t, { writing: t.id, form: "running", text: "", failed: null });
    expect(html).toContain("spine-live");
    expect(html).toMatch(/aria-label="Notes" aria-busy="true"/);
    // Not dimmed: the lines stay readable while they change.
    expect(html).toContain('<div class="notes">');
    expect(html).toContain(drawnLine("Si apre."));
    expect(html).toContain("The model is reading what was said since.");
    expect(html).not.toContain("Updated at");
    expect(panel(room(), { writing: room().id, form: "running", text: "", failed: null })).toContain("The model is reading the transcript.");
    expect(notesButton(t, { writing: t.id, form: "running", text: "", failed: null }, null)).toEqual({ label: "Stop", disabled: false, stops: true });
  });

  it("marks the lines an update changed: ember on added and changed ones, a removed one struck and dimmed, none when the time is up", () => {
    const t = written(room());
    const first = (t.notes?.lines ?? [])[0];
    const now = 5_000_000;
    const gone = { ...first, id: "N9", text: "Una riga tolta." };
    const marks: NoteMark[] = [
      { id: first.id, kind: "fresh", until: now + 1_000, line: first, index: 0 },
      { id: "N9", kind: "gone", until: now + 1_000, line: gone, index: 1 },
    ];
    const html = panel(t, idle, {}, { marks, now });
    expect(html).toContain(drawnLine(first.text, "model", "fresh"));
    expect(html).toContain('<li data-author="model" data-mark="gone"><del>Una riga tolta.</del></li>');
    // Their time is up: the notes are as they were written, with no mark.
    const spent = panel(t, idle, {}, { marks, now: now + 1_000 });
    expect(spent).not.toContain("data-mark");
    expect(spent).not.toContain("Una riga tolta.");
    expect(spent).toBe(panel(t, idle, {}, { marks: [], now }));
  });

  /** Notes with a line of the model's past the hour and one of the user's. */
  const mixed = (t: Transcript): Transcript => ({
    ...written(t),
    notes: {
      ...written(t).notes!,
      lines: [
        { id: "N1", section: "Now", text: "Si apre [64:09].", author: "model" },
        { id: "N2", section: "People", text: "Ada: acquisti.", author: "user" },
      ],
      nextId: 3,
    },
  });

  it("lets each line be edited with a click and removed with ×, adds a line at each section's foot, and marks the user's lines", () => {
    const html = panel(mixed(room()));
    expect(html).toContain(`${drawnLine('Si apre <a href="#t=1:04:09" target="_blank" rel="noreferrer noopener">1:04:09</a>.')}<button type="button" aria-label="Remove line N1" title="Remove the line" class="note-remove">`);
    expect(html).toContain(`${drawnLine("Ada: acquisti.", "user")}<span class="sr-only"> (your line)</span><button type="button" aria-label="Remove line N2"`);
    expect([...html.matchAll(/aria-label="Add a line to ([^"]+)"/g)].map((m) => m[1])).toEqual(["Now", "People", "Figures &amp; dates", "Asked of me", "Commitments heard"]);
    expect(html).not.toContain("<textarea");
  });

  it("edits a line in place, past the hour as the view shows it, and adds one in a field at its section's foot", () => {
    const t = mixed(room());
    const [model, user] = t.notes!.lines;
    const editing = panel(t, idle, {}, undefined, { form: "running", line: model });
    expect(editing).toMatch(/<li data-author="model"><span class="note-editor"><textarea aria-label="Line N1 of Now" rows="1"[^>]*>Si apre \[1:04:09\]\.<\/textarea>/);
    expect(editing).toContain("Enter saves, Escape cancels; emptied, the line goes.");
    expect(editing).not.toContain('aria-label="Remove line N1"');
    expect(panel(t, idle, {}, undefined, { form: "running", line: user })).toMatch(/<li data-author="user"><span class="note-editor"><textarea aria-label="Line N2 of People"[^>]*>Ada: acquisti\.<\/textarea>/);
    const adding = panel(t, idle, {}, undefined, { form: "running", section: "People" });
    expect(adding).toMatch(/<h2>People<\/h2><ul>.*<\/ul><span class="note-editor"><textarea aria-label="New line in People" rows="1"[^>]*><\/textarea><span[^>]*>Enter adds it, Escape cancels\.<\/span>/);
    expect(adding).not.toContain('aria-label="Add a line to People"');
    // An edit in the recap opens nothing in the running notes.
    expect(panel(t, idle, {}, undefined, { form: "recap", line: model })).not.toContain("<textarea");
  });

  it("keeps a line being edited in view when an update removed it meanwhile: the user's edit wins", () => {
    const t = mixed(room());
    const typing = { id: "N7", section: "People", text: "Bea: fornitore.", author: "model" as const };
    expect(panel(t, idle, {}, undefined, { form: "running", line: typing })).toMatch(
      /<h2>People<\/h2><ul><li data-author="user">.*?<\/li><li data-author="model"><span class="note-editor"><textarea aria-label="Line N7 of People"[^>]*>Bea: fornitore\.<\/textarea>/,
    );
  });

  it("offers Rewrite from scratch in a menu, once the model wrote a line and while nothing is under way", () => {
    const t = written(room());
    const html = panel(t);
    expect(html).toMatch(/<summary[^>]*>.*More.*<\/summary>/s);
    expect(html).toMatch(/<button type="button"[^>]*>.*Rewrite from scratch.*<\/button>/s);
    expect(html).toContain("Drops the model&#x27;s lines and writes the notes again from the transcript.");
    expect(rewriteButton(t, idle, null)).toEqual({ disabled: false });
    expect(rewriteButton(room(), idle, null)).toEqual({ disabled: true });
    expect(rewriteButton(t, { ...idle, writing: 9 }, null)).toEqual({ disabled: true });
    expect(rewriteButton(t, idle, "No text model serves: the notes and the suggestions need one. Load it in the Server view.")).toEqual({ disabled: true });
  });

  it("cannot write with no text model, nothing heard, or another Transcript's notes under way, and says why it failed", () => {
    const t = room();
    expect(notesButton(t, idle, "No text model serves: the notes and the suggestions need one. Load it in the Server view.").disabled).toBe(true);
    expect(notesButton(createTranscript(2), idle, null).disabled).toBe(true);
    expect(notesButton(t, { writing: 9, form: "running", text: "", failed: null }, null).disabled).toBe(true);
    expect(panel(t, idle, { textBlocked: "The image model is loaded: the notes and the suggestions need a text model. Switch in the Server view." })).toContain("The image model is loaded");
    expect(panel(t, { ...idle, failed: { transcript: t.id, form: "running", message: "The model wrote nothing." } })).toContain("The notes were not written: The model wrote nothing.");
  });

  const recapped = (t: Transcript): Transcript => ({
    ...written(t),
    recap: {
      kind: "work",
      lines: [
        { id: "N1", section: "Summary", text: "Il budget del lancio.", author: "model" },
        { id: "N2", section: "Figures & dates", text: "40k: il tetto [00:01]", author: "user" },
      ],
      removed: [],
      nextId: 3,
      writtenAs: "work",
      covered: 1,
      writtenAt: new Date(2026, 9, 27, 16, 20).getTime(),
      fromMs: null,
    },
  });

  /** Where the running notes fold under the recap. */
  const foldAt = (html: string) => html.lastIndexOf("<details", html.indexOf(">Running notes</summary>"));

  it("offers Write the recap at any moment, and shows no recap and no fold before one is written", () => {
    const html = panel(written(room()));
    expect(html).toContain(">Write the recap<");
    expect(html).not.toContain("Running notes</summary>");
    expect(html).toContain(">Copy notes<");
    expect(recapButton(room(), idle, null)).toEqual({ label: "Write the recap", disabled: false, stops: false });
  });

  it("shows the recap on top once it exists, in its sections, the running notes folded under it", () => {
    const html = panel(recapped(room()));
    const recap = html.indexOf('aria-label="Recap"');
    const fold = foldAt(html);
    expect(recap).toBeGreaterThan(0);
    expect(fold).toBeGreaterThan(recap);
    expect(html.slice(recap, fold)).toContain(`<h2>Figures &amp; dates</h2><ul>${drawnLine('40k: il tetto <a href="#t=00:01" target="_blank" rel="noreferrer noopener">00:01</a>', "user")}`);
    expect(html.slice(recap, fold)).toContain("Written at 16:20 from 1 utterance.");
    expect(html).toMatch(/<details[^>]*><summary[^>]*>Running notes<\/summary>/);
    expect(html).not.toMatch(/<details[^>]*open/);
    expect(html.slice(fold)).toContain("Si apre.");
    expect(html).toContain(">Copy recap<");
  });

  it("lets the recap's lines be edited, removed and added the same way, in the recap's sections", () => {
    const t = recapped(room());
    const html = panel(t);
    const recap = html.slice(html.indexOf('aria-label="Recap"'), foldAt(html));
    expect(recap).toContain(`${drawnLine("Il budget del lancio.")}<button type="button" aria-label="Remove line N1"`);
    expect([...recap.matchAll(/aria-label="Add a line to ([^"]+)"/g)].map((m) => m[1])).toEqual(["Summary", "Topics", "People", "Figures &amp; dates", "Decisions", "Action items", "Open questions"]);
    const editing = panel(t, idle, {}, undefined, { form: "recap", line: t.recap!.lines[0] });
    expect(editing.slice(editing.indexOf('aria-label="Recap"'), foldAt(editing))).toContain('<textarea aria-label="Line N1 of Summary"');
    // The running notes' line of the same id is not the one edited.
    expect(editing.slice(foldAt(editing))).not.toContain("<textarea");
  });

  it("streams the recap on top while it is written, its button a Stop, the notes' button waiting", () => {
    const t = written(room());
    const writing: NotesWriterState = { writing: t.id, form: "recap", text: "## Summary\n- Il budget", failed: null };
    const html = panel(t, writing);
    const recap = html.indexOf('aria-label="Recap"');
    expect(recap).toBeGreaterThan(0);
    expect(html.slice(recap, foldAt(html))).toContain("spine-live");
    expect(html.slice(recap, foldAt(html))).toContain("<li>Il budget</li>");
    // A recap on its way is not edited.
    expect(html.slice(recap, foldAt(html))).not.toContain("Add a line");
    expect(recapButton(t, writing, null)).toEqual({ label: "Stop", disabled: false, stops: true });
    expect(notesButton(t, writing, null)).toMatchObject({ label: "Update notes", disabled: true, stops: false });
    expect(recapButton(t, { ...writing, form: "running" }, null)).toMatchObject({ disabled: true, stops: false });
    expect(recapButton(createTranscript(2), idle, null).disabled).toBe(true);
    expect(panel(t, { ...idle, failed: { transcript: t.id, form: "recap", message: "ignis answered 500" } })).toContain("The recap was not written: ignis answered 500");
  });
});

const idleHandle = { getState: () => IDLE, subscribe: () => () => {} };
const assistants = {
  notes: { getState: (): NotesWriterState => ({ writing: null, form: "running", text: "", failed: null }), subscribe: () => () => {} },
  analyst: { getState: (): AnalystState => ({ looking: null, nextAt: {}, failed: null }), subscribe: () => () => {} },
  rewriter: { getState: () => ({ transcript: null, utterance: null, writing: false, proposal: null, failed: null }), subscribe: () => () => {} },
  asker: { getState: () => ({ transcript: null, question: "", untilMs: 0, text: "", writing: false, failed: null }), subscribe: () => () => {} },
};

describe("the view", () => {
  it("heads the transcript with its exports, and opens the right panel on the Call tab, beside Notes and Settings (Q129)", () => {
    const html = renderToStaticMarkup(
      <TranscribeView state={state({ active: room() })} actions={actions} listening={idleHandle} assistants={assistants} drawer={null} onDrawer={none} onOpenInChat={none} />,
    );
    expect(html.indexOf('aria-label="Transcript title"')).toBeLessThan(html.indexOf(">Download"));
    expect(html).toContain(">Open in chat<");
    expect(html).toContain(">Send to workspace<");
    // The Download menu opens downward: its chevron points down while it is closed.
    expect(html).toContain("rotate-180 transition-transform group-open:rotate-0");
    // It offers the four files, then PDF, whose dialog asks what the PDF carries.
    const menu = html.slice(html.indexOf(">Download"), html.indexOf(">Open in chat<"));
    expect([...menu.matchAll(/<button[^>]*>([^<]+)<\/button>/g)].map((m) => m[1])).toEqual([".txt", ".md", ".srt", ".vtt", "PDF…"]);
    expect(html).not.toContain('aria-label="Download as PDF"');
    expect(html).toMatch(/role="tab" id="transcribe-tab-call" aria-selected="true"/);
    expect(html).toMatch(/role="tab" id="transcribe-tab-notes" aria-selected="false"/);
    expect(html).toMatch(/role="tab" id="transcribe-tab-settings" aria-selected="false"/);
    expect(html).toContain(">Context<");
    expect(html).toContain('id="transcribe-speakers"');
    expect(html).not.toContain("Utterance ends after");
    expect(html).toContain(">Suggest now<");
    expect(html).not.toContain("suggestions waiting");
  });

  it("rings the bell with the open Transcript's suggestions waiting (Q132)", () => {
    const t = suggest(suggest(room(), { kind: "topic", topic: "Supporto IT" }, 1), { kind: "terms", terms: ["PostgreSQL"] }, 2);
    const html = renderToStaticMarkup(
      <TranscribeView state={state({ active: t })} actions={actions} listening={idleHandle} assistants={assistants} drawer={null} onDrawer={none} onOpenInChat={none} />,
    );
    expect(html).toContain('aria-label="2 suggestions"');
  });
});

describe("the search (Q136)", () => {
  const field = (query: string, count: number, at = 0) => renderToStaticMarkup(<SearchField query={query} count={count} at={at} onQuery={none} onStep={none} />);

  it("says where it stands among the hits, offers the previous and the next, and says when there are none", () => {
    expect(field("budget", 12, 2)).toContain(">3 of 12<");
    expect(field("budget", 12, 2)).toContain('aria-label="Next match"');
    expect(field("budget", 0)).toContain(">None<");
    expect(field("", 0)).not.toContain("of");
  });

  it("names a listed Transcript's matches, or that its title holds the query", () => {
    expect(listMatch(1)).toBe("1 match");
    expect(listMatch(4)).toBe("4 matches");
    expect(listMatch(0)).toBe("In the title");
  });
});

describe("the Transcripts list", () => {
  it("says how many utterances each holds, or that it is the one listening", () => {
    const t = room();
    expect(transcriptSummary(t, IDLE)).toBe("1 utterance");
    expect(transcriptSummary(createTranscript(2), IDLE)).toBe("Nothing heard yet");
    expect(transcriptSummary(t, at({ phase: "paused", transcript: 1 }))).toBe("Paused");
  });

  it("names a download after the title, and tells a Transcript heard over more than one session", () => {
    expect(fileStem("25 Oct 15:42 · Call")).toBe("25-Oct-15-42-Call");
    const t = room();
    expect(sessionsWithSpeakers(t)).toBe(1);
    expect(sessionsWithSpeakers(addUtterance(t, { stream: "mic-2", startMs: 9_000, endMs: 9_500, text: "Eccomi.", voice: { kind: "speaker", index: 0 } }))).toBe(2);
  });
});

describe("the context images (#431)", () => {
  const SLIDE = "e".repeat(64);
  const shown = (t: Transcript) =>
    renderToStaticMarkup(
      <StaticPictures urls={{ [`thumb:${SLIDE}`]: "blob:slide" }}>
        <Sheet transcript={t} live={[]} phase="idle" call={false} here actions={actions} deleted={null} hits={[]} current={0} rewrite={null} canRewrite readTo={null} seek={0} autoCheck asking={null} />
      </StaticPictures>,
    );

  it("stand on the spine at their moments, the picture and its size beside the time", () => {
    let t = addContextImage(room(), { image: SLIDE, atMs: 0, width: 1280, height: 720 }, 1);
    t = addContextImage(t, { image: SLIDE, atMs: 9_000, width: 640, height: 360 }, 2);
    const html = shown(t);
    const start = html.indexOf('aria-label="Context image at 00:00"');
    const said = html.indexOf(">00:04<");
    const later = html.indexOf('aria-label="Context image at 00:09"');
    expect(start).toBeGreaterThan(-1);
    expect(said).toBeGreaterThan(start);
    expect(later).toBeGreaterThan(said);
    expect(html).toContain('src="blob:slide"');
    expect(html).toContain("Context image, 1280×720");
  });

  it("stand in place of the empty sheet when a picture came before anyone spoke", () => {
    const html = shown(addContextImage(createTranscript(1), { image: SLIDE, atMs: 0, width: 800, height: 600 }, 1));
    expect(html).toContain('aria-label="Context image at 00:00"');
    expect(html).not.toContain("Nothing heard yet");
  });

  const card = (t: Transcript, sharing: boolean, error: string | null = null) =>
    renderToStaticMarkup(
      <Call transcript={t} state={state({ active: t })} actions={actions} live={sharing} saved={null} flagged={false} sharing={sharing} adding={{ open: none, capture: none, capturing: false, error }} />,
    );

  it("are added in the Call tab from a file or a paste, and by Capture only while a call shares a picture", () => {
    const idle = card(room(), false);
    expect(idle).toContain(">Images</h2>");
    expect(idle).toContain(">From a file</button>");
    expect(idle).toContain('accept="image/*"');
    expect(idle).toContain("or paste one");
    expect(idle).not.toContain(">Capture</button>");
    expect(idle).toContain("On a call, Capture takes a frame of what it shares.");
    const sharing = card(room(), true);
    expect(sharing).toContain(">Capture</button>");
    expect(sharing).not.toContain("On a call, Capture");
  });

  it("say why a picture was refused, and how many the spine holds", () => {
    expect(card(room(), false, "slides.pdf is not an image.")).toContain("slides.pdf is not an image.");
    const two = addContextImage(addContextImage(room(), { image: SLIDE, atMs: 0, width: 1, height: 1 }, 1), { image: SLIDE, atMs: 5_000, width: 1, height: 1 }, 2);
    expect(card(two, false)).toContain("2 images on the spine.");
  });
});
