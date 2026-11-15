import { useEffect, useRef, useState } from "react";
import { canSend, loadAwaited, type ModelState } from "../api/model.ts";
import { conversationTurns, type Exchange, type Settings, thinkingBudgetOf, type ToolExtras } from "../api/request.ts";
import type { ToolCall } from "../api/sse.ts";
import { streamChat } from "../api/stream.ts";
import { switchAndWait } from "../api/switching.ts";
import type { NextModel } from "../conversation/nextModel.ts";
import { type PromptImage, redrawn } from "../conversation/images.ts";
import { CUT_NOT_RUN, endedAtTokenLimit } from "../metrics/CutAtLimit.tsx";
import { computeFigures, type Figures } from "../metrics/figures.ts";
import { mergeLoaded } from "../store/docs.ts";
import type { ImageId, ImageStore } from "../store/imageStore.ts";
import { pageStore } from "../store/index.ts";
import { keepPlain } from "../store/pictures.ts";
import { useStoredSessions } from "../store/useStoredSessions.ts";
import { createFrameBatch, type FrameBatch } from "./frameBatch.ts";
import {
  addLogRow,
  addMessages,
  bornInto,
  clearAttachments,
  createSession,
  editMessage,
  failedReply,
  forkSession,
  type LogRow,
  type Message,
  moveWorkspace,
  openSession,
  pickHeld,
  removeAttachment,
  removeSession,
  type SessionList,
  titleOf,
  truncateFrom,
  type TurnSwitch,
  UNTITLED,
  updateMessage,
} from "../sessions/sessions.ts";
import { CHAT_SESSIONS, freedBy, keepPictures, picturesOf, restoreChats, toStored, withPictures, worthKeeping } from "../sessions/stored.ts";
import type { PlaygroundSettings } from "../settings/defaults.ts";
import { type AgentRun, parseAgentCall, runAgents, toolResult } from "../tools/agents/agents.ts";
import { askToolResult, parseAskCall } from "../tools/ask/ask.ts";
import { unknownCall } from "../tools/errors.ts";
import { type ImageRun, imageToolResult, parseImageCall, runImages } from "../tools/image/image.ts";
import { type LocalContext, type LocalRun, localToolImages, localToolResult, runLocalCalls, shownRun, startedRun } from "../tools/local/local.ts";
import { agentExtras, routeCall, toolExtras, type ToolsState } from "../tools/index.ts";
import { browserMemory } from "../tools/local/memory.ts";
import { agentNotice, turnNotice } from "../tools/notice.ts";
import { getTavilyKey } from "../tools/web/tavilyKey.ts";
import { parseWebCall, runWeb, type WebRun, webToolResult } from "../tools/web/web.ts";
import { type AttachedFile, attachedChip, fromLegacy, readAttached, type Taken, takeTurnFiles, turnAdds, withAttached } from "../workspace/attach.ts";
import { noticeFiles } from "../workspace/notice.ts";
import type { ChatNames } from "../workspace/panel.ts";
import { boxPictures, madeIn } from "../workspace/pictures.ts";
import { seenIn } from "../workspace/tools.ts";
import { useVision } from "../workspace/vision.ts";
import { type ChatBinding, pageWorkspaces, type Workspaces } from "../workspace/workspaces.ts";

// The conversation loop: the sessions, the replies streaming into them, and
// the tools a reply calls. The sessions are kept in the page's store and read
// back on the next load (sessions/stored.ts, store/useStoredSessions.ts). A
// turn does not wait for them: they come back with ids past every one the page
// has handed out, below what it was doing meanwhile, and a store that never
// answers leaves the chat as it was before it kept anything.
//
// A session streams one reply at a time — its history is a line, and two
// replies writing into it would fork it. Whether *other* sessions may stream
// meanwhile is the caller's choice (`parallel`): off, the page runs one turn
// and every Send waits for it; on, each session runs its own. On the socket
// (GitHub #283) each session is a stream of its own, and each of its agents
// too, all on one connection; over HTTP the browser's stream budget (GitHub
// #220) queues what the connection cannot carry, so parallel sessions share
// the same five streams on localhost.

/**
 * A turn may start in `sessionId`: nothing streams there, and — unless
 * sessions run in parallel — nothing streams anywhere else either.
 */
export function canStartTurn(streaming: ReadonlySet<number>, sessionId: number, parallel: boolean): boolean {
  return parallel ? !streaming.has(sessionId) : streaming.size === 0;
}

/** A session's stream on the socket: its turns continue one another there, and its agents' streams are named under it. */
export const sessionStream = (sessionId: number) => `session-${sessionId}`;

/**
 * Why a reply's tool calls do not run, each call's result, or `undefined` when they run: a reply cut at the token
 * limit is incomplete, none of its calls run, the closed ones included (GitHub #583); a stopped turn runs nothing
 * more; and a turn calls tools at most `maxRounds` times in a row.
 */
export function callsNotRun(figures: Figures | undefined, aborted: boolean, round: number, maxRounds: number): string | undefined {
  if (endedAtTokenLimit(figures)) return CUT_NOT_RUN;
  if (aborted) return "Stopped before the call ran.";
  if (round >= maxRounds) return `Not run: a turn can call tools at most ${maxRounds} times in a row.`;
  return undefined;
}

/**
 * What a turn brings beside its prompt: the box's files and pictures for the workspace, the names a resent prompt
 * already attached there, the message a resend or a regenerate replaces, and whom to tell once the prompt is sent,
 * or not sent because the store refused its files, with what the page says of a refusal.
 */
type TurnInput = {
  add: AttachedFile[];
  pictures: AttachedFile[];
  named: string[];
  replaces?: number;
  sent?: (sent: boolean, said?: string) => void;
};

const NO_FILES: TurnInput = { add: [], pictures: [], named: [] };

/**
 * A message as the turns of a request read it: a prompt with the files it attached named after its words, and a
 * prompt sent before the turn notice with its moment as its notice.
 */
export const exchangeOf = (m: Message): Exchange => ({
  role: m.role,
  content: withAttached(m.content, m.attached),
  images: m.images,
  reasoning: m.reasoning,
  failed: failedReply(m),
  toolCalls: m.toolCalls,
  toolCallId: m.toolCallId,
  notice: m.notice ?? m.dateTime,
});

export function useConversation({
  model,
  next,
  refreshModel,
  settings,
  tools,
  parallel,
  store = pageStore(),
  workspaces = pageWorkspaces(),
  live,
}: {
  model: ModelState;
  /** The model the next turn goes to, and whether it switches there first (the composer's model picker). */
  next: NextModel;
  /** Reads which model is loaded now. */
  refreshModel: () => void;
  settings: PlaygroundSettings;
  tools: ToolsState;
  /** Sessions other than the one streaming may start their own turn. */
  parallel: boolean;
  store?: ImageStore;
  /** Where the chats' workspaces are kept (spec #170). */
  workspaces?: Workspaces;
  /**
   * The store's pictures the Image view's sessions name now, or null while it has not read them: deleting a chat
   * session keeps them, and while null deletes no picture.
   */
  live?: () => Iterable<ImageId> | null;
}) {
  // Session and message ids. The counter lives in a ref, not in a module
  // variable: a hot reload keeps component state but re-runs the module, and
  // a restarted count would hand out ids still on screen (duplicate keys).
  const ids = useRef(2);
  const newId = () => ids.current++;
  const [list, commitList] = useState<SessionList>(() => ({ sessions: [createSession(1)], activeId: 1 }));
  // What streams in — text, reasoning, calls, the tools as they run — waits
  // for the next frame and lands as one update for every session
  // (`frameBatch.ts`); any other change lands at once, after what waited.
  const frames = useRef<FrameBatch<SessionList> | null>(null);
  frames.current ??= createFrameBatch(commitList);
  const setList = (update: (l: SessionList) => SessionList) => {
    frames.current?.flush();
    commitList(update);
  };
  const setSoon = (key: string, update: (l: SessionList) => SessionList) => frames.current?.set(key, update);
  const listNow = useRef(list);
  listNow.current = list;
  // Whether the model the chat talks to has vision, read again after each switch, and asked when a tool reads a picture.
  const vision = useVision(model);
  const storage = useStoredSessions({
    store,
    collection: CHAT_SESSIONS,
    view: "chat",
    items: list.sessions,
    activeId: list.activeId,
    toStored,
    worthKeeping,
    adopt: async (values, activeKey) => {
      const restored = restoreChats(values, ids.current);
      ids.current = restored.nextId;
      // A picture the store cannot read leaves its image with the redraw alone, not the session unread.
      const loaded = await Promise.all(restored.sessions.map((s) => withPictures(s, (id) => store.getImage(id).catch(() => null))));
      const reopened = [...restored.stamps].find(([, stamp]) => stamp.key === activeKey)?.[0];
      setList((l) => {
        const merged = mergeLoaded(l.sessions, l.activeId, loaded, reopened, worthKeeping);
        return { sessions: merged.items, activeId: merged.activeId };
      });
      return { loaded, stamps: restored.stamps };
    },
    onRemoved: (gone, left) => {
      // What the workspaces hold stays too: a prompt's picture is one of its workspace's files (#394).
      void workspaces.pictures().then(
        (held) => {
          const elsewhere = live ? live() : [];
          for (const id of freedBy(gone, left, elsewhere === null ? null : [...elsewhere, ...held])) void store.deleteImage(id).catch(() => {});
        },
        () => {},
      );
    },
  });
  // The sessions a reply is streaming into, for the render.
  const [streaming, setStreaming] = useState<ReadonlySet<number>>(() => new Set());
  // The same, as the turns themselves see it: a ref settles two Sends in one
  // tick, which the state — one render behind — would let both through.
  const controllers = useRef(new Map<number, AbortController>());
  const running = () => new Set(controllers.current.keys());
  const [attachError, setAttachError] = useState<string | null>(null);
  // The files the next prompt attaches. Like its images, they belong to the box, not to a session: sending adds
  // them to the chat's workspace, and the box is empty again.
  const [attached, setAttached] = useState<AttachedFile[]>([]);
  // The workspace each session's turns made it, before the list shows the session bound to it.
  const born = useRef(new Map<number, string>());
  // Questions waiting for the user, by `messageId:callId`: each settles with the answer.
  const waiting = useRef(new Map<string, (answer: string | null) => void>());
  // Whether the reader is at the bottom of the conversation: only then do
  // new tokens pull the view down.
  const following = useRef(true);

  const active = list.sessions.find((s) => s.id === list.activeId) ?? list.sessions[0];
  /** Anything at all is streaming: what the header's pulse and the model dot show. */
  const busy = streaming.size > 0;
  const streamingHere = streaming.has(active.id);
  /** A model can take the next turn: the one loaded, or the one the turn switches to first. */
  const canTake = canSend(model) && next.id !== null;
  /** A new turn can start in the session on screen, and a model can take it. */
  const canRun = canStartTurn(streaming, active.id, parallel) && canTake;
  /** Send waits: this session streams, or another one does and turns are not parallel. */
  const sendBlocked = !canStartTurn(streaming, active.id, parallel);

  // A turn sent while the model was unloaded loads it (spec model-switch/02). The read at the turn's start comes
  // back `unloaded` before its request reaches the server, so the page reads again each second until it sees the
  // load — which it then follows as it follows a switch — and the header says what the wait is.
  const awaitingLoad = loadAwaited(model, busy);
  useEffect(() => {
    if (!awaitingLoad) return;
    const timer = setInterval(refreshModel, 1000);
    return () => clearInterval(timer);
  }, [awaitingLoad, refreshModel]);

  function selectSession(id: number) {
    following.current = true;
    setList((l) => ({ ...l, activeId: id }));
  }

  function newSession() {
    following.current = true;
    const id = newId();
    setList((l) => openSession(l, id));
  }

  /**
   * The session's hold on its workspace, as the chat its store key names: its
   * own, or one born with the first file and named after the chat (`title`
   * when it is not named yet). The chat's reader has seen what `history`
   * read or made of the files, the turn's model having that in its context.
   */
  function chatBinding(sessionId: number, history: Message[], title?: string): ChatBinding {
    const now = () => listNow.current.sessions.find((s) => s.id === sessionId);
    return workspaces.chat(storage.keyOf(sessionId), {
      // The one born on this page first: until the list shows it, the chat names the deleted one it replaces.
      bound: () => born.current.get(sessionId) ?? now()?.workspaceId,
      // A chat not named yet leaves the workspace the module's own name.
      name: () => title ?? (now()?.title === UNTITLED ? "" : (now()?.title ?? "")),
      onBorn: (id, replaced) => {
        born.current.set(sessionId, id);
        setList((l) => ({ ...l, sessions: bornInto(l.sessions, sessionId, id, replaced) }));
      },
      seen: seenIn(history),
    });
  }

  /** Why the chat on screen may not pick or move its workspace now, or null (`pickHeld`): what the picker shows. */
  const workspaceHeld = pickHeld(streaming, active.id);

  /**
   * The picker's choice for the chat on screen: an existing workspace before its first file, or once it has one a
   * move there, or out of its own to none (`to` null), `leftName` naming the workspace it leaves (`moveWorkspace`).
   * While a reply of the chat runs nothing changes, by the same rule as the picker's, read from the turns
   * themselves: why, or null once done.
   */
  function pickWorkspace(to: string | null, leftName?: string): string | null {
    const sessionId = active.id;
    const held = pickHeld(running(), sessionId);
    if (held) return held;
    // A workspace born for the chat on this page stands in for its binding no longer: the list's is the one now.
    born.current.delete(sessionId);
    const departure = { at: Date.now(), ...(leftName !== undefined ? { name: leftName } : {}) };
    setList((l) => ({ ...l, sessions: moveWorkspace(l.sessions, sessionId, to ?? undefined, departure) }));
    return null;
  }

  /**
   * Deletes a workspace, freeing the pictures neither a chat nor an image
   * session names (none while either has not read its sessions). The chats
   * that worked in it keep their messages and still name it, as a chat
   * whose workspace another tab deleted does: each moves to an existing
   * workspace from the picker, or its next file starts a new workspace
   * (`bornInto`). A store that refuses the delete rejects.
   */
  async function deleteWorkspace(workspaceId: string) {
    const chats = pictures();
    const images = live ? live() : [];
    await workspaces.remove(workspaceId, chats === null || images === null ? null : [...chats, ...images]);
  }

  function deleteSession(sessionId: number) {
    // A session that goes takes its turn with it: the loop would otherwise
    // keep a stream open, writing into messages nobody can read.
    controllers.current.get(sessionId)?.abort();
    const id = newId();
    setList((l) => removeSession(l, sessionId, id));
  }

  /**
   * One turn, streamed into `sessionId` after `history`: to `prompt` sent as
   * a new user message (send, resend), with `images` on it when it carries
   * any and the files `input` attaches, or to the history as it stands when
   * `prompt` is null (regenerate).
   * A reply that calls agents starts them, hands their answers back and
   * streams the next reply, until one answers without calling a tool. Stop
   * ends the whole turn, agents included.
   */
  async function exchange(
    sessionId: number,
    history: Message[],
    prompt: string | null,
    images: PromptImage[] = [],
    input: TurnInput = NO_FILES,
  ) {
    if (!canStartTurn(running(), sessionId, parallel) || !canTake) return input.sent?.(false);
    // The model the turn goes to, when it is not the one loaded: the owner's choice in the picker, or the way back
    // from the image model. The page switches there before the first request, which names no model either way.
    const target = next.switches && next.id !== null ? { to: next.id, ...(next.from ? { from: next.from } : {}) } : null;
    // A request names no model: the server serves the one it has loaded, and a name could only switch it. The
    // page asks again which that is, so the header follows a switch another client made.
    refreshModel();
    const requestSettings: Settings = settings;
    // The files the tools write into the prompt are as they were when the turn started. The date and
    // time is the session's, not this turn's: it sits ahead of the whole conversation, so a moment that moved
    // between two turns would change the prompt's first tokens and cost a prefill of everything (GitHub #186).
    // What changes between turns, the memory notes and "update every prompt"'s moment, rides in the turn notice.
    const session = list.sessions.find((s) => s.id === sessionId);
    const promptContext = { now: session?.startedAt ?? new Date() };
    const extras = toolExtras(tools, promptContext);
    const agentTools = agentExtras(tools, promptContext);
    // The rounds the turn was started with: the same budget for this loop and for each agent's own.
    const maxRounds = Math.max(1, Math.floor(tools.maxRounds));
    // A first prompt names the chat, and so the workspace its files bring to life.
    const firstTitle =
      prompt !== null && history.length === 0
        ? titleOf({ id: 0, role: "user", content: prompt, images, attached: input.add.map((f) => f.name), reasoning: "", streaming: false })
        : undefined;
    const chat = chatBinding(sessionId, history, firstTitle);
    const local: Omit<LocalContext, "settings" | "signal"> = {
      jsSafetyCheck: tools.jsSafetyCheck,
      workspace: { workspaces, chat, vision, scale: redrawn },
      onCheck: (figures, error) =>
        logRow(sessionId, { laneTag: "agent", reasoningEffort: "none", figures, error, agent: "run_js safety check" }),
    };
    const abort = new AbortController();
    controllers.current.set(sessionId, abort);
    // Sending is a request to see the answer: follow it from the bottom. A turn
    // always starts in the session on screen; one left streaming in a session
    // the reader walked away from moves nothing, since the transcript scrolls
    // on the session it shows.
    following.current = true;
    setStreaming((s) => new Set(s).add(sessionId));
    const end = () => {
      controllers.current.delete(sessionId);
      setStreaming((s) => {
        const rest = new Set(s);
        rest.delete(sessionId);
        return rest;
      });
    };

    // The files go to the workspace before the turn is sent, while the Workspace switch is on: the prompt's own, and
    // an older chat's attachments, which become the first files of its workspace (spec #170). The prompt names them.
    // Files the store refuses leave the turn unsent, with nothing lost; a picture it refuses still goes inline.
    const adds = turnAdds(tools.enabled && tools.workspace, session?.attachments ?? [], { files: input.add, pictures: input.pictures });
    const taken: Taken = adds.files.length + adds.pictures.length > 0 ? await takeTurnFiles(workspaces, chat, adds) : { sent: true, named: [] };
    if (!taken.sent) {
      end();
      input.sent?.(false, taken.said);
      return;
    }
    if (adds.clearsLegacy) setList((l) => ({ ...l, sessions: clearAttachments(l.sessions, sessionId) }));
    const named = [...input.named, ...taken.named];
    // What the turn replaces goes once its files are in.
    const replaces = input.replaces;
    if (replaces !== undefined) setList((l) => ({ ...l, sessions: truncateFrom(l.sessions, sessionId, replaces) }));

    let conversation = history;
    // The prompt the switch is shown on: the new one, or the one a regenerate answers again.
    let switchOn = [...history].reverse().find((m) => m.role === "user")?.id;
    if (prompt !== null) {
      // The notice tells of the workspace as it is now, the prompt's own files in it; a store that cannot say leaves it out.
      const workspaceFiles = tools.enabled && tools.workspace ? await noticeFiles(workspaces, chat.bound()) : undefined;
      const user: Message = {
        id: newId(),
        role: "user",
        content: prompt,
        reasoning: "",
        streaming: false,
        ...(images.length > 0 ? { images } : {}),
        ...(named.length > 0 ? { attached: named } : {}),
        ...turnNotice(tools, history, { now: new Date(), notes: browserMemory.list(), files: workspaceFiles, attached: named, moved: session?.moved }),
      };
      setList((l) => ({ ...l, sessions: addMessages(l.sessions, sessionId, [user]) }));
      void keepSent(sessionId, user);
      conversation = [...conversation, user];
      switchOn = user.id;
    }
    input.sent?.(true, taken.said);
    const markSwitch = (state: TurnSwitch) => {
      if (switchOn === undefined) return;
      const id = switchOn;
      setList((l) => ({ ...l, sessions: updateMessage(l.sessions, sessionId, id, (m) => ({ ...m, switch: state })) }));
    };
    // Calls of the latest reply still without a result, if the loop breaks while they run.
    let unanswered: ToolCall[] = [];
    // The pictures this turn's calls made, by the round that made them: `made-N` counts on across rounds.
    const turnImages: ImageRun[] = [];
    try {
      if (target) {
        markSwitch({ ...target, status: "switching" });
        const failed = await switchAndWait(target.to, { signal: abort.signal, onModelChange: refreshModel });
        if (failed === "stopped") {
          markSwitch({ ...target, status: "stopped" });
          return;
        }
        if (failed) {
          markSwitch({ ...target, status: "failed", error: failed });
          return;
        }
        markSwitch({ ...target, status: "switched" });
      }
      for (let round = 0; ; round++) {
        const reply = await streamReply(sessionId, conversation, requestSettings, extras, abort.signal);
        conversation = [...conversation, reply];
        const calls = reply.toolCalls ?? [];
        if (calls.length === 0 || reply.error) break;
        // Every call gets a result, even one that never ran: the history must stay well-formed.
        const notRun = callsNotRun(reply.figures, abort.signal.aborted, round, maxRounds);
        if (notRun) {
          addToolResults(sessionId, calls.map((c) => ({ callId: c.id, content: notRun })));
          break;
        }
        unanswered = calls;
        conversation = [
          ...conversation,
          ...(await runToolCalls(sessionId, conversation, turnImages, reply, requestSettings, extras, agentTools, local, maxRounds, abort.signal)),
        ];
        unanswered = [];
        if (abort.signal.aborted) break;
      }
    } catch (err) {
      // A fault in the loop must not leave the page busy for good: it ends the turn, says what broke and
      // answers the calls it left open, so the next turn starts from a well-formed history.
      console.error(err);
      const message = `The Playground failed: ${err instanceof Error ? err.message : String(err)}`;
      if (unanswered.length > 0) addToolResults(sessionId, unanswered.map((c) => ({ callId: c.id, content: message })));
      const failure: Message = { id: newId(), role: "assistant", content: "", reasoning: "", streaming: false, error: message };
      setList((l) => ({
        ...l,
        sessions: addMessages(
          l.sessions.map((s) =>
            s.id === sessionId ? { ...s, messages: s.messages.map((m) => (m.streaming ? { ...m, streaming: false } : m)) } : s,
          ),
          sessionId,
          [failure],
        ),
      }));
    } finally {
      end();
    }
  }

  /** Streams one assistant reply and logs it; resolves with the reply as it ended. */
  async function streamReply(
    sessionId: number,
    conversation: Message[],
    requestSettings: Settings,
    extras: ToolExtras,
    signal: AbortSignal,
  ): Promise<Message> {
    const turns = conversationTurns(conversation.map(exchangeOf));
    let reply: Message = { id: newId(), role: "assistant", content: "", reasoning: "", streaming: true };
    setList((l) => ({ ...l, sessions: addMessages(l.sessions, sessionId, [reply]) }));
    // The reply as it streams waits for the frame; its end lands at once, so a
    // turn never waits on a frame and the next request sees the whole text.
    const change = (f: (m: Message) => Message, now = false) => {
      reply = f(reply);
      const next = reply;
      const update = (l: SessionList) => ({ ...l, sessions: updateMessage(l.sessions, sessionId, next.id, () => next) });
      if (now) setList(update);
      else setSoon(`${sessionId}:${next.id}`, update);
    };
    const result = await streamChat({
      request: { settings: requestSettings, turns, extras },
      streamId: sessionStream(sessionId),
      signal,
      onQueued: () => change((m) => ({ ...m, queued: true })),
      onStart: () => {
        if (reply.queued) change((m) => ({ ...m, queued: false }));
      },
      onEvent: (event) => {
        if (event.kind === "reasoning") change((m) => ({ ...m, reasoning: m.reasoning + event.text }));
        if (event.kind === "content") change((m) => ({ ...m, content: m.content + event.text }));
        if (event.kind === "tool_call") change((m) => ({ ...m, toolCalls: [...(m.toolCalls ?? []), event.call] }));
      },
    });
    const figures = result.ok ? computeFigures(result.timeline) : null;
    const error = result.ok ? undefined : result.message;
    change((m) => ({ ...m, streaming: false, queued: false, figures: figures ?? undefined, error }), true);
    logRow(sessionId, {
      laneTag: requestSettings.laneTag,
      reasoningEffort: requestSettings.reasoningEffort,
      thinkingBudget: thinkingBudgetOf(requestSettings),
      figures,
      error,
    });
    return reply;
  }

  /**
   * Runs a reply's calls — agents and web calls at once — shown on the reply
   * as they go; resolves with the tool results in call order. Picture calls
   * come last, on one visit to the image model: the others need the text
   * model loaded, and a switch would take it from under them.
   */
  async function runToolCalls(
    sessionId: number,
    conversation: Message[],
    /** The picture calls of this turn's earlier rounds: their replies in `conversation` do not carry them. */
    turnImages: ImageRun[],
    reply: Message,
    requestSettings: Settings,
    extras: ToolExtras,
    agentTools: ToolExtras,
    local: Omit<LocalContext, "settings" | "signal">,
    maxRounds: number,
    signal: AbortSignal,
  ) {
    const calls = reply.toolCalls ?? [];
    const available = (extras.tools ?? []).map((t) => t.function.name);
    const parsed = calls.filter((c) => routeCall(c.name, available) === "agent").map((c) => parseAgentCall(c, available));
    const tasks = parsed.flatMap((p) => (p.ok ? [p.task] : []));
    let runs: AgentRun[] = parsed.map((p) => (p.ok ? { ...p.task, status: "queued", reasoning: "", content: "" } : p.run));
    const webParsed = calls.filter((c) => routeCall(c.name, available) === "web").map((c) => parseWebCall(c, available));
    const webTasks = webParsed.flatMap((p) => (p.ok ? [p.task] : []));
    let webRuns: WebRun[] = webParsed.map((p) => (p.ok ? { ...p.task, status: "running" } : p.run));
    const unknownTools = calls.filter((c) => routeCall(c.name, available) === "unknown").map((c) => unknownCall(c, available));
    let questions = calls.filter((c) => routeCall(c.name, available) === "ask").map(parseAskCall);
    const localCalls = calls.filter((c) => routeCall(c.name, available) === "local");
    let localRuns: LocalRun[] = localCalls.map(startedRun);
    const imageParsed = calls.filter((c) => routeCall(c.name, available) === "image").map((c) => parseImageCall(c, available));
    const imageTasks = imageParsed.flatMap((p) => (p.ok ? [p.task] : []));
    let imageRuns: ImageRun[] = imageParsed.map((p) => (p.ok ? { ...p.task, status: "queued" } : p.run));
    const show = () => {
      const agents = runs;
      const web = webRuns;
      const asked = questions;
      const localNow = localRuns;
      const pictures = imageRuns;
      setSoon(`${sessionId}:${reply.id}:tools`, (l) => ({
        ...l,
        sessions: updateMessage(l.sessions, sessionId, reply.id, (m) => ({
          ...m,
          agents,
          web,
          unknownTools,
          questions: asked,
          local: localNow.map(shownRun),
          ...(pictures.length > 0 ? { imageRuns: pictures } : {}),
        })),
      }));
    };
    show();
    // Each waiting question settles when the user answers it (answer) or the turn is stopped.
    const asking = questions
      .filter((q) => q.status === "waiting")
      .map(
        (q) =>
          new Promise<void>((resolve) => {
            const key = `${reply.id}:${q.callId}`;
            const settle = (text: string | null) => {
              waiting.current.delete(key);
              signal.removeEventListener("abort", onAbort);
              questions = questions.map((x) =>
                x.callId !== q.callId ? x : text === null ? { ...x, status: "skipped" } : { ...x, status: "answered", answer: text },
              );
              show();
              resolve();
            };
            const onAbort = () => settle(null);
            if (signal.aborted) return settle(null);
            signal.addEventListener("abort", onAbort, { once: true });
            waiting.current.set(key, settle);
          }),
      );
    // The agents hear the workspace's files as a chat's first turn does: the workspace prompt they share says a notice brings them.
    const agentFiles = tasks.length > 0 && tools.enabled && tools.workspace && local.workspace ? await noticeFiles(workspaces, local.workspace.chat.bound()) : undefined;
    await Promise.all([
      runAgents(tasks, {
        settings: requestSettings,
        streamId: sessionStream(sessionId),
        tools: agentTools,
        notice: agentNotice(tools, browserMemory.list(), agentFiles),
        tavilyKey: getTavilyKey(),
        local,
        maxRounds,
        signal,
        onUpdate: (run) => {
          runs = runs.map((r) => (r.callId === run.callId ? run : r));
          show();
        },
      }),
      runWeb(webTasks, {
        tavilyKey: getTavilyKey(),
        signal,
        onUpdate: (run) => {
          webRuns = webRuns.map((r) => (r.callId === run.callId ? run : r));
          show();
        },
      }),
      runLocalCalls(localCalls, { ...local, settings: requestSettings, signal }, (run) => {
        localRuns = localRuns.map((r) => (r.callId === run.callId ? run : r));
        show();
      }),
      ...asking,
    ]);
    const pictured = await runImages(imageTasks, {
      messages: [...conversation, { role: "assistant", imageRuns: turnImages }],
      signal,
      onModelChange: refreshModel,
      // While the Workspace switch is on the pictures land in the chat's workspace (#394), which the reply shows them from.
      ...(tools.workspace && local.workspace ? madeIn(local.workspace.workspaces, local.workspace.chat) : {}),
      onUpdate: (run) => {
        imageRuns = imageRuns.map((r) => (r.callId === run.callId ? run : r));
        show();
      },
    });
    turnImages.push(...pictured);
    // A row per request an agent made; a run that failed adds one for its error.
    for (const run of runs) {
      if (!tasks.some((t) => t.callId === run.callId)) continue;
      const row = {
        laneTag: "agent" as const,
        reasoningEffort: requestSettings.reasoningEffort,
        // What agentRequest sent: the turn's own budget.
        thinkingBudget: thinkingBudgetOf(requestSettings),
        agent: run.name,
      };
      for (const figures of run.rounds ?? []) logRow(sessionId, { ...row, figures });
      if (run.error) logRow(sessionId, { ...row, figures: null, error: run.error });
    }
    const results = new Map([
      ...runs.map((run) => [run.callId, toolResult(run)] as const),
      ...webRuns.map((run) => [run.callId, webToolResult(run)] as const),
      ...unknownTools.map((call) => [call.callId, call.error] as const),
      ...questions.map((q) => [q.callId, askToolResult(q)] as const),
      ...localRuns.map((run) => [run.callId, localToolResult(run)] as const),
      ...imageRuns.map((run) => [run.callId, imageToolResult(run)] as const),
    ]);
    // A picture a read_file call read with vision rides on its result, which keeps it: every later request sends the same bytes.
    const readPictures = new Map(localRuns.map((run) => [run.callId, localToolImages(run)] as const));
    return addToolResults(sessionId, calls.map((c) => ({ callId: c.id, content: results.get(c.id) ?? "", images: readPictures.get(c.id) })));
  }

  /** The pictures beside a sent prompt's redraws, into the store, and named on its images once there. */
  async function keepSent(sessionId: number, message: Message) {
    const images = await keepPictures(message.images ?? [], (blob) => keepPlain(store, blob).then((p) => p.id));
    if (images) setList((l) => ({ ...l, sessions: updateMessage(l.sessions, sessionId, message.id, (m) => ({ ...m, images })) }));
  }

  /** The store's pictures the chat's sessions name — what deleting an Image view session must keep — or null until they are read. */
  function pictures(): Set<ImageId> | null {
    return storage.loaded ? new Set(listNow.current.sessions.flatMap((s) => [...picturesOf(s)])) : null;
  }

  function addToolResults(sessionId: number, results: { callId: string; content: string; images?: PromptImage[] }[]): Message[] {
    const messages = results.map(
      (r): Message => ({
        id: newId(),
        role: "tool",
        content: r.content,
        ...(r.images?.length ? { images: r.images } : {}),
        toolCallId: r.callId,
        reasoning: "",
        streaming: false,
      }),
    );
    setList((l) => ({ ...l, sessions: addMessages(l.sessions, sessionId, messages) }));
    return messages;
  }

  function logRow(sessionId: number, row: Omit<LogRow, "n" | "at">) {
    setList((l) => ({ ...l, sessions: addLogRow(l.sessions, sessionId, { at: new Date().toLocaleTimeString(), ...row }) }));
  }

  /**
   * Sends `prompt`, and the images and files picked with it, in the active session when a turn can start. Resolves
   * once the prompt is sent, or with false when it is not: the store refused its files, which the box gets back,
   * as the caller gives it back the prompt and its images.
   */
  function send(prompt: string, images: PromptImage[] = []): Promise<boolean> {
    if (!canRun) return Promise.resolve(false);
    // With the Workspace switch off the box hides its files and keeps them for when it is back on.
    const workspace = tools.enabled && tools.workspace;
    const files = workspace ? attached : [];
    if (files.length > 0) setAttached([]);
    setAttachError(null);
    return new Promise((resolve) => {
      // The prompt's pictures go inline as they always did, and become files of the workspace too (#394).
      const input: TurnInput = {
        add: files,
        pictures: workspace ? boxPictures(images) : [],
        named: [],
        sent: (sent, said) => {
          if (!sent && files.length > 0) setAttached((current) => [...files, ...current]);
          if (said) setAttachError(said);
          resolve(sent);
        },
      };
      void exchange(active.id, active.messages, prompt, images, input);
    });
  }

  /**
   * Drops `messageId` and what follows, then streams again: an edited
   * prompt, or a fresh reply. An edited prompt keeps the images the original
   * was sent with — editing the words is not a reason to take the picture away.
   */
  function rerun(messageId: number, prompt: string | null) {
    const index = active.messages.findIndex((m) => m.id === messageId);
    if (index === -1 || !canStartTurn(running(), active.id, parallel) || !canTake) return;
    const sessionId = active.id;
    const images = active.messages[index].images ?? [];
    // Its files too: they are in the workspace already, and the prompt names them again.
    const named = active.messages[index].attached ?? [];
    // The turn drops the message and what follows once an older chat's attachments are in its workspace: a store that refuses them leaves the chat as it was.
    const sent = (_sent: boolean, said?: string) => {
      if (said) setAttachError(said);
    };
    const input: TurnInput = { ...NO_FILES, named: prompt === null ? [] : named, replaces: messageId, sent };
    void exchange(sessionId, active.messages.slice(0, index), prompt, prompt === null ? [] : images, input);
  }

  function saveEdit(messageId: number, text: string) {
    const sessionId = active.id;
    setList((l) => ({ ...l, sessions: editMessage(l.sessions, sessionId, messageId, text) }));
  }

  function fork(messageId: number) {
    following.current = true;
    const id = newId();
    const sessionId = active.id;
    setList((l) => forkSession(l, sessionId, messageId, id));
  }

  /** Reads files for the next prompt, text, PDF or audio; the ones that cannot be read are named in `attachError`. */
  async function attach(files: File[]) {
    const taken = [...attached.map((f) => f.name), ...active.attachments.map((a) => a.name)];
    const added: AttachedFile[] = [];
    const errors: string[] = [];
    for (const file of files) {
      const result = await readAttached(file, [...taken, ...added.map((f) => f.name)]);
      if (result.ok) added.push(result.file);
      else errors.push(result.error);
    }
    setAttachError(errors.length > 0 ? errors.join(" ") : null);
    if (added.length > 0) setAttached((current) => [...current, ...added]);
  }

  /** Takes a file out of the next prompt, or out of an older chat's attachments before they reach its workspace. */
  function detach(name: string) {
    if (attached.some((f) => f.name === name)) {
      setAttached((current) => current.filter((f) => f.name !== name));
      return;
    }
    const sessionId = active.id;
    setList((l) => ({ ...l, sessions: removeAttachment(l.sessions, sessionId, name) }));
  }

  /** The user's answer to a question a reply is waiting on. */
  function answer(messageId: number, callId: string, text: string) {
    waiting.current.get(`${messageId}:${callId}`)?.(text);
  }

  /** Stops the turn of the session on screen, which is the one Stop is shown for. */
  function stop() {
    controllers.current.get(active.id)?.abort();
  }

  return {
    list,
    active,
    streaming,
    streamingHere,
    busy,
    sendBlocked,
    canRun,
    following,
    selectSession,
    newSession,
    deleteSession,
    send,
    rerun,
    saveEdit,
    fork,
    answer,
    attach,
    detach,
    /** The chat on screen's hold on its workspace: what the workspace panel makes files through. */
    binding: () => chatBinding(active.id, active.messages),
    /** The chat on screen, and each chat's title by its store key: who made a version, as the workspace panel names it. */
    chatNames: {
      here: (key: string) => storage.idOf(key) === active.id,
      title: (key: string) => {
        const id = storage.idOf(key);
        const title = list.sessions.find((s) => s.id === id)?.title;
        return title === UNTITLED ? undefined : title;
      },
    } satisfies ChatNames,
    workspaceHeld,
    pickWorkspace,
    deleteWorkspace,
    /** What the prompt box shows of the files the next turn adds to the workspace: the box's, then an older chat's. */
    attachments: [...attached, ...active.attachments.map(fromLegacy)].map(attachedChip),
    attachError,
    stop,
    pictures,
    /** What the store refused, when it did: the sessions go on in memory. */
    storeError: storage.error,
  };
}
