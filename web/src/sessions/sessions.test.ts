import { describe, expect, it } from "vitest";
import {
  addExchange,
  addLogRow,
  addMessages,
  addReply,
  createSession,
  editMessage,
  forkSession,
  type Message,
  openSession,
  removeSession,
  type SessionList,
  titleFrom,
  truncateFrom,
  UNTITLED,
  updateMessage,
} from "./sessions.ts";

const message = (id: number, role: Message["role"], content: string): Message => ({
  id,
  role,
  content,
  reasoning: "",
  streaming: role === "assistant",
});

const withMessages = (id: number): SessionList["sessions"][number] => ({
  ...createSession(id),
  messages: [message(100 + id, "user", "hi")],
});

describe("titleFrom", () => {
  it("takes the first non-blank line with its spaces collapsed", () => {
    expect(titleFrom("\n  Explain   the KV cache \nin detail")).toBe("Explain the KV cache");
  });

  it("shortens a long line", () => {
    const title = titleFrom("a".repeat(80));
    expect(title).toHaveLength(48);
    expect(title.endsWith("…")).toBe(true);
  });

  it("falls back to the untitled name", () => {
    expect(titleFrom("  \n ")).toBe(UNTITLED);
  });
});

describe("openSession", () => {
  it("reuses an active session that has no messages", () => {
    const list = { sessions: [createSession(1)], activeId: 1 };
    expect(openSession(list, 2)).toBe(list);
  });

  it("opens a new session at the top when the active one has messages", () => {
    const next = openSession({ sessions: [withMessages(1)], activeId: 1 }, 2);
    expect(next.activeId).toBe(2);
    expect(next.sessions.map((s) => s.id)).toEqual([2, 1]);
  });
});

describe("removeSession", () => {
  const list = { sessions: [withMessages(3), withMessages(2), withMessages(1)], activeId: 2 };

  it("keeps the active session when another one goes", () => {
    const next = removeSession(list, 3, 9);
    expect(next.sessions.map((s) => s.id)).toEqual([2, 1]);
    expect(next.activeId).toBe(2);
  });

  it("moves to the session that took the removed one's place", () => {
    expect(removeSession(list, 2, 9).activeId).toBe(1);
    expect(removeSession({ ...list, activeId: 1 }, 1, 9).activeId).toBe(2);
  });

  it("leaves a fresh session when the last one goes", () => {
    const next = removeSession({ sessions: [withMessages(1)], activeId: 1 }, 1, 9);
    expect(next).toEqual({ sessions: [createSession(9)], activeId: 9 });
  });

  it("ignores an unknown session", () => {
    expect(removeSession(list, 42, 9)).toBe(list);
  });
});

describe("exchanges and log", () => {
  it("names the session after its first prompt only", () => {
    let sessions = [createSession(1), createSession(2)];
    sessions = addExchange(sessions, 1, message(1, "user", "First prompt"), message(2, "assistant", ""));
    sessions = addExchange(sessions, 1, message(3, "user", "Second prompt"), message(4, "assistant", ""));
    expect(sessions[0].title).toBe("First prompt");
    expect(sessions[0].messages).toHaveLength(4);
    expect(sessions[1]).toEqual(createSession(2));
  });

  it("changes one message in one session", () => {
    let sessions = addExchange([createSession(1)], 1, message(1, "user", "q"), message(2, "assistant", ""));
    sessions = updateMessage(sessions, 1, 2, (m) => ({ ...m, content: m.content + "tok" }));
    expect(sessions[0].messages.map((m) => m.content)).toEqual(["q", "tok"]);
  });

  it("edits a message's text and marks it, leaving an unchanged one alone", () => {
    let sessions = addExchange([createSession(1)], 1, message(1, "user", "q"), message(2, "assistant", "a"));
    sessions = editMessage(sessions, 1, 2, "better answer");
    expect(sessions[0].messages[1]).toMatchObject({ content: "better answer", edited: true });
    expect(sessions[0].messages[0].edited).toBeUndefined();
    expect(editMessage(sessions, 1, 1, "q")[0].messages[0].edited).toBeUndefined();
  });

  it("drops a message and what follows it, keeping the log", () => {
    const row = { at: "12:00", laneTag: "interactive" as const, reasoningEffort: "xhigh" as const, figures: null };
    let sessions = addExchange([createSession(1)], 1, message(1, "user", "q1"), message(2, "assistant", "a1"));
    sessions = addExchange(sessions, 1, message(3, "user", "q2"), message(4, "assistant", "a2"));
    sessions = addLogRow(sessions, 1, row);
    const cut = truncateFrom(sessions, 1, 3);
    expect(cut[0].messages.map((m) => m.id)).toEqual([1, 2]);
    expect(cut[0].log).toHaveLength(1);
    expect(truncateFrom(sessions, 1, 99)).toEqual(sessions);
  });

  it("appends messages and names an empty session after the user message among them", () => {
    const [named] = addMessages([createSession(1)], 1, [message(1, "user", "Plan the work"), message(2, "assistant", "")]);
    expect(named.title).toBe("Plan the work");
    const [kept] = addMessages([named], 1, [message(3, "tool", "result")]);
    expect(kept.title).toBe("Plan the work");
    expect(kept.messages.map((m) => m.id)).toEqual([1, 2, 3]);
  });

  it("adds a reply without a prompt", () => {
    const sessions = addReply([withMessages(1)], 1, message(7, "assistant", ""));
    expect(sessions[0].messages.map((m) => m.id)).toEqual([101, 7]);
  });

  it("forks a session up to and including a message", () => {
    let sessions = addExchange([createSession(1)], 1, message(1, "user", "q1"), message(2, "assistant", "a1"));
    sessions = addExchange(sessions, 1, message(3, "user", "q2"), message(4, "assistant", "a2"));
    const list = { sessions, activeId: 1 };
    const next = forkSession(list, 1, 2, 9);
    expect(next.activeId).toBe(9);
    expect(next.sessions.map((s) => s.id)).toEqual([9, 1]);
    expect(next.sessions[0]).toMatchObject({ title: "q1 (fork)", log: [] });
    expect(next.sessions[0].messages.map((m) => m.id)).toEqual([1, 2]);
    expect(next.sessions[0].messages.every((m) => !m.streaming)).toBe(true);
    expect(next.sessions[1]).toBe(sessions[0]);
    expect(forkSession(list, 1, 99, 9)).toBe(list);
  });

  it("numbers log rows per session", () => {
    const row = { at: "12:00", laneTag: "interactive" as const, reasoningEffort: "xhigh" as const, figures: null };
    let sessions = [createSession(1), createSession(2)];
    sessions = addLogRow(sessions, 1, row);
    sessions = addLogRow(sessions, 1, row);
    sessions = addLogRow(sessions, 2, row);
    expect(sessions[0].log.map((r) => r.n)).toEqual([1, 2]);
    expect(sessions[1].log.map((r) => r.n)).toEqual([1]);
  });
});
