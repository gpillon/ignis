import { describe, expect, it } from "vitest";
import type { Attachment } from "../tools/local/attachments.ts";
import { addAttachments, createSession, forkSession, removeAttachment } from "./sessions.ts";

const file = (name: string): Attachment => ({ name, size: 1, kind: "text", text: "x" });

describe("session attachments", () => {
  it("adds and removes files on one session only", () => {
    let sessions = [createSession(1), createSession(2)];
    sessions = addAttachments(sessions, 1, [file("a.txt"), file("b.txt")]);
    expect(sessions[0].attachments.map((a) => a.name)).toEqual(["a.txt", "b.txt"]);
    expect(sessions[1].attachments).toEqual([]);
    sessions = removeAttachment(sessions, 1, "a.txt");
    expect(sessions[0].attachments.map((a) => a.name)).toEqual(["b.txt"]);
  });

  it("keeps the files in a fork", () => {
    const withFile = addAttachments([createSession(1)], 1, [file("a.txt")]);
    const source = {
      ...withFile[0],
      messages: [{ id: 5, role: "user" as const, content: "q", reasoning: "", streaming: false }],
    };
    const next = forkSession({ sessions: [source], activeId: 1 }, 1, 5, 9);
    expect(next.sessions[0].attachments.map((a) => a.name)).toEqual(["a.txt"]);
  });
});
