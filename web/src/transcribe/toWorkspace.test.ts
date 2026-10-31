import { describe, expect, it } from "vitest";
import { createMemoryStore } from "../store/memory.ts";
import { createWorkspaces, type Workspaces } from "../workspace/workspaces.ts";
import { toMarkdown } from "./markdown.ts";
import { sendTranscript, transcriptFile, workspaceNameFor } from "./toWorkspace.ts";
import { addUtterance, createTranscript, type Transcript } from "./transcript.ts";

// Send to workspace (spec #170, ticket 399): the Transcript becomes one
// Markdown file of a workspace, the text the Markdown export writes.

function call(): Transcript {
  let t = createTranscript(1, { source: "call", at: new Date(2026, 9, 25, 15, 42) });
  t = addUtterance(t, { stream: "mic-1", startMs: 1_200, endMs: 3_900, text: "Mi sentite?", voice: { kind: "me" } });
  return addUtterance(t, { stream: "call-1", startMs: 4_480, endMs: 9_020, text: "Sì, forte e chiaro.", voice: { kind: "speaker", index: 0 } });
}

function module(): Workspaces {
  let n = 0;
  return createWorkspaces(createMemoryStore(), { now: () => 1_000 + n, newId: () => `id${++n}` });
}

describe("the file a Transcript becomes", () => {
  it("holds exactly the Markdown export, under the name its download would carry", () => {
    const t = call();
    expect(transcriptFile(t)).toEqual({ name: "25-Oct-15-42-Call.md", text: toMarkdown(t) });
    expect(transcriptFile(t).text).toContain("**Me** (00:01): Mi sentite?");
  });

  it("names a new workspace after the Transcript", () => {
    expect(workspaceNameFor(call())).toBe("25 Oct 15:42 · Call");
  });
});

describe("sending a Transcript to a workspace", () => {
  it("to a new workspace named after it: one file, the export byte for byte, authored by the user", async () => {
    const w = module();
    const t = call();
    const sent = await sendTranscript(w, t, { kind: "new", name: workspaceNameFor(t) });
    expect(sent.workspace.name).toBe("25 Oct 15:42 · Call");
    expect(sent.written).toMatchObject({ name: "25-Oct-15-42-Call.md", version: 1 });
    const files = await w.files(sent.workspace.id);
    expect(files).toHaveLength(1);
    expect(files[0].kind).toBe("text");
    expect(files[0].versions[0].text).toBe(toMarkdown(t));
    expect(files[0].versions[0].author).toEqual({ by: "user" });
  });

  it("to an existing workspace, beside its files; a second send takes name (2)", async () => {
    const w = module();
    const t = call();
    const home = await w.create("Notes");
    await w.writeText(home.id, "plan.md", "# Plan", { by: "user" });
    await sendTranscript(w, t, { kind: "existing", id: home.id });
    const again = await sendTranscript(w, t, { kind: "existing", id: home.id });
    expect(again.written.name).toBe("25-Oct-15-42-Call (2).md");
    expect((await w.files(home.id)).map((f) => f.name)).toEqual(["25-Oct-15-42-Call (2).md", "25-Oct-15-42-Call.md", "plan.md"]);
    expect((await w.list()).length).toBe(1);
  });

  it("leaves the Transcript as it was, and fails on a workspace that is gone", async () => {
    const w = module();
    const t = call();
    const before = JSON.stringify(t);
    await sendTranscript(w, t, { kind: "new", name: "x" });
    expect(JSON.stringify(t)).toBe(before);
    await expect(sendTranscript(w, t, { kind: "existing", id: "nope" })).rejects.toThrow("gone");
  });
});
