import { sendText, type Destination, type Sent } from "../workspace/send.ts";
import type { Workspaces } from "../workspace/workspaces.ts";
import { fileStem } from "./export.ts";
import { transcriptMarkdown } from "./markdown.ts";
import { titleOf } from "./sessions.ts";
import type { Transcript } from "./transcript.ts";

// Send to workspace: the Transcript as a Markdown file of a workspace (spec
// #170). The text is the transcript alone as the `.md` export writes it, its
// notes, answers and pictures left out (spec #428), and the name is the one that
// download would have; the Transcript keeps its own record.

/** The file a Transcript becomes: the transcript alone as Markdown, under the name the `.md` download would carry. */
export const transcriptFile = (t: Transcript): { name: string; text: string } => ({ name: `${fileStem(titleOf(t))}.md`, text: transcriptMarkdown(t) });

/** What a new workspace for the Transcript is called: the Transcript's title. */
export const workspaceNameFor = (t: Transcript): string => titleOf(t);

export const sendTranscript = (workspaces: Workspaces, t: Transcript, destination: Destination): Promise<Sent> =>
  sendText(workspaces, destination, transcriptFile(t));
