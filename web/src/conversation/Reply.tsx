import { useState } from "react";
import { Readout } from "../metrics/Readout.tsx";
import type { Message } from "../sessions/sessions.ts";
import { AgentStrip } from "../tools/agents/AgentStrip.tsx";
import { IconFork, IconPencil, IconRegenerate } from "../ui/icons.tsx";
import { Markdown } from "../ui/Markdown.tsx";
import { MessageEditor } from "./MessageEditor.tsx";
import { ActionButton, ActionRow, EditedMark, RERUN_BLOCKED, type TurnActions } from "./TurnActions.tsx";

export function Reply(props: {
  message: Message;
  markdown: boolean;
  last: boolean;
  actions: TurnActions;
  onRegenerate: () => void;
  openCallId: string | null;
  onOpenAgent: (callId: string) => void;
}) {
  const { message: m, markdown, actions } = props;
  const [editing, setEditing] = useState(false);
  const thinking = m.streaming && !m.content;
  return (
    <article className="group/turn relative flex flex-col gap-3">
      {/* The i-dot of the wordmark marks what ignis wrote. */}
      <span className="cut absolute top-[0.4em] -left-6 hidden size-2.5 bg-ember [--cut-size:4px] md:block" aria-hidden />
      {m.reasoning && (
        <details className="reasoning" open={thinking}>
          <summary className="cursor-pointer select-none font-display text-[13px] font-medium text-ash hover:text-ink">
            {thinking ? "Thinking…" : "Reasoning"}
          </summary>
          <pre className="mt-2 max-h-80 overflow-y-auto border-l-2 border-line pl-3 whitespace-pre-wrap break-words font-sans text-[13px] leading-normal text-ash">
            {m.reasoning}
          </pre>
        </details>
      )}
      {editing ? (
        <MessageEditor
          initial={m.content}
          onCancel={() => setEditing(false)}
          buttons={[
            {
              label: "Save",
              primary: true,
              onClick: (text) => {
                actions.onSave(text);
                setEditing(false);
              },
            },
          ]}
        />
      ) : (
        (m.content || (m.streaming && !m.reasoning)) &&
        (markdown && m.content ? (
          <Markdown text={m.content} streaming={m.streaming} />
        ) : (
          <pre className="whitespace-pre-wrap break-words font-sans text-[15px] leading-normal">
            {m.content}
            {m.streaming && <span className="caret" aria-hidden />}
          </pre>
        ))
      )}
      {m.agents && m.agents.length > 0 && (
        <AgentStrip runs={m.agents} openCallId={props.openCallId} onOpen={props.onOpenAgent} />
      )}
      {m.error && (
        <p className="border-l-2 border-fault pl-3 text-sm text-fault" role="alert">
          {m.error}
        </p>
      )}
      {m.figures && <Readout figures={m.figures} />}
      {!m.streaming && !editing && (
        <ActionRow className="-mt-1 -ml-2" pinned={props.last}>
          <ActionButton
            icon={<IconRegenerate />}
            label="Regenerate"
            disabled={!actions.canRerun}
            title={actions.canRerun ? "Drops this reply and what follows, then answers the same prompt again" : RERUN_BLOCKED}
            onClick={props.onRegenerate}
          />
          <ActionButton icon={<IconPencil />} label="Edit" onClick={() => setEditing(true)} />
          <ActionButton icon={<IconFork />} label="Fork" title="Start a new session from this reply" onClick={actions.onFork} />
          {m.edited && <EditedMark />}
        </ActionRow>
      )}
    </article>
  );
}
