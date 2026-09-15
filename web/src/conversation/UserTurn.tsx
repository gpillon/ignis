import { useState } from "react";
import type { Message } from "../sessions/sessions.ts";
import { IconFork, IconPencil } from "../ui/icons.tsx";
import { MessageEditor } from "./MessageEditor.tsx";
import { ActionButton, ActionRow, EditedMark, RERUN_BLOCKED, type TurnActions } from "./TurnActions.tsx";

export function UserTurn({ message: m, actions, onResend }: { message: Message; actions: TurnActions; onResend: (text: string) => void }) {
  const [editing, setEditing] = useState(false);
  if (editing) {
    return (
      <MessageEditor
        initial={m.content}
        onCancel={() => setEditing(false)}
        buttons={[
          {
            label: "Save",
            onClick: (text) => {
              actions.onSave(text);
              setEditing(false);
            },
          },
          {
            label: "Save and resend",
            primary: true,
            disabled: !actions.canRerun,
            title: actions.canRerun ? "Drops everything after this prompt and sends it again" : RERUN_BLOCKED,
            onClick: (text) => {
              setEditing(false);
              onResend(text);
            },
          },
        ]}
      />
    );
  }
  return (
    <div className="group/turn flex flex-col items-end gap-1 self-end md:max-w-[85%]">
      <article className="cut bg-surface px-4 py-3 [--cut-size:12px]">
        <pre className="whitespace-pre-wrap break-words font-sans text-[15px] leading-normal">{m.content}</pre>
      </article>
      <ActionRow className="-mr-2">
        {m.edited && <EditedMark />}
        <ActionButton icon={<IconPencil />} label="Edit" onClick={() => setEditing(true)} />
        <ActionButton icon={<IconFork />} label="Fork" title="Start a new session from this message" onClick={actions.onFork} />
      </ActionRow>
    </div>
  );
}
