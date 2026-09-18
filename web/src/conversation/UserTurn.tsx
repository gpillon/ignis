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
      {m.images && m.images.length > 0 && (
        <div className="flex flex-wrap justify-end gap-2">
          {/* A download, not a new tab: Chrome refuses to navigate the top frame to a `data:` URL. */}
          {m.images.map((image, index) => (
            <a key={`${image.name}:${index}`} href={image.url} download={image.name} title={`Save ${image.name}`}>
              <img src={image.url} alt={image.name} className="cut block max-h-56 w-auto max-w-full [--cut-size:10px]" />
            </a>
          ))}
        </div>
      )}
      {m.content !== "" && (
        <article className="cut bg-surface px-4 py-3 [--cut-size:12px]">
          <pre className="whitespace-pre-wrap break-words font-sans text-[15px] leading-normal">{m.content}</pre>
        </article>
      )}
      <ActionRow className="-mr-2">
        {m.edited && <EditedMark />}
        <ActionButton icon={<IconPencil />} label="Edit" onClick={() => setEditing(true)} />
        <ActionButton icon={<IconFork />} label="Fork" title="Start a new session from this message" onClick={actions.onFork} />
      </ActionRow>
    </div>
  );
}
