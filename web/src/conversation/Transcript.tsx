import { type RefObject, useLayoutEffect, useRef } from "react";
import type { ModelState } from "../api/model.ts";
import type { Session } from "../sessions/sessions.ts";
import { isAtBottom } from "../ui/scroll.ts";
import { EmptyState } from "./EmptyState.tsx";
import { Reply } from "./Reply.tsx";
import type { TurnActions } from "./TurnActions.tsx";
import { UserTurn } from "./UserTurn.tsx";

/** The agent open in the reader: the reply that started it, and its call. */
export type OpenAgent = { messageId: number; callId: string };

/**
 * The session's turns, scrolling. `following` says whether the reader is at
 * the bottom: only then do new tokens pull the view down.
 */
export function Transcript(props: {
  session: Session;
  model: ModelState;
  markdown: boolean;
  canRerun: boolean;
  following: RefObject<boolean>;
  openAgent: OpenAgent | null;
  onOpenAgent: (agent: OpenAgent) => void;
  onRerun: (messageId: number, prompt: string | null) => void;
  onSave: (messageId: number, text: string) => void;
  onFork: (messageId: number) => void;
  onAnswer: (messageId: number, callId: string, text: string) => void;
}) {
  const { session: active, following, openAgent } = props;
  const transcript = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    const el = transcript.current;
    if (el && following.current) el.scrollTop = el.scrollHeight;
  }, [active.messages, active.id]);

  return (
    <div
      ref={transcript}
      onScroll={(e) => (following.current = isAtBottom(e.currentTarget))}
      className="flex min-h-0 flex-1 flex-col overflow-y-auto px-4 py-6 md:px-10"
    >
      {active.messages.length === 0 ? (
        <EmptyState model={props.model} />
      ) : (
        <div className="mx-auto flex w-full max-w-3xl flex-col gap-7">
          {active.messages.map((m, i) => {
            // A tool result shows through the agent card or web row of the call it answers.
            if (m.role === "tool") return null;
            const actions: TurnActions = {
              canRerun: props.canRerun,
              onSave: (text) => props.onSave(m.id, text),
              onFork: () => props.onFork(m.id),
            };
            // Keyed by session too: a fork shares message ids with its source.
            const key = `${active.id}:${m.id}`;
            return m.role === "user" ? (
              <UserTurn key={key} message={m} actions={actions} onResend={(text) => props.onRerun(m.id, text)} />
            ) : (
              <Reply
                key={key}
                message={m}
                markdown={props.markdown}
                last={i === active.messages.length - 1}
                actions={actions}
                onRegenerate={() => props.onRerun(m.id, null)}
                openCallId={openAgent?.messageId === m.id ? openAgent.callId : null}
                onOpenAgent={(callId) => props.onOpenAgent({ messageId: m.id, callId })}
                onAnswer={(callId, text) => props.onAnswer(m.id, callId, text)}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}
