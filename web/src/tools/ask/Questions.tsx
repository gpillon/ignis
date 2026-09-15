import { useState } from "react";
import { field } from "../../ui/classes.ts";
import type { Question } from "./ask.ts";

// The questions a reply asked the user, inside the reply. A waiting question
// takes an option or a written answer; the turn goes on once every question
// in the reply is settled.

export function Questions({ questions, onAnswer }: { questions: Question[]; onAnswer: (callId: string, text: string) => void }) {
  return (
    <section aria-label="Questions" className="flex flex-col gap-3">
      {questions.map((q, i) => (
        <QuestionCard key={`${i}:${q.callId}`} question={q} onAnswer={(text) => onAnswer(q.callId, text)} />
      ))}
    </section>
  );
}

const STATUS_LABEL = { waiting: "Waiting for your answer", answered: "Answered", skipped: "Not answered", invalid: "Not asked" };

function QuestionCard({ question: q, onAnswer }: { question: Question; onAnswer: (text: string) => void }) {
  const [text, setText] = useState("");
  const waiting = q.status === "waiting";
  const send = (value: string) => {
    const answer = value.trim();
    if (answer) onAnswer(answer);
  };
  return (
    <div className={`cut flex flex-col gap-3 bg-surface px-4 py-3 [--cut-size:12px] ${waiting ? "shadow-[inset_0_0_0_1px_var(--ember)]" : ""}`}>
      <div className="flex flex-wrap items-baseline justify-between gap-x-3">
        <h3 className={`font-display text-[13px] font-semibold ${waiting ? "text-ember" : "text-ink"}`}>Question for you</h3>
        <span className="font-display text-xs text-ash">{STATUS_LABEL[q.status]}</span>
      </div>
      <p className="whitespace-pre-wrap break-words text-[15px] leading-normal text-ink">{q.question}</p>
      {waiting ? (
        <>
          {q.options.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {q.options.map((option, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => send(option)}
                  className="cut bg-ground px-3 py-1.5 text-left text-sm text-ink [--cut-size:8px] hover:bg-line/60 hover:text-ember"
                >
                  {option}
                </button>
              ))}
            </div>
          )}
          <form
            className="flex items-end gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              send(text);
            }}
          >
            <textarea
              className={`${field} flex-1 resize-none leading-snug`}
              rows={1}
              name="answer"
              aria-label="Your answer"
              placeholder={q.options.length > 0 ? "Or write your own answer" : "Your answer"}
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                  e.preventDefault();
                  send(text);
                }
              }}
            />
            <button
              type="submit"
              disabled={!text.trim()}
              className="cut bg-ember px-3 py-2 font-display text-sm font-semibold text-[#1c2026] [--cut-size:8px] disabled:opacity-40"
            >
              Answer
            </button>
          </form>
        </>
      ) : q.status === "answered" ? (
        <p className="border-l-2 border-ember pl-3 text-sm whitespace-pre-wrap text-ink">{q.answer}</p>
      ) : (
        <p className="text-sm text-ash">{q.status === "invalid" ? q.error : "The turn was stopped before you answered."}</p>
      )}
    </div>
  );
}
