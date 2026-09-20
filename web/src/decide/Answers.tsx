import { useState } from "react";
import { asText } from "./json.ts";
import { Bar, BoxGlyph, DigitTrace, DistributionRow, ImageMark, NoulMark, PointGlyph, ScoreMark } from "./marks.tsx";
import { type Draft, evidenceImage, type Question } from "./model.ts";
import { type Answer, AXES, levelOrder, type Run } from "./request.ts";

// What came back (GitHub #247), read in the order the request declared.
//
// The response's own order is not that: `answers`, `probabilities` and `legend`
// are `BTreeMap`s on the wire, so they arrive sorted alphabetically — and a
// score's level keys are index strings, where "10" sorts before "2". Every
// panel walks the sent draft instead and parses a level key as a number.

export function Answers({ draft, run }: { draft: Draft; run: Run }) {
  const anyGenerated = run.response.usage.output_tokens > 0;
  return (
    <div className="flex min-w-0 flex-col gap-5">
      <Cost run={run} anyGenerated={anyGenerated} />
      <ol className="flex flex-col gap-5">
        {draft.questions.map((question) => (
          <li key={question.uid}>
            <AnswerPanel question={question} answer={run.response.answers[question.id]} draft={draft} />
          </li>
        ))}
      </ol>
      <AnswerMassNote />
      <Raw run={run} />
    </div>
  );
}

/** The wire cost, and what it says about which primitives were asked. */
function Cost({ run, anyGenerated }: { run: Run; anyGenerated: boolean }) {
  const { input_tokens, output_tokens } = run.response.usage;
  return (
    <div className="border-b border-line pb-3">
      <p className="flex flex-wrap items-baseline gap-x-4 gap-y-1 font-display text-[13px] tabular-nums text-ash">
        <span>
          <span className="text-ink">{input_tokens.toLocaleString()}</span> prompt tokens
        </span>
        <span>
          <span className="text-ink">{output_tokens.toLocaleString()}</span> generated
        </span>
        <span>
          <span className="text-ink">{Math.round(run.elapsedMs)}</span> ms
        </span>
        <span className="text-ash/80">{run.response.model}</span>
      </p>
      <p className="mt-1.5 max-w-[68ch] text-[12px] leading-snug text-ash">
        {anyGenerated
          ? "A number, point or box generates a digit per step, so those tokens are real. The readouts beside them generated none."
          : "Nothing was generated: every answer was read from the logits of one position, out of a single prefill of the evidence."}
      </p>
    </div>
  );
}

function AnswerPanel({ question, answer, draft }: { question: Question; answer: Answer | undefined; draft: Draft }) {
  // What was asked is the heading. The name is how the answer is *keyed*,
  // which matters to the caller writing code against it and not to the reader
  // looking at the answer, so it sits beside the primitive as a label.
  const asked = asText(question.instructions).trim();
  return (
    <section>
      <header className="flex items-baseline justify-between gap-3">
        <h3 className="min-w-0 font-display text-[17px] font-semibold leading-snug text-ink">{asked || question.id}</h3>
        <span className="flex shrink-0 items-baseline gap-2 font-display text-[11px] text-ash">
          <span className="font-mono" title="The key this answer came back under">
            {question.id}
          </span>
          <span className="cut bg-surface px-2 py-0.5 [--cut-size:5px]">{question.kind}</span>
        </span>
      </header>
      <div className="mt-3">
        {answer === undefined ? (
          <Missing id={question.id} />
        ) : answer.type === "error" ? (
          <Failed code={answer.code} message={answer.message} />
        ) : (
          <Body question={question} answer={answer} draft={draft} />
        )}
      </div>
    </section>
  );
}

function Body({ question, answer, draft }: { question: Question; answer: Answer; draft: Draft }) {
  switch (answer.type) {
    case "noul":
      return <NoulMark value={answer.noul} yes={question.yes.trim() || "Yes"} no={question.no.trim() || "No"} />;
    case "choice":
      return <ChoiceBody question={question} answer={answer} />;
    case "score":
      return <ScoreBody question={question} answer={answer} />;
    case "number":
      return <NumberBody answer={answer} />;
    case "point":
    case "box":
      return <SpatialBody answer={answer} draft={draft} />;
    default:
      return null;
  }
}

function ChoiceBody({ question, answer }: { question: Question; answer: Extract<Answer, { type: "choice" }> }) {
  const declared = question.options.map((option) => option.key);
  const extra = Object.keys(answer.probabilities).filter((key) => !declared.includes(key));
  const rows = [...declared, ...extra];
  return (
    <div>
      <p className="font-display text-[28px] font-semibold leading-none text-ink">{answer.choice}</p>
      <ol className="mt-3 max-h-[19rem] overflow-y-auto pr-1">
        {rows.map((key) => (
          <DistributionRow
            key={key}
            label={key}
            detail={question.options.find((o) => o.key === key)?.description || key}
            value={answer.probabilities[key] ?? 0}
            winner={key === answer.choice}
          />
        ))}
      </ol>
      <Confidence value={answer.confidence} note="the winner's own probability" />
    </div>
  );
}

function ScoreBody({ question, answer }: { question: Question; answer: Extract<Answer, { type: "score" }> }) {
  const order = levelOrder(answer.probabilities);
  const levels = order.map((key, position) => ({
    index: position,
    // The legend is what the server echoed back; the draft's level is the
    // fallback for a response that carried none.
    label: answer.legend[key] ?? question.levels[Number(key)] ?? key,
    probability: answer.probabilities[key] ?? 0,
  }));
  return (
    <div>
      <ScoreMark score={answer.score} levels={levels} probabilities={levels.map((l) => l.probability)} />
      <Confidence value={answer.confidence} note="1 − spread over half the level range, so two adjacent levels are confident" />
    </div>
  );
}

function NumberBody({ answer }: { answer: Extract<Answer, { type: "number" }> }) {
  return (
    <div>
      <p className="font-display text-[34px] font-semibold leading-none tabular-nums text-ink">
        {answer.number.toLocaleString()}
        <span className="ml-2 align-baseline font-display text-[15px] font-medium text-ash">± {answer.uncertainty.toFixed(1)}</span>
      </p>
      <div className="mt-3">
        <DigitTrace digits={answer.digits} />
      </div>
      <p className="mt-2 max-w-[68ch] text-[12px] leading-snug text-ash">
        One column per place: the digit it committed and how sure it was. The uncertainty is in units of the number, and it is the
        model's own reckoning — not a bound.
      </p>
    </div>
  );
}

function SpatialBody({ answer, draft }: { answer: Extract<Answer, { type: "point" | "box" }>; draft: Draft }) {
  const image = evidenceImage(draft.evidence);
  const axes = AXES[answer.type];
  return (
    <div className="flex flex-col gap-3">
      {image ? (
        <ImageMark
          image={image}
          caption="The halo is the model's own uncertainty on each axis, in pixels. It is a self-report, not a bound."
        >
          {(natural) => {
            // A hairline on a 1,500-pixel image has to be drawn in that image's
            // units, so strokes scale with its longest side.
            const scale = Math.max(natural.width, natural.height) / 600;
            return answer.type === "point" ? (
              <PointGlyph
                x={answer.pixels.x ?? 0}
                y={answer.pixels.y ?? 0}
                sigmaX={answer.uncertainty.x ?? 0}
                sigmaY={answer.uncertainty.y ?? 0}
                scale={scale}
              />
            ) : (
              <BoxGlyph pixels={answer.pixels} sigma={answer.uncertainty} scale={scale} />
            );
          }}
        </ImageMark>
      ) : (
        <p className="text-[13px] text-ash">The evidence image is no longer on this page, so there is nothing to draw on.</p>
      )}
      <table className="w-full text-[13px] tabular-nums">
        <thead>
          <tr className="font-display text-[11px] text-ash">
            <th className="w-10 text-left font-medium">axis</th>
            <th className="text-right font-medium">pixels</th>
            <th className="text-right font-medium">± px</th>
            <th className="text-right font-medium">on the 0–scale</th>
          </tr>
        </thead>
        <tbody>
          {axes.map((axis) => (
            <tr key={axis} className="border-t border-line/60">
              <td className="py-1 font-display text-ash">{axis}</td>
              <td className="py-1 text-right text-ink">{answer.pixels[axis] ?? "—"}</td>
              <td className="py-1 text-right text-ash">{(answer.uncertainty[axis] ?? 0).toFixed(1)}</td>
              <td className="py-1 text-right text-ash">{answer.normalized[axis] ?? "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <details className="text-[12px] text-ash">
        <summary className="cursor-pointer font-display text-ink">Digit trace per axis</summary>
        <ol className="mt-2 flex flex-col gap-2">
          {axes.map((axis, index) => (
            <li key={axis} className="flex items-end gap-3">
              <span className="w-6 font-display text-ash">{axis}</span>
              <DigitTrace digits={answer.digits[axis] ?? []} offset={index * (answer.digits[axis]?.length ?? 0)} />
            </li>
          ))}
        </ol>
      </details>
    </div>
  );
}

function Confidence({ value, note }: { value: number; note: string }) {
  return (
    <div className="mt-3 flex items-center gap-3">
      <span className="font-display text-[11px] text-ash">confidence</span>
      <span className="w-14 shrink-0">
        <Bar value={value} dim />
      </span>
      <span className="font-display text-[13px] tabular-nums text-ink">{value.toFixed(3)}</span>
      <span className="min-w-0 truncate text-[12px] text-ash" title={note}>
        {note}
      </span>
    </div>
  );
}

function Missing({ id }: { id: string }) {
  return (
    <p className="border-l-2 border-warn pl-3 text-[13px] text-ash">
      ignis answered without <span className="font-display text-ink">{id}</span>. The other answers stand.
    </p>
  );
}

/**
 * One question's runtime failure, beside its siblings' answers rather than in
 * place of them: everything a caller could get wrong refuses the whole request
 * before the first prefill, so what lands here was already paid for.
 */
function Failed({ code, message }: { code: string; message: string }) {
  return (
    <div className="border-l-2 border-fault pl-3">
      <p className="font-display text-[13px] font-semibold text-fault">{code}</p>
      <p className="mt-0.5 max-w-[68ch] text-[13px] leading-snug text-ash">{message}</p>
    </div>
  );
}

/**
 * The one thing these answers cannot show, said once.
 *
 * The probabilities are renormalized over the declared options, so they always
 * sum to 1 and nothing in the body can say how much of the model's own
 * distribution the options held. That figure is the endpoint's single silent
 * failure, and it is a histogram on the metrics listener instead.
 */
function AnswerMassNote() {
  return (
    <p className="max-w-[68ch] border-t border-line pt-3 text-[12px] leading-snug text-ash">
      These probabilities are shared out among the declared options alone, so they always add to 1. How much of the model's own
      distribution those options actually held — the answer mass — is not in the response; the Monitor charts it.
    </p>
  );
}

function Raw({ run }: { run: Run }) {
  const [copied, setCopied] = useState(false);
  return (
    <details>
      <summary className="cursor-pointer font-display text-[13px] text-ash hover:text-ink">Response body</summary>
      <div className="md-code cut mt-2 [--cut-size:12px]">
        <div className="md-code-bar">
          <span>application/json</span>
          <button
            type="button"
            onClick={() => {
              void navigator.clipboard?.writeText(run.raw);
              setCopied(true);
              setTimeout(() => setCopied(false), 1200);
            }}
          >
            {copied ? "Copied" : "Copy"}
          </button>
        </div>
        <pre>{pretty(run.raw)}</pre>
      </div>
    </details>
  );
}

const pretty = (raw: string) => {
  try {
    return JSON.stringify(JSON.parse(raw), null, 2);
  } catch {
    return raw;
  }
};
