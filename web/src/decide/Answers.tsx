import { useState } from "react";
import { asText } from "./json.ts";
import { Bar, BoxGlyph, DigitTrace, DistributionRow, ImageMark, NoulMark, PointGlyph, ScoreMark } from "./marks.tsx";
import { DEFAULT_CEILING, type Draft, evidenceImage, type Question } from "./model.ts";
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
      <ol className="flex flex-col">
        {draft.questions.map((question, index) => (
          <li key={question.uid} className={index > 0 ? "mt-6 border-t border-line pt-6" : ""}>
            <AnswerPanel question={question} answer={run.response.answers[question.id]} draft={draft} />
          </li>
        ))}
      </ol>
      {/* The receipt, under what it paid for: what the request cost, what the
          answers cannot say, and the bytes they came back as. */}
      <footer className="flex flex-col gap-3 border-t border-line pt-4">
        <Cost run={run} anyGenerated={anyGenerated} />
        <AnswerMassNote />
        <Raw run={run} />
      </footer>
    </div>
  );
}

/** The wire cost, and what it says about which primitives were asked. */
function Cost({ run, anyGenerated }: { run: Run; anyGenerated: boolean }) {
  const { input_tokens, output_tokens } = run.response.usage;
  return (
    <div>
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
      <p className="mt-1.5 text-[12px] leading-snug text-ash">
        {anyGenerated
          ? "A number, and a point or a box answered by the digit chain, generate a digit per step, and a scalar generates until its number is complete, so those tokens are real. The readouts beside them generated none, and neither did a point or a box read off the calibrated heads."
          : "Nothing was generated: out of a single prefill of the evidence, every answer was read from the logits of one position — or, for a point or a box off the calibrated heads, from their attention over the image."}
      </p>
    </div>
  );
}

/** What each method did, for the chip that names it. */
const METHOD_TITLE: Record<"head" | "chain", string> = {
  head: "Read in one pass off the calibrated heads' attention over the image — no decode round ran",
  chain: "Written one digit at a time under a constrained decode — one decode round per digit",
};

function AnswerPanel({ question, answer, draft }: { question: Question; answer: Answer | undefined; draft: Draft }) {
  // What was asked is the heading. The name is how the answer is *keyed*,
  // which matters to the caller writing code against it and not to the reader
  // looking at the answer, so it sits beside the primitive as a label.
  const asked = asText(question.instructions).trim();
  return (
    <section>
      <header className="flex items-baseline justify-between gap-3">
        <h3 className="min-w-0 font-display text-[17px] font-bold leading-snug text-ink">{asked || question.id}</h3>
        <span className="flex shrink-0 items-baseline gap-2 font-display text-[11px] text-ash">
          <span className="font-mono" title="The key this answer came back under">
            {question.id}
          </span>
          <span className="cut bg-surface px-2 py-0.5 [--cut-size:5px]">{question.kind}</span>
          {/* Which method actually answered (GitHub #260, #263) — the answer's
              fact and not the question's, because a point or a box sent with
              no `method` is answered by whichever default applies. */}
          {(answer?.type === "point" || answer?.type === "box") && answer.method && (
            <span className="cut bg-surface px-2 py-0.5 text-ember [--cut-size:5px]" title={METHOD_TITLE[answer.method]}>
              {answer.method}
            </span>
          )}
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
      return <NoulBody question={question} answer={answer} />;
    case "choice":
      return <ChoiceBody question={question} answer={answer} />;
    case "score":
      return <ScoreBody question={question} answer={answer} />;
    case "number":
      return <NumberBody answer={answer} />;
    case "scalar":
      return <ScalarBody question={question} answer={answer} />;
    case "point":
    case "box":
      return <SpatialBody answer={answer} draft={draft} />;
    default:
      return null;
  }
}

/**
 * Jev's `noul` answer carries no confidence field, because the number *is*
 * the confidence — so the winning option's own probability is what reads
 * here, which is `p` for a yes and `1 - p` for a no. The raw `p` stays on the
 * page beside it: the answer above is the reading, and this is the figure it
 * was read from.
 */
function NoulBody({ question, answer }: { question: Question; answer: Extract<Answer, { type: "noul" }> }) {
  const yes = question.yes.trim() || "Yes";
  const no = question.no.trim() || "No";
  return (
    <div>
      <NoulMark value={answer.noul} yes={yes} no={no} />
      <Confidence value={Math.max(answer.noul, 1 - answer.noul)} note={`p(yes) = ${answer.noul.toFixed(3)}`} />
    </div>
  );
}

function ChoiceBody({ question, answer }: { question: Question; answer: Extract<Answer, { type: "choice" }> }) {
  const declared = question.options.map((option) => option.key);
  const extra = Object.keys(answer.probabilities).filter((key) => !declared.includes(key));
  const rows = [...declared, ...extra];
  return (
    <div>
      <p className="font-display text-[28px] font-semibold leading-none text-ember">{answer.choice}</p>
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
      <Confidence
        value={answer.confidence}
        note="the winner's own share"
        detail="A choice's confidence is the chosen option's own probability, nothing more."
      />
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
      <Confidence
        value={answer.confidence}
        note="how tightly the levels cluster"
        detail={
          "Not the tallest bar: 1 − the spread of the distribution over half the level range. " +
          "Mass on two neighbouring levels is a precise answer and stays confident; the same mass split " +
          "between the two ends averages to a middle nobody voted for, and reads 0."
        }
      />
    </div>
  );
}

function NumberBody({ answer }: { answer: Extract<Answer, { type: "number" }> }) {
  return (
    <div>
      <p className="font-display text-[34px] font-semibold leading-none tabular-nums text-ember">
        {answer.number.toLocaleString()}
        <span className="ml-2 align-baseline font-display text-[15px] font-medium text-ash">± {answer.uncertainty.toFixed(1)}</span>
      </p>
      <div className="mt-3">
        <DigitTrace digits={answer.digits} />
      </div>
      <p className="mt-2 text-[12px] leading-snug text-ash">
        One column per place: the digit it committed and how sure it was. The uncertainty is in units of the number, and it is the
        model's own reckoning — not a bound.
      </p>
    </div>
  );
}

/**
 * A `scalar`: what the model wrote, what that parses to, and what the run
 * cost.
 *
 * Both spellings are on the page because they are not the same fact — `3` and
 * `3.0` are one number written two ways, and a caller checking a reading
 * against its trace wants the one that produced it.
 *
 * The rounds are here rather than only in the receipt below: a scalar's whole
 * argument is that it spends what its answer needs, and that is invisible
 * unless the field it did *not* have to fill is named beside it.
 */
function ScalarBody({ question, answer }: { question: Question; answer: Extract<Answer, { type: "scalar" }> }) {
  // What the run spent, read off the spelling: one token per character, plus
  // the brace that closed the object. The response bills the whole request in
  // one figure and never per question, so this is derived rather than
  // reported — and it holds because the alphabet is one token per character,
  // which `scalar.rs` verifies against the loaded tokenizer at load.
  const rounds = [...answer.text].length + 1;
  const ceiling = question.ceiling ?? DEFAULT_CEILING;
  return (
    <div>
      <p className="font-display text-[34px] font-semibold leading-none tabular-nums text-ember">
        {answer.text}
        <span className="ml-2 align-baseline font-display text-[15px] font-medium text-ash">± {sigma(answer.uncertainty)}</span>
      </p>
      <p className="mt-2 flex flex-wrap items-baseline gap-x-4 gap-y-1 font-display text-[12px] tabular-nums text-ash">
        <span title="The f64 the spelling above parsed to. `3` and `3.0` are the same number and not the same answer.">
          reads as <span className="text-ink">{String(answer.value)}</span>
        </span>
        <span title="The characters it wrote, plus the brace that closed the object. A number spends its whole field whatever the answer is.">
          <span className="text-ink">{rounds}</span> round{rounds === 1 ? "" : "s"}, under a ceiling of {ceiling} digits
        </span>
      </p>
      <div className="mt-3">
        <DigitTrace digits={answer.digits} />
      </div>
      <p className="mt-2 text-[12px] leading-snug text-ash">
        One column per digit, and how sure it was of it. The decimal point, the sign and the closing brace were steps of the run
        too, but they hold no place — a trace shorter than the spelling is right. The uncertainty is in units of the value, each
        digit weighted by the place the point put it, and it is the model's own reckoning — not a bound.
      </p>
    </div>
  );
}

/**
 * A sigma in the value's own units, which for a scalar is a small number: a
 * fixed decimal place would print every one of them as `0.0`, and a
 * percentage would be a different quantity altogether.
 */
function sigma(value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  return Math.abs(value) >= 1 ? value.toFixed(1) : String(Number(value.toPrecision(2)));
}

/**
 * Decimal places on a point's or a box's per-axis figure.
 *
 * Three, whatever the method wrote there, by the owner's call: a head
 * answer's figure is a fraction of an image token and a chain's is a
 * self-reported sigma, and at one place a reading a quarter of a cell finer
 * than another printed as the same number. It is also what `region.share` and
 * every confidence on this page already read to, so the panel has one
 * precision rather than two.
 */
const SPATIAL_DECIMALS = 3;

/**
 * What the drawing means, for a head answer — which is not one sentence.
 *
 * The three readings differ in what they can claim: the head set outlines an
 * object, so its point has an extent and its box *is* that extent; the
 * pointing head alone has no set to outline with, and its point sits on the
 * part of the object that head reads.
 */
function headNote(kind: "point" | "box", extent: boolean): string {
  if (kind === "box") {
    return "The box is the extent the head set outlines around the pointing head's point, and the band around it is what the reading resolves — half an image token per edge, the same on every answer — not a spread. Ask for the chain on a target under two image tokens tall, where it was measured the better box.";
  }
  if (extent) {
    return "The dashed outline is the extent the head set read around the object, and the crosshair is its centre. The solid rectangle is what the reading resolves — half an image token per axis, the same on every answer — not a spread. Ask for the chain when you need finer than that.";
  }
  return "The rectangle is one image token — what the pointing head's map can resolve, the same on every answer — and not a spread. This load has no head set, so there is no extent and the point sits on the part of the object that head reads: on a labelled target, where the label begins rather than its centre. Ask for the chain when you need finer than a token.";
}

/**
 * A point or a box: the picture and its figures side by side, because they are
 * one answer read two ways — the drawing says where, the table says how
 * precisely.
 *
 * A head answer (GitHub #260, #263) is the same answer with a different
 * second reading: its per-axis figure is the reading's **resolution**, the
 * same on every answer — not a spread that falls off from the centre — and
 * what varies between a strong answer and a weak one is the share of the
 * heads' attention the region held. So the figure draws as the rectangle it
 * is, the column that reports it is named for what it is, and the share reads
 * underneath the way a confidence does.
 */
function SpatialBody({ answer, draft }: { answer: Extract<Answer, { type: "point" | "box" }>; draft: Draft }) {
  const image = evidenceImage(draft.evidence);
  const axes = AXES[answer.type];
  // A head answer is read in one pass and has no digit trace.
  const digits = answer.digits;
  const head = answer.method === "head";
  // The box the head set outlined around the point, where there was a set to
  // outline with: a head point off the pointing head alone carries none.
  const extent = answer.type === "point" ? answer.extent : undefined;
  // An anchored reading resolves a peak *inside* its cell (GitHub #264), so
  // it answers to half a token; the pointing head alone answers to a whole
  // one. A head box is always the set's, so it is always the half.
  const subCell = head && (answer.type === "box" || extent !== undefined);
  return (
    <div className="grid items-start gap-5 sm:grid-cols-2">
      <div className="flex justify-center">
        {image ? (
          <ImageMark image={image}>
            {(natural) =>
              answer.type === "point" ? (
                <PointGlyph
                  x={answer.pixels.x ?? 0}
                  y={answer.pixels.y ?? 0}
                  sigmaX={answer.uncertainty.x ?? 0}
                  sigmaY={answer.uncertainty.y ?? 0}
                  span={Math.min(natural.width, natural.height)}
                  cell={head}
                  extent={extent}
                />
              ) : (
                <BoxGlyph pixels={answer.pixels} sigma={answer.uncertainty} cell={head} />
              )
            }
          </ImageMark>
        ) : (
          <p className="text-[13px] text-ash">The evidence image is no longer on this page, so there is nothing to draw on.</p>
        )}
      </div>

      <div className="flex min-w-0 flex-col gap-3">
        <table className="w-full text-[13px] tabular-nums">
          <thead>
            <tr className="font-display text-[11px] text-ash">
              <th className="w-10 text-left font-medium">axis</th>
              <th className="text-right font-medium">pixels</th>
              <th
                className="text-right font-medium"
                title={
                  head
                    ? subCell
                      ? "What the reading resolves on this axis: half an image token, because the head set resolves its peak inside the cell"
                      : "What the reading resolves on this axis: one whole image token, the pointing head's own map"
                    : "The model's own uncertainty on this axis"
                }
              >
                {head ? "cell px" : "± px"}
              </th>
              <th className="text-right font-medium">on the 0–scale</th>
            </tr>
          </thead>
          <tbody>
            {axes.map((axis) => (
              <tr key={axis} className="border-t border-line/60">
                <td className="py-1 font-display text-ash">{axis}</td>
                <td className="py-1 text-right text-ink">{answer.pixels[axis] ?? "—"}</td>
                <td className="py-1 text-right text-ash">{(answer.uncertainty[axis] ?? 0).toFixed(SPATIAL_DECIMALS)}</td>
                <td className="py-1 text-right text-ash">{answer.normalized[axis] ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {extent && <Extent extent={extent} />}
        {answer.region && <Region region={answer.region} />}
        {digits && <details className="text-[12px] text-ash">
          <summary className="cursor-pointer font-display text-ink">Digit trace per axis</summary>
          {/* Two to a row: a box has four axes, and a column of four traces is
              taller than the picture they belong to. */}
          <ol className="mt-2 grid grid-cols-2 gap-x-5 gap-y-3">
            {axes.map((axis, index) => (
              <li key={axis} className="flex items-end gap-3">
                <span className="w-6 font-display text-ash">{axis}</span>
                <DigitTrace digits={digits[axis] ?? []} offset={index * (digits[axis]?.length ?? 0)} />
              </li>
            ))}
          </ol>
        </details>}
      </div>

      <p className="text-[12px] leading-snug text-ash sm:col-span-2">
        {head ? headNote(answer.type, extent !== undefined) : "The halo is the model's own uncertainty on each axis, in pixels. It is a self-report, not a bound."}
      </p>
    </div>
  );
}

/**
 * The extent a head set read around a point (GitHub #263), as text.
 *
 * The outline on the picture says where it is and this says what it is: every
 * figure in a panel has to be on the page as a number, or a reader who wants
 * the box the point came out of has to measure it off a drawing.
 */
function Extent({ extent }: { extent: Record<string, number> }) {
  return (
    <div className="flex items-baseline gap-3">
      <span
        className="shrink-0 font-display text-[11px] text-ash"
        title="The box the head set outlined around the object, in pixels of the submitted image. The point is its centre."
      >
        extent
      </span>
      {/* One span per corner, not one string: HTML collapses the run of
          spaces that separated them, so the four figures ran together. */}
      <span className="flex min-w-0 flex-wrap gap-x-3 font-display text-[13px] tabular-nums text-ink">
        {CORNERS.map((corner) => (
          <span key={corner}>
            <span className="text-ash">{corner}</span> {extent[corner] ?? "—"}
          </span>
        ))}
      </span>
    </div>
  );
}

/** The corners an extent is written in, in the order the wire writes them. */
const CORNERS = ["x0", "y0", "x1", "y1"] as const;

/**
 * How concentrated the heads' attention was (GitHub #260, #263).
 *
 * The share is the confidence to act on — a diffuse map is a weaker answer —
 * and it is not a calibrated probability, which is why it does not read as
 * one: the number is beside its bar and the cells it was taken over are
 * beside that, because a large share over many cells is not the same answer
 * as the same share over one. It is the **pointing head's** on every answer,
 * a box's included: the set's other heads outline what that head found, they
 * do not vote on whether it found anything.
 */
function Region({ region }: { region: { cells: number; share: number } }) {
  return (
    <div className="flex items-center gap-3">
      <span className="font-display text-[11px] text-ash">attention</span>
      <span className="w-14 shrink-0">
        <Bar value={region.share} dim />
      </span>
      <span className="font-display text-[13px] tabular-nums text-ink">{region.share.toFixed(3)}</span>
      <span
        className="min-w-0 truncate text-[12px] text-ash"
        title="The share of the pointing head's attention over the image that the cells it read held. It separates hits from misses on the measured scenes; it is not a calibrated probability."
      >
        over {region.cells} cell{region.cells === 1 ? "" : "s"}
      </span>
    </div>
  );
}

/**
 * The confidence row. `note` is what fits on the line; `detail` is what a
 * reader who stops on it wants — the arithmetic, which is ours and not Jev's,
 * so it has to be sayable somewhere.
 */
function Confidence({ value, note, detail }: { value: number; note: string; detail?: string }) {
  return (
    <div className="mt-3 flex items-center gap-3">
      <span className="font-display text-[11px] text-ash">confidence</span>
      <span className="w-14 shrink-0">
        <Bar value={value} dim />
      </span>
      <span className="font-display text-[13px] tabular-nums text-ink">{value.toFixed(3)}</span>
      <span className="min-w-0 truncate text-[12px] text-ash" title={detail ?? note}>
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
      <p className="mt-0.5 text-[13px] leading-snug text-ash">{message}</p>
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
    <p className="text-[12px] leading-snug text-ash">
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
