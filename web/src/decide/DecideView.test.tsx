import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { DecideView } from "./DecideView.tsx";
import { EvidenceEditor } from "./EvidenceEditor.tsx";
import { EXAMPLES } from "./examples.ts";
import { jsonString } from "./json.ts";
import { newQuestion, type Primitive, type Question, validate } from "./model.ts";
import { QuestionCard } from "./QuestionCard.tsx";

// The bench (GitHub #247): the opening screen, and one card per primitive.

const card = (question: Question, faults = validate({ evidence: { mode: "text", text: "e" }, questions: [question], extras: [] })) =>
  renderToStaticMarkup(
    <QuestionCard question={question} faults={faults} first last onChange={() => {}} onRemove={() => {}} onMove={() => {}} />,
  );

const ask = (kind: Primitive, extra: Partial<Question> = {}): Question => ({
  ...newQuestion(kind, kind),
  instructions: jsonString("Decide."),
  ...extra,
});

describe("DecideView", () => {
  it("opens on what the endpoint does and every example it ships", () => {
    const html = renderToStaticMarkup(<DecideView ready />);
    expect(html).toContain("Nothing is generated");
    for (const example of EXAMPLES) {
      expect(html, example.id).toContain(example.name);
      expect(html, example.id).toContain(example.shows);
    }
  });

  it("holds Decide back until there is something to decide", () => {
    const html = renderToStaticMarkup(<DecideView ready />);
    expect(html).toMatch(/<button[^>]*disabled[^>]*>Decide<\/button>/);
    expect(html).toContain("Add a question");
  });

  it("says it is waiting for the model rather than offering a button that cannot work", () => {
    expect(renderToStaticMarkup(<DecideView ready={false} />)).toContain("Waiting for the model");
  });

  it("offers every primitive as a starting point", () => {
    const html = renderToStaticMarkup(<DecideView ready />);
    for (const kind of ["noul", "choice", "score", "number", "point", "box"]) expect(html, kind).toContain(`>${kind}</button>`);
  });

  it("gives every form field an id or a name, as the browser asks", () => {
    const html =
      renderToStaticMarkup(<DecideView ready />) +
      card(ask("noul")) +
      card(ask("choice", { options: [{ key: "a", description: "the a" }] })) +
      card(ask("score", { levels: ["Low", "High"] })) +
      card(ask("number")) +
      renderToStaticMarkup(<EvidenceEditor evidence={{ mode: "text", text: "" }} onChange={() => {}} />) +
      renderToStaticMarkup(<EvidenceEditor evidence={{ mode: "json", text: "" }} onChange={() => {}} />) +
      renderToStaticMarkup(<EvidenceEditor evidence={{ mode: "image", images: [], text: "" }} onChange={() => {}} />);
    const fields = html.match(/<(input|textarea|select)\b[^>]*>/g) ?? [];
    expect(fields.length).toBeGreaterThan(10);
    for (const field of fields) expect(field).toMatch(/\s(id|name)="[^"]+"/);
  });
});

describe("QuestionCard", () => {
  it("shows a noul's two descriptions, with the server's defaults as the placeholders", () => {
    const html = card(ask("noul"));
    expect(html).toContain("A yes means");
    expect(html).toContain('placeholder="Yes"');
    expect(html).toContain('placeholder="No"');
  });

  it("shows a choice's options in declared order and says the order is the prompt", () => {
    const html = card(ask("choice", { options: [{ key: "zulu", description: "z" }, { key: "alpha", description: "a" }] }));
    expect(html).toContain("in the order the prompt lists them");
    expect(html.indexOf('value="zulu"')).toBeLessThan(html.indexOf('value="alpha"'));
  });

  it("numbers a score's levels, since the answer is a weighted average of their positions", () => {
    const html = card(ask("score", { levels: ["Low", "Mid", "High"] }));
    expect(html).toContain("Levels, lowest first");
    expect(html).toContain('value="Low"');
    expect(html).toContain('value="High"');
  });

  it("offers digits only for a constrained decode, and says what scale they declare", () => {
    expect(card(ask("number", { digits: 4 }))).toContain("Up to 9999");
    expect(card(ask("point", { digits: 3 }))).toContain("0–999 scale on each axis");
    expect(card(ask("noul"))).not.toContain("Digits per number");
  });

  it("shows a question's own faults on the question", () => {
    const html = card(ask("choice", { options: [] }));
    expect(html).toContain("declares no options");
    expect(html).toContain("border-l-warn");
  });

  it("shows a JSON instruction as JSON and offers to turn it into text", () => {
    const html = card({ ...ask("noul"), instructions: { kind: "object", entries: [{ key: "ask", value: jsonString("why?") }] } });
    expect(html).toContain("Edit as text");
    expect(html).toContain('&quot;ask&quot;: &quot;why?&quot;');
  });
});

describe("EvidenceEditor", () => {
  it("offers the three shapes state accepts and says what each is for", () => {
    const html = renderToStaticMarkup(<EvidenceEditor evidence={{ mode: "json", text: "{}" }} onChange={() => {}} />);
    for (const label of ["Text", "JSON", "Image"]) expect(html, label).toContain(`>${label}</button>`);
    expect(html).toContain("not as a quoted string");
  });

  it("shows the parse error it was handed", () => {
    const html = renderToStaticMarkup(<EvidenceEditor evidence={{ mode: "json", text: "{" }} onChange={() => {}} invalid="not JSON (line 1, column 2)" />);
    expect(html).toContain("not JSON (line 1, column 2)");
    expect(html).toContain("border-warn");
  });
});
