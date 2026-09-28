import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { DecideView } from "./DecideView.tsx";
import { EvidenceEditor } from "./EvidenceEditor.tsx";
import { EXAMPLES } from "./examples.ts";
import { jsonString } from "./json.ts";
import { EMPTY_DRAFT, EMPTY_SPARE, type Evidence, newQuestion, type Primitive, type Question, validate } from "./model.ts";
import { isClipped, QuestionCard } from "./QuestionCard.tsx";
import { createDecision } from "./sessions.ts";
import { Sessions } from "./Sessions.tsx";

// The bench (GitHub #247): the opening screen, and one card per primitive.

const card = (
  question: Question,
  evidence: Evidence = { mode: "text", text: "e" },
  faults = validate({ evidence, questions: [question], extras: [] }),
) =>
  renderToStaticMarkup(
    <QuestionCard question={question} evidence={evidence} faults={faults} first last onChange={() => {}} onRemove={() => {}} onMove={() => {}} />,
  );

const ask = (kind: Primitive, extra: Partial<Question> = {}): Question => ({
  ...newQuestion(kind, kind),
  instructions: jsonString("Decide."),
  ...extra,
});

describe("DecideView", () => {
  it("opens on what the endpoint does and every example it ships", () => {
    const html = renderToStaticMarkup(<DecideView ready drawer={null} onDrawer={() => {}} />);
    expect(html).toContain("Nothing is generated");
    for (const example of EXAMPLES) {
      expect(html, example.id).toContain(example.name);
      expect(html, example.id).toContain(example.shows);
    }
  });

  it("ships every text and JSON example sendable as it loads, the locates' pointers included", async () => {
    // The image example fetches its picture, which is the browser's to do.
    for (const example of EXAMPLES.filter((e) => e.id !== "onimage")) {
      expect(validate(await example.build()), example.id).toEqual([]);
    }
    const listitem = await EXAMPLES.find((e) => e.id === "listitem")!.build();
    expect(listitem.questions.map((q) => q.within)).toEqual(["/tickets", "/tickets", "/notes"]);
  });

  it("asks the Three Laws as choices whose every option argues its case", async () => {
    const laws = await EXAMPLES.find((e) => e.id === "laws")!.build();
    expect(laws.questions).toHaveLength(8);
    for (const q of laws.questions) {
      expect(q.kind, q.id).toBe("choice");
      // A bare key would be an action with no reason, which is the half of
      // the dilemma the example exists to put in front of the model.
      for (const option of q.options) expect(option.description, `${q.id}.${option.key}`).toMatch(/: /);
    }
    expect(laws.evidence.mode === "text" && laws.evidence.text).toContain("in this order of precedence");
  });

  it("holds Decide back until there is something to decide", () => {
    const html = renderToStaticMarkup(<DecideView ready drawer={null} onDrawer={() => {}} />);
    expect(html).toMatch(/<button[^>]*disabled[^>]*>Decide<\/button>/);
    expect(html).toContain("Add a question");
  });

  it("says it is waiting for the model rather than offering a button that cannot work", () => {
    expect(renderToStaticMarkup(<DecideView ready={false} drawer={null} onDrawer={() => {}} />)).toContain("Waiting for the model");
  });

  it("offers every primitive as a starting point", () => {
    const html = renderToStaticMarkup(<DecideView ready drawer={null} onDrawer={() => {}} />);
    for (const kind of ["noul", "choice", "score", "number", "scalar", "point", "box", "locate"]) expect(html, kind).toContain(`>${kind}</button>`);
  });

  it("gives every form field an id or a name, as the browser asks", () => {
    const html =
      renderToStaticMarkup(<DecideView ready drawer={null} onDrawer={() => {}} />) +
      card(ask("noul")) +
      card(ask("choice", { options: [{ key: "a", description: "the a" }] })) +
      card(ask("score", { levels: ["Low", "High"] })) +
      card(ask("number")) +
      card(ask("scalar")) +
      card(ask("locate")) +
      renderToStaticMarkup(<EvidenceEditor evidence={{ mode: "text", text: "" }} spare={EMPTY_SPARE} onChange={() => {}} onSpare={() => {}} />) +
      renderToStaticMarkup(<EvidenceEditor evidence={{ mode: "json", text: "" }} spare={EMPTY_SPARE} onChange={() => {}} onSpare={() => {}} />) +
      renderToStaticMarkup(<EvidenceEditor evidence={{ mode: "image", images: [], text: "" }} spare={EMPTY_SPARE} onChange={() => {}} onSpare={() => {}} />);
    const fields = html.match(/<(input|textarea|select)\b[^>]*>/g) ?? [];
    expect(fields.length).toBeGreaterThan(10);
    for (const field of fields) expect(field).toMatch(/\s(id|name)="[^"]+"/);
  });
});

describe("QuestionCard", () => {
  it("leads with what the question asks, and labels the answer name it keys", () => {
    const html = card(ask("choice", { options: [{ key: "a", description: "the a" }] }));
    // The name is a key, so it is labelled and it comes last; the field that
    // is prominent and unlabelled is the one whose placeholder explains itself.
    expect(html).toContain('placeholder="What should the model decide?"');
    expect(html).toContain("Answer name");
    expect(html.indexOf("What should the model decide?")).toBeLessThan(html.indexOf("Answer name"));
  });

  it("offers a plain way to delete the question", () => {
    expect(card(ask("noul"))).toContain('aria-label="Delete this question"');
  });

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

  it("calls a field clipped only when it is really showing less than it holds", () => {
    // The rule the folding row turns on. A pixel of rounding is not an
    // overflow: a box whose content is one pixel over still reads as full.
    expect(isClipped({ scrollWidth: 400, clientWidth: 200 })).toBe(true);
    expect(isClipped({ scrollWidth: 200, clientWidth: 200 })).toBe(false);
    expect(isClipped({ scrollWidth: 201, clientWidth: 200 })).toBe(false);
    expect(isClipped({ scrollWidth: 202, clientWidth: 200 })).toBe(true);
  });

  it("folds an option's name and delete away while its description is being written", () => {
    const html = card(ask("choice", { options: [{ key: "billing", description: "Payments" }] }));
    // At rest both neighbours are there at their own width, and the width is
    // on a wrapper rather than on the control: a second width utility beside
    // the control's own would not win.
    expect(html).toContain("mr-1.5 w-[7.5rem]");
    expect(html).toContain("ml-1.5 w-7");
    expect(html).toContain("transition-[width,margin]");
    expect(html).toContain("motion-reduce:transition-none");
    // And they are still real controls while folded, so the row stays
    // reachable by keyboard.
    expect(html).toContain('aria-label="Option 1 name"');
    expect(html).toContain('aria-label="Remove option 1"');
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

  it("offers a point its two methods, and a default that is the load's and not the tab's", () => {
    // GitHub #260. The empty value is the load's own choice — head where the
    // artifact has a calibrated pointing head, chain where it has none — and
    // the tab cannot resolve it, so it must not present one of the two as if
    // it had.
    const html = card(ask("point"));
    expect(html).toContain("Answered by");
    expect(html).toContain("this load&#x27;s own</option>");
    expect(html).toContain('<option value="head">head</option>');
    expect(html).toContain('<option value="chain">chain</option>');
    expect(html).toContain("calibrated pointing head");
    // What each one is for, once it has been chosen.
    expect(card(ask("point", { method: "head" }))).toContain("no decode round");
    expect(card(ask("point", { method: "chain" }))).toContain("one decode round per digit");
    // A readout has one way of being answered, and the endpoint refuses the
    // field on it.
    expect(card(ask("number"))).not.toContain("Answered by");
    expect(card(ask("noul"))).not.toContain("Answered by");
  });

  it("offers a box the same two methods, and a default that is the endpoint's and not the load's", () => {
    // GitHub #263: a box answers with the chain on **every** load — the head
    // box was measured no better on a small button — so the empty value here
    // is not "whatever this load serves" the way a point's is, and saying so
    // would tell the reader a box is load-dependent when it is not.
    const html = card(ask("box"));
    expect(html).toContain("Answered by");
    expect(html).toContain("the endpoint&#x27;s</option>");
    expect(html).toContain('<option value="head">head</option>');
    expect(html).toContain('<option value="chain">chain</option>');
    expect(html).toContain("The chain, on every load");
    expect(html).not.toContain("this load&#x27;s own");
    // Its head reads the set's extent, and it is refused where there is no
    // set rather than quietly answered by the chain.
    const head = card(ask("box", { method: "head" }));
    expect(head).toContain("head set&#x27;s extent");
    expect(head).toContain("refuses it");
    expect(card(ask("box", { method: "chain" }))).toContain("four edges");
  });

  it("shows a question's own faults on the question", () => {
    const html = card(ask("choice", { options: [] }));
    expect(html).toContain("declares no options");
    expect(html).toContain("border-l-warn");
  });

  it("offers a locate a place to search within and says what it will choose between there", () => {
    const log: Evidence = { mode: "text", text: "a\n\nb\nc" };
    const html = card(ask("locate"), log);
    expect(html).toContain('placeholder="the whole evidence"');
    // Four lines, one of them blank: the blank one keeps its number and is
    // never named.
    expect(html).toContain("4 lines to choose between, numbered from 0");
    expect(html).toContain("1 of them blank");
    expect(card(ask("noul"), log)).not.toContain("Search within");
  });

  it("offers a locate its kind, method and compression, each defaulting to the endpoint's", () => {
    // GitHub #278, in the shape of a point's method: the empty choice is the
    // default and sends no field. `auto` and the `shortlist` are the
    // defaults' own names, so they are the empty choice and not offered
    // twice; a compression's default follows the kind, and says so.
    const html = card(ask("locate"), { mode: "text", text: "a\nb" });
    expect(html).toContain("Text kind");
    expect(html).toMatch(/<select[^>]*name="[^"]+-kind"[^>]*><option value="" selected="">auto<\/option><option value="log">log<\/option><option value="prose">prose<\/option><option value="records">records<\/option><\/select>/);
    expect(html).toMatch(/<select[^>]*name="[^"]+-method"[^>]*><option value="" selected="">shortlist<\/option><option value="vote">vote<\/option><\/select>/);
    expect(html).toMatch(
      /<select[^>]*name="[^"]+-compression"[^>]*><option value="" selected="">the kind&#x27;s own<\/option><option value="template_fold">template_fold<\/option><option value="none">none<\/option><\/select>/,
    );
    // What each default does, beside it.
    expect(html).toContain("Told for you");
    expect(html).toContain("The default: the calibrated heads narrow");
    expect(html).toContain("a log is folded into templates, prose and records are read as they are");
    // None of them on a question that is not a locate.
    for (const kind of ["noul", "point"] as const) expect(card(ask(kind)), kind).not.toContain("Text kind");
  });

  it("shows the choice it was given, and what that choice does", () => {
    const html = card(ask("locate", { textKind: "prose", locateMethod: "vote", compression: "none" }), { mode: "text", text: "a\nb" });
    expect(html).toContain('<option value="prose" selected="">prose</option>');
    expect(html).toContain('<option value="vote" selected="">vote</option>');
    expect(html).toContain('<option value="none" selected="">none</option>');
    expect(html).toContain("Sentences of a document");
    expect(html).toContain("The head vote served before");
    expect(html).toContain("Read the text as it is");
  });

  it("says auto reads an array of JSON objects as records, which its shape alone decides", () => {
    const state: Evidence = { mode: "json", text: '{"tickets": [{"id": 1}, {"id": 2}]}' };
    expect(card(ask("locate", { within: "/tickets" }), state)).toContain("auto reads it as records");
    expect(card(ask("locate"), { mode: "text", text: "a\nb" })).not.toContain("auto reads it as records");
  });

  it("reports a route the endpoint would refuse on the card", () => {
    const fold = card(ask("locate", { locateMethod: "vote", compression: "template_fold" }), { mode: "text", text: "a\nb" });
    expect(fold).toContain("asks for a vote over a fold");
    expect(fold).toContain("border-l-warn");
    const state: Evidence = { mode: "json", text: '{"tickets": [{"id": 1}, {"id": 2}]}' };
    expect(card(ask("locate", { within: "/tickets", textKind: "log" }), state)).toContain("which is read as records");
  });

  it("counts the items of the list a locate points into, and reports a pointer that names nothing", () => {
    const state: Evidence = { mode: "json", text: '{"tickets": [{"id": 1}, {"id": 2}, {"id": 3}]}' };
    expect(card(ask("locate", { within: "/tickets" }), state)).toContain("3 items to choose between");
    const missing = card(ask("locate", { within: "/ticket" }), state);
    expect(missing).toContain("names nothing in the evidence");
    expect(missing).toContain("border-l-warn");
  });

  it("shows a JSON instruction as JSON and offers to turn it into text", () => {
    const html = card({ ...ask("noul"), instructions: { kind: "object", entries: [{ key: "ask", value: jsonString("why?") }] } });
    expect(html).toContain("Edit as text");
    expect(html).toContain('&quot;ask&quot;: &quot;why?&quot;');
  });
});

describe("EvidenceEditor", () => {
  it("offers the three shapes state accepts and says what each is for", () => {
    const html = renderToStaticMarkup(<EvidenceEditor evidence={{ mode: "json", text: "{}" }} spare={EMPTY_SPARE} onChange={() => {}} onSpare={() => {}} />);
    for (const label of ["Text", "JSON", "Image"]) expect(html, label).toContain(`>${label}</button>`);
    expect(html).toContain("not as a quoted string");
  });

  it("shows the parse error it was handed", () => {
    const html = renderToStaticMarkup(<EvidenceEditor evidence={{ mode: "json", text: "{" }} spare={EMPTY_SPARE} onChange={() => {}} onSpare={() => {}} invalid="not JSON (line 1, column 2)" />);
    expect(html).toContain("not JSON (line 1, column 2)");
    expect(html).toContain("border-warn");
  });
});

describe("Sessions", () => {
  const list = {
    decisions: [
      { ...createDecision(2), draft: { ...EMPTY_DRAFT, questions: [ask("noul"), ask("choice")] } },
      { ...createDecision(1), running: true },
    ],
    activeId: 2,
  };

  it("names each decision by what it asks and says how much is in it", () => {
    const html = renderToStaticMarkup(<Sessions list={list} open onNew={() => {}} onSelect={() => {}} onRemove={() => {}} />);
    expect(html).toContain("Decide.");
    expect(html).toContain("2 questions");
    // A decision with a send in flight says so, and cannot be deleted under it.
    expect(html).toContain("Deciding");
    expect(html).toMatch(/aria-label="Delete decision: New decision"[^>]*disabled/);
  });

  it("marks the active one and offers a new one", () => {
    const html = renderToStaticMarkup(<Sessions list={list} open onNew={() => {}} onSelect={() => {}} onRemove={() => {}} />);
    expect(html).toContain('aria-current="true"');
    expect(html).toContain("New decision");
  });
});
