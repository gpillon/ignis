// The examples the Decide tab opens with (GitHub #247). They double as the
// empty state: the fastest way to learn what the endpoint answers is to send
// something it answers well.
//
// Three of them are Jev's own documented requests over Jev's own state, so a
// reader who knows that API recognises the tab immediately; the rest show what
// this server adds — a JSON evidence, a generated number, and a position on an
// image.

import flame from "../brand/flame.webp";
import { imageFromFile, type PromptImage } from "../conversation/images.ts";
import { jsonString } from "./json.ts";
import { type Draft, newQuestion, type Option, type Primitive, type Question } from "./model.ts";

export type Example = {
  id: string;
  name: string;
  /** What picking this one shows, in one line. */
  shows: string;
  build: () => Promise<Draft>;
};

/** A question, written the way an example wants to read. */
function question(id: string, kind: Primitive, instructions: string, extra: Partial<Question> = {}): Question {
  return { ...newQuestion(kind, id), instructions: jsonString(instructions), ...extra };
}

const options = (pairs: [string, string][]): Option[] => pairs.map(([key, description]) => ({ key, description }));

const PAYOUTS = "Help! My payouts have been failing for 3 days. I've emailed twice and nobody has replied. If this isn't fixed by Friday I'm moving to another provider.";

const ORDER = `{
  "order": "A-4471",
  "placed": "2026-09-14",
  "items": [
    { "sku": "KLN-200", "qty": 2, "price": 149.0 },
    { "sku": "EMB-010", "qty": 1, "price": 38.5 }
  ],
  "shipping": { "country": "IT", "method": "standard" },
  "note": "Please deliver before the 20th, it is a gift."
}`;

export const EXAMPLES: Example[] = [
  {
    id: "triage",
    name: "Triage one message",
    shows: "Three questions over one piece of evidence — prefilled once, answered together.",
    build: async () => ({
      evidence: { mode: "text", text: PAYOUTS },
      extras: [],
      questions: [
        question("is_urgent", "noul", "Does this convey urgency?", {
          yes: "Explicitly time-sensitive",
          no: "No urgency expressed",
        }),
        question("department", "choice", "Which team should handle this?", {
          options: options([
            ["billing", "Payments, invoicing, refunds"],
            ["technical", "Bugs, outages, integrations"],
            ["sales", "Pricing, upgrades, new accounts"],
          ]),
        }),
        question("frustration", "score", "How frustrated is the customer?", {
          levels: ["Calm", "Frustrated", "Very angry"],
        }),
      ],
    }),
  },
  {
    id: "routing",
    name: "Route to one of twelve",
    shows: "A wide choice: the distribution stays readable well past a handful of options.",
    build: async () => ({
      evidence: { mode: "text", text: PAYOUTS },
      extras: [],
      questions: [
        question("queue", "choice", "Which queue should this ticket land in?", {
          options: options([
            ["payouts", "Money leaving the account: payouts, transfers, settlement"],
            ["billing", "Money coming in: invoices, cards, refunds"],
            ["onboarding", "Account creation, verification, first transaction"],
            ["api", "Integration and SDK questions"],
            ["outage", "Something is down right now for many customers"],
            ["fraud", "Suspected fraud or an account takeover"],
            ["compliance", "KYC, sanctions, regulatory paperwork"],
            ["tax", "Invoices, VAT, year-end statements"],
            ["sales", "Pricing, plans, upgrades"],
            ["partners", "Resellers and platform partners"],
            ["press", "Media and analyst enquiries"],
            ["other", "None of the above"],
          ]),
        }),
        question("churn_risk", "score", "How likely is this customer to leave?", {
          levels: ["Settled", "Unhappy", "Shopping around", "One foot out the door", "Already gone"],
        }),
      ],
    }),
  },
  {
    id: "record",
    name: "Read a structured record",
    shows: "The evidence is JSON, not prose — an object or an array goes in as itself.",
    build: async () => ({
      evidence: { mode: "json", text: ORDER },
      extras: [],
      questions: [
        question("needs_express", "noul", "Should this order be upgraded to express shipping?", {
          yes: "The note asks for a date standard shipping would miss",
          no: "Standard shipping is fine",
        }),
        question("gift", "noul", "Is this order a gift?"),
        question("segment", "choice", "How would you describe this basket?", {
          options: options([
            ["single", "One item, low value"],
            ["multi", "Several items, ordinary value"],
            ["bulk", "Many units of the same item"],
          ]),
        }),
      ],
    }),
  },
  {
    id: "count",
    name: "Count something",
    shows: "A number is generated a digit at a time, and each digit reports its own confidence.",
    build: async () => ({
      evidence: { mode: "text", text: PAYOUTS },
      extras: [],
      questions: [
        question("days_failing", "number", "For how many days have the payouts been failing?", { digits: 2 }),
        question("emails_sent", "number", "How many emails has the customer sent?", { digits: 1 }),
      ],
    }),
  },
  {
    id: "onimage",
    name: "Find it on an image",
    shows: "A point and a box answer in the pixels of the image you submitted.",
    build: async () => ({
      evidence: { mode: "image", images: [await brandImage()], text: "The ignis mark." },
      extras: [],
      questions: [
        question("hottest", "point", "Where is the brightest part of the flame?"),
        question("mark", "box", "Draw a box around the whole flame."),
        question("on_dark", "noul", "Is this mark on a dark background?"),
      ],
    }),
  },
];

/**
 * The bundled mark as an evidence image.
 *
 * Fetched and re-encoded through the chat's own `imageFromFile`, so the
 * example's `state` carries a `data:` URI exactly like a picked file does —
 * the server never has to reach back to this page for it.
 */
async function brandImage(): Promise<PromptImage> {
  const response = await fetch(flame);
  const blob = await response.blob();
  const read = await imageFromFile(new File([blob], "ignis-flame.webp", { type: blob.type || "image/webp" }));
  if (!read.ok) throw new Error(read.error);
  return read.image;
}
